--------------------------------------------------------------------------------
-- gemm_seq.vhd -- sequences one matrix multiply through the PE array
--
-- Every matmul in the network is the same shape from this module's point of
-- view: PE_ROWS tokens by K reduction steps by N output channels, computed as
-- ceil(N / PE_COLS) tiles of K cycles each. The caller supplies the sizes and
-- the base addresses; this module walks the memories, drives the array, and
-- writes the requantised results back.
--
-- Two operand sources
-- -------------------
-- Most stages multiply activations by trained weights out of weight_rom. The
-- two attention products do not: Q.K^T and probs.V multiply one activation
-- tensor by another. Because act_ram is feature-major, word[k] already holds
-- all PE_ROWS tokens' k-th feature -- which is exactly the operand the weight
-- side needs, just sliced down to the PE_COLS channels of the current tile. So
-- attention costs no special datapath, only a mux and a second read port.
--
-- Pipelining
-- ----------
-- The memories read synchronously, so an address issued on cycle i yields data
-- on cycle i+1. Address generation therefore runs one cycle ahead of the
-- array, and the valid/first/last flags are delayed by one cycle to line up
-- with the data they describe.
--
-- Writeback overlap
-- -----------------
-- A finished tile is read out of the array's back accumulator bank while the
-- next tile accumulates into the front one, so writeback is normally free. It
-- takes PE_ROWS cycles to transpose the tile (the array reads out one token at
-- a time, but act_ram stores one feature at a time) plus PE_COLS cycles to
-- write. When K is shorter than that -- only the attention context product,
-- where K = SEQ_LEN = 16 -- the accumulator side stalls until writeback frees
-- the bank, which is correct rather than fast, and costs a handful of cycles
-- on the cheapest stage in the network.
--------------------------------------------------------------------------------

library ieee;
    use ieee.std_logic_1164.all;
    use ieee.numeric_std.all;

    use work.vit_pkg.all;

entity gemm_seq is
    generic (
        ACT_DEPTH : positive := 512;
        ROM_DEPTH : positive := 11456
    );
    port (
        clk  : in std_logic;
        rstn : in std_logic;

        -- ── Job description, sampled on i_start ─────────────────────────────
        i_start   : in std_logic;
        i_k_len   : in unsigned(clog2(ACT_DEPTH) - 1 downto 0);  -- K
        i_n_tiles : in unsigned(clog2(ACT_DEPTH) - 1 downto 0);  -- ceil(N/PE_COLS)
        i_a_base  : in unsigned(clog2(ACT_DEPTH) - 1 downto 0);  -- activations
        i_w_base  : in unsigned(clog2(ROM_DEPTH) - 1 downto 0);  -- weight ROM
        i_wa_base : in unsigned(clog2(ACT_DEPTH) - 1 downto 0);  -- weights-from-act
        i_o_base  : in unsigned(clog2(ACT_DEPTH) - 1 downto 0);  -- destination
        i_shift   : in unsigned(4 downto 0);                     -- requant shift
        i_act_w   : in std_logic;   -- '1': weight operand comes from act_ram

        o_busy : out std_logic;
        o_done : out std_logic;     -- one-cycle strobe when the matmul finishes

        -- ── Bias, indexed by the tile currently being started ───────────────
        o_tile   : out unsigned(clog2(ACT_DEPTH) - 1 downto 0);
        i_bias   : in  acc_col_t;

        -- ── act_ram port A: the activation operand ──────────────────────────
        o_ra_addr : out unsigned(clog2(ACT_DEPTH) - 1 downto 0);
        o_ra_re   : out std_logic;
        i_ra_data : in  q8_row_t;

        -- ── act_ram port B: the weight operand for attention stages ─────────
        o_rb_addr : out unsigned(clog2(ACT_DEPTH) - 1 downto 0);
        o_rb_re   : out std_logic;
        i_rb_data : in  q8_row_t;

        -- ── act_ram write port ──────────────────────────────────────────────
        o_wr_addr : out unsigned(clog2(ACT_DEPTH) - 1 downto 0);
        o_wr_data : out q8_row_t;
        o_wr_we   : out std_logic;

        -- ── weight ROM ──────────────────────────────────────────────────────
        o_rom_addr : out unsigned(clog2(ROM_DEPTH) - 1 downto 0);
        o_rom_re   : out std_logic;
        i_rom_data : in  q8_col_t;

        -- ── PE array ────────────────────────────────────────────────────────
        o_pe_a     : out q8_row_t;
        o_pe_w     : out q8_col_t;
        o_pe_bias  : out acc_col_t;
        o_pe_valid : out std_logic;
        o_pe_first : out std_logic;
        o_pe_last  : out std_logic;
        o_pe_rd    : out unsigned(clog2(PE_ROWS) - 1 downto 0);
        i_pe_acc   : in  acc_col_t
    );
end entity gemm_seq;

architecture rtl of gemm_seq is

    subtype act_addr_t is unsigned(clog2(ACT_DEPTH) - 1 downto 0);
    subtype rom_addr_t is unsigned(clog2(ROM_DEPTH) - 1 downto 0);

    -- ── Latched job parameters ──────────────────────────────────────────────
    signal k_len   : act_addr_t := (others => '0');
    signal n_tiles : act_addr_t := (others => '0');
    signal a_base  : act_addr_t := (others => '0');
    signal w_base  : rom_addr_t := (others => '0');
    signal wa_base : act_addr_t := (others => '0');
    signal o_base  : act_addr_t := (others => '0');
    signal shift_r : unsigned(4 downto 0) := (others => '0');
    signal act_w   : std_logic := '0';

    -- ── Accumulate side ─────────────────────────────────────────────────────
    type acc_state_t is (A_IDLE, A_RUN, A_DRAIN);
    signal a_state : acc_state_t := A_IDLE;

    signal k_idx : act_addr_t := (others => '0');   -- reduction index
    signal t_idx : act_addr_t := (others => '0');   -- tile being accumulated

    -- Control flags delayed one cycle to meet the memory data they describe.
    signal d_valid, d_first, d_last : std_logic := '0';
    signal d_tile : act_addr_t := (others => '0');

    -- ── Writeback side ──────────────────────────────────────────────────────
    type wb_state_t is (W_IDLE, W_READ, W_WRITE);
    signal w_state : wb_state_t := W_IDLE;

    signal wb_m    : unsigned(clog2(PE_ROWS) - 1 downto 0) := (others => '0');
    signal wb_c    : unsigned(clog2(PE_COLS) - 1 downto 0) := (others => '0');
    signal wb_tile : act_addr_t := (others => '0');
    signal wb_req  : std_logic := '0';
    signal wb_busy : std_logic := '0';

    -- Transposed tile: one q8_row_t per output channel of the tile, filled a
    -- token at a time as the array is read out.
    type col_buf_t is array (0 to PE_COLS - 1) of q8_row_t;
    signal col_buf : col_buf_t := (others => (others => (others => '0')));

    -- Running weight-ROM pointer. A tile consumes K consecutive ROM words and
    -- tiles run back to back, so a plain increment replaces what would
    -- otherwise be a (tile * K) multiplier sitting in the address path.
    signal rom_ptr : rom_addr_t := (others => '0');

    signal last_tile : std_logic;
    signal can_close : std_logic;

begin

    -- The final cycle of a tile hands the bank over to writeback, so it may
    -- only happen when writeback is not still holding the other bank. Every
    -- stage except the attention context product has K >= WB_CYCLES, so this
    -- condition is already true and costs nothing.
    can_close <= '1' when (wb_busy = '0' and wb_req = '0') else '0';
    last_tile <= '1' when t_idx = n_tiles - 1 else '0';

    o_busy <= '0' when a_state = A_IDLE and w_state = W_IDLE else '1';
    o_tile <= t_idx;

    -- ── Address generation, one cycle ahead of the array ────────────────────
    o_ra_addr <= a_base + k_idx;
    o_ra_re   <= '1' when a_state = A_RUN else '0';

    o_rb_addr <= wa_base + k_idx;
    o_rb_re   <= '1' when (a_state = A_RUN and act_w = '1') else '0';

    o_rom_addr <= rom_ptr;
    o_rom_re   <= '1' when (a_state = A_RUN and act_w = '0') else '0';

    -- ── Operand routing into the array ──────────────────────────────────────
    o_pe_a <= i_ra_data;

    -- Weight side: either a ROM word, or the PE_COLS-wide slice of the
    -- activation word belonging to this tile's output channels.
    p_wsel : process(all) is
        variable base : natural;
    begin
        if act_w = '1' then
            base := to_integer(d_tile) * PE_COLS;
            for c in 0 to PE_COLS - 1 loop
                if base + c < PE_ROWS then
                    o_pe_w(c) <= i_rb_data(base + c);
                else
                    o_pe_w(c) <= (others => '0');
                end if;
            end loop;
        else
            o_pe_w <= i_rom_data;
        end if;
    end process p_wsel;

    o_pe_bias  <= i_bias;
    o_pe_valid <= d_valid;
    o_pe_first <= d_first;
    o_pe_last  <= d_last;
    o_pe_rd    <= wb_m;

    -- ── Accumulate FSM ──────────────────────────────────────────────────────
    p_acc : process(clk) is
    begin
        if rising_edge(clk) then
            o_done  <= '0';
            wb_req  <= '0';
            d_valid <= '0';
            d_first <= '0';
            d_last  <= '0';

            if rstn = '0' then
                a_state <= A_IDLE;
                k_idx   <= (others => '0');
                t_idx   <= (others => '0');

            else
                case a_state is

                    when A_IDLE =>
                        if i_start = '1' then
                            k_len   <= i_k_len;
                            n_tiles <= i_n_tiles;
                            a_base  <= i_a_base;
                            w_base  <= i_w_base;
                            wa_base <= i_wa_base;
                            o_base  <= i_o_base;
                            shift_r <= i_shift;
                            act_w   <= i_act_w;
                            k_idx   <= (others => '0');
                            t_idx   <= (others => '0');
                            rom_ptr <= i_w_base;
                            a_state <= A_RUN;
                        end if;

                    when A_RUN =>
                        -- Issue the flags describing the address presented on
                        -- the previous cycle; the data for it arrives now.
                        d_tile <= t_idx;

                        if k_idx = k_len - 1 then
                            -- Closing the tile also releases it to writeback,
                            -- so wait if writeback has not finished the last.
                            if can_close = '1' then
                                d_valid <= '1';
                                d_first <= '1' when k_idx = 0 else '0';
                                d_last  <= '1';
                                wb_req  <= '1';
                                wb_tile <= t_idx;
                                k_idx   <= (others => '0');
                                rom_ptr <= rom_ptr + 1;

                                if last_tile = '1' then
                                    a_state <= A_DRAIN;
                                else
                                    t_idx <= t_idx + 1;
                                end if;
                            end if;
                        else
                            d_valid <= '1';
                            d_first <= '1' when k_idx = 0 else '0';
                            d_last  <= '0';
                            k_idx   <= k_idx + 1;
                            rom_ptr <= rom_ptr + 1;
                        end if;

                    when A_DRAIN =>
                        -- The last tile still has to be written back.
                        if wb_busy = '0' and wb_req = '0' then
                            o_done  <= '1';
                            a_state <= A_IDLE;
                        end if;

                end case;
            end if;
        end if;
    end process p_acc;

    -- ── Writeback FSM: transpose the tile, then store it ────────────────────
    p_wb : process(clk) is
        variable acc : acc_t;
    begin
        if rising_edge(clk) then
            o_wr_we <= '0';

            if rstn = '0' then
                w_state <= W_IDLE;
                wb_busy <= '0';
                wb_m    <= (others => '0');
                wb_c    <= (others => '0');

            else
                case w_state is

                    when W_IDLE =>
                        wb_busy <= '0';
                        if wb_req = '1' then
                            wb_busy <= '1';
                            wb_m    <= (others => '0');
                            w_state <= W_READ;
                        end if;

                    -- Walk the array's rows, requantising each accumulator as
                    -- it appears. The array's readout mux is combinational, so
                    -- the accumulators visible this cycle are the ones for the
                    -- row currently driven on o_pe_rd -- no pipeline delay.
                    when W_READ =>
                        for c in 0 to PE_COLS - 1 loop
                            acc := i_pe_acc(c);
                            col_buf(c)(to_integer(wb_m)) <=
                                requantize(acc, to_integer(shift_r));
                        end loop;

                        if wb_m = PE_ROWS - 1 then
                            wb_c    <= (others => '0');
                            w_state <= W_WRITE;
                        else
                            wb_m <= wb_m + 1;
                        end if;

                    -- One word per output channel of the tile.
                    when W_WRITE =>
                        o_wr_addr <= o_base +
                                     resize(wb_tile * PE_COLS + wb_c,
                                            o_wr_addr'length);
                        o_wr_data <= col_buf(to_integer(wb_c));
                        o_wr_we   <= '1';

                        if wb_c = PE_COLS - 1 then
                            wb_busy <= '0';
                            w_state <= W_IDLE;
                        else
                            wb_c <= wb_c + 1;
                        end if;

                end case;
            end if;
        end if;
    end process p_wb;

end architecture rtl;
