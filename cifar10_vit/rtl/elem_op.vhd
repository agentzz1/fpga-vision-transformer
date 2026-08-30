--------------------------------------------------------------------------------
-- elem_op.vhd -- the elementwise operations between matrix multiplies
--
-- Everything in the network that is not a matmul, a softmax or a LayerNorm
-- happens here. All four operations are streaming passes over the same
-- feature-major activation words, one word per cycle, so they share one address
-- generator and one writeback path instead of scattering four small state
-- machines through the controller.
--
--   E_POS    out = sat8(a + pos_embed)   positional embedding, after the stem
--   E_ADD    out = sat8(a + b)           the two residual connections
--   E_GELU   out = GELU_LUT[a]           FFN activation, one ROM read
--   E_GAP    out(0) = sat8(sum(a) >> GAP_SHIFT), rest zero
--
-- E_GAP is the odd one out: it reduces ACROSS the word rather than elementwise
-- along it. That falls out of the memory layout -- a word holds one feature
-- across all SEQ_LEN tokens, and global average pooling averages over exactly
-- those tokens, so the reduction the classifier needs is a sum of the bytes
-- within a single word. Averaging is a shift because SEQ_LEN is a power of two.
--
-- The result lands in byte 0 with the other lanes zeroed, which puts the pooled
-- vector in row 0 of the PE array for the classifier matmul. The remaining rows
-- compute against zeros and their outputs are ignored; at 192 cycles for the
-- whole classifier it is not worth special-casing the array to avoid that.
--
-- Timing: addresses are issued one cycle ahead of the data, matching the
-- synchronous memories, so a pass over N words takes N + 2 cycles.
--------------------------------------------------------------------------------

library ieee;
    use ieee.std_logic_1164.all;
    use ieee.numeric_std.all;

    use work.vit_pkg.all;

entity elem_op is
    generic (
        ACT_DEPTH : positive := 640
    );
    port (
        clk  : in std_logic;
        rstn : in std_logic;

        i_start : in std_logic;
        i_mode  : in std_logic_vector(1 downto 0);
        i_len   : in unsigned(clog2(ACT_DEPTH) - 1 downto 0);   -- words
        i_a_base : in unsigned(clog2(ACT_DEPTH) - 1 downto 0);
        i_b_base : in unsigned(clog2(ACT_DEPTH) - 1 downto 0);
        i_o_base : in unsigned(clog2(ACT_DEPTH) - 1 downto 0);

        o_busy : out std_logic;
        o_done : out std_logic;                 -- 1-cycle strobe

        -- act_ram port A (operand a)
        o_ra_addr : out unsigned(clog2(ACT_DEPTH) - 1 downto 0);
        o_ra_re   : out std_logic;
        i_ra_data : in  q8_row_t;

        -- act_ram port B (operand b, E_ADD only)
        o_rb_addr : out unsigned(clog2(ACT_DEPTH) - 1 downto 0);
        o_rb_re   : out std_logic;
        i_rb_data : in  q8_row_t;

        -- positional embedding ROM (E_POS only), same word layout
        o_pos_addr : out unsigned(clog2(D_MODEL) - 1 downto 0);
        o_pos_re   : out std_logic;
        i_pos_data : in  q8_row_t;

        -- act_ram write port
        o_wr_addr : out unsigned(clog2(ACT_DEPTH) - 1 downto 0);
        o_wr_data : out q8_row_t;
        o_wr_we   : out std_logic
    );
end entity elem_op;

architecture rtl of elem_op is

    subtype addr_t is unsigned(clog2(ACT_DEPTH) - 1 downto 0);

    type state_t is (E_IDLE, E_RUN, E_DRAIN);
    signal state : state_t := E_IDLE;

    signal mode   : std_logic_vector(1 downto 0) := (others => '0');
    signal len_r  : addr_t := (others => '0');
    signal a_base : addr_t := (others => '0');
    signal b_base : addr_t := (others => '0');
    signal o_base : addr_t := (others => '0');

    signal idx   : addr_t := (others => '0');    -- word being addressed
    signal d_idx : addr_t := (others => '0');    -- word whose data is present
    signal d_val : std_logic := '0';

begin

    o_busy <= '0' when state = E_IDLE else '1';

    o_ra_addr <= a_base + idx;
    o_rb_addr <= b_base + idx;
    -- The positional ROM is indexed by feature, and the stem writes exactly
    -- D_MODEL words, so the low bits of the word index address it directly.
    o_pos_addr <= resize(idx, o_pos_addr'length);

    o_ra_re  <= '1' when state = E_RUN else '0';
    o_rb_re  <= '1' when (state = E_RUN and mode = ELEM_ADD) else '0';
    o_pos_re <= '1' when (state = E_RUN and mode = ELEM_POS) else '0';

    p_elem : process(clk) is
        variable sum  : signed(15 downto 0);
        variable acc  : signed(15 downto 0);
        variable byte : integer range 0 to 255;
    begin
        if rising_edge(clk) then
            o_wr_we <= '0';
            o_done  <= '0';

            if rstn = '0' then
                state <= E_IDLE;
                idx   <= (others => '0');
                d_val <= '0';

            else
                case state is

                    when E_IDLE =>
                        d_val <= '0';
                        if i_start = '1' then
                            mode   <= i_mode;
                            len_r  <= i_len;
                            a_base <= i_a_base;
                            b_base <= i_b_base;
                            o_base <= i_o_base;
                            idx    <= (others => '0');
                            state  <= E_RUN;
                        end if;

                    when E_RUN =>
                        -- Advance the address stream, remembering which word
                        -- the data arriving next cycle belongs to.
                        d_idx <= idx;
                        d_val <= '1';
                        if idx = len_r - 1 then
                            state <= E_DRAIN;
                        else
                            idx <= idx + 1;
                        end if;

                    when E_DRAIN =>
                        -- One extra cycle so the final word's data lands.
                        d_val <= '0';
                        if d_val = '0' then
                            o_done <= '1';
                            state  <= E_IDLE;
                        end if;

                end case;

                -- Compute and write whenever a fetched word is available. This
                -- runs in both E_RUN and E_DRAIN, which is what lets the last
                -- word be written after address generation has finished.
                if d_val = '1' then
                    o_wr_addr <= o_base + d_idx;
                    o_wr_we   <= '1';

                    case mode is

                        when ELEM_POS =>
                            for m in 0 to PE_ROWS - 1 loop
                                o_wr_data(m) <= add_sat8(i_ra_data(m),
                                                         i_pos_data(m));
                            end loop;

                        when ELEM_ADD =>
                            for m in 0 to PE_ROWS - 1 loop
                                o_wr_data(m) <= add_sat8(i_ra_data(m),
                                                         i_rb_data(m));
                            end loop;

                        when ELEM_GELU =>
                            for m in 0 to PE_ROWS - 1 loop
                                -- Index by the raw bit pattern, so a negative
                                -- activation selects entry 256+v exactly as
                                -- the reference's `x & 0xFF` does.
                                byte := to_integer(unsigned(std_logic_vector(
                                            i_ra_data(m))));
                                o_wr_data(m) <= to_signed(GELU_LUT(byte), 8);
                            end loop;

                        when others =>          -- ELEM_GAP
                            acc := (others => '0');
                            for m in 0 to PE_ROWS - 1 loop
                                acc := acc + resize(i_ra_data(m), 16);
                            end loop;
                            sum := shift_right(acc, GAP_SHIFT);
                            if sum > to_signed(INT8_MAX, 16) then
                                o_wr_data(0) <= to_signed(INT8_MAX, 8);
                            elsif sum < to_signed(INT8_MIN, 16) then
                                o_wr_data(0) <= to_signed(INT8_MIN, 8);
                            else
                                o_wr_data(0) <= resize(sum, 8);
                            end if;
                            for m in 1 to PE_ROWS - 1 loop
                                o_wr_data(m) <= (others => '0');
                            end loop;

                    end case;
                end if;
            end if;
        end if;
    end process p_elem;

end architecture rtl;
