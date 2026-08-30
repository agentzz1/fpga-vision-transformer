--------------------------------------------------------------------------------
-- tb_gemm_seq.vhd -- integration test: sequencer + PE array + memories
--
-- Builds the real datapath (gemm_seq driving pe_array, reading act_ram and
-- weight_rom, writing results back into act_ram) and checks the result of a
-- complete matrix multiply against vectors produced by the Python golden
-- model's `matmul_q`.
--
-- This is the test that matters: the PE array on its own was already shown to
-- accumulate correctly, but that says nothing about address generation,
-- operand alignment, the one-cycle memory read latency, requantisation, the
-- transpose on writeback, or the accumulator bank swap. All of those are
-- exercised here, and all of them have to be right for a single output word to
-- match.
--
-- Vectors come from sw/gen_gemm_vectors.py:
--     sim/vectors/gemm_act.txt     activations, feature-major, one word/line
--     sim/vectors/gemm_rom.hex     weight ROM image, one byte/line
--     sim/vectors/gemm_bias.txt    bias per output channel, pre-scaled
--     sim/vectors/gemm_expect.txt  expected output words
--     sim/vectors/gemm_meta.txt    K, tile count, requantisation shift
--------------------------------------------------------------------------------

library ieee;
    use ieee.std_logic_1164.all;
    use ieee.numeric_std.all;

library std;
    use std.textio.all;

    use work.vit_pkg.all;

entity tb_gemm_seq is
end entity tb_gemm_seq;

architecture sim of tb_gemm_seq is

    constant CLK_PERIOD : time     := 10 ns;
    constant ACT_DEPTH  : positive := 512;
    constant ROM_DEPTH  : positive := 1024;     -- enough for the test case

    constant VEC_DIR : string := "vectors/";

    -- Where the test places the operands and expects the results.
    constant ACT_IN_BASE : natural := 0;
    constant ACT_OUT_BASE : natural := 256;

    signal clk  : std_logic := '0';
    signal rstn : std_logic := '0';
    signal done : boolean := false;

    -- ── gemm_seq job interface ──────────────────────────────────────────────
    signal start   : std_logic := '0';
    signal k_len   : unsigned(clog2(ACT_DEPTH) - 1 downto 0) := (others => '0');
    signal n_tiles : unsigned(clog2(ACT_DEPTH) - 1 downto 0) := (others => '0');
    signal a_base  : unsigned(clog2(ACT_DEPTH) - 1 downto 0) :=
                     to_unsigned(ACT_IN_BASE, clog2(ACT_DEPTH));
    signal w_base  : unsigned(clog2(ROM_DEPTH) - 1 downto 0) := (others => '0');
    signal wa_base : unsigned(clog2(ACT_DEPTH) - 1 downto 0) := (others => '0');
    signal o_base  : unsigned(clog2(ACT_DEPTH) - 1 downto 0) :=
                     to_unsigned(ACT_OUT_BASE, clog2(ACT_DEPTH));
    signal shift_s : unsigned(4 downto 0) := (others => '0');
    signal act_w   : std_logic := '0';
    signal busy    : std_logic;
    signal gdone   : std_logic;

    signal cur_tile : unsigned(clog2(ACT_DEPTH) - 1 downto 0);
    signal bias_s   : acc_col_t := (others => (others => '0'));

    -- ── act_ram wiring (two read ports = two identically written copies) ────
    signal ra_addr, rb_addr : unsigned(clog2(ACT_DEPTH) - 1 downto 0);
    signal ra_re, rb_re     : std_logic;
    signal ra_data, rb_data : q8_row_t;

    -- Port B is the sequencer's attention-weight path during a run and the
    -- checker's readback path afterwards, so both drivers are muxed onto it.
    signal seq_rb_addr : unsigned(clog2(ACT_DEPTH) - 1 downto 0);
    signal seq_rb_re   : std_logic;

    signal wr_addr : unsigned(clog2(ACT_DEPTH) - 1 downto 0);
    signal wr_data : q8_row_t;
    signal wr_we   : std_logic;

    -- The testbench preloads activations, so the write port is muxed between
    -- the testbench (during load) and the sequencer (during the run).
    signal tb_wr_addr : unsigned(clog2(ACT_DEPTH) - 1 downto 0) := (others => '0');
    signal tb_wr_data : q8_row_t := (others => (others => '0'));
    signal tb_wr_we   : std_logic := '0';
    signal loading    : std_logic := '1';

    signal ram_waddr : unsigned(clog2(ACT_DEPTH) - 1 downto 0);
    signal ram_wdata : q8_row_t;
    signal ram_we    : std_logic;

    -- Readback port, driven by the testbench after the run.
    signal chk_addr : unsigned(clog2(ACT_DEPTH) - 1 downto 0) := (others => '0');
    signal chk_re   : std_logic := '0';

    -- ── weight ROM ──────────────────────────────────────────────────────────
    signal rom_addr : unsigned(clog2(ROM_DEPTH) - 1 downto 0);
    signal rom_re   : std_logic;
    signal rom_data : q8_col_t;

    -- ── PE array ────────────────────────────────────────────────────────────
    signal pe_a     : q8_row_t;
    signal pe_w     : q8_col_t;
    signal pe_bias  : acc_col_t;
    signal pe_valid : std_logic;
    signal pe_first : std_logic;
    signal pe_last  : std_logic;
    signal pe_rd    : unsigned(clog2(PE_ROWS) - 1 downto 0);
    signal pe_acc   : acc_col_t;
    signal pe_tv    : std_logic;

    -- Bias table, indexed by tile.
    type bias_tab_t is array (0 to 255) of acc_col_t;
    signal bias_tab : bias_tab_t := (others => (others => (others => '0')));

begin

    clk <= not clk after CLK_PERIOD / 2 when not done else '0';

    -- Write-port mux: testbench during preload, sequencer during the run.
    ram_waddr <= tb_wr_addr when loading = '1' else wr_addr;
    ram_wdata <= tb_wr_data when loading = '1' else wr_data;
    ram_we    <= tb_wr_we   when loading = '1' else wr_we;

    -- Two copies of the activation memory so the activation operand and the
    -- attention weight operand can be fetched in the same cycle. Both see
    -- identical writes, so they always hold identical contents.
    u_act_a : entity work.act_ram
        generic map (DEPTH => ACT_DEPTH)
        port map (
            clk     => clk,
            i_raddr => ra_addr,
            i_re    => ra_re,
            o_rdata => ra_data,
            i_waddr => ram_waddr,
            i_wdata => ram_wdata,
            i_we    => ram_we
        );

    u_act_b : entity work.act_ram
        generic map (DEPTH => ACT_DEPTH)
        port map (
            clk     => clk,
            i_raddr => rb_addr,
            i_re    => rb_re,
            o_rdata => rb_data,
            i_waddr => ram_waddr,
            i_wdata => ram_wdata,
            i_we    => ram_we
        );

    -- Port B doubles as the checker's readback path once the run is over.
    rb_addr <= chk_addr when chk_re = '1' else seq_rb_addr;
    rb_re   <= chk_re or seq_rb_re;

    u_rom : entity work.weight_rom
        generic map (DEPTH => ROM_DEPTH, INIT_FILE => VEC_DIR & "gemm_rom.hex")
        port map (
            clk     => clk,
            i_addr  => rom_addr,
            i_re    => rom_re,
            o_wdata => rom_data
        );

    u_pe : entity work.pe_array
        port map (
            clk          => clk,
            rstn         => rstn,
            i_a          => pe_a,
            i_w          => pe_w,
            i_bias       => pe_bias,
            i_valid      => pe_valid,
            i_first      => pe_first,
            i_last       => pe_last,
            o_tile_valid => pe_tv,
            i_rd_row     => pe_rd,
            o_acc        => pe_acc
        );

    bias_s <= bias_tab(to_integer(cur_tile));

    u_seq : entity work.gemm_seq
        generic map (ACT_DEPTH => ACT_DEPTH, ROM_DEPTH => ROM_DEPTH)
        port map (
            clk       => clk,
            rstn      => rstn,
            i_start   => start,
            i_k_len   => k_len,
            i_n_tiles => n_tiles,
            i_a_base  => a_base,
            i_w_base  => w_base,
            i_wa_base => wa_base,
            i_o_base  => o_base,
            i_shift   => shift_s,
            i_act_w   => act_w,
            o_busy    => busy,
            o_done    => gdone,
            o_tile    => cur_tile,
            i_bias    => bias_s,
            o_ra_addr => ra_addr,
            o_ra_re   => ra_re,
            i_ra_data => ra_data,
            o_rb_addr => seq_rb_addr,
            o_rb_re   => seq_rb_re,
            i_rb_data => rb_data,
            o_wr_addr => wr_addr,
            o_wr_data => wr_data,
            o_wr_we   => wr_we,
            o_rom_addr => rom_addr,
            o_rom_re   => rom_re,
            i_rom_data => rom_data,
            o_pe_a     => pe_a,
            o_pe_w     => pe_w,
            o_pe_bias  => pe_bias,
            o_pe_valid => pe_valid,
            o_pe_first => pe_first,
            o_pe_last  => pe_last,
            o_pe_rd    => pe_rd,
            i_pe_acc   => pe_acc
        );

    p_main : process is
        file     f      : text;
        variable status : file_open_status;
        variable l      : line;
        variable v      : integer;
        variable errors : natural := 0;
        variable checks : natural := 0;
        variable k_v, tiles_v, shift_v : integer;
        variable n_words : integer;
        variable cycles  : natural := 0;

        procedure open_vec (file fh : text; name : string) is
        begin
            file_open(status, fh, VEC_DIR & name, read_mode);
            assert status = open_ok
                report "cannot open vector file " & VEC_DIR & name &
                       " (run: cd ../sw && python3 gen_gemm_vectors.py)"
                severity failure;
        end procedure;
    begin
        rstn    <= '0';
        loading <= '1';
        wait for 5 * CLK_PERIOD;
        wait until rising_edge(clk);
        rstn <= '1';
        wait until rising_edge(clk);

        -- ── Read the job description ────────────────────────────────────────
        open_vec(f, "gemm_meta.txt");
        readline(f, l); read(l, k_v);
        readline(f, l); read(l, tiles_v);
        readline(f, l); read(l, shift_v);
        file_close(f);
        report "job: K=" & integer'image(k_v) &
               " tiles=" & integer'image(tiles_v) &
               " shift=" & integer'image(shift_v);

        k_len   <= to_unsigned(k_v, k_len'length);
        n_tiles <= to_unsigned(tiles_v, n_tiles'length);
        shift_s <= to_unsigned(shift_v, shift_s'length);

        -- ── Preload activations, one feature-major word per line ────────────
        open_vec(f, "gemm_act.txt");
        for kk in 0 to k_v - 1 loop
            readline(f, l);
            for m in 0 to PE_ROWS - 1 loop
                read(l, v);
                tb_wr_data(m) <= to_signed(v, 8);
            end loop;
            tb_wr_addr <= to_unsigned(ACT_IN_BASE + kk, tb_wr_addr'length);
            tb_wr_we   <= '1';
            wait until rising_edge(clk);
        end loop;
        tb_wr_we <= '0';
        file_close(f);

        -- ── Load the bias table, PE_COLS entries per tile ───────────────────
        open_vec(f, "gemm_bias.txt");
        for t in 0 to tiles_v - 1 loop
            for c in 0 to PE_COLS - 1 loop
                readline(f, l); read(l, v);
                bias_tab(t)(c) <= to_signed(v, ACC_BITS);
            end loop;
        end loop;
        file_close(f);
        wait until rising_edge(clk);

        -- ── Run the matmul ──────────────────────────────────────────────────
        loading <= '0';
        wait until rising_edge(clk);
        start <= '1';
        wait until rising_edge(clk);
        start <= '0';

        while gdone = '0' loop
            wait until rising_edge(clk);
            cycles := cycles + 1;
            assert cycles < 200000
                report "TB FAIL: matmul never completed" severity failure;
        end loop;
        report "matmul completed in " & integer'image(cycles) & " cycles";

        wait until rising_edge(clk);

        -- ── Compare every output word against the golden model ──────────────
        n_words := tiles_v * PE_COLS;
        open_vec(f, "gemm_expect.txt");
        for nn in 0 to n_words - 1 loop
            chk_addr <= to_unsigned(ACT_OUT_BASE + nn, chk_addr'length);
            chk_re   <= '1';
            wait until rising_edge(clk);
            wait for 1 ns;                  -- synchronous read latency

            readline(f, l);
            for m in 0 to PE_ROWS - 1 loop
                read(l, v);
                checks := checks + 1;
                if to_integer(rb_data(m)) /= v then
                    if errors < 20 then
                        report "MISMATCH word " & integer'image(nn) &
                               " token " & integer'image(m) &
                               ": expected " & integer'image(v) &
                               " got " & integer'image(to_integer(rb_data(m)))
                            severity error;
                    end if;
                    errors := errors + 1;
                end if;
            end loop;
        end loop;
        chk_re <= '0';
        file_close(f);

        report "checked " & integer'image(checks) & " values, " &
               integer'image(errors) & " mismatches";
        assert errors = 0
            report "TB FAIL: " & integer'image(errors) &
                   " values differ from the golden model"
            severity failure;
        assert checks = n_words * PE_ROWS
            report "TB FAIL: wrong number of comparisons" severity failure;

        report "GEMM_SEQ PASS: " & integer'image(checks) &
               " outputs bit-exact vs golden model, " &
               integer'image(cycles) & " cycles";
        done <= true;
        wait;
    end process p_main;

    p_watchdog : process is
    begin
        wait for 40 ms;
        if not done then
            report "TB FAIL: watchdog timeout" severity failure;
        end if;
        wait;
    end process p_watchdog;

end architecture sim;
