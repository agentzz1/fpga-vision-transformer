--------------------------------------------------------------------------------
-- tb_pe_array.vhd -- self-checking test for the output-stationary MAC array
--
-- Drives the array with pseudo-random operands and compares every accumulator
-- against an independently computed reference sum. The reference is a plain
-- nested loop written straight from the definition of a matrix product, so it
-- shares no logic with the design under test -- a control or dataflow bug in
-- the array cannot hide by being mirrored in the reference.
--
-- Covers:
--   * three K values, including the largest the design uses (PATCH_DIM = 192)
--     and the smallest (SEQ_LEN = 16, the attention context product)
--   * bias preload on the first cycle
--   * back-to-back tiles with no idle cycle, which is what the double-buffered
--     accumulators exist to allow -- if the banks were swapped incorrectly,
--     tile N would read tile N-1's data and the check would fail
--   * stalling (i_valid low) mid-tile, which must not disturb accumulation
--------------------------------------------------------------------------------

library ieee;
    use ieee.std_logic_1164.all;
    use ieee.numeric_std.all;

    use work.vit_pkg.all;

entity tb_pe_array is
end entity tb_pe_array;

architecture sim of tb_pe_array is

    constant CLK_PERIOD : time := 10 ns;

    signal clk  : std_logic := '0';
    signal rstn : std_logic := '0';

    signal a_in    : q8_row_t  := (others => (others => '0'));
    signal w_in    : q8_col_t  := (others => (others => '0'));
    signal bias_in : acc_col_t := (others => (others => '0'));
    signal valid   : std_logic := '0';
    signal first   : std_logic := '0';
    signal last    : std_logic := '0';

    signal tile_valid : std_logic;
    signal rd_row     : unsigned(clog2(PE_ROWS) - 1 downto 0) := (others => '0');
    signal acc_out    : acc_col_t;

    signal done : boolean := false;

    -- Reference storage: the operands actually presented to the array, kept so
    -- the expected result can be recomputed from scratch after the fact.
    -- Process variables rather than signals, so a value written while driving
    -- cycle k is readable immediately instead of a delta cycle later.
    type a_mat_t is array (0 to 255, 0 to PE_ROWS - 1) of integer;
    type w_mat_t is array (0 to 255, 0 to PE_COLS - 1) of integer;

begin

    clk <= not clk after CLK_PERIOD / 2 when not done else '0';

    dut : entity work.pe_array
        port map (
            clk          => clk,
            rstn         => rstn,
            i_a          => a_in,
            i_w          => w_in,
            i_bias       => bias_in,
            i_valid      => valid,
            i_first      => first,
            i_last       => last,
            o_tile_valid => tile_valid,
            i_rd_row     => rd_row,
            o_acc        => acc_out
        );

    p_main : process is
        -- Counters must be variables: `sig <= sig + 1` inside a loop reads the
        -- same pre-delta value every iteration, so a signal would count one
        -- increment per delta cycle instead of one per check.
        variable errors : natural := 0;
        variable checks : natural := 0;

        variable a_hist : a_mat_t;
        variable w_hist : w_mat_t;

        -- Deterministic pseudo-random source. A maximal-length 32-bit LFSR
        -- rather than a multiplicative generator: VHDL's integer is 32-bit
        -- signed, so `seed * 1103515245` would overflow and abort the run.
        variable lfsr : unsigned(31 downto 0) := x"12345678";

        impure function rnd (lo, hi : integer) return integer is
            variable fb : std_logic;
        begin
            for i in 1 to 8 loop            -- a few steps between draws
                fb   := lfsr(31) xor lfsr(21) xor lfsr(1) xor lfsr(0);
                lfsr := lfsr(30 downto 0) & fb;
            end loop;
            return lo + (to_integer(lfsr(23 downto 8)) mod (hi - lo + 1));
        end function;

        -- Run one tile of K cycles, recording operands as it goes.
        -- stall_at >= 0 inserts an idle cycle before that k, to prove that
        -- deasserting i_valid pauses rather than corrupts the accumulation.
        procedure run_tile (k_len : integer; use_bias : boolean;
                            stall_at : integer) is
            variable b : integer;
        begin
            -- Bias is held for the whole tile; the array only samples it on
            -- the first cycle.
            for n in 0 to PE_COLS - 1 loop
                if use_bias then
                    b := rnd(-100, 100) * Q_SCALE;
                else
                    b := 0;
                end if;
                bias_in(n) <= to_signed(b, ACC_BITS);
            end loop;

            for k in 0 to k_len - 1 loop
                if k = stall_at then
                    valid <= '0';
                    first <= '0';
                    last  <= '0';
                    wait until rising_edge(clk);
                end if;

                for m in 0 to PE_ROWS - 1 loop
                    a_hist(k, m) := rnd(-128, 127);
                end loop;
                for n in 0 to PE_COLS - 1 loop
                    w_hist(k, n) := rnd(-128, 127);
                end loop;

                for m in 0 to PE_ROWS - 1 loop
                    a_in(m) <= to_signed(a_hist(k, m), 8);
                end loop;
                for n in 0 to PE_COLS - 1 loop
                    w_in(n) <= to_signed(w_hist(k, n), 8);
                end loop;

                valid <= '1';
                if k = 0 then first <= '1'; else first <= '0'; end if;
                if k = k_len - 1 then last <= '1'; else last <= '0'; end if;

                wait until rising_edge(clk);
            end loop;

            valid <= '0';
            first <= '0';
            last  <= '0';
        end procedure;

        -- Recompute the tile from the recorded operands and compare.
        procedure check_tile (k_len : integer; use_bias : boolean;
                              tag : string) is
            variable expected : integer;
            variable got      : integer;
        begin
            -- The DUT schedules tile_valid and the bank swap on the same
            -- rising edge this process just woke on, so let those updates
            -- settle before sampling them.
            wait for 1 ns;

            assert tile_valid = '1'
                report tag & ": array never reported a completed tile"
                severity failure;

            for m in 0 to PE_ROWS - 1 loop
                rd_row <= to_unsigned(m, rd_row'length);
                wait for 1 ns;              -- combinational readout mux
                for n in 0 to PE_COLS - 1 loop
                    expected := 0;
                    if use_bias then
                        expected := to_integer(bias_in(n));
                    end if;
                    for k in 0 to k_len - 1 loop
                        expected := expected + a_hist(k, m) * w_hist(k, n);
                    end loop;

                    got := to_integer(acc_out(n));
                    checks := checks + 1;
                    if got /= expected then
                        report tag & ": acc(" & integer'image(m) & "," &
                               integer'image(n) & ") expected " &
                               integer'image(expected) & " got " &
                               integer'image(got)
                            severity error;
                        errors := errors + 1;
                    end if;
                end loop;
            end loop;
        end procedure;

    begin
        rstn <= '0';
        wait for 5 * CLK_PERIOD;
        wait until rising_edge(clk);
        rstn <= '1';
        wait until rising_edge(clk);

        -- K = 64: the most common shape (QKV, out-proj, FC1, scores).
        run_tile(64, false, -1);
        check_tile(64, false, "K=64 no-bias");

        -- Bias preload.
        run_tile(64, true, -1);
        check_tile(64, true, "K=64 bias");

        -- K = PATCH_DIM: the widest accumulation, worst case for overflow.
        run_tile(PATCH_DIM, true, -1);
        check_tile(PATCH_DIM, true, "K=PATCH_DIM bias");

        -- K = SEQ_LEN: the narrowest, used by the attention context product.
        run_tile(SEQ_LEN, false, -1);
        check_tile(SEQ_LEN, false, "K=SEQ_LEN");

        -- A stall in the middle must pause, not corrupt.
        run_tile(64, true, 30);
        check_tile(64, true, "K=64 stalled");

        -- Back-to-back tiles with no gap: the second tile starts accumulating
        -- in the other bank on the cycle right after the first finishes.
        run_tile(32, true, -1);
        check_tile(32, true, "back-to-back tile 1");
        run_tile(32, true, -1);
        check_tile(32, true, "back-to-back tile 2");

        report "checks=" & integer'image(checks) &
               " errors=" & integer'image(errors);
        assert errors = 0
            report "TB FAIL: " & integer'image(errors) & " accumulator mismatches"
            severity failure;
        assert checks = 7 * PE_ROWS * PE_COLS
            report "TB FAIL: expected " & integer'image(7 * PE_ROWS * PE_COLS) &
                   " checks, ran " & integer'image(checks)
            severity failure;

        report "PE_ARRAY PASS: " & integer'image(checks) &
               " accumulators bit-exact across 7 tiles";
        done <= true;
        wait;
    end process p_main;

    p_watchdog : process is
    begin
        wait for 5 ms;
        if not done then
            report "TB FAIL: watchdog timeout" severity failure;
        end if;
        wait;
    end process p_watchdog;

end architecture sim;
