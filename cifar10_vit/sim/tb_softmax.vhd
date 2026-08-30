--------------------------------------------------------------------------------
-- tb_softmax.vhd -- checks softmax.vhd against the Python golden model
--
-- Reads sim/vectors/softmax_vectors.txt, produced by sw/gen_softmax_vectors.py
-- by calling `softmax_q` directly. The testbench never recomputes the expected
-- values itself: a second copy of the arithmetic here could repeat the same
-- mistake as the RTL and cancel it out.
--
-- The vector set is adversarial by design -- constant rows, single spikes,
-- rows straddling the table's magnitude clamp, and a sweep that reaches every
-- one of the 256 possible magnitudes and therefore every reachable table
-- entry and step boundary.
--
-- File format, one vector per line, '#' comments and blank lines skipped:
--     <SEQ_LEN int8 scores>  <SEQ_LEN int8 expected probabilities>
--------------------------------------------------------------------------------

library ieee;
    use ieee.std_logic_1164.all;
    use ieee.numeric_std.all;

library std;
    use std.textio.all;

    use work.vit_pkg.all;

entity tb_softmax is
end entity tb_softmax;

architecture sim of tb_softmax is

    constant CLK_PERIOD : time   := 10 ns;
    constant VEC_FILE   : string := "vectors/softmax_vectors.txt";

    signal clk  : std_logic := '0';
    signal rstn : std_logic := '0';
    signal done : boolean := false;

    signal row_in  : q8_row_t := (others => (others => '0'));
    signal valid   : std_logic := '0';
    signal row_out : q8_row_t;
    signal ovalid  : std_logic;
    signal busy    : std_logic;

begin

    clk <= not clk after CLK_PERIOD / 2 when not done else '0';

    dut : entity work.softmax
        port map (
            clk     => clk,
            rstn    => rstn,
            i_row   => row_in,
            i_valid => valid,
            o_row   => row_out,
            o_valid => ovalid,
            o_busy  => busy
        );

    p_main : process is
        file     f      : text;
        variable status : file_open_status;
        variable l      : line;
        variable v      : integer;
        variable skip   : boolean;
        variable expect : q8_row_t;
        variable errors : natural := 0;
        variable vectors_run : natural := 0;
        variable checks : natural := 0;
        variable sum    : integer;
        variable worst_deficit : integer := 0;
    begin
        rstn <= '0';
        wait for 5 * CLK_PERIOD;
        wait until rising_edge(clk);
        rstn <= '1';
        wait until rising_edge(clk);

        file_open(status, f, VEC_FILE, read_mode);
        assert status = open_ok
            report "cannot open " & VEC_FILE &
                   " (run: cd ../sw && python3 gen_softmax_vectors.py)"
            severity failure;

        while not endfile(f) loop
            readline(f, l);

            -- Skip blank lines and '#' comments. The first non-space
            -- character is inspected in place; `read` for an integer skips
            -- leading whitespace on its own, so nothing has to be consumed
            -- and pushed back.
            skip := true;
            if l.all'length > 0 then
                for i in l.all'range loop
                    if l.all(i) /= ' ' then
                        skip := (l.all(i) = '#');
                        exit;
                    end if;
                end loop;
            end if;
            next when skip;

            for i in 0 to PE_ROWS - 1 loop
                read(l, v);
                row_in(i) <= to_signed(v, 8);
            end loop;
            for i in 0 to PE_ROWS - 1 loop
                read(l, v);
                expect(i) := to_signed(v, 8);
            end loop;

            -- Drive one row and wait for the result.
            wait until rising_edge(clk) and busy = '0';
            valid <= '1';
            wait until rising_edge(clk);
            valid <= '0';
            wait until rising_edge(clk) and ovalid = '1';
            wait for 1 ns;

            vectors_run := vectors_run + 1;
            sum := 0;
            for i in 0 to PE_ROWS - 1 loop
                checks := checks + 1;
                sum := sum + to_integer(row_out(i));
                if row_out(i) /= expect(i) then
                    if errors < 15 then
                        report "MISMATCH vector " & integer'image(vectors_run) &
                               " element " & integer'image(i) &
                               ": expected " & integer'image(to_integer(expect(i))) &
                               " got " & integer'image(to_integer(row_out(i)))
                            severity error;
                    end if;
                    errors := errors + 1;
                end if;
            end loop;

            -- Track how far the row sum falls below Q_SCALE. Each element is
            -- floored, so a deficit of up to SEQ_LEN is expected and bounded;
            -- anything larger would mean the normalisation is wrong even if
            -- it happened to match the reference.
            if Q_SCALE - sum > worst_deficit then
                worst_deficit := Q_SCALE - sum;
            end if;
            assert sum <= Q_SCALE
                report "vector " & integer'image(vectors_run) &
                       " sums to " & integer'image(sum) & ", above Q_SCALE"
                severity error;
        end loop;
        file_close(f);

        report "vectors=" & integer'image(vectors_run) &
               " checks=" & integer'image(checks) &
               " errors=" & integer'image(errors) &
               " worst row-sum deficit=" & integer'image(worst_deficit);

        assert vectors_run > 400
            report "TB FAIL: only " & integer'image(vectors_run) &
                   " vectors read; the vector file looks truncated"
            severity failure;
        assert errors = 0
            report "TB FAIL: " & integer'image(errors) & " mismatches"
            severity failure;
        assert worst_deficit <= PE_ROWS
            report "TB FAIL: row-sum deficit " & integer'image(worst_deficit) &
                   " exceeds the one-LSB-per-element bound"
            severity failure;

        report "SOFTMAX PASS: " & integer'image(checks) &
               " values bit-exact vs golden model across " &
               integer'image(vectors_run) & " vectors";
        done <= true;
        wait;
    end process p_main;

    p_watchdog : process is
    begin
        wait for 60 ms;
        if not done then
            report "TB FAIL: watchdog timeout" severity failure;
        end if;
        wait;
    end process p_watchdog;

end architecture sim;
