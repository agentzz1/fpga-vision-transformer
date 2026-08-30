--------------------------------------------------------------------------------
-- tb_layernorm.vhd -- checks layernorm.vhd against the Python golden model
--
-- Reads sim/vectors/layernorm_vectors.txt, produced by
-- sw/gen_layernorm_vectors.py calling `layernorm_q` directly. The testbench
-- never recomputes the expected values, so a mistake shared between the two
-- implementations cannot hide.
--
-- The cases deliberately target where this algorithm breaks: zero-variance
-- rows (which must not divide by zero), the int8 extremes, single outliers,
-- mostly-negative rows (which catch a truncating rather than flooring output
-- shift), and variances placed either side of a power of two, where the
-- leading-one detector changes the chosen shift.
--
-- File format:
--     CASE <name>
--     <D_MODEL input words>    each SEQ_LEN values, feature-major
--     <D_MODEL output words>   each SEQ_LEN values, feature-major
--------------------------------------------------------------------------------

library ieee;
    use ieee.std_logic_1164.all;
    use ieee.numeric_std.all;

library std;
    use std.textio.all;

    use work.vit_pkg.all;

entity tb_layernorm is
end entity tb_layernorm;

architecture sim of tb_layernorm is

    constant CLK_PERIOD : time   := 10 ns;
    constant VEC_FILE   : string := "vectors/layernorm_vectors.txt";

    signal clk  : std_logic := '0';
    signal rstn : std_logic := '0';
    signal done : boolean := false;

    signal start   : std_logic := '0';
    signal word_in : q8_row_t := (others => (others => '0'));
    signal valid   : std_logic := '0';
    signal pass    : std_logic;
    signal word_out : q8_row_t;
    signal ovalid  : std_logic;
    signal busy    : std_logic;
    signal ldone   : std_logic;

    type token_block_t is array (0 to D_MODEL - 1) of q8_row_t;

begin

    clk <= not clk after CLK_PERIOD / 2 when not done else '0';

    dut : entity work.layernorm
        port map (
            clk     => clk,
            rstn    => rstn,
            i_start => start,
            i_word  => word_in,
            i_valid => valid,
            o_pass  => pass,
            o_word  => word_out,
            o_valid => ovalid,
            o_busy  => busy,
            o_done  => ldone
        );

    p_main : process is
        file     f      : text;
        variable status : file_open_status;
        variable l      : line;
        variable v      : integer;
        variable skip   : boolean;
        variable is_case : boolean;
        variable name_s : string(1 to 40);
        variable inputs  : token_block_t;
        variable expect  : token_block_t;
        variable errors  : natural := 0;
        variable checks  : natural := 0;
        variable cases   : natural := 0;
        variable case_errors : natural;

        -- Read the next content line, skipping blanks and '#' comments.
        -- Sets is_case when the line begins a new CASE block.
        procedure next_line (variable eof : out boolean) is
            variable c : character;
        begin
            eof := false;
            loop
                if endfile(f) then
                    eof := true;
                    return;
                end if;
                readline(f, l);
                skip := true;
                is_case := false;
                if l.all'length > 0 then
                    for i in l.all'range loop
                        if l.all(i) /= ' ' then
                            c := l.all(i);
                            skip := (c = '#');
                            is_case := (c = 'C');
                            exit;
                        end if;
                    end loop;
                end if;
                exit when not skip;
            end loop;
        end procedure;

        variable eof_v : boolean;
    begin
        rstn <= '0';
        wait for 5 * CLK_PERIOD;
        wait until rising_edge(clk);
        rstn <= '1';
        wait until rising_edge(clk);

        file_open(status, f, VEC_FILE, read_mode);
        assert status = open_ok
            report "cannot open " & VEC_FILE &
                   " (run: cd ../sw && python3 gen_layernorm_vectors.py)"
            severity failure;

        loop
            next_line(eof_v);
            exit when eof_v;
            next when not is_case;              -- expect a CASE header here

            cases := cases + 1;
            case_errors := 0;

            -- Read the input block, then the expected output block.
            for k in 0 to D_MODEL - 1 loop
                next_line(eof_v);
                assert not eof_v report "vector file truncated" severity failure;
                for m in 0 to PE_ROWS - 1 loop
                    read(l, v);
                    inputs(k)(m) := to_signed(v, 8);
                end loop;
            end loop;
            for k in 0 to D_MODEL - 1 loop
                next_line(eof_v);
                assert not eof_v report "vector file truncated" severity failure;
                for m in 0 to PE_ROWS - 1 loop
                    read(l, v);
                    expect(k)(m) := to_signed(v, 8);
                end loop;
            end loop;

            -- ── Pass 1: stream the block so statistics accumulate ───────────
            wait until rising_edge(clk) and busy = '0';
            start <= '1';
            wait until rising_edge(clk);
            start <= '0';

            for k in 0 to D_MODEL - 1 loop
                word_in <= inputs(k);
                valid   <= '1';
                wait until rising_edge(clk);
            end loop;
            valid <= '0';

            -- ── Pass 2: re-stream once the module asks for it ───────────────
            wait until rising_edge(clk) and pass = '1';
            for k in 0 to D_MODEL - 1 loop
                word_in <= inputs(k);
                valid   <= '1';
                wait until rising_edge(clk);
                wait for 1 ns;
                if ovalid = '1' then
                    for m in 0 to PE_ROWS - 1 loop
                        checks := checks + 1;
                        if word_out(m) /= expect(k)(m) then
                            if errors < 15 then
                                report "MISMATCH case " & integer'image(cases) &
                                       " feature " & integer'image(k) &
                                       " token " & integer'image(m) &
                                       ": expected " &
                                       integer'image(to_integer(expect(k)(m))) &
                                       " got " &
                                       integer'image(to_integer(word_out(m)))
                                    severity error;
                            end if;
                            errors := errors + 1;
                            case_errors := case_errors + 1;
                        end if;
                    end loop;
                end if;
            end loop;
            valid <= '0';
            wait until rising_edge(clk) and busy = '0';
        end loop;
        file_close(f);

        report "cases=" & integer'image(cases) &
               " checks=" & integer'image(checks) &
               " errors=" & integer'image(errors);

        assert cases >= 30
            report "TB FAIL: only " & integer'image(cases) &
                   " cases read; the vector file looks truncated"
            severity failure;
        assert checks = cases * D_MODEL * PE_ROWS
            report "TB FAIL: expected " &
                   integer'image(cases * D_MODEL * PE_ROWS) &
                   " comparisons, made " & integer'image(checks) &
                   " (the module did not emit every word)"
            severity failure;
        assert errors = 0
            report "TB FAIL: " & integer'image(errors) & " mismatches"
            severity failure;

        report "LAYERNORM PASS: " & integer'image(checks) &
               " values bit-exact vs golden model across " &
               integer'image(cases) & " cases";
        done <= true;
        wait;
    end process p_main;

    p_watchdog : process is
    begin
        wait for 80 ms;
        if not done then
            report "TB FAIL: watchdog timeout" severity failure;
        end if;
        wait;
    end process p_watchdog;

end architecture sim;
