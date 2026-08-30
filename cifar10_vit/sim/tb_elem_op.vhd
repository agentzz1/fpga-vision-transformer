--------------------------------------------------------------------------------
-- tb_elem_op.vhd -- checks elem_op.vhd against the Python golden model
--
-- Exercises all four elementwise modes with vectors from
-- sw/gen_elem_vectors.py, which calls the same primitives the golden model
-- uses. The testbench stands in for the memories: it answers the module's read
-- requests from the vector data and captures its writes, so the address
-- generation and the one-cycle read latency are checked too, not just the
-- arithmetic.
--
-- File format:
--     MODE <name> <n_words>
--     <SEQ_LEN a> <SEQ_LEN b> <SEQ_LEN expected>   x n_words
--------------------------------------------------------------------------------

library ieee;
    use ieee.std_logic_1164.all;
    use ieee.numeric_std.all;

library std;
    use std.textio.all;

    use work.vit_pkg.all;

entity tb_elem_op is
end entity tb_elem_op;

architecture sim of tb_elem_op is

    constant CLK_PERIOD : time     := 10 ns;
    constant ACT_DEPTH  : positive := 640;
    constant MAX_WORDS  : positive := 256;
    constant VEC_FILE   : string   := "vectors/elem_vectors.txt";

    signal clk  : std_logic := '0';
    signal rstn : std_logic := '0';
    signal done : boolean := false;

    signal start  : std_logic := '0';
    signal mode   : std_logic_vector(1 downto 0) := (others => '0');
    signal len_s  : unsigned(clog2(ACT_DEPTH) - 1 downto 0) := (others => '0');
    signal a_base : unsigned(clog2(ACT_DEPTH) - 1 downto 0) := (others => '0');
    signal b_base : unsigned(clog2(ACT_DEPTH) - 1 downto 0) := (others => '0');
    signal o_base : unsigned(clog2(ACT_DEPTH) - 1 downto 0) := (others => '0');
    signal busy   : std_logic;
    signal edone  : std_logic;

    signal ra_addr, rb_addr : unsigned(clog2(ACT_DEPTH) - 1 downto 0);
    signal ra_re, rb_re     : std_logic;
    signal ra_data, rb_data : q8_row_t := (others => (others => '0'));

    signal pos_addr : unsigned(clog2(D_MODEL) - 1 downto 0);
    signal pos_re   : std_logic;
    signal pos_data : q8_row_t := (others => (others => '0'));

    signal wr_addr : unsigned(clog2(ACT_DEPTH) - 1 downto 0);
    signal wr_data : q8_row_t;
    signal wr_we   : std_logic;

    -- Vector storage, doubling as the memory the module reads from.
    type block_t is array (0 to MAX_WORDS - 1) of q8_row_t;
    signal a_mem  : block_t := (others => (others => (others => '0')));
    signal b_mem  : block_t := (others => (others => (others => '0')));
    signal e_mem  : block_t := (others => (others => (others => '0')));
    signal got    : block_t := (others => (others => (others => '0')));
    -- Written-word tracker. Only p_mem drives it; p_main asks for a clear via
    -- clear_wrote rather than driving the signal too, because two processes
    -- driving one resolved signal would leave it at 'X' rather than '0'.
    signal wrote       : std_logic_vector(0 to MAX_WORDS - 1) := (others => '0');
    signal clear_wrote : std_logic := '0';

begin

    clk <= not clk after CLK_PERIOD / 2 when not done else '0';

    dut : entity work.elem_op
        generic map (ACT_DEPTH => ACT_DEPTH)
        port map (
            clk        => clk,
            rstn       => rstn,
            i_start    => start,
            i_mode     => mode,
            i_len      => len_s,
            i_a_base   => a_base,
            i_b_base   => b_base,
            i_o_base   => o_base,
            o_busy     => busy,
            o_done     => edone,
            o_ra_addr  => ra_addr,
            o_ra_re    => ra_re,
            i_ra_data  => ra_data,
            o_rb_addr  => rb_addr,
            o_rb_re    => rb_re,
            i_rb_data  => rb_data,
            o_pos_addr => pos_addr,
            o_pos_re   => pos_re,
            i_pos_data => pos_data,
            o_wr_addr  => wr_addr,
            o_wr_data  => wr_data,
            o_wr_we    => wr_we
        );

    -- Stand-in memories: synchronous read, exactly like act_ram, so the
    -- module's one-cycle-ahead address generation is genuinely exercised.
    p_mem : process(clk) is
    begin
        if rising_edge(clk) then
            if ra_re = '1' and to_integer(ra_addr) < MAX_WORDS then
                ra_data <= a_mem(to_integer(ra_addr));
            end if;
            if rb_re = '1' and to_integer(rb_addr) < MAX_WORDS then
                rb_data <= b_mem(to_integer(rb_addr));
            end if;
            if pos_re = '1' then
                pos_data <= b_mem(to_integer(pos_addr));
            end if;
            if clear_wrote = '1' then
                wrote <= (others => '0');
            elsif wr_we = '1' and to_integer(wr_addr) < MAX_WORDS then
                got(to_integer(wr_addr))   <= wr_data;
                wrote(to_integer(wr_addr)) <= '1';
            end if;
        end if;
    end process p_mem;

    p_main : process is
        file     f      : text;
        variable status : file_open_status;
        variable l      : line;
        variable v      : integer;
        variable skip, is_mode : boolean;
        variable c      : character;
        variable tag_v  : string(1 to 4);
        variable code_v : integer;
        variable words  : integer;
        variable errors : natural := 0;
        variable checks : natural := 0;
        variable modes  : natural := 0;
        variable eof_v  : boolean;

        procedure next_line (variable eof : out boolean) is
        begin
            eof := false;
            loop
                if endfile(f) then
                    eof := true;
                    return;
                end if;
                readline(f, l);
                skip := true;
                is_mode := false;
                if l.all'length > 0 then
                    for i in l.all'range loop
                        if l.all(i) /= ' ' then
                            c := l.all(i);
                            skip := (c = '#');
                            is_mode := (c = 'M');
                            exit;
                        end if;
                    end loop;
                end if;
                exit when not skip;
            end loop;
        end procedure;
    begin
        rstn <= '0';
        wait for 5 * CLK_PERIOD;
        wait until rising_edge(clk);
        rstn <= '1';
        wait until rising_edge(clk);

        file_open(status, f, VEC_FILE, read_mode);
        assert status = open_ok
            report "cannot open " & VEC_FILE &
                   " (run: cd ../sw && python3 gen_elem_vectors.py)"
            severity failure;

        loop
            next_line(eof_v);
            exit when eof_v;
            next when not is_mode;

            -- "MODE <code> <n_words>". The mode is a number, not a name:
            -- VHDL's string `read` does not skip leading whitespace, so
            -- parsing names here silently reads " POS" and picks the wrong
            -- mode. Integer `read` does skip whitespace.
            read(l, tag_v);                     -- "MODE"
            read(l, code_v);
            read(l, words);
            modes := modes + 1;

            mode <= std_logic_vector(to_unsigned(code_v, 2));

            assert words <= MAX_WORDS
                report "vector block larger than MAX_WORDS" severity failure;

            for k in 0 to words - 1 loop
                next_line(eof_v);
                assert not eof_v report "vector file truncated" severity failure;
                for m in 0 to PE_ROWS - 1 loop
                    read(l, v);
                    a_mem(k)(m) <= to_signed(v, 8);
                end loop;
                for m in 0 to PE_ROWS - 1 loop
                    read(l, v);
                    b_mem(k)(m) <= to_signed(v, 8);
                end loop;
                for m in 0 to PE_ROWS - 1 loop
                    read(l, v);
                    e_mem(k)(m) <= to_signed(v, 8);
                end loop;
            end loop;
            clear_wrote <= '1';
            wait until rising_edge(clk);
            clear_wrote <= '0';
            wait until rising_edge(clk);

            -- All three regions overlap at base 0 here: the stand-in memories
            -- are separate arrays, so a/b/out do not alias the way they would
            -- in the real act_ram.
            len_s  <= to_unsigned(words, len_s'length);
            a_base <= (others => '0');
            b_base <= (others => '0');
            o_base <= (others => '0');
            wait until rising_edge(clk);

            start <= '1';
            wait until rising_edge(clk);
            start <= '0';
            wait until rising_edge(clk) and edone = '1';
            wait for 1 ns;

            for k in 0 to words - 1 loop
                assert wrote(k) = '1'
                    report "mode " & integer'image(code_v) &
                           " word " & integer'image(k) &
                           " was never written"
                    severity failure;
                for m in 0 to PE_ROWS - 1 loop
                    checks := checks + 1;
                    if got(k)(m) /= e_mem(k)(m) then
                        if errors < 15 then
                            report "MISMATCH mode " & integer'image(code_v) &
                                   " word " & integer'image(k) &
                                   " lane " & integer'image(m) &
                                   ": expected " &
                                   integer'image(to_integer(e_mem(k)(m))) &
                                   " got " &
                                   integer'image(to_integer(got(k)(m)))
                                severity error;
                        end if;
                        errors := errors + 1;
                    end if;
                end loop;
            end loop;
        end loop;
        file_close(f);

        report "modes=" & integer'image(modes) &
               " checks=" & integer'image(checks) &
               " errors=" & integer'image(errors);

        assert modes = 4
            report "TB FAIL: expected 4 modes, ran " & integer'image(modes)
            severity failure;
        assert errors = 0
            report "TB FAIL: " & integer'image(errors) & " mismatches"
            severity failure;

        report "ELEM_OP PASS: " & integer'image(checks) &
               " values bit-exact vs golden model across all 4 modes";
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
