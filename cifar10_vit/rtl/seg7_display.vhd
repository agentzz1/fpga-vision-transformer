--------------------------------------------------------------------------------
-- seg7_display.vhd -- Basys 3 four-digit seven-segment driver
--
-- Time-multiplexes the board's four shared-cathode digits. The Basys 3 wires
-- all four digits to one set of seven segment lines and selects a digit with
-- the anode lines, so exactly one digit may be lit at a time; cycling faster
-- than the eye can follow makes all four appear lit at once.
--
-- Both seg and an are ACTIVE LOW on this board (a '0' lights a segment /
-- enables a digit).
--
-- Displays i_value as four hexadecimal nibbles, digit 0 being the rightmost.
-- i_blank blanks individual digits (bit 0 = rightmost), so the caller can show
-- a single class index without three leading zeros.
--
-- REFRESH_DIV sets the per-digit dwell time in clock cycles. The full frame is
-- 4 * REFRESH_DIV cycles; aim for a 1-2 ms frame (about 500-1000 Hz) to avoid
-- visible flicker without drawing a visible ghost on the neighbouring digit.
--     100 MHz, REFRESH_DIV = 50_000 -> 500 us/digit -> 2 ms frame -> 500 Hz
--------------------------------------------------------------------------------

library ieee;
    use ieee.std_logic_1164.all;
    use ieee.numeric_std.all;

entity seg7_display is
    generic (
        REFRESH_DIV : positive := 50_000        -- clocks per digit
    );
    port (
        clk     : in  std_logic;
        rstn    : in  std_logic;                -- synchronous, active low
        i_value : in  std_logic_vector(15 downto 0);   -- 4 hex nibbles
        i_blank : in  std_logic_vector(3 downto 0);    -- '1' blanks that digit
        o_seg   : out std_logic_vector(6 downto 0);    -- CA..CG, active low
        o_an    : out std_logic_vector(3 downto 0);    -- digit enable, active low
        o_dp    : out std_logic                        -- decimal point, active low
    );
end entity seg7_display;

architecture rtl of seg7_display is

    signal refresh_cnt : integer range 0 to REFRESH_DIV - 1 := 0;
    signal digit_sel   : unsigned(1 downto 0) := (others => '0');
    signal nibble      : std_logic_vector(3 downto 0);
    signal blank_this  : std_logic;

    -- Segment patterns, active low, bit order seg(0)=CA .. seg(6)=CG.
    function hex_to_seg (h : std_logic_vector(3 downto 0))
        return std_logic_vector is
    begin
        case h is
            when x"0"   => return "1000000";
            when x"1"   => return "1111001";
            when x"2"   => return "0100100";
            when x"3"   => return "0110000";
            when x"4"   => return "0011001";
            when x"5"   => return "0010010";
            when x"6"   => return "0000010";
            when x"7"   => return "1111000";
            when x"8"   => return "0000000";
            when x"9"   => return "0010000";
            when x"A"   => return "0001000";
            when x"B"   => return "0000011";
            when x"C"   => return "1000110";
            when x"D"   => return "0100001";
            when x"E"   => return "0000110";
            when others => return "0001110";    -- F
        end case;
    end function hex_to_seg;

begin

    o_dp <= '1';                                -- decimal point never lit

    -- Digit scan counter
    p_scan : process(clk) is
    begin
        if rising_edge(clk) then
            if rstn = '0' then
                refresh_cnt <= 0;
                digit_sel   <= (others => '0');
            elsif refresh_cnt = REFRESH_DIV - 1 then
                refresh_cnt <= 0;
                digit_sel   <= digit_sel + 1;
            else
                refresh_cnt <= refresh_cnt + 1;
            end if;
        end if;
    end process p_scan;

    -- Select the nibble and blanking bit belonging to the active digit.
    with digit_sel select nibble <=
        i_value(3  downto  0) when "00",
        i_value(7  downto  4) when "01",
        i_value(11 downto  8) when "10",
        i_value(15 downto 12) when others;

    with digit_sel select blank_this <=
        i_blank(0) when "00",
        i_blank(1) when "01",
        i_blank(2) when "10",
        i_blank(3) when others;

    -- Drive segments and anodes. Registered so the outputs are glitch-free and
    -- the segment pattern settles with the anode select rather than after it.
    p_drive : process(clk) is
    begin
        if rising_edge(clk) then
            if rstn = '0' then
                o_seg <= (others => '1');       -- all segments off
                o_an  <= (others => '1');       -- all digits off
            else
                if blank_this = '1' then
                    o_seg <= (others => '1');
                else
                    o_seg <= hex_to_seg(nibble);
                end if;

                case digit_sel is
                    when "00"   => o_an <= "1110";
                    when "01"   => o_an <= "1101";
                    when "10"   => o_an <= "1011";
                    when others => o_an <= "0111";
                end case;
            end if;
        end if;
    end process p_drive;

end architecture rtl;
