--------------------------------------------------------------------------------
-- uart_rx.vhd -- 8N1 UART receiver, LSB-first
--
-- Clock-frequency agnostic: CLKS_PER_BIT is a generic, so the same module works
-- whatever core clock the accelerator ends up closing timing at.
--     CLKS_PER_BIT = f_clk / baud     e.g. 100e6 / 115200 = 868
--
-- Sampling: the start bit is detected on its falling edge, the counter then
-- waits half a bit period to land in the centre of the start bit, and every
-- subsequent bit is sampled one full bit period later -- i.e. at its centre.
-- This is the same scheme as the proven MNIST basys3_top.vhd receiver, lifted
-- into a standalone entity.
--
-- o_valid is a single-cycle strobe; o_data is held until the next byte lands.
--------------------------------------------------------------------------------

library ieee;
    use ieee.std_logic_1164.all;
    use ieee.numeric_std.all;

entity uart_rx is
    generic (
        CLKS_PER_BIT : positive := 868          -- 100 MHz / 115200 baud
    );
    port (
        clk     : in  std_logic;
        rstn    : in  std_logic;                -- synchronous, active low
        i_rxd   : in  std_logic;                -- serial input (already synced)
        o_data  : out std_logic_vector(7 downto 0);
        o_valid : out std_logic                 -- 1-cycle strobe on new byte
    );
end entity uart_rx;

architecture rtl of uart_rx is

    type state_t is (S_IDLE, S_START, S_DATA, S_STOP);
    signal state   : state_t := S_IDLE;

    signal clk_cnt : integer range 0 to CLKS_PER_BIT - 1 := 0;
    signal bit_cnt : integer range 0 to 7 := 0;
    signal shreg   : std_logic_vector(7 downto 0) := (others => '0');

    -- Two-flop synchroniser: i_rxd crosses from the asynchronous pad domain.
    signal rxd_meta : std_logic := '1';
    signal rxd_sync : std_logic := '1';

begin

    p_sync : process(clk) is
    begin
        if rising_edge(clk) then
            rxd_meta <= i_rxd;
            rxd_sync <= rxd_meta;
        end if;
    end process p_sync;

    p_rx : process(clk) is
    begin
        if rising_edge(clk) then
            o_valid <= '0';                     -- default: deassert strobe

            if rstn = '0' then
                state   <= S_IDLE;
                clk_cnt <= 0;
                bit_cnt <= 0;
            else
                case state is

                    -- Idle high; a low level is the start bit.
                    when S_IDLE =>
                        if rxd_sync = '0' then
                            clk_cnt <= 0;
                            state   <= S_START;
                        end if;

                    -- Wait half a bit to reach the centre of the start bit.
                    -- Re-check the line there: a glitch shorter than half a bit
                    -- is rejected rather than shifted in as data.
                    when S_START =>
                        if clk_cnt = CLKS_PER_BIT / 2 - 1 then
                            clk_cnt <= 0;
                            bit_cnt <= 0;
                            if rxd_sync = '0' then
                                state <= S_DATA;
                            else
                                state <= S_IDLE;        -- false start, abort
                            end if;
                        else
                            clk_cnt <= clk_cnt + 1;
                        end if;

                    -- Sample 8 data bits, one per bit period, at bit centre.
                    -- UART is LSB-first, so shifting right with the new bit
                    -- entering the MSB leaves bit7..bit0 correctly placed.
                    when S_DATA =>
                        if clk_cnt = CLKS_PER_BIT - 1 then
                            clk_cnt <= 0;
                            shreg   <= rxd_sync & shreg(7 downto 1);
                            if bit_cnt = 7 then
                                state <= S_STOP;
                            else
                                bit_cnt <= bit_cnt + 1;
                            end if;
                        else
                            clk_cnt <= clk_cnt + 1;
                        end if;

                    -- One bit period of stop bit, then publish the byte.
                    when S_STOP =>
                        if clk_cnt = CLKS_PER_BIT - 1 then
                            clk_cnt <= 0;
                            o_data  <= shreg;
                            o_valid <= '1';
                            state   <= S_IDLE;
                        else
                            clk_cnt <= clk_cnt + 1;
                        end if;

                end case;
            end if;
        end if;
    end process p_rx;

end architecture rtl;
