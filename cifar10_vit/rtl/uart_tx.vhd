--------------------------------------------------------------------------------
-- uart_tx.vhd -- 8N1 UART transmitter, LSB-first
--
-- Companion to uart_rx.vhd; same CLKS_PER_BIT generic convention.
--     CLKS_PER_BIT = f_clk / baud     e.g. 100e6 / 115200 = 868
--
-- Handshake: assert i_send for one cycle with i_data valid while o_busy is low.
-- o_busy rises in the same cycle the byte is accepted and stays high until the
-- stop bit has finished, so a caller can simply gate on "not o_busy".
-- Requests asserted while o_busy is high are ignored (not queued).
--------------------------------------------------------------------------------

library ieee;
    use ieee.std_logic_1164.all;
    use ieee.numeric_std.all;

entity uart_tx is
    generic (
        CLKS_PER_BIT : positive := 868          -- 100 MHz / 115200 baud
    );
    port (
        clk    : in  std_logic;
        rstn   : in  std_logic;                 -- synchronous, active low
        i_data : in  std_logic_vector(7 downto 0);
        i_send : in  std_logic;                 -- 1-cycle request strobe
        o_txd  : out std_logic;                 -- serial output (idles high)
        o_busy : out std_logic
    );
end entity uart_tx;

architecture rtl of uart_tx is

    type state_t is (S_IDLE, S_START, S_DATA, S_STOP);
    signal state   : state_t := S_IDLE;

    signal clk_cnt : integer range 0 to CLKS_PER_BIT - 1 := 0;
    signal bit_cnt : integer range 0 to 7 := 0;
    signal shreg   : std_logic_vector(7 downto 0) := (others => '1');
    signal txd_r   : std_logic := '1';          -- line idles high
    signal busy_r  : std_logic := '0';

begin

    o_txd  <= txd_r;
    o_busy <= busy_r;

    p_tx : process(clk) is
    begin
        if rising_edge(clk) then
            if rstn = '0' then
                state   <= S_IDLE;
                clk_cnt <= 0;
                bit_cnt <= 0;
                shreg   <= (others => '1');
                txd_r   <= '1';
                busy_r  <= '0';
            else
                case state is

                    when S_IDLE =>
                        txd_r  <= '1';
                        if i_send = '1' then
                            shreg   <= i_data;
                            clk_cnt <= 0;
                            bit_cnt <= 0;
                            txd_r   <= '0';         -- drive the start bit now
                            busy_r  <= '1';
                            state   <= S_START;
                        else
                            busy_r <= '0';
                        end if;

                    -- Hold the start bit for one full bit period, then present
                    -- data bit 0.
                    when S_START =>
                        if clk_cnt = CLKS_PER_BIT - 1 then
                            clk_cnt <= 0;
                            txd_r   <= shreg(0);
                            state   <= S_DATA;
                        else
                            clk_cnt <= clk_cnt + 1;
                        end if;

                    -- Shift out bits 1..7, LSB-first, one per bit period.
                    when S_DATA =>
                        if clk_cnt = CLKS_PER_BIT - 1 then
                            clk_cnt <= 0;
                            if bit_cnt = 7 then
                                txd_r <= '1';       -- stop bit
                                state <= S_STOP;
                            else
                                shreg   <= '0' & shreg(7 downto 1);
                                txd_r   <= shreg(1);
                                bit_cnt <= bit_cnt + 1;
                            end if;
                        else
                            clk_cnt <= clk_cnt + 1;
                        end if;

                    -- Hold the stop bit for a full bit period before accepting
                    -- another request.
                    when S_STOP =>
                        if clk_cnt = CLKS_PER_BIT - 1 then
                            clk_cnt <= 0;
                            busy_r  <= '0';
                            state   <= S_IDLE;
                        else
                            clk_cnt <= clk_cnt + 1;
                        end if;

                end case;
            end if;
        end if;
    end process p_tx;

end architecture rtl;
