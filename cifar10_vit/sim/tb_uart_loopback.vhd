--------------------------------------------------------------------------------
-- tb_uart_loopback.vhd -- wires uart_tx straight into uart_rx and checks that
-- every byte sent comes back out bit-identical.
--
-- This is a real end-to-end check of the serial framing (start bit, LSB-first
-- data order, stop bit, and the receiver's half-bit centring), not just a
-- compile check: a byte-order or off-by-one-bit-period bug shows up as a
-- mismatch or a timeout here.
--
--   ghdl -a --std=08 uart_rx.vhd uart_tx.vhd tb_uart_loopback.vhd
--   ghdl -e --std=08 tb_uart_loopback && ghdl -r --std=08 tb_uart_loopback
--------------------------------------------------------------------------------

library ieee;
    use ieee.std_logic_1164.all;
    use ieee.numeric_std.all;

entity tb_uart_loopback is
end entity tb_uart_loopback;

architecture sim of tb_uart_loopback is

    -- A small CLKS_PER_BIT keeps the simulation short while exercising exactly
    -- the same logic; the real design uses 868 (100 MHz / 115200 baud).
    constant CLKS_PER_BIT : positive := 16;
    constant CLK_PERIOD   : time     := 10 ns;

    signal clk  : std_logic := '0';
    signal rstn : std_logic := '0';

    signal tx_data : std_logic_vector(7 downto 0) := (others => '0');
    signal tx_send : std_logic := '0';
    signal tx_busy : std_logic;
    signal line    : std_logic;

    signal rx_data  : std_logic_vector(7 downto 0);
    signal rx_valid : std_logic;

    signal sent_count : natural := 0;
    signal recv_count : natural := 0;
    signal errors     : natural := 0;
    signal done       : boolean := false;

    -- Test vector: values chosen to catch bit-order and stuck-bit faults
    -- (all-zeroes, all-ones, alternating, walking one, ASCII).
    type byte_array_t is array (natural range <>) of std_logic_vector(7 downto 0);
    constant TEST_BYTES : byte_array_t := (
        x"00", x"FF", x"AA", x"55", x"01", x"80", x"7F", x"3C", x"A5", x"42"
    );

begin

    clk <= not clk after CLK_PERIOD / 2 when not done else '0';

    u_tx : entity work.uart_tx
        generic map (CLKS_PER_BIT => CLKS_PER_BIT)
        port map (
            clk    => clk,
            rstn   => rstn,
            i_data => tx_data,
            i_send => tx_send,
            o_txd  => line,
            o_busy => tx_busy
        );

    u_rx : entity work.uart_rx
        generic map (CLKS_PER_BIT => CLKS_PER_BIT)
        port map (
            clk     => clk,
            rstn    => rstn,
            i_rxd   => line,
            o_data  => rx_data,
            o_valid => rx_valid
        );

    -- Checker: every received byte must equal the byte sent in the same order.
    p_check : process(clk) is
    begin
        if rising_edge(clk) then
            if rx_valid = '1' then
                if recv_count < TEST_BYTES'length then
                    if rx_data /= TEST_BYTES(recv_count) then
                        report "MISMATCH at byte " & integer'image(recv_count) &
                               ": expected 0x" & to_hstring(TEST_BYTES(recv_count)) &
                               " got 0x" & to_hstring(rx_data)
                            severity error;
                        errors <= errors + 1;
                    end if;
                else
                    report "unexpected extra byte received" severity error;
                    errors <= errors + 1;
                end if;
                recv_count <= recv_count + 1;
            end if;
        end if;
    end process p_check;

    p_stim : process is
    begin
        rstn <= '0';
        wait for 10 * CLK_PERIOD;
        wait until rising_edge(clk);
        rstn <= '1';
        wait for 10 * CLK_PERIOD;

        for i in TEST_BYTES'range loop
            wait until rising_edge(clk) and tx_busy = '0';
            tx_data <= TEST_BYTES(i);
            tx_send <= '1';
            wait until rising_edge(clk);
            tx_send <= '0';
            sent_count <= i + 1;
            -- Wait for this byte to finish on the wire before queueing the next.
            wait until rising_edge(clk) and tx_busy = '1';
            wait until rising_edge(clk) and tx_busy = '0';
        end loop;

        -- Allow the receiver to finish framing the final byte.
        wait for 4 * CLKS_PER_BIT * CLK_PERIOD;

        report "sent=" & integer'image(sent_count) &
               " received=" & integer'image(recv_count) &
               " errors=" & integer'image(errors);

        assert recv_count = TEST_BYTES'length
            report "TB FAIL: expected " & integer'image(TEST_BYTES'length) &
                   " bytes, got " & integer'image(recv_count)
            severity failure;
        assert errors = 0
            report "TB FAIL: " & integer'image(errors) & " byte mismatches"
            severity failure;

        report "UART LOOPBACK PASS: " & integer'image(recv_count) &
               "/" & integer'image(TEST_BYTES'length) & " bytes bit-exact";
        done <= true;
        wait;
    end process p_stim;

    -- Watchdog: fail loudly instead of hanging if framing never completes.
    p_watchdog : process is
    begin
        wait for TEST_BYTES'length * 20 * CLKS_PER_BIT * CLK_PERIOD + 10 us;
        if not done then
            report "TB FAIL: watchdog timeout, recv_count=" &
                   integer'image(recv_count) severity failure;
        end if;
        wait;
    end process p_watchdog;

end architecture sim;
