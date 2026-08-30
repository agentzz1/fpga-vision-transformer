--------------------------------------------------------------------------------
-- pe_array.vhd -- output-stationary multiply-accumulate array
--
-- This is the engine the whole accelerator is built around. Every matrix
-- multiply in the network runs on it: the patch projection, the fused QKV
-- projection, the attention score and context products, the output
-- projection, both FFN layers, and the classifier.
--
-- Structure
-- ---------
-- PE_ROWS x PE_COLS cells, one DSP48E1 each. Rows map to tokens (the M
-- dimension), columns to a tile of PE_COLS output channels (the N dimension).
-- Cell (m,n) owns the accumulator for output element (m,n) and keeps it in
-- place for the whole tile -- hence "output stationary". Nothing moves between
-- cells, so there is no systolic skew to get wrong and no pipeline to drain:
--
--   every cycle:  acc(m,n) += i_a(m) * i_w(n)
--
-- One activation broadcasts along its row, one weight down its column, so
-- PE_ROWS + PE_COLS operands per cycle drive PE_ROWS * PE_COLS multipliers.
-- With PE_ROWS = SEQ_LEN every row holds a real token and no cell idles.
--
-- A tile of shape (PE_ROWS x PE_COLS) therefore takes exactly K cycles, and a
-- full (M x K x N) matmul takes ceil(N / PE_COLS) * K cycles.
--
-- Double buffering
-- ----------------
-- Accumulators are duplicated into two banks. The array accumulates into one
-- while the previous tile's results are read out of the other, so readout is
-- free rather than a stall between tiles. i_last swaps them.
--
-- Bias
-- ----
-- i_bias is added on the first cycle of a tile instead of clearing to zero,
-- which is how the bias term costs nothing: it is a preload, not an extra
-- pass. Callers pass the bias already scaled by Q_SCALE (matching the golden
-- model's `bias * Q_SCALE` term) and drive it to zero for unbiased stages.
--
-- Accumulator width
-- -----------------
-- Worst case is K * 128 * 128 plus the preloaded bias. For the largest K in
-- this design (PATCH_DIM = 192) that is about 3.2 million, needing 23 bits
-- plus sign; ACC_BITS = 32 leaves generous headroom, and an assertion below
-- catches a future K that would overflow.
--------------------------------------------------------------------------------

library ieee;
    use ieee.std_logic_1164.all;
    use ieee.numeric_std.all;

    use work.vit_pkg.all;

entity pe_array is
    port (
        clk   : in  std_logic;
        rstn  : in  std_logic;                  -- synchronous, active low

        -- Operand stream. One k-slice per cycle while i_valid is high.
        i_a     : in  q8_row_t;                 -- activations, one per row
        i_w     : in  q8_col_t;                 -- weights, one per column
        i_bias  : in  acc_col_t;                -- preload, one per column
        i_valid : in  std_logic;
        i_first : in  std_logic;                -- k = 0: preload bias
        i_last  : in  std_logic;                -- k = K-1: publish the tile

        -- Readout of the most recently published tile.
        o_tile_valid : out std_logic;           -- a tile is available
        i_rd_row     : in  unsigned(clog2(PE_ROWS) - 1 downto 0);
        o_acc        : out acc_col_t
    );
end entity pe_array;

architecture rtl of pe_array is

    type acc_bank_t is array (0 to PE_ROWS - 1) of acc_col_t;

    -- Two banks: one accumulating, one being read out.
    signal bank0, bank1 : acc_bank_t;

    -- '0' -> accumulate into bank0 and read bank1, '1' -> the other way round.
    signal acc_sel   : std_logic := '0';
    signal tile_rdy  : std_logic := '0';

begin

    -- The array is only correct while the accumulator cannot overflow. This is
    -- a static property of the configuration, so check it once at elaboration
    -- rather than paying for saturation logic in every cell.
    -- Expressed in bits rather than as a magnitude comparison: 2**(ACC_BITS-1)
    -- would itself overflow VHDL's 32-bit universal integer.
    assert clog2((PATCH_DIM + 1) * 128 * 128) + 1 <= ACC_BITS
        report "ACC_BITS too narrow for the largest K in this configuration"
        severity failure;

    o_tile_valid <= tile_rdy;

    -- Readout mux: PE_COLS accumulators from the bank that is not accumulating.
    o_acc <= bank1(to_integer(i_rd_row)) when acc_sel = '0'
             else bank0(to_integer(i_rd_row));

    p_mac : process(clk) is
        variable prod : signed(15 downto 0);
        variable acc  : acc_t;
    begin
        if rising_edge(clk) then
            if rstn = '0' then
                acc_sel  <= '0';
                tile_rdy <= '0';
                for m in 0 to PE_ROWS - 1 loop
                    for n in 0 to PE_COLS - 1 loop
                        bank0(m)(n) <= (others => '0');
                        bank1(m)(n) <= (others => '0');
                    end loop;
                end loop;

            elsif i_valid = '1' then
                for m in 0 to PE_ROWS - 1 loop
                    for n in 0 to PE_COLS - 1 loop
                        -- One DSP48E1: 8x8 signed multiply into the wide
                        -- accumulator held in the DSP's own P register.
                        prod := i_a(m) * i_w(n);

                        if i_first = '1' then
                            -- Preload with the bias rather than clearing, so
                            -- the bias costs no extra cycle.
                            acc := i_bias(n) + resize(prod, ACC_BITS);
                        elsif acc_sel = '0' then
                            acc := bank0(m)(n) + resize(prod, ACC_BITS);
                        else
                            acc := bank1(m)(n) + resize(prod, ACC_BITS);
                        end if;

                        if acc_sel = '0' then
                            bank0(m)(n) <= acc;
                        else
                            bank1(m)(n) <= acc;
                        end if;
                    end loop;
                end loop;

                -- Finishing a tile hands it to the readout side and moves
                -- accumulation to the other bank.
                if i_last = '1' then
                    acc_sel  <= not acc_sel;
                    tile_rdy <= '1';
                end if;
            end if;
        end if;
    end process p_mac;

end architecture rtl;
