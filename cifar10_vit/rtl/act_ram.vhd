--------------------------------------------------------------------------------
-- act_ram.vhd -- unified activation memory, laid out feature-major
--
-- Every intermediate tensor in the network lives here: the projected tokens,
-- Q/K/V, the attention scores and context, both LayerNorm outputs, and the
-- FFN hidden layer. Regions are carved out by base address (see act_map.vhd),
-- reusing space once a tensor is dead.
--
-- Layout -- the important design decision
-- ---------------------------------------
-- One word holds ONE FEATURE ACROSS ALL SEQ_LEN TOKENS, not one token across
-- all its features:
--
--     word[k] = { x[0][k], x[1][k], ... x[PE_ROWS-1][k] }
--
-- That is exactly the slice the PE array consumes: on cycle k it needs
-- activation x[m][k] for every row m at once, which is a single word read.
-- The alternative (token-major) layout would need PE_ROWS separate reads per
-- cycle and would make the array memory-bound instead of compute-bound.
--
-- The cost is on the write side. A finished tile produces PE_COLS features for
-- one token at a time, so results have to be transposed before they can be
-- written back -- that is what tile_writeback.vhd does, accumulating a full
-- column of PE_ROWS tokens before issuing one word write.
--
-- Implementation
-- --------------
-- Simple dual port: one synchronous read, one synchronous write, no
-- read-during-write forwarding (the sequencer never reads a word in the same
-- cycle it writes it, because a stage's output region is always distinct from
-- its input region). A synchronous read with no reset on the output register
-- is the idiom Vivado infers as true block RAM; adding a reset there would
-- force it into distributed LUT RAM and blow the LUT budget, which is the very
-- thing this architecture exists to avoid.
--
-- PE_ROWS bytes wide x DEPTH deep = 16 x 512 = 8 KB, which maps onto four
-- BRAM18 blocks operating in parallel (each contributing 36 of the 144 bits).
--------------------------------------------------------------------------------

library ieee;
    use ieee.std_logic_1164.all;
    use ieee.numeric_std.all;

    use work.vit_pkg.all;

entity act_ram is
    generic (
        DEPTH : positive := 512
    );
    port (
        clk : in std_logic;

        -- Read port: one feature slice (PE_ROWS activations) per cycle.
        i_raddr : in  unsigned(clog2(DEPTH) - 1 downto 0);
        i_re    : in  std_logic;
        o_rdata : out q8_row_t;

        -- Write port: one complete feature slice per cycle.
        i_waddr : in  unsigned(clog2(DEPTH) - 1 downto 0);
        i_wdata : in  q8_row_t;
        i_we    : in  std_logic
    );
end entity act_ram;

architecture rtl of act_ram is

    type ram_t is array (0 to DEPTH - 1) of q8_row_t;

    -- Initialised to zero so simulation starts from a defined state rather
    -- than 'U', which would otherwise propagate through the whole datapath and
    -- make an unrelated bug look like an arithmetic one.
    signal ram : ram_t := (others => (others => (others => '0')));

begin

    p_ram : process(clk) is
    begin
        if rising_edge(clk) then
            if i_we = '1' then
                ram(to_integer(i_waddr)) <= i_wdata;
            end if;

            -- No rstn on this register: keeping it reset-free is what lets the
            -- synthesiser map the whole array into block RAM.
            if i_re = '1' then
                o_rdata <= ram(to_integer(i_raddr));
            end if;
        end if;
    end process p_ram;

end architecture rtl;
