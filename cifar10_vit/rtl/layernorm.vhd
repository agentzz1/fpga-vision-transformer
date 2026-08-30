--------------------------------------------------------------------------------
-- layernorm.vhd -- LayerNorm with no multiplier in the inner loop and no divider
--
-- Standardising a token means subtracting its mean and dividing by its
-- standard deviation. The division is the expensive part, and this design
-- avoids it entirely: a leading-one detector on the variance gives log2(var)
-- directly, half of that is log2(std), so dividing by the standard deviation
-- becomes a shift by (lod + 1) / 2.
--
-- The approximation is coarse -- the effective divisor is rounded to a power of
-- two -- but QAT trains the network with this exact behaviour in the loop, so
-- the weights adapt to it. What it buys is a priority encoder and a shifter in
-- place of a square root and a divider.
--
-- All SEQ_LEN tokens at once
-- --------------------------
-- LayerNorm normalises across features within a token, which looks like the
-- wrong shape for a feature-major memory. It is actually the right one: each
-- act_ram word holds one feature across every token, so streaming D_MODEL
-- words feeds all SEQ_LEN tokens' accumulators in the same cycles. Every token
-- is normalised concurrently and no transpose is needed anywhere.
--
-- Two passes over the data
-- ------------------------
-- The mean and variance are not known until the whole token has been seen, so
-- the caller streams the same D_MODEL words twice:
--
--   L_ACC    D_MODEL cycles   accumulate sum and sum-of-squares per token
--   L_CALC   SEQ_LEN cycles   mean, variance and shift, one token per cycle
--   L_NORM   D_MODEL cycles   emit normalised words
--
-- About 144 cycles per LayerNorm and 288 per image, against roughly 12,000 for
-- the matrix multiplies.
--
-- mean*mean is a real multiply, but it happens once per token rather than once
-- per element, so a single multiplier is time-shared across the SEQ_LEN tokens
-- during L_CALC instead of instantiating SEQ_LEN of them.
--
-- The output shift must be arithmetic
-- -----------------------------------
-- `shift_right` on a signed value rounds toward negative infinity, matching
-- Python's `>>`. Dividing by 2**n instead would truncate toward zero and be
-- wrong by one on every negative element -- the single easiest way to break
-- bit-exactness here, which is why the vectors include mostly-negative rows.
--------------------------------------------------------------------------------

library ieee;
    use ieee.std_logic_1164.all;
    use ieee.numeric_std.all;

    use work.vit_pkg.all;

entity layernorm is
    port (
        clk  : in std_logic;
        rstn : in std_logic;                    -- synchronous, active low

        i_start : in std_logic;                 -- begin a new token block

        -- Feature-major words: i_word(m) is token m's current feature.
        -- The same D_MODEL words are streamed twice, first to accumulate
        -- statistics and then to be normalised; o_pass says which is wanted.
        i_word  : in  q8_row_t;
        i_valid : in  std_logic;

        o_pass  : out std_logic;                -- '0' accumulate, '1' normalise
        o_word  : out q8_row_t;
        o_valid : out std_logic;
        o_busy  : out std_logic;
        o_done  : out std_logic                 -- 1-cycle strobe at the end
    );
end entity layernorm;

architecture rtl of layernorm is

    -- log2(D_MODEL): mean and variance divide by the feature count, which is a
    -- power of two, so both divisions are shifts.
    constant VAR_BITS : natural := clog2(D_MODEL);
    -- Output fractional bits after the headroom shift (see quant_ops).
    constant FRAC     : natural := Q_BITS - LN_HEADROOM;

    -- Sum of D_MODEL int8 values: at most 64 * 128 = 8192.
    subtype sum_t is signed(17 downto 0);
    -- Sum of squares: at most 64 * 128^2 = 1,048,576.
    subtype ssq_t is unsigned(25 downto 0);

    type sum_arr_t is array (0 to PE_ROWS - 1) of sum_t;
    type ssq_arr_t is array (0 to PE_ROWS - 1) of ssq_t;
    type mean_arr_t is array (0 to PE_ROWS - 1) of signed(9 downto 0);
    type shift_arr_t is array (0 to PE_ROWS - 1) of natural range 0 to 15;

    signal sum_acc : sum_arr_t := (others => (others => '0'));
    signal ssq_acc : ssq_arr_t := (others => (others => '0'));
    signal mean_r  : mean_arr_t := (others => (others => '0'));
    signal shift_r : shift_arr_t := (others => 0);

    type state_t is (L_IDLE, L_ACC, L_CALC, L_NORM);
    signal state : state_t := L_IDLE;

    signal k_idx : integer range 0 to D_MODEL - 1 := 0;
    signal t_idx : integer range 0 to PE_ROWS - 1 := 0;

begin

    o_busy <= '0' when state = L_IDLE else '1';
    o_pass <= '1' when state = L_NORM else '0';

    p_ln : process(clk) is
        variable mean_v  : signed(9 downto 0);
        variable msq_v   : unsigned(25 downto 0);
        variable mm_v    : signed(19 downto 0);
        variable var_v   : signed(25 downto 0);
        variable diff_v  : signed(17 downto 0);
        variable net_v   : integer;
        variable shifted : signed(17 downto 0);
    begin
        if rising_edge(clk) then
            o_valid <= '0';
            o_done  <= '0';

            if rstn = '0' then
                state <= L_IDLE;
                k_idx <= 0;
                t_idx <= 0;

            else
                case state is

                    when L_IDLE =>
                        if i_start = '1' then
                            for m in 0 to PE_ROWS - 1 loop
                                sum_acc(m) <= (others => '0');
                                ssq_acc(m) <= (others => '0');
                            end loop;
                            k_idx <= 0;
                            state <= L_ACC;
                        end if;

                    -- Pass 1: one word per cycle feeds every token's
                    -- accumulators at once.
                    when L_ACC =>
                        if i_valid = '1' then
                            for m in 0 to PE_ROWS - 1 loop
                                sum_acc(m) <= sum_acc(m) + resize(i_word(m), sum_t'length);
                                ssq_acc(m) <= ssq_acc(m) +
                                    resize(unsigned(std_logic_vector(
                                        i_word(m) * i_word(m))), ssq_t'length);
                            end loop;

                            if k_idx = D_MODEL - 1 then
                                k_idx <= 0;
                                t_idx <= 0;
                                state <= L_CALC;
                            else
                                k_idx <= k_idx + 1;
                            end if;
                        end if;

                    -- One token per cycle: mean, variance, and the shift that
                    -- stands in for dividing by the standard deviation. A
                    -- single multiplier is reused across the tokens here.
                    when L_CALC =>
                        mean_v := resize(shift_right(sum_acc(t_idx), VAR_BITS), 10);
                        msq_v  := shift_right(ssq_acc(t_idx), VAR_BITS);
                        mm_v   := resize(mean_v * mean_v, 20);
                        var_v  := signed(resize(msq_v, 26)) - resize(mm_v, 26);
                        if var_v < 0 then
                            var_v := (others => '0');
                        end if;

                        mean_r(t_idx)  <= mean_v;
                        shift_r(t_idx) <=
                            (leading_one(unsigned(std_logic_vector(var_v))) + 1) / 2;

                        if t_idx = PE_ROWS - 1 then
                            t_idx <= 0;
                            k_idx <= 0;
                            state <= L_NORM;
                        else
                            t_idx <= t_idx + 1;
                        end if;

                    -- Pass 2: the caller re-streams the same words and gets
                    -- the standardised values back.
                    when L_NORM =>
                        if i_valid = '1' then
                            for m in 0 to PE_ROWS - 1 loop
                                diff_v := resize(i_word(m), 18) -
                                          resize(mean_r(m), 18);
                                net_v  := shift_r(m) - FRAC;
                                if net_v >= 0 then
                                    -- Arithmetic shift: rounds toward negative
                                    -- infinity, as the reference does.
                                    shifted := shift_right(diff_v, net_v);
                                else
                                    shifted := shift_left(diff_v, -net_v);
                                end if;

                                if shifted > to_signed(INT8_MAX, 18) then
                                    o_word(m) <= to_signed(INT8_MAX, 8);
                                elsif shifted < to_signed(INT8_MIN, 18) then
                                    o_word(m) <= to_signed(INT8_MIN, 8);
                                else
                                    o_word(m) <= resize(shifted, 8);
                                end if;
                            end loop;
                            o_valid <= '1';

                            if k_idx = D_MODEL - 1 then
                                k_idx  <= 0;
                                o_done <= '1';
                                state  <= L_IDLE;
                            else
                                k_idx <= k_idx + 1;
                            end if;
                        end if;

                end case;
            end if;
        end if;
    end process p_ln;

end architecture rtl;
