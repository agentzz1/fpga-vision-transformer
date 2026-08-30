#!/usr/bin/env python3
"""
model_config.py -- the single source of truth for the accelerator's shape.

These numbers appear in three places that must never disagree: the QAT training
model, the integer golden model, and the VHDL package the RTL compiles against.
They are defined here, and `export_vhdl_package()` generates the VHDL constants
from them so the hardware cannot drift out of sync with the software.

Why this shape
--------------
The workload is dominated by two costs that pull in opposite directions:

  * The patch-embedding projection costs (#pixels x D_MODEL) multiply-
    accumulates -- 3072 x 64 here -- and that total is INVARIANT to patch size.
    Splitting the image into more, smaller patches does not make the stem
    cheaper; it only makes attention and the FFN more expensive.
  * Attention costs scale with SEQ_LEN^2 and the FFN with SEQ_LEN x D_MODEL x
    D_FF, so token count is the knob that actually moves the back half.

8x8 patches give 16 tokens, which is enough spatial structure for attention to
have something to do, keeps the score matrix at a trivial 16x16, and divides
evenly into a 16-row PE array so no row ever idles.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:
    # ── Input ───────────────────────────────────────────────────────────────
    img_h: int = 32
    img_w: int = 32
    img_c: int = 3

    # ── Tokenisation ────────────────────────────────────────────────────────
    patch: int = 8                  # 8x8 patches -> a 4x4 grid of tokens

    # ── Transformer ─────────────────────────────────────────────────────────
    d_model: int = 64               # power of two: LayerNorm divides by shifting
    d_ff: int = 128                 # 2x expansion, as in the MNIST design
    n_heads: int = 1
    n_layers: int = 1
    n_classes: int = 10

    # ── Fixed-point / hardware ──────────────────────────────────────────────
    q_bits: int = 7                 # Q1.7
    ln_headroom: int = 2            # extra >>2 on LayerNorm output
    acc_bits: int = 32              # PE accumulator width

    # ── PE array ────────────────────────────────────────────────────────────
    # Output-stationary array of PE_ROWS x PE_COLS multiply-accumulate cells,
    # one DSP48E1 each. Rows map to tokens (M), columns to a tile of output
    # channels (N). One MAC per PE per cycle, so a tile takes K cycles.
    pe_rows: int = 16               # == seq_len, so every row is always busy
    pe_cols: int = 4                # 16 x 4 = 64 DSPs of the 90 available

    @property
    def grid(self) -> int:
        """Tokens per side of the patch grid (4 for 8x8 patches on 32x32)."""
        return self.img_h // self.patch

    @property
    def seq_len(self) -> int:
        """Number of tokens. No CLS token: the head global-average-pools."""
        return self.grid * self.grid

    @property
    def patch_dim(self) -> int:
        """Values per patch: patch x patch x channels."""
        return self.patch * self.patch * self.img_c

    @property
    def n_pe(self) -> int:
        return self.pe_rows * self.pe_cols

    @property
    def attn_shift(self) -> int:
        """Requantisation shift after Q.K^T.

        The usual 1/sqrt(head_dim) scaling folds into the requantisation shift
        for free when head_dim is a power of two: shifting by an extra
        log2(sqrt(d)) costs nothing, whereas a real divide would cost a lot.
        """
        head_dim = self.d_model // self.n_heads
        log2_head = head_dim.bit_length() - 1
        if (1 << log2_head) != head_dim:
            raise ValueError(f"head_dim {head_dim} must be a power of two")
        return self.q_bits + log2_head // 2

    @property
    def gap_shift(self) -> int:
        """Global average pool over seq_len tokens is a shift, not a divide."""
        s = self.seq_len.bit_length() - 1
        if (1 << s) != self.seq_len:
            raise ValueError(f"seq_len {self.seq_len} must be a power of two")
        return s

    # ── Cost model ──────────────────────────────────────────────────────────
    def mac_counts(self) -> dict[str, int]:
        """Multiply-accumulates per image, per stage."""
        c, s, d, f = self, self.seq_len, self.d_model, self.d_ff
        return {
            "patch_embed": s * c.patch_dim * d,
            "qkv":         s * d * (3 * d),
            "scores":      s * s * d,
            "attn_v":      s * s * d,
            "out_proj":    s * d * d,
            "fc1":         s * d * f,
            "fc2":         s * f * d,
            "classifier":  d * c.n_classes,
        }

    def gemm_cycles(self) -> dict[str, int]:
        """Cycles each stage occupies the PE array.

        A stage of shape (M=seq_len, K, N) runs as ceil(N / pe_cols) tiles of K
        cycles each. Readout of a finished tile overlaps the next tile's
        accumulation via double-buffered accumulators, so it costs nothing
        extra as long as K >= pe_rows -- which holds for every stage here
        except attn_v, where K == 16 == pe_rows exactly.
        """
        s, d, f = self.seq_len, self.d_model, self.d_ff

        def tiles(n: int) -> int:
            return -(-n // self.pe_cols)        # ceil

        return {
            "patch_embed": tiles(d) * self.patch_dim,
            "qkv":         tiles(3 * d) * d,
            "scores":      tiles(s) * d,
            "attn_v":      tiles(d) * s,
            "out_proj":    tiles(d) * d,
            "fc1":         tiles(f) * d,
            "fc2":         tiles(d) * f,
            "classifier":  tiles(self.n_classes) * d,
        }

    def param_counts(self) -> dict[str, int]:
        """int8 parameter bytes per tensor."""
        d, f, s = self.d_model, self.d_ff, self.seq_len
        return {
            "patch_proj_w": d * self.patch_dim, "patch_proj_b": d,
            "pos_embed": s * d,
            "wq": d * d, "wk": d * d, "wv": d * d, "wo": d * d,
            "w1": f * d, "b1": f,
            "w2": d * f, "b2": d,
            "cls_w": self.n_classes * d, "cls_b": self.n_classes,
        }

    def summary(self) -> str:
        macs = self.mac_counts()
        cycles = self.gemm_cycles()
        params = self.param_counts()
        total_mac = sum(macs.values())
        total_cyc = sum(cycles.values())
        total_par = sum(params.values())
        lines = [
            f"tokens {self.seq_len} ({self.grid}x{self.grid} of {self.patch}x{self.patch}), "
            f"patch_dim {self.patch_dim}, d_model {self.d_model}, d_ff {self.d_ff}",
            f"PE array {self.pe_rows}x{self.pe_cols} = {self.n_pe} MACs/cycle",
            "",
            f"{'stage':<14}{'MACs':>12}{'cycles':>10}{'PE util':>10}",
        ]
        for k in macs:
            util = macs[k] / (cycles[k] * self.n_pe) if cycles[k] else 0.0
            lines.append(f"{k:<14}{macs[k]:>12,}{cycles[k]:>10,}{util:>9.0%}")
        lines += [
            f"{'TOTAL':<14}{total_mac:>12,}{total_cyc:>10,}"
            f"{total_mac / (total_cyc * self.n_pe):>9.0%}",
            "",
            f"parameters: {total_par:,} int8 bytes ({total_par / 1024:.1f} KiB)",
        ]
        return "\n".join(lines)


CONFIG = ModelConfig()


# ── VHDL export ─────────────────────────────────────────────────────────────
VHDL_HEADER = """\
--------------------------------------------------------------------------------
-- vit_pkg.vhd -- generated by cifar10_vit/sw/model_config.py; DO NOT EDIT.
--
-- Regenerate with:  python3 cifar10_vit/sw/model_config.py --export
--
-- Every dimension and fixed-point constant the RTL uses comes from here, and
-- from the same Python object the golden model and the QAT training code use,
-- so the three implementations cannot drift apart.
--------------------------------------------------------------------------------

library ieee;
    use ieee.std_logic_1164.all;
    use ieee.numeric_std.all;

package vit_pkg is
"""

VHDL_FOOTER = """
    -- Saturating requantiser: arithmetic shift right, then clamp to int8.
    -- Shared by every stage that closes out a matrix multiply.
    function requantize (acc : signed; shift : natural) return signed;

    -- Saturating int8 add, used by the residual connections.
    function add_sat8 (a : signed(7 downto 0); b : signed(7 downto 0))
        return signed;

    -- Index of the most significant set bit (leading-one detector), 0 for 0.
    function leading_one (v : unsigned) return natural;

    -- Bits needed to index n distinct values, minimum 1.
    function clog2 (n : positive) return natural;

end package vit_pkg;

package body vit_pkg is

    function requantize (acc : signed; shift : natural) return signed is
        variable shifted : signed(acc'length - 1 downto 0);
    begin
        -- shift_right on a signed value is arithmetic and rounds toward
        -- negative infinity, matching Python's and numpy's >> operators.
        shifted := shift_right(acc, shift);
        if shifted > to_signed(INT8_MAX, acc'length) then
            return to_signed(INT8_MAX, 8);
        elsif shifted < to_signed(INT8_MIN, acc'length) then
            return to_signed(INT8_MIN, 8);
        else
            return resize(shifted, 8);
        end if;
    end function requantize;

    function add_sat8 (a : signed(7 downto 0); b : signed(7 downto 0))
        return signed is
        variable sum : signed(8 downto 0);
    begin
        sum := resize(a, 9) + resize(b, 9);
        if sum > to_signed(INT8_MAX, 9) then
            return to_signed(INT8_MAX, 8);
        elsif sum < to_signed(INT8_MIN, 9) then
            return to_signed(INT8_MIN, 8);
        else
            return resize(sum, 8);
        end if;
    end function add_sat8;

    function leading_one (v : unsigned) return natural is
    begin
        -- Priority encoder: scan from the top, return the first set bit.
        for i in v'length - 1 downto 0 loop
            if v(i) = '1' then
                return i;
            end if;
        end loop;
        return 0;
    end function leading_one;

    function clog2 (n : positive) return natural is
        variable bits : natural := 0;
        variable v    : positive := 1;
    begin
        while v < n loop
            v    := v * 2;
            bits := bits + 1;
        end loop;
        if bits = 0 then
            return 1;                       -- always give at least one bit
        end if;
        return bits;
    end function clog2;

end package body vit_pkg;
"""


def export_vhdl_package(cfg: ModelConfig = CONFIG) -> str:
    """Render vit_pkg.vhd from the config."""
    p = cfg.param_counts()
    total_params = sum(p.values())
    total_cycles = sum(cfg.gemm_cycles().values())

    def c(name: str, value: int, comment: str = "") -> str:
        line = f"    constant {name:<18}: natural := {value};"
        return f"{line:<52}-- {comment}" if comment else line

    body = "\n".join([
        "",
        "    -- ── Input ───────────────────────────────────────────────────",
        c("IMG_H", cfg.img_h),
        c("IMG_W", cfg.img_w),
        c("IMG_C", cfg.img_c),
        c("IMG_PIXELS", cfg.img_h * cfg.img_w * cfg.img_c, "bytes per image"),
        "",
        "    -- ── Tokenisation ────────────────────────────────────────────",
        c("PATCH", cfg.patch),
        c("GRID", cfg.grid, "patches per side"),
        c("SEQ_LEN", cfg.seq_len, "tokens"),
        c("PATCH_DIM", cfg.patch_dim, "values per patch"),
        "",
        "    -- ── Transformer ─────────────────────────────────────────────",
        c("D_MODEL", cfg.d_model),
        c("D_FF", cfg.d_ff),
        c("N_HEADS", cfg.n_heads),
        c("N_LAYERS", cfg.n_layers),
        c("N_CLASSES", cfg.n_classes),
        "",
        "    -- ── Fixed point ─────────────────────────────────────────────",
        c("Q_BITS", cfg.q_bits, "Q1.7"),
        c("Q_SCALE", 1 << cfg.q_bits, "integer representing 1.0"),
        c("ACC_BITS", cfg.acc_bits, "PE accumulator width"),
        c("LN_HEADROOM", cfg.ln_headroom, "extra >> on LayerNorm out"),
        c("ATTN_SHIFT", cfg.attn_shift, "Q.K^T requant, folds 1/sqrt(d)"),
        c("GAP_SHIFT", cfg.gap_shift, "average pool over SEQ_LEN"),
        "    constant INT8_MIN         : integer := -128;",
        "    constant INT8_MAX         : integer := 127;",
        "",
        "    -- ── PE array ────────────────────────────────────────────────",
        c("PE_ROWS", cfg.pe_rows, "maps to tokens (M)"),
        c("PE_COLS", cfg.pe_cols, "maps to an N tile"),
        c("N_PE", cfg.n_pe, "MACs per cycle"),
        "",
        "    -- ── Derived sizes (for reference / sizing memories) ─────────",
        c("TOTAL_PARAM_BYTES", total_params),
        c("TOTAL_GEMM_CYCLES", total_cycles, "PE-array cycles per image"),
        "",
        "    -- ── Types ───────────────────────────────────────────────────",
        "    subtype q8_t   is signed(7 downto 0);",
        f"    subtype acc_t  is signed({cfg.acc_bits - 1} downto 0);",
        "",
        "    -- One PE-array column slice: PE_ROWS activations read in parallel.",
        "    type q8_row_t  is array (0 to PE_ROWS - 1) of q8_t;",
        "    -- One weight slice: PE_COLS weights for the current k.",
        "    type q8_col_t  is array (0 to PE_COLS - 1) of q8_t;",
        "    -- Accumulator readout: one column's worth of results.",
        "    type acc_col_t is array (0 to PE_COLS - 1) of acc_t;",
        "",
        _render_exp_table(),
        "",
        _render_gelu_table(),
    ])
    return VHDL_HEADER + body + VHDL_FOOTER


def _wrap_table(values, per_line: int = 16, indent: str = "        ") -> str:
    rows = []
    for i in range(0, len(values), per_line):
        rows.append(indent + ", ".join(str(v) for v in values[i:i + per_line]))
    return ",\n".join(rows)


def _render_exp_table() -> str:
    """Emit the softmax exponential as a table indexed by magnitude.

    The reference `_exp_q7_from_diff` clamps (row_max - score) at Q_SCALE and
    then maps it through index arithmetic into a 256-entry Q16 table. Only 26
    of those entries are ever reachable, and every step of the arithmetic
    depends on nothing but the magnitude -- so evaluating the reference over
    its entire input domain (0..Q_SCALE) collapses the whole path to a single
    129-byte lookup. This table is generated by calling the Python function
    itself, so it is bit-identical by construction rather than by inspection.
    """
    from quant_ops import Q_SCALE as _QS, _exp_q7_from_diff
    vals = [_exp_q7_from_diff(-m) for m in range(_QS + 1)]
    return (
        "    -- exp(score - row_max) in Q1.7, indexed by the clamped magnitude\n"
        "    -- of the difference. Generated from quant_ops._exp_q7_from_diff\n"
        "    -- over its whole input domain; see model_config._render_exp_table.\n"
        f"    type exp_table_t is array (0 to {_QS}) of integer range 0 to 127;\n"
        "    constant EXP_BY_MAG : exp_table_t := (\n"
        f"{_wrap_table(vals)}\n"
        "    );"
    )


def _render_gelu_table() -> str:
    """Emit the GELU ROM, indexed by the raw activation byte."""
    from quant_ops import GELU_LUT
    vals = [int(v) for v in GELU_LUT]
    return (
        "    -- GELU, indexed by the raw int8 activation bit pattern (0..255,\n"
        "    -- two's complement). Generated from quant_ops.build_gelu_lut.\n"
        "    type gelu_table_t is array (0 to 255) of integer range -128 to 127;\n"
        "    constant GELU_LUT : gelu_table_t := (\n"
        f"{_wrap_table(vals)}\n"
        "    );"
    )


def _main() -> None:
    import sys
    cfg = CONFIG
    print(cfg.summary())
    if "--export" in sys.argv:
        out = Path(__file__).resolve().parents[1] / "rtl" / "vit_pkg.vhd"
        out.write_text(export_vhdl_package(cfg))
        print(f"\nwrote {out}")


if __name__ == "__main__":
    _main()
