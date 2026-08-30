#!/usr/bin/env python3
"""
quant_ops.py -- the integer primitives shared by the whole cifar10_vit flow.

Everything the accelerator does numerically is built out of the handful of
operations in this file. The QAT training emulation, the bit-exact golden
model and the RTL all have to agree on them exactly, so they are defined
once, here, and each one documents the hardware it corresponds to.

Numeric convention (inherited from the proven MNIST design):
  * Activations and weights are signed 8-bit, interpreted as Q1.7 -- i.e. an
    integer v represents v/128.
  * Products accumulate in a wide signed accumulator (int32 in hardware).
  * Requantising back to Q1.7 is an arithmetic right shift by 7 followed by
    saturation to [-128, 127].

Two details that are easy to get wrong and are pinned down deliberately:

  * Shifts are ARITHMETIC and round toward NEGATIVE INFINITY (floor), not
    toward zero. Python's `>>` and numpy's `>>` on signed integers both floor,
    and VHDL's `shift_right` on a `signed` value floors too, so the three
    implementations agree without special-casing negatives.
  * Saturation CLAMPS, it does not wrap. -300 becomes -128, not +212.
"""
from __future__ import annotations

import math

import numpy as np

# ── Fixed-point format ──────────────────────────────────────────────────────
Q_BITS = 7                      # fractional bits: Q1.7
Q_SCALE = 1 << Q_BITS           # 128 -> the integer representing 1.0
INT8_MIN = -128
INT8_MAX = 127


def sat8(x):
    """Clamp to the signed 8-bit range. Accepts a scalar or an ndarray.

    Mirrors the RTL's saturating requantiser: values outside the representable
    range stick at the endpoint instead of wrapping around.
    """
    if isinstance(x, np.ndarray):
        return np.clip(x, INT8_MIN, INT8_MAX).astype(np.int32)
    return max(INT8_MIN, min(INT8_MAX, int(x)))


def requantize(acc, shift: int = Q_BITS):
    """Wide accumulator -> int8: arithmetic shift right, then saturate.

    This is the single most common operation in the datapath -- it closes out
    every matrix multiply. `shift` is a per-stage constant chosen at design
    time, never a runtime value, so it costs nothing in hardware (it is a wire
    reindexing, not a barrel shifter).
    """
    if isinstance(acc, np.ndarray):
        return sat8(acc.astype(np.int64) >> shift)
    return sat8(acc >> shift)


def matmul_q(a: np.ndarray, b: np.ndarray, bias: np.ndarray | None = None,
             shift: int = Q_BITS) -> np.ndarray:
    """int8 (M,K) x int8 (K,N) -> int8 (M,N), the way the PE array does it.

    Accumulation happens in int64 here purely so numpy cannot overflow while
    checking the model; the values stay well inside the int32 the hardware
    accumulator actually provides (see `check_accum_width`).

    `bias` is in the same Q1.7 units as the output, so it is pre-scaled by
    Q_SCALE before being added into the accumulator -- exactly what the RTL
    does when it preloads the accumulator with the bias term.
    """
    acc = a.astype(np.int64) @ b.astype(np.int64)
    if bias is not None:
        acc = acc + bias.astype(np.int64).reshape(1, -1) * Q_SCALE
    return requantize(acc, shift)


def check_accum_width(k: int, acc_bits: int = 32) -> bool:
    """Does a K-term int8 dot product (plus a Q1.7 bias) fit the accumulator?

    Worst case magnitude is K * 128 * 128 for the products, plus 128 * 128 for
    the pre-scaled bias. Called by the tests so a future change to K or to the
    accumulator width fails loudly rather than silently wrapping in hardware.
    """
    worst = (k + 1) * 128 * 128
    return worst < (1 << (acc_bits - 1))


# ── GELU, as a 256-entry byte ROM ───────────────────────────────────────────
_SQRT_2_OVER_PI = 0.7978845608028654


def build_gelu_lut() -> np.ndarray:
    """256-entry int8 GELU table indexed by the raw int8 activation byte.

    The hardware indexes this ROM with the activation's 8 raw bits, so entry i
    holds GELU of the value that bit pattern represents: i for i < 128, and
    i - 256 for i >= 128 (two's complement).

    Truncation toward zero (Python's int()) is deliberate -- it matches the
    MNIST psum_activation.vhd table this is carried over from, and the table is
    baked into the bitstream, so the only thing that matters is that the golden
    model builds it the same way.
    """
    lut = np.zeros(256, dtype=np.int32)
    for i in range(256):
        x_int = i if i < 128 else i - 256
        x = x_int / Q_SCALE
        t = _SQRT_2_OVER_PI * (x + 0.044715 * x ** 3)
        y = 0.5 * x * (1.0 + math.tanh(t))
        lut[i] = sat8(int(y * Q_SCALE))
    return lut


GELU_LUT = build_gelu_lut()


def gelu_q(x: np.ndarray) -> np.ndarray:
    """Elementwise GELU via the ROM: one lookup, one cycle, no arithmetic."""
    return GELU_LUT[x.astype(np.int32) & 0xFF]


# ── Softmax, numerically stable, exp from a ROM ─────────────────────────────
SM_LUT_DEPTH = 256
SM_X_MIN = -10.0                # exp() is tabulated over [-10, 0]


def build_exp_lut() -> np.ndarray:
    """256-entry Q16 table of exp(x) for x spanning [SM_X_MIN, 0]."""
    lut = np.zeros(SM_LUT_DEPTH, dtype=np.int64)
    for i in range(SM_LUT_DEPTH):
        x = SM_X_MIN + i * (-SM_X_MIN / (SM_LUT_DEPTH - 1))
        lut[i] = int(math.floor(math.exp(x) * (1 << 16) + 0.5))
    return lut


EXP_LUT_Q16 = build_exp_lut()


def _exp_q7_from_diff(diff: int) -> int:
    """exp(diff) in Q1.7, for diff = score - row_max, i.e. diff <= 0.

    The index arithmetic is integer-only and mirrors softmax.vhd: the
    magnitude of the difference is clamped to one Q1.7 unit (128, i.e. -1.0 in
    real terms, the bottom of the useful range), scaled into the table, and
    the Q16 result shifted down to Q7.
    """
    if diff >= 0:
        return 127
    magnitude = min(-diff, Q_SCALE)
    scaled = (magnitude * (SM_LUT_DEPTH - 1)) // (10 * Q_SCALE)
    idx = min(SM_LUT_DEPTH - 1, (SM_LUT_DEPTH - 1) - scaled)
    return min(127, int(EXP_LUT_Q16[idx]) >> 9)


def softmax_q(scores: np.ndarray) -> np.ndarray:
    """Row-wise integer softmax on an (M,N) int8 score matrix.

    Subtracting the row maximum first keeps every exponent argument <= 0, which
    is what lets a table covering only [-10, 0] be sufficient. The normalising
    divide is the one true division in the datapath; the RTL implements it as a
    reciprocal multiply.
    """
    out = np.zeros_like(scores, dtype=np.int32)
    for m in range(scores.shape[0]):
        row = scores[m].astype(np.int32)
        row_max = int(row.max())
        exps = np.array([_exp_q7_from_diff(int(v) - row_max) for v in row],
                        dtype=np.int64)
        denom = int(exps.sum())
        if denom <= 0:
            # Degenerate row: fall back to a uniform distribution rather than
            # dividing by zero. Unreachable in practice (the row max always
            # contributes 127) but the hardware needs a defined behaviour.
            out[m] = Q_SCALE // scores.shape[1]
        else:
            out[m] = sat8((exps * Q_SCALE) // denom)
    return out


# ── LayerNorm without multipliers or dividers ───────────────────────────────
def leading_one(x: int) -> int:
    """Index of the most significant set bit, 0 for x == 0.

    In hardware this is a priority encoder (leading-one detector), not a loop.
    """
    if x <= 0:
        return 0
    return x.bit_length() - 1


def layernorm_q(tokens: np.ndarray, headroom: int = 2) -> np.ndarray:
    """Integer LayerNorm using leading-one detection instead of 1/sqrt(var).

    Standardising divides by the standard deviation. Rather than compute a
    reciprocal square root, the hardware finds the position of the leading one
    in the variance, which gives log2(var); half of that is log2(std), so the
    division collapses into a shift by (lod+1)/2. The approximation is coarse
    -- the effective divisor is rounded to a power of two -- but it costs a
    priority encoder and a shifter instead of a divider, and QAT trains the
    model with this exact behaviour in the loop, so the network adapts to it.

    `headroom` is an extra right shift on the output. Standardised values have
    unit variance and reach roughly +-3, which does not fit Q1.7's [-1, 1)
    range; shifting down by 2 (a divide by 4) keeps the tail from hard
    clipping. It must match LN_HEADROOM in the RTL and in the QAT emulation.

    Mean and variance use a shift instead of a divide, so the feature dimension
    must be a power of two.
    """
    d_model = tokens.shape[1]
    var_bits = leading_one(d_model)
    if (1 << var_bits) != d_model:
        raise ValueError(f"layernorm needs a power-of-two width, got {d_model}")
    frac = Q_BITS - headroom

    out = np.zeros_like(tokens, dtype=np.int32)
    for m in range(tokens.shape[0]):
        row = tokens[m].astype(np.int64)
        mean = int(row.sum()) >> var_bits
        mean_sq = int((row * row).sum()) >> var_bits
        var = max(0, mean_sq - mean * mean)
        shift = (leading_one(var) + 1) // 2
        diff = row - mean
        net = shift - frac
        if net >= 0:
            out[m] = sat8(diff >> net)
        else:
            out[m] = sat8(diff << (-net))
    return out


def add_sat(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Saturating elementwise add -- the residual connections."""
    return sat8(a.astype(np.int32) + b.astype(np.int32))


def argmax_first(logits: np.ndarray) -> int:
    """argmax with first-maximum-wins, matching a strict-greater RTL compare.

    Ties matter: a comparator built with `>` keeps the earlier index, one built
    with `>=` keeps the later one. numpy's argmax also keeps the first, so this
    is consistent, but it is spelled out so the RTL has an unambiguous rule.
    """
    best_i, best_v = 0, int(logits[0])
    for i in range(1, len(logits)):
        v = int(logits[i])
        if v > best_v:
            best_v, best_i = v, i
    return best_i


# ── Input quantisation ──────────────────────────────────────────────────────
def quantize_pixels(img_u8: np.ndarray, mean, std) -> np.ndarray:
    """uint8 image -> int8 Q1.7, applying the training normalisation.

    Done in float here and on the host side in the real system: the FPGA
    receives raw bytes over UART and applies an equivalent integer affine map,
    so this stays the single definition of what the network expects to see.
    """
    x = img_u8.astype(np.float64) / 255.0
    mean = np.asarray(mean, dtype=np.float64).reshape(-1, 1, 1)
    std = np.asarray(std, dtype=np.float64).reshape(-1, 1, 1)
    x = (x - mean) / std
    return sat8(np.rint(x * Q_SCALE).astype(np.int64))


# ── Self-tests ──────────────────────────────────────────────────────────────
def _test() -> None:
    # Saturation clamps rather than wrapping.
    assert sat8(300) == 127 and sat8(-300) == -128 and sat8(5) == 5

    # Shifts floor toward negative infinity, and numpy agrees with Python.
    assert (-1) >> 1 == -1, "python >> must floor"
    assert int(np.int64(-1) >> 1) == -1, "numpy >> must floor"
    assert requantize(-1) == -1 and requantize(-129 * 128) == -128

    # Accumulator width is adequate for the largest K the design uses.
    for k in (16, 64, 128, 192, 256):
        assert check_accum_width(k), f"K={k} overflows a 32-bit accumulator"

    # GELU table: near-zero maps to near-zero, large positive is ~identity,
    # large negative is squashed toward zero.
    assert GELU_LUT[0] == 0
    assert GELU_LUT[127] > 100, "GELU(~1.0) should pass through most of x"
    # GELU(-1.0) = -0.1588, i.e. -20 in Q1.7 after truncation toward zero.
    assert GELU_LUT[128] == -20, f"GELU(-1.0) should be -20, got {GELU_LUT[128]}"

    # Softmax rows are non-negative and sum to roughly Q_SCALE (128).
    rng = np.random.default_rng(0)
    scores = rng.integers(-128, 128, size=(8, 16)).astype(np.int32)
    p = softmax_q(scores)
    assert p.min() >= 0
    # Each entry is floored, losing under 1 LSB, so a row of N entries sums to
    # somewhere in (Q_SCALE - N, Q_SCALE]. The deficit is real and the network
    # is trained with it in the loop; what matters is that it is bounded.
    sums = p.sum(axis=1)
    n_cols = scores.shape[1]
    assert np.all(sums <= Q_SCALE), f"softmax rows overshoot: {sums}"
    assert np.all(sums > Q_SCALE - n_cols), f"softmax rows sum to {sums}"
    # The largest score must receive the largest probability.
    for m in range(scores.shape[0]):
        assert p[m].argmax() == scores[m].argmax()

    # LayerNorm zero-centres each row.
    toks = rng.integers(-128, 128, size=(4, 64)).astype(np.int32)
    ln = layernorm_q(toks)
    assert ln.shape == toks.shape
    assert np.all(np.abs(ln.mean(axis=1)) < 12), "LN rows should be ~zero-mean"
    # A constant row has zero variance and must not blow up or divide by zero.
    flat = layernorm_q(np.full((1, 64), 42, dtype=np.int32))
    assert np.all(flat == 0), f"constant row should normalise to 0, got {flat}"

    # leading_one is a bit-position, and handles the zero case.
    assert leading_one(0) == 0 and leading_one(1) == 0 and leading_one(255) == 7

    # Residual add saturates.
    assert add_sat(np.array([100]), np.array([100]))[0] == 127

    # argmax keeps the first maximum on a tie.
    assert argmax_first(np.array([5, 9, 9, 1])) == 1

    # matmul_q against an explicit reference loop.
    a = rng.integers(-128, 128, size=(3, 16)).astype(np.int32)
    b = rng.integers(-128, 128, size=(16, 5)).astype(np.int32)
    bias = rng.integers(-20, 20, size=(5,)).astype(np.int32)
    got = matmul_q(a, b, bias)
    for m in range(3):
        for n in range(5):
            acc = int(bias[n]) * Q_SCALE + sum(int(a[m, k]) * int(b[k, n])
                                               for k in range(16))
            assert got[m, n] == sat8(acc >> 7), (m, n)

    print("quant_ops self-tests OK")


if __name__ == "__main__":
    _test()
