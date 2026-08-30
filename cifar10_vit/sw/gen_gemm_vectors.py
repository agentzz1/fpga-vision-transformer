#!/usr/bin/env python3
"""
gen_gemm_vectors.py -- test vectors for the gemm_seq + pe_array integration test.

Emits, for one matrix multiply:
  gemm_act.txt      activations, one feature-major word per line (PE_ROWS bytes)
  gemm_rom.hex      weight ROM image, one byte per line, in array-consumption order
  gemm_bias.txt     one bias per output channel, already scaled by Q_SCALE
  gemm_expect.txt   expected result, one feature-major word per line
  gemm_meta.txt     K, N_TILES, SHIFT

The expected values come from `matmul_q` in the golden model, so this checks
the hardware against the same reference the network's accuracy is measured
with -- not against a second hand-written copy of the same arithmetic.

The ROM ordering is the interesting part. The array wants, on cycle k of
output tile t, the PE_COLS weights belonging to output channels
t*PE_COLS .. t*PE_COLS+3 at reduction index k. Laying the ROM out in exactly
that order offline turns weight fetch into a bare address increment at run
time, which is why the weight side never stalls.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from model_config import CONFIG
from quant_ops import Q_SCALE, matmul_q


def emit(out_dir: Path, k: int, n: int, shift: int, seed: int,
         use_bias: bool, scale: int = 127) -> None:
    cfg = CONFIG
    m = cfg.pe_rows
    cols = cfg.pe_cols
    if n % cols:
        raise ValueError(f"N={n} must be a multiple of PE_COLS={cols}")

    # `scale` bounds the operand magnitude. Full-range int8 operands over a
    # long reduction saturate most outputs at +-127, and a wall of clipped
    # values would hide an arithmetic bug behind the clamp -- so the default
    # test case is scaled to keep most results inside the range, with a
    # separate full-range case run afterwards to exercise saturation itself.
    rng = np.random.default_rng(seed)
    a = rng.integers(-scale, scale + 1, size=(m, k)).astype(np.int32)   # (M, K)
    w = rng.integers(-scale, scale + 1, size=(n, k)).astype(np.int32)   # (N, K)
    bias = (rng.integers(-60, 61, size=(n,)).astype(np.int32)
            if use_bias else np.zeros(n, dtype=np.int32))

    expected = matmul_q(a, w.T, bias, shift)                       # (M, N)

    out_dir.mkdir(parents=True, exist_ok=True)

    def word_line(vals) -> str:
        # Feature-major word: one byte per token, token 0 first.
        return " ".join(str(int(v)) for v in vals)

    # Activations: word[k] = column k of A, i.e. every token's k-th feature.
    (out_dir / "gemm_act.txt").write_text(
        "".join(word_line(a[:, kk]) + "\n" for kk in range(k)))

    # Weight ROM, in array-consumption order.
    rom_bytes: list[int] = []
    for t in range(n // cols):
        for kk in range(k):
            for c in range(cols):
                rom_bytes.append(int(w[t * cols + c, kk]) & 0xFF)
    (out_dir / "gemm_rom.hex").write_text(
        "".join(f"{b:02X}\n" for b in rom_bytes))

    # Bias, pre-scaled exactly as the accumulator preload expects.
    (out_dir / "gemm_bias.txt").write_text(
        "".join(f"{int(b) * Q_SCALE}\n" for b in bias))

    # Expected output, in the same feature-major layout act_ram stores.
    (out_dir / "gemm_expect.txt").write_text(
        "".join(word_line(expected[:, nn]) + "\n" for nn in range(n)))

    (out_dir / "gemm_meta.txt").write_text(
        f"{k}\n{n // cols}\n{shift}\n")

    print(f"K={k} N={n} tiles={n // cols} shift={shift} bias={use_bias}")
    print(f"  activations {k} words, rom {len(rom_bytes)} bytes, "
          f"expected {n} words")
    print(f"  expected range [{expected.min()}, {expected.max()}], "
          f"saturated {int((np.abs(expected) == 127).sum())}/{expected.size}")
    print(f"  wrote to {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--shift", type=int, default=7)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--no-bias", action="store_true")
    ap.add_argument("--scale", type=int, default=127,
                    help="bound on operand magnitude (lower = less saturation)")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parents[1] / "sim" / "vectors")
    args = ap.parse_args()
    emit(args.out, args.k, args.n, args.shift, args.seed, not args.no_bias,
         args.scale)


if __name__ == "__main__":
    main()
