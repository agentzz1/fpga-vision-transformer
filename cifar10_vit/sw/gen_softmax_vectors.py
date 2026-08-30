#!/usr/bin/env python3
"""gen_softmax_vectors.py -- golden vectors for sim/tb_softmax.vhd.

`softmax_q` in quant_ops.py is the specification for rtl/softmax.vhd. Rather
than re-implement that arithmetic in the testbench (where a shared mistake
would cancel out and hide a bug), the testbench reads the answers this script
produces by calling the Python model itself. The two implementations then have
nothing in common but the numbers.

Output format -- plain text, one vector per line:

    <16 int8 scores>  <16 int8 expected probabilities>

Blank lines and lines starting with '#' are comments and are skipped by the
testbench, so the file stays readable.

The vector set is deliberately adversarial. The interesting corners of this
particular softmax are all in the exponent index arithmetic:

  * `_exp_q7_from_diff` clamps the magnitude of (score - row_max) at Q_SCALE
    (128), so every diff from -128 to -255 lands on the same table entry. Rows
    that straddle that clamp are generated explicitly.
  * the index step `(magnitude * 255) // 1280` only moves 26 times over the
    whole input domain, so the sweep below hits every magnitude 0..255 -- and
    therefore every reachable table entry and every step boundary -- at least
    once. Coverage is asserted before the file is written.
  * a row whose scores are all equal, all INT8_MIN or all INT8_MAX exercises
    the tie-breaking in the row-max search and the uniform output it produces.

Usage:  python3 gen_softmax_vectors.py [output_path]
Default output: ../sim/vectors/softmax_vectors.txt
"""
from __future__ import annotations

import os
import sys

import numpy as np

from quant_ops import INT8_MAX, INT8_MIN, Q_SCALE, softmax_q

SEQ_LEN = 16


def clamp8(v: int) -> int:
    return max(INT8_MIN, min(INT8_MAX, int(v)))


class VectorSet:
    """Accumulates rows plus a human-readable tag for each group."""

    def __init__(self) -> None:
        self.groups: list[tuple[str, list[list[int]]]] = []

    def add(self, tag: str, rows) -> None:
        rows = [[clamp8(v) for v in r] for r in rows]
        for r in rows:
            assert len(r) == SEQ_LEN, f"{tag}: row has {len(r)} elements"
        self.groups.append((tag, rows))

    def all_rows(self) -> list[list[int]]:
        return [r for _, rows in self.groups for r in rows]


def build_vectors() -> VectorSet:
    vs = VectorSet()
    rng = np.random.default_rng(20240517)

    # ── Degenerate rows: every element identical ────────────────────────────
    # Every diff is 0, so every exponent is the `diff >= 0 -> 127` special
    # case and the row normalises to a uniform 128/16 = 8. This also pins down
    # the row-max tie rule: with all elements equal it cannot matter, which is
    # exactly why it is worth checking.
    vs.add("all-equal",
           [[v] * SEQ_LEN for v in (INT8_MIN, -100, -37, -1, 0, 1, 37, 100,
                                    INT8_MAX)])

    # ── Saturated rows, called out separately because they are the endpoints
    # the RTL's 8-bit registers are most likely to get wrong ────────────────
    vs.add("all-INT8_MIN", [[INT8_MIN] * SEQ_LEN])
    vs.add("all-INT8_MAX", [[INT8_MAX] * SEQ_LEN])

    # ── Single spike ────────────────────────────────────────────────────────
    # One large score against a floor of small ones: the widest dynamic range
    # the format can present, and the case where the diff clamp bites hardest
    # (127 - (-128) = 255, clamped to 128). The spike is walked across every
    # column so a row-max search that only looks at, say, element 0 fails.
    spikes = []
    for pos in range(SEQ_LEN):
        row = [INT8_MIN] * SEQ_LEN
        row[pos] = INT8_MAX
        spikes.append(row)
    for pos in (0, 5, 15):
        row = [0] * SEQ_LEN
        row[pos] = INT8_MAX
        spikes.append(row)
        row = [-1] * SEQ_LEN
        row[pos] = 0
        spikes.append(row)
    vs.add("single-spike", spikes)

    # ── All-negative and all-positive rows ──────────────────────────────────
    # The row max is negative here, so `score - row_max` still has to come out
    # non-positive; an implementation that forgot the sign would produce
    # positive diffs and take the `>= 0 -> 127` path for every element.
    vs.add("all-negative",
           [rng.integers(INT8_MIN, 0, size=SEQ_LEN).tolist() for _ in range(40)])
    vs.add("all-positive",
           [rng.integers(1, INT8_MAX + 1, size=SEQ_LEN).tolist()
            for _ in range(40)])

    # ── Exhaustive magnitude sweep ──────────────────────────────────────────
    # Element 0 is the row max, so element j sees a diff of exactly -(start+j).
    # Sweeping `start` in steps of 15 walks every magnitude from 0 to 255 --
    # every reachable ROM entry, both sides of every index step, and both
    # sides of the magnitude clamp at 128.
    sweep = []
    for start in range(0, 256, 15):
        sweep.append([INT8_MAX] +
                     [INT8_MAX - (start + j) for j in range(1, SEQ_LEN)])
    # The same sweep with a non-extreme row max, so the max value itself is
    # not always 127.
    for start in range(0, 129, 15):
        sweep.append([0] + [-(start + j) for j in range(1, SEQ_LEN)])
    vs.add("magnitude-sweep", sweep)

    # ── The magnitude clamp, up close ───────────────────────────────────────
    # 128 is the last magnitude before the clamp flattens everything; 129 and
    # 255 must land on the same table entry as 128.
    clamp_rows = []
    for base in (126, 127, 128, 129, 130, 200, 255):
        row = [INT8_MAX] + [INT8_MAX - base] * (SEQ_LEN - 1)
        clamp_rows.append(row)
    # Mixed: half the row inside the clamp, half beyond it.
    clamp_rows.append([INT8_MAX] + [INT8_MAX - 127] * 7 +
                      [INT8_MAX - 200] * 8)
    vs.add("clamp-boundary", clamp_rows)

    # ── Rows whose exponentials are as small as the format allows ───────────
    # Nothing here actually underflows to zero: the clamp stops the exponent
    # argument at -1.0, where exp() is 0.368, i.e. 47/128 -- see the note in
    # the report. These rows sit at that floor, which is the closest the
    # arithmetic can get to an underflow, and confirm the RTL agrees there.
    underflow = []
    for hi in (INT8_MAX, 100, 0, -1):
        underflow.append([hi] + [INT8_MIN] * (SEQ_LEN - 1))
        underflow.append([INT8_MIN] * (SEQ_LEN - 1) + [hi])
    vs.add("exp-floor", underflow)

    # ── Ties for the row maximum ────────────────────────────────────────────
    # Several elements share the maximum. Every one of them must take the
    # `diff = 0 -> 127` path; a `>=` vs `>` slip in the max search cannot be
    # seen here, but a max search that stops early can.
    ties = []
    for n_max in (2, 3, 8, 15):
        row = [-40] * SEQ_LEN
        for k in range(n_max):
            row[(k * 5) % SEQ_LEN] = 60
        ties.append(row)
    ties.append([70, 70] + [69] * 14)
    ties.append([69] * 14 + [70, 70])
    vs.add("max-ties", ties)

    # ── Narrow-range rows ───────────────────────────────────────────────────
    # Diffs of a handful of LSBs, where the index arithmetic truncates to the
    # same table entry for several adjacent scores and the output is nearly
    # uniform. Off-by-one errors in the index show up here.
    vs.add("narrow-range",
           [rng.integers(-4, 5, size=SEQ_LEN).tolist() for _ in range(40)] +
           [rng.integers(120, 128, size=SEQ_LEN).tolist() for _ in range(20)] +
           [rng.integers(-128, -120, size=SEQ_LEN).tolist() for _ in range(20)])

    # ── Bulk random ─────────────────────────────────────────────────────────
    vs.add("random-full-range",
           [rng.integers(INT8_MIN, INT8_MAX + 1, size=SEQ_LEN).tolist()
            for _ in range(240)])

    return vs


def magnitude_coverage(rows) -> set[int]:
    """Which |score - row_max| values the vector set actually exercises."""
    seen = set()
    for row in rows:
        row_max = max(row)
        for v in row:
            seen.add(row_max - v)
    return seen


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    default = os.path.join(here, "..", "sim", "vectors",
                           "softmax_vectors.txt")
    out_path = sys.argv[1] if len(sys.argv) > 1 else default
    out_path = os.path.normpath(out_path)

    vs = build_vectors()
    rows = vs.all_rows()

    # Coverage gate: if the sweep ever stops hitting every magnitude the RTL
    # can see, the vectors have silently gone blind and this fails here rather
    # than in a green testbench run.
    seen = magnitude_coverage(rows)
    missing = sorted(set(range(0, 256)) - seen)
    assert not missing, f"magnitudes never exercised: {missing[:20]}"

    scores = np.array(rows, dtype=np.int32)
    expected = softmax_q(scores)

    # Sanity properties of the golden answers themselves, so a broken
    # quant_ops cannot quietly redefine "correct".
    assert expected.min() >= 0, "probabilities must be non-negative"
    assert expected.max() <= INT8_MAX
    sums = expected.sum(axis=1)
    assert np.all(sums <= Q_SCALE), "a row sums past 1.0"
    assert np.all(sums > Q_SCALE - SEQ_LEN), "a row loses more than 1 LSB/element"

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("# softmax golden vectors -- generated by "
                "sw/gen_softmax_vectors.py; DO NOT EDIT.\n")
        f.write("# Each data line: %d int8 scores, then the %d int8 outputs\n"
                % (SEQ_LEN, SEQ_LEN))
        f.write("# of quant_ops.softmax_q() on that row.\n")
        f.write("# vectors: %d   magnitudes covered: %d/256\n"
                % (len(rows), len(seen & set(range(256)))))
        n = 0
        for tag, group in vs.groups:
            f.write("#\n# %s (%d rows)\n" % (tag, len(group)))
            for row in group:
                exp_row = expected[n]
                n += 1
                f.write(" ".join("%4d" % v for v in row) + "   " +
                        " ".join("%4d" % v for v in exp_row) + "\n")
        assert n == len(rows)

    print("wrote %s: %d vectors, %d groups, magnitudes covered %d/256"
          % (out_path, len(rows), len(vs.groups), len(seen & set(range(256)))))
    print("row sums: min %d max %d (Q_SCALE = %d)"
          % (sums.min(), sums.max(), Q_SCALE))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
