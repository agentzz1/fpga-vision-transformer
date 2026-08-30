#!/usr/bin/env python3
"""
eval_golden.py -- run the integer golden model over the CIFAR-10 test set.

This closes the loop between training and hardware. The QAT model reports an
accuracy while carrying integer-valued tensors through PyTorch; the golden
model recomputes the same network in pure Python integer arithmetic from the
exported int8 weight files. If the two disagree by more than rounding, then the
training emulation and the hardware contract have drifted apart, and the RTL --
which is verified against the golden model -- would inherit the discrepancy.

    python3 eval_golden.py                 # whole test set
    python3 eval_golden.py --limit 1000    # quick check
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from cifar_data import CLASS_NAMES, load_test
from golden_model import GoldenModel, Weights


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--weights", type=Path,
                    default=Path(__file__).resolve().parents[1] / "weights_int8")
    args = ap.parse_args()

    images, labels = load_test()
    if args.limit:
        images, labels = images[:args.limit], labels[:args.limit]

    model = GoldenModel(Weights.load(args.weights))

    t0 = time.time()
    preds = np.empty(len(images), dtype=np.int64)
    for i, img in enumerate(images):
        preds[i] = model.predict(img)
        if (i + 1) % 1000 == 0:
            rate = (i + 1) / (time.time() - t0)
            print(f"  {i + 1}/{len(images)}  {rate:.0f} img/s", flush=True)
    dt = time.time() - t0

    acc = float((preds == labels).mean())
    print(f"\ngolden model accuracy: {acc * 100:.2f}% "
          f"on {len(images)} images ({dt:.1f}s)")

    # A model that has collapsed onto one or two classes can still post a
    # respectable-looking number, so show the per-class breakdown rather than
    # just the headline.
    print(f"\n{'class':<12}{'recall':>8}{'predicted':>11}")
    for c, name in enumerate(CLASS_NAMES):
        mask = labels == c
        recall = float((preds[mask] == c).mean()) if mask.any() else 0.0
        print(f"{name:<12}{recall * 100:>7.1f}%{int((preds == c).sum()):>11}")

    used = len(np.unique(preds))
    print(f"\ndistinct classes predicted: {used}/10")
    if used < 5:
        print("WARNING: the model has collapsed onto a few classes")


if __name__ == "__main__":
    main()
