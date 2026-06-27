"""real_ssvep_benchmark.py — validate FBCCA on REAL 8-channel SSVEP (Nakanishi2015,
a near-perfect Unicorn analog: 8 ch, 256 Hz, 12 flicker targets). Training-free.
"""
from __future__ import annotations
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from ssvep_cca import classify


from ssvep_cca import classify as _classify


def run(subjects=(1, 2, 3)):
    from moabb.datasets import Nakanishi2015
    from moabb.paradigms import SSVEP
    ds = Nakanishi2015()
    para = SSVEP(n_classes=12)
    print("Nakanishi2015 — REAL 8-ch SSVEP, training-free FBCCA (within-subject):\n")
    accs = []
    for s in subjects:
        X, y, meta = para.get_data(ds, [s])
        fs = int(para.resample or 256)
        freqs = sorted({float(v) for v in np.unique(y)})
        labmap = {f"{f}": i for i, f in enumerate(freqs)}
        yi = np.array([labmap[str(float(v))] if str(float(v)) in labmap
                       else freqs.index(min(freqs, key=lambda z: abs(z-float(v)))) for v in y])
        correct = 0
        for k in range(len(X)):
            correct += (classify(X[k], freqs=freqs, fs=fs)[0] == yi[k])
        acc = correct / len(X); accs.append(acc)
        print(f"  S{s}: n={len(X):3d} trials, {len(freqs)} targets, FBCCA acc={acc:.2f}  (chance={1/len(freqs):.2f})")
    print(f"\n  mean FBCCA acc = {np.mean(accs):.2f}  (12-class, real 8-ch, NO training)")


if __name__ == "__main__":
    run()
