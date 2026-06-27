"""real_ssvep_benchmark.py — validate FBCCA on REAL 8-channel SSVEP (Nakanishi2015,
a near-perfect Unicorn analog: 8 ch, 256 Hz, 12 flicker targets). Training-free.
"""
from __future__ import annotations
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from ssvep_cca import reference, _cca_corr, _bandpass


def fbcca_predict(X, freqs, fs):
    """X (ch, T) -> index into freqs via filter-bank CCA."""
    Xt = (X - X.mean(1, keepdims=True)).T
    n = Xt.shape[0]
    bands = [(6, 90), (14, 90), (22, 90)]
    wts = [(k + 1) ** -1.25 + 0.25 for k in range(len(bands))]
    scores = np.zeros(len(freqs))
    for w, (lo, hi) in zip(wts, bands):
        Xb = _bandpass(Xt, lo, hi, fs)
        for i, f in enumerate(freqs):
            scores[i] += w * _cca_corr(Xb, reference(f, n, fs)) ** 2
    return int(np.argmax(scores))


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
            correct += (fbcca_predict(X[k], freqs, fs) == yi[k])
        acc = correct / len(X); accs.append(acc)
        print(f"  S{s}: n={len(X):3d} trials, {len(freqs)} targets, FBCCA acc={acc:.2f}  (chance={1/len(freqs):.2f})")
    print(f"\n  mean FBCCA acc = {np.mean(accs):.2f}  (12-class, real 8-ch, NO training)")


if __name__ == "__main__":
    run()
