"""real_ssvep_trca.py — FBCCA (zero-train) vs TRCA (calibrated) on REAL 8-ch SSVEP
(Nakanishi2015). Proves whether a short per-user calibration lifts the flagship.
"""
from __future__ import annotations
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from ssvep_trca import calibrate, classify_trca
from ssvep_cca import classify


def run(subjects=(1, 2, 3), cal_frac=0.5):
    from moabb.datasets import Nakanishi2015
    from moabb.paradigms import SSVEP
    ds = Nakanishi2015(); para = SSVEP(n_classes=12)
    print("Nakanishi2015 — FBCCA (0-train) vs TRCA (calibrated), real 8-ch:\n")
    fb_all, tr_all = [], []
    for s in subjects:
        X, y, meta = para.get_data(ds, [s]); fs = 256
        freqs = sorted({float(v) for v in np.unique(y)})
        yi = np.array([freqs.index(min(freqs, key=lambda z: abs(z-float(v)))) for v in y])
        rng = np.random.default_rng(0)
        cal_idx, te_idx = [], []
        for c in range(len(freqs)):
            ci = np.where(yi == c)[0]; rng.shuffle(ci)
            k = int(len(ci)*cal_frac); cal_idx += list(ci[:k]); te_idx += list(ci[k:])
        cal_idx, te_idx = np.array(cal_idx), np.array(te_idx)
        by_cls = [X[cal_idx][yi[cal_idx] == c] for c in range(len(freqs))]
        model = calibrate(by_cls)
        fb = np.mean([classify(X[i], freqs=freqs, fs=fs)[0] == yi[i] for i in te_idx])
        tr = np.mean([classify_trca(X[i], model)[0] == yi[i] for i in te_idx])
        fb_all.append(fb); tr_all.append(tr)
        print(f"  S{s}: FBCCA={fb:.2f}  TRCA={tr:.2f}  (+{tr-fb:+.2f}, {len(te_idx)} test)")
    print(f"\n  mean: FBCCA={np.mean(fb_all):.2f}  TRCA={np.mean(tr_all):.2f}  "
          f"-> {'TRCA' if np.mean(tr_all)>np.mean(fb_all) else 'FBCCA'} wins on real data")


if __name__ == "__main__":
    run()
