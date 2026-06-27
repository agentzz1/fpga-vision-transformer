"""real_ssvep_montage_ablation.py — headset-realistic SSVEP number.

Nakanishi2015 is an 8-electrode OCCIPITAL cluster (PO7,PO3,POz,PO4,PO8,O1,Oz,O2). The
Unicorn Hybrid Black has only 4 posterior channels (Pz,PO7,Oz,PO8); its other 4
(Fz,C3,Cz,C4) contribute little SSVEP. So the all-8-occipital FBCCA number is an UPPER
bound. This ablation restricts Nakanishi to the 4 Unicorn-posterior-equivalent electrodes
(PO7, PO8, Oz, POz~=Pz) and reports FBCCA on THAT — the headset-realistic estimate.

    python real_ssvep_montage_ablation.py
"""
from __future__ import annotations
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from ssvep_cca import classify

# Nakanishi order: PO7,PO3,POz,PO4,PO8,O1,Oz,O2 -> Unicorn-posterior-equiv indices:
#   PO7(0), PO8(4), Oz(6), POz~=Pz(2)
UNICORN4 = [0, 4, 6, 2]


def run(subjects=tuple(range(1, 10)), require_n=9):
    from moabb.datasets import Nakanishi2015
    from moabb.paradigms import SSVEP
    ds = Nakanishi2015(); para = SSVEP(n_classes=12); fs = 256
    print("FBCCA (0-train): all-8-occipital vs Unicorn-4-posterior-equiv "
          "[PO7,PO8,Oz,POz], Nakanishi2015 12-class:\n")
    a8, a4, missing = [], [], []
    for s in subjects:
        try:
            X, y, _ = para.get_data(ds, [s])
        except Exception as e:
            print(f"  S{s}: skipped ({type(e).__name__})"); missing.append(s); continue
        freqs = sorted({float(v) for v in np.unique(y)})
        yi = np.array([freqs.index(min(freqs, key=lambda z: abs(z-float(v)))) for v in y])
        c8 = np.mean([classify(X[k], freqs=freqs, fs=fs)[0] == yi[k] for k in range(len(X))])
        c4 = np.mean([classify(X[k][UNICORN4], freqs=freqs, fs=fs)[0] == yi[k] for k in range(len(X))])
        a8.append(c8); a4.append(c4)
        print(f"  S{s}: 8-occ={c8:.2f}  Unicorn-4={c4:.2f}")
    print(f"\n  8-occipital      : {np.mean(a8):.2f} +/- {np.std(a8):.2f}  (upper bound)")
    print(f"  Unicorn-4-posterior: {np.mean(a4):.2f} +/- {np.std(a4):.2f}  "
          f"(HEADSET-REALISTIC, n={len(a4)})")
    if len(a4) < require_n:
        print(f"\n  FAIL: only {len(a4)}/{require_n} subjects (missing {missing}); RE-RUN.")
        sys.exit(1)


if __name__ == "__main__":
    run()
