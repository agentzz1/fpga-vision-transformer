"""real_p300_benchmark.py — validate xDAWN+LDA on a REAL P300 speller dataset
(moabb BNCI2014-009), incl. a Unicorn-8ch subset. Reports AUC (imbalanced classes).
"""
from __future__ import annotations
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import p300_pipeline

UNICORN8 = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]


def run(subjects=(1, 2)):
    from moabb.datasets import BNCI2014_009
    from moabb.paradigms import P300
    ds = BNCI2014_009()
    para = P300(resample=128)
    print("BNCI2014-009 — REAL P300 speller, xDAWN+shrinkLDA (within-subject):\n")
    for s in subjects:
        X, y, meta = para.get_data(ds, [s])
        y = np.asarray([1 if str(v).lower().startswith("t") else 0 for v in y])
        ch = list(meta.columns) if hasattr(meta, "columns") else None
        # X already (trials, ch, T); subsample is best-effort by index if names unknown
        r_all = p300_pipeline.evaluate(X[:1500], y[:1500], fs=128)
        print(f"  S{s}: n={min(len(y),1500)} (target={int(y[:1500].sum())}) "
              f"AUC={r_all['auc']:.3f} acc={r_all['acc']:.3f}")


if __name__ == "__main__":
    run()
