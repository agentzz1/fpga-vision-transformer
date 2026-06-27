"""real_p300_benchmark.py — validate xDAWN+LDA on a REAL P300 speller dataset
(moabb BNCI2014-009). Reports AUC for the FULL 16-ch montage AND for the actual
Unicorn-8ch subset (the headset-realistic number). Classes are ~1:5 imbalanced → AUC.
"""
from __future__ import annotations
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import p300_pipeline

# Canonical Unicorn montage. All 8 exist in BNCI2014-009 (verified), so the
# subset is exact, not nearest-neighbour.
UNICORN8 = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]


def _binc(y):
    return np.asarray([1 if str(v).lower().startswith("t") else 0 for v in y])


def _groups(meta, n):
    """Group id per epoch = (session, run) so same-run flashes never split across
    train/test (leak-free CV). Falls back to None if meta lacks the columns."""
    try:
        cols = [c for c in ("session", "run") if c in meta.columns]
        if not cols:
            return None
        g = meta[cols].astype(str).agg("/".join, axis=1).to_numpy()
        return g[:n]
    except Exception:
        return None


def run(subjects=(1, 2, 3, 4, 5), cap=1500, require_n=5):    # BNCI2014-009 has 10 subjects
    import sys as _sys
    from moabb.datasets import BNCI2014_009
    from moabb.paradigms import P300
    ds = BNCI2014_009()
    para_full = P300(resample=128)
    para_8 = P300(resample=128, channels=UNICORN8)  # <-- actually subset to 8
    print("BNCI2014-009 — REAL P300 speller, xDAWN+shrinkLDA (within-subject):")
    print("  16-ch vs Unicorn-8ch; GROUP-AWARE CV (GroupKFold by session/run = leak-free)\n")
    f_aucs, e_aucs = [], []; missing = []
    for s in subjects:
        try:
            Xf, yf, mf = para_full.get_data(ds, [s])
            Xe, ye, me = para_8.get_data(ds, [s])
        except Exception as e:
            print(f"  S{s}: skipped ({type(e).__name__})"); missing.append(s); continue
        gf, ge = _groups(mf, min(len(yf), cap)), _groups(me, min(len(ye), cap))
        yf, ye = _binc(yf), _binc(ye)
        rf = p300_pipeline.evaluate(Xf[:cap], yf[:cap], fs=128, groups=gf)
        re = p300_pipeline.evaluate(Xe[:cap], ye[:cap], fs=128, groups=ge)
        f_aucs.append(rf["auc"]); e_aucs.append(re["auc"])
        print(f"  S{s}: n={min(len(yf),cap)} (target={int(yf[:cap].sum())})  "
              f"16-ch AUC={rf['auc']:.3f}  |  Unicorn-8ch AUC={re['auc']:.3f} "
              f"acc={re['acc']:.3f}  [{re['cv']}]")
    print(f"\n  mean +/- std ({len(e_aucs)} subjects): 16-ch AUC={np.mean(f_aucs):.3f}  |  "
          f"Unicorn-8ch AUC={np.mean(e_aucs):.3f} +/- {np.std(e_aucs):.3f}  "
          f"(8-ch, leak-free, is the number to quote)")
    if len(e_aucs) < require_n:
        print(f"\n  FAIL: only {len(e_aucs)}/{require_n} subjects (missing {missing}); RE-RUN. "
              f"The headline (8-ch AUC 0.94) is the n={require_n} cohort, not a partial run.")
        _sys.exit(1)


if __name__ == "__main__":
    run()
