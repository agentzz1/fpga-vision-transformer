"""real_mi_benchmark.py — validate MI pipelines on REAL PhysioNet data, including a
Unicorn-8-channel subset so numbers reflect the actual headset (not a 64-ch lab cap).

Task: imagined LEFT vs RIGHT fist (eegbci runs 4,8,12). Within-subject CV.
"""
from __future__ import annotations
import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import mne
from mne.datasets import eegbci
mne.set_log_level("ERROR")

import mi_pipeline, riemann_pipeline
try:
    import eegnet; HAS_EEGNET = True
except Exception:
    HAS_EEGNET = False

UNICORN8 = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]


def load_subject(sub, runs=(4, 8, 12), tmin=0.5, tmax=2.5):
    paths = eegbci.load_data(sub, list(runs), update_path=True)
    raws = [mne.io.read_raw_edf(p, preload=True) for p in paths]
    raw = mne.concatenate_raws(raws)
    eegbci.standardize(raw)
    raw.set_montage("standard_1005", on_missing="ignore")
    raw.filter(8., 30., verbose="ERROR")
    events, eid = mne.events_from_annotations(raw)
    # T1 = left fist, T2 = right fist (imagined for these runs)
    want = {k: v for k, v in eid.items() if k in ("T1", "T2")}
    ep = mne.Epochs(raw, events, want, tmin=tmin, tmax=tmax, baseline=None,
                    preload=True, verbose="ERROR")
    X = ep.get_data() * 1e6                              # (trials, ch, T) microvolts
    y = ep.events[:, -1]; y = (y == y.max()).astype(int)  # 0/1
    fs = int(round(raw.info['sfreq']))
    return raw.ch_names, X, y, fs


def subset(ch_names, X, names):
    idx = [ch_names.index(c) for c in names if c in ch_names]
    return idx, X[:, idx, :]


def run(subjects=(1, 2, 3)):
    print(f"PhysioNet MI (imagined L/R fist) — within-subject 5-fold CV\n")
    agg = {}
    for s in subjects:
        ch, X, y, fs = load_subject(s)
        configs = [("64ch", list(range(len(ch))), X)]
        idx8, X8 = subset(ch, X, UNICORN8)
        configs.append((f"Unicorn-{len(idx8)}ch", idx8, X8))
        import fbcsp
        for tag, _, Xc in configs:
            res = {"CSP": mi_pipeline.evaluate(Xc, y, fs=fs)["acc"],
                   "Riemann": riemann_pipeline.evaluate(Xc, y, fs=fs)["acc"],
                   "FBCSP": fbcsp.evaluate(Xc, y, fs=fs)["acc"]}
            if HAS_EEGNET:
                res["EEGNet"] = eegnet.evaluate(Xc, y, folds=5, epochs=50)["acc"]
            print(f"  S{s:03d} [{tag:11}] n={len(y):3d} " +
                  "  ".join(f"{k}={v:.2f}" for k, v in res.items()))
            agg.setdefault(tag, {k: [] for k in res})
            for k, v in res.items(): agg[tag][k].append(v)
    print("\n=== mean across subjects ===")
    for tag, d in agg.items():
        print(f"  [{tag:11}] " + "  ".join(f"{k}={np.mean(v):.2f}" for k, v in d.items()))


if __name__ == "__main__":
    run()
