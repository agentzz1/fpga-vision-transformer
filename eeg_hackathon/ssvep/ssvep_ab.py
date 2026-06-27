"""ssvep_ab.py — demo-day A/B: training-free FBCCA vs calibrated TRCA on REAL data.

Record a short calibration block (look at each arrow N times), then this fits
TRCA and compares both decoders on held-out trials so you pick the winner before
going live. Honest empiricism beats trusting synthetic.

    python ssvep_ab.py --record    # guided live recording via Unicorn LSL
    python ssvep_ab.py --npz cal.npz   # evaluate from a saved recording
Recording format (cal.npz): trials (n,ch,T) float32, labels (n,) int in 0..3.
"""
from __future__ import annotations
import os as _os, sys as _sys
_HB = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_HB, _os.path.join(_HB, "ssvep"), _os.path.join(_HB, "app")):
    if _p not in _sys.path: _sys.path.insert(0, _p)
import argparse
import numpy as np

from ssvep_cca import classify, FREQS, ARROWS, FS
from ssvep_trca import calibrate, classify_trca


def evaluate(trials, labels, k_folds=5):
    trials = np.asarray(trials, float); labels = np.asarray(labels, int)
    idx = np.arange(len(labels)); rng = np.random.default_rng(0); rng.shuffle(idx)
    folds = np.array_split(idx, k_folds)
    fb_acc, tr_acc = [], []
    for i in range(k_folds):
        te = folds[i]; tr = np.concatenate([folds[j] for j in range(k_folds) if j != i])
        # FBCCA: training-free -> just classify test
        fb = np.mean([classify(trials[t])[0] == labels[t] for t in te])
        # TRCA: calibrate on train split
        by_cls = [trials[tr][labels[tr] == c] for c in range(len(FREQS))]
        if min(len(c) for c in by_cls) >= 2:
            model = calibrate(by_cls)
            trc = np.mean([classify_trca(trials[t], model)[0] == labels[t] for t in te])
        else:
            trc = float("nan")
        fb_acc.append(fb); tr_acc.append(trc)
    print(f"FBCCA (no training): {np.nanmean(fb_acc):.3f}")
    print(f"TRCA  (calibrated) : {np.nanmean(tr_acc):.3f}")
    print(f"-> use {'TRCA' if np.nanmean(tr_acc) > np.nanmean(fb_acc) else 'FBCCA'} live")


def record(n_per_class=10, win_s=2.5):
    import time
    from acquire import LSLAcquirer
    from ssvep_online import _occipital_idx
    occ = _occipital_idx(); acq = LSLAcquirer().start()
    trials, labels = [], []
    print("Calibration: look at each arrow when prompted.")
    for c, arrow in enumerate(ARROWS):
        for r in range(n_per_class):
            input(f"  [{arrow}] trial {r+1}/{n_per_class} — fixate the {arrow} flicker, press Enter")
            time.sleep(win_s)
            d, _ = acq.get_data(seconds=win_s)
            trials.append(d[occ, -int(win_s*FS):]); labels.append(c)
    acq.stop()
    trials = np.asarray(trials, np.float32); labels = np.asarray(labels, int)
    np.savez("cal.npz", trials=trials, labels=labels)
    print(f"saved cal.npz {trials.shape}"); return trials, labels


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true"); ap.add_argument("--npz")
    a = ap.parse_args()
    if a.record:
        t, y = record()
    elif a.npz:
        z = np.load(a.npz); t, y = z["trials"], z["labels"]
    else:
        # synthetic smoke test of the A/B harness
        from ssvep_cca import synth_ssvep
        rng = np.random.default_rng(0); t, y = [], []
        for c, f in enumerate(FREQS):
            for _ in range(12):
                t.append(synth_ssvep(f, 2.0, n_ch=4, snr=0.5, rng=rng)); y.append(c)
        t = np.asarray(t)
    evaluate(t, y)
