"""ssvep_ab.py — demo-day A/B: training-free FBCCA vs calibrated TRCA on REAL data.

Record a short calibration block (look at each arrow N times) WHILE the flickering
stimulus is on screen, then this fits TRCA and compares both decoders on held-out
trials so you pick the winner before going live. Honest empiricism beats trusting
synthetic.

    # 1) start the flicker in another terminal (it prints its refresh-locked freqs):
    python ssvep_stim.py
    # 2) record against it, telling this script the SAME monitor refresh:
    python ssvep_ab.py --record --refresh 60
    # 3) or evaluate a saved recording:
    python ssvep_ab.py --npz cal.npz

cal.npz format: trials (n,ch,T) float32, labels (n,) int 0..3, freqs (4,) float
(the exact achievable_freqs(refresh) the flicker used — replayed at decode time so
FBCCA/TRCA reference the stimulus that was actually shown).
"""
from __future__ import annotations
import os as _os, sys as _sys
_HB = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_HB, _os.path.join(_HB, "ssvep"), _os.path.join(_HB, "app")):
    if _p not in _sys.path: _sys.path.insert(0, _p)
import argparse
import numpy as np

from ssvep_cca import classify, FREQS, ARROWS, FS, achievable_freqs
from ssvep_trca import calibrate, classify_trca


def evaluate(trials, labels, freqs=FREQS, k_folds=5):
    """A/B FBCCA vs TRCA. `freqs` MUST be the frequencies the flicker actually used
    (stored in cal.npz); otherwise FBCCA scores against the wrong references."""
    trials = np.asarray(trials, float); labels = np.asarray(labels, int)
    n_cls = len(freqs)
    k_folds = max(2, min(k_folds, np.min(np.bincount(labels, minlength=n_cls))))
    idx = np.arange(len(labels)); rng = np.random.default_rng(0); rng.shuffle(idx)
    folds = np.array_split(idx, k_folds)
    fb_acc, tr_acc = [], []
    for i in range(k_folds):
        te = folds[i]; tr = np.concatenate([folds[j] for j in range(k_folds) if j != i])
        fb = np.mean([classify(trials[t], freqs=freqs)[0] == labels[t] for t in te])
        by_cls = [trials[tr][labels[tr] == c] for c in range(n_cls)]
        if min(len(c) for c in by_cls) >= 2:
            model = calibrate(by_cls)
            trc = np.mean([classify_trca(trials[t], model)[0] == labels[t] for t in te])
        else:
            trc = float("nan")
        fb_acc.append(fb); tr_acc.append(trc)
    fb_m, tr_m = float(np.nanmean(fb_acc)), float(np.nanmean(tr_acc))
    print(f"FBCCA (no training): {fb_m:.3f}")
    print(f"TRCA  (calibrated) : {tr_m:.3f}")
    print(f"-> use {'TRCA' if tr_m > fb_m else 'FBCCA'} live "
          f"(freqs={[round(f,2) for f in freqs]})")
    return fb_m, tr_m


def record(n_per_class=10, win_s=2.5, refresh=60):
    """Guided recording. Requires the flicker (ssvep_stim.py) to be visible on a
    `refresh`-Hz monitor; we store the matching achievable_freqs so decode replays
    the exact stimulus frequencies."""
    import time
    from acquire import LSLAcquirer
    from ssvep_online import _occipital_idx
    freqs, _ = achievable_freqs(refresh, n=4)
    occ = _occipital_idx(); acq = LSLAcquirer().start()
    trials, labels = [], []
    print(f"Calibration on a {refresh}Hz monitor. Flicker freqs (Hz): "
          f"{[round(f,2) for f in freqs]}.")
    print("Keep `python ssvep_stim.py` visible and FIXATE the prompted arrow each trial.")
    for c, arrow in enumerate(ARROWS):
        for r in range(n_per_class):
            input(f"  [{arrow}] trial {r+1}/{n_per_class} — fixate the {arrow} flicker, press Enter")
            time.sleep(win_s)
            d, _ = acq.get_data(seconds=win_s)
            trials.append(d[occ, -int(win_s*FS):]); labels.append(c)
    acq.stop()
    trials = np.asarray(trials, np.float32); labels = np.asarray(labels, int)
    np.savez("cal.npz", trials=trials, labels=labels, freqs=np.asarray(freqs, float))
    print(f"saved cal.npz {trials.shape}  freqs={[round(f,2) for f in freqs]}")
    return trials, labels, freqs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true"); ap.add_argument("--npz")
    ap.add_argument("--refresh", type=int, default=60,
                    help="monitor refresh Hz of the flicker (for frequency picking)")
    a = ap.parse_args()
    freqs = FREQS
    if a.record:
        t, y, freqs = record(refresh=a.refresh)
    elif a.npz:
        z = np.load(a.npz); t, y = z["trials"], z["labels"]
        if "freqs" in z: freqs = list(z["freqs"])
    else:
        # synthetic smoke test of the A/B harness (use the detected-refresh freqs)
        from ssvep_cca import synth_ssvep
        freqs, _ = achievable_freqs(a.refresh, n=4)
        rng = np.random.default_rng(0); t, y = [], []
        for c, f in enumerate(freqs):
            for _ in range(12):
                t.append(synth_ssvep(f, 2.0, n_ch=4, snr=0.5, rng=rng)); y.append(c)
        t = np.asarray(t)
    evaluate(t, y, freqs=freqs)
