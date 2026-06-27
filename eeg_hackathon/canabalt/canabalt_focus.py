"""canabalt_focus.py — UHB-robust binary control: FOCUS vs REST from band-power.

Motivated by Natalizio et al. 2024 (~94.6% focus/rest on the Unicorn, live, during
Tetris) and Pontifex & Coffman 2023 (the UHB's SPECTRAL data is reliable, r~0.84,
while its ERPs are only moderate). So we drive Canabalt's single key from a
band-power 'concentration' state instead of a noisy time-domain readiness potential.

Mechanism: focus suppresses occipital/parietal ALPHA (8-13 Hz) and tends to raise
frontal-midline THETA / BETA. We classify focus vs rest on log band-power -> when
'focus' is detected, press SPACE (jump).

    python canabalt_focus.py             # synthetic train + CV
    python canabalt_focus.py --realtime  # live: SPACE while you concentrate
"""
from __future__ import annotations
import os as _os, sys as _sys
_HB = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_HB, _os.path.join(_HB, "ssvep"), _os.path.join(_HB, "app")):
    if _p not in _sys.path: _sys.path.insert(0, _p)
import argparse, sys, time
import numpy as np

from eeg_common import FS, CHANNEL_NAMES, BANDS
from features import bandpower_features
from model import build_classifier, evaluate


def _epochs(data, events, win_s=1.0):
    n = int(win_s * FS); X, y = [], []
    for s, lab in events:
        s0 = int(s); s1 = s0 + n
        if 0 <= s0 and s1 <= data.shape[1]:
            X.append(data[:, s0:s1]); y.append(int(lab))
    return np.asarray(X), np.asarray(y, int)


def train(data, events, win_s=1.0):
    X, y = _epochs(data, events, win_s)
    F, _ = bandpower_features(X)
    print("focus/rest epochs:", X.shape, "feat:", F.shape,
          f"(focus={int((y==1).sum())}, rest={int((y==0).sum())})")
    print("CV:", evaluate(F, y, "rf"))
    return build_classifier("rf").fit(F, y)


def _synth_focus(n=80, win_s=1.0, seed=0):
    """Synthetic: 'focus' epochs have suppressed alpha + raised frontal theta/beta."""
    rng = np.random.default_rng(seed); T = int(win_s*FS); t = np.arange(T)/FS
    data_list, ev, cur = [], [], 0
    ai = [CHANNEL_NAMES.index(c) for c in ("Pz","Oz","PO7","PO8")]
    fi = [CHANNEL_NAMES.index(c) for c in ("Fz","Cz")]
    for _ in range(n):
        for lab in (0, 1):
            seg = 6*rng.standard_normal((len(CHANNEL_NAMES), T))
            alpha = np.sin(2*np.pi*10*t + rng.uniform(0,6))
            beta  = np.sin(2*np.pi*20*t + rng.uniform(0,6))
            theta = np.sin(2*np.pi*6*t + rng.uniform(0,6))
            if lab == 0:                       # rest: strong alpha
                for c in ai: seg[c] += 7*alpha
            else:                              # focus: alpha suppressed, theta/beta up
                for c in ai: seg[c] += 1.5*alpha
                for c in fi: seg[c] += 4*theta + 3*beta
            data_list.append(seg); ev.append([cur, lab]); cur += T
    return np.concatenate(data_list, axis=1), np.asarray(ev, int)


def calibrate_live(acq, win_s=1.0, n_per=15):
    """Record real focus vs rest windows from the live LSL stream and train on THEM."""
    import numpy as np
    n = int(win_s * FS); X, y = [], []
    for label, name in ((1, "CONCENTRATE hard (mental math)"), (0, "RELAX / rest")):
        for r in range(n_per):
            input(f"  [{name}] trial {r+1}/{n_per} — get ready, press Enter then hold {win_s:.0f}s")
            time.sleep(win_s)
            data, _ = acq.get_data(seconds=win_s)
            if data.shape[1] >= n:
                F, _ = bandpower_features(data[:, -n:][None, ...]); X.append(F[0]); y.append(label)
    return np.asarray(X), np.asarray(y)


def realtime(thresh=0.6, win_s=1.0, step_s=0.2, calibrate=False):
    from acquire import LSLAcquirer
    acq = LSLAcquirer().start()
    if calibrate:
        import numpy as np
        from sklearn.ensemble import RandomForestClassifier
        print("LIVE calibration: real focus vs rest from your EEG.")
        X, y = calibrate_live(acq, win_s)
        clf = RandomForestClassifier(n_estimators=200, random_state=0).fit(X, y)
        print(f"calibrated on {len(y)} real trials.")
    else:
        print("WARNING: no --calibrate -> classifier trained on SYNTHETIC focus/rest; this is "
              "a code-path demo, NOT a validated live detector. Use --calibrate for real use.")
        d, ev = _synth_focus(); clf = train(d, ev, win_s)
    from pynput.keyboard import Controller
    kb = Controller(); n = int(win_s*FS); last = 0.0
    print("Focus->jump live (concentrate to jump). Ctrl-C to stop.")
    try:
        while True:
            data, _ = acq.get_data(seconds=win_s)
            if data.shape[1] >= n:
                F, _ = bandpower_features(data[:, -n:][None, ...])
                if float(clf.predict_proba(F)[0,1]) >= thresh and time.time()-last > 0.5:
                    last = time.time(); kb.press(' '); kb.release(' '); print("JUMP")
            time.sleep(step_s)
    except KeyboardInterrupt:
        acq.stop()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--realtime", action="store_true")
    ap.add_argument("--calibrate", action="store_true",
                    help="record REAL focus/rest from LSL before going live (recommended)")
    a = ap.parse_args()
    if a.realtime: realtime(calibrate=a.calibrate)
    else:
        d, ev = _synth_focus(); train(d, ev)
