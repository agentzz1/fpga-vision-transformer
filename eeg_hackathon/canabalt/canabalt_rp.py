"""canabalt_rp.py — readiness-potential 'jump' detector for Canabalt (1 binary key).

The on-brief, motor-intention route: Canabalt = one action (spacebar). We detect
the movement-related cortical potential (MRCP / Bereitschaftspotential) — a slow
negative shift over the motor strip (C3/Cz/C4) that precedes a voluntary press —
and fire a spacebar. Self-paced (no cue): slide a window, score 'about-to-jump'.

This is the literal answer to the Zeiss brief ('detect when the player intends to
press a key'). It's harder/noisier than SSVEP (expect ~70-80% on 8ch), so it's the
'depth/innovation' tier, not the primary demo.

    python canabalt_rp.py            # synthetic train + CV
    python canabalt_rp.py --realtime # live: press SPACE on detected intention
"""
from __future__ import annotations
import os as _os, sys as _sys
_HB = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_HB, _os.path.join(_HB, "ssvep"), _os.path.join(_HB, "app")):
    if _p not in _sys.path: _sys.path.insert(0, _p)
import argparse, sys, time
import numpy as np
from scipy.signal import butter, filtfilt

from eeg_common import FS, CHANNEL_NAMES
from model import build_classifier, evaluate

MOTOR = ["C3", "Cz", "C4"]
RP_WIN = (-0.2, 0.8)          # validatable motor-evoked window. For REAL anticipatory RP: use (-1.0, 0.2) + LP_HZ=3.


def _motor_idx():
    return [CHANNEL_NAMES.index(c) for c in MOTOR]


LP_HZ = 12.0


def _lowpass(x, hz=LP_HZ, fs=FS):
    b, a = butter(3, hz / (fs / 2), btype="low")
    return filtfilt(b, a, x, axis=-1)


def rp_features(epochs: np.ndarray) -> np.ndarray:
    """epochs (n, n_motor_ch, T) -> slow-potential features per motor channel:
    [late mean amplitude, slope of the slow shift, min (negativity)]."""
    X = _lowpass(np.asarray(epochs, float))
    T = X.shape[-1]
    late = X[..., int(0.6 * T):].mean(-1)              # late-window mean
    slope = (X[..., -1] - X[..., 0])                   # net slow drift
    neg = X.min(-1)                                    # peak negativity (RP)
    return np.concatenate([late, slope, neg], axis=1).astype(np.float32)


def _epochs_from(data, events, idx, win=RP_WIN):
    pre = int(round(-win[0] * FS)); n = int(round((win[1] - win[0]) * FS))
    X, y = [], []
    for s, lab in events:
        s0 = int(s) - pre; s1 = s0 + n
        if 0 <= s0 and s1 <= data.shape[1]:
            ep = data[idx, s0:s1].astype(float)
            ep -= ep[:, :max(int(0.1 * FS), 1)].mean(1, keepdims=True)
            X.append(ep); y.append(int(lab))
    return np.asarray(X), np.asarray(y, int)


def train_synth():
    from acquire import SyntheticAcquirer, build_events
    d, ts, p = SyntheticAcquirer(180, 140, seed=7).get_data()
    ev = build_events(ts[p], ts)
    X, y = _epochs_from(d, ev, _motor_idx())
    F = rp_features(X)
    print("RP epochs:", X.shape, "features:", F.shape,
          f"(jump={int((y==1).sum())}, idle={int((y==0).sum())})")
    print("CV:", evaluate(F, y, "lda"))
    return build_classifier("lda").fit(F, y)


def realtime(thresh=0.6, win_s=0.6, step_s=0.1):
    from acquire import LSLAcquirer
    from pynput.keyboard import Controller
    clf = train_synth()                      # replace with a model trained on real recording
    idx = _motor_idx(); acq = LSLAcquirer().start(); kb = Controller()
    n = int((RP_WIN[1] - RP_WIN[0]) * FS); last = 0.0
    print("Canabalt RP detector live (Ctrl-C to stop)...")
    try:
        while True:
            data, _ = acq.get_data(seconds=win_s + 0.5)
            if data.shape[1] >= n:
                F = rp_features(data[idx, -n:][None, ...])
                pjump = float(clf.predict_proba(F)[0, 1])
                if pjump >= thresh and time.time() - last > 0.4:
                    last = time.time(); kb.press(' '); kb.release(' '); print(f"JUMP p={pjump:.2f}")
            time.sleep(step_s)
    except KeyboardInterrupt:
        acq.stop()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--realtime", action="store_true")
    a = ap.parse_args()
    realtime() if a.realtime else train_synth()
