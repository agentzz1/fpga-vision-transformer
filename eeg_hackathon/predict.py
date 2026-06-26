"""predict.py — use a trained model.

Realtime (live Unicorn via LSL):
    python predict.py --realtime
Batch (a recorded CSV + presses, or synthetic):
    python predict.py --csv eeg.csv --presses presses.txt
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import joblib

from eeg_common import FS, TMIN, TMAX, N_TIMES
from features import make_features
from preprocess import _bandpass_notch

CKPT = os.path.join(os.path.dirname(__file__), "checkpoints", "model.joblib")


def _load():
    m = joblib.load(CKPT)
    return m["pipeline"], m["features"]


class RealtimePredictor:
    """Slide a 1 s window over the live LSL stream; emit press probability."""

    def __init__(self, thresh: float = 0.5):
        self.pipe, self.feat = _load()
        self.thresh = thresh

    def run(self):
        from acquire import LSLAcquirer
        acq = LSLAcquirer().start()
        win_s = TMAX - TMIN
        print("Realtime spacebar detector running (Ctrl-C to stop)...")
        try:
            while True:
                data, ts = acq.get_data(seconds=win_s)
                if data.shape[1] >= N_TIMES:
                    seg = _bandpass_notch(data)[:, -N_TIMES:][None, ...]
                    F, _ = make_features(seg, self.feat)
                    p = float(self.pipe.predict_proba(F)[0, 1])
                    if p >= self.thresh:
                        print(f"SPACE  p={p:.2f}")
                time.sleep(0.1)
        except KeyboardInterrupt:
            acq.stop()


def batch(args):
    from acquire import CSVAcquirer, SyntheticAcquirer, build_events, presses_to_events, sample_rest_events
    from preprocess import epoch_pipeline
    pipe, feat = _load()
    if args.csv:
        eeg, ts = CSVAcquirer(args.csv).get_data()
        press = presses_to_events(np.loadtxt(args.presses).ravel(), ts)
        rest = sample_rest_events(len(press), len(ts), press[:, 0])
        ev = np.concatenate([press, rest]); ev = ev[np.argsort(ev[:, 0])]
    else:
        d, ts, p = SyntheticAcquirer(60, 50).get_data(); eeg = d
        ev = build_events(ts[p], ts)
    X, y = epoch_pipeline(eeg, ts, ev)
    F, _ = make_features(X, feat)
    pred = pipe.predict(F)
    print(f"accuracy = {(pred == y).mean():.3f} on {len(y)} epochs")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--realtime", action="store_true")
    ap.add_argument("--csv"); ap.add_argument("--presses")
    ap.add_argument("--thresh", type=float, default=0.5)
    a = ap.parse_args()
    if a.realtime:
        RealtimePredictor(a.thresh).run()
    else:
        batch(a)
