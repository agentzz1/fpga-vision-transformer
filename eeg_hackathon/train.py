"""train.py — train a spacebar detector and save it.

Sources:
  --csv eeg.csv --presses presses.txt   (recorded session)
  (default) synthetic data, for a smoke test.

Saves checkpoints/model.joblib (pipeline + feature config).
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import joblib

from acquire import CSVAcquirer, SyntheticAcquirer, build_events, presses_to_events, sample_rest_events
from preprocess import epoch_pipeline
from features import make_features
from model import build_classifier, evaluate

CKPT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")


def load_session(args):
    if args.csv:
        eeg, ts = CSVAcquirer(args.csv).get_data()
        press_t = np.loadtxt(args.presses).ravel() if args.presses else np.array([])
        press = presses_to_events(press_t, ts)
        rest = sample_rest_events(len(press), len(ts), press[:, 0])
        ev = np.concatenate([press, rest])[np.argsort(np.concatenate([press, rest])[:, 0])]
        return eeg, ts, ev
    d, ts, p = SyntheticAcquirer(args.duration, args.n_presses).get_data()
    return d, ts, build_events(ts[p], ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv"); ap.add_argument("--presses")
    ap.add_argument("--duration", type=float, default=180)
    ap.add_argument("--n_presses", type=int, default=140)
    ap.add_argument("--features", default="all", choices=["bandpower", "erp", "all", "tsfresh"])
    ap.add_argument("--model", default="rf", choices=["lda", "rf", "xgb"])
    ap.add_argument("--use_mne", action="store_true")
    args = ap.parse_args()

    eeg, ts, ev = load_session(args)
    X, y = epoch_pipeline(eeg, ts, ev, use_mne=args.use_mne)
    F, names = make_features(X, args.features)
    report = evaluate(F, y, args.model)
    print("CV:", report)

    pipe = build_classifier(args.model).fit(F, y)
    os.makedirs(CKPT_DIR, exist_ok=True)
    joblib.dump({"pipeline": pipe, "features": args.features,
                 "feature_names": names, "use_mne": args.use_mne}, 
                os.path.join(CKPT_DIR, "model.joblib"))
    json.dump(report, open(os.path.join(CKPT_DIR, "cv_report.json"), "w"), indent=2)
    print(f"saved -> {os.path.join(CKPT_DIR, 'model.joblib')}")


if __name__ == "__main__":
    main()
