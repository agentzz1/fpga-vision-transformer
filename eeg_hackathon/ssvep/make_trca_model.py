"""make_trca_model.py — turn a recorded calibration (cal.npz: trials,labels) into a
TRCA model .npy that ssvep_2048_app.py --model loads for calibrated live decode.

    python ssvep_ab.py --record      # -> cal.npz
    python make_trca_model.py cal.npz cal.npy
    python ../app/ssvep_2048_app.py --live --model cal.npy
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from ssvep_trca import calibrate


def main(npz="cal.npz", out="cal.npy"):
    z = np.load(npz); X, y = z["trials"], z["labels"]
    by_cls = [X[y == c] for c in sorted(set(y.tolist()))]
    model = calibrate(by_cls)
    np.save(out, model, allow_pickle=True)
    print(f"saved TRCA model -> {out}  ({len(by_cls)} classes)")


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
