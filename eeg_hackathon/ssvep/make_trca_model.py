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
    # Stamp the EXACT flicker frequencies the recording used (from ssvep_ab.record ->
    # achievable_freqs(refresh)), so the live app can assert model freqs == monitor freqs
    # and fall back to FBCCA on a mismatch instead of silently decoding wrong templates.
    if "freqs" in z:
        model["freqs"] = [float(f) for f in z["freqs"]]
        print(f"  stamped recording freqs: {[round(f,2) for f in model['freqs']]}")
    else:
        print("  WARNING: cal.npz has no 'freqs' — cannot verify model matches the live monitor.")
    np.save(out, model, allow_pickle=True)
    print(f"saved TRCA model -> {out}  ({len(by_cls)} classes)")


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
