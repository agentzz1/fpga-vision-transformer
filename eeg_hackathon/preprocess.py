"""preprocess.py — filtering + epoching for the Unicorn pipeline.

Primary path uses MNE-Python (matches the repo); a pure-SciPy fallback runs
when MNE isn't installed, so the demo works anywhere. Same I/O either way.

Contract (see eeg_common.py):
  data       (8, n_samples) float32, timestamps (n_samples,) float64
  events     (n_events, 2) [sample_index, label]
  ->  X      (n_epochs, 8, N_TIMES) float32,  y (n_epochs,) int
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

from eeg_common import (
    FS, EEG_CH, BANDPASS, NOTCH_FREQ, TMIN, TMAX, N_TIMES,
    BASELINE, EPOCH_DTYPE,
)

try:
    import mne  # noqa: F401
    _HAS_MNE = True
except Exception:
    _HAS_MNE = False


# ------------------------------ SciPy path --------------------------------- #
def _bandpass_notch(data: np.ndarray, l=BANDPASS[0], h=BANDPASS[1],
                    notch=NOTCH_FREQ, fs=FS) -> np.ndarray:
    """Zero-phase bandpass + mains notch on (8, n) data."""
    x = np.asarray(data, dtype=np.float64)
    nyq = fs / 2.0
    b, a = butter(4, [max(l, 0.01) / nyq, min(h, nyq - 1) / nyq], btype="band")
    x = filtfilt(b, a, x, axis=1)
    if notch and notch < nyq:
        bn, an = iirnotch(notch / nyq, Q=30.0)
        x = filtfilt(bn, an, x, axis=1)
    return x


def _epoch_scipy(data, events) -> Tuple[np.ndarray, np.ndarray]:
    pre = int(round(-TMIN * FS))
    n = N_TIMES
    base_n = int(round((BASELINE[1] - BASELINE[0]) * FS))
    X, y = [], []
    for s, lab in events:
        s0 = int(s) - pre
        s1 = s0 + n
        if s0 < 0 or s1 > data.shape[1]:
            continue
        ep = data[:, s0:s1].astype(np.float64)
        ep -= ep[:, :max(base_n, 1)].mean(axis=1, keepdims=True)  # baseline corr
        X.append(ep)
        y.append(int(lab))
    if not X:
        return (np.empty((0, EEG_CH, n), EPOCH_DTYPE), np.empty((0,), int))
    return np.asarray(X, EPOCH_DTYPE), np.asarray(y, int)


# ------------------------------- MNE path ---------------------------------- #
def build_raw(data, timestamps=None):
    """Return an mne.io.RawArray (microvolts -> volts for MNE convention)."""
    from eeg_common import make_info
    info = make_info()
    return mne.io.RawArray(np.asarray(data, float) * 1e-6, info, verbose="error")


def filter_raw(raw, l_freq=BANDPASS[0], h_freq=BANDPASS[1], notch=NOTCH_FREQ):
    raw = raw.copy().filter(l_freq, h_freq, verbose="error")
    if notch:
        raw = raw.notch_filter(freqs=[notch], verbose="error")
    return raw


def make_epochs(raw, events):
    ev = np.column_stack([events[:, 0],
                          np.zeros(len(events), int), events[:, 1]])
    epochs = mne.Epochs(raw, ev, tmin=TMIN, tmax=TMAX, baseline=BASELINE,
                        preload=True, verbose="error",
                        event_id={"rest": 0, "space": 1})
    return epochs


def get_epoch_array(epochs) -> Tuple[np.ndarray, np.ndarray]:
    X = (epochs.get_data() * 1e6).astype(EPOCH_DTYPE)       # back to microvolts
    if X.shape[-1] != N_TIMES:                              # align length
        X = X[..., :N_TIMES] if X.shape[-1] > N_TIMES else np.pad(
            X, ((0, 0), (0, 0), (0, N_TIMES - X.shape[-1])))
    y = epochs.events[:, -1].astype(int)
    return X, y


# ------------------------------ unified API -------------------------------- #
def epoch_pipeline(data, timestamps, events, use_mne: bool = False
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """Filter then epoch. SciPy by default (fast, dependency-light); MNE if asked."""
    if use_mne and _HAS_MNE:
        raw = filter_raw(build_raw(data, timestamps))
        return get_epoch_array(make_epochs(raw, np.asarray(events)))
    filt = _bandpass_notch(data)
    return _epoch_scipy(filt, np.asarray(events))


if __name__ == "__main__":
    from acquire import SyntheticAcquirer, build_events
    d, ts, presses = SyntheticAcquirer(60, 60).get_data()
    ev = build_events(ts[presses], ts)
    X, y = epoch_pipeline(d, ts, ev)
    print(f"MNE available : {_HAS_MNE}")
    print(f"epochs X      : {X.shape} dtype={X.dtype}")
    print(f"labels y      : {y.shape} (space={int((y==1).sum())}, rest={int((y==0).sum())})")
