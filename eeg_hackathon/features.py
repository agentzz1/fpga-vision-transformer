"""features.py — features from epochs X (n_epochs, 8, N_TIMES).

  * bandpower_features  : Welch PSD integrated per band per channel (log) -> (n, 40)
  * erp_features        : mean amplitude in post-stimulus windows per channel
  * tsfresh_features    : optional, automated time-series features (guarded)
  * make_features(kind) : 'bandpower' | 'erp' | 'all' | 'tsfresh'
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
from scipy.signal import welch

_TRAPZ = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz

from eeg_common import FS, EEG_CH, CHANNEL_NAMES, BANDS


def bandpower_features(X, fs: int = FS) -> Tuple[np.ndarray, List[str]]:
    X = np.asarray(X, float)
    n_ep = X.shape[0]
    feats = np.zeros((n_ep, EEG_CH * len(BANDS)), dtype=np.float32)
    names: List[str] = []
    nperseg = min(X.shape[-1], 128)
    for ci, ch in enumerate(CHANNEL_NAMES):
        f, P = welch(X[:, ci, :], fs=fs, nperseg=nperseg, axis=-1)  # (n_ep, nf)
        for bi, (band, (lo, hi)) in enumerate(BANDS.items()):
            m = (f >= lo) & (f < hi)
            bp = _TRAPZ(P[:, m], f[m], axis=-1) if m.any() else np.zeros(n_ep)
            feats[:, ci * len(BANDS) + bi] = np.log(bp + 1e-12)
            if ci == 0 or True:
                names.append(f"{ch}_{band}")
    return feats, names


def erp_features(X, fs: int = FS) -> Tuple[np.ndarray, List[str]]:
    """Mean amplitude in 3 windows: pre(0-100ms), N1(100-200), P3(250-400) post-onset."""
    from eeg_common import TMIN
    X = np.asarray(X, float)
    onset = int(round(-TMIN * fs))
    wins = {"w1": (0.00, 0.10), "w2": (0.10, 0.20), "w3": (0.25, 0.40)}
    cols, names = [], []
    for ci, ch in enumerate(CHANNEL_NAMES):
        for wn, (a, b) in wins.items():
            s0, s1 = onset + int(a * fs), onset + int(b * fs)
            cols.append(X[:, ci, s0:s1].mean(axis=1))
            names.append(f"{ch}_{wn}")
    return np.asarray(cols, np.float32).T, names


def tsfresh_features(X) -> "pd.DataFrame":
    """Optional automated TS features (compact MinimalFCParameters for speed)."""
    import pandas as pd
    from tsfresh import extract_features
    from tsfresh.feature_extraction import MinimalFCParameters
    from tsfresh.utilities.dataframe_functions import impute
    X = np.asarray(X, float)
    rows = []
    for i in range(X.shape[0]):
        for ci, ch in enumerate(CHANNEL_NAMES):
            for t in range(X.shape[2]):
                rows.append((i, t, ch, X[i, ci, t]))
    long = pd.DataFrame(rows, columns=["id", "time", "kind", "value"])
    feat = extract_features(long, column_id="id", column_sort="time",
                            column_kind="kind", column_value="value",
                            default_fc_parameters=MinimalFCParameters(),
                            disable_progressbar=True, n_jobs=0)
    return impute(feat)


def make_features(X, kind: str = "bandpower") -> Tuple[np.ndarray, List[str]]:
    if kind == "bandpower":
        return bandpower_features(X)
    if kind == "erp":
        return erp_features(X)
    if kind == "all":
        a, na = bandpower_features(X)
        b, nb = erp_features(X)
        return np.concatenate([a, b], axis=1), na + nb
    if kind == "tsfresh":
        df = tsfresh_features(X)
        return df.to_numpy().astype(np.float32), list(df.columns)
    raise ValueError(f"unknown feature kind: {kind}")


if __name__ == "__main__":
    from acquire import SyntheticAcquirer, build_events
    from preprocess import epoch_pipeline
    d, ts, p = SyntheticAcquirer(60, 60).get_data()
    X, y = epoch_pipeline(d, ts, build_events(ts[p], ts))
    F, names = make_features(X, "all")
    print(f"features : {F.shape} ({len(names)} names) e.g. {names[:4]}")
