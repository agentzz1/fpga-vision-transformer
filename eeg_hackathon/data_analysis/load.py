"""load.py — robust loader for BR41N.IO / g.tec EEG .mat datasets.

Hackathon .mat files vary (continuous data+trigger, or pre-epoched trials+labels).
This auto-detects the layout and returns a uniform (trials, channels, samples) array
+ integer labels, ready for mi_pipeline / p300_pipeline / riemann_pipeline.
"""
from __future__ import annotations
from typing import Tuple, Optional
import numpy as np


def _as_2d_data(v):
    """Return (channels, samples) from a 2-D array, orienting channels<=samples."""
    v = np.asarray(v)
    if v.ndim != 2: return None
    return v if v.shape[0] <= v.shape[1] else v.T


def load_mat(path, fs: int = 250, tmin: float = 0.0, tmax: float = 0.8,
             label_keys=("y", "labels", "label", "trig", "trigger", "stim", "Y"),
             ) -> Tuple[np.ndarray, np.ndarray, int]:
    """Best-effort -> (X (trials,ch,T), y (trials,), fs).

    Handles: (a) already-epoched 3-D array (trials,ch,T) or (ch,T,trials);
             (b) continuous 2-D data + a trigger/label vector (events from nonzero).
    """
    from scipy.io import loadmat
    m = {k: v for k, v in loadmat(path).items() if not k.startswith("__")}

    # (a) pre-epoched 3-D. Orient to (trials, ch, T):
    #   channel axis = smallest dim; trials axis = the one matching the label vector
    #   length (size alone can't tell trials from samples); samples = remaining.
    for v in m.values():
        a = np.asarray(v)
        if a.ndim == 3:
            ch_ax = int(np.argmin(a.shape))
            others = [i for i in range(3) if i != ch_ax]
            lab = _label_vector(m, label_keys)
            if lab is not None and len(lab) in (a.shape[others[0]], a.shape[others[1]]):
                t_ax = others[0] if len(lab) == a.shape[others[0]] else others[1]
            else:                                  # no labels: assume samples > trials
                t_ax = others[0] if a.shape[others[0]] < a.shape[others[1]] else others[1]
            s_ax = others[1] if t_ax == others[0] else others[0]
            X = np.transpose(a, (t_ax, ch_ax, s_ax))
            y = (lab.astype(int) if lab is not None and len(lab) == X.shape[0]
                 else _find_labels(m, len(X), label_keys))
            return X.astype(float), y, fs

    # (b) continuous + trigger
    data2d = max((x for x in (_as_2d_data(v) for v in m.values()) if x is not None),
                 key=lambda x: x.size, default=None)
    if data2d is None:
        raise ValueError("No 2-D/3-D EEG array found in .mat")
    trig = _find_trigger(m, data2d.shape[1], label_keys)
    if trig is None:
        raise ValueError("No trigger/label vector found; pass events manually.")
    onsets = np.where(np.diff((trig != 0).astype(int)) > 0)[0] + 1
    labels = trig[onsets].astype(int)
    pre, n = int(-tmin * fs), int(round((tmax - tmin) * fs))
    X, y = [], []
    for o, lab in zip(onsets, labels):
        s0 = o - pre
        if 0 <= s0 and s0 + n <= data2d.shape[1]:
            X.append(data2d[:, s0:s0 + n]); y.append(lab)
    return np.asarray(X, float), np.asarray(y, int), fs


def _label_vector(m, keys):
    for k in keys:
        if k in m:
            a = np.asarray(m[k]).ravel()
            if a.ndim == 1 and 1 < a.size < 100000 and np.all(a == a.astype(int)):
                return a
    return None


def _find_labels(m, n, keys):
    for k in keys:
        if k in m:
            y = np.asarray(m[k]).ravel()
            if len(y) == n: return y.astype(int)
    # fallback: any 1-D int-ish vector of length n
    for v in m.values():
        a = np.asarray(v).ravel()
        if a.size == n and np.all(a == a.astype(int)):
            return a.astype(int)
    return np.zeros(n, int)


def _find_trigger(m, n, keys):
    for k in keys:
        if k in m:
            t = np.asarray(m[k]).ravel()
            if t.size == n: return t
    for v in m.values():
        a = np.asarray(v).ravel()
        if a.size == n and (a != 0).sum() < n * 0.5:   # sparse -> looks like a trigger
            return a
    return None


if __name__ == "__main__":
    # round-trip self-test: write a synthetic continuous+trigger .mat, load it back
    from scipy.io import savemat
    fs, ch, T = 250, 8, 60 * 250
    rng = np.random.default_rng(0)
    data = rng.standard_normal((ch, T))
    trig = np.zeros(T, int)
    onsets = np.arange(500, T - 500, 600)
    for i, o in enumerate(onsets):
        trig[o] = 1 + (i % 2)               # labels 1/2
    savemat("/tmp/_test.mat", {"EEG": data, "trigger": trig})
    X, y, f = load_mat("/tmp/_test.mat", fs=fs, tmin=0.0, tmax=0.8)
    print(f"loaded X={X.shape} y={y.shape} classes={sorted(set(y.tolist()))} fs={f}")
    assert X.shape[1] == ch and len(X) == len(onsets)
    print("load.py round-trip OK")
