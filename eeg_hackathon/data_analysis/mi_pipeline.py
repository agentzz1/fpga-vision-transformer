"""mi_pipeline.py — Motor-Imagery decoding (CSP + LDA), the BR41N.IO 'Stroke Rehab /
Motor Imagery Data Analysis' SOTA baseline. Pure numpy/scipy/sklearn (no MNE needed).

CSP (Common Spatial Patterns) learns spatial filters maximizing variance ratio
between two MI classes (e.g. left vs right hand) — the canonical motor-imagery method.

Loader adapts to g.tec .mat (channels x samples + trigger) or any (trials,ch,T) array.
"""
from __future__ import annotations
from typing import Tuple
import numpy as np
from scipy.linalg import eigh
from scipy.signal import butter, filtfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score


def bandpass(X, lo=8, hi=30, fs=250):
    b, a = butter(4, [lo/(fs/2), hi/(fs/2)], btype="band")
    return filtfilt(b, a, X, axis=-1)


class CSP:
    """Common Spatial Patterns: top-k vs bottom-k filters."""
    def __init__(self, n_components=6):
        self.n_components = n_components
        self.filters_ = None

    def fit(self, X, y):
        X = np.asarray(X, float); y = np.asarray(y)
        classes = np.unique(y)
        assert len(classes) == 2, "CSP here handles 2 classes"
        def cov(trials):
            cs = []
            for t in trials:
                t = t - t.mean(1, keepdims=True)
                c = t @ t.T; cs.append(c / np.trace(c))
            return np.mean(cs, axis=0)
        c1, c2 = cov(X[y == classes[0]]), cov(X[y == classes[1]])
        w, V = eigh(c1, c1 + c2)                     # generalized eigvecs
        order = np.argsort(w)[::-1]
        V = V[:, order]
        k = self.n_components // 2
        self.filters_ = np.concatenate([V[:, :k], V[:, -k:]], axis=1).T  # (2k, ch)
        return self

    def transform(self, X):
        feats = []
        for t in np.asarray(X, float):
            z = self.filters_ @ t                    # (2k, T)
            v = np.var(z, axis=1)
            feats.append(np.log(v / (v.sum() + 1e-12) + 1e-12))
        return np.asarray(feats)


def evaluate(X, y, fs=250, n_components=6, folds=5):
    """Bandpass -> CSP -> LDA, cross-validated. X:(trials,ch,T)."""
    X = np.asarray(X, float); y = np.asarray(y)
    if len(y) < 4 or len(np.unique(y)) < 2:
        raise ValueError(f"need >=4 trials across >=2 classes for CV; got n={len(y)}, "
                         f"classes={np.unique(y).tolist()}")
    # clamp folds to the smallest class so StratifiedKFold never errors on few trials
    folds = max(2, min(folds, int(np.min(np.bincount(y)))))
    Xf = bandpass(X, fs=fs)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)
    accs = []
    for tr, te in skf.split(Xf, y):
        csp = CSP(n_components).fit(Xf[tr], y[tr])
        clf = LinearDiscriminantAnalysis().fit(csp.transform(Xf[tr]), y[tr])
        accs.append((clf.predict(csp.transform(Xf[te])) == y[te]).mean())
    return {"method": "CSP+LDA", "acc": float(np.mean(accs)),
            "std": float(np.std(accs)), "folds": folds, "n": len(y)}


def load_gtec_mat(path, fs=250, tmin=0.5, tmax=2.5):
    """Best-effort loader for a g.tec MI .mat (continuous data + trigger channel)."""
    from scipy.io import loadmat
    m = loadmat(path)
    arrs = {k: v for k, v in m.items() if not k.startswith("__")}
    # heuristic: largest 2-D array = data (ch x samples) or (samples x ch)
    data = max((v for v in arrs.values() if getattr(v, "ndim", 0) == 2),
               key=lambda v: v.size)
    if data.shape[0] > data.shape[1]:
        data = data.T                                # -> (ch, samples)
    return data  # caller supplies events; structure varies per dataset


def _synth_mi(n_per=60, ch=8, T=500, fs=250, seed=0):
    """Synthetic MI: class 0 has stronger mu-power on 'left' channels, class 1 'right'."""
    rng = np.random.default_rng(seed)
    X, y = [], []
    t = np.arange(T) / fs
    for cls in (0, 1):
        strong = [1, 2] if cls == 0 else [5, 6]      # lateralized channels
        for _ in range(n_per):
            sig = 0.5 * rng.standard_normal((ch, T))
            mu = np.sin(2*np.pi*11*t + rng.uniform(0, 6))
            for c in strong:
                sig[c] += 2.5 * mu                    # class-specific mu rhythm
            X.append(sig); y.append(cls)
    return np.asarray(X), np.asarray(y)


if __name__ == "__main__":
    X, y = _synth_mi()
    print("Motor-Imagery CSP+LDA (synthetic, lateralized mu):")
    print(" ", evaluate(X, y))
