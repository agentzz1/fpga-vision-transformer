"""p300_pipeline.py — P300 speller decoding (xDAWN-style enhancement + shrinkage LDA),
the BR41N.IO 'P300 Speller Data Analysis' SOTA baseline. numpy/scipy/sklearn only.

P300: target (attended) flashes evoke a ~300 ms parietal positivity; non-targets
don't. Classic strong pipeline: bandpass 1-12 Hz -> baseline -> spatial enhancement
-> temporal decimation -> shrinkage LDA. Scored by AUC (classes are imbalanced ~1:5).
"""
from __future__ import annotations
import numpy as np
from scipy.linalg import eigh
from scipy.signal import butter, filtfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import roc_auc_score


def bandpass(X, lo=1.0, hi=12.0, fs=250):
    b, a = butter(4, [lo/(fs/2), hi/(fs/2)], btype="band")
    return filtfilt(b, a, X, axis=-1)


class XdawnLite:
    """Lightweight xDAWN: spatial filters maximizing evoked (target-ERP) power."""
    def __init__(self, n_filters=4):
        self.n_filters = n_filters; self.W_ = None

    def fit(self, X, y):
        X = np.asarray(X, float)
        target = X[y == 1].mean(0)                       # evoked template (ch,T)
        Rsig = target @ target.T                         # signal covariance
        allc = np.concatenate(X, axis=1)
        Rall = allc @ allc.T + 1e-6*np.eye(X.shape[1])   # total covariance
        w, V = eigh(Rsig, Rall)
        self.W_ = V[:, np.argsort(w)[::-1][:self.n_filters]].T   # (k, ch)
        return self

    def transform(self, X):
        return np.asarray([self.W_ @ t for t in np.asarray(X, float)])  # (n,k,T)


def _features(X, fs=250, decim=6):
    """baseline-correct + decimate + vectorize."""
    X = X - X[..., :max(int(0.1*fs),1)].mean(-1, keepdims=True)
    Xd = X[..., ::decim]
    return Xd.reshape(len(Xd), -1)


def evaluate(epochs, y, fs=250, folds=5, groups=None):
    """bandpass -> xDAWN -> decimate -> shrinkage LDA. epochs:(n,ch,T). Reports AUC+acc.

    groups : optional array of run/character ids, one per epoch. If given, uses
    GroupKFold so epochs from the same stimulation run never split across
    train/test — this removes the optimistic bias of a plain shuffled split
    (correlated 1-target/5-nontarget flashes from one character leaking both ways).
    """
    Xf = bandpass(np.asarray(epochs, float), fs=fs)
    y = np.asarray(y)
    if groups is not None:
        groups = np.asarray(groups)
        n_g = len(np.unique(groups))
        splitter = GroupKFold(n_splits=min(folds, n_g))
        split_iter = splitter.split(Xf, y, groups)
        leak = "group-aware (GroupKFold)"
    else:
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)
        split_iter = splitter.split(Xf, y)
        leak = "stratified (may be optimistic if epochs are grouped)"
    aucs, accs = [], []
    for tr, te in split_iter:
        xd = XdawnLite().fit(Xf[tr], y[tr])
        Ftr, Fte = _features(xd.transform(Xf[tr]), fs), _features(xd.transform(Xf[te]), fs)
        clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(Ftr, y[tr])
        prob = clf.predict_proba(Fte)[:, 1]
        aucs.append(roc_auc_score(y[te], prob))
        accs.append((clf.predict(Fte) == y[te]).mean())
    return {"method": "xDAWN+shrinkLDA", "auc": float(np.mean(aucs)),
            "acc": float(np.mean(accs)), "folds": folds, "n": len(y),
            "n_target": int((y == 1).sum()), "cv": leak}


def _synth_p300(n_target=60, ratio=5, ch=8, T=200, fs=250, seed=0):
    """Synthetic P300: targets get a ~300 ms parietal positivity; non-targets flat."""
    rng = np.random.default_rng(seed)
    t = np.arange(T)/fs
    erp = np.exp(-((t-0.30)**2)/(2*0.04**2))             # P300 bump
    spatial = np.array([0.1,0.2,0.3,0.4,0.9,0.7,1.0,0.7])[:ch]   # parietal-weighted
    X, y = [], []
    for _ in range(n_target):
        e = 6*rng.standard_normal((ch,T)) + 4.0*np.outer(spatial, erp)
        X.append(e); y.append(1)
        for _ in range(ratio):
            X.append(6*rng.standard_normal((ch,T))); y.append(0)
    X, y = np.asarray(X), np.asarray(y)
    idx = rng.permutation(len(y)); return X[idx], y[idx]


if __name__ == "__main__":
    X, y = _synth_p300()
    print("P300 xDAWN+shrinkLDA (synthetic, imbalanced ~1:5):")
    print(" ", evaluate(X, y))
