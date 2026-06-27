"""baseline.py — naive raw-feature LDA, the honest baseline to ablate SOTA against."""
from __future__ import annotations
import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score


def evaluate(X, y, fs=250, lo=1, hi=40, decim=8, folds=5):
    b, a = butter(4, [lo/(fs/2), hi/(fs/2)], btype="band")
    Xf = filtfilt(b, a, np.asarray(X, float), axis=-1)[..., ::decim]
    F = Xf.reshape(len(Xf), -1)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)
    acc = cross_val_score(LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
                          F, y, cv=skf, scoring="accuracy")
    return {"method": "raw+shrinkLDA(baseline)", "acc": float(acc.mean()),
            "std": float(acc.std()), "folds": folds, "n": len(y)}
