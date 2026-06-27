"""riemann_pipeline.py — Riemannian tangent-space decoding (SOTA for MI/P300).

Covariance matrices live on a curved (SPD) manifold; projecting them to the tangent
space at the data mean linearizes the geometry, after which a plain linear classifier
is very strong — often beating CSP, and a top performer on BCI Competition data.
Pure numpy/scipy/sklearn (a minimal pyriemann-style implementation).

    from riemann_pipeline import evaluate
    evaluate(X, y)        # X:(trials, ch, T)
"""
from __future__ import annotations
import numpy as np
from scipy.linalg import sqrtm, logm, inv
from scipy.signal import butter, filtfilt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


def bandpass(X, lo=8, hi=30, fs=250):
    b, a = butter(4, [lo/(fs/2), hi/(fs/2)], btype="band")
    return filtfilt(b, a, X, axis=-1)


def _covs(X):
    out = []
    for t in np.asarray(X, float):
        t = t - t.mean(1, keepdims=True)
        c = (t @ t.T) / t.shape[1]
        out.append(c + 1e-6 * np.trace(c) / c.shape[0] * np.eye(c.shape[0]))
    return np.asarray(out)


def _mean_cov(C):                      # Euclidean mean ref (robust, cheap)
    return C.mean(0)


def _tangent(C, ref):
    p = inv(sqrtm(ref)).real
    n = C.shape[1]
    iu = np.triu_indices(n)
    w = np.sqrt(2) * np.ones((n, n)); np.fill_diagonal(w, 1.0)
    feats = []
    for c in C:
        S = logm(p @ c @ p).real
        feats.append((S * w)[iu])
    return np.asarray(feats)


def evaluate(X, y, fs=250, folds=5, lo=8, hi=30):
    Xf = bandpass(np.asarray(X, float), lo, hi, fs)
    y = np.asarray(y)
    C = _covs(Xf)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)
    accs, aucs = [], []
    for tr, te in skf.split(C, y):
        ref = _mean_cov(C[tr])
        Ftr, Fte = _tangent(C[tr], ref), _tangent(C[te], ref)
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(Ftr, y[tr])
        pred = clf.predict(Fte); accs.append((pred == y[te]).mean())
        try:
            aucs.append(roc_auc_score(y[te], clf.predict_proba(Fte)[:, 1]))
        except Exception:
            pass
    return {"method": "Riemann-TS+LR", "acc": float(np.mean(accs)),
            "auc": float(np.mean(aucs)) if aucs else None,
            "std": float(np.std(accs)), "folds": folds, "n": len(y)}


if __name__ == "__main__":
    from mi_pipeline import _synth_mi, evaluate as csp_eval
    X, y = _synth_mi()
    print("Motor-Imagery on synthetic (lateralized mu):")
    print("  CSP+LDA   :", csp_eval(X, y))
    print("  Riemann-TS:", evaluate(X, y))
