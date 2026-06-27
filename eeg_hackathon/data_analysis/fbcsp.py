"""fbcsp.py — Filter-Bank CSP (Ang et al. 2008), the BCI-Competition-IV-winning
motor-imagery method. CSP in several frequency sub-bands + MI feature selection + LDA.
Pure numpy/scipy/sklearn.
"""
from __future__ import annotations
import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from mi_pipeline import CSP

BANDS = [(4, 8), (8, 12), (12, 16), (16, 20), (20, 26), (26, 32)]


def _bp(X, lo, hi, fs):
    b, a = butter(4, [lo/(fs/2), min(hi, fs/2-1)/(fs/2)], btype="band")
    return filtfilt(b, a, X, axis=-1)


class FBCSP:
    def __init__(self, n_components=4, bands=BANDS, fs=250):
        self.n_components, self.bands, self.fs = n_components, bands, fs
        self.csps_ = []

    def fit(self, X, y):
        self.csps_ = []
        for lo, hi in self.bands:
            c = CSP(self.n_components).fit(_bp(X, lo, hi, self.fs), y)
            self.csps_.append(c)
        return self

    def transform(self, X):
        feats = [self.csps_[i].transform(_bp(X, lo, hi, self.fs))
                 for i, (lo, hi) in enumerate(self.bands)]
        return np.concatenate(feats, axis=1)


def evaluate(X, y, fs=250, n_components=4, k=8, folds=5):
    X = np.asarray(X, float); y = np.asarray(y)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)
    accs = []
    for tr, te in skf.split(X, y):
        fb = FBCSP(n_components, fs=fs).fit(X[tr], y[tr])
        Ftr, Fte = fb.transform(X[tr]), fb.transform(X[te])
        sel = SelectKBest(mutual_info_classif, k=min(k, Ftr.shape[1])).fit(Ftr, y[tr])
        clf = LinearDiscriminantAnalysis().fit(sel.transform(Ftr), y[tr])
        accs.append((clf.predict(sel.transform(Fte)) == y[te]).mean())
    return {"method": "FBCSP+LDA", "acc": float(np.mean(accs)),
            "std": float(np.std(accs)), "folds": folds, "n": len(y)}


if __name__ == "__main__":
    import sys, warnings; warnings.filterwarnings("ignore")
    sys.path.insert(0, ".")
    from real_mi_benchmark import load_subject, subset, UNICORN8
    import mi_pipeline
    print("FBCSP vs CSP on REAL PhysioNet MI (Unicorn-8ch), fs=160:\n")
    for s in (1, 2, 3):
        ch, X, y = load_subject(s)
        _, X8 = subset(ch, X, UNICORN8)
        csp = mi_pipeline.evaluate(X8, y, fs=160)["acc"]
        fbc = evaluate(X8, y, fs=160)["acc"]
        print(f"  S{s:03d}: CSP={csp:.2f}  FBCSP={fbc:.2f}  (+{fbc-csp:+.2f})")
