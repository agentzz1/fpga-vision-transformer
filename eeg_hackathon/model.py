"""model.py — spacebar-vs-rest classifier + cross-validated evaluation."""
from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score


def build_classifier(name: str = "lda") -> Pipeline:
    name = name.lower()
    if name == "lda":
        clf = LinearDiscriminantAnalysis()
    elif name == "rf":
        clf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=0)
    elif name == "xgb":
        try:
            from xgboost import XGBClassifier
            clf = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.1,
                                subsample=0.9, eval_metric="logloss", n_jobs=-1)
        except Exception:
            clf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=0)
    else:
        raise ValueError(name)
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def evaluate(Xf, y, name: str = "lda", folds: int = 5) -> Dict:
    Xf, y = np.asarray(Xf, float), np.asarray(y, int)
    folds = min(folds, np.bincount(y).min())
    folds = max(folds, 2)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)
    pipe = build_classifier(name)
    acc = cross_val_score(pipe, Xf, y, cv=skf, scoring="accuracy")
    try:
        auc = cross_val_score(pipe, Xf, y, cv=skf, scoring="roc_auc")
        auc_m = float(auc.mean())
    except Exception:
        auc_m = float("nan")
    return {"model": name, "acc": float(acc.mean()), "acc_std": float(acc.std()),
            "auc": auc_m, "folds": folds, "n": len(y)}
