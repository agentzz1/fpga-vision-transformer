"""ssvep_trca.py — TRCA (Task-Related Component Analysis) calibration for SSVEP.

Training-free FBCCA is the baseline that works instantly. TRCA adds a short
(~60 s) per-user calibration that learns spatial filters + templates per target,
typically beating CCA/FBCCA — especially at SHORT windows. This is the algorithm
behind the highest-ITR EEG BCI on record (Nakanishi et al. 2018).

Workflow:
    model = calibrate(trials_by_class)        # trials_by_class[k]: (n_trials, ch, T)
    cls, scores = classify_trca(window, model) # window: (ch, T) -> class idx
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import scipy.linalg as la

from ssvep_cca import FREQS, ARROWS, FS


def _trca_filter(trials: np.ndarray) -> np.ndarray:
    """Leading TRCA spatial filter for one class. trials: (n_trials, ch, T)."""
    n_tr, ch, T = trials.shape
    X = trials - trials.mean(axis=2, keepdims=True)
    # S = sum of cross-covariances between distinct trials
    S = np.zeros((ch, ch))
    for i in range(n_tr):
        for j in range(n_tr):
            if i != j:
                S += X[i] @ X[j].T
    # Q = covariance of all trials concatenated
    Xcat = X.transpose(1, 0, 2).reshape(ch, -1)
    Q = Xcat @ Xcat.T + 1e-6 * np.eye(ch)
    eigvals, eigvecs = la.eig(S, Q)
    return eigvecs[:, np.argmax(eigvals.real)].real


def calibrate(trials_by_class: List[np.ndarray]) -> Dict:
    """Build ensemble-TRCA model: a spatial filter + template per class."""
    filters, templates = [], []
    for trials in trials_by_class:
        trials = np.asarray(trials, float)
        filters.append(_trca_filter(trials))
        templates.append(trials.mean(axis=0))            # (ch, T) trial-average
    W = np.asarray(filters).T                            # (ch, K) ensemble filters
    return {"W": W, "templates": templates, "freqs": FREQS}


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel(), b.ravel()
    a = a - a.mean(); b = b - b.mean()
    d = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(a @ b / d)


def classify_trca(window: np.ndarray, model: Dict) -> Tuple[int, np.ndarray]:
    """Ensemble-TRCA: correlate filtered test vs each filtered template."""
    X = np.asarray(window, float)
    X = X - X.mean(axis=1, keepdims=True)
    W, templates = model["W"], model["templates"]
    Xf = W.T @ X                                          # (K, T)
    scores = np.array([_corr(Xf, (W.T @ templates[k])) for k in range(len(templates))])
    return int(np.argmax(scores)), scores


if __name__ == "__main__":
    # A/B vs FBCCA on synthetic SSVEP at a hard (short window, low SNR) setting.
    from ssvep_cca import synth_ssvep, classify
    rng = np.random.default_rng(3)
    n_ch, snr = 4, 0.40
    for win_s in (0.75, 1.0, 1.5):
        # calibration: 8 trials/class
        cal = [np.stack([synth_ssvep(f, win_s, n_ch=n_ch, snr=snr, rng=rng) for _ in range(8)])
               for f in FREQS]
        model = calibrate(cal)
        fb = trca = n = 0
        for _ in range(60):
            true = rng.integers(len(FREQS))
            w = synth_ssvep(FREQS[true], win_s, n_ch=n_ch, snr=snr, rng=rng)
            fb += (classify(w)[0] == true)
            trca += (classify_trca(w, model)[0] == true)
            n += 1
        print(f"win={win_s:.2f}s SNR={snr}: FBCCA acc={fb/n:.2f}  "
              f"TRCA acc={trca/n:.2f}  (n={n}, 4-class)")
