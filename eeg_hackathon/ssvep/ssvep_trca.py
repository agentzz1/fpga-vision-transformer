"""ssvep_trca.py — TRCA (Task-Related Component Analysis) calibration for SSVEP.

Training-free FBCCA is the baseline that works instantly. TRCA adds a short
(~100 s, see REAL_BENCHMARK.md) per-user calibration that learns spatial filters +
templates per target, typically beating CCA/FBCCA — especially at SHORT windows. This
is the algorithm behind the highest-ITR EEG BCI on record (Nakanishi et al. 2018).

NOTE: TRCA's advantage shows on REAL EEG (real_ssvep_trca.py: FBCCA 0.93 -> TRCA 0.99).
On the PURELY synthetic sinusoids below, plain CCA/FBCCA is already near-perfect and can
match or beat TRCA — synthetic SSVEP has no task-related component for TRCA to exploit,
so do NOT read the synthetic self-test as the TRCA verdict. The real-data benchmark is.

Workflow:
    model = calibrate(trials_by_class)        # trials_by_class[k]: (n_trials, ch, T)
    cls, scores = classify_trca(window, model) # window: (ch, T) -> class idx
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import scipy.linalg as la
from scipy.signal import butter, filtfilt

from ssvep_cca import FREQS, ARROWS, FS

TRCA_BAND = (6.0, 50.0)   # SSVEP passband applied identically in calibrate + classify


def _bp(X, lo=TRCA_BAND[0], hi=TRCA_BAND[1], fs=FS):
    """Zero-phase Butterworth bandpass on the LAST (time) axis. Bypasses if window
    is too short for filtfilt's padding (keeps the synthetic smoke test working)."""
    X = np.asarray(X, float)
    if X.shape[-1] < 28:
        return X
    b, a = butter(4, [lo / (fs / 2), min(hi, fs / 2 - 1) / (fs / 2)], btype="band")
    return filtfilt(b, a, X, axis=-1)


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


def calibrate(trials_by_class: List[np.ndarray], band=TRCA_BAND) -> Dict:
    """Build ensemble-TRCA model: a spatial filter + template per class.

    Epochs are BANDPASSED to the SSVEP band before computing TRCA filters/templates,
    so the spatial filter isn't dominated by drift/mains; classify_trca applies the
    identical band to the test window (stored in the model)."""
    filters, templates = [], []
    for trials in trials_by_class:
        trials = _bp(np.asarray(trials, float), band[0], band[1])
        filters.append(_trca_filter(trials))
        templates.append(trials.mean(axis=0))            # (ch, T) trial-average
    W = np.asarray(filters).T                            # (ch, K) ensemble filters
    return {"W": W, "templates": templates, "freqs": FREQS, "band": band}


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel(), b.ravel()
    a = a - a.mean(); b = b - b.mean()
    d = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(a @ b / d)


def classify_trca(window: np.ndarray, model: Dict) -> Tuple[int, np.ndarray]:
    """Ensemble-TRCA: correlate filtered test vs each filtered template."""
    X = _bp(np.asarray(window, float), *model.get("band", TRCA_BAND))  # same passband as calibrate
    X = X - X.mean(axis=1, keepdims=True)
    W, templates = model["W"], model["templates"]
    Xf = W.T @ X                                          # (K, T)
    scores = np.array([_corr(Xf, (W.T @ templates[k])) for k in range(len(templates))])
    return int(np.argmax(scores)), scores


def _selftest_phaselocked(seed=3):
    """Offline-reproducible correctness check for the eTRCA math.

    The point is to PROVE the calibrate()/classify_trca() pipeline decodes phase-locked
    SSVEP well (templates + ensemble spatial filters are computed correctly), NOT to
    claim TRCA beats CCA on synthetic. Real SSVEP is phase-locked, so each class gets a
    fixed phase (+ small jitter) and a fixed scalp pattern; we assert TRCA decodes this
    far above the 0.25 chance level. (Whether TRCA > FBCCA is settled on REAL data only —
    real_ssvep_trca.py: 0.93 -> 0.99 — because clean synthetic sinusoids already saturate
    CCA, leaving nothing for a spatial filter to improve.)
    """
    rng = np.random.default_rng(seed)
    n_ch, win_s, fs = 4, 1.0, FS
    n = int(win_s * fs); t = np.arange(n) / fs
    class_phase = {i: rng.uniform(0, 2 * np.pi) for i in range(len(FREQS))}
    spatial = np.array([1.0, 0.6, 0.3, 0.8])           # fixed SSVEP scalp pattern
    def trial(c):
        ph = class_phase[c] + rng.uniform(-0.15, 0.15)
        x = spatial[:, None] * np.sin(2 * np.pi * FREQS[c] * t + ph)[None, :]
        x = x + 1.5 * rng.standard_normal((n_ch, n))   # noisy (SNR<1 per channel)
        return x.astype(np.float32)
    from ssvep_cca import classify
    cal = [np.stack([trial(c) for _ in range(10)]) for c in range(len(FREQS))]
    model = calibrate(cal)
    fb = tr = m = 0
    for _ in range(80):
        c = int(rng.integers(len(FREQS)))
        w = trial(c)
        fb += (classify(w, fbcca=True)[0] == c)
        tr += (classify_trca(w, model)[0] == c)
        m += 1
    print(f"phase-locked SNR<1 win={win_s}s: FBCCA={fb/m:.2f}  TRCA={tr/m:.2f}  (n={m}, 4-class, chance=0.25)")
    return tr / m, fb / m


if __name__ == "__main__":
    print("[note] This is a MATH correctness check (eTRCA decodes phase-locked data), not "
          "a TRCA-vs-FBCCA verdict. On clean synthetic CCA already saturates; the real\n"
          "verdict is real_ssvep_trca.py (FBCCA 0.93 -> TRCA 0.99).\n")
    tr, fb = _selftest_phaselocked()
    assert tr >= 0.80, f"eTRCA should decode phase-locked SSVEP well; got {tr:.2f} (chance 0.25)"
    print(f"OK: eTRCA decodes phase-locked synthetic at {tr:.2f} (>> 0.25 chance) — math verified.")
