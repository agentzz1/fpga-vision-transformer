"""ssvep_cca.py — training-free SSVEP decoding via CCA / FBCCA.

The winning core for "2048 with your mind": 4 arrows flicker at 4 frequencies;
the occipital EEG resonates at whichever the user looks at. CCA correlates the
EEG window against sine/cosine references per frequency — NO training needed —
and picks the max. Filter-bank CCA (FBCCA) sums sub-band correlations for a
robust boost.

Frequencies are chosen as 60Hz-monitor sub-harmonics so the flicker is exact:
  60/7=8.57, 60/6=10, 60/5=12, 60/4=15  ->  ARROWS = up,down,left,right
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.cross_decomposition import CCA

FS = 250
FREQS = [8.57, 10.0, 12.0, 15.0]          # Hz, one per arrow
ARROWS = ["up", "down", "left", "right"]
N_HARMONICS = 3


def reference(freq: float, n: int, fs: int = FS, n_harm: int = N_HARMONICS) -> np.ndarray:
    """(n, 2*n_harm) sine/cosine reference bank for one frequency."""
    t = np.arange(n) / fs
    cols = []
    for h in range(1, n_harm + 1):
        cols.append(np.sin(2 * np.pi * h * freq * t))
        cols.append(np.cos(2 * np.pi * h * freq * t))
    return np.asarray(cols).T


def _cca_corr(X: np.ndarray, Y: np.ndarray) -> float:
    """Max canonical correlation between X (n,ch) and Y (n,2h)."""
    n = min(len(X), len(Y))
    cca = CCA(n_components=1)
    try:
        cca.fit(X[:n], Y[:n])
        xc, yc = cca.transform(X[:n], Y[:n])
        return abs(np.corrcoef(xc[:, 0], yc[:, 0])[0, 1])
    except Exception:
        return 0.0


def _bandpass(x, lo, hi, fs=FS):
    b, a = butter(4, [lo / (fs / 2), min(hi, fs / 2 - 1) / (fs / 2)], btype="band")
    return filtfilt(b, a, x, axis=0)


def classify(eeg_win: np.ndarray, freqs: List[float] = FREQS, fs: int = FS,
             fbcca: bool = True) -> Tuple[int, np.ndarray]:
    """eeg_win: (channels, samples) occipital EEG -> (class_idx, scores).

    class_idx indexes FREQS/ARROWS. scores are per-frequency CCA correlations.
    """
    X = np.asarray(eeg_win, float).T                     # (samples, ch)
    X = X - X.mean(0, keepdims=True)
    n = X.shape[0]
    if not fbcca:
        scores = np.array([_cca_corr(X, reference(f, n, fs)) for f in freqs])
        return int(np.argmax(scores)), scores
    # FBCCA: sub-bands emphasising successive harmonics, weighted a^-b + c
    bands = [(6, 50), (14, 50), (22, 50)]
    weights = [(k + 1) ** -1.25 + 0.25 for k in range(len(bands))]
    scores = np.zeros(len(freqs))
    for w, (lo, hi) in zip(weights, bands):
        Xb = _bandpass(X, lo, hi, fs)
        for i, f in enumerate(freqs):
            scores[i] += w * _cca_corr(Xb, reference(f, n, fs)) ** 2
    return int(np.argmax(scores)), scores


def itr_bits_per_min(n_classes: int, accuracy: float, t_select_s: float) -> float:
    """Wolpaw information transfer rate (bits/min)."""
    p, N = accuracy, n_classes
    if p <= 1.0 / N:
        bits = 0.0
    elif p >= 1.0:
        bits = np.log2(N)
    else:
        bits = (np.log2(N) + p * np.log2(p) + (1 - p) * np.log2((1 - p) / (N - 1)))
    return bits * (60.0 / t_select_s)


# --------------------------- synthetic validation -------------------------- #
def synth_ssvep(freq: float, dur_s: float, n_ch: int = 4, snr: float = 0.6,
                fs: int = FS, rng=None) -> np.ndarray:
    """Synthetic occipital SSVEP at `freq` (+harmonics) in 1/f noise -> (ch, n)."""
    rng = rng or np.random.default_rng()
    n = int(dur_s * fs); t = np.arange(n) / fs
    sig = np.zeros((n_ch, n))
    for h, amp in zip((1, 2, 3), (1.0, 0.5, 0.3)):
        ph = rng.uniform(0, 2 * np.pi)
        sig += amp * np.sin(2 * np.pi * h * freq * t + ph)[None, :]
    noise = rng.standard_normal((n_ch, n))
    # pink-ish noise
    for c in range(n_ch):
        noise[c] = np.cumsum(noise[c]); noise[c] -= noise[c].mean()
        noise[c] /= noise[c].std() + 1e-9
    return (snr * sig + noise).astype(np.float32)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    for dur in (1.0, 2.0, 3.0):
        correct = 0; trials = 40
        for _ in range(trials):
            true = rng.integers(len(FREQS))
            win = synth_ssvep(FREQS[true], dur, n_ch=4, snr=0.55, rng=rng)
            pred, _ = classify(win, fbcca=True)
            correct += (pred == true)
        acc = correct / trials
        itr = itr_bits_per_min(len(FREQS), acc, dur)
        print(f"window={dur:.1f}s  acc={acc:.2f}  ITR={itr:5.1f} bits/min  "
              f"(4-class, synthetic SNR~0.55)")
