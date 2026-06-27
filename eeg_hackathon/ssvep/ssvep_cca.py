"""ssvep_cca.py — training-free SSVEP decoding via CCA / FBCCA.

The winning core for "2048 with your mind": 4 arrows flicker at 4 frequencies;
the occipital EEG resonates at whichever the user looks at. CCA correlates the
EEG window against sine/cosine references per frequency — NO training needed —
and picks the max. Filter-bank CCA (FBCCA) sums sub-band correlations for a
robust boost.

A flicker frequency f is only renderable EXACTLY on a display if the half-period
refresh/(2f) is an integer number of frames. The nominal default set below is the
120Hz set (where all four are exact: 120/{8,10,12,14}=15/12/10/8.57). On a 60Hz
monitor those round badly (8.57->7.5, 12->15, colliding with 15). So the live app
must NOT hard-code FREQS — it calls `achievable_freqs(refresh)` at startup to get
the exact realizable frequencies for the detected refresh, and feeds THOSE same
values to both the renderer and the decoder (`classify(..., freqs=...)`). This
guarantees rendered flicker == decoder reference on any monitor.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.cross_decomposition import CCA

FS = 250
# Default reference set = the 120Hz-exact frequencies. The LIVE app overrides this
# per-monitor via achievable_freqs(refresh); these are the nominal targets / the
# values used for synthetic tests where there is no real display.
FREQS = [8.57, 10.0, 12.0, 15.0]          # Hz, one per arrow
ARROWS = ["up", "down", "left", "right"]
N_HARMONICS = 3


def achievable_freqs(refresh: float, n: int = 4) -> Tuple[List[float], List[int]]:
    """Exact, distinct flicker frequencies renderable on a `refresh`-Hz display.

    A square-wave flicker toggles every `half` frames, giving f = refresh/(2*half).
    Only integer `half` is exactly renderable. We pick `n` consecutive integer
    half-periods starting near the 15Hz end of the usable SSVEP band, so the
    frequencies are guaranteed distinct and physically exact on this monitor.

    Returns (freqs, halves) where freqs[i] = refresh/(2*halves[i]).
    Example: 60Hz -> [15, 10, 7.5, 6] (halves 2,3,4,5);
             120Hz -> [15, 12, 10, 8.571] (halves 4,5,6,7).
    """
    refresh = float(refresh)
    half_min = max(2, int(round(refresh / (2.0 * 15.0))))  # highest freq ~15Hz
    halves = [half_min + k for k in range(n)]
    freqs = [refresh / (2.0 * h) for h in halves]
    return freqs, halves


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
    # FBCCA (Chen et al. 2015): combined feature rho~_k = sum_n w(n)*(rho_k^n)^2,
    # with w(n)=n^-a + b. The SQUARING is per the original paper (eq. for rho~),
    # not a bug; sub-bands emphasise successive harmonics.
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
                fs: int = FS, rng=None, phase=None) -> np.ndarray:
    """Synthetic occipital SSVEP at `freq` (+harmonics) in 1/f noise -> (ch, n)."""
    rng = rng or np.random.default_rng()
    n = int(dur_s * fs); t = np.arange(n) / fs
    sig = np.zeros((n_ch, n))
    for h, amp in zip((1, 2, 3), (1.0, 0.5, 0.3)):
        ph = rng.uniform(0, 2 * np.pi) if phase is None else h * phase
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
