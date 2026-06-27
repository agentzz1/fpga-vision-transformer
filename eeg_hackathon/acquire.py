"""acquire.py — EEG acquisition for the Unicorn pipeline.

Three interchangeable sources, all returning the canonical raw buffer
(8, n_samples) float32 + timestamps (n_samples,) float64 seconds:

  * LSLAcquirer      — live stream from the Unicorn Suite "Unicorn LSL" app.
  * CSVAcquirer      — a recording from the Unicorn Recorder (CSV).
  * SyntheticAcquirer— hardware-free generator with injected spacebar ERPs.

Plus a KeyLogger (pynput) for real spacebar timestamps and event helpers.
"""
from __future__ import annotations

import threading
import time
from typing import List, Optional, Tuple

import numpy as np

from eeg_common import (
    FS, EEG_CH, CHANNEL_NAMES, RAW_DTYPE, TIME_DTYPE,
    LABEL_SPACE, LABEL_REST, N_TIMES, TMIN,
)


# --------------------------- signal-quality gate --------------------------- #
def channel_quality(window, flat_uv=0.5, sat_uv=200.0):
    """Per-channel electrode-contact check for a live window (ch, samples), units uV.

    Returns (ok_mask, reasons): ok_mask[i] False if channel i is flat (dead/disconnected),
    railed/saturated, or NaN/Inf. The #1 dry-electrode demo failure is a lifting electrode;
    this surfaces it instead of silently decoding noise.
    """
    X = np.asarray(window, float)
    sd = X.std(axis=1)
    pp = X.max(axis=1) - X.min(axis=1)
    ok, reasons = [], []
    for i in range(X.shape[0]):
        if not np.all(np.isfinite(X[i])):
            ok.append(False); reasons.append("NaN/Inf")
        elif sd[i] < flat_uv:
            ok.append(False); reasons.append("flat")
        elif pp[i] > 2 * sat_uv:
            ok.append(False); reasons.append("railed")
        else:
            ok.append(True); reasons.append("ok")
    return np.array(ok), reasons


def notch_filter(X, fs=FS, freq=50.0, q=30.0):
    """Apply a mains notch (50 or 60 Hz) to a (ch, samples) array."""
    from scipy.signal import iirnotch, filtfilt
    b, a = iirnotch(freq / (fs / 2), q)
    return filtfilt(b, a, np.asarray(X, float), axis=-1)


# ----------------------------- event helpers ------------------------------- #
def presses_to_events(press_times, timestamps, label: int = LABEL_SPACE) -> np.ndarray:
    """Map press timestamps (sec, same clock as `timestamps`) -> events (n,2)."""
    ts = np.asarray(timestamps, dtype=float)
    out: List[List[int]] = []
    for t in np.atleast_1d(press_times):
        idx = int(np.searchsorted(ts, float(t)))
        if 0 <= idx < ts.size:
            out.append([idx, int(label)])
    return np.asarray(out, dtype=int).reshape(-1, 2)


def sample_rest_events(n_rest, n_samples, avoid_idx, margin_s=0.6, edge_s=0.9,
                       rng=None) -> np.ndarray:
    """Random no-press baseline events, kept >= margin_s away from presses."""
    rng = rng or np.random.default_rng(0)
    margin = int(margin_s * FS)
    avoid = np.asarray(list(avoid_idx), dtype=int)
    lo, hi = int(edge_s * FS), n_samples - int(edge_s * FS)
    out: List[List[int]] = []
    tries = 0
    while len(out) < n_rest and tries < n_rest * 100 and hi > lo:
        tries += 1
        c = int(rng.integers(lo, hi))
        if avoid.size and np.min(np.abs(avoid - c)) < margin:
            continue
        out.append([c, LABEL_REST])
    return np.asarray(out, dtype=int).reshape(-1, 2)


def build_events(press_times, timestamps, n_rest=None, rng=None) -> np.ndarray:
    """press events + matched rest events, concatenated & sorted by sample."""
    press = presses_to_events(press_times, timestamps)
    n_rest = len(press) if n_rest is None else n_rest
    rest = sample_rest_events(n_rest, len(timestamps), press[:, 0], rng=rng)
    ev = np.concatenate([press, rest], axis=0) if len(rest) else press
    return ev[np.argsort(ev[:, 0])]


# --------------------------- synthetic source ------------------------------ #
def _erp_template(n: Optional[int] = None) -> np.ndarray:
    """Per-channel evoked template (8, n_times): motor RP on C3/Cz/C4, P300 on
    parietal/occipital, ~zero on Fz."""
    n = N_TIMES if n is None else n
    t = np.linspace(TMIN, TMIN + (n - 1) / FS, n)
    tmpl = np.zeros((EEG_CH, n), dtype=np.float64)
    motor = (-2.5 * np.exp(-((t + 0.05) ** 2) / (2 * 0.05 ** 2))
             + 4.0 * np.exp(-((t - 0.15) ** 2) / (2 * 0.06 ** 2)))
    for ch in ("C3", "Cz", "C4"):
        tmpl[CHANNEL_NAMES.index(ch)] += motor
    p300 = 5.0 * np.exp(-((t - 0.30) ** 2) / (2 * 0.07 ** 2))
    for ch in ("Pz", "Oz", "PO7", "PO8"):
        tmpl[CHANNEL_NAMES.index(ch)] += 0.7 * p300
    return tmpl


class SyntheticAcquirer:
    """8ch/250Hz EEG with injected spacebar ERPs; returns ground-truth events."""

    def __init__(self, duration_s=120.0, n_presses=120, noise_uv=8.0, seed=7):
        self.duration_s = float(duration_s)
        self.n_presses = int(n_presses)
        self.noise_uv = float(noise_uv)
        self.rng = np.random.default_rng(seed)

    def _pink(self, n):
        white = self.rng.standard_normal((EEG_CH, n))
        x = np.zeros_like(white); a = 0.97; x[:, 0] = white[:, 0]
        for i in range(1, n):
            x[:, i] = a * x[:, i - 1] + (1 - a) * white[:, i]
        return x / (x.std(axis=1, keepdims=True) + 1e-9)

    def get_data(self):
        n = int(self.duration_s * FS)
        t = np.arange(n) / FS
        data = self.noise_uv * self._pink(n)
        alpha = np.sin(2 * np.pi * 10.0 * t + self.rng.uniform(0, 2 * np.pi))
        for ch in ("Pz", "Oz", "PO7", "PO8"):
            data[CHANNEL_NAMES.index(ch)] += 6.0 * alpha
        lo, hi = int(1.0 * FS), n - int(1.0 * FS)
        evt = np.sort(self.rng.choice(np.arange(lo, hi), size=self.n_presses,
                                      replace=False))
        keep = [evt[0]]
        for e in evt[1:]:
            if e - keep[-1] >= int(0.5 * FS):
                keep.append(e)
        evt = np.asarray(keep, dtype=int)
        tmpl = _erp_template(); pre = int(round(-TMIN * FS))
        for e in evt:
            s0, s1 = e - pre, e - pre + tmpl.shape[1]
            if s0 >= 0 and s1 <= n:
                data[:, s0:s1] += (1.0 + 0.15 * self.rng.standard_normal()) * tmpl
        return data.astype(RAW_DTYPE), t.astype(TIME_DTYPE), evt.astype(int)


# ------------------------------ CSV source --------------------------------- #
class CSVAcquirer:
    """Load a Unicorn Recorder CSV; first 8 numeric columns are EEG."""

    def __init__(self, path): self.path = path

    def get_data(self):
        import pandas as pd
        df = pd.read_csv(self.path)
        num = df.select_dtypes(include="number")
        eeg = num.iloc[:, :EEG_CH].to_numpy().T.astype(RAW_DTYPE)
        n = eeg.shape[1]
        tcol = next((c for c in df.columns if c.lower() in
                     ("timestamp", "time", "times", "lsl_timestamp")), None)
        ts = (df[tcol].to_numpy().astype(TIME_DTYPE) if tcol is not None
              else (np.arange(n) / FS).astype(TIME_DTYPE))
        return eeg, ts


# ------------------------------ LSL source --------------------------------- #
class LSLAcquirer:
    """Live Unicorn EEG via Lab Streaming Layer; rolling ring buffer."""

    def __init__(self, stream_name=None, buffer_s=30.0):
        self.stream_name = stream_name
        self.buffer_n = int(buffer_s * FS)
        self._buf = np.zeros((EEG_CH, self.buffer_n), dtype=RAW_DTYPE)
        self._ts = np.zeros(self.buffer_n, dtype=TIME_DTYPE)
        self._filled = 0
        self._lock = threading.Lock()
        self._run = False
        self._thread = None
        self._inlet = None
        self._ch_idx = list(range(EEG_CH))   # set properly in _resolve (by label if available)

    def _stream_labels(self, info):
        """Read per-channel labels from the LSL stream description XML, if present."""
        try:
            labels, ch = [], info.desc().child("channels").child("channel")
            for _ in range(info.channel_count()):
                if ch.empty():
                    break
                labels.append(ch.child_value("label") or ch.child_value("name") or "")
                ch = ch.next_sibling()
            return labels
        except Exception:
            return []

    def _resolve(self):
        from pylsl import resolve_streams, StreamInlet
        streams = resolve_streams(wait_time=3.0)
        if not streams:
            raise RuntimeError("No LSL streams. Start 'Unicorn LSL' in Unicorn Suite.")
        chosen = None
        for s in streams:
            if self.stream_name and self.stream_name.lower() in s.name().lower():
                chosen = s; break
            if s.type().upper() == "EEG":
                chosen = s; break
        chosen = chosen or max(streams, key=lambda s: s.channel_count())
        self._inlet = StreamInlet(chosen, max_buflen=60)
        info = self._inlet.info()
        # validate sampling rate (wrong/IRREGULAR srate => every freq reference is wrong)
        srate = info.nominal_srate()
        if srate and abs(srate - FS) > 1.0:
            print(f"[LSL] WARNING: stream srate={srate} Hz != expected FS={FS} Hz. "
                  f"Frequency references will be MISTUNED — set the Unicorn LSL rate to {FS}.")
        # map channels by LABEL when available; fall back to positional first-8
        labels = self._stream_labels(info)
        idx = []
        if labels:
            lut = {l.strip().lower(): i for i, l in enumerate(labels)}
            for name in CHANNEL_NAMES:
                idx.append(lut.get(name.lower()))
        if labels and all(i is not None for i in idx):
            self._ch_idx = idx
            mapping = ", ".join(f"{n}->#{i}" for n, i in zip(CHANNEL_NAMES, idx))
            print(f"[LSL] '{info.name()}' channel map (by label): {mapping}")
        else:
            self._ch_idx = list(range(EEG_CH))
            print(f"[LSL] '{info.name()}': no usable channel labels; using POSITIONAL "
                  f"first {EEG_CH} channels as {CHANNEL_NAMES}. VERIFY this matches your cap!")
        return chosen

    def start(self):
        self._resolve(); self._run = True
        self._thread = threading.Thread(target=self._pull, daemon=True)
        self._thread.start(); return self

    def _pull(self):
        while self._run:
            chunk, stamps = self._inlet.pull_chunk(timeout=0.2, max_samples=64)
            if not chunk:
                continue
            arr = np.asarray(chunk, dtype=np.float32)[:, self._ch_idx].T
            stamps = np.asarray(stamps, dtype=TIME_DTYPE); k = arr.shape[1]
            with self._lock:
                self._buf = np.roll(self._buf, -k, axis=1); self._buf[:, -k:] = arr
                self._ts = np.roll(self._ts, -k); self._ts[-k:] = stamps
                self._filled = min(self.buffer_n, self._filled + k)

    def get_data(self, seconds=None):
        with self._lock:
            n = self.buffer_n if seconds is None else min(self.buffer_n, int(seconds * FS))
            n = min(n, self._filled)
            return self._buf[:, -n:].copy(), self._ts[-n:].copy()

    def stop(self):
        self._run = False
        if self._thread:
            self._thread.join(timeout=1.0)


# ----------------------------- key logger ---------------------------------- #
class KeyLogger:
    """Record spacebar press timestamps (pynput) on the same clock as the EEG."""

    def __init__(self, clock=time.time, key="space"):
        self.clock = clock; self.key = key
        self.presses: List[float] = []; self._listener = None

    def start(self):
        from pynput import keyboard

        def on_press(k):
            name = getattr(k, "name", None) or getattr(k, "char", None)
            if name == self.key:
                self.presses.append(float(self.clock()))

        self._listener = keyboard.Listener(on_press=on_press)
        self._listener.start(); return self

    def stop(self):
        if self._listener:
            self._listener.stop()

    def save(self, path):
        np.savetxt(path, np.asarray(self.presses), header="press_time_s")


if __name__ == "__main__":
    acq = SyntheticAcquirer(duration_s=60.0, n_presses=60)
    data, ts, presses = acq.get_data()
    events = build_events(ts[presses], ts)
    print(f"buffer shape : {data.shape} dtype={data.dtype}")
    print(f"timestamps   : {ts.shape} ({ts[0]:.2f}..{ts[-1]:.2f}s)")
    print(f"press events : {len(presses)}")
    print(f"events array : {events.shape} "
          f"(press={int((events[:,1]==LABEL_SPACE).sum())}, "
          f"rest={int((events[:,1]==LABEL_REST).sum())})")
