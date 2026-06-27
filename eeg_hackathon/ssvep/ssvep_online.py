"""ssvep_online.py — live SSVEP -> 2048 arrow-key control.

Pipeline: LSLAcquirer (Unicorn) -> occipital channels -> sliding window ->
FBCCA classify -> confidence gate + dwell -> press the arrow key (pynput) into
whatever window has focus (the 2048 game).

    python ssvep_online.py                 # live (needs Unicorn LSL + display)
    python ssvep_online.py --simulate      # offline logic test (no hardware)

Confidence gate: emit only if the top FBCCA score beats the runner-up by `margin`
and the same arrow is stable for `dwell` consecutive windows -> robust, few false
triggers (critical for a clean live demo).
"""
from __future__ import annotations
import os as _os, sys as _sys
_HB = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_HB, _os.path.join(_HB, "ssvep"), _os.path.join(_HB, "app")):
    if _p not in _sys.path: _sys.path.insert(0, _p)
import argparse
import sys
import time
from collections import deque

import numpy as np

from ssvep_cca import classify, synth_ssvep, FREQS, ARROWS, FS

OCCIPITAL = ["Oz", "PO7", "PO8", "Pz"]     # Unicorn channels used for SSVEP


def _occipital_idx():
    from eeg_common import CHANNEL_NAMES
    return [CHANNEL_NAMES.index(c) for c in OCCIPITAL]


class SSVEPController:
    """Turns a stream of EEG windows into stable arrow decisions."""

    def __init__(self, margin=0.15, dwell=2, refractory_s=0.8, freqs=FREQS):
        self.margin = margin
        self.dwell = dwell
        self.refractory_s = refractory_s
        self.hist = deque(maxlen=dwell)
        self.last_fire = 0.0
        # MUST match the stimulus frequencies (refresh-locked). For the two-terminal
        # stim+online workflow, pass the same achievable_freqs(refresh) used by ssvep_stim.
        self.freqs = freqs

    def step(self, win, now=None):
        """win: (n_occipital_ch, samples). Returns arrow str or None."""
        now = time.time() if now is None else now
        idx, scores = classify(win, freqs=self.freqs)
        order = np.argsort(scores)[::-1]
        top, second = scores[order[0]], scores[order[1]]
        conf_ok = (top - second) >= self.margin * (abs(top) + 1e-9)
        self.hist.append(idx if conf_ok else -1)
        stable = conf_ok and len(self.hist) == self.dwell and len(set(self.hist)) == 1
        if stable and (now - self.last_fire) >= self.refractory_s:
            self.last_fire = now
            return ARROWS[idx]
        return None


def _press(arrow: str):
    from pynput.keyboard import Controller, Key
    kb = Controller()
    key = {"up": Key.up, "down": Key.down, "left": Key.left, "right": Key.right}[arrow.lower()]
    kb.press(key); kb.release(key)


def _detect_refresh(default=60):
    """Best-effort monitor refresh via pygame; falls back to `default` headless."""
    try:
        import pygame
        pygame.display.init()
        r = int(round(pygame.display.get_current_refresh_rate()))
        pygame.display.quit()
        return r or default
    except Exception:
        return default


def run_live(window_s=2.0, step_s=0.4, refresh=None, notch=50.0):
    from acquire import LSLAcquirer, channel_quality, notch_filter
    from ssvep_cca import achievable_freqs
    if refresh is None:
        refresh = _detect_refresh()
    occ = _occipital_idx()
    acq = LSLAcquirer().start()
    freqs, _ = achievable_freqs(refresh, n=4)
    ctrl = SSVEPController(freqs=freqs)
    print(f"SSVEP->2048 live (refresh={refresh}Hz, freqs={[round(f,2) for f in freqs]}, "
          f"notch={notch}Hz). Look at an arrow. Channels={OCCIPITAL}. Ctrl-C to stop.")
    try:
        while True:
            data, _ = acq.get_data(seconds=window_s)
            if data.shape[1] >= int(window_s * FS):
                win = data[occ, -int(window_s * FS):]
                ok, reasons = channel_quality(win)        # electrode-contact gate
                if not ok.all():
                    bad = [OCCIPITAL[i] for i, g in enumerate(ok) if not g]
                    print(f"  [BAD CONTACT: {','.join(bad)}] — fix electrodes, not decoding")
                else:
                    if notch:
                        win = notch_filter(win, fs=FS, freq=notch)   # kill 50/60Hz mains
                    arrow = ctrl.step(win)
                    if arrow:
                        print(f"  -> {arrow}")
                        _press(arrow)
            time.sleep(step_s)
    except KeyboardInterrupt:
        acq.stop()


def run_simulate(trials=24, window_s=2.0):
    """Offline check of the full decision logic on synthetic SSVEP (no hardware)."""
    rng = np.random.default_rng(1)
    ctrl = SSVEPController(dwell=1)            # single-window for the test
    correct = 0
    for _ in range(trials):
        true = rng.integers(len(FREQS))
        win = synth_ssvep(FREQS[true], window_s, n_ch=len(OCCIPITAL), snr=0.5, rng=rng)
        idx, scores = classify(win)
        correct += (idx == true)
    print(f"[simulate] decision-logic accuracy = {correct}/{trials} "
          f"= {correct/trials:.2f} (4-class synthetic, {window_s}s windows)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--refresh", type=int, default=None,
                    help="monitor refresh Hz (must match ssvep_stim); auto-detected if omitted")
    ap.add_argument("--notch", type=float, default=50.0, help="mains notch Hz (50 EU/60 US; 0=off)")
    a = ap.parse_args()
    run_simulate() if a.simulate else run_live(refresh=a.refresh, notch=(a.notch or None))
