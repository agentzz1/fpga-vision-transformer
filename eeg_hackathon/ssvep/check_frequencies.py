"""check_frequencies.py — lock the best 4 SSVEP frequencies on YOUR screen+subject.

Records a few seconds while you fixate each flicker, prints per-frequency FBCCA
separability so you can confirm (or swap) the 4 targets before going live.

    python check_frequencies.py            # synthetic self-check (no hardware)
    python check_frequencies.py --live     # record per-target from Unicorn LSL
"""
from __future__ import annotations
import os as _os, sys as _sys
_HB = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_HB, _os.path.join(_HB, "ssvep")):
    if _p not in _sys.path: _sys.path.insert(0, _p)
import argparse, time
import numpy as np
from ssvep_cca import classify, synth_ssvep, FREQS, ARROWS, FS


def report(trials_by_target):
    """trials_by_target[i]: list of (ch,T) windows recorded while gazing target i."""
    print(f"{'target':8}{'freq':>7}{'self-score':>12}{'best-other':>12}{'sep':>7}")
    correct = 0; total = 0
    for i, wins in enumerate(trials_by_target):
        for w in wins:
            _, sc = classify(w)
            self_s = sc[i]; other = max(sc[j] for j in range(len(sc)) if j != i)
            correct += (np.argmax(sc) == i); total += 1
        # aggregate on the last window for display
        _, sc = classify(wins[-1])
        self_s = sc[i]; other = max(sc[j] for j in range(len(sc)) if j != i)
        print(f"{ARROWS[i]:8}{FREQS[i]:>7.2f}{self_s:>12.3f}{other:>12.3f}"
              f"{self_s-other:>7.3f}")
    print(f"\noverall decode accuracy: {correct}/{total} = {correct/total:.2f}")
    print("-> any target with sep<=0 is weak: set its FREQS entry to another "
          "refresh/integer value in ssvep_cca.py")


def live(n=6, win_s=2.5):
    from acquire import LSLAcquirer
    from ssvep_online import _occipital_idx
    occ = _occipital_idx(); acq = LSLAcquirer().start()
    out = []
    for i, f in enumerate(FREQS):
        input(f"Fixate the {ARROWS[i]} flicker ({f:.2f} Hz), press Enter...")
        wins = []
        for _ in range(n):
            time.sleep(win_s)
            d, _ = acq.get_data(seconds=win_s); wins.append(d[occ, -int(win_s*FS):])
        out.append(wins)
    acq.stop(); report(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--live", action="store_true")
    a = ap.parse_args()
    if a.live: live()
    else:
        rng = np.random.default_rng(0)
        tbt = [[synth_ssvep(f, 2.5, n_ch=4, snr=0.5, rng=rng) for _ in range(6)] for f in FREQS]
        report(tbt)
