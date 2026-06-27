"""real_ssvep_live_regime.py — the SSVEP number that matches the LIVE GAME.

The flagship headline (FBCCA 0.93) is measured at Nakanishi's 4.2 s epoch / 12 classes.
The live 2048 game uses a 2 s window and only 4 targets. Those push in opposite directions
(4-class easier, 2 s harder), so the net is genuinely different — this script measures it
directly so PITCH/DEMO_DAY can quote the LIVE-REGIME number, not the 12-class one.

    python real_ssvep_live_regime.py
"""
from __future__ import annotations
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from ssvep_cca import classify


# Nakanishi order: PO7,PO3,POz,PO4,PO8,O1,Oz,O2 -> Unicorn-4-posterior = PO7,PO8,Oz,POz
UNICORN4 = [0, 4, 6, 2]


def run(subjects=tuple(range(1, 10)), win_s=2.0, n_targets=4, require_n=9):
    from moabb.datasets import Nakanishi2015
    from moabb.paradigms import SSVEP
    ds = Nakanishi2015(); para = SSVEP(n_classes=12); fs = 256
    nsamp = int(win_s * fs)
    print(f"Nakanishi2015 — LIVE-REGIME FBCCA: {win_s}s window, {n_targets} targets, "
          f"Unicorn-4-posterior channels (ALL THREE live penalties stacked), n up to {len(subjects)}:\n")
    accs, missing = [], []
    for s in subjects:
        try:
            X, y, _ = para.get_data(ds, [s])
        except Exception as e:
            print(f"  S{s}: skipped ({type(e).__name__})"); missing.append(s); continue
        freqs_all = sorted({float(v) for v in np.unique(y)})
        # pick n_targets well-separated frequencies (like 4 arrows spread across the band)
        pick = list(np.linspace(0, len(freqs_all) - 1, n_targets).round().astype(int))
        sub_freqs = [freqs_all[i] for i in pick]
        yi = np.array([freqs_all.index(min(freqs_all, key=lambda z: abs(z - float(v)))) for v in y])
        mask = np.isin(yi, pick)
        # 2 s window + 4-class subset + 4 Unicorn-posterior channels = full live regime
        Xs, ys = X[mask][:, UNICORN4, :nsamp], yi[mask]
        lut = {p: k for k, p in enumerate(pick)}
        correct = np.mean([classify(Xs[k], freqs=sub_freqs, fs=fs)[0] == lut[ys[k]]
                           for k in range(len(Xs))])
        accs.append(correct)
        print(f"  S{s}: {len(Xs)} trials, {n_targets}-class, {win_s}s, 4ch -> FBCCA acc={correct:.2f}")
    accs = np.array(accs)
    print(f"\n  LIVE-REGIME FBCCA: mean={accs.mean():.2f} +/- {accs.std():.2f}  "
          f"(n={len(accs)}, {n_targets}-class, {win_s}s, Unicorn-4-posterior, 0-train, "
          f"chance={1/n_targets:.2f})")
    if len(accs) < require_n:
        print(f"\n  FAIL: only {len(accs)}/{require_n} subjects (missing {missing}); RE-RUN.")
        sys.exit(1)


if __name__ == "__main__":
    run()
