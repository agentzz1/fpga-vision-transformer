"""verify_all.py — one-command readiness check. Runs EVERY no-hardware component
and prints a GREEN/RED report. If all green, the kit is demo-ready.

    python verify_all.py
"""
import os, sys, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
for sub in ("ssvep", "app", "canabalt", "data_analysis"):
    sys.path.insert(0, os.path.join(HERE, sub))

results = []
def check(name, fn):
    try:
        ok, detail = fn(); results.append((name, ok, detail))
    except Exception as e:
        results.append((name, False, f"ERR {type(e).__name__}: {e}"))

import numpy as np

def _ssvep():
    from ssvep_cca import classify, synth_ssvep, FREQS
    rng = np.random.default_rng(0); c = 0
    for _ in range(40):
        t = rng.integers(4); c += classify(synth_ssvep(FREQS[t], 2.0, 4, 0.55, rng=rng))[0] == t
    return c/40 >= 0.9, f"SSVEP FBCCA acc={c/40:.2f}"

def _game():
    from game_2048 import Game2048
    g = Game2048(seed=1); m = 0
    while g.can_move() and m < 500:
        for d in ("up","left","down","right"):
            if g.move(d): break
        m += 1
    return g.score > 0, f"2048 engine score={g.score}, max={g.max_tile()}"

def _mi():
    from mi_pipeline import _synth_mi, evaluate
    r = evaluate(*_synth_mi()); return r["acc"] >= 0.8, f"MI CSP+LDA acc={r['acc']:.2f}"

def _riemann():
    from riemann_pipeline import evaluate
    from mi_pipeline import _synth_mi
    r = evaluate(*_synth_mi()); return r["acc"] >= 0.8, f"Riemann acc={r['acc']:.2f}"

def _p300():
    from p300_pipeline import _synth_p300, evaluate
    r = evaluate(*_synth_p300()); return r["auc"] >= 0.8, f"P300 xDAWN+LDA auc={r['auc']:.2f}"

def _focus():
    from canabalt_focus import _synth_focus, _epochs
    from features import bandpower_features
    from model import evaluate as ev
    d, e = _synth_focus(); X, y = _epochs(d, e); F, _ = bandpower_features(X)
    r = ev(F, y, "rf"); return r["acc"] >= 0.8, f"focus acc={r['acc']:.2f}"


def _eegnet():
    import importlib
    if not importlib.util.find_spec("torch"):
        return True, "EEGNet skipped (no torch) — numpy pipelines cover it"
    from eegnet import evaluate
    from mi_pipeline import _synth_mi
    r = evaluate(*_synth_mi(n_per=40), epochs=30)
    return r["acc"] >= 0.8, f"EEGNet acc={r['acc']:.2f}"

def _baseline():
    from baseline import evaluate
    from mi_pipeline import _synth_mi
    r = evaluate(*_synth_mi()); return True, f"baseline acc={r['acc']:.2f} (ablation ref)"

for n, f in [("SSVEP decoder", _ssvep), ("EEGNet", _eegnet), ("Baseline", _baseline), ("2048 game", _game), ("Motor Imagery", _mi),
             ("Riemannian", _riemann), ("P300 speller", _p300), ("Focus trigger", _focus)]:
    check(n, f)

print("\n==== EEG HACKATHON KIT — READINESS ====")
allok = True
for name, ok, detail in results:
    print(f"  [{'GREEN' if ok else 'RED  '}] {name:16} {detail}")
    allok &= ok
print("=" * 42)
print("ALL GREEN — kit is demo-ready." if allok else "SOME RED — see above.")
sys.exit(0 if allok else 1)
