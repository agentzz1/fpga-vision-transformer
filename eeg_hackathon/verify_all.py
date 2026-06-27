"""verify_all.py — one-command readiness check.

Default (no args): runs EVERY no-hardware component on SYNTHETIC signals and prints
a GREEN/RED report. This proves the code paths run end-to-end; it does NOT prove
accuracy on real EEG (synthetic 100%s mean nothing on their own).

    python verify_all.py            # fast synthetic smoke test (seconds, no network)
    python verify_all.py --real     # ALSO run the REAL public-data benchmarks
                                    # (downloads via MOABB/MNE; minutes; needs net).

The real numbers (the ones to quote) live in the committed run logs:
  ssvep/RUN_LOG_ssvep_trca.txt, data_analysis/RUN_LOG_p300.txt, and REAL_BENCHMARK.md.
"""
import os, sys, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
REAL = "--real" in sys.argv
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

print("\n==== EEG HACKATHON KIT — READINESS (SYNTHETIC smoke test) ====")
allok = True
for name, ok, detail in results:
    print(f"  [{'GREEN' if ok else 'RED  '}] {name:16} {detail}")
    allok &= ok
print("=" * 42)
print("ALL GREEN (synthetic) — code paths run end-to-end."
      if allok else "SOME RED — see above.")
print("Note: these are SYNTHETIC checks. Real-EEG numbers: see REAL_BENCHMARK.md /"
      " RUN_LOG_*.txt, or run `python verify_all.py --real`.")

if REAL:
    import re
    print("\n==== REAL public-data benchmarks (downloads via MOABB/MNE) ====")
    print("  Each ASSERTS its headline number is at/above the committed RUN_LOG floor.\n")
    # (script, [(label, regex capturing a float, min_threshold), ...])
    # Thresholds are conservative floors below the committed measured values, so a
    # GREEN here means 'the headline number reproduced', not merely 'exited 0'.
    # Floors track the COMMITTED HEADLINE numbers (RUN_LOG_*.txt), set ~0.05 below the
    # mean to allow seed/cohort jitter. A GREEN here means the quoted headline reproduced.
    checks = [
        ("ssvep/real_ssvep_trca.py", [
            ("SSVEP FBCCA mean", r"FBCCA:\s*mean=([0-9.]+)", 0.88),   # headline 0.93 (n=9)
            ("SSVEP TRCA mean",  r"TRCA\s*:\s*mean=([0-9.]+)",  0.95),   # headline 0.99 (n=9)
        ]),
        ("ssvep/real_ssvep_montage_ablation.py", [
            ("SSVEP Unicorn-4-posterior", r"Unicorn-4-posterior:\s*([0-9.]+)", 0.85),  # headline 0.91
        ]),
        ("data_analysis/real_mi_benchmark.py", [
            ("MI Unicorn-8ch CSP", r"\[Unicorn-8ch\]\s*CSP=([0-9.]+)", 0.58),  # headline 0.64 (n=8)
        ]),
        ("data_analysis/real_p300_benchmark.py", [
            ("P300 Unicorn-8ch AUC", r"Unicorn-8ch AUC=([0-9.]+)", 0.90),  # headline 0.937 (n=5)
        ]),
    ]
    for rel, asserts in checks:
        print(f"--- {rel} ---")
        out = subprocess.run([sys.executable, os.path.join(HERE, rel)],
                             capture_output=True, text=True)
        sys.stdout.write(out.stdout[-1500:])
        if out.returncode != 0:
            print(f"  [RED] {rel} exited {out.returncode}\n{out.stderr[-500:]}"); allok = False; continue
        text = out.stdout
        for label, pat, thr in asserts:
            m = re.findall(pat, text, re.M)
            if not m:
                print(f"  [RED] {label}: number not found in output"); allok = False
            else:
                val = float(m[-1])
                ok = val >= thr
                allok &= ok
                print(f"  [{'GREEN' if ok else 'RED  '}] {label}={val:.3f} (floor {thr})")
        print()

sys.exit(0 if allok else 1)
