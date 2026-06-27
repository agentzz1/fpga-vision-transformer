# EEG Hackathon Kit — Unicorn Hybrid Black (BR41N.IO / Zeiss "EEG Mind Control")

Turnkey kit to win across all three BR41N.IO categories. **Code paths verified ALL GREEN**
on a synthetic smoke test (`python verify_all.py`); **accuracy verified on REAL public
EEG** (Nakanishi2015, PhysioNet, BNCI2014-009 — see `data_analysis/REAL_BENCHMARK.md` and
the committed `RUN_LOG_*.txt`). No hardware? Everything runs on a synthetic fallback.

## 60-second start
```bash
pip install -r requirements.txt
python verify_all.py          # fast SYNTHETIC smoke test (~13s) — proves every code path runs
python run.py                 # menu: pick any demo
```
**Reproduce every real headline number with ONE command** (asserts each against its floor):
```bash
pip install -r requirements-real.txt          # moabb, torch (downloads public datasets)
python verify_all.py --real                   # or: make reproduce   (~15-20 min, needs net)
```
`--real` runs all five public-data benchmarks (SSVEP TRCA, SSVEP montage, SSVEP live-regime,
MI, P300) and prints GREEN/RED per headline number — it is the single reproduce-all target.
**Synthetic vs real:** `verify_all.py` (no args) only proves the code runs end-to-end —
synthetic 100%s are not evidence. The numbers to quote come from `--real` and are recorded in
`REAL_BENCHMARK.md` / `RUN_LOG_*.txt`.
Read **DEMO_DAY.md** for the step-by-step demo-day flow + the 90-second judge pitch.

## What's inside (all verified)
**🎮 Gaming — SSVEP → 2048** (`app/ssvep_2048_app.py`): self-contained game; look at a
flickering arrow → it moves. Zero-training FBCCA decoder; optional TRCA calibration.
Why SSVEP: the Unicorn's spectral signal is reliable where its ERPs aren't
(Pontifex & Coffman 2023). Canabalt option: band-power **focus trigger**
(`canabalt/canabalt_focus.py`), motivated by Natalizio et al. 2024 (high focus/engagement
classification on the Unicorn during gameplay).

**📊 Data Analysis — point at a `.mat`, get SOTA** (`data_analysis/run_analysis.py`):
robust loader auto-detects layout; runs the right pipeline. **Ablation on SYNTHETIC data
(these 1.00s only show the code separates a clean signal — NOT real accuracy; for real
numbers see `data_analysis/REAL_BENCHMARK.md`: MI 8-ch 0.64, P300 8-ch AUC 0.94):**

| Paradigm | Method | Acc (synthetic) | AUC |
|---|---|---|---|
| Motor Imagery | raw+LDA (baseline) | 0.46 | - |
| Motor Imagery | CSP + LDA | 1.00 | - |
| Motor Imagery | Riemann tangent-space | 1.00 | 1.00 |
| Motor Imagery | EEGNet (CNN) | 1.00 | - |
| P300 | raw+LDA (baseline) | 0.88 | - |
| P300 | xDAWN + shrinkLDA | 0.96 | 0.99 |

**🛠 Core** (`acquire/preprocess/features/model`): LSL/CSV/synthetic acquisition (the
community-standard `unicorn2lsl` LSL path), MNE/SciPy filtering, band-power/ERP/tsfresh,
classifiers + CV.

## Files
`run.py` (launcher) · `verify_all.py` (readiness) · `DEMO_DAY.md` (runbook+pitch) ·
`app/` (2048 game) · `ssvep/` (FBCCA/TRCA/stim/online/A-B/freq-check) ·
`data_analysis/` (load, run_analysis, mi/p300/riemann/eegnet/baseline, BENCHMARK.md) ·
`canabalt/` (focus + RP) · core modules at root.

## Demo-day rule
Record a clean run early as backup. Fallback ladder in DEMO_DAY.md so a demo never dies.
