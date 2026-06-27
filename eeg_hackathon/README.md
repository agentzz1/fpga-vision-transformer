# EEG Hackathon Kit — Unicorn Hybrid Black (BR41N.IO / Zeiss "EEG Mind Control")

Turnkey kit to win across all three BR41N.IO categories. **Code paths verified ALL GREEN**
on a synthetic smoke test (`python verify_all.py`); **accuracy verified on REAL public
EEG** (Nakanishi2015, PhysioNet, BNCI2014-009 — see `data_analysis/REAL_BENCHMARK.md` and
the committed `RUN_LOG_*.txt`). No hardware? Everything runs on a synthetic fallback.

## 60-second start
```bash
pip install -r requirements.txt
python verify_all.py          # fast SYNTHETIC smoke test — proves every code path runs
python verify_all.py --real   # ALSO run the REAL public-data benchmarks (downloads; minutes)
pip install -r requirements-real.txt   # needed for --real (moabb, torch)
python run.py                 # menu: pick any demo
```
**Synthetic vs real:** `verify_all.py` (no args) only proves the code runs end-to-end —
synthetic 100%s are not evidence. The numbers to quote come from the real-data benchmarks
(`--real`) and are recorded in `REAL_BENCHMARK.md` / `RUN_LOG_*.txt`.
Read **DEMO_DAY.md** for the step-by-step demo-day flow + the 90-second judge pitch.

## What's inside (all verified)
**🎮 Gaming — SSVEP → 2048** (`app/ssvep_2048_app.py`): self-contained game; look at a
flickering arrow → it moves. Zero-training FBCCA decoder; optional TRCA calibration.
Why SSVEP: the Unicorn's spectral signal is reliable where its ERPs aren't
(Pontifex & Coffman 2023). Canabalt option: band-power **focus trigger**
(`canabalt/canabalt_focus.py`), motivated by Natalizio et al. 2024 (high focus/engagement
classification on the Unicorn during gameplay).

**📊 Data Analysis — point at a `.mat`, get SOTA** (`data_analysis/run_analysis.py`):
robust loader auto-detects layout; runs the right pipeline. Methods + synthetic ablation:

| Paradigm | Method | Acc | AUC |
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
