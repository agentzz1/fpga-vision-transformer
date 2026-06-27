# PITCH — "Mind-Controlled 2048 on an 8-channel headset" (BR41N.IO / Zeiss)

## The one-liner
We control a real game with brain signals on a **consumer 8-channel Unicorn** — using
the **same SSVEP paradigm behind the world's highest-bitrate BCIs**, validated on **real
public EEG**, and we picked every method by **measured real-data performance, not hype.**

## The demo (60–90 s, live)
1. Put on the Unicorn → start Unicorn LSL.
2. Look at a flickering arrow → **2048 moves by brain**. (4 frequencies ↔ 4 arrows.)
3. Zero-training FBCCA already averages **0.93** (n=9); a short ~100-s TRCA calibration
   rescues weak responders → cohort **0.99**.
4. (Backup) pre-recorded clip; (fallback) synthetic showcase — demo never dies.

## Why we win — evidence, not claims (all on REAL public datasets)
| Paradigm | Real dataset (8-ch where possible) | Result |
|---|---|---|
| **SSVEP (our game)** | Nakanishi2015, 8-ch, 12-class, n=9 | FBCCA **0.93** → **TRCA 0.99** (0/~100-s train) |
| Motor Imagery | PhysioNet, n=8, 8-ch 0.64 ≈ 64-ch 0.65 (CSP) | 8 ch keep the full MI signal |
| P300 speller | BNCI2014-009 (n=5) | xDAWN+LDA **AUC 0.94 ± 0.05** (leak-free CV) |

## Three insights that show depth (judges reward these)
1. **8 channels retain the full MI signal** here (8-ch 0.64 ≈ 64-ch 0.65, n=8) — the signal is
   sensorimotor/occipital, which the Unicorn has. The headset isn't the bottleneck; trial count is.
2. **The Unicorn's spectral signal is reliable where its ERPs aren't** (Pontifex 2023),
   which is exactly *why* we chose SSVEP over P300/readiness-potential for live control.
3. **On hackathon-sized data, the deep net loses** — EEGNet is worst at 45 trials, CSP/Riemann/FBCSP tie;
   we auto-select method by data size. We measured this; we didn't guess.

## What we built (turnkey, `verify_all.py` = ALL GREEN)
Self-contained SSVEP→2048 game · zero-train FBCCA + calibrated TRCA · band-power focus
trigger (Canabalt) · full data-analysis suite (CSP, FBCSP, Riemann, EEGNet, xDAWN) with
auto method-selection · LSL/CSV/synthetic acquisition · one-command launcher + runbook.

## Ask
We're the team that brought a **measured, real-data-validated, reproducible** BCI — not a
lucky live fluke. Every number above reproduces with one command.
