# PITCH — "Mind-Controlled 2048 on an 8-channel headset" (BR41N.IO / Zeiss)

## The one-liner
We control a real game with brain signals on a **consumer 8-channel Unicorn** — using
the **same SSVEP paradigm behind the world's highest-bitrate BCIs**, validated on **real
public EEG**, and we picked every method by **measured real-data performance, not hype.**

## The demo (60–90 s, live)
1. Put on the Unicorn → start Unicorn LSL.
2. Look at a flickering arrow → **2048 moves by brain**. (4 frequencies ↔ 4 arrows.)
3. Show the **60-s calibration** flip: zero-training **0.82 → calibrated 0.98**.
4. (Backup) pre-recorded clip; (fallback) synthetic showcase — demo never dies.

## Why we win — evidence, not claims (all on REAL public datasets)
| Paradigm | Real dataset (8-ch where possible) | Result |
|---|---|---|
| **SSVEP (our game)** | Nakanishi2015, 8-ch, 12-class | FBCCA 0.83 → **TRCA 0.98**, 0/60-s train |
| Motor Imagery | PhysioNet, **8-ch = 64-ch** (0.64 vs 0.65) | CSP/Riemann robust on small data |
| P300 speller | BNCI2014-009 | xDAWN+LDA **AUC 0.96** |

## Three insights that show depth (judges reward these)
1. **8 channels retain most of the MI signal** for this task — the signal is sensorimotor/occipital,
   which the Unicorn has. The headset isn't the bottleneck; trial count is.
2. **The Unicorn's spectral signal is reliable where its ERPs aren't** (Pontifex 2023),
   which is exactly *why* we chose SSVEP over P300/readiness-potential for live control.
3. **On hackathon-sized data, simpler wins** — CSP/Riemann beat FBCSP/EEGNet at 45 trials;
   we auto-select method by data size. We measured this; we didn't guess.

## What we built (turnkey, `verify_all.py` = ALL GREEN)
Self-contained SSVEP→2048 game · zero-train FBCCA + calibrated TRCA · band-power focus
trigger (Canabalt) · full data-analysis suite (CSP, FBCSP, Riemann, EEGNet, xDAWN) with
auto method-selection · LSL/CSV/synthetic acquisition · one-command launcher + runbook.

## Ask
We're the team that brought a **measured, real-data-validated, reproducible** BCI — not a
lucky live fluke. Every number above reproduces with one command.
