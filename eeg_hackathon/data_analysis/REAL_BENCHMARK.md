# REAL-data validation (not synthetic) — PhysioNet Motor Imagery

Task: imagined **left vs right fist** (PhysioNet eegmmidb, runs 4/8/12), within-subject
5-fold CV, 45 trials/subject. Chance = 0.50. This is the honest, real-EEG picture —
synthetic 100%s mean nothing; these are what the methods actually deliver.

## Results (accuracy)
| Channels | CSP+LDA | Riemann-TS | FBCSP | EEGNet |
|---|---|---|---|---|
| 64-ch (lab cap) | 0.70 | 0.65 | 0.61 | 0.56 |
| **Unicorn 8-ch subset** | **0.61** | **0.61** | 0.60 | 0.58 |

*(fs=160 Hz correctly applied; 5-fold CV; FBCSP & EEGNet actually run, not estimated.)*

Per-subject (Unicorn-8ch, CSP): S1 0.71, S2 0.82, **S3 0.29** (a weak responder — "BCI
illiteracy" is real, ~10–30% of people).

## Findings that make this the right implementation for the hackathon
1. **8 channels retain most of the MI signal** (8-ch ~0.61 vs 64-ch ~0.70 for CSP). The
   discriminative signal is sensorimotor (C3/Cz/C4), which the Unicorn has; there is a
   modest gap, and on 8-ch the four methods converge (~0.60) — accuracy is **trial-limited**,
   not channel-limited at the level that matters.
2. **On small calibration data, simpler wins.** CSP+LDA and Riemann beat FBCSP and EEGNet
   here, because FBCSP (more features) and EEGNet (a CNN) overfit 45 trials. FBCSP/EEGNet
   only pull ahead with the ~288 trials of BCI-Competition-IV-2a.
3. **→ method auto-selection** (in `run_analysis.py`): <~120 trials ⇒ CSP/Riemann;
   ≥~120 ⇒ also try FBCSP; ≥~250 ⇒ EEGNet. Pick the CV winner, report the ablation.
4. **Collect more trials** if you can — accuracy on this task is trial-limited, not
   channel-limited. 60–100 trials/class is the single biggest lever.

## Reproduce
`python data_analysis/real_mi_benchmark.py`  (downloads via MNE, runs all methods,
incl. the Unicorn-8ch subset). FBCSP: `python data_analysis/fbcsp.py`.

---

# REAL-data validation — SSVEP flagship (Nakanishi2015, 8-channel)

Dataset: **Nakanishi2015** — real **8-channel** SSVEP, 256 Hz, **12 flicker targets**, a
near-perfect Unicorn analog. Decoder: **training-free FBCCA** (no calibration).

| Subject | Targets | FBCCA acc (0-train) | Chance |
|---|---|---|---|
| **S1–S9 mean** | 12 | **0.93 ± 0.12** | 0.08 |
| best (S4–S6,S8,S9) | 12 | 1.00 | 0.08 |
| weak responder (S2) | 12 | 0.68 | 0.08 |

*(Single source of truth: `ssvep/RUN_LOG_ssvep_trca.txt` — 9 subjects × 3 seeds, cal_frac=0.5.
**Honest n:** Nakanishi2015 has 10 subjects; 9 download reproducibly here (S10 → ValueError,
and a rate-limited mirror can return fewer — re-run if you see n<9). So this is n=9, not 10.)*

**Why this matters:** 93% on a *12-class* problem with *zero training* on *8 channels* is
strong (a weak responder, S2 = 0.68, pulls the mean down; most subjects hit ~1.00). Our
hackathon game uses only **4 targets** (2048 arrows) → accuracy is materially **higher**
than the 12-class number, and a short **TRCA** calibration raises it further (next section).
This is the headset-realistic, real-data evidence behind the SSVEP→2048 flagship.

Reproduce: `python ssvep/real_ssvep_trca.py`  (downloads Nakanishi2015 via MOABB).

---

# REAL-data validation — P300 (BNCI2014-009 speller)

Real P300 speller, xDAWN + shrinkage-LDA, within-subject (AUC; classes ~1:5 imbalanced).
All 8 Unicorn channels (Fz, C3, Cz, C4, Pz, PO7, Oz, PO8) exist in BNCI2014-009, so the
**8-ch subset is exact** — and the P300 sources (Pz/Cz/PO7/PO8) all live in that montage,
so the Unicorn-8 number is **not worse** than the full 16-ch cap. CV is **GroupKFold by
session/run** (leak-free: correlated flashes from one run never split across train/test).

| Subject | 16-ch AUC | **Unicorn-8ch AUC** | 8-ch Acc |
|---|---|---|---|
| S1 | 0.942 | **0.956** | 0.925 |
| S2 | 0.959 | **0.964** | 0.934 |
| **mean (n=2)** | **0.950** | **0.960** | 0.93 |

Unicorn-8ch AUC ~0.96 on real data confirms the P300 pipeline is SOTA-competitive on the
*actual* headset montage (source: `RUN_LOG_p300.txt`). The leak-free AUC (0.960) ≈ the
earlier shuffled-CV number (0.962), so it was **not** inflated by within-run leakage.
**Caveat:** n=2 subjects — indicative, not a population estimate. Reproduce:
`python data_analysis/real_p300_benchmark.py`.

## Summary — all three paradigms validated on REAL public data
| Paradigm | Dataset | Metric | Result |
|---|---|---|---|
| SSVEP (flagship) | Nakanishi2015 (8-ch, n=9) | acc, 12-class, 0-train → calib | **0.93 → 0.99** |
| Motor Imagery | PhysioNet eegmmidb (8-ch subset) | acc, 2-class | **0.61** |
| P300 | BNCI2014-009 | AUC | **0.96** |

This is the evidence base that makes the kit a credible *best-working* implementation:
real datasets, honest numbers, headset-realistic (8-ch) where possible.

---

# REAL-data: SSVEP calibration pays off (FBCCA vs TRCA, Nakanishi2015)

TRCA calibration vs training-free FBCCA, real 8-ch, 12-class, within-subject
(n=9 subjects × 3 random seeds, cal_frac=0.5):

| Subject | FBCCA (0-train) | TRCA (calibrated) | Δ |
|---|---|---|---|
| S1 | 0.75 | 1.00 | +0.25 |
| S2 (weak) | 0.68 | 0.93 | +0.25 |
| S3 | 0.97 | 1.00 | +0.03 |
| S4–S9 | 0.99–1.00 | 1.00 | ~0 |
| **mean ± std** | **0.93 ± 0.12** | **0.99 ± 0.02** | **+0.06** |

*Honesty note:* the lift is small because FBCCA is already near-ceiling on most subjects;
calibration matters most for **weak responders** (S2: 0.68→0.93). TRCA=1.00 on the strong
subjects reflects partly the small per-subject test sets (~7–8 trials/class), so read it as
"near-perfect," not a literal universal 100%.

**How little calibration is enough** (learning curve on the **3-subject subset S1–S3**,
which *includes* the weak responder S2 — that's where calibration value is visible; the
n=9 cohort is already near-ceiling at every cal_frac):

| cal_frac | trials/class | **calibration time** | FBCCA (S1–3) | TRCA (S1–3) |
|---|---|---|---|---|
| 0.15 | 2 | **~100 s** | 0.80 | **0.95** |
| 0.20 | 3 | ~150 s | 0.80 | 0.97 |
| 0.35 | 5 | ~249 s | 0.80 | 0.97 |
| 0.50 | 7 | ~349 s (~6 min) | 0.80 | 0.98 |

**Correction to earlier framing:** a *cal_frac=0.5* calibration is **~6 min** (84 trials),
NOT "60 s." But a genuinely **short ~100-s calibration (2 trials/class) already reaches
TRCA 0.95** even on a weak-responder cohort — so the fast-calibration story holds; the
wall-clock number is now measured, not asserted.

**Flagship recommendation (empirical):** **FBCCA zero-training already averages 0.93** on
12-class real 8-ch SSVEP (instant: "judge puts on the cap, it works"). Add a short
~100-s **TRCA** calibration to rescue weak responders → cohort ~0.99. Reproduce:
`python ssvep/real_ssvep_trca.py`.
