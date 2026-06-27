# REAL-data validation (not synthetic) — PhysioNet Motor Imagery

Task: imagined **left vs right fist** (PhysioNet eegmmidb, runs 4/8/12), within-subject
5-fold CV, 45 trials/subject, **n=8 subjects** (mean ± std). Chance = 0.50. This is the
honest, real-EEG picture — synthetic 100%s mean nothing; these are what the methods deliver.

## Results (accuracy, n=8, mean ± std)
| Channels | CSP+LDA | Riemann-TS | FBCSP | EEGNet |
|---|---|---|---|---|
| 64-ch (lab cap) | 0.65 ± 0.16 | 0.59 ± 0.20 | 0.61 ± 0.15 | 0.53 ± 0.11 |
| **Unicorn 8-ch subset** | **0.64 ± 0.18** | **0.62 ± 0.18** | 0.64 ± 0.18 | 0.51 ± 0.09 |

*(fs=160 Hz; 5-fold CV; FBCSP & EEGNet actually run, not estimated. Source: `RUN_LOG_mi.txt`.
EEGNet now seeds torch + uses deterministic algorithms, but its 0.51 is **indicative**
(seed/BLAS/hardware-sensitive) — it's the worst method here regardless, not a precise headline.)*

Per-subject Unicorn-8ch CSP spans **0.29 → 0.91** (S3=0.29 weak responder vs S7=0.91); the
large std is real between-subject variance — "BCI illiteracy" affects ~10–30% of people.

## Findings that make this the right implementation for the hackathon
1. **8 channels retain ALL of the MI signal here** (8-ch CSP 0.64 ≈ 64-ch 0.65; across all
   four methods 8-ch ties or beats 64-ch). The discriminative signal is sensorimotor
   (C3/Cz/C4), which the Unicorn has, so dropping the other 56 channels costs ~nothing —
   accuracy is **trial-limited**, not channel-limited.
2. **On small calibration data, this EEGNet recipe loses.** EEGNet (45 trials, 50 epochs,
   no early-stopping/augmentation) is worst at 0.51; CSP/Riemann/FBCSP cluster at 0.62–0.64.
   *Scoped claim:* this shows the tested CNN config overfits ~45 trials, not that deep nets
   are categorically worse — a tuned EEGNet with augmentation could close the gap, and EEGNet
   pulls ahead anyway with the ~288 trials of BCI-Competition-IV-2a. With n=8 and ±0.18 std
   the CSP/Riemann/FBCSP differences are **not** statistically separable; treat them as tied.
3. **→ method auto-selection** (in `run_analysis.py`): <~120 trials ⇒ CSP/Riemann;
   ≥~120 ⇒ also try FBCSP; ≥~250 ⇒ EEGNet. Pick the CV winner, report the ablation.
4. **Collect more trials** if you can — accuracy on this task is trial-limited, not
   channel-limited. 60–100 trials/class is the single biggest lever.

## Reproduce
`python data_analysis/real_mi_benchmark.py`  (downloads via MNE, runs all methods,
incl. the Unicorn-8ch subset). FBCSP: `python data_analysis/fbcsp.py`.

---

# REAL-data validation — SSVEP flagship (Nakanishi2015, 8-channel)

Dataset: **Nakanishi2015** — real **8-channel** SSVEP, 256 Hz, **12 flicker targets**.
Decoder: **training-free FBCCA** (no calibration).

*Montage — quantified, not just caveated:* Nakanishi's 8 electrodes are an **occipital
cluster** (PO7,PO3,POz,PO4,PO8,O1,Oz,O2); the Unicorn has only **4 posterior** channels
(Pz,PO7,Oz,PO8). So we ran a **montage ablation** restricting Nakanishi to the 4
Unicorn-posterior-equivalent electrodes (PO7,PO8,Oz,POz≈Pz):

| Montage | FBCCA (0-train, n=9) |
|---|---|
| all-8 occipital (upper bound) | 0.93 ± 0.12 |
| **Unicorn-4-posterior (headset-realistic)** | **0.91 ± 0.16** |

Dropping to the Unicorn's 4 posterior channels costs only **~2 points** — SSVEP lives at
PO/O, which the Unicorn has. **Quote 0.91 as the headset-realistic 12-class number** (source:
`ssvep/RUN_LOG_ssvep_montage.txt`).

*Two honest approximations in this proxy:* (1) Nakanishi has no Pz, so we substitute **POz**
for the Unicorn's **Pz** — POz is slightly more posterior, so it carries marginally *more*
SSVEP, making the proxy a touch **optimistic**; (2) Nakanishi has **zero frontal/central
electrodes**, so the Unicorn's Fz/C3/Cz/C4 (which contribute little SSVEP anyway) are simply
absent — the proxy is occipital-only. Net: read 0.91 as a mild upper bound, not a floor. Remaining honest gap: this is still Nakanishi-recorded,
not Unicorn-recorded data — we have **no real Unicorn SSVEP recording** (the one gap to the
live headset, and a reason FBCCA/SSVEP, the spectral paradigm, was chosen — it transfers best).

| Subject | Targets | FBCCA acc (0-train) | Chance |
|---|---|---|---|
| **S1–S9 mean** | 12 | **0.93 ± 0.12** | 0.08 |
| best (S4–S6,S8,S9) | 12 | 1.00 | 0.08 |
| weak responder (S2) | 12 | 0.68 | 0.08 |

*(Single source of truth: `ssvep/RUN_LOG_ssvep_trca.txt` — 9 subjects × 3 seeds, cal_frac=0.5.
**Honest n:** Nakanishi2015 has 10 subjects; 9 download reproducibly here (S10 → ValueError,
and a rate-limited mirror can return fewer — re-run if you see n<9). So this is n=9, not 10.)*

**Distribution (n=9, not just the mean):** FBCCA per-subject = {0.68, 0.75, 0.97, 0.99, 1.00,
1.00, 1.00, 1.00, 1.00} → **median 1.00, range 0.68–1.00**. The mean (0.93) is dragged by one
weak responder (S2); 7 of 9 subjects are ≥0.97. Per-subject test sets are ~7–8 trials/class,
so the 1.00s are near-ceiling, not infinitely precise — read the spread, not just the mean.

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

| Subject | 16-ch AUC | **Unicorn-8ch AUC** |
|---|---|---|
| S1 | 0.942 | **0.956** |
| S2 | 0.959 | **0.964** |
| S3 (lower) | 0.844 | **0.833** |
| S4 | 0.964 | **0.959** |
| S5 | 0.972 | **0.975** |
| **mean ± std (n=5)** | **0.936** | **0.937 ± 0.052** |

Unicorn-8ch AUC **0.94 ± 0.05** (n=5) confirms the P300 pipeline is SOTA-competitive on the
*actual* headset montage (source: `RUN_LOG_p300.txt`); 8-ch ≈ 16-ch because the P300 sources
(Pz/Cz/PO7/PO8) all live in the Unicorn montage. S3 is a lower responder (0.83), which the
±0.05 std reflects. Reproduce: `python data_analysis/real_p300_benchmark.py`.

## Summary — all paradigms validated on REAL public data
| Paradigm | Dataset | Hardware | Metric | Result |
|---|---|---|---|---|
| SSVEP (flagship) | Nakanishi2015 (8-ch, n=9) | public, gel occipital cap — **NOT Unicorn** | acc, 12-class, 0-train → calib | **0.93 → 0.99** |
| SSVEP (**live regime**) | Nakanishi2015 (n=9) | public, **4 Unicorn-posterior ch proxy** — NOT Unicorn | acc, 4-class, 2-s, 0-train | **0.90 ± 0.16** |
| Motor Imagery | PhysioNet eegmmidb (8-ch, n=8) | public, gel 64-cap subset — NOT Unicorn | acc, 2-class | **0.64 ± 0.18** |
| P300 | BNCI2014-009 (8-ch, n=5) | public, gel 16-cap subset — NOT Unicorn | AUC, leak-free | **0.94 ± 0.05** |

**Live-regime optimism note:** the 4 targets in the live-regime row are picked **well-separated**
across the band (the favorable case). Adjacent flicker frequencies are more confusable, so 0.90
is a mild upper bound for an arbitrary 4-arrow layout. On a **60 Hz** screen the achievable set
[15,10,7.5,6] has a harmonic collision (15 = 2×7.5) — the app warns and recommends a **120 Hz**
display (clean set 15/12/10/8.57); see `achievable_freqs` / `harmonic_collisions`.

**Honest hardware caveat:** *every* row is a public dataset with a **channel-subset proxy** for
the Unicorn montage — **none is Unicorn-recorded**. The dominant real-world SSVEP penalty
(dry electrodes + real monitor flicker) is therefore **unmodeled**; treat these as upper
bounds. This is the single gap to the live headset (and why SSVEP, the most transferable
spectral paradigm, is the flagship). Drop a real `.npz` from the headset and the benchmark
hooks fold it in.

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
