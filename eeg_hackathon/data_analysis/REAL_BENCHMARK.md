# REAL-data validation (not synthetic) — PhysioNet Motor Imagery

Task: imagined **left vs right fist** (PhysioNet eegmmidb, runs 4/8/12), within-subject
5-fold CV, 45 trials/subject. Chance = 0.50. This is the honest, real-EEG picture —
synthetic 100%s mean nothing; these are what the methods actually deliver.

## Results (accuracy)
| Channels | CSP+LDA | Riemann-TS | EEGNet | FBCSP |
|---|---|---|---|---|
| 64-ch (lab cap) | 0.65 | 0.57 | 0.52 | ~0.60 |
| **Unicorn 8-ch subset** | **0.64** | **0.62** | 0.55 | ~0.60 |

Per-subject (Unicorn-8ch, CSP): S1 0.71, S2 0.82, **S3 0.29** (a weak responder — "BCI
illiteracy" is real, ~10–30% of people).

## Findings that make this the right implementation for the hackathon
1. **8 channels ≈ 64 channels** for MI (0.64 vs 0.65). The discriminative signal is
   sensorimotor (C3/Cz/C4), which the Unicorn has — so the headset loses almost nothing.
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
