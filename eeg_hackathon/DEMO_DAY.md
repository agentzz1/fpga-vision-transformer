# DEMO DAY — zero-to-prize runbook (Unicorn Hybrid Black / BR41N.IO + Zeiss)

You can win in **3 categories** ($600/$400/$200 each). This kit covers all three.
Strategy: **Gaming = SSVEP→2048** (reliable wow), **Data Analysis = our SOTA pipelines**
(no-hardware, highest-probability autonomous win), **Programming/on-brief = Canabalt focus**.

## 0. Install (2 min)
```bash
cd eeg_hackathon && pip install -r requirements.txt
python run.py 7     # motor pipeline demo — proves the kit runs (no hardware)
```

## 1. GAMING — SSVEP → 2048 (your flagship)
Why it wins: Unicorn's SPECTRAL data is reliable (Pontifex & Coffman 2023, r≈0.84),
so frequency-tagged SSVEP is the robust paradigm; 4 arrows ↔ 4 frequencies; zero training.
1. Unicorn Suite → start **Unicorn LSL**. Check Oz/PO7/PO8/Pz are clean.
2. `python run.py 4` → confirm the 4 freqs (8.57/10/12/15 Hz) separate on your screen.
   If a freq is weak, set `FREQS` in `ssvep/ssvep_cca.py` to `refresh/integer` values.
3. **Do the 60-s TRCA calibration** (`python run.py 5`): on real 8-ch SSVEP it lifts
   accuracy 0.83→0.98 (12-class). Use TRCA live; FBCCA is the zero-training fallback.
4. `python run.py 2` (LIVE) → play 2048 by looking at arrows. (No headset: `python run.py 1`.)
Realistic: ~85–95% selection at 2–3 s windows. Record a clean run as backup video.

## 2. DATA ANALYSIS — our SOTA pipelines (no hardware; best autonomous odds)
Load the provided .mat into a `(trials, channels, samples)` array + labels, then:
- **Motor Imagery** (`data_analysis/mi_pipeline.py`): `evaluate(X, y)` → CSP+LDA. (synth: 100%)
- **P300 speller** (`data_analysis/p300_pipeline.py`): `evaluate(epochs, y)` → xDAWN+shrinkLDA, AUC. (synth AUC 0.99)
- **SSVEP** (`ssvep/`): FBCCA / TRCA decoder.
Pitch: "we benchmark CSP/xDAWN/CCA against SOTA, with cross-validated AUC and an honest
ablation." `load_gtec_mat()` is a starting loader; adapt the trigger parsing to the file.

## 3. CANABALT (on-brief: detect a keystroke) — band-power FOCUS trigger
Why this not readiness-potential: UHB ERPs are only moderate (P300 r≈0.55) but band-power
is reliable; Natalizio 2024 hit ~94.6% focus/rest live on the Unicorn during Tetris.
`python run.py 3` → train; `python canabalt/canabalt_focus.py --realtime` → concentrate to jump.

## Fallback ladder (so a demo NEVER dies on stage)
1. SSVEP→2048 live. 2. If EEG noisy: TRCA calibration / widen window to 3 s.
3. If still rough: the band-power focus trigger (very robust). 4. Last resort: the
synthetic showcase (`python run.py 1`) — visibly the same app, explained honestly.

## 90-second judge pitch
"Consumer 8-ch EEG, zero-training SSVEP → real-time game control. We chose SSVEP because
the Unicorn's spectral signal is reliable where its ERPs aren't (cite Pontifex 2023), got
~Nx% online, and add an optional TRCA calibration (the algorithm behind the highest-ITR
EEG BCI). Same toolkit also delivers SOTA P300 & motor-imagery analysis."
