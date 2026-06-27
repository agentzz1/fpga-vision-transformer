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
Why it wins: Pontifex & Coffman (2023) found the Unicorn gives **valid spectral measures
even without gel**, while its ERPs need conductive solution — so frequency-tagged SSVEP is
the robust paradigm; 4 arrows ↔ 4 frequencies; zero training.
1. Unicorn Suite → start **Unicorn LSL**.
2. `python run.py 4` (**Setup: stimulus + freqs**, = `ssvep/ssvep_stim.py`) → shows the
   flickering arrows and **prints the 4 refresh-locked frequencies** (60 Hz → 15/10/7.5/6;
   120 Hz → 15/12/10/8.57). Same values drive flicker AND decoder, so they can't diverge.
3. **Optional short TRCA calibration** (`python run.py 5`): zero-training FBCCA already
   averages **0.93** on real 8-ch 12-class SSVEP (n=9); a ~100-s calibration rescues weak
   responders → ~0.99. Use TRCA live for weak responders; FBCCA is the instant fallback.
4. `python run.py 2` (**LIVE game**, = `app/ssvep_2048_app.py`) → play 2048 by looking at
   arrows. The **electrode-quality enforcement lives HERE in the game** (not the step-2 stim):
   it maps channels by LSL label, applies a 50/60 Hz notch (`--notch 60` US), prints the
   resolved channel map at startup, shows a red **"CHECK ELECTRODES: PO7…"** banner + refuses
   to decode on a flat/railed channel, and warns on refresh/frame-drop mismatch. (No headset:
   `python run.py 1` → clearly watermarked **SYNTHETIC — NO HEADSET**.)
Expectation (MEASURED, not hand-waved): at the **live regime — 2 s window, 4 targets** —
FBCCA scores **0.90 ± 0.16** (n=9, `RUN_LOG_ssvep_live_regime.txt`) — this stacks ALL three
real penalties: 4 targets, 2 s window, AND the Unicorn's 4 posterior channels. The remaining
unmodeled gap is dry electrodes + real monitor flicker, so treat **~0.90 as the live upper bound**. Record a
clean run as a backup video.

## 2. DATA ANALYSIS — our SOTA pipelines (no hardware; best autonomous odds)
Load the provided .mat into a `(trials, channels, samples)` array + labels, then:
- **Motor Imagery** (`data_analysis/mi_pipeline.py`): `evaluate(X, y)` → CSP+LDA. (synth: 100%)
- **P300 speller** (`data_analysis/p300_pipeline.py`): `evaluate(epochs, y)` → xDAWN+shrinkLDA, AUC. (synth AUC 0.99)
- **SSVEP** (`ssvep/`): FBCCA / TRCA decoder.
Pitch: "we benchmark CSP/xDAWN/CCA against SOTA, with cross-validated AUC and an honest
ablation." `load_gtec_mat()` is a starting loader; adapt the trigger parsing to the file.

## 3. CANABALT (on-brief: detect a keystroke) — band-power FOCUS trigger
Why this not readiness-potential: on the Unicorn, band-power is more reliable than ERPs
(Pontifex & Coffman 2023); Natalizio et al. 2024 reported high focus/engagement
classification live on the Unicorn during Tetris.
`python run.py 3` → train; `python canabalt/canabalt_focus.py --realtime` → concentrate to jump.

## Fallback ladder (so a demo NEVER dies on stage)
1. SSVEP→2048 live. 2. If EEG noisy: TRCA calibration / widen window to 3 s.
3. If still rough: the band-power focus trigger (very robust). 4. Last resort: the
synthetic showcase (`python run.py 1`) — visibly the same app, explained honestly.

## 90-second judge pitch
"Consumer 8-ch EEG, zero-training SSVEP → real-time game control. We chose SSVEP because
the Unicorn's spectral signal is reliable where its ERPs aren't (cite Pontifex 2023), got
FBCCA 0.93 offline on real 8-ch data, and add an optional TRCA calibration (the algorithm behind the highest-ITR
EEG BCI). Same toolkit also delivers SOTA P300 & motor-imagery analysis."
