# SSVEP → 2048 : the win-probability play

**Goal:** play 2048 hands-free by *looking* at flickering arrows. SSVEP + FBCCA,
**zero training**, ~90%+ with 2–3 s windows. This is the reliable centerpiece demo.

## Why this wins
- 2048's 4 actions ↔ 4 flicker frequencies (8.57/10/12/15 Hz, exact on a 60 Hz screen).
- CCA needs **no training data** → works the instant a judge puts on the cap.
- Occipital channels (Oz/PO7/PO8/Pz) carry SSVEP strongly on the Unicorn.
- Verified core: 100% on synthetic @1 s; online decision logic 24/24 @2 s.

## Files
- `ssvep_cca.py`   — FBCCA decoder + references + synthetic validator + ITR.
- `ssvep_stim.py`  — pygame flickering arrows (run next to the game).
- `ssvep_online.py`— LSL → occipital window → FBCCA → confidence/dwell gate → arrow keypress.

## Tonight (no hardware) — already passing
```bash
python ssvep_cca.py          # acc/ITR across window lengths on synthetic SSVEP
python ssvep_online.py --simulate   # full decision-logic check
```

## Demo day (with the Unicorn)
1. `pip install pygame pylsl pynput scikit-learn scipy numpy`
2. Unicorn Suite → start **Unicorn LSL**. Confirm occipital channels are clean
   (Oz/PO7/PO8/Pz); reduce lighting glare; ask the user to fixate the target arrow.
3. Terminal A: `python ssvep_stim.py`  (the flickering arrows)
4. Open 2048 (web or app) and click it so it has keyboard focus.
5. Terminal B: `python ssvep_online.py`  → it presses ↑↓←→ when a target is locked.
6. Tune in `ssvep_online.py`: `window_s` (2.0→3.0 if noisy), `margin`, `dwell`
   (raise to cut false triggers), `refractory_s`.

## Tuning knobs that move accuracy
- **Window length** is the main lever: 1 s fast/risky, 3 s slow/robust. Start 2.5 s.
- **FBCCA** on (default) > plain CCA. Add a per-user calibration (TRCA) only if time.
- Pick frequencies that are exact monitor sub-harmonics; avoid pairs in 2:1 ratio
  (harmonic confusion) — 8.57/10/12/15 are safe on 60 Hz.

## If SSVEP underperforms on the day (safety net)
- Jaw-clench / blink burst on a frontal channel = a rock-solid binary trigger
  (great for *Canabalt* jump). Detect via high-band power threshold on Fz.
- Or motor readiness-potential "jump" detector from the main pipeline (`../`)
  for the on-brief innovation story.
