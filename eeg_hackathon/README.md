# Unicorn EEG → Keystroke Pipeline (Zeiss "EEG Mind Control" hackathon)

Live + offline pipeline to **detect/classify keystrokes from g.tec Unicorn EEG**
(8 ch @ 250 Hz) and drive a game (Canabalt spacebar / 2048 arrows).
Acquisition (LSL / CSV / synthetic) → MNE/SciPy preprocessing → features
(band-power + ERP + optional tsfresh) → classifier (LDA/RF/XGB) → realtime predict.

## 0. No hardware? Prove it works now
```bash
pip install numpy scipy scikit-learn pandas xgboost   # core (already enough for the demo)
python run_demo.py        # synthetic EEG with injected motor/P300 ERPs -> CV acc ~0.72 / AUC ~0.79
```

## 1. Install (full, on the hackathon machine)
```bash
conda create -n eeg python=3.11 -y && conda activate eeg
pip install -r requirements.txt          # adds mne, pylsl, tsfresh, pynput
```

## 2. Stream EEG from the Unicorn
1. Unicorn Suite → pair the headset (Bluetooth) → start **"Unicorn LSL"** (streams 17 ch; EEG = first 8).
2. Sanity check the stream is seen: `python -c "import acquire,eeg_common; a=acquire.LSLAcquirer().start(); import time; time.sleep(3); print(a.get_data(2)[0].shape)"`
   *(BrainFlow is an alternative driver if LSL is flaky — the Unicorn board is supported there too.)*

## 3. Record a labeled session (play Canabalt while logging spacebars)
```python
from acquire import LSLAcquirer, KeyLogger
from pylsl import local_clock
acq = LSLAcquirer().start()
log = KeyLogger(clock=local_clock).start()    # same clock as LSL timestamps == alignment
# ... play Canabalt for a few minutes ...
acq.stop(); log.stop()
import numpy as np; data, ts = acq.get_data(seconds=99999); np.save("eeg.npy", data)
log.save("presses.txt")
```
(Or use the Unicorn Recorder → CSV and `CSVAcquirer`.)

## 4. Train
```bash
python train.py --csv eeg.csv --presses presses.txt --features all --model rf
# -> checkpoints/model.joblib + cv_report.json
```

## 5. Predict live (replace the keystroke)
```bash
python predict.py --realtime          # prints SPACE when press intention is detected
```
Wire the detection to a key event (e.g. `pynput.keyboard.Controller().press(' ')`)
to actually control the game — see `utils` hook in predict.RealtimePredictor.

## Data contract (every module conforms)
- raw `data` (8, n_samples) float32 µV, `timestamps` (n_samples,) float64 s
- `events` (n_events, 2) = [sample_index, label]  (1=space, 0=rest)
- `epochs` X (n_epochs, 8, 251) float32, `y` (n_epochs,)
- channels (fixed order): Fz, C3, Cz, C4, Pz, PO7, Oz, PO8 — bandpass 0.1–50 Hz, notch 50 Hz, epoch −0.2…+0.8 s.

## Files
`eeg_common.py` (constants/contract) · `acquire.py` (LSL/CSV/synthetic + keylogger) ·
`preprocess.py` (filter+epoch, MNE or SciPy) · `features.py` (band-power/ERP/tsfresh) ·
`model.py` (classifiers+CV) · `train.py` · `predict.py` · `run_demo.py`.

## Hackathon tips (the informative signal)
- Best channels: **C3/Cz/C4** (motor / readiness potential) + **Pz/Oz** (visual response).
- The discriminative signal is the **movement-related cortical potential** (starts ~0.5–1 s
  *before* the press) + the post-stimulus response. Try epochs starting earlier (−1.0 s) to
  exploit pre-movement signal for *anticipatory* control.
- Strong baselines for motor EEG: **CSP + LDA**, **Riemannian/covariance (pyriemann)**, or a
  small **EEGNet** CNN. Band-power+RF here is the quick baseline; CSP/Riemannian usually beats it.
- 2048 (4 classes) is harder than Canabalt (1 binary press) — start with Canabalt.
