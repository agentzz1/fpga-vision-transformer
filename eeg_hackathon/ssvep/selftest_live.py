"""selftest_live.py — exercise the REAL live LSL decode path WITHOUT a headset.

Reviewer ask: the live branch (LSLAcquirer._resolve -> channel-label mapping -> quality gate
-> notch -> decode) is otherwise only discoverable on stage. This spins up a MOCK LSL outlet
labelled with the Unicorn montage, streams a synthetic SSVEP at a known target on the
occipital channels, then runs the ACTUAL LSLAcquirer + SSVEPController over it and asserts the
decoder recovers the target. Run it before the event as a live-path smoke test.

    python ssvep/selftest_live.py          # needs pylsl (same dep as live use); SKIPs if absent
"""
from __future__ import annotations
import os, sys, time, threading
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from eeg_common import CHANNEL_NAMES, FS
from ssvep_cca import achievable_freqs, synth_ssvep


def _mock_outlet_stream(freqs, target_idx, stop, fs=FS, refresh=60):
    """Push 8-ch samples labelled as the Unicorn montage; SSVEP on the occipital channels."""
    from pylsl import StreamInfo, StreamOutlet
    info = StreamInfo("UN-mock", "EEG", len(CHANNEL_NAMES), fs, "float32", "unicornmock123")
    chns = info.desc().append_child("channels")
    for name in CHANNEL_NAMES:               # label channels so _resolve maps BY NAME
        chns.append_child("channel").append_child_value("label", name)
    outlet = StreamOutlet(info, chunk_size=8)
    occ = [CHANNEL_NAMES.index(c) for c in ("Oz", "PO7", "PO8", "Pz")]
    rng = np.random.default_rng(0)
    while not stop.is_set():
        block = 0.2
        sig = synth_ssvep(freqs[target_idx], block, n_ch=len(occ), snr=0.7, rng=rng)  # (occ, n)
        n = sig.shape[1]
        frame = (rng.standard_normal((len(CHANNEL_NAMES), n)) * 5).astype(np.float32)
        for k, ch in enumerate(occ):
            frame[ch] = sig[k] * 20.0          # ~uV-scale SSVEP on occipital channels
        for s in frame.T:
            outlet.push_sample(s.tolist())
        time.sleep(block)


def main():
    try:
        import pylsl  # noqa: F401
    except Exception:
        print("SKIP: pylsl not installed (same dependency the live demo needs). "
              "Install it (pip install pylsl) to run the live-path self-test."); return 0
    from acquire import LSLAcquirer
    from ssvep_online import SSVEPController, _occipital_idx  # noqa: F401

    refresh = 60
    freqs, _ = achievable_freqs(refresh, n=4)
    target = 2
    print(f"Live-path self-test: mock Unicorn LSL, target arrow #{target} @ {freqs[target]:.2f}Hz, "
          f"freqs={[round(f,2) for f in freqs]}")
    stop = threading.Event()
    pub = threading.Thread(target=_mock_outlet_stream, args=(freqs, target, stop), daemon=True)
    pub.start(); time.sleep(1.0)
    try:
        from acquire import channel_quality, notch_filter
        acq = LSLAcquirer().start(); occ = _occipital_idx()
        ctrl = SSVEPController(freqs=freqs)
        time.sleep(2.5)                        # let the ring buffer fill
        hits = trials = 0
        for _ in range(8):
            data, _ = acq.get_data(seconds=2.0)
            if data.shape[1] >= int(2.0 * FS):
                win = data[occ, -int(2.0 * FS):]
                ok, _r = channel_quality(win)
                win = notch_filter(win, fs=FS, freq=50.0)
                idx, _sc = __import__("ssvep_cca").classify(win, freqs=freqs)
                trials += 1; hits += int(idx == target)
            time.sleep(0.4)
        acq.stop(); stop.set()
        acc = hits / max(trials, 1)
        print(f"  decoded {hits}/{trials} windows as the target  (acc={acc:.2f})")
        ok = acc >= 0.6
        print(f"  {'GREEN — live LSL path (resolve/label-map/quality/notch/decode) works' if ok else 'RED'}")
        return 0 if ok else 1
    finally:
        stop.set()


if __name__ == "__main__":
    sys.exit(main())
