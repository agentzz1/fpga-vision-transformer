# GAPS: implemented vs. needs-a-device / needs-Xcode

Honest status of the SSVEP-2048 iOS app. **It has not been compiled in this
environment** (Linux, no Xcode/Swift toolchain). The code is assembled to be
contract-complete and internally consistent, but the first real build and every
hardware claim must be validated on a Mac / real headset.

## Implemented (and reviewed for consistency, not compiled)

- **Single coherent target.** All Swift lives under one `Sources/` tree compiled by
  one XcodeGen target (`SSVEP2048`). The earlier split second tree
  (`ios/Sources/`) has been consolidated in; the duplicate is gone.
- **All previously-missing contract types now exist** (each defined exactly once):
  - `Core/Constants.swift` — `EEGConfig`, `Direction`.
  - `Decode/Decision.swift` — `FlickerTarget` (with `phase` default 0), `Decision`.
  - `BLE/EEGSource.swift` — `EEGSource` protocol, `EEGConnectionState`.
  - `DSP/Biquad.swift` — `Biquad`, `BiquadDesign` (Butterworth band-pass, RBJ notch).
  - `Decode/ReferenceBank.swift` — `ReferenceBank`, `makeReferenceBank(...)`.
  - `BLE/BLEConstants.swift` — `UnicornBLE`.
  - `UI/ContentView.swift` — root view composing flicker + board + status chrome.
- **Pure 2048 engine** (`Game2048`) — seedable/deterministic, slide/merge/spawn,
  game-over detection. The most testable, hardware-independent piece.
- **Decision gate** (`SSVEPController`) — confidence margin + dwell + refractory,
  pure and event-driven; thresholds now stated consistently (default 1.2).
- **DSP**: CCA via LAPACK QR + SVD (rank-tolerance now tied to `Float.ulpOfOne`);
  FBCCA with Chen sub-band weighting; sub-bands now designed **directly from
  `FBCCAConfig.subBands` corners** (custom sub-bands are honored, not ignored);
  sub-band high corner reconciled to the 80 Hz prefilter corner.
- **`achievableFreqs`** — frame-locked target selection; **no longer emits duplicate
  frequencies** (pads from `EEGConfig.defaultFrequencies`, skipping collisions).
- **Synthetic source** — injects a real multi-harmonic SSVEP + pink/mains noise so
  the whole pipeline runs in the Simulator; single demo setter
  (`setTargetFrequency`) driven by the exact realized target frequency.
- **BLE manager** — full CoreBluetooth state machine (scan → connect → discover →
  notify → START), reconnect/backoff, capability-based characteristic fallback.
- **Frame parser** — resync/reassembly across notifications; endianness is now a
  single named constant and all framing constants are demoted to **VERIFY**.
- **`project.yml`** — iOS 16+, SwiftUI app, `Sources/` source path,
  `NSBluetoothAlwaysUsageDescription`, bundle id `com.ssvep2048.app`. YAML validated.

## Needs Xcode / a Mac to verify

- **It compiles at all.** No Swift compiler was available here. Likely first-build
  items to check: Accelerate/LAPACK symbol availability and `__LAPACK_int` typing in
  `CCA.swift`; `vDSP_biquad` setup path in `Filters.swift`; SwiftUI availability
  attributes; the `ControllerProxy` Combine bridge in `ContentView`.
- **Numerical correctness of the new `BiquadDesign`.** The Butterworth band-pass is
  derived analytically (prototype poles → LP→BP → bilinear) with a center-frequency
  gain normalization. It should be sanity-checked against a reference (e.g. SciPy
  `butter(2,[lo,hi],'bandpass')`) and for stability at the 24–80 Hz / fs=250 band.
- **End-to-end synthetic decode accuracy** — confirm the dwell gate actually commits
  the correct arrow for an injected frequency, and tune `confidenceThreshold` /
  `dwellWindows` against the synthetic ground truth.
- **Refresh handling** — `AppViewModel` defaults to `refresh: 60`; on ProMotion the
  detected refresh (`UIScreen.maximumFramesPerSecond`) should be plumbed into the
  view model so flicker targets and decoder references match the real 120 Hz.

## Needs a real Unicorn headset to verify

- **Every item in the README "BLE protocol VERIFY checklist"** — transport viability
  (Classic vs BLE), UUIDs, START/STOP commands, frame layout, **24-bit endianness**,
  and the µV scale. These are placeholders; live data is untrusted until confirmed
  against BrainFlow `unicorn_board.cpp` + an nRF Connect sniff.
- **Channel order / montage** match to the physical cap.
- **Mains frequency** (50 vs 60 Hz) for the deployment region.
- **Signal-quality / electrode-contact** thresholds (the `signalQuality` heuristic
  and CHECK ELECTRODES trigger are coarse and untuned against real EEG).

## Known smaller gaps / follow-ups

- No automated tests are included; `Game2048` and `SSVEPController` are pure and
  should get XCTest coverage first.
- `battery(_:)` offset/encoding in the parser is assumed (VERIFY) and currently
  unused by the UI.
- The electrode-quality estimate is derived only from decode confidence, not from
  per-channel railing/flatline checks (the Python pipeline's `channel_quality` does
  the latter and could be ported).
