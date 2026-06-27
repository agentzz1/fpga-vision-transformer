# SSVEP 2048 (iOS)

Play **2048 with your eyes.** Four arrow tiles flicker at distinct, refresh-locked
frequencies around the board. When you gaze at one, your visual cortex produces a
**steady-state visual evoked potential (SSVEP)** at that flicker frequency. The app
reads 8-channel EEG from a **g.tec Unicorn Hybrid Black**, decodes which arrow you
are looking at with **filter-bank CCA (FBCCA, Chen et al. 2015)**, and — after a
short dwell — makes that 2048 move.

It runs **fully headset-free in the iOS Simulator** using a synthetic EEG source
that injects a real SSVEP at the target frequency, so the entire pipeline
(samples → ring buffer → pre-filter → FBCCA → dwell gate → game) is exercisable
without hardware.

> ⚠️ **Honesty note.** This project has **not been compiled** in the environment it
> was assembled in (Linux, no Xcode/Swift toolchain). It is written to be coherent
> and contract-complete, but the first real build happens on your Mac. The BLE
> protocol constants are **placeholders that must be verified on a real device** —
> see the [BLE protocol VERIFY checklist](#ble-protocol-verify-checklist) and
> [`GAPS.md`](GAPS.md).

---

## Architecture

Single Xcode target **`SSVEP2048`**, all Swift under `Sources/`:

```
Sources/
  App/      SSVEP2048App.swift      @main entry point
            AppViewModel.swift      @MainActor pipeline coordinator
  BLE/      EEGSource.swift         protocol + EEGConnectionState
            BLEConstants.swift      UnicornBLE (UUIDs/commands/frame geometry — VERIFY)
            UnicornBLEManager.swift CoreBluetooth central (live source)
            UnicornPacketParser.swift  45-byte frame → 8×µV (VERIFY)
            SyntheticEEGSource.swift   headset-free synthetic SSVEP source
  Core/     Constants.swift         EEGConfig + Direction
            EEGRingBuffer.swift     thread-safe sliding-window store
  DSP/      Biquad.swift            Biquad + BiquadDesign (Butterworth BP, notch)
            Filters.swift           prefilter + sub-band cascades (vDSP)
            FBCCA.swift             filter-bank CCA decoder + achievableFreqs()
  Decode/   CCA.swift               canonical correlation (LAPACK QR + SVD)
            ReferenceBank.swift     sin/cos harmonic references
            Decision.swift          FlickerTarget + Decision
  Game/     Game2048.swift          pure 2048 engine (seedable)
            SSVEPController.swift    decision → move dwell/refractory gate
  UI/       ContentView.swift       root screen (composes the below)
            FlickerView.swift       CADisplayLink refresh-locked flicker tiles
            Game2048View.swift      board + status chrome
```

Pipeline (see `AppViewModel`): `EEGSource.samples()` → `EEGRingBuffer` → every
~0.5 s pull the latest 2 s occipital window → per-channel 6–80 Hz Butterworth +
mains notch prefilter → `FBCCA.classify` → `Decision` → `SSVEPController` dwell gate
→ one 2048 move.

---

## Build

Requires a **Mac with Xcode 15+** and [XcodeGen](https://github.com/yonyz/XcodeGen).

```sh
brew install xcodegen
cd ios/UnicornSSVEP
xcodegen generate
open *.xcodeproj
```

`xcodegen generate` reads `project.yml` and produces `SSVEP2048.xcodeproj` with one
iOS app target (iOS 16+, SwiftUI), bundle id `com.ssvep2048.app`, and an Info.plist
containing `NSBluetoothAlwaysUsageDescription`.

For **device** builds, set your signing team: open the project and pick a team under
*Signing & Capabilities*, or set `DEVELOPMENT_TEAM` in `project.yml` and regenerate.

---

## Run: synthetic (Simulator, no headset)

1. In Xcode pick an **iOS Simulator** (e.g. iPhone 15) and Run (⌘R).
2. The Simulator has no Bluetooth radio, so the app **automatically** uses
   `SyntheticEEGSource` (you'll see the **"SYNTHETIC — NO HEADSET"** watermark).
3. The synthetic source injects an SSVEP at the first target's frequency by
   default, so the decoder should converge and the board should start making moves
   after the dwell period (~0.75 s of sustained high-confidence windows).
4. To drive a specific arrow programmatically in a demo, call
   `AppViewModel.setSyntheticGaze(_ arrowIndex:)` (0=up, 1=down, 2=left, 3=right);
   it maps the index to the **exact realized** flicker frequency the decoder uses.

> Note: the Simulator renders flicker at 60 Hz; SSVEP discrimination is best on a
> real ProMotion (120 Hz) device, but the synthetic path validates correctness end
> to end regardless.

---

## Run: live (real Unicorn headset)

> ⚠️ Live BLE is **unverified** — read the checklist below first. The stock Unicorn
> streams over **Bluetooth Classic (RFCOMM/SPP)**, which iOS CoreBluetooth **cannot
> use**. Live mode will only work if your unit exposes a **BLE/GATT** data path (or
> a bridge does), and only after the constants in `BLEConstants.swift` are corrected.

1. Run on a **physical iOS device** (not the Simulator).
2. Power on and wear the Unicorn; ensure good electrode contact (esp. PO7/Oz/PO8).
3. Launch the app and accept the **Bluetooth permission** prompt.
4. The app scans for peripherals advertising the `UN-` name prefix, connects,
   discovers the notify/command characteristics, subscribes, and writes START.
5. The status bar shows Disconnected → Connecting → **Streaming**. Once streaming,
   gaze at an arrow and hold ~1 s to commit a move.
6. If a **CHECK ELECTRODES** banner appears, decode quality is low — re-seat the
   occipital electrodes / re-gel.

---

## BLE protocol VERIFY checklist

**Every constant below is an unverified placeholder.** A real device run **must**
confirm each one against (a) BrainFlow's `unicorn_board.cpp`
(<https://github.com/brainflow-dev/brainflow> → `src/board_controller/unicorn/`)
and (b) a live **nRF Connect** sniff of the headset. Until then, treat live data as
untrusted. All constants are centralized in `Sources/BLE/BLEConstants.swift` so each
fix is a one-line change.

### Transport (the big one)
- [ ] **Does the unit even expose BLE/GATT?** The stock Unicorn uses Bluetooth
      Classic RFCOMM/SPP (BrainFlow opens a serial/COM port, not GATT).
      CoreBluetooth cannot speak RFCOMM. Confirm a GATT data characteristic exists
      (nRF Connect) or that you have a BLE bridge. If not, this app's live path is
      not viable as-is.

### Advertising / identification
- [ ] `UnicornBLE.advertisedNamePrefix` (`"UN-"`) — exact advertised local-name
      prefix and casing (nRF Connect → device name).

### GATT UUIDs (currently Nordic UART placeholders — almost certainly wrong)
- [ ] `UnicornBLE.serviceUUID` — the primary data/control service UUID.
- [ ] `UnicornBLE.notifyCharUUID` — characteristic carrying EEG frames; confirm it
      has the **Notify** property.
- [ ] `UnicornBLE.commandCharUUID` — characteristic for START/STOP; confirm
      **Write** / **Write Without Response**.
      (The manager also falls back to capability matching, but verify anyway.)

### Commands
- [ ] `UnicornBLE.startCommand` — exact begin-acquisition byte sequence
      (BrainFlow `unicorn_board.cpp`).
- [ ] `UnicornBLE.stopCommand` — exact stop-acquisition byte sequence.

### Frame geometry (currently assumed 45-byte frame)
- [ ] `UnicornBLE.frameLength` (45) — actual notification/frame length.
- [ ] `UnicornBLE.header` (`0xC0 0x00`) — sync/header bytes, if any.
- [ ] `UnicornBLE.footer` (`0x0D 0x0A`) — trailer bytes, if any.
- [ ] `UnicornBLE.eegByteOffset` (2) — first EEG byte. Watch for a leading
      status/battery byte that shifts EEG to offset 3.
- [ ] EEG block size: `eegChannelCount` (8) × `bytesPerChannel` (3) = 24 bytes,
      contiguous and non-overlapping with the accel/gyro/counter trailer.

### Sample encoding (decode-killer if wrong)
- [ ] **`UnicornBLE.eegLittleEndian` (currently `true`)** — the 24-bit signed
      sample byte order. Unicorn is *very likely* little-endian (BrainFlow decodes
      LE), but it is **not confirmed for a GATT transport**. If wrong, every sample
      is byte-swapped garbage (sign bit in the wrong byte) and SSVEP decoding fails.
      Confirm against a known-amplitude signal, then flip this one constant.
- [ ] `UnicornBLE.microvoltsPerCount` (`4500000/50331642 ≈ 0.0894073 µV/LSB`) —
      the count→µV scale. Confirm against the SDK/datasheet or a calibration.
- [ ] 24-bit two's-complement sign extension (sign bit `0x00800000`).

### Trailer (unused by decode today, but documented)
- [ ] accel `27..32` (3×int16), gyro `33..38` (3×int16), counter `39..42` (uint32),
      battery/status byte at `26`. Confirm offsets/encodings before wiring these up.

### Acquisition parameters
- [ ] `EEGConfig.fs` (250 Hz) and `EEGConfig.channelNames`
      (`Fz,C3,Cz,C4,Pz,PO7,Oz,PO8`) — confirm rate and channel order/montage match
      your cap (these match the project's Python `eeg_common.py`).
- [ ] `EEGConfig.mainsHz` — 50 Hz (EU/UK) vs **60 Hz** (North America).
