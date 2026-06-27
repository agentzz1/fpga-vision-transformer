//
//  SyntheticEEGSource.swift
//  SSVEP2048
//
//  An `EEGSource` that SYNTHESIZES EEG-like data containing a steady-state
//  visual evoked potential (SSVEP) at a chosen target frequency. This lets the
//  whole app run end-to-end in the iOS Simulator (or on device) with NO headset
//  attached: the pipeline (samples → ring buffer → pre-filter → FBCCA → game)
//  receives realistic length-8 µV samples at 250 Hz exactly as it would from
//  `UnicornBLEManager`.
//
//  Signal model per channel (units µV):
//    - 1/f-ish background EEG noise (pink-ish) + white sensor noise.
//    - On occipital channels (`EEGConfig.occipitalIndices`), an additional SSVEP
//      = fundamental sin + two harmonics (3 sinusoids total) at the *target*
//      frequency, with mild per-harmonic amplitude falloff and per-channel phase
//      jitter so it is not a perfectly clean tone.
//    - A 50/60 Hz mains hum component so the downstream notch filter has work to
//      do (region taken from `EEGConfig.mainsHz`).
//
//  Conforms to the SAME `EEGSource` protocol as the BLE manager (see
//  BLE/EEGSource.swift), so `AppViewModel` can swap sources transparently.
//
//  NOTE ON FILE LOCATION: The interface contract (§A / §B.4) lists this type
//  under `Sources/BLE/`. It is written here under `Sources/Core/` per the build
//  task's explicit output path. The Xcode target compiles everything under
//  `Sources/`, so either location is on the compile path; the public API is
//  unchanged. // TODO(contract): reconcile folder (BLE/ vs Core/).
//

import Foundation

/// `EEGSource` implementation that generates synthetic SSVEP data for
/// headset-free development and demos.
///
/// Threading model (mirrors the BLE manager, per contract §D.5):
///   - A private serial queue (`genQueue`) owns all mutable state and drives a
///     repeating timer that yields one sample per 250 Hz tick.
///   - `samples()` / `states()` expose `AsyncStream`s fed from that queue.
///   - The class holds no UI state and never touches the main actor itself.
final class SyntheticEEGSource: EEGSource {

    // MARK: - EEGSource conformance (read-only surface)

    /// Channel order matches `EEGConfig.channelNames` (Fz,C3,Cz,C4,Pz,PO7,Oz,PO8).
    let channelNames: [String] = EEGConfig.channelNames

    /// Fixed 250 Hz acquisition rate, identical to the real Unicorn.
    let sampleRate: Double = EEGConfig.fs

    /// Current connection state. Mutated only on `genQueue`; cross-thread reads
    /// observe a consistent (possibly slightly stale) snapshot.
    var connectionState: EEGConnectionState {
        genQueue.sync { _connectionState }
    }

    // MARK: - Configuration

    /// The frequency (Hz) currently embedded in the occipital channels. `nil`
    /// means "no SSVEP" — pure noise (useful for testing the no-decision path).
    /// Mutated only on `genQueue`.
    private var targetFrequency: Double?

    /// Amplitude (µV) of the SSVEP fundamental on occipital channels.
    private let ssvepAmplitude: Double = 6.0

    /// Amplitude (µV) of the broadband background EEG.
    private let backgroundAmplitude: Double = 12.0

    /// Amplitude (µV) of the mains hum component.
    private let mainsAmplitude: Double = 3.0

    /// Number of SSVEP sinusoids: fundamental + 2 harmonics.
    private let harmonicCount: Int = 3

    /// injectFrequency: optional Hz to embed in occipital channels for testing.
    /// Pass `nil` to start with pure noise; a target can be chosen later via
    /// `setTarget(_:)` / `setTargetFrequency(_:)`.
    init(injectFrequency: Double? = nil) {
        self.targetFrequency = injectFrequency
    }

    // MARK: - Generation state (touched only on genQueue)

    /// Dedicated serial queue. The generation timer fires here and all mutable
    /// state below is read/written only here.
    private let genQueue = DispatchQueue(label: "com.ssvep2048.synthetic",
                                         qos: .userInitiated)

    /// Repeating timer that emits one sample per 250 Hz tick.
    private var timer: DispatchSourceTimer?

    /// Backing store for `connectionState`; mutated only on `genQueue`.
    private var _connectionState: EEGConnectionState = .disconnected {
        didSet {
            guard oldValue != _connectionState else { return }
            statesContinuation?.yield(_connectionState)
        }
    }

    /// Sample index since `start()`, used to compute the continuous phase of all
    /// deterministic (SSVEP, mains) oscillators. Wraps are avoided by using
    /// fractional-second time; with Double this stays exact for hours.
    private var sampleIndex: UInt64 = 0

    /// Per-channel random phase offsets for the SSVEP, so channels are not
    /// phase-identical. Index 0..7. Regenerated on `start()`.
    private var channelPhase: [Double] = []

    /// Pink-noise generator state per channel (Voss-McCartney style accumulators).
    private var pinkState: [PinkNoise] = []

    /// Deterministic RNG so demos are reproducible within a run.
    private var rng = SystemRandomNumberGenerator()

    // MARK: - AsyncStream plumbing

    private var samplesContinuation: AsyncStream<[Double]>.Continuation?
    private var statesContinuation: AsyncStream<EEGConnectionState>.Continuation?

    /// Continuous stream of length-8 µV samples, one element per 250 Hz scan.
    /// A new subscription replaces any prior continuation.
    func samples() -> AsyncStream<[Double]> {
        AsyncStream(bufferingPolicy: .unbounded) { continuation in
            self.genQueue.async {
                // Replace any previous subscriber.
                self.samplesContinuation?.finish()
                self.samplesContinuation = continuation
                continuation.onTermination = { [weak self] _ in
                    self?.genQueue.async {
                        self?.samplesContinuation = nil
                    }
                }
            }
        }
    }

    /// State changes, with the initial value emitted on subscribe.
    func states() -> AsyncStream<EEGConnectionState> {
        AsyncStream(bufferingPolicy: .bufferingNewest(8)) { continuation in
            self.genQueue.async {
                self.statesContinuation?.finish()
                self.statesContinuation = continuation
                // Emit current state immediately on subscribe (contract B.3).
                continuation.yield(self._connectionState)
                continuation.onTermination = { [weak self] _ in
                    self?.genQueue.async {
                        self?.statesContinuation = nil
                    }
                }
            }
        }
    }

    // MARK: - EEGSource control

    /// Begin generating samples. Transitions disconnected → connecting →
    /// streaming (the brief `connecting` step mimics the BLE handshake so the UI
    /// behaves identically across sources).
    func start() {
        genQueue.async {
            guard self.timer == nil else { return }

            self._connectionState = .connecting

            // Fresh per-run randomness so each demo run is slightly different.
            self.sampleIndex = 0
            self.channelPhase = (0..<EEGConfig.channelCount).map { _ in
                Double.random(in: 0..<(2 * .pi), using: &self.rng)
            }
            self.pinkState = (0..<EEGConfig.channelCount).map { _ in
                PinkNoise(seed: UInt64.random(in: .min ... .max, using: &self.rng))
            }

            // Create the 250 Hz emission timer.
            let interval = 1.0 / EEGConfig.fs
            let timer = DispatchSource.makeTimerSource(queue: self.genQueue)
            timer.schedule(deadline: .now() + interval,
                           repeating: interval,
                           leeway: .milliseconds(1))
            timer.setEventHandler { [weak self] in
                self?.emitSample()
            }
            self.timer = timer

            // Simulate a short connect delay, then go live.
            self.genQueue.asyncAfter(deadline: .now() + 0.2) {
                guard self.timer != nil else { return }
                self._connectionState = .streaming
                timer.resume()
            }
        }
    }

    /// Stop generation and tear down the stream. Idempotent.
    func stop() {
        genQueue.async {
            self.timer?.cancel()
            self.timer = nil
            self._connectionState = .disconnected
        }
    }

    // MARK: - Demo / gaze target control (synthetic-only API)

    /// Set the gaze target by ARROWS index (0..3). The corresponding frequency
    /// from `EEGConfig.defaultFrequencies` is embedded in the occipital channels,
    /// simulating the user looking at that flicker tile. Pass `nil` (or an
    /// out-of-range index) to simulate gazing away (pure noise / no decision).
    ///
    /// This is the "gaze/target setter for the demo" — it is intentionally NOT
    /// part of the `EEGSource` protocol and is only available on this concrete
    /// type, used by demo controls to drive a known ground-truth direction.
    func setTarget(_ arrowIndex: Int?) {
        genQueue.async {
            guard let i = arrowIndex,
                  EEGConfig.defaultFrequencies.indices.contains(i) else {
                self.targetFrequency = nil
                return
            }
            self.targetFrequency = EEGConfig.defaultFrequencies[i]
        }
    }

    /// Set the embedded SSVEP frequency directly in Hz, bypassing the ARROWS
    /// table. Pass `nil` to simulate gazing away (pure noise). Useful when the
    /// flicker frequencies are recomputed at runtime via `achievableFreqs`.
    func setTargetFrequency(_ hz: Double?) {
        genQueue.async {
            self.targetFrequency = hz
        }
    }

    // MARK: - Sample synthesis (genQueue only)

    /// Build and yield one length-8 µV sample for the current `sampleIndex`.
    private func emitSample() {
        guard let continuation = samplesContinuation else {
            // No subscriber yet; still advance time so phase stays continuous.
            sampleIndex &+= 1
            return
        }

        let t = Double(sampleIndex) / EEGConfig.fs   // seconds since start
        let twoPi = 2.0 * Double.pi
        let target = targetFrequency

        var sample = [Double](repeating: 0, count: EEGConfig.channelCount)
        let occipital = Set(EEGConfig.occipitalIndices)

        for ch in 0..<EEGConfig.channelCount {
            // --- Background EEG: 1/f-ish pink noise + a little white noise. ---
            var value = backgroundAmplitude * pinkState[ch].next()
            value += 1.5 * gaussian()   // white sensor noise (µV)

            // --- Mains hum (region-configurable; downstream notch removes it). ---
            value += mainsAmplitude * sin(twoPi * EEGConfig.mainsHz * t)

            // --- SSVEP on occipital channels only. ---
            if let f = target, occipital.contains(ch) {
                let basePhase = channelPhase[ch]
                for h in 1...harmonicCount {
                    // Harmonic amplitude falloff ~ 1/h so the fundamental
                    // dominates, matching real SSVEP spectra.
                    let amp = ssvepAmplitude / Double(h)
                    let phase = basePhase * Double(h)
                    value += amp * sin(twoPi * f * Double(h) * t + phase)
                }
            }

            sample[ch] = value
        }

        continuation.yield(sample)
        sampleIndex &+= 1
    }

    /// Standard-normal sample via Box–Muller, using the instance RNG.
    private func gaussian() -> Double {
        let u1 = Double.random(in: Double.ulpOfOne...1, using: &rng)
        let u2 = Double.random(in: 0..<1, using: &rng)
        return (-2.0 * log(u1)).squareRoot() * cos(2.0 * .pi * u2)
    }
}

// MARK: - Pink (1/f) noise generator

/// Voss–McCartney pink-noise generator producing roughly unit-variance output.
/// Each instance is seeded so per-channel noise is decorrelated and the run is
/// reproducible. Not thread-safe; each channel owns its own instance and is only
/// touched on `genQueue`.
private struct PinkNoise {
    /// Number of white-noise "octave" rows summed to approximate 1/f.
    private static let rows = 16

    private var values: [Double]
    private var rng: SeededGenerator
    private var counter: UInt32 = 0
    private var runningSum: Double

    init(seed: UInt64) {
        rng = SeededGenerator(seed: seed)
        values = [Double](repeating: 0, count: PinkNoise.rows)
        // Prime each row with an initial white sample.
        for i in 0..<PinkNoise.rows {
            values[i] = Double.random(in: -1...1, using: &rng)
        }
        runningSum = values.reduce(0, +)
    }

    /// Next pink-noise sample, scaled to ~unit variance.
    mutating func next() -> Double {
        // Update exactly one row per call, chosen by the lowest set bit of the
        // counter (the Voss–McCartney trick), giving the 1/f spectrum cheaply.
        counter &+= 1
        if counter != 0 {
            let row = Int(counter.trailingZeroBitCount) % PinkNoise.rows
            runningSum -= values[row]
            values[row] = Double.random(in: -1...1, using: &rng)
            runningSum += values[row]
        }
        // White-noise dither + normalize by row count.
        let white = Double.random(in: -1...1, using: &rng)
        return (runningSum + white) / Double(PinkNoise.rows + 1) * 3.0
    }
}

// MARK: - Seeded RNG

/// A small deterministic SplitMix64 generator so synthetic runs are reproducible
/// given a seed (handy for tests and demos). Conforms to `RandomNumberGenerator`.
private struct SeededGenerator: RandomNumberGenerator {
    private var state: UInt64

    init(seed: UInt64) {
        // Avoid a zero state.
        state = seed == 0 ? 0x9E3779B97F4A7C15 : seed
    }

    mutating func next() -> UInt64 {
        state &+= 0x9E3779B97F4A7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }
}
