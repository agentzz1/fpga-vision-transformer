//
//  AppViewModel.swift
//  SSVEP2048
//
//  Top-level application coordinator. This is the single object that wires the
//  whole real-time SSVEP pipeline together and publishes its observable outputs
//  to SwiftUI:
//
//      EEGSource  ──samples()──▶  EEGRingBuffer  ──occipital window──▶  Filters
//          │                                                              │
//          └──states()──▶ connectionState                                ▼
//                                                                       FBCCA
//                                                                         │
//                                                                     Decision
//                                                                         │
//                                                       SSVEPController.ingest(_:)
//                                                                         │
//                                                          game / dwell / quality
//
//  Responsibilities (interface contract §B.18):
//    - Choose the EEG source: `SyntheticEEGSource` in the Simulator or when
//      `forceSynthetic` is requested, otherwise `UnicornBLEManager`.
//    - Compute the on-screen flicker targets for the device refresh rate via
//      `achievableFreqs(refresh:)` (index-aligned to `EEGConfig.ARROWS`).
//    - Drain the source's `samples()` stream into the `EEGRingBuffer` and drain
//      its `states()` stream into the published `connectionState`.
//    - Run a periodic decode on a background task: every ~0.5 s pull the latest
//      2 s occipital window, pre-filter it (6–80 Hz Butterworth + mains notch),
//      classify it with FBCCA, and feed the resulting `Decision` to the
//      `SSVEPController` (whose dwell/refractory gate turns sustained intent into
//      a single 2048 move). A simple signal-quality estimate is published too.
//
//  Threading (contract §D.5):
//    - This type is `@MainActor`; every `@Published` mutation happens on main.
//    - Sample ingestion and the DSP/FBCCA decode run OFF main, inside detached
//      `Task`s; only the final `Decision` and quality value hop back to main.
//    - The `EEGRingBuffer` is internally thread-safe, so the producing task and
//      the decoding task can touch it concurrently.
//

import SwiftUI
import Combine

#if canImport(UIKit)
import UIKit
#endif

@MainActor
final class AppViewModel: ObservableObject {

    // MARK: - Published UI state (contract surface)

    /// Live connection state of the active EEG source, mirrored from
    /// `EEGSource.states()`. Drives the connection banner in the UI.
    @Published private(set) var connectionState: EEGConnectionState

    /// On-screen flicker targets, index-aligned to `EEGConfig.ARROWS`
    /// (ARROWS[i] ↔ targets[i] ↔ Decision.target == i). Computed once at init
    /// from the device refresh rate.
    @Published private(set) var targets: [FlickerTarget]

    /// Whether the synthetic source is in use (Simulator, forced, or BLE absent).
    /// Lets the UI show a "DEMO / synthetic EEG" indicator.
    @Published private(set) var usingSynthetic: Bool

    // MARK: - Published quality (beyond contract minimum, additive UI signal)

    /// A coarse 0…1 estimate of decode quality for the current window, derived
    /// from the latest `Decision.confidence` (margin of winner over runner-up).
    /// 0 == no usable signal / no decision; 1 == a strong, well-separated peak.
    /// This is an *additive* convenience for the UI; it is not part of the
    /// decode/game contract and never feeds back into the pipeline.
    @Published private(set) var signalQuality: Double = 0

    // MARK: - Owned subsystems

    /// Decode → game bridge with dwell/refractory gating. Public per §B.18 so the
    /// UI can observe `controller.game`, `controller.dwellProgress`, etc.
    let controller: SSVEPController

    // MARK: - Pipeline configuration

    /// Display refresh used to derive `targets`. Captured at init.
    private let refresh: Double

    /// The active EEG source (synthetic or BLE). Held as the protocol type so the
    /// pipeline is source-agnostic. Strong reference keeps it alive for the app's
    /// lifetime; `EEGSource` is a class-bound (`AnyObject`) protocol.
    private let source: EEGSource

    /// Concrete synthetic source, when one is in use. Lets demo controls (e.g.
    /// `setSyntheticGaze`) drive a known ground-truth direction. `nil` when the
    /// real BLE source is active.
    private let syntheticSource: SyntheticEEGSource?

    /// Sliding-window store fed by the sample stream, read by the decoder.
    private let ringBuffer = EEGRingBuffer()

    /// FBCCA decoder, built once for the chosen target frequencies.
    private let decoder: FBCCA

    /// Per-occipital-channel pre-filter cascades (6–80 Hz Butterworth + mains
    /// notch). One stateful cascade per channel so IIR delay state persists across
    /// overlapping windows. Count == `EEGConfig.occipitalIndices.count`.
    private let prefilters: [BiquadCascade]

    // MARK: - Decode cadence

    /// Decode period in seconds. The contract calls for overlapping 2 s windows
    /// produced at ~0.5 s cadence; with a 2 s window that is a 75% overlap.
    private let decodeInterval: TimeInterval = 0.5

    /// Window length (samples) handed to the decoder = 2 s @ fs.
    private let windowSamples = EEGConfig.windowSamples

    // MARK: - Running tasks

    /// Drains `source.samples()` into the ring buffer (off main).
    private var sampleTask: Task<Void, Never>?

    /// Drains `source.states()` into `connectionState` (results applied on main).
    private var stateTask: Task<Void, Never>?

    /// Periodic decode loop: window → prefilter → FBCCA → controller (off main,
    /// hops to main only to deliver the Decision + quality).
    private var decodeTask: Task<Void, Never>?

    /// Guards against double `start()`.
    private var isRunning = false

    // MARK: - Init

    /// - Parameters:
    ///   - forceSynthetic: force `SyntheticEEGSource` even on a real device. The
    ///     Simulator is *always* synthetic (no CoreBluetooth radio) regardless.
    ///   - refresh: display refresh rate in Hz used to derive flicker targets
    ///     (e.g. 60 or 120). Defaults to 60.
    init(forceSynthetic: Bool = false, refresh: Double = 60) {
        self.refresh = refresh

        // --- Targets: realized frame-locked frequencies for this refresh. ------
        // Index identity with ARROWS is load-bearing (contract §D.4).
        let targets = achievableFreqs(refresh: refresh)
        self.targets = targets

        // --- Source selection --------------------------------------------------
        // The Simulator cannot run CoreBluetooth, so it must use the synthetic
        // source; on device we honour `forceSynthetic`.
        let useSynthetic = forceSynthetic || Self.isSimulator
        self.usingSynthetic = useSynthetic

        if useSynthetic {
            // Seed the synthetic source with the first target's frequency so the
            // app shows a plausible "up" decision out of the box; demo controls
            // can change the gaze afterwards.
            let seedFreq = targets.first?.frequency
            let synth = SyntheticEEGSource(injectFrequency: seedFreq)
            self.syntheticSource = synth
            self.source = synth
        } else {
            self.syntheticSource = nil
            self.source = UnicornBLEManager(namePrefix: UnicornBLE.advertisedNamePrefix)
        }

        self.connectionState = self.source.connectionState

        // --- Decoder -----------------------------------------------------------
        self.decoder = FBCCA(targets: targets)

        // --- Game controller ---------------------------------------------------
        self.controller = SSVEPController()

        // --- Pre-filter cascades (one per occipital channel) -------------------
        let sections = Filters.ssvepPrefilter(fs: Float(EEGConfig.fs),
                                              mains: Float(EEGConfig.mainsHz))
        self.prefilters = EEGConfig.occipitalIndices.map { _ in
            BiquadCascade(sections: sections)
        }
    }

    // MARK: - Lifecycle

    /// Start the source and spin up the three pipeline tasks. Idempotent.
    func start() {
        guard !isRunning else { return }
        isRunning = true

        // Fresh state for a clean run.
        ringBuffer.reset()
        prefilters.forEach { $0.reset() }

        // Bring the source online (power-on/scan for BLE; timer for synthetic).
        source.start()

        startStateTask()
        startSampleTask()
        startDecodeTask()
    }

    /// Tear down all tasks and stop the source. Idempotent.
    func stop() {
        guard isRunning else { return }
        isRunning = false

        sampleTask?.cancel();  sampleTask = nil
        stateTask?.cancel();   stateTask = nil
        decodeTask?.cancel();  decodeTask = nil

        source.stop()
    }

    /// Reset the game (and the gate) without disturbing the EEG pipeline.
    func resetGame() {
        controller.reset()
    }

    // MARK: - Demo controls (synthetic source only)

    /// When running on the synthetic source, point the simulated gaze at the
    /// given ARROWS index (0…3) so the decoder should converge on that target.
    /// No-op on the real BLE source. Useful for UI demos/tests without a headset.
    func setSyntheticGaze(_ arrowIndex: Int?) {
        guard let synth = syntheticSource else { return }
        if let i = arrowIndex, targets.indices.contains(i) {
            // Drive by the *exact realized* frequency so it matches the reference
            // bank the decoder built for that target.
            synth.setTargetFrequency(targets[i].frequency)
        } else {
            synth.setTargetFrequency(nil)
        }
    }

    // MARK: - Pipeline tasks

    /// Mirror `EEGSource.states()` into the published `connectionState`.
    private func startStateTask() {
        let stream = source.states()
        stateTask = Task { [weak self] in
            for await state in stream {
                if Task.isCancelled { break }
                await MainActor.run { self?.connectionState = state }
            }
        }
    }

    /// Drain length-8 µV samples into the ring buffer off the main actor.
    private func startSampleTask() {
        let stream = source.samples()
        let buffer = ringBuffer
        sampleTask = Task.detached(priority: .userInitiated) {
            for await sample in stream {
                if Task.isCancelled { break }
                // Defensive: only push correctly-shaped scans; the ring also
                // asserts/guards on length internally.
                if sample.count == EEGConfig.channelCount {
                    buffer.push(sample)
                }
            }
        }
    }

    /// Periodic decode loop. Runs off main; pulls the latest occipital window,
    /// pre-filters it in place per channel, runs FBCCA, and dispatches the
    /// `Decision` (plus a quality estimate) back to the main actor.
    private func startDecodeTask() {
        // Capture immutable pipeline pieces so the detached task does not touch
        // `self` (an actor-isolated type) on a background thread.
        let buffer = ringBuffer
        let decoder = self.decoder
        let prefilters = self.prefilters
        let occipital = EEGConfig.occipitalIndices
        let n = windowSamples
        let intervalNanos = UInt64(decodeInterval * 1_000_000_000)

        decodeTask = Task.detached(priority: .userInitiated) { [weak self] in
            while !Task.isCancelled {
                // Wait one cadence period between decodes (~0.5 s).
                try? await Task.sleep(nanoseconds: intervalNanos)
                if Task.isCancelled { break }

                // Need a full 2 s of data before the first decode.
                guard let raw = buffer.latestWindow(n: n, channels: occipital) else {
                    continue
                }

                // Pre-filter each occipital channel block in place (persistent
                // IIR state across overlapping windows). `raw` is channel-major:
                // channel c occupies [c*n, c*n + n).
                var filtered = raw
                let channelCount = occipital.count
                filtered.withUnsafeMutableBufferPointer { buf in
                    guard let base = buf.baseAddress else { return }
                    for c in 0..<channelCount {
                        let chan = base + c * n
                        // In-place filtering: read and write the same block.
                        prefilters[c].process(chan, chan, n: n)
                    }
                }

                // Classify the filtered occipital window.
                let decision: Decision = filtered.withUnsafeBufferPointer { buf in
                    guard let base = buf.baseAddress else {
                        return Decision(target: -1, scores: [], confidence: 0)
                    }
                    return decoder.classify(window: base, channelCount: channelCount)
                }

                // Deliver to the gate + UI on the main actor.
                let quality = Self.quality(from: decision)
                await MainActor.run {
                    guard let self, self.isRunning else { return }
                    self.signalQuality = quality
                    self.controller.ingest(decision)
                }
            }
        }
    }

    // MARK: - Helpers

    /// Map a `Decision`'s confidence (max/secondMax, >= 1) onto a 0…1 quality
    /// bar. A confidence of 1 (no separation) → 0; confidence >= 2 (winner is
    /// twice the runner-up) → 1. Linear in between. A "no decision" (target -1)
    /// reports 0.
    private static func quality(from decision: Decision) -> Double {
        guard decision.target >= 0 else { return 0 }
        let c = Double(decision.confidence)
        let normalized = (c - 1.0)        // 0 at conf 1, 1 at conf 2
        return min(1.0, max(0.0, normalized))
    }

    /// True when building/running for the iOS Simulator, which has no Bluetooth
    /// radio and therefore must use the synthetic source.
    private static var isSimulator: Bool {
        #if targetEnvironment(simulator)
        return true
        #else
        return false
        #endif
    }
}
