//
//  FBCCA.swift
//  SSVEP2048
//
//  Filter-bank Canonical Correlation Analysis (FBCCA) decoder for SSVEP, per
//  Chen et al. (2015), "Filter bank canonical correlation analysis for
//  implementing a high-speed SSVEP-based brain-computer interface",
//  J. Neural Eng. 12(4):046008.
//
//  This file builds directly on top of:
//    - `CCA.canonicalCorrelationMax(...)`  (Decode/CCA.swift)        — ρ between
//      a multichannel EEG sub-band window and a sin/cos reference matrix.
//    - `makeReferenceBank(...)` / `ReferenceBank` (Decode/ReferenceBank.swift)
//      — the per-frequency sin/cos harmonic reference matrices.
//    - `Filters.subBand(_:fs:)` / `BiquadCascade` (DSP/Filters.swift)
//      — the Chen sub-band band-pass filters applied before each CCA.
//
//  Algorithm summary (Chen 2015):
//    For a single window X (channel-major, occipital subset already selected by
//    the caller) and a set of target frequencies f_t:
//      1. Decompose X into M sub-bands  SB_m(X)  (each a band-pass of X with a
//         successively higher low corner — `Filters.subBand(m, fs:)`).
//      2. For each sub-band m and each target t, compute the canonical
//         correlation  ρ_{m,t} = CCA( SB_m(X) , Y_t )  where Y_t is the
//         frequency-t harmonic reference bank.
//      3. Combine across sub-bands with the Chen weighting:
//             ρ²_t = Σ_m  w(m) · ρ²_{m,t},     w(m) = m^(-a) + b
//         here a = `weightA` (1.25), b = `weightB` (0.25), m 1-based.
//      4. Decision = argmax_t ρ²_t ; confidence = max / second-max.
//
//  Data conventions (interface contract §B.0 / §D):
//    - `window` is channel-major Float, length `channelCount * N`, each channel a
//      contiguous N-block (= LAPACK column-major N×channelCount). The occipital
//      subset has ALREADY been selected by the caller; `channelCount` is the
//      subset size.
//    - `targets` are index-aligned to `EEGConfig.ARROWS`; `Decision.target` is
//      that same index (or -1 for "no decision").
//

import Accelerate
import Foundation

// MARK: - FBCCAConfig

/// Static configuration for the filter-bank CCA decoder.
///
/// Defaults follow Chen et al. (2015) and the interface contract (§B.12):
///   - three sub-bands `[(8,80),(16,80),(24,80)]` Hz. NOTE: the high corner is
///     capped at 80 Hz to match the broadband pre-filter's 80 Hz upper corner
///     (`Filters.ssvepPrefilter`); content above 80 Hz is already removed before
///     FBCCA, so a wider sub-band corner would be dead range. (Chen used 90 Hz;
///     reconciled to the prefilter here.)
///   - weighting exponents `a = 1.25`, `b = 0.25`,
///   - `N = EEGConfig.windowSamples` samples at `fs = EEGConfig.fs`,
///   - `harmonics = EEGConfig.harmonics` (3) sin/cos pairs per reference.
struct FBCCAConfig {
    /// Sampling rate in Hz.
    let fs: Double
    /// Window length in samples (must match the window the caller passes in).
    let N: Int
    /// Number of harmonics (sin/cos pairs) in each reference bank.
    let harmonics: Int
    /// Sub-band band-pass corner frequencies (low, high) in Hz, one per sub-band.
    /// The number of entries determines the number of sub-bands M.
    let subBands: [(lowHz: Float, highHz: Float)]
    /// Chen weighting exponent `a` in `w(m) = m^(-a) + b`.
    let weightA: Float
    /// Chen weighting offset `b` in `w(m) = m^(-a) + b`.
    let weightB: Float

    init(fs: Double = EEGConfig.fs,
         N: Int = EEGConfig.windowSamples,
         harmonics: Int = EEGConfig.harmonics,
         subBands: [(lowHz: Float, highHz: Float)] = [(8, 80), (16, 80), (24, 80)],
         weightA: Float = 1.25, weightB: Float = 0.25) {
        self.fs = fs
        self.N = N
        self.harmonics = harmonics
        self.subBands = subBands
        self.weightA = weightA
        self.weightB = weightB
    }
}

// MARK: - FBCCA

/// Filter-bank CCA SSVEP classifier.
///
/// One instance is built per set of on-screen targets (frequencies). Reference
/// banks and the per-sub-band weights are computed once at init; `classify` is
/// then called once per sliding decode window.
final class FBCCA {

    // MARK: Stored configuration

    /// Decoder configuration (frequencies, harmonics, sub-bands, weights).
    private let config: FBCCAConfig

    /// Target frequencies in Hz, index-aligned to `EEGConfig.ARROWS`.
    private let frequencies: [Double]

    /// Per-target harmonic reference banks (column-major N × (2*harmonics)).
    /// `referenceBanks[t]` is the sin/cos matrix for `frequencies[t]`.
    private let referenceBanks: [ReferenceBank]

    /// Chen sub-band weights `w(m) = m^(-a) + b`, one per sub-band (1-based m).
    private let subBandWeights: [Float]

    /// Pre-built sub-band biquad section sets (one section list per sub-band).
    /// Stored as plain coefficients; a fresh, stateful `BiquadCascade` is created
    /// per `classify` call per channel so that decode windows are filtered
    /// independently (no cross-window/cross-channel delay leakage), which is the
    /// correct semantics for one-shot block CCA on an isolated window.
    private let subBandSections: [[Biquad]]

    // MARK: Init

    /// - Parameters:
    ///   - targets: flicker targets index-aligned to `EEGConfig.ARROWS`. Only the
    ///     `frequency` field is used by the decoder; phase/frames are display-side.
    ///   - config: decoder configuration; defaults to Chen (2015) parameters.
    init(targets: [FlickerTarget], config: FBCCAConfig = FBCCAConfig()) {
        self.config = config
        self.frequencies = targets.map { $0.frequency }

        // Build one harmonic reference bank per target frequency, once.
        self.referenceBanks = targets.map { target in
            makeReferenceBank(freq: target.frequency,
                              harmonics: config.harmonics,
                              fs: config.fs,
                              N: config.N)
        }

        // Pre-compute Chen sub-band weights w(m) = m^(-a) + b for 1-based m.
        self.subBandWeights = (1...max(1, config.subBands.count)).map { m in
            powf(Float(m), -config.weightA) + config.weightB
        }

        // Pre-design the band-pass sections for each sub-band DIRECTLY from the
        // configured corner frequencies, so a caller passing custom sub-bands
        // (e.g. [(10,40)]) actually gets those corners. `config.subBands` is the
        // single source of truth — NOT the hardcoded Chen 8/16/24–90 lows in
        // `Filters.subBand`. (The default config still yields Chen's sub-bands.)
        let fsF = Float(config.fs)
        let nyquist = fsF / 2
        if config.subBands.isEmpty {
            // Degenerate config: fall back to one Chen sub-band so M >= 1.
            self.subBandSections = [Filters.subBand(1, fs: fsF)]
        } else {
            self.subBandSections = config.subBands.map { band in
                // Clamp the high corner strictly below Nyquist for stability.
                let high = min(band.highHz, nyquist * 0.95)
                return BiquadDesign.butterBandpass(lowHz: band.lowHz,
                                                   highHz: high, fs: fsF)
            }
        }
    }

    // MARK: Classification

    /// Classify a single decode window.
    ///
    /// - Parameters:
    ///   - window: channel-major Float, length `channelCount * config.N`. The
    ///     occipital subset must already be selected by the caller.
    ///   - channelCount: number of channels in `window` (the subset size).
    /// - Returns: a `Decision` whose `scores[t] = Σ_m w(m)·ρ²_{m,t}`,
    ///   `target == argmax_t scores[t]`, and `confidence == max / second-max`.
    ///   If there are no targets, returns the empty "no decision" result.
    func classify(window: UnsafePointer<Float>, channelCount: Int) -> Decision {
        let nTargets = frequencies.count
        let N = config.N
        let M = subBandSections.count

        // Degenerate guards: no targets / empty window / no channels.
        guard nTargets > 0, N > 0, channelCount > 0, M > 0 else {
            return Decision(target: -1, scores: [], confidence: 0)
        }

        // ----- 1) Pre-filter the window into each sub-band, once per classify. -----
        // For each sub-band m we produce a channel-major buffer `SB_m` of the same
        // shape as `window` (channelCount * N), filtered per channel with a fresh
        // stateful cascade (block-isolated filtering of this window).
        var subBandWindows: [[Float]] = []
        subBandWindows.reserveCapacity(M)

        for m in 0..<M {
            var filtered = [Float](repeating: 0, count: channelCount * N)
            let sections = subBandSections[m]

            // Filter each contiguous channel block independently. A new cascade
            // per channel keeps delay state from leaking between channels.
            filtered.withUnsafeMutableBufferPointer { outBuf in
                guard let outBase = outBuf.baseAddress else { return }
                for c in 0..<channelCount {
                    let cascade = BiquadCascade(sections: sections)
                    let inChan = window + c * N
                    let outChan = outBase + c * N
                    cascade.process(inChan, outChan, n: N)
                }
            }
            subBandWindows.append(filtered)
        }

        // ----- 2) CCA per (sub-band, target); 3) weighted sum of squares. -----
        var scores = [Float](repeating: 0, count: nTargets)

        for m in 0..<M {
            let weight = subBandWeights[m]
            // Bind the sub-band window pointer once for all targets in this band.
            subBandWindows[m].withUnsafeBufferPointer { sbBuf in
                guard let X = sbBuf.baseAddress else { return }

                for t in 0..<nTargets {
                    let bank = referenceBanks[t]
                    // Reference banks must match the configured window length; if a
                    // mismatched bank slips through, skip it rather than read OOB.
                    guard bank.N == N else { continue }

                    bank.Y.withUnsafeBufferPointer { yBuf in
                        guard let Y = yBuf.baseAddress else { return }

                        // Largest canonical correlation between the sub-band EEG
                        // (N × channelCount, column-major) and the reference
                        // (N × bank.cols, column-major).
                        let rho = CCA.canonicalCorrelationMax(X: X, p: channelCount,
                                                              Y: Y, q: bank.cols,
                                                              N: N)
                        // Chen: accumulate weighted SQUARED correlation.
                        scores[t] += weight * rho * rho
                    }
                }
            }
        }

        // ----- 4) Argmax + confidence (max / second-max). -----
        return Self.makeDecision(scores: scores)
    }

    // MARK: - Decision assembly

    /// Build a `Decision` from per-target scores: argmax target and a confidence
    /// ratio of the best to the second-best score (>= 1; defined as 1 when there
    /// is only one target or the runner-up is ~0).
    private static func makeDecision(scores: [Float]) -> Decision {
        guard !scores.isEmpty else {
            return Decision(target: -1, scores: [], confidence: 0)
        }

        // Find the best and second-best scores in one pass.
        var bestIdx = 0
        var best: Float = scores[0]
        var second: Float = -.greatestFiniteMagnitude
        for i in 1..<scores.count {
            let s = scores[i]
            if s > best {
                second = best
                best = s
                bestIdx = i
            } else if s > second {
                second = s
            }
        }

        // Confidence = best / second-best, clamped to >= 1. With a single target
        // or a non-positive runner-up the ratio is undefined → report 1.
        let confidence: Float
        if scores.count < 2 || second <= 0 {
            confidence = 1
        } else {
            confidence = max(1, best / second)
        }

        return Decision(target: bestIdx, scores: scores, confidence: confidence)
    }
}

// MARK: - achievableFreqs

/// Compute the realized SSVEP target frequencies for a given display refresh.
///
/// Frame-locked flicker can only realize frequencies that divide the refresh by
/// an integer number of frames per *half* period (the tile toggles every `k`
/// frames, so one full on/off cycle takes `2k` frames):
///
///     f(k) = refresh / (2 * k)
///
/// We enumerate integer half-period counts `k` whose realized frequency lands in
/// the practical SSVEP band [6, 16] Hz (good SNR, comfortable, low harmonic
/// overlap), then pick 4 **maximally separated** frequencies from that set so
/// neighbouring targets are easy to discriminate. Results are returned sorted
/// **descending by frequency** and are index-aligned to `EEGConfig.ARROWS`.
///
/// Each returned `FlickerTarget` carries its exact realized `frequency` and the
/// integer `framesPerHalfPeriod` (k) the renderer (`FlickerView`) uses to drive
/// opacity from an integer frame counter (never `Date()`), per the contract.
///
/// - Parameter refresh: display refresh rate in Hz (e.g. 60 or 120).
/// - Returns: exactly 4 `FlickerTarget`s sorted descending by frequency. If the
///   band yields fewer than 4 distinct candidates, falls back to as many as are
///   available padded with the highest available candidate (never returns junk
///   outside the band).
func achievableFreqs(refresh: Double) -> [FlickerTarget] {

    // Practical SSVEP band for frame-locked stimulation.
    let lowHz = 6.0
    let highHz = 16.0
    let desiredCount = 4

    // 1) Enumerate all integer half-period candidates k with f(k) in [low, high].
    //    f decreases as k increases, so iterate k upward and keep in-band hits.
    var candidates: [(k: Int, f: Double)] = []
    if refresh > 0 {
        // Smallest k that keeps f <= highHz: k >= refresh / (2*highHz).
        let kMin = max(1, Int(ceil(refresh / (2 * highHz))))
        // Largest k that keeps f >= lowHz: k <= refresh / (2*lowHz).
        let kMax = max(kMin, Int(floor(refresh / (2 * lowHz))))
        for k in kMin...kMax {
            let f = refresh / (2.0 * Double(k))
            if f >= lowHz - 1e-9 && f <= highHz + 1e-9 {
                candidates.append((k, f))
            }
        }
    }

    // Defensive fallback: if the refresh is degenerate or no in-band candidate
    // exists, return the contract's default frequencies as plain targets.
    guard !candidates.isEmpty else {
        return EEGConfig.defaultFrequencies.map { f in
            FlickerTarget(frequency: f, framesPerHalfPeriod: 0)
        }
    }

    // Sort candidates ascending by frequency for the spacing selection below.
    candidates.sort { $0.f < $1.f }

    // 2) Pick `desiredCount` maximally-separated candidates.
    //    If we have exactly enough (or fewer), take them all; otherwise greedily
    //    choose endpoints first, then repeatedly insert the candidate that
    //    maximizes the minimum spacing to the already-chosen set. This yields a
    //    well-spread subset across the band for good inter-target separation.
    let selected: [(k: Int, f: Double)]
    if candidates.count <= desiredCount {
        selected = candidates
    } else {
        // Start with the band extremes (lowest and highest realized freq).
        var chosenIdx: [Int] = [0, candidates.count - 1]
        while chosenIdx.count < desiredCount {
            var bestCandidate = -1
            var bestMinGap = -1.0
            for i in 0..<candidates.count where !chosenIdx.contains(i) {
                // Minimum distance from candidate i to any already-chosen freq.
                var minGap = Double.greatestFiniteMagnitude
                for j in chosenIdx {
                    minGap = min(minGap, abs(candidates[i].f - candidates[j].f))
                }
                if minGap > bestMinGap {
                    bestMinGap = minGap
                    bestCandidate = i
                }
            }
            if bestCandidate < 0 { break }
            chosenIdx.append(bestCandidate)
        }
        selected = chosenIdx.map { candidates[$0] }
    }

    // 3) Sort descending by frequency (contract: highest first) and map to
    //    FlickerTargets carrying the exact realized frequency and k.
    let descending = selected.sorted { $0.f > $1.f }
    var targets = descending.map { cand in
        FlickerTarget(frequency: cand.f, framesPerHalfPeriod: cand.k)
    }

    // Guarantee exactly `desiredCount` entries (index identity with ARROWS is
    // load-bearing). If the band yielded fewer than `desiredCount` DISTINCT
    // frame-locked frequencies, do NOT duplicate (two arrows sharing a frequency
    // are indistinguishable to FBCCA). Instead, fill the remaining slots from
    // `EEGConfig.defaultFrequencies`, skipping any value already realized, so
    // every emitted target frequency is unique.
    if targets.count < desiredCount {
        var used = Set(targets.map { Self_round($0.frequency) })
        for f in EEGConfig.defaultFrequencies {
            guard targets.count < desiredCount else { break }
            let key = Self_round(f)
            if used.contains(key) { continue }
            used.insert(key)
            // framesPerHalfPeriod 0 marks a non-frame-locked fallback target;
            // the renderer clamps k to >= 1 so it still flickers.
            targets.append(FlickerTarget(frequency: f, framesPerHalfPeriod: 0))
        }
    }
    return Array(targets.prefix(desiredCount))
}

/// Quantize a frequency to a stable key for duplicate detection (~0.01 Hz).
private func Self_round(_ f: Double) -> Int {
    Int((f * 100).rounded())
}
