//
//  Filters.swift
//  SSVEP2048
//
//  DSP filtering facade for the SSVEP pipeline.
//
//  Provides:
//   - `BiquadCascade`: a per-channel, stateful cascade of `Biquad` sections that
//     filters `[Float]` windows in place using Accelerate's vDSP biquad engine.
//   - `Filters`: a small enum facade that assembles the standard SSVEP pre-filter
//     chain (broadband Butterworth band-pass + mains notch) and the FBCCA
//     sub-band band-pass sections.
//
//  Data conventions (per the interface contract, §D):
//   - A "window" is channel-major Float, length C*N, each channel a contiguous
//     N-block. `BiquadCascade` is intended to be instantiated ONE PER CHANNEL so
//     that its biquad delay state corresponds to a single continuous channel
//     signal; the caller advances pointer/length into each channel block.
//   - All section coefficients are `Float` (a0 normalized to 1), matching
//     `Biquad` (§B.8) and the FBCCA / decode path which is single-precision.
//
//  Implementation note on the vDSP path:
//   Accelerate exposes a clean single-precision multi-section IIR engine via
//   `vDSP_biquad` together with a `vDSP_biquad_Setup` (created from a packed
//   coefficient array and a persistent delay buffer). This is the path used
//   below. It keeps delay state across calls (required so windows stride
//   continuously without edge transients at window boundaries). If for some
//   reason the vDSP setup cannot be created, we fall back to a hand-rolled
//   Direct-Form-II Transposed biquad cascade in pure Swift (mathematically
//   identical, also state-persistent) — see `processFallback`.
//

import Accelerate
import Foundation

// MARK: - BiquadCascade

/// One per channel — owns persistent biquad delay state across windows.
///
/// Wraps `vDSP_biquad` (single precision). The cascade applies each `Biquad`
/// section in series; delay memory is retained between `process(_:_:n:)` calls so
/// that successive (strided / overlapping) windows of the *same* channel are
/// filtered as one continuous stream.
final class BiquadCascade {

    /// The cascade's sections (kept for reset / fallback / re-setup).
    private let sections: [Biquad]

    /// Number of biquad sections.
    private let sectionCount: vDSP_Length

    /// Opaque vDSP biquad setup (coefficients + section count). `nil` triggers the
    /// pure-Swift fallback path.
    private var setup: vDSP_biquad_Setup?

    /// Persistent delay state required by `vDSP_biquad`.
    ///
    /// vDSP requires a delay buffer of length `2 * numSections + 2` Floats, which
    /// it reads at entry and writes at exit so state survives across calls.
    private var delays: [Float]

    /// Fallback Direct-Form-II Transposed state: two delay registers per section.
    /// Only used when `setup == nil`.
    private var z1: [Float]
    private var z2: [Float]

    /// - Parameter sections: ordered biquad sections to apply in series.
    init(sections: [Biquad]) {
        self.sections = sections
        self.sectionCount = vDSP_Length(sections.count)

        // vDSP delay buffer length: 2*N + 2 floats, zero-initialized.
        self.delays = [Float](repeating: 0, count: 2 * sections.count + 2)

        // Fallback state, two registers per section.
        self.z1 = [Float](repeating: 0, count: sections.count)
        self.z2 = [Float](repeating: 0, count: sections.count)

        self.setup = BiquadCascade.makeSetup(from: sections)
    }

    deinit {
        if let setup = setup {
            vDSP_biquad_DestroySetup(setup)
        }
    }

    /// Build the packed coefficient array vDSP expects and create the setup.
    ///
    /// vDSP wants 5 Double coefficients per section in the order
    /// `[b0, b1, b2, a1, a2]` with a0 already normalized to 1 — exactly the
    /// `Biquad` layout. Returns `nil` (→ fallback) if there are no sections or the
    /// setup cannot be created.
    private static func makeSetup(from sections: [Biquad]) -> vDSP_biquad_Setup? {
        guard !sections.isEmpty else { return nil }

        // vDSP_biquad_CreateSetup takes Double coefficients regardless of the
        // Float processing precision used by vDSP_biquad.
        var coeffs = [Double]()
        coeffs.reserveCapacity(sections.count * 5)
        for s in sections {
            coeffs.append(Double(s.b0))
            coeffs.append(Double(s.b1))
            coeffs.append(Double(s.b2))
            coeffs.append(Double(s.a1))
            coeffs.append(Double(s.a2))
        }
        return vDSP_biquad_CreateSetup(coeffs, vDSP_Length(sections.count))
    }

    /// In-place per-channel filtering, length `n`. Delays persist across calls.
    ///
    /// `x` and `y` may alias (the SSVEP pipeline filters in place by passing the
    /// same buffer for input and output). `vDSP_biquad` supports in-place use.
    ///
    /// - Parameters:
    ///   - x: input samples (one channel block), length `n`.
    ///   - y: output samples (one channel block), length `n`.
    ///   - n: number of samples.
    func process(_ x: UnsafePointer<Float>, _ y: UnsafeMutablePointer<Float>, n: Int) {
        guard n > 0 else { return }

        if let setup = setup {
            // vDSP biquad engine. `delays` is read in / written out so state
            // persists across windows.
            vDSP_biquad(setup,
                        &delays,
                        x, 1,
                        y, 1,
                        vDSP_Length(n))
        } else {
            processFallback(x, y, n: n)
        }
    }

    /// Pure-Swift Direct-Form-II Transposed cascade (state-persistent), used only
    /// if the vDSP setup is unavailable. Mathematically equivalent to the vDSP
    /// path; kept for robustness / portability.
    ///
    /// Per section, DF2T recurrence (a0 == 1):
    ///   y[k] = b0*x[k] + z1
    ///   z1   = b1*x[k] - a1*y[k] + z2
    ///   z2   = b2*x[k] - a2*y[k]
    private func processFallback(_ x: UnsafePointer<Float>, _ y: UnsafeMutablePointer<Float>, n: Int) {
        // Process sample-by-sample threading through all sections so we can run in
        // place even when x and y alias.
        for k in 0..<n {
            var sample = x[k]
            for s in 0..<sections.count {
                let sec = sections[s]
                let out = sec.b0 * sample + z1[s]
                z1[s] = sec.b1 * sample - sec.a1 * out + z2[s]
                z2[s] = sec.b2 * sample - sec.a2 * out
                sample = out
            }
            y[k] = sample
        }
    }

    /// Clear all persistent delay/state. Call when starting a new continuous
    /// stream (e.g. on (re)connect) to avoid carrying over stale transients.
    func reset() {
        for i in delays.indices { delays[i] = 0 }
        for i in z1.indices { z1[i] = 0 }
        for i in z2.indices { z2[i] = 0 }
    }
}

// MARK: - Filters facade

enum Filters {

    /// Standard SSVEP pre-filter chain (3 sections): 6–80 Hz Butterworth
    /// band-pass (4th order ⇒ 2 cascaded biquads) plus a mains notch (1 biquad).
    ///
    /// - Parameters:
    ///   - fs: sampling rate in Hz (e.g. `EEGConfig.fs` = 250).
    ///   - mains: mains frequency to notch (50 or 60 Hz; `EEGConfig.mainsHz`).
    /// - Returns: ordered sections `[bp0, bp1, notch]`, total 3 biquads.
    static func ssvepPrefilter(fs: Float, mains: Float) -> [Biquad] {
        // Broadband band-pass: spans the SSVEP fundamentals and first harmonics
        // while removing slow drift (<6 Hz) and HF noise (>80 Hz).
        var sections = BiquadDesign.butterBandpass(lowHz: 6, highHz: 80, fs: fs)

        // Single-frequency mains notch. Q ≈ 30 → ~ (mains/30) Hz −3 dB bandwidth,
        // i.e. a narrow notch that suppresses line interference with minimal
        // distortion of nearby SSVEP content.
        let notch = BiquadDesign.iirNotch(f0: mains, q: 30, fs: fs)
        sections.append(notch)

        return sections
    }

    /// Chen et al. (2015) FBCCA sub-band band-pass sections for filter-bank index
    /// `m` (1-based).
    ///
    /// Each sub-band m passes [m * 8 Hz, 80 Hz], i.e. successively higher
    /// high-pass corners so that higher sub-bands isolate higher harmonics. The
    /// high corner is capped at 80 Hz to match the broadband pre-filter's 80 Hz
    /// upper corner (`ssvepPrefilter`) — content above 80 Hz is already removed
    /// before FBCCA, so a wider corner would be dead range and is also a steeper,
    /// more marginally-stable band-pass at fs=250. (Chen used 90 Hz; reconciled
    /// to the prefilter here.)
    ///
    /// NOTE: this helper is now only the degenerate fallback when an FBCCAConfig
    /// supplies no sub-bands; the normal path designs each sub-band directly from
    /// `FBCCAConfig.subBands` corners (see `FBCCA.init`).
    ///
    /// - Parameters:
    ///   - m: 1-based sub-band index (clamped to ≥ 1).
    ///   - fs: sampling rate in Hz.
    /// - Returns: a 4th-order Butterworth band-pass as 2 cascaded biquads.
    static func subBand(_ m: Int, fs: Float) -> [Biquad] {
        let band = max(1, m)

        // Low corner steps up by 8 Hz per sub-band (Chen's design).
        let low = Float(band) * 8.0

        // High corner matches the prefilter (80 Hz); keep strictly below Nyquist.
        let nyquist = fs / 2.0
        let high = min(Float(80.0), nyquist * 0.95)

        return BiquadDesign.butterBandpass(lowHz: low, highHz: high, fs: fs)
    }
}
