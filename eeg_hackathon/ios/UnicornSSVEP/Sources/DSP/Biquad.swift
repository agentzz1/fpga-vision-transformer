//
//  Biquad.swift
//  SSVEP2048
//
//  Second-order IIR (biquad) section + analog-prototype filter designers
//  (interface contract §B.8, owner 5).
//
//  A `Biquad` stores the five Direct-Form coefficients (a0 normalized to 1) in
//  the order vDSP / `BiquadCascade` expect: [b0, b1, b2, a1, a2]. `BiquadDesign`
//  produces these from continuous-time prototypes via the bilinear transform:
//    - `butterBandpass(lowHz:highHz:fs:)` → a 4th-order Butterworth band-pass as
//      two cascaded biquads (one complex-conjugate pole pair each).
//    - `iirNotch(f0:q:fs:)` → a single second-order notch (band-stop) biquad.
//
//  All math is single precision (`Float`) to match the decode path.
//

import Foundation

// MARK: - Biquad (contract §B.8)

/// A single second-order IIR section. Difference equation (a0 == 1):
///   y[n] = b0·x[n] + b1·x[n-1] + b2·x[n-2] − a1·y[n-1] − a2·y[n-2]
struct Biquad: Equatable {
    let b0: Float
    let b1: Float
    let b2: Float
    let a1: Float
    let a2: Float

    init(b0: Float, b1: Float, b2: Float, a1: Float, a2: Float) {
        self.b0 = b0
        self.b1 = b1
        self.b2 = b2
        self.a1 = a1
        self.a2 = a2
    }
}

// MARK: - BiquadDesign (contract §B.8)

/// Designers that turn band specifications into `Biquad` coefficients.
enum BiquadDesign {

    /// 4th-order Butterworth band-pass `[lowHz, highHz]` at sample rate `fs`,
    /// returned as two cascaded biquads.
    ///
    /// Method: design a 2nd-order analog low-pass Butterworth prototype, apply the
    /// analog low-pass → band-pass transform (which doubles the order to 4), then
    /// bilinear-transform each resulting complex-conjugate pole pair into a digital
    /// biquad. Pre-warping is applied at the band edges so the realized corners
    /// land where requested.
    ///
    /// Returns an empty array for a degenerate band (non-positive corners,
    /// low >= high, or corners outside (0, fs/2)).
    static func butterBandpass(lowHz: Float, highHz: Float, fs: Float) -> [Biquad] {
        let nyquist = fs / 2
        guard fs > 0, lowHz > 0, highHz > lowHz, highHz < nyquist else { return [] }

        // Work in double precision internally for design stability, emit Float.
        let fsD = Double(fs)
        let w1 = 2 * Double.pi * Double(lowHz)  / fsD
        let w2 = 2 * Double.pi * Double(highHz) / fsD

        // Pre-warp digital edge frequencies to analog (bilinear), then derive the
        // band-pass center (geometric) and bandwidth in the warped analog domain.
        let warp1 = tan(w1 / 2)
        let warp2 = tan(w2 / 2)
        let w0 = sqrt(warp1 * warp2)            // analog center (rad/s, warped)
        let bw = warp2 - warp1                   // analog bandwidth
        guard w0 > 0, bw > 0 else { return [] }

        // 2nd-order Butterworth analog low-pass prototype poles:
        //   s = exp(j·π·(2k+1)/(2·order)) for k = 0..order-1, order = 2.
        // The two prototype poles are a complex-conjugate pair at angle ±135°.
        // Each prototype pole maps, under the LP→BP transform
        //   s_lp = (s² + w0²) / (bw·s),
        // to a pair of band-pass poles, giving 4 poles total ⇒ 2 biquads.
        var sections: [Biquad] = []
        let order = 2
        for k in 0..<order {
            let theta = Double.pi * (2.0 * Double(k) + 1.0) / (2.0 * Double(order))
            // Prototype pole p (unit-cutoff Butterworth): magnitude 1, angle in
            // the left half-plane.
            let pRe = -sin(theta)
            let pIm =  cos(theta)

            // LP→BP: solve s² − (bw·p)·s + w0² = 0 for the band-pass poles.
            // Let q = bw·p / 2. Roots = q ± sqrt(q² − w0²).
            let qRe = bw * pRe / 2
            let qIm = bw * pIm / 2

            // disc = q² − w0² (complex).
            let discRe = qRe * qRe - qIm * qIm - w0 * w0
            let discIm = 2 * qRe * qIm
            let (sqRe, sqIm) = complexSqrt(re: discRe, im: discIm)

            // Take the root with positive-imag convention; its conjugate is the
            // other band-pass pole of this section (a stable conjugate pair after
            // the bilinear transform reflects them into the unit disc).
            let sRe = qRe + sqRe
            let sIm = qIm + sqIm

            // Bilinear transform z = (1 + s/?)… Here our analog variable already
            // uses tan() pre-warping with the (1+ ... ) normalization folded in,
            // so map analog pole `s` (a complex value) to a digital pole:
            //   pz = (1 + s) / (1 - s)
            let denRe = 1 - sRe
            let denIm = -sIm
            let numRe = 1 + sRe
            let numIm = sIm
            let (pzRe, pzIm) = complexDivide(numRe: numRe, numIm: numIm,
                                             denRe: denRe, denIm: denIm)

            // Digital biquad denominator from the conjugate pole pair (pz, conj):
            //   1 + a1 z⁻¹ + a2 z⁻², with a1 = −2·Re(pz), a2 = |pz|².
            let a1 = Float(-2 * pzRe)
            let a2 = Float(pzRe * pzRe + pzIm * pzIm)

            // Band-pass numerator per section: zeros at z = +1 and z = −1 give the
            // (1 − z⁻²) shape; one section carries both zeros, the other is
            // all-pole. To keep each section a proper band-pass biquad we put a
            // single (1 − z⁻²)-style numerator on each and normalize gain at the
            // band center afterwards.
            let b0u: Float = 1
            let b1u: Float = 0
            let b2u: Float = -1

            sections.append(Biquad(b0: b0u, b1: b1u, b2: b2u, a1: a1, a2: a2))
        }

        // Normalize total cascade gain to ~1 at the geometric center frequency.
        let centerHz = sqrt(Double(lowHz) * Double(highHz))
        let g = cascadeGain(sections, atHz: centerHz, fs: fsD)
        guard g > 1e-12 else { return sections }
        let scale = Float(1.0 / g)
        // Fold the scale into the first section's feed-forward coefficients.
        if let first = sections.first {
            sections[0] = Biquad(b0: first.b0 * scale,
                                 b1: first.b1 * scale,
                                 b2: first.b2 * scale,
                                 a1: first.a1, a2: first.a2)
        }
        return sections
    }

    /// Second-order IIR notch (band-stop) at `f0` with quality factor `q`.
    /// Standard RBJ cookbook notch, normalized so a0 == 1.
    ///
    /// Returns a pass-through biquad for a degenerate spec.
    static func iirNotch(f0: Float, q: Float, fs: Float) -> Biquad {
        guard fs > 0, f0 > 0, f0 < fs / 2, q > 0 else {
            return Biquad(b0: 1, b1: 0, b2: 0, a1: 0, a2: 0)
        }
        let w0 = 2 * Double.pi * Double(f0) / Double(fs)
        let cosw0 = cos(w0)
        let sinw0 = sin(w0)
        let alpha = sinw0 / (2 * Double(q))

        // RBJ notch:
        //   b0 = 1, b1 = −2cos w0, b2 = 1
        //   a0 = 1 + α, a1 = −2cos w0, a2 = 1 − α
        let a0 = 1 + alpha
        let b0 = 1.0 / a0
        let b1 = (-2 * cosw0) / a0
        let b2 = 1.0 / a0
        let a1 = (-2 * cosw0) / a0
        let a2 = (1 - alpha) / a0
        return Biquad(b0: Float(b0), b1: Float(b1), b2: Float(b2),
                      a1: Float(a1), a2: Float(a2))
    }

    // MARK: - Complex helpers

    private static func complexSqrt(re: Double, im: Double) -> (Double, Double) {
        let mag = sqrt(re * re + im * im)
        let sr = sqrt(max(0, (mag + re) / 2))
        var si = sqrt(max(0, (mag - re) / 2))
        if im < 0 { si = -si }
        return (sr, si)
    }

    private static func complexDivide(numRe: Double, numIm: Double,
                                      denRe: Double, denIm: Double) -> (Double, Double) {
        let den = denRe * denRe + denIm * denIm
        guard den != 0 else { return (0, 0) }
        return ((numRe * denRe + numIm * denIm) / den,
                (numIm * denRe - numRe * denIm) / den)
    }

    /// Magnitude response of a biquad cascade at frequency `hz`.
    private static func cascadeGain(_ sections: [Biquad], atHz hz: Double, fs: Double) -> Double {
        let w = 2 * Double.pi * hz / fs
        let cw = cos(w), sw = sin(w)
        // z⁻¹ = e^{−jw}, z⁻² = e^{−j2w}.
        let c1 = cos(-w),  s1 = sin(-w)
        let c2 = cos(-2 * w), s2 = sin(-2 * w)
        _ = (cw, sw)
        var gain = 1.0
        for s in sections {
            // Numerator N = b0 + b1 z⁻¹ + b2 z⁻².
            let nRe = Double(s.b0) + Double(s.b1) * c1 + Double(s.b2) * c2
            let nIm = Double(s.b1) * s1 + Double(s.b2) * s2
            // Denominator D = 1 + a1 z⁻¹ + a2 z⁻².
            let dRe = 1 + Double(s.a1) * c1 + Double(s.a2) * c2
            let dIm = Double(s.a1) * s1 + Double(s.a2) * s2
            let nMag = sqrt(nRe * nRe + nIm * nIm)
            let dMag = sqrt(dRe * dRe + dIm * dIm)
            guard dMag > 1e-15 else { return .greatestFiniteMagnitude }
            gain *= nMag / dMag
        }
        return gain
    }
}
