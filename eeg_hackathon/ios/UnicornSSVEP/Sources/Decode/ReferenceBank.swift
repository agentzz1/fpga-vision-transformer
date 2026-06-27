//
//  ReferenceBank.swift
//  SSVEP2048
//
//  Harmonic sin/cos reference matrices for CCA-based SSVEP decoding (interface
//  contract §B.10, owner 6).
//
//  For a candidate stimulus frequency f, the CCA reference is the matrix
//  Y ∈ ℝ^{N × 2H} whose columns are, for each harmonic h = 1..H:
//      sin(2π·h·f·t),  cos(2π·h·f·t),   t = n/fs, n = 0..N-1
//  This is the classic Lin/Bin/Chen SSVEP reference set. `FBCCA` correlates each
//  sub-band-filtered EEG window against this bank via `CCA`.
//
//  Storage is column-major Float (each column a contiguous N-block), matching the
//  LAPACK layout `CCA.canonicalCorrelationMax` consumes directly.
//

import Foundation

// MARK: - ReferenceBank (contract §B.10)

/// A sin/cos harmonic reference matrix for one stimulus frequency.
struct ReferenceBank {

    /// Column-major samples × columns matrix, length `N * cols`. Column `c`
    /// occupies `Y[c*N ..< (c+1)*N]`.
    let Y: [Float]

    /// Number of rows (samples) — must equal the decode window length.
    let N: Int

    /// Number of columns == `2 * harmonics` (a sin and a cos per harmonic).
    let cols: Int
}

// MARK: - makeReferenceBank (contract §B.10)

/// Build the harmonic reference bank for `freq` (Hz).
///
/// - Parameters:
///   - freq: stimulus fundamental frequency in Hz.
///   - harmonics: number of harmonics H (each contributes a sin and a cos column).
///   - fs: sampling rate in Hz.
///   - N: window length in samples (rows of Y).
/// - Returns: a `ReferenceBank` with `cols == 2*harmonics`, or an empty bank
///   (cols 0) for a degenerate spec.
func makeReferenceBank(freq: Double, harmonics: Int, fs: Double, N: Int) -> ReferenceBank {
    let H = max(0, harmonics)
    let cols = 2 * H
    guard N > 0, fs > 0, H > 0, freq > 0 else {
        return ReferenceBank(Y: [], N: max(0, N), cols: 0)
    }

    var Y = [Float](repeating: 0, count: N * cols)
    let twoPi = 2.0 * Double.pi

    // Column layout (column-major): for harmonic h (1-based), columns
    //   2*(h-1)   = sin(2π·h·f·t)
    //   2*(h-1)+1 = cos(2π·h·f·t)
    for h in 1...H {
        let omega = twoPi * Double(h) * freq / fs   // radians per sample
        let sinCol = 2 * (h - 1)
        let cosCol = sinCol + 1
        let sinBase = sinCol * N
        let cosBase = cosCol * N
        for n in 0..<N {
            let phase = omega * Double(n)
            Y[sinBase + n] = Float(sin(phase))
            Y[cosBase + n] = Float(cos(phase))
        }
    }

    return ReferenceBank(Y: Y, N: N, cols: cols)
}
