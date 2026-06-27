//
//  Decision.swift
//  SSVEP2048
//
//  Decode-side value types shared by the renderer and the decoder (interface
//  contract §B.2, owner 7):
//    - `FlickerTarget`: one on-screen flicker stimulus (frequency + the integer
//      frame schedule the refresh-locked renderer uses).
//    - `Decision`: the FBCCA decoder's per-window output (winning target, raw
//      per-target scores, and a confidence ratio).
//

import Foundation

// MARK: - FlickerTarget (contract §B.2)

/// One SSVEP flicker stimulus.
///
/// The renderer (`FlickerView`) drives each tile from an integer frame counter
/// (never `Date()`), toggling every `framesPerHalfPeriod` frames so the realized
/// frequency is exactly `refresh / (2 * framesPerHalfPeriod)`. The decoder
/// (`FBCCA`) builds its reference bank from `frequency`. The two MUST come from
/// the same `achievableFreqs(refresh:)` call so stimulus and references agree.
struct FlickerTarget: Equatable {

    /// Exact realized flicker frequency in Hz.
    let frequency: Double

    /// Frames the tile holds each on/off state (the integer half-period count
    /// `k`). One full period is `2 * k` frames. `0` means "not frame-locked"
    /// (used only by the defaultFrequencies fallback, which the renderer guards
    /// against by clamping to at least 1).
    let framesPerHalfPeriod: Int

    /// Optional starting phase in radians (renderer-side; default 0). Carried for
    /// completeness; the current square-wave renderer starts every tile ON.
    let phase: Double

    init(frequency: Double, framesPerHalfPeriod: Int, phase: Double = 0) {
        self.frequency = frequency
        self.framesPerHalfPeriod = framesPerHalfPeriod
        self.phase = phase
    }
}

// MARK: - Decision (contract §B.2)

/// The decoder's output for a single decode window.
struct Decision: Equatable {

    /// Index of the winning target, index-aligned to `EEGConfig.ARROWS`
    /// (`ARROWS[target]` is the move). `-1` is the "no decision" sentinel
    /// (empty/degenerate window, no targets).
    let target: Int

    /// Per-target combined scores (`Σ_m w(m)·ρ²_{m,t}`), index-aligned to targets.
    /// Empty for a "no decision".
    let scores: [Float]

    /// Confidence = best score / second-best score, clamped to `>= 1`. `1` means
    /// no separation (or a single target); higher means a cleaner winner. The
    /// gate (`SSVEPController`) thresholds this.
    let confidence: Float

    init(target: Int, scores: [Float], confidence: Float) {
        self.target = target
        self.scores = scores
        self.confidence = confidence
    }
}
