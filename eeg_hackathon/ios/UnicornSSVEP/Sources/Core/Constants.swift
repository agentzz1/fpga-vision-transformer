//
//  Constants.swift
//  SSVEP2048
//
//  Single source of truth for app-wide constants (interface contract §B.1,
//  owner 4). Everything that needs the acquisition geometry, channel montage,
//  SSVEP band parameters, or the arrow/direction mapping reads it from here so
//  there is exactly ONE definition of each value across the target.
//
//  Hardware: g.tec Unicorn Hybrid Black — 8 EEG channels @ 250 Hz. The montage
//  order below is the canonical Unicorn order and matches the Python pipeline's
//  `eeg_common.CHANNEL_NAMES` (Fz,C3,Cz,C4,Pz,PO7,Oz,PO8). Do NOT reorder: the
//  index into this list is the channel axis index everywhere downstream.
//

import Foundation

// MARK: - Direction (contract §B.1)

/// A 2048 move direction. Index identity with `EEGConfig.ARROWS` is load-bearing
/// (contract §D.4): `ARROWS[Decision.target]` is the intended move.
enum Direction: Int, CaseIterable, Equatable {
    case up = 0
    case down = 1
    case left = 2
    case right = 3
}

// MARK: - EEGConfig (contract §B.1)

/// App-wide EEG / SSVEP configuration. All values are compile-time constants.
enum EEGConfig {

    // MARK: Acquisition geometry

    /// Sampling rate in Hz. The Unicorn Hybrid Black streams at a fixed 250 Hz.
    static let fs: Double = 250

    /// Number of EEG channels per scan.
    static let channelCount: Int = 8

    /// Canonical Unicorn montage order. INDEX in this list == channel axis index
    /// everywhere downstream (ring buffer rows, parser output, decoder input).
    /// Matches the Python pipeline's `eeg_common.CHANNEL_NAMES`.
    static let channelNames: [String] = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]

    /// Occipital / parieto-occipital channel indices used for SSVEP decoding.
    /// These are the informative channels for visual stimulation: PO7(5), Oz(6),
    /// PO8(7), plus Pz(4) for a little extra parietal coverage.
    static let occipitalIndices: [Int] = [4, 5, 6, 7]

    // MARK: Decode window

    /// Decode window length in samples = 2 s @ fs (500 @ 250 Hz). This is also
    /// the default ring-buffer depth.
    static let windowSamples: Int = 500

    /// Number of harmonics (sin/cos pairs) per CCA reference bank.
    static let harmonics: Int = 3

    // MARK: Mains

    /// Mains hum frequency to notch (Hz). 50 in EU/UK/most of the world; set to
    /// 60 for North America. VERIFY against the deployment region.
    static let mainsHz: Double = 50

    // MARK: Arrows / targets

    /// The four flicker directions, index-aligned to the on-screen targets and to
    /// `Decision.target`. `ARROWS[i]` is the move committed when target `i` wins.
    /// Order MUST match `FlickerView`'s tile layout (up, down, left, right).
    static let ARROWS: [Direction] = [.up, .down, .left, .right]

    /// Fallback flicker frequencies (Hz), index-aligned to `ARROWS`, used when a
    /// display refresh cannot realize four well-separated frame-locked targets
    /// (see `achievableFreqs(refresh:)`). Sorted descending to mirror the
    /// realized-target ordering. Distinct values so two arrows never collide.
    static let defaultFrequencies: [Double] = [15.0, 12.0, 10.0, 8.57]

    // MARK: BLE

    /// Advertised-name prefix used to filter discovered Unicorn peripherals.
    /// The Unicorn Hybrid Black advertises as "UN-<serial>". VERIFY on device.
    static let advertisedNamePrefix: String = "UN-"
}
