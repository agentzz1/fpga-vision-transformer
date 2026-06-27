//
//  EEGSource.swift
//  SSVEP2048
//
//  The source-agnostic EEG input abstraction (interface contract §B.3, owner 3).
//  Both `UnicornBLEManager` (live headset over CoreBluetooth) and
//  `SyntheticEEGSource` (headset-free simulator/demo) conform to this protocol so
//  `AppViewModel` can swap them transparently.
//

import Foundation

// MARK: - EEGConnectionState (contract §B.3)

/// Lifecycle state of an `EEGSource`. Emitted on `states()` and mirrored into the
/// UI. `Equatable` so sources can de-dup state changes (`didSet` guards) and the
/// status bar can switch over it.
enum EEGConnectionState: Equatable {
    /// Idle: not connected and not trying to connect.
    case disconnected
    /// Powering on / scanning / connecting / discovering (BLE), or the brief
    /// synthetic warm-up.
    case connecting
    /// Live: samples are flowing.
    case streaming
    /// A terminal-ish error with a human-readable reason (permission denied,
    /// BT off, discovery failure, link drop, …). The source may still attempt
    /// to reconnect depending on its policy.
    case failed(String)
}

// MARK: - EEGSource (contract §B.3)

/// A streaming multichannel EEG source.
///
/// Sample contract: `samples()` yields one length-`EEGConfig.channelCount` row of
/// µV values per acquisition scan, in `channelNames` order, at `sampleRate` Hz.
/// `states()` yields connection-state changes and MUST emit the current state
/// immediately on subscribe so consumers never miss the initial value.
///
/// Class-bound (`AnyObject`) so `AppViewModel` can hold it by reference and so the
/// concrete sources can manage CoreBluetooth / timer state internally.
protocol EEGSource: AnyObject {

    /// Channel order of each emitted sample. Matches `EEGConfig.channelNames`.
    var channelNames: [String] { get }

    /// Acquisition rate in Hz. Matches `EEGConfig.fs` for the Unicorn.
    var sampleRate: Double { get }

    /// Current connection state (a consistent, possibly slightly stale snapshot).
    var connectionState: EEGConnectionState { get }

    /// Continuous stream of length-`channelCount` µV samples, one per scan.
    func samples() -> AsyncStream<[Double]>

    /// Stream of connection-state changes; emits the current value on subscribe.
    func states() -> AsyncStream<EEGConnectionState>

    /// Bring the source online (scan/connect for BLE; start the timer for synthetic).
    func start()

    /// Take the source offline and release resources.
    func stop()
}
