//
//  BLEConstants.swift
//  SSVEP2048
//
//  Central registry of Unicorn BLE identifiers, commands, and frame constants
//  (interface contract §C, owner 1).
//
//  ⚠️ ENDIANNESS / PROTOCOL WARNING — almost everything in this file is
//  UNVERIFIED and marked `VERIFY`. The stock g.tec Unicorn Hybrid Black streams
//  over Bluetooth *Classic* (RFCOMM/SPP), which CoreBluetooth (BLE/GATT) cannot
//  use. The GATT service/characteristic UUIDs, the START/STOP command bytes, and
//  the on-air 45-byte frame layout below are PLACEHOLDERS that MUST be confirmed
//  against (a) a real-device nRF Connect capture and (b) BrainFlow's
//  `unicorn_board.cpp`. See README.md → "BLE protocol VERIFY checklist".
//
//  Reserve the `VERIFIED` tag ONLY for a value cross-checked against a cited
//  capture or vendor doc; cite the source inline when you promote one.
//

import Foundation
import CoreBluetooth

/// Unicorn Hybrid Black BLE identifiers, commands, and frame geometry.
enum UnicornBLE {

    // MARK: - Discovery

    /// Advertised-name prefix. The Unicorn advertises "UN-<serial>".
    /// VERIFY: confirm exact prefix/casing with nRF Connect.
    static let advertisedNamePrefix: String = EEGConfig.advertisedNamePrefix

    // MARK: - GATT UUIDs (ALL VERIFY — placeholders)
    //
    // These Nordic UART Service (NUS) UUIDs are a common default for vendors that
    // tunnel a serial protocol over BLE. They are a GUESS for the Unicorn and are
    // very likely wrong. The BLE manager also falls back to capability-based
    // matching (notify/write properties) so a UUID swap needs no code change.

    /// Primary data/control service. VERIFY (placeholder: Nordic UART Service).
    static let serviceUUID = CBUUID(string: "6E400001-B5A3-F393-E0A9-E50E24DCCA9E")

    /// Notify (TX → host) characteristic carrying EEG frames.
    /// VERIFY (placeholder: NUS TX).
    static let notifyCharUUID = CBUUID(string: "6E400003-B5A3-F393-E0A9-E50E24DCCA9E")

    /// Write (host → device) characteristic for START/STOP commands.
    /// VERIFY (placeholder: NUS RX).
    static let commandCharUUID = CBUUID(string: "6E400002-B5A3-F393-E0A9-E50E24DCCA9E")

    // MARK: - Commands (VERIFY)
    //
    // BrainFlow issues a START acquisition command and a STOP command to the
    // Unicorn. The exact byte sequences over a BLE characteristic are UNKNOWN
    // here and MUST be taken from `unicorn_board.cpp` / a capture.

    /// Begin-acquisition command bytes. VERIFY against BrainFlow `unicorn_board.cpp`.
    static let startCommand: [UInt8] = [0x61, 0x7C, 0x87]

    /// Stop-acquisition command bytes. VERIFY against BrainFlow `unicorn_board.cpp`.
    static let stopCommand: [UInt8] = [0x63, 0x5C, 0xC5]

    // MARK: - Frame geometry (VERIFY)
    //
    // Single source of truth for the parser. The 45-byte framing, header/footer,
    // EEG offset, and scale are ALL unconfirmed for a BLE/GATT transport.

    /// Total validated frame length in bytes. VERIFY.
    static let frameLength: Int = 45

    /// Frame header bytes. VERIFY.
    static let header: [UInt8] = [0xC0, 0x00]

    /// Frame footer bytes. VERIFY (frame rejected if these do not match).
    static let footer: [UInt8] = [0x0D, 0x0A]

    /// Byte offset of the first EEG byte within a frame. VERIFY.
    static let eegByteOffset: Int = 2

    /// Number of EEG channels per frame. VERIFY (Unicorn has 8 EEG channels).
    static let eegChannelCount: Int = EEGConfig.channelCount

    /// Bytes per EEG channel sample (24-bit). VERIFY.
    static let bytesPerChannel: Int = 3

    /// Byte order of each 24-bit EEG sample.
    ///
    /// `true`  → little-endian: raw = b0 | b1<<8 | b2<<16
    /// `false` → big-endian:    raw = b0<<16 | b1<<8 | b2
    ///
    /// VERIFY. The Unicorn is *very likely* little-endian (BrainFlow decodes the
    /// SDK byte stream LE), but this has NOT been confirmed against a capture for
    /// this transport. Flip this one constant once verified.
    static let eegLittleEndian: Bool = true

    /// µV per raw count. VERIFY. The frequently-quoted Unicorn scale is
    /// 4500000 / 50331642 ≈ 0.0894073 µV/LSB; confirm against the SDK/datasheet.
    static let microvoltsPerCount: Double = 4_500_000.0 / 50_331_642.0

    // MARK: - Notes

    /// Human-readable reminder surfaced in logs/UI if anyone wires it up.
    static let verifyNote =
        "Unicorn BLE constants are UNVERIFIED placeholders. The stock device uses " +
        "Bluetooth Classic (RFCOMM/SPP), not GATT. Confirm UUIDs, commands, and " +
        "the 45-byte frame layout with nRF Connect + BrainFlow unicorn_board.cpp " +
        "before trusting live data."
}
