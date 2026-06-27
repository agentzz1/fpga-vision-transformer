import Foundation

/// Parses raw Unicorn Hybrid Black notifications into length-8 µV EEG samples.
///
/// ⚠️ ALL framing constants below are UNVERIFIED (`VERIFY`). The stock Unicorn
/// streams over Bluetooth Classic (RFCOMM/SPP), not BLE/GATT, so a GATT-delivered
/// 45-byte frame with these exact fields cannot be trusted until confirmed against
/// (a) a real-device nRF Connect capture and (b) BrainFlow `unicorn_board.cpp`.
/// See README.md → "BLE protocol VERIFY checklist". Constants live in `UnicornBLE`
/// (BLE/BLEConstants.swift) so a firmware variant can be retargeted in one place.
///
/// ASSUMED 45-byte frame layout (VERIFY every row):
/// ```
/// offset  size  field
/// 0..1     2    header        0xC0 0x00            (VERIFY)
/// 2..25   24    EEG           8 ch × 24-bit signed (VERIFY endianness!)
/// 26      1     battery/status                     (VERIFY)
/// 27..32   6    accel         3 × int16            (VERIFY)
/// 33..38   6    gyro          3 × int16            (VERIFY)
/// 39..42   4    counter       uint32               (VERIFY)
/// 43..44   2    footer        0x0D 0x0A            (VERIFY: reject if mismatch)
/// ```
/// NOTE: the EEG block (offset 2, 8×3 = 24 bytes) ends at index 25 inclusive, so
/// the trailer must start at 26 — the offsets above are contiguous and
/// non-overlapping. Only the EEG block is exposed via `EEGSource`; the trailer is
/// skipped during decode (battery is available via `battery(_:)`).
///
/// The exact EEG byte offset is taken from `UnicornBLE.eegByteOffset`; if a real
/// capture shows EEG starting at byte 3 (with a status byte at offset 2 first),
/// change that ONE constant in BLEConstants.swift and the parser follows.
struct UnicornPacketParser {

    // MARK: - Frame geometry (all VERIFY; sourced from UnicornBLE)

    /// Total length of one validated Unicorn frame, in bytes. VERIFY.
    static let frameLength = UnicornBLE.frameLength

    private static let header: [UInt8] = UnicornBLE.header        // VERIFY
    private static let footer: [UInt8] = UnicornBLE.footer        // VERIFY
    private static let eegByteOffset = UnicornBLE.eegByteOffset   // VERIFY
    private static let eegChannelCount = UnicornBLE.eegChannelCount // VERIFY
    private static let bytesPerChannel = UnicornBLE.bytesPerChannel // VERIFY

    /// Byte order of each 24-bit EEG sample. VERIFY. Single named constant so the
    /// whole parser flips with one edit once a capture confirms the order.
    /// Unicorn is *very likely* little-endian (BrainFlow decodes LE).
    private static let littleEndian = UnicornBLE.eegLittleEndian  // VERIFY

    /// µV per raw count. VERIFY ≈ 0.0894073 µV/LSB.
    private static let microvoltsPerCount = UnicornBLE.microvoltsPerCount // VERIFY

    /// Sign bit for a 24-bit two's-complement sample.
    private static let signBit24: Int32 = 0x0080_0000
    /// Sign-extension mask for negative 24-bit samples (0xFF00_0000).
    private static let signExtend24: Int32 = ~0x00FF_FFFF

    // MARK: - Resync buffer

    /// Rolling buffer holding bytes that have not yet formed a complete, validated
    /// frame. Survives across `append` calls so frames split across notifications
    /// (or arriving with leading garbage after a reconnect) are reassembled.
    private var buffer: [UInt8] = []

    init() {
        buffer.reserveCapacity(Self.frameLength * 4)
    }

    // MARK: - Streaming API

    /// Append raw notification bytes and emit every complete, validated frame as a
    /// length-8 µV sample (channel order == `EEGConfig.channelNames`).
    ///
    /// Resync strategy: scan for the header; if the bytes at the header position
    /// form a frame whose footer validates, emit it and advance. If a candidate
    /// header is found but the frame is incomplete, keep it buffered for the next
    /// call. If a header's frame fails footer validation, drop a single byte and
    /// re-scan so a spurious header inside the trailer cannot wedge the stream.
    mutating func append(_ data: Data) -> [[Double]] {
        guard !data.isEmpty else { return [] }
        buffer.append(contentsOf: data)

        var samples: [[Double]] = []
        var index = 0
        let n = buffer.count

        while index + Self.frameLength <= n {
            guard buffer[index] == Self.header[0],
                  buffer[index + 1] == Self.header[1] else {
                index += 1
                continue
            }

            let frameSlice = buffer[index ..< index + Self.frameLength]
            let frame = Data(frameSlice)

            if let sample = Self.decodeFrame(frame) {
                samples.append(sample)
                index += Self.frameLength
            } else {
                index += 1
            }
        }

        if index > 0 {
            buffer.removeFirst(min(index, buffer.count))
        }

        // Guard against unbounded growth if a stream never resynchronizes.
        if buffer.count > Self.frameLength * 8 {
            buffer.removeFirst(buffer.count - Self.frameLength)
        }

        return samples
    }

    // MARK: - Single-frame decode

    /// Decode exactly one candidate 45-byte frame into 8 µV channel values.
    ///
    /// Returns `nil` if the frame is the wrong length, or if the header/footer do
    /// not validate. EEG channels are decoded as 24-bit signed (byte order per
    /// `littleEndian`), sign-extended, then scaled to µV. Frame order ==
    /// `EEGConfig.channelNames`.
    static func decodeFrame(_ frame: Data) -> [Double]? {
        guard frame.count == frameLength else { return nil }
        let base = frame.startIndex

        guard frame[base] == header[0],
              frame[base + 1] == header[1] else { return nil }
        guard frame[base + frameLength - 2] == footer[0],
              frame[base + frameLength - 1] == footer[1] else { return nil }

        var channels = [Double](repeating: 0, count: eegChannelCount)
        for ch in 0 ..< eegChannelCount {
            let off = base + eegByteOffset + ch * bytesPerChannel
            let bA = Int32(frame[off])
            let bB = Int32(frame[off + 1])
            let bC = Int32(frame[off + 2])

            // Assemble 24-bit value per the (VERIFY) byte order.
            var raw: Int32
            if littleEndian {
                raw = (bC << 16) | (bB << 8) | bA   // b0 is least significant
            } else {
                raw = (bA << 16) | (bB << 8) | bC   // b0 is most significant
            }

            // Sign-extend the 24-bit two's-complement value into Int32.
            if raw & signBit24 != 0 {
                raw |= signExtend24
            }

            channels[ch] = Double(raw) * microvoltsPerCount
        }
        return channels
    }

    /// Battery percentage from a validated frame, read from the byte immediately
    /// after the EEG block. VERIFY: offset and encoding are unconfirmed.
    /// Returns `nil` if the frame is the wrong length.
    static func battery(_ frame: Data) -> Int? {
        guard frame.count == frameLength else { return nil }
        let base = frame.startIndex
        let statusOffset = eegByteOffset + eegChannelCount * bytesPerChannel // VERIFY
        guard statusOffset < frameLength else { return nil }
        let nibble = Int(frame[base + statusOffset] & 0x0F)
        return 100 * nibble / 15
    }
}
