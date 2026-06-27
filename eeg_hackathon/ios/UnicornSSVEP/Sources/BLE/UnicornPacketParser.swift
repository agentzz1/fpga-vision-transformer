import Foundation

/// Parses raw Unicorn Hybrid Black BLE notifications into length-8 µV EEG samples.
///
/// Assumed 45-byte frame layout (per interface contract §C / `UnicornBLE`):
/// ```
/// offset  size  field
/// 0..1     2    header        0xC0 0x00            (VERIFIED)
/// 2        1    battery/status low nibble = battery (0..15)
/// 3..26    24   EEG           8 ch × 24-bit signed BIG-endian (VERIFIED)
/// 25..30   6    accel         3 × int16            (VERIFY: 25 vs 27 disputed)
/// 31..36   6    gyro          3 × int16            (VERIFY)
/// 37..38   2    reserved/status                    (VERIFY)
/// 39..42   4    counter       uint32 little-endian (VERIFIED)
/// 43..44   2    footer        0x0D 0x0A            (VERIFIED: reject if mismatch)
/// ```
/// Only the EEG block is exposed via `EEGSource`; the accel/gyro/counter/battery
/// trailer is skipped during decode (battery is available via `battery(_:)`).
struct UnicornPacketParser {

    /// Total length of one validated Unicorn frame, in bytes.
    static let frameLength = 45

    // MARK: - VERIFY constants (frame geometry & scale)
    //
    // The contract marks header/footer/EEG-offset/scale as VERIFIED, but they are
    // kept here as named constants so a firmware variant can be retargeted without
    // touching decode logic. Disputed accel/gyro offsets are NOT used here.

    /// Frame header bytes. VERIFIED.
    private static let header: [UInt8] = [0xC0, 0x00]
    /// Frame footer bytes. VERIFIED — frame is rejected if these do not match.
    private static let footer: [UInt8] = [0x0D, 0x0A]
    /// Byte offset of the first EEG byte. VERIFIED: EEG occupies bytes 3..26.
    private static let eegByteOffset = 3
    /// Number of EEG channels in a frame. VERIFIED.
    private static let eegChannelCount = 8
    /// Bytes per EEG channel (24-bit sample). VERIFIED.
    private static let bytesPerChannel = 3
    /// µV per raw count. VERIFIED ≈ 0.0894073 µV/LSB.
    private static let microvoltsPerCount: Double = 4_500_000.0 / 50_331_642.0
    /// Sign bit for a 24-bit two's-complement sample. VERIFIED.
    private static let signBit24: Int32 = 0x0080_0000
    /// Sign-extension mask for negative 24-bit samples. VERIFIED.
    private static let signExtend24: Int32 = ~0x00FF_FFFF // 0xFF00_0000

    // MARK: - Resync buffer

    /// Rolling buffer holding bytes that have not yet formed a complete, validated
    /// frame. Survives across `append` calls so frames split across notifications
    /// (or arriving with leading garbage after a reconnect) are reassembled.
    private var buffer: [UInt8] = []

    init() {
        // Reserve a little headroom to avoid early reallocations under streaming.
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
    ///
    /// Partial frames are tolerated: nothing is emitted until a full 45-byte,
    /// footer-validated frame is available.
    mutating func append(_ data: Data) -> [[Double]] {
        guard !data.isEmpty else { return [] }
        buffer.append(contentsOf: data)

        var samples: [[Double]] = []
        var index = 0
        let n = buffer.count

        while index + Self.frameLength <= n {
            // Require a valid header at the current position.
            guard buffer[index] == Self.header[0],
                  buffer[index + 1] == Self.header[1] else {
                // Not a header here — slide forward one byte to resync.
                index += 1
                continue
            }

            let frameSlice = buffer[index ..< index + Self.frameLength]
            let frame = Data(frameSlice)

            if let sample = Self.decodeFrame(frame) {
                samples.append(sample)
                index += Self.frameLength
            } else {
                // Header matched but footer failed: treat header as coincidental,
                // skip one byte and keep scanning.
                index += 1
            }
        }

        // Drop everything we consumed/skipped; keep the unparsed tail (a possible
        // partial frame straddling the next notification).
        if index > 0 {
            buffer.removeFirst(min(index, buffer.count))
        }

        // Guard against unbounded growth if a stream never resynchronizes:
        // retain at most one frame's worth of trailing bytes.
        if buffer.count > Self.frameLength * 8 {
            buffer.removeFirst(buffer.count - Self.frameLength)
        }

        return samples
    }

    // MARK: - Single-frame decode

    /// Decode exactly one candidate 45-byte frame into 8 µV channel values.
    ///
    /// Returns `nil` if the frame is the wrong length, or if the header/footer do
    /// not validate. EEG channels are decoded as 24-bit signed big-endian, sign-
    /// extended, then scaled to µV. Frame order == `EEGConfig.channelNames`.
    static func decodeFrame(_ frame: Data) -> [Double]? {
        guard frame.count == frameLength else { return nil }

        // Index into the Data via a stable base so non-zero startIndex slices work.
        let base = frame.startIndex

        // Validate header.
        guard frame[base] == header[0],
              frame[base + 1] == header[1] else { return nil }

        // Validate footer (last two bytes).
        guard frame[base + frameLength - 2] == footer[0],
              frame[base + frameLength - 1] == footer[1] else { return nil }

        var channels = [Double](repeating: 0, count: eegChannelCount)
        for ch in 0 ..< eegChannelCount {
            let off = base + eegByteOffset + ch * bytesPerChannel
            // 24-bit big-endian: most-significant byte first.
            let b0 = Int32(frame[off])
            let b1 = Int32(frame[off + 1])
            let b2 = Int32(frame[off + 2])
            var raw = (b0 << 16) | (b1 << 8) | b2

            // Sign-extend the 24-bit two's-complement value into Int32.
            if raw & signBit24 != 0 {
                raw |= signExtend24
            }

            channels[ch] = Double(raw) * microvoltsPerCount
        }
        return channels
    }

    /// Battery percentage from a validated frame: `100 * (frame[2] & 0x0F) / 15`.
    /// Returns `nil` if the frame is the wrong length.
    static func battery(_ frame: Data) -> Int? {
        guard frame.count == frameLength else { return nil }
        let base = frame.startIndex
        let nibble = Int(frame[base + 2] & 0x0F)
        return 100 * nibble / 15
    }
}
