import Foundation

/// Thread-safe, fixed-size ring buffer of multichannel EEG samples.
///
/// Stores the most recent `capacitySamples` scans, each scan being a
/// length-`channelCount` row of µV values in `EEGConfig.channelNames` order
/// (see contract §D.1). The producer is the `EEGSource` stream (one `push`
/// per 250 Hz scan, off the main thread); the consumer is the DSP/decode
/// pipeline which reads sliding windows via `latestWindow(...)`.
///
/// Windows are returned **channel-major** `[Float]`: a length-`channelCount * n`
/// array where each channel occupies a contiguous `n`-element block
/// (`[ch0_n0…ch0_n(n-1), ch1_n0…]`). When viewed as an `n × channelCount`
/// matrix this is LAPACK column-major, matching what `CCA`/`FBCCA` expect
/// (contract §D.2).
///
/// Thread safety is provided by an internal `os_unfair_lock`. All public
/// methods take the lock for the duration of their access, so concurrent
/// `push` and `latestWindow` calls are safe. The lock is uncontended in the
/// common single-producer/single-consumer case and far cheaper than a
/// dispatch queue per sample at 250 Hz.
final class EEGRingBuffer {

    // MARK: Immutable configuration

    /// Number of channels in every sample (length of each pushed row).
    let channelCount: Int

    /// Maximum number of samples retained (the ring's physical size).
    let capacitySamples: Int

    // MARK: Storage

    /// Flat backing store laid out **sample-major**: row `r` occupies
    /// `storage[r*channelCount ..< (r+1)*channelCount]`. Sample-major makes
    /// `push` a single contiguous copy; the channel-major transpose happens
    /// lazily in `latestWindow`. Indexed as a ring via `head`/`filled`.
    private var storage: [Double]

    /// Index of the slot that will receive the *next* pushed sample.
    private var head: Int = 0

    /// Number of valid samples currently stored (0...capacitySamples).
    private var filled: Int = 0

    /// Guards all mutable state. `os_unfair_lock` is allocated on the heap so
    /// its address is stable for the lifetime of the buffer (required — the
    /// struct must not be copied while locking).
    private let lock: UnsafeMutablePointer<os_unfair_lock>

    // MARK: Init / deinit

    /// - Parameters:
    ///   - channelCount: samples per scan; defaults to `EEGConfig.channelCount` (8).
    ///   - capacitySamples: ring depth; defaults to `EEGConfig.windowSamples` (500).
    init(channelCount: Int = EEGConfig.channelCount,
         capacitySamples: Int = EEGConfig.windowSamples) {
        precondition(channelCount > 0, "channelCount must be positive")
        precondition(capacitySamples > 0, "capacitySamples must be positive")
        self.channelCount = channelCount
        self.capacitySamples = capacitySamples
        self.storage = [Double](repeating: 0, count: channelCount * capacitySamples)
        self.lock = UnsafeMutablePointer<os_unfair_lock>.allocate(capacity: 1)
        self.lock.initialize(to: os_unfair_lock())
    }

    deinit {
        lock.deinitialize(count: 1)
        lock.deallocate()
    }

    // MARK: Mutation

    /// Append one length-`channelCount` µV sample. Thread-safe.
    ///
    /// Samples whose length does not match `channelCount` are dropped (with an
    /// assertion in debug builds) so a malformed BLE frame cannot corrupt the
    /// ring layout. Overwrites the oldest sample once the ring is full.
    func push(_ sample: [Double]) {
        assert(sample.count == channelCount,
               "EEGRingBuffer.push expected \(channelCount) channels, got \(sample.count)")
        guard sample.count == channelCount else { return }

        os_unfair_lock_lock(lock)
        defer { os_unfair_lock_unlock(lock) }

        let base = head * channelCount
        for c in 0..<channelCount {
            storage[base + c] = sample[c]
        }
        head = (head + 1) % capacitySamples
        if filled < capacitySamples { filled += 1 }
    }

    // MARK: Reads

    /// Latest `n` samples as a channel-major `Float` window
    /// (length `channelCount * n`), all channels in storage order.
    ///
    /// Returns `nil` if fewer than `n` samples are available, or if `n` is
    /// non-positive or exceeds `capacitySamples`.
    func latestWindow(n: Int) -> [Float]? {
        // Reuse the subset path with the full, in-order channel list.
        latestWindow(n: n, channels: Array(0..<channelCount))
    }

    /// Channel-major `Float` window for an explicit subset of channel indices
    /// (length `channels.count * n`), e.g. the occipital subset
    /// `EEGConfig.occipitalIndices` before FBCCA (contract §D.3).
    ///
    /// Channels appear in the order given by `channels`. Returns `nil` if fewer
    /// than `n` samples are available, `n` is out of range, or any requested
    /// channel index is out of bounds.
    func latestWindow(n: Int, channels: [Int]) -> [Float]? {
        guard n > 0, n <= capacitySamples else { return nil }
        for c in channels where c < 0 || c >= channelCount { return nil }

        os_unfair_lock_lock(lock)
        defer { os_unfair_lock_unlock(lock) }

        guard filled >= n else { return nil }

        // Oldest of the requested window sits `n` slots behind `head`,
        // wrapped into the ring.
        let start = ((head - n) % capacitySamples + capacitySamples) % capacitySamples
        let outChannels = channels.count
        var out = [Float](repeating: 0, count: outChannels * n)

        // Transpose sample-major storage → channel-major output.
        for i in 0..<n {
            let row = (start + i) % capacitySamples
            let rowBase = row * channelCount
            for (outC, srcC) in channels.enumerated() {
                // Channel block `outC` is contiguous; sample `i` is offset `i`.
                out[outC * n + i] = Float(storage[rowBase + srcC])
            }
        }
        return out
    }

    /// Number of valid samples currently stored (0...capacitySamples). Thread-safe.
    var count: Int {
        os_unfair_lock_lock(lock)
        defer { os_unfair_lock_unlock(lock) }
        return filled
    }

    /// Discard all stored samples. Thread-safe. Capacity is preserved.
    func reset() {
        os_unfair_lock_lock(lock)
        defer { os_unfair_lock_unlock(lock) }
        head = 0
        filled = 0
        // No need to zero `storage`; `filled`/`head` make stale data unreachable.
    }
}
