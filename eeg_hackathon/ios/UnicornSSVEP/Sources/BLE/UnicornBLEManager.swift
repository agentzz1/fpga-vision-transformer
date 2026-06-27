//
//  UnicornBLEManager.swift
//  SSVEP2048
//
//  CoreBluetooth central that scans for Unicorn Hybrid Black headsets (advertised
//  name prefix "UN-"), connects, discovers the data (notify) and control (write)
//  characteristics, subscribes to notifications, writes the START command, and
//  republishes reassembled 45-byte frames as length-8 µV samples.
//
//  Conforms to `EEGSource` (see BLE/EEGSource.swift). Frame parsing is delegated
//  to `UnicornPacketParser` (see BLE/UnicornPacketParser.swift). BLE identifiers
//  and commands come from `UnicornBLE` (see BLE/BLEConstants.swift).
//
//  IMPORTANT: The stock Unicorn streams over Bluetooth Classic RFCOMM/SPP, which
//  CoreBluetooth cannot use. The GATT service/characteristic UUIDs in `UnicornBLE`
//  are marked VERIFY and MUST be confirmed against the actual firmware (e.g. with
//  nRF Connect). This manager is written so that substituting those UUIDs requires
//  no API changes here.
//

import Foundation
import CoreBluetooth

/// `EEGSource` implementation backed by a live Unicorn headset over CoreBluetooth.
///
/// Threading model (per contract §D.5):
///   - All CoreBluetooth work happens on a private serial queue (`bleQueue`).
///   - `samples()` / `states()` expose `AsyncStream`s; their continuations are
///     fed from the BLE queue. Consumers may iterate from any actor/task.
///   - The class holds no UI state and never touches the main actor itself.
final class UnicornBLEManager: NSObject, EEGSource {

    // MARK: - EEGSource conformance (read-only surface)

    /// Channel order matches `EEGConfig.channelNames` (Fz,C3,Cz,C4,Pz,PO7,Oz,PO8).
    let channelNames: [String] = EEGConfig.channelNames

    /// Fixed 250 Hz acquisition rate of the Unicorn Hybrid Black.
    let sampleRate: Double = EEGConfig.fs

    /// Current connection state. Reads are serialized through `bleQueue`; the
    /// stored value is updated only from that queue, so cross-thread reads observe
    /// a consistent (if possibly slightly stale) snapshot.
    var connectionState: EEGConnectionState {
        bleQueue.sync { _connectionState }
    }

    // MARK: - Configuration

    /// Advertised-name prefix used to filter discovered peripherals.
    private let namePrefix: String

    /// Filters advertised peripherals by name prefix. Unicorn advertises "UN-".
    init(namePrefix: String = "UN-") {
        self.namePrefix = namePrefix
        super.init()
    }

    // MARK: - CoreBluetooth state

    /// Dedicated serial queue. Every CBCentralManager / CBPeripheral delegate
    /// callback is delivered here, and all mutable state below is touched only here.
    private let bleQueue = DispatchQueue(label: "com.ssvep2048.ble", qos: .userInitiated)

    private var central: CBCentralManager?
    private var peripheral: CBPeripheral?
    private var notifyCharacteristic: CBCharacteristic?
    private var commandCharacteristic: CBCharacteristic?

    /// Frame reassembler. Mutated only on `bleQueue`.
    private var parser = UnicornPacketParser()

    /// Backing store for `connectionState`; mutated only on `bleQueue`.
    private var _connectionState: EEGConnectionState = .disconnected {
        didSet {
            guard oldValue != _connectionState else { return }
            statesContinuation?.yield(_connectionState)
        }
    }

    // MARK: - Lifecycle flags (bleQueue only)

    /// True between `start()` and `stop()`. Drives whether disconnects trigger a
    /// reconnect or are treated as a clean teardown.
    private var isActive = false

    /// True once the START command has been written, so we don't write it twice
    /// (e.g. if discovery callbacks fire more than once).
    private var didSendStart = false

    // MARK: - Reconnect / backoff (bleQueue only)

    /// Current reconnect attempt count, used for capped exponential backoff.
    private var reconnectAttempt = 0
    /// Base delay for backoff (seconds).
    private let reconnectBaseDelay: TimeInterval = 1.0
    /// Maximum backoff delay (seconds).
    private let reconnectMaxDelay: TimeInterval = 30.0
    /// In-flight scheduled reconnect, so we can cancel it on `stop()`.
    private var reconnectWorkItem: DispatchWorkItem?

    // MARK: - AsyncStream plumbing

    private var samplesContinuation: AsyncStream<[Double]>.Continuation?
    private var statesContinuation: AsyncStream<EEGConnectionState>.Continuation?

    /// Continuous stream of length-8 µV samples, one element per 250 Hz scan.
    ///
    /// A single long-lived stream is created lazily and shared. If a previous
    /// stream was finished, a fresh one is vended. Buffering is unbounded so a
    /// slow consumer never drops samples silently — callers that fall behind are
    /// responsible for windowing/decimation downstream.
    func samples() -> AsyncStream<[Double]> {
        AsyncStream(bufferingPolicy: .unbounded) { continuation in
            bleQueue.async { [weak self] in
                guard let self else { continuation.finish(); return }
                // Replace any prior continuation; only the latest consumer is fed.
                self.samplesContinuation?.finish()
                self.samplesContinuation = continuation
                continuation.onTermination = { [weak self] _ in
                    self?.bleQueue.async {
                        // Only clear if we still own this continuation.
                        self?.samplesContinuation = nil
                    }
                }
            }
        }
    }

    /// Stream of connection-state changes. The current state is emitted
    /// immediately on subscribe so consumers don't miss the initial value.
    func states() -> AsyncStream<EEGConnectionState> {
        AsyncStream(bufferingPolicy: .bufferingNewest(8)) { continuation in
            bleQueue.async { [weak self] in
                guard let self else { continuation.finish(); return }
                self.statesContinuation?.finish()
                self.statesContinuation = continuation
                continuation.onTermination = { [weak self] _ in
                    self?.bleQueue.async {
                        self?.statesContinuation = nil
                    }
                }
                // Emit current state on subscribe (contract: initial value).
                continuation.yield(self._connectionState)
            }
        }
    }

    // MARK: - EEGSource control

    /// Power-on → scan → connect → discover → notify → write START.
    ///
    /// Idempotent-ish: calling `start()` while already active is a no-op beyond
    /// ensuring the central exists. The actual scan is kicked off (or deferred)
    /// from `centralManagerDidUpdateState`.
    func start() {
        bleQueue.async { [weak self] in
            guard let self else { return }
            self.isActive = true
            self.reconnectAttempt = 0

            if self.central == nil {
                // Creating the central triggers `centralManagerDidUpdateState`,
                // where scanning actually begins once Bluetooth is powered on.
                self.central = CBCentralManager(delegate: self, queue: self.bleQueue)
            } else if self.central?.state == .poweredOn {
                self.beginScan()
            }
        }
    }

    /// Write STOP (best effort), cancel the connection, and tear everything down.
    func stop() {
        bleQueue.async { [weak self] in
            guard let self else { return }
            self.isActive = false
            self.cancelPendingReconnect()

            // Best-effort STOP command before dropping the link.
            if let p = self.peripheral, let cmd = self.commandCharacteristic {
                let stop = Data(UnicornBLE.stopCommand)
                let type: CBCharacteristicWriteType =
                    cmd.properties.contains(.write) ? .withResponse : .withoutResponse
                p.writeValue(stop, for: cmd, type: type)
            }

            self.central?.stopScan()
            if let p = self.peripheral {
                self.central?.cancelPeripheralConnection(p)
            }
            self.teardownConnectionState()
            self.setState(.disconnected)
        }
    }

    // MARK: - Internal: scanning & connection (bleQueue only)

    /// Begin scanning for peripherals advertising the configured service.
    ///
    /// We scan filtered by `UnicornBLE.serviceUUID` when possible (more power
    /// efficient and works while backgrounded). Because that UUID is VERIFY and
    /// may be wrong, we also inspect advertised names in
    /// `didDiscover` and accept anything matching `namePrefix`.
    private func beginScan() {
        guard isActive, let central, central.state == .poweredOn else { return }
        setState(.connecting)
        // Filtering by service UUID is preferred, but the UUID is unverified;
        // passing nil would scan for everything. We pass the service filter and
        // additionally name-match in didDiscover as a belt-and-suspenders guard.
        central.scanForPeripherals(
            withServices: [UnicornBLE.serviceUUID], // VERIFY: service UUID unconfirmed
            options: [CBCentralManagerScanOptionAllowDuplicatesKey: false]
        )
    }

    /// Attempt to connect to the chosen peripheral.
    private func connect(to peripheral: CBPeripheral) {
        guard let central else { return }
        central.stopScan()
        self.peripheral = peripheral
        peripheral.delegate = self
        setState(.connecting)
        central.connect(peripheral, options: nil)
    }

    /// Schedule a reconnect with capped exponential backoff, if still active.
    private func scheduleReconnect() {
        guard isActive else { return }
        cancelPendingReconnect()

        let delay = min(reconnectMaxDelay,
                        reconnectBaseDelay * pow(2.0, Double(reconnectAttempt)))
        reconnectAttempt += 1

        let work = DispatchWorkItem { [weak self] in
            guard let self else { return }
            // Reset per-connection state and rescan from scratch.
            self.parser = UnicornPacketParser()
            self.didSendStart = false
            self.beginScan()
        }
        reconnectWorkItem = work
        bleQueue.asyncAfter(deadline: .now() + delay, execute: work)
    }

    private func cancelPendingReconnect() {
        reconnectWorkItem?.cancel()
        reconnectWorkItem = nil
    }

    /// Clear references to the current peripheral/characteristics.
    private func teardownConnectionState() {
        notifyCharacteristic = nil
        commandCharacteristic = nil
        peripheral = nil
        didSendStart = false
    }

    /// Update state on the BLE queue (the only place `_connectionState` is set).
    private func setState(_ newState: EEGConnectionState) {
        _connectionState = newState
    }

    /// Write the START acquisition command once notifications are confirmed.
    private func sendStartIfReady() {
        guard isActive,
              !didSendStart,
              let peripheral,
              let notify = notifyCharacteristic,
              notify.isNotifying,                 // gate START on confirmed notifications
              let command = commandCharacteristic
        else { return }

        didSendStart = true
        let start = Data(UnicornBLE.startCommand)
        let type: CBCharacteristicWriteType =
            command.properties.contains(.write) ? .withResponse : .withoutResponse
        peripheral.writeValue(start, for: command, type: type)

        // Streaming is considered live once START has been issued on a notifying
        // characteristic. Subsequent frames will start arriving in didUpdateValue.
        reconnectAttempt = 0
        setState(.streaming)
    }
}

// MARK: - CBCentralManagerDelegate

extension UnicornBLEManager: CBCentralManagerDelegate {

    func centralManagerDidUpdateState(_ central: CBCentralManager) {
        // All delegate callbacks already run on `bleQueue`.
        switch central.state {
        case .poweredOn:
            if isActive { beginScan() }

        case .poweredOff:
            setState(.failed("Bluetooth is powered off"))
            teardownConnectionState()

        case .unauthorized:
            setState(.failed("Bluetooth permission denied"))

        case .unsupported:
            setState(.failed("Bluetooth LE not supported on this device"))

        case .resetting, .unknown:
            // Transient; wait for the next state update.
            setState(.connecting)

        @unknown default:
            setState(.failed("Unknown Bluetooth state"))
        }
    }

    func centralManager(_ central: CBCentralManager,
                        didDiscover peripheral: CBPeripheral,
                        advertisementData: [String: Any],
                        rssi RSSI: NSNumber) {
        // Prefer the advertised local name, falling back to the peripheral name.
        let advName = advertisementData[CBAdvertisementDataLocalNameKey] as? String
        let name = advName ?? peripheral.name ?? ""

        // Name-match guard: only connect to devices matching the configured prefix.
        // (Service-UUID filtering in beginScan is unverified, so this is the real
        // selection criterion.)
        guard name.hasPrefix(namePrefix) else { return }

        connect(to: peripheral)
    }

    func centralManager(_ central: CBCentralManager,
                        didConnect peripheral: CBPeripheral) {
        // Discover the (VERIFY) primary service. Restricting to a known UUID
        // speeds discovery; pass the service so only it is enumerated.
        peripheral.discoverServices([UnicornBLE.serviceUUID]) // VERIFY: service UUID
    }

    func centralManager(_ central: CBCentralManager,
                        didFailToConnect peripheral: CBPeripheral,
                        error: Error?) {
        let msg = error?.localizedDescription ?? "Connection failed"
        setState(.failed(msg))
        teardownConnectionState()
        scheduleReconnect()
    }

    func centralManager(_ central: CBCentralManager,
                        didDisconnectPeripheral peripheral: CBPeripheral,
                        error: Error?) {
        teardownConnectionState()
        if isActive {
            // Unexpected drop while we still want data → backoff + reconnect.
            if let error {
                setState(.failed("Disconnected: \(error.localizedDescription)"))
            } else {
                setState(.connecting)
            }
            scheduleReconnect()
        } else {
            // Clean teardown initiated by stop().
            setState(.disconnected)
        }
    }
}

// MARK: - CBPeripheralDelegate

extension UnicornBLEManager: CBPeripheralDelegate {

    func peripheral(_ peripheral: CBPeripheral, didDiscoverServices error: Error?) {
        if let error {
            setState(.failed("Service discovery failed: \(error.localizedDescription)"))
            scheduleReconnect()
            return
        }
        guard let services = peripheral.services, !services.isEmpty else {
            setState(.failed("No services found"))
            scheduleReconnect()
            return
        }

        // Discover the notify + command characteristics within each service.
        // We request both known (VERIFY) UUIDs; if the service hosting them
        // differs, requesting per-service still surfaces them when present.
        for service in services {
            peripheral.discoverCharacteristics(
                [UnicornBLE.notifyCharUUID, UnicornBLE.commandCharUUID], // VERIFY: char UUIDs
                for: service
            )
        }
    }

    func peripheral(_ peripheral: CBPeripheral,
                    didDiscoverCharacteristicsFor service: CBService,
                    error: Error?) {
        if let error {
            setState(.failed("Characteristic discovery failed: \(error.localizedDescription)"))
            scheduleReconnect()
            return
        }
        guard let characteristics = service.characteristics else { return }

        for characteristic in characteristics {
            // Match by UUID first; if UUIDs are wrong/substituted, fall back to
            // capability-based matching so the manager still works after a UUID
            // swap (per contract §D.6: tolerate UUID substitution).
            if characteristic.uuid == UnicornBLE.notifyCharUUID {
                notifyCharacteristic = characteristic
            } else if characteristic.uuid == UnicornBLE.commandCharUUID {
                commandCharacteristic = characteristic
            } else if notifyCharacteristic == nil,
                      characteristic.properties.contains(.notify) ||
                      characteristic.properties.contains(.indicate) {
                notifyCharacteristic = characteristic
            } else if commandCharacteristic == nil,
                      characteristic.properties.contains(.write) ||
                      characteristic.properties.contains(.writeWithoutResponse) {
                commandCharacteristic = characteristic
            }
        }

        // Subscribe to the notify characteristic. The START command is gated on
        // the resulting `didUpdateNotificationStateFor` confirming isNotifying.
        if let notify = notifyCharacteristic, !notify.isNotifying {
            peripheral.setNotifyValue(true, for: notify)
        } else {
            // Already notifying (rare) — try to send START now.
            sendStartIfReady()
        }
    }

    func peripheral(_ peripheral: CBPeripheral,
                    didUpdateNotificationStateFor characteristic: CBCharacteristic,
                    error: Error?) {
        if let error {
            setState(.failed("Enable notify failed: \(error.localizedDescription)"))
            scheduleReconnect()
            return
        }
        guard characteristic.uuid == notifyCharacteristic?.uuid else { return }

        if characteristic.isNotifying {
            // Notifications confirmed — now (and only now) write START.
            sendStartIfReady()
        }
    }

    func peripheral(_ peripheral: CBPeripheral,
                    didUpdateValueFor characteristic: CBCharacteristic,
                    error: Error?) {
        if error != nil { return }
        guard characteristic.uuid == notifyCharacteristic?.uuid,
              let data = characteristic.value, !data.isEmpty else { return }

        // Feed raw notification bytes to the reassembler. It handles fragmentation
        // and resync, returning zero or more complete length-8 µV samples.
        let newSamples = parser.append(data)
        guard !newSamples.isEmpty, let continuation = samplesContinuation else { return }
        for sample in newSamples {
            continuation.yield(sample)
        }
    }

    func peripheral(_ peripheral: CBPeripheral,
                    didWriteValueFor characteristic: CBCharacteristic,
                    error: Error?) {
        // Surface command-write failures (e.g. START) as a failed state so the
        // pipeline can react. Successful writes need no action.
        if let error {
            setState(.failed("Command write failed: \(error.localizedDescription)"))
        }
    }
}
