//
//  FlickerView.swift
//  SSVEP2048
//
//  CADisplayLink-driven SSVEP flicker tiles.
//
//  This view renders 4 arrow flicker squares as *refresh-locked square waves*.
//  Each tile toggles on/off purely as a function of an integer frame counter
//  driven by CADisplayLink — never Date() — so the realized stimulation
//  frequency is exactly an integer sub-multiple of the display refresh rate.
//
//  Contract role (§B.15):
//    - `targets` is index-aligned to EEGConfig.ARROWS:
//        targets[0] -> .up, [1] -> .down, [2] -> .left, [3] -> .right
//    - The frequencies driven here MUST be the same FlickerTarget values the
//      decoder (FBCCA) uses, so the on-screen stimulus and the reference banks
//      agree. The caller obtains both from achievableFreqs(detectedRefresh).
//    - `highlightedTarget` highlights the current gaze/dwell target (or nil).
//
//  Square-wave rule (load-bearing):
//      tile i is ON  <=>  (frameCount / target.framesPerHalfPeriod) % 2 == 0
//  i.e. the tile holds each state for `framesPerHalfPeriod` (k) frames, giving a
//  realized frequency of refresh / (2 * k) Hz — exactly FlickerTarget.frequency.
//

import SwiftUI

#if canImport(UIKit)
import UIKit
#endif

// MARK: - FlickerView

/// CADisplayLink-driven flicker tiles. Drives opacity from an integer frame
/// counter (refresh-locked, never Date()). `targets` index-aligned to ARROWS.
struct FlickerView: View {

    /// Flicker targets, index-aligned to `EEGConfig.ARROWS`.
    let targets: [FlickerTarget]

    /// Optional dwell/gaze highlight: index into `targets`, or nil for none.
    let highlightedTarget: Int?

    init(targets: [FlickerTarget], highlightedTarget: Int? = nil) {
        self.targets = targets
        self.highlightedTarget = highlightedTarget
    }

    var body: some View {
        // The driver publishes a monotonically increasing frame counter that is
        // ticked once per display refresh by CADisplayLink. The on/off state of
        // every tile is a pure function of this counter, so all tiles stay
        // perfectly phase-locked to the display.
        GeometryReader { geo in
            FlickerDriverView(targets: targets) { frameCount in
                arrowGrid(in: geo.size, frameCount: frameCount)
            }
        }
    }

    // MARK: Layout

    /// Lays out the four directional tiles in a plus / diamond arrangement:
    ///
    ///         [ up ]
    ///   [ left ]   [ right ]
    ///        [ down ]
    ///
    /// The center is left empty for the game board (composed by ContentView).
    @ViewBuilder
    private func arrowGrid(in size: CGSize, frameCount: Int) -> some View {
        // Tile side length: a fraction of the smaller dimension so the diamond
        // fits regardless of orientation.
        let tile = min(size.width, size.height) * 0.26
        let cx = size.width / 2
        let cy = size.height / 2
        // Radial offset from center to each tile center.
        let r = min(size.width, size.height) * 0.36

        ZStack {
            // Up (index 0)
            tileView(index: 0, side: tile, frameCount: frameCount)
                .position(x: cx, y: cy - r)
            // Down (index 1)
            tileView(index: 1, side: tile, frameCount: frameCount)
                .position(x: cx, y: cy + r)
            // Left (index 2)
            tileView(index: 2, side: tile, frameCount: frameCount)
                .position(x: cx - r, y: cy)
            // Right (index 3)
            tileView(index: 3, side: tile, frameCount: frameCount)
                .position(x: cx + r, y: cy)
        }
        .frame(width: size.width, height: size.height)
    }

    /// A single flicker tile. Renders nothing if `index` is out of range so the
    /// view tolerates fewer than 4 targets gracefully.
    @ViewBuilder
    private func tileView(index: Int, side: CGFloat, frameCount: Int) -> some View {
        if index < targets.count {
            let target = targets[index]
            let isOn = Self.isOn(frameCount: frameCount, target: target)
            let isHighlighted = (highlightedTarget == index)

            ZStack {
                // The flicker square itself: a high-contrast on/off square wave.
                // We toggle a fully-opaque white square against a black square,
                // which maximizes contrast (and thus SSVEP SNR) and keeps the
                // mean luminance comparable across phases.
                RoundedRectangle(cornerRadius: 8)
                    .fill(isOn ? Color.white : Color.black)

                // Directional glyph overlaid on the flicker square. Kept a mid
                // gray so it is visible in both phases without dominating the
                // luminance modulation that the SSVEP relies on.
                Image(systemName: Self.glyphName(for: index))
                    .font(.system(size: side * 0.4, weight: .bold))
                    .foregroundColor(.gray)

                // Dwell / gaze highlight ring. Purely cosmetic — it does NOT
                // change the flicker phase or luminance schedule.
                if isHighlighted {
                    RoundedRectangle(cornerRadius: 8)
                        .strokeBorder(Color.green, lineWidth: side * 0.06)
                }
            }
            .frame(width: side, height: side)
            // Disable implicit SwiftUI animation: the flicker MUST be a hard,
            // single-frame square-wave transition, not an interpolated fade.
            .animation(nil, value: isOn)
            .accessibilityLabel(Self.accessibilityLabel(for: index, target: target))
        } else {
            EmptyView()
        }
    }

    // MARK: Square-wave state

    /// The core refresh-locked square-wave rule.
    ///
    /// A tile holds each state (on, then off) for `framesPerHalfPeriod` frames.
    /// One full period is therefore `2 * framesPerHalfPeriod` frames, giving a
    /// realized frequency of `refresh / (2 * framesPerHalfPeriod)` Hz.
    ///
    /// Returns ON when the integer half-period index is even.
    static func isOn(frameCount: Int, target: FlickerTarget) -> Bool {
        let k = max(1, target.framesPerHalfPeriod) // guard against /0
        return (frameCount / k) % 2 == 0
    }

    // MARK: Glyphs / accessibility

    /// SF Symbol arrow glyph for each ARROWS index (0..3 == up/down/left/right).
    private static func glyphName(for index: Int) -> String {
        switch index {
        case 0: return "arrow.up"
        case 1: return "arrow.down"
        case 2: return "arrow.left"
        case 3: return "arrow.right"
        default: return "questionmark"
        }
    }

    private static func directionName(for index: Int) -> String {
        switch index {
        case 0: return "Up"
        case 1: return "Down"
        case 2: return "Left"
        case 3: return "Right"
        default: return "Unknown"
        }
    }

    private static func accessibilityLabel(for index: Int, target: FlickerTarget) -> String {
        // e.g. "Up, 12.0 hertz flicker"
        let hz = String(format: "%.2f", target.frequency)
        return "\(directionName(for: index)), \(hz) hertz flicker"
    }
}

// MARK: - CADisplayLink driver

/// Bridges a `CADisplayLink` into SwiftUI. Owns the integer frame counter and
/// re-renders its content on every display refresh.
///
/// Implemented as a `UIViewRepresentable` so the `CADisplayLink` is tied to the
/// view's lifecycle (created on appear, invalidated on disappear) and so we can
/// detect the screen's `maximumFramesPerSecond` (the achievable refresh rate).
private struct FlickerDriverView<Content: View>: UIViewRepresentable {

    let targets: [FlickerTarget]
    /// Content builder, called with the current integer frame counter.
    let content: (Int) -> Content

    func makeCoordinator() -> Coordinator {
        Coordinator()
    }

    func makeUIView(context: Context) -> HostingFrameView {
        let view = HostingFrameView()
        // The coordinator owns the CADisplayLink and pumps the frame counter
        // into the hosting view's SwiftUI content.
        context.coordinator.attach(to: view) { frameCount in
            content(frameCount)
        }
        return view
    }

    func updateUIView(_ uiView: HostingFrameView, context: Context) {
        // Re-bind the content closure so updated `targets` / `highlightedTarget`
        // (captured by the closure) are reflected on the next tick.
        context.coordinator.updateContentBuilder { frameCount in
            content(frameCount)
        }
    }

    static func dismantleUIView(_ uiView: HostingFrameView, coordinator: Coordinator) {
        coordinator.invalidate()
    }

    // MARK: Coordinator

    final class Coordinator {
        private var displayLink: CADisplayLink?
        private weak var hostView: HostingFrameView?
        private var contentBuilder: ((Int) -> Content)?

        /// Monotonic frame counter, ticked once per display refresh.
        private var frameCount: Int = 0

        /// Attach a CADisplayLink to the given view and start ticking.
        func attach(to view: HostingFrameView, builder: @escaping (Int) -> Content) {
            self.hostView = view
            self.contentBuilder = builder

            // Render the initial frame immediately so the first paint is correct.
            renderCurrentFrame()

            let link = CADisplayLink(target: self, selector: #selector(tick))
            // Request the full native refresh rate. On ProMotion (up to 120 Hz)
            // this drives the highest achievable cadence, which the frequency
            // selection (achievableFreqs(detectedRefresh)) is computed against.
            //
            // NOTE on detected refresh: the display's achievable refresh rate is
            // `UIScreen.main.maximumFramesPerSecond`. The *caller* reads this and
            // passes the resulting FlickerTargets in; here we simply lock the
            // CADisplayLink to that maximum so the integer frame counter advances
            // at the same rate the targets were designed for.
            if #available(iOS 15.0, *) {
                let maxFPS = view.detectedRefresh
                link.preferredFrameRateRange = CAFrameRateRange(
                    minimum: Float(maxFPS),
                    maximum: Float(maxFPS),
                    preferred: Float(maxFPS)
                )
            }
            link.add(to: .main, forMode: .common)
            self.displayLink = link
        }

        func updateContentBuilder(_ builder: @escaping (Int) -> Content) {
            self.contentBuilder = builder
            // Repaint with the current counter so prop changes (e.g. highlight)
            // are visible without waiting for state to differ.
            renderCurrentFrame()
        }

        @objc private func tick(_ link: CADisplayLink) {
            // Advance the refresh-locked counter exactly once per refresh.
            frameCount &+= 1
            renderCurrentFrame()
        }

        private func renderCurrentFrame() {
            guard let host = hostView, let builder = contentBuilder else { return }
            host.setContent(builder(frameCount))
        }

        func invalidate() {
            displayLink?.invalidate()
            displayLink = nil
            hostView = nil
            contentBuilder = nil
        }

        deinit {
            invalidate()
        }
    }
}

// MARK: - Hosting view

/// A UIView that hosts SwiftUI content and exposes the screen's achievable
/// refresh rate. Re-hosts content cheaply by reusing a single `UIHostingController`.
private final class HostingFrameView: UIView {

    private var hostingController: UIHostingController<AnyView>?

    /// The display's achievable refresh rate (Hz). On ProMotion this is 120;
    /// on standard displays, 60. Falls back to 60 if a screen is unavailable.
    var detectedRefresh: Int {
        // Prefer the screen actually showing this view (multi-display / Stage
        // Manager correctness); fall back to the main screen.
        let screen = window?.windowScene?.screen ?? UIScreen.main
        let fps = screen.maximumFramesPerSecond
        return fps > 0 ? fps : 60
    }

    /// Install or update the hosted SwiftUI content.
    func setContent<V: View>(_ view: V) {
        let wrapped = AnyView(view)
        if let hc = hostingController {
            // Reuse the existing hosting controller — just swap the root view.
            hc.rootView = wrapped
        } else {
            let hc = UIHostingController(rootView: wrapped)
            hc.view.backgroundColor = .clear
            hc.view.translatesAutoresizingMaskIntoConstraints = false
            addSubview(hc.view)
            NSLayoutConstraint.activate([
                hc.view.leadingAnchor.constraint(equalTo: leadingAnchor),
                hc.view.trailingAnchor.constraint(equalTo: trailingAnchor),
                hc.view.topAnchor.constraint(equalTo: topAnchor),
                hc.view.bottomAnchor.constraint(equalTo: bottomAnchor),
            ])
            hostingController = hc
        }
    }
}

// MARK: - Preview

#if DEBUG
struct FlickerView_Previews: PreviewProvider {
    static var previews: some View {
        // 60 Hz fallback example: k=2 -> 15 Hz, k=3 -> 10 Hz, etc.
        // (In the app these come from achievableFreqs(detectedRefresh).)
        let targets = [
            FlickerTarget(frequency: 15.0, framesPerHalfPeriod: 2),
            FlickerTarget(frequency: 10.0, framesPerHalfPeriod: 3),
            FlickerTarget(frequency: 7.5,  framesPerHalfPeriod: 4),
            FlickerTarget(frequency: 6.0,  framesPerHalfPeriod: 5),
        ]
        return FlickerView(targets: targets, highlightedTarget: 0)
            .background(Color.black)
            .ignoresSafeArea()
    }
}
#endif
