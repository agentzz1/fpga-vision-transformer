//
//  ContentView.swift
//  SSVEP2048
//
//  Root screen (interface contract §B.17, owner 11). Composes the SSVEP flicker
//  tiles (`FlickerView`) around the 2048 board (`Game2048View`) and overlays the
//  status chrome (connection state + dwell progress, synthetic watermark,
//  electrode warning) defined in `Game2048View.swift`.
//
//  All game/decoder state comes from the injected `AppViewModel` (and the
//  `SSVEPController` it owns). This view holds NO pipeline state of its own.
//

import SwiftUI
import Combine

/// The single root view of the app.
struct ContentView: View {

    /// The shared pipeline coordinator, injected by `SSVEP2048App`.
    @EnvironmentObject var viewModel: AppViewModel

    /// Observe the controller directly so game/dwell updates re-render the board.
    @ObservedObject private var controllerProxy = ControllerProxy()

    var body: some View {
        // Bridge the controller (owned by the view model) into this view's
        // observation graph so @Published changes refresh the UI.
        let controller = viewModel.controller

        ZStack {
            Color.black.ignoresSafeArea()

            GeometryReader { geo in
                ZStack {
                    // SSVEP flicker tiles fill the screen; the board sits in the
                    // center "hole" of the diamond arrangement.
                    FlickerView(targets: viewModel.targets,
                                highlightedTarget: highlightedTarget(controller))
                        .ignoresSafeArea()

                    // Center game board, sized to the smaller screen dimension.
                    Game2048View(game: controller.game)
                        .frame(width: min(geo.size.width, geo.size.height) * 0.42,
                               height: min(geo.size.width, geo.size.height) * 0.42)
                        .position(x: geo.size.width / 2, y: geo.size.height / 2)
                }
            }

            // Top status chrome: connection + dwell.
            VStack {
                SSVEPStatusBar(state: viewModel.connectionState,
                               dwellProgress: controller.dwellProgress)
                    .padding(.horizontal, 12)
                    .padding(.top, 8)

                if viewModel.signalQuality < 0.15 &&
                   viewModel.connectionState == .streaming {
                    ElectrodeWarningBanner(detail: nil)
                        .padding(.horizontal, 12)
                }

                Spacer()

                // Reset control along the bottom.
                Button(action: { viewModel.resetGame() }) {
                    Label("Reset", systemImage: "arrow.counterclockwise")
                        .font(.system(size: 15, weight: .semibold, design: .rounded))
                        .foregroundColor(.white)
                        .padding(.horizontal, 18)
                        .padding(.vertical, 10)
                        .background(Capsule().fill(Color.white.opacity(0.15)))
                }
                .padding(.bottom, 16)
            }

            // Synthetic-source watermark overlay.
            if viewModel.usingSynthetic {
                SyntheticWatermark()
            }
        }
        .statusBarHidden(true)
        // Keep the view subscribed to the controller's published updates.
        .onAppear { controllerProxy.bind(controller) }
    }

    /// Highlight the target currently being dwelled on, if any, by reading the
    /// last decision's winning target while dwell is in progress.
    private func highlightedTarget(_ controller: SSVEPController) -> Int? {
        guard controller.dwellProgress > 0,
              let decision = controller.lastDecision,
              decision.target >= 0 else { return nil }
        return decision.target
    }
}

// MARK: - Controller observation bridge

/// `AppViewModel.controller` is a separate `ObservableObject`. SwiftUI only
/// re-renders for `ObservableObject`s it observes directly, so this tiny proxy
/// republishes the controller's `objectWillChange` into the view's graph.
private final class ControllerProxy: ObservableObject {
    private var cancellable: Any?

    func bind(_ controller: SSVEPController) {
        // Re-publish the controller's change notifications as our own so the
        // view refreshes when game/dwell/lastDecision change.
        cancellable = controller.objectWillChange.sink { [weak self] _ in
            self?.objectWillChange.send()
        }
    }
}

#if DEBUG
struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
            .environmentObject(AppViewModel(forceSynthetic: true, refresh: 60))
    }
}
#endif
