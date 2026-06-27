//
//  Game2048View.swift
//  SSVEP2048
//
//  SwiftUI rendering of the 2048 board: tiles, classic tile colors, and the
//  running score, given a pure `Game2048` model value.
//
//  Contract role (§B.16):
//    - `Game2048View` is a *pure* view of a `Game2048` snapshot. Its only public
//      API is `init(game: Game2048)` and `body`. It owns no state and never
//      mutates the model — moves are driven elsewhere (SSVEPController).
//
//  This file additionally provides the on-screen *status chrome* described for
//  this view's role — but, to honor the contract's exact `Game2048View`
//  signature (§B.16), that chrome lives in small, file-internal helper views
//  (`ElectrodeWarningBanner`, `SyntheticWatermark`, `SSVEPStatusBar`). The root
//  `ContentView` (owner 11) composes these together with `FlickerView` so the
//  arrows sit around the board. Keeping them here keeps the board + status
//  styling in one place without widening the public `Game2048View` API.
//
//  Index identity (§D.4) is respected indirectly: this view only renders the
//  board; arrow placement / decoder alignment is handled by FlickerView.
//

import SwiftUI

// MARK: - Game2048View (contract §B.16)

/// Pure SwiftUI rendering of a 2048 board snapshot.
///
/// Draws an N×N grid (N = `game.board.count`) of tiles using the classic 2048
/// palette, plus the score and a game-over overlay. Holds no state; re-renders
/// whenever the injected `game` value changes.
struct Game2048View: View {

    /// Immutable snapshot of the game to render.
    let game: Game2048

    /// Contract initializer (§B.16). The view is a function of this value only.
    init(game: Game2048) {
        self.game = game
    }

    var body: some View {
        VStack(spacing: 16) {
            scoreHeader
            board
        }
        .padding(Game2048Style.boardPadding)
        .background(
            RoundedRectangle(cornerRadius: Game2048Style.boardCornerRadius)
                .fill(Game2048Style.boardBackground)
        )
    }

    // MARK: Score header

    /// Title + current score chip.
    private var scoreHeader: some View {
        HStack(alignment: .center) {
            Text("2048")
                .font(.system(size: 34, weight: .heavy, design: .rounded))
                .foregroundColor(Game2048Style.titleColor)

            Spacer()

            VStack(spacing: 2) {
                Text("SCORE")
                    .font(.system(size: 11, weight: .bold, design: .rounded))
                    .foregroundColor(Game2048Style.scoreLabelColor)
                Text("\(game.score)")
                    .font(.system(size: 22, weight: .heavy, design: .rounded))
                    .foregroundColor(.white)
                    .monospacedDigit()
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 8)
            .background(
                RoundedRectangle(cornerRadius: 6)
                    .fill(Game2048Style.scoreChipBackground)
            )

            if game.hasWon {
                wonBadge
            }
        }
    }

    /// Small "WIN" badge shown once any tile reaches 2048.
    private var wonBadge: some View {
        Text("WIN!")
            .font(.system(size: 13, weight: .black, design: .rounded))
            .foregroundColor(.white)
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
            .background(Capsule().fill(Game2048Style.winBadgeColor))
            .padding(.leading, 8)
    }

    // MARK: Board grid

    /// The square tile grid. Uses a `GeometryReader` so tiles stay square and
    /// scale to the available width regardless of board size.
    private var board: some View {
        GeometryReader { geo in
            let n = max(game.board.count, 1)
            let spacing = Game2048Style.tileSpacing
            // Solve for tile side so n tiles + (n+1) gaps fill the smaller edge.
            let side = (min(geo.size.width, geo.size.height) - spacing * CGFloat(n + 1))
                / CGFloat(n)
            let tileSide = max(side, 0)

            ZStack {
                RoundedRectangle(cornerRadius: Game2048Style.gridCornerRadius)
                    .fill(Game2048Style.gridBackground)

                VStack(spacing: spacing) {
                    ForEach(0..<n, id: \.self) { row in
                        HStack(spacing: spacing) {
                            ForEach(0..<n, id: \.self) { col in
                                tileView(value: valueAt(row: row, col: col),
                                         side: tileSide)
                            }
                        }
                    }
                }
                .padding(spacing)

                if game.isGameOver {
                    gameOverOverlay
                }
            }
        }
        .aspectRatio(1, contentMode: .fit)
    }

    /// Safe accessor in case `board` is ragged or smaller than expected.
    private func valueAt(row: Int, col: Int) -> Int {
        guard row < game.board.count, col < game.board[row].count else { return 0 }
        return game.board[row][col]
    }

    /// A single tile (empty or with a value).
    private func tileView(value: Int, side: CGFloat) -> some View {
        RoundedRectangle(cornerRadius: Game2048Style.tileCornerRadius)
            .fill(Game2048Style.tileBackground(for: value))
            .frame(width: side, height: side)
            .overlay(
                Group {
                    if value > 0 {
                        Text("\(value)")
                            .font(.system(
                                size: Game2048Style.fontSize(forValue: value, tileSide: side),
                                weight: .heavy,
                                design: .rounded))
                            .foregroundColor(Game2048Style.tileTextColor(for: value))
                            .minimumScaleFactor(0.4)
                            .lineLimit(1)
                            .padding(2)
                    }
                }
            )
            // Animate value changes (spawns / merges) when the model updates.
            .animation(.easeInOut(duration: 0.12), value: value)
    }

    /// Dim overlay with "GAME OVER" when no moves remain.
    private var gameOverOverlay: some View {
        ZStack {
            RoundedRectangle(cornerRadius: Game2048Style.gridCornerRadius)
                .fill(Color.black.opacity(0.5))
            Text("GAME OVER")
                .font(.system(size: 30, weight: .black, design: .rounded))
                .foregroundColor(.white)
        }
        .transition(.opacity)
    }
}

// MARK: - Status chrome (consumed by ContentView, owner 11)
//
// These are deliberately NOT part of the `Game2048View` public surface (§B.16
// fixes its init to `init(game:)`). They are file-internal SwiftUI views so the
// root screen can overlay them on/around the board + FlickerView.

/// Red "CHECK ELECTRODES" banner.
///
/// Surfaced when the EEG signal quality is poor (e.g. railed channels / no
/// contact). ContentView decides visibility from the pipeline; this view is
/// purely presentational.
struct ElectrodeWarningBanner: View {
    /// Optional detail (e.g. which channels are bad). Shown under the headline.
    let detail: String?

    init(detail: String? = nil) {
        self.detail = detail
    }

    var body: some View {
        VStack(spacing: 2) {
            Label("CHECK ELECTRODES", systemImage: "exclamationmark.triangle.fill")
                .font(.system(size: 15, weight: .heavy, design: .rounded))
                .foregroundColor(.white)
            if let detail, !detail.isEmpty {
                Text(detail)
                    .font(.system(size: 12, weight: .semibold, design: .rounded))
                    .foregroundColor(.white.opacity(0.9))
            }
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 10)
        .background(Color.red)
        .accessibilityElement(children: .combine)
        .accessibilityLabel("Check electrodes")
    }
}

/// Large diagonal "SYNTHETIC — NO HEADSET" watermark.
///
/// Overlaid across the whole screen whenever the active `EEGSource` is the
/// `SyntheticEEGSource` (simulator or `forceSynthetic`). Non-interactive.
struct SyntheticWatermark: View {
    var body: some View {
        Text("SYNTHETIC — NO HEADSET")
            .font(.system(size: 40, weight: .black, design: .rounded))
            .foregroundColor(Color.orange.opacity(0.22))
            .rotationEffect(.degrees(-30))
            .fixedSize()
            .allowsHitTesting(false)          // never intercepts gaze/touch input
            .accessibilityHidden(true)
    }
}

/// Compact status bar showing the BLE/source connection state and (optionally)
/// dwell progress toward committing a move.
///
/// `EEGConnectionState` comes from the contract's `BLE/EEGSource.swift` (§B.3);
/// this view maps each case to a label + color dot.
struct SSVEPStatusBar: View {
    /// Current source connection state.
    let state: EEGConnectionState
    /// Dwell progress 0..1 (from `SSVEPController.dwellProgress`); nil hides it.
    let dwellProgress: Double?

    init(state: EEGConnectionState, dwellProgress: Double? = nil) {
        self.state = state
        self.dwellProgress = dwellProgress
    }

    var body: some View {
        VStack(spacing: 6) {
            HStack(spacing: 8) {
                Circle()
                    .fill(stateColor)
                    .frame(width: 10, height: 10)
                Text(stateText)
                    .font(.system(size: 14, weight: .semibold, design: .rounded))
                    .foregroundColor(Game2048Style.titleColor)
                Spacer()
            }

            if let dwellProgress {
                ProgressView(value: max(0, min(1, dwellProgress)))
                    .progressViewStyle(.linear)
                    .tint(Game2048Style.winBadgeColor)
                    .accessibilityLabel("Dwell progress")
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(
            RoundedRectangle(cornerRadius: 8)
                .fill(Game2048Style.scoreChipBackground.opacity(0.15))
        )
    }

    /// Human-readable status text per connection state.
    private var stateText: String {
        switch state {
        case .disconnected:      return "Disconnected"
        case .connecting:        return "Connecting…"
        case .streaming:         return "Streaming"
        case .failed(let msg):   return "Failed: \(msg)"
        }
    }

    /// Status dot color per connection state.
    private var stateColor: Color {
        switch state {
        case .disconnected:  return .gray
        case .connecting:    return .yellow
        case .streaming:     return .green
        case .failed:        return .red
        }
    }
}

// MARK: - Styling helpers

/// Centralized colors / metrics for the 2048 board, using the classic palette.
/// Internal (not part of the contract); shared by the views in this file.
enum Game2048Style {

    // Layout metrics.
    static let boardPadding: CGFloat = 16
    static let boardCornerRadius: CGFloat = 12
    static let gridCornerRadius: CGFloat = 8
    static let tileCornerRadius: CGFloat = 6
    static let tileSpacing: CGFloat = 8

    // Background / chrome colors.
    static let boardBackground = Color(red: 0.98, green: 0.97, blue: 0.94)
    static let gridBackground  = Color(red: 0.73, green: 0.68, blue: 0.63)
    static let titleColor      = Color(red: 0.46, green: 0.43, blue: 0.40)
    static let scoreLabelColor = Color(red: 0.93, green: 0.89, blue: 0.85)
    static let scoreChipBackground = Color(red: 0.73, green: 0.68, blue: 0.63)
    static let winBadgeColor   = Color(red: 0.93, green: 0.76, blue: 0.18)

    /// Empty-cell color (board grid showing through).
    private static let emptyTile = Color(red: 0.80, green: 0.76, blue: 0.71)

    /// Classic 2048 tile background by value. Falls back to the dark "super"
    /// color for anything above the standard table.
    static func tileBackground(for value: Int) -> Color {
        switch value {
        case 0:     return emptyTile
        case 2:     return Color(red: 0.93, green: 0.89, blue: 0.85)
        case 4:     return Color(red: 0.93, green: 0.88, blue: 0.78)
        case 8:     return Color(red: 0.95, green: 0.69, blue: 0.47)
        case 16:    return Color(red: 0.96, green: 0.58, blue: 0.39)
        case 32:    return Color(red: 0.96, green: 0.49, blue: 0.37)
        case 64:    return Color(red: 0.96, green: 0.37, blue: 0.23)
        case 128:   return Color(red: 0.93, green: 0.81, blue: 0.45)
        case 256:   return Color(red: 0.93, green: 0.80, blue: 0.38)
        case 512:   return Color(red: 0.93, green: 0.78, blue: 0.31)
        case 1024:  return Color(red: 0.93, green: 0.77, blue: 0.25)
        case 2048:  return Color(red: 0.93, green: 0.76, blue: 0.18)
        default:    return Color(red: 0.24, green: 0.23, blue: 0.20)  // >2048
        }
    }

    /// Tile text color: dark on light (2/4) tiles, white otherwise.
    static func tileTextColor(for value: Int) -> Color {
        switch value {
        case 2, 4:  return Color(red: 0.46, green: 0.43, blue: 0.40)
        default:    return .white
        }
    }

    /// Font size scaled by tile side and digit count so big numbers still fit.
    static func fontSize(forValue value: Int, tileSide: CGFloat) -> CGFloat {
        let digits = String(value).count
        // Base size is a fraction of the tile; shrink as digit count grows.
        let base = tileSide * 0.5
        let shrink: CGFloat
        switch digits {
        case 0...2: shrink = 1.0
        case 3:     shrink = 0.85
        case 4:     shrink = 0.70
        default:    shrink = 0.55
        }
        return max(base * shrink, 10)
    }
}

// MARK: - Previews

#if DEBUG
struct Game2048View_Previews: PreviewProvider {
    /// Build a deterministic, populated board for previewing tile colors.
    private static func sampleGame() -> Game2048 {
        var g = Game2048(seed: 42)
        // Drive a few moves so the preview shows several tile values.
        _ = g.move(.up)
        _ = g.move(.left)
        _ = g.move(.down)
        _ = g.move(.right)
        return g
    }

    static var previews: some View {
        ZStack {
            Color(red: 0.98, green: 0.97, blue: 0.94).ignoresSafeArea()

            VStack(spacing: 12) {
                SSVEPStatusBar(state: .streaming, dwellProgress: 0.6)
                ElectrodeWarningBanner(detail: "PO7, Oz railed")
                Game2048View(game: sampleGame())
            }
            .padding()

            SyntheticWatermark()
        }
        .previewDisplayName("Board + status chrome")
    }
}
#endif
