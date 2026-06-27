//
//  SSVEPController.swift
//  SSVEP2048
//
//  Bridges raw decoder output (`Decision`) to discrete 2048 game moves.
//
//  This is the "decision gate": a small, pure, unit-testable state machine that
//  sits between the FBCCA decoder (which fires ~4 Decisions/second as the 2 s
//  window slides) and the game model (which wants at most one deliberate move
//  per intended glance). It ports the proven gating recipe:
//
//    1. Confidence margin   — reject decisions whose winning target does not beat
//                             the runner-up by the configured margin. Default
//                             threshold 1.2 (max >= 1.2 * secondMax, i.e. a >=0.2
//                             relative margin with headroom over the proven >=0.15
//                             recipe). `Decision.confidence` is defined by the
//                             contract as max/secondMax (>= 1), so the margin gate
//                             is `confidence >= confidenceThreshold` (default 1.2).
//    2. Dwell               — require `dwellWindows` *consecutive, agreeing* high-
//                             confidence windows before committing (default 3 ≈
//                             0.75 s of sustained intent at the 0.25 s cadence).
//    3. Refractory          — after committing a move, ignore everything for ~1 s
//                             so a single sustained glance produces exactly one
//                             move, not a burst.
//
//  Per the interface contract (§B.14) the *public* surface is exactly:
//    @Published game, lastDecision, dwellProgress; init(...); ingest(_:); reset().
//  The internal gate state (listening / lowConfidence / locked / refractory) is
//  modelled privately via the enum below and surfaced only through dwellProgress.
//
//  Threading: this is an ObservableObject whose @Published properties drive UI,
//  so `ingest` is expected to be called on the main actor (the pipeline hops to
//  main before delivering Decisions). The logic itself is otherwise pure.
//

import Combine
import Foundation

/// Bridges decoder output to game moves with dwell/debounce.
final class SSVEPController: ObservableObject {

    // MARK: - Public, contract-defined surface (§B.14)

    /// The live game model. Mutated only by committing a fully-dwelled move.
    @Published private(set) var game: Game2048

    /// The most recent `Decision` fed to `ingest`, regardless of whether it
    /// passed the gate. Useful for diagnostics / on-screen confidence readout.
    @Published private(set) var lastDecision: Decision?

    /// Progress in [0, 1] toward committing the *current* candidate move.
    /// 0 == not dwelling (listening, low confidence, or refractory),
    /// 1 == about to commit on the next agreeing window.
    @Published private(set) var dwellProgress: Double

    // MARK: - Configuration (captured at init; not part of public surface)

    /// Consecutive agreeing, high-confidence windows required to emit a move.
    private let dwellWindows: Int

    /// Confidence gate. A `Decision` is "high confidence" iff
    /// `confidence >= confidenceThreshold`. Contract default 1.2 corresponds to
    /// the proven >=0.15 *relative* margin with headroom (max >= 1.2 * 2nd).
    private let confidenceThreshold: Float

    /// Refractory period after a committed move, expressed in *windows* rather
    /// than wall-clock seconds so the gate stays pure and frame-rate independent.
    /// At the contract's ~0.25 s decision cadence (windowStride 62 @ 250 Hz),
    /// 4 windows ≈ 1.0 s — matching the proven ~1 s refractory.
    private let refractoryWindows: Int

    // MARK: - Internal gate state

    /// Coarse phase of the decision gate. Exposed indirectly via `dwellProgress`;
    /// kept private to honour "do not invent public API beyond the contract."
    private enum GateState: Equatable {
        /// No agreeing streak in progress; waiting for a confident decision.
        case listening
        /// Saw a decision but it failed the confidence gate.
        case lowConfidence
        /// Accumulating an agreeing, confident streak toward a commit.
        case dwelling(target: Int, count: Int)
        /// A move was just committed; ignoring input until refractory elapses.
        case refractory(remaining: Int)
    }

    private var state: GateState = .listening

    // MARK: - Init

    /// - Parameters:
    ///   - game: initial game model (defaults to a fresh board).
    ///   - dwellWindows: consecutive agreeing windows required to emit a move.
    ///   - confidenceThreshold: minimum `Decision.confidence` (max/secondMax) to
    ///     count a decision as "confident". Default 1.2 (>=0.15 relative margin).
    init(game: Game2048 = Game2048(),
         dwellWindows: Int = 3,
         confidenceThreshold: Float = 1.2) {
        self.game = game
        // Guard against degenerate configuration: at least one agreeing window.
        self.dwellWindows = max(1, dwellWindows)
        self.confidenceThreshold = confidenceThreshold
        // ~1.0 s refractory at the standard 0.25 s cadence.
        self.refractoryWindows = 4
        self.lastDecision = nil
        self.dwellProgress = 0
    }

    // MARK: - Ingest

    /// Feed one decoder `Decision`. Advances the dwell/refractory state machine
    /// and, when a candidate target survives the confidence + dwell gates,
    /// applies the corresponding move (`EEGConfig.ARROWS[target]`) to `game`.
    ///
    /// Every call counts as exactly one "window" for dwell/refractory accounting,
    /// which keeps the gate purely event-driven and trivially unit-testable.
    func ingest(_ decision: Decision) {
        lastDecision = decision

        // --- Refractory: one window elapses per ingest; swallow input meanwhile.
        if case .refractory(let remaining) = state {
            let next = remaining - 1
            if next <= 0 {
                state = .listening
            } else {
                state = .refractory(remaining: next)
            }
            dwellProgress = 0
            return
        }

        // --- Validity: a real decision must name a target and beat the gate.
        //     `target == -1` is the contract's "no decision" sentinel.
        let isConfident =
            decision.target >= 0 &&
            decision.target < EEGConfig.ARROWS.count &&
            decision.confidence >= confidenceThreshold

        guard isConfident else {
            // Any non-confident window breaks an in-progress streak: a glance must
            // be *sustained*, so we restart rather than tolerate gaps.
            state = .lowConfidence
            dwellProgress = 0
            return
        }

        // --- Dwell accumulation -------------------------------------------------
        let target = decision.target
        let newCount: Int
        switch state {
        case .dwelling(let current, let count) where current == target:
            // Streak continues for the same target.
            newCount = count + 1
        default:
            // Fresh streak: either we were idle/low-confidence, or the confident
            // target changed (which resets accumulated dwell).
            newCount = 1
        }

        if newCount >= dwellWindows {
            // Commit: the intent has been held long enough.
            commitMove(target: target)
        } else {
            // Still accumulating; report fractional progress toward the threshold.
            state = .dwelling(target: target, count: newCount)
            dwellProgress = Double(newCount) / Double(dwellWindows)
        }
    }

    // MARK: - Reset

    /// Reset both the game and the gate to their initial state.
    func reset() {
        game.reset()
        state = .listening
        lastDecision = nil
        dwellProgress = 0
    }

    // MARK: - Private

    /// Apply the dwelled move and enter the refractory period.
    private func commitMove(target: Int) {
        // Index identity (contract §D.4): ARROWS[target] is the intended Direction.
        let direction = EEGConfig.ARROWS[target]
        game.move(direction)               // no-op-safe; @discardableResult

        // Enter refractory so one sustained glance yields exactly one move.
        state = .refractory(remaining: refractoryWindows)
        // A momentary full bar signals the commit; it falls back to 0 next window.
        dwellProgress = 1
    }
}
