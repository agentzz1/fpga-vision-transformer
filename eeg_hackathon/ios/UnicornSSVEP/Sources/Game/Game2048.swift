//
//  Game2048.swift
//  SSVEP2048
//
//  Pure model/logic for the 2048 game engine.
//
//  This file is intentionally UI-free and dependency-free (aside from
//  Foundation and the `Direction` enum defined in Core/Constants.swift).
//  It is a value type (`struct`) so it can be copied freely, compared with
//  `==`, and driven deterministically in unit tests via an explicit RNG seed.
//
//  Conforms exactly to INTERFACE CONTRACT §B.13.
//

import Foundation

// MARK: - Deterministic RNG

/// A small, deterministic pseudo-random number generator.
///
/// `Game2048` must be reproducible in tests: given the same seed and the same
/// sequence of moves, the spawned tiles (and therefore the entire board) must
/// be identical run-to-run. Swift's default `SystemRandomNumberGenerator` is
/// not seedable, so we use a SplitMix64 generator — tiny, fast, well-distributed,
/// and trivially seedable.
///
/// This type is `private` to the file: it is an implementation detail and not
/// part of the public contract.
private struct SplitMix64: RandomNumberGenerator {
    private var state: UInt64

    init(seed: UInt64) {
        self.state = seed
    }

    mutating func next() -> UInt64 {
        // SplitMix64 (Steele, Lea & Flood, 2014). Public-domain reference algorithm.
        state &+= 0x9E3779B97F4A7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }
}

// MARK: - Game2048

/// The 2048 board engine.
///
/// Board layout: `board[row][col]`, with `row == 0` at the top and `col == 0`
/// at the left. A cell value of `0` means empty; any non-zero value is a tile
/// (always a power of two >= 2).
///
/// Moves slide every tile as far as possible in the chosen direction, merging
/// each pair of equal adjacent tiles exactly once per move (the classic 2048
/// rule: a tile that has just been formed by a merge cannot merge again in the
/// same move). If the board changed as a result of the slide/merge, a new tile
/// (`2` with 90% probability, `4` with 10%) is spawned in a random empty cell.
struct Game2048: Equatable {

    // MARK: Public state (contract §B.13)

    /// The `size` x `size` grid. `0` == empty, otherwise the tile's value.
    private(set) var board: [[Int]]

    /// Running score: incremented by the value of each newly merged tile.
    private(set) var score: Int

    /// `true` once no move in any direction can change the board.
    private(set) var isGameOver: Bool

    // MARK: Private state

    /// Board dimension (default 4 → standard 4x4 game).
    private let size: Int

    /// Seeded RNG used for tile spawning. When `seed` is `nil` we still use a
    /// `SplitMix64` (seeded from the system generator) so that *all* randomness
    /// flows through a single, replaceable source — this keeps spawn behaviour
    /// uniform and testable.
    private var rng: SplitMix64

    // MARK: Init (contract §B.13)

    /// Create a new game.
    ///
    /// - Parameters:
    ///   - size: Grid dimension. Defaults to `4` (standard 2048).
    ///   - seed: Optional RNG seed. Pass a fixed value for deterministic,
    ///           reproducible games (used by tests). When `nil`, a seed is
    ///           drawn from the system generator.
    init(size: Int = 4, seed: UInt64? = nil) {
        // Guard against degenerate sizes; the contract default is 4.
        self.size = max(1, size)
        self.board = Array(repeating: Array(repeating: 0, count: self.size),
                           count: self.size)
        self.score = 0
        self.isGameOver = false

        if let seed {
            self.rng = SplitMix64(seed: seed)
        } else {
            // Derive a one-off seed from the system RNG so unseeded games are
            // still routed through SplitMix64.
            var system = SystemRandomNumberGenerator()
            self.rng = SplitMix64(seed: system.next())
        }

        // A fresh game starts with two tiles, exactly like the original.
        spawnTile()
        spawnTile()
    }

    // MARK: Public API

    /// Apply a move in `direction`.
    ///
    /// Slides and merges tiles, updates `score`, and — if the board changed —
    /// spawns one new tile and refreshes `isGameOver`.
    ///
    /// - Returns: `true` if the board changed (and a tile was spawned),
    ///            `false` if the move was a no-op (board left untouched).
    @discardableResult
    mutating func move(_ direction: Direction) -> Bool {
        // Ignore moves once the game is over.
        guard !isGameOver else { return false }

        let before = board
        let (newBoard, gained) = Self.applyMove(to: board, direction: direction)

        // A move is only valid if it actually changed the board.
        guard newBoard != before else { return false }

        board = newBoard
        score += gained
        spawnTile()
        updateGameOver()
        return true
    }

    /// Reset to a fresh game (same `size`), re-seeding two starting tiles.
    ///
    /// The RNG stream is *not* reset; the game simply continues drawing from
    /// the existing generator. Tests that need a fully reproducible reset
    /// should construct a new `Game2048(seed:)` instead.
    mutating func reset() {
        board = Array(repeating: Array(repeating: 0, count: size), count: size)
        score = 0
        isGameOver = false
        spawnTile()
        spawnTile()
    }

    /// `true` if any tile has reached the 2048 win threshold.
    var hasWon: Bool {
        for row in board {
            for value in row where value >= 2048 {
                return true
            }
        }
        return false
    }

    // MARK: Convenience accessors (non-contract conveniences)

    /// The largest tile currently on the board (`0` if the board is empty).
    var maxTile: Int {
        var best = 0
        for row in board {
            for value in row where value > best {
                best = value
            }
        }
        return best
    }

    /// `true` if at least one move in some direction would change the board.
    /// (Inverse of `isGameOver`, computed on demand from the current board.)
    var canMove: Bool {
        !Self.isStuck(board)
    }

    // MARK: - Move mechanics (pure, static, testable)

    /// Apply a move to a board without mutating game state.
    ///
    /// Implemented by reducing every direction to a left-slide:
    ///   - left  : rows as-is
    ///   - right : reverse each row, slide left, reverse back
    ///   - up    : transpose, slide left, transpose back
    ///   - down  : transpose, reverse, slide left, reverse, transpose back
    ///
    /// - Returns: the resulting board and the score gained from merges.
    private static func applyMove(to board: [[Int]],
                                  direction: Direction) -> (board: [[Int]], gained: Int) {
        var working = board

        switch direction {
        case .left:
            break
        case .right:
            working = working.map { $0.reversed() }
        case .up:
            working = transpose(working)
        case .down:
            working = transpose(working).map { $0.reversed() }
        }

        var gained = 0
        working = working.map { row -> [Int] in
            let (collapsed, points) = collapseRowLeft(row)
            gained += points
            return collapsed
        }

        // Undo the orientation transform applied above.
        switch direction {
        case .left:
            break
        case .right:
            working = working.map { $0.reversed() }
        case .up:
            working = transpose(working)
        case .down:
            working = transpose(working.map { $0.reversed() })
        }

        return (working, gained)
    }

    /// Slide and merge a single row to the left.
    ///
    /// 1. Drop zeros (compact non-empty tiles to the left).
    /// 2. Merge equal adjacent tiles once, left-to-right (a merged tile is
    ///    locked and cannot merge again this move).
    /// 3. Re-pad with zeros on the right to the original length.
    ///
    /// - Returns: the new row and the points gained (sum of merged tile values).
    private static func collapseRowLeft(_ row: [Int]) -> (row: [Int], points: Int) {
        let length = row.count
        let tiles = row.filter { $0 != 0 }

        var merged: [Int] = []
        merged.reserveCapacity(tiles.count)
        var points = 0
        var index = 0

        while index < tiles.count {
            // If the next tile equals the current one, merge them into one.
            if index + 1 < tiles.count, tiles[index] == tiles[index + 1] {
                let value = tiles[index] * 2
                merged.append(value)
                points += value
                index += 2          // consume both tiles; result tile is locked
            } else {
                merged.append(tiles[index])
                index += 1
            }
        }

        // Pad the remainder with empties to keep the row length stable.
        if merged.count < length {
            merged.append(contentsOf: repeatElement(0, count: length - merged.count))
        }

        return (merged, points)
    }

    /// Transpose a square matrix (rows <-> columns).
    private static func transpose(_ matrix: [[Int]]) -> [[Int]] {
        guard let first = matrix.first else { return matrix }
        let cols = first.count
        let rows = matrix.count
        var result = Array(repeating: Array(repeating: 0, count: rows), count: cols)
        for r in 0..<rows {
            for c in 0..<cols {
                result[c][r] = matrix[r][c]
            }
        }
        return result
    }

    // MARK: - Spawning

    /// Place a new tile (`2` 90% of the time, `4` 10%) into a random empty cell.
    /// No-op if the board is full.
    private mutating func spawnTile() {
        // Collect the coordinates of all empty cells.
        var empties: [(row: Int, col: Int)] = []
        for r in 0..<size {
            for c in 0..<size where board[r][c] == 0 {
                empties.append((r, c))
            }
        }
        guard !empties.isEmpty else { return }

        // Pick an empty cell deterministically via our seeded RNG.
        let pickIndex = Int(rng.next() % UInt64(empties.count))
        let cell = empties[pickIndex]

        // 90% chance of a 2, 10% chance of a 4.
        let roll = rng.next() % 10
        board[cell.row][cell.col] = (roll == 0) ? 4 : 2
    }

    // MARK: - Game-over detection

    /// Recompute and cache `isGameOver` from the current board.
    private mutating func updateGameOver() {
        isGameOver = Self.isStuck(board)
    }

    /// `true` if no move in any direction would change `board`:
    /// the board is full AND no two orthogonally adjacent tiles are equal.
    private static func isStuck(_ board: [[Int]]) -> Bool {
        let rows = board.count
        guard rows > 0 else { return true }
        let cols = board[0].count

        for r in 0..<rows {
            for c in 0..<cols {
                let value = board[r][c]
                // Any empty cell means a move is still possible.
                if value == 0 { return false }
                // A matching right neighbour can be merged.
                if c + 1 < cols, board[r][c + 1] == value { return false }
                // A matching bottom neighbour can be merged.
                if r + 1 < rows, board[r + 1][c] == value { return false }
            }
        }
        return true
    }

    // MARK: - Equatable

    /// Equality intentionally ignores the RNG state and `size`-derived caches;
    /// two games are equal when their *observable* state matches. This makes
    /// `Game2048` usable as `@Published` value with `objectWillChange` dedup and
    /// keeps test assertions focused on board/score/over-state.
    static func == (lhs: Game2048, rhs: Game2048) -> Bool {
        lhs.board == rhs.board
            && lhs.score == rhs.score
            && lhs.isGameOver == rhs.isGameOver
    }
}
