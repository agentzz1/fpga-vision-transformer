"""game_2048.py — pure 2048 engine (no GUI), so it's fully unit-testable headless.

The SSVEP/EEG controller calls move('up'|'down'|'left'|'right'); the GUI just
renders state. Keeping logic separate = reliable demo + testable tonight.
"""
from __future__ import annotations
import random
from typing import List, Tuple

SIZE = 4


class Game2048:
    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.board = [[0] * SIZE for _ in range(SIZE)]
        self.score = 0
        self._spawn(); self._spawn()

    def _empty(self) -> List[Tuple[int, int]]:
        return [(r, c) for r in range(SIZE) for c in range(SIZE) if self.board[r][c] == 0]

    def _spawn(self):
        empty = self._empty()
        if empty:
            r, c = self.rng.choice(empty)
            self.board[r][c] = 4 if self.rng.random() < 0.1 else 2

    @staticmethod
    def _compress_merge(row: List[int]) -> Tuple[List[int], int]:
        """Slide non-zeros left and merge equal neighbours once. Returns (row, gained)."""
        vals = [v for v in row if v != 0]
        out, gained, i = [], 0, 0
        while i < len(vals):
            if i + 1 < len(vals) and vals[i] == vals[i + 1]:
                out.append(vals[i] * 2); gained += vals[i] * 2; i += 2
            else:
                out.append(vals[i]); i += 1
        out += [0] * (SIZE - len(out))
        return out, gained

    def _move_left(self, board):
        moved, gained, new = False, 0, []
        for row in board:
            nr, g = self._compress_merge(row)
            gained += g
            if nr != row: moved = True
            new.append(nr)
        return new, moved, gained

    @staticmethod
    def _rot(board):  # rotate clockwise
        return [list(r) for r in zip(*board[::-1])]

    def move(self, direction: str) -> bool:
        """Apply a move; spawn a tile if the board changed. Returns moved?."""
        b = self.board
        k = {"left": 0, "up": 3, "right": 2, "down": 1}[direction]
        for _ in range(k): b = self._rot(b)
        b, moved, gained = self._move_left(b)
        for _ in range((4 - k) % 4): b = self._rot(b)
        if moved:
            self.board = b; self.score += gained; self._spawn()
        return moved

    def can_move(self) -> bool:
        if self._empty(): return True
        for r in range(SIZE):
            for c in range(SIZE):
                v = self.board[r][c]
                if (c + 1 < SIZE and self.board[r][c + 1] == v) or \
                   (r + 1 < SIZE and self.board[r + 1][c] == v):
                    return True
        return False

    def max_tile(self) -> int:
        return max(max(row) for row in self.board)


if __name__ == "__main__":
    # headless self-test: random play should run, score should grow, merges work
    g = Game2048(seed=1)
    assert g._compress_merge([2, 2, 2, 2]) == ([4, 4, 0, 0], 8)
    assert g._compress_merge([2, 0, 2, 4]) == ([4, 4, 0, 0], 4)
    assert g._compress_merge([0, 0, 0, 2]) == ([2, 0, 0, 0], 0)
    moves = 0
    while g.can_move() and moves < 2000:
        for d in ("up", "left", "down", "right"):
            if g.move(d): break
        moves += 1
    print(f"2048 engine OK: {moves} moves, score={g.score}, max tile={g.max_tile()}")
