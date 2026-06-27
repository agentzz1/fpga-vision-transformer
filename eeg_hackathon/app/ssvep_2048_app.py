"""ssvep_2048_app.py — FLAGSHIP gaming demo: play 2048 with SSVEP.

Self-contained pygame app: renders the 2048 board with 4 flickering arrows around
it, decodes occipital EEG (live Unicorn LSL, or SYNTHETIC fallback so it runs with
no hardware), and applies the chosen move. Zero browser dependency = reliable demo.

    python ssvep_2048_app.py              # live (Unicorn LSL) or auto-synthetic
    python ssvep_2048_app.py --synthetic  # force the no-hardware showcase
    python ssvep_2048_app.py --model cal.npz   # use a TRCA calibration

Controls: arrow keys also work (manual fallback for the judges); ESC quits.
"""
from __future__ import annotations
import os as _os, sys as _sys
_HB = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_HB, _os.path.join(_HB, "ssvep"), _os.path.join(_HB, "app")):
    if _p not in _sys.path: _sys.path.insert(0, _p)
import argparse, sys, time
import numpy as np

from game_2048 import Game2048, SIZE
from ssvep_cca import classify, FREQS, ARROWS, FS, synth_ssvep

ARROW_DIRS = ["up", "down", "left", "right"]
TILE_COLORS = {
    0:(205,193,180),2:(238,228,218),4:(237,224,200),8:(242,177,121),
    16:(245,149,99),32:(246,124,95),64:(246,94,59),128:(237,207,114),
    256:(237,204,97),512:(237,200,80),1024:(237,197,63),2048:(237,194,46),
}


class EEGSource:
    """Live LSL occipital source, or synthetic generator keyed to a 'gaze' target."""
    def __init__(self, synthetic=True):
        self.synthetic = synthetic
        self.rng = np.random.default_rng(0)
        self._acq = None; self._occ = None
        if not synthetic:
            try:
                from acquire import LSLAcquirer
                from ssvep_online import _occipital_idx
                self._acq = LSLAcquirer().start(); self._occ = _occipital_idx()
            except Exception as e:
                print(f"[EEG] LSL unavailable ({e}); using SYNTHETIC."); self.synthetic = True

    def window(self, win_s, gaze_idx=None):
        if not self.synthetic and self._acq is not None:
            data, _ = self._acq.get_data(seconds=win_s)
            return data[self._occ, -int(win_s*FS):] if data.shape[1] >= int(win_s*FS) else None
        # synthetic: emit SSVEP at the gazed arrow's frequency (demo/fallback)
        f = FREQS[gaze_idx if gaze_idx is not None else self.rng.integers(len(FREQS))]
        return synth_ssvep(f, win_s, n_ch=4, snr=0.5, rng=self.rng)


def run(synthetic=True, win_s=2.0):
    import pygame
    pygame.init()
    W = 560; H = 760
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("SSVEP 2048 — control with your brain")
    font = pygame.font.SysFont("arial", 40, bold=True)
    big = pygame.font.SysFont("arial", 28, bold=True)
    small = pygame.font.SysFont("arial", 20)
    clock = pygame.time.Clock()
    game = Game2048()
    src = EEGSource(synthetic=synthetic)

    board_px, margin, top = 480, 40, 200
    cell = board_px // SIZE
    gaze_idx = [0]          # which arrow the user is "looking at" (synthetic demo: cycle)
    last_decode = [0.0]; scores = [np.zeros(4)]
    frame = 0; refresh = 60

    def draw_board():
        for r in range(SIZE):
            for c in range(SIZE):
                v = game.board[r][c]
                x = margin + c*cell; y = top + r*cell
                pygame.draw.rect(screen, TILE_COLORS.get(v,(60,58,50)),
                                 (x+4,y+4,cell-8,cell-8), border_radius=6)
                if v:
                    t = font.render(str(v), True, (119,110,101) if v<=4 else (249,246,242))
                    screen.blit(t, t.get_rect(center=(x+cell//2, y+cell//2)))

    def draw_flickers(t):
        # 4 flicker bars at the edges, frequency per arrow
        bars = {0:(W//2-40,10,80,30), 1:(W//2-40,H-40,80,30),
                2:(10,top+board_px//2-40,30,80), 3:(W-40,top+board_px//2-40,30,80)}
        for i,(f,rect) in enumerate(zip(FREQS,[bars[0],bars[1],bars[2],bars[3]])):
            on = np.sin(2*np.pi*f*t) > 0
            col = (255,255,255) if on else (40,40,40)
            if i == gaze_idx[0]: pygame.draw.rect(screen,(90,160,255),
                                 (rect[0]-3,rect[1]-3,rect[2]+6,rect[3]+6),border_radius=4)
            pygame.draw.rect(screen, col, rect, border_radius=4)

    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type==pygame.KEYDOWN and e.key==pygame.K_ESCAPE):
                running = False
            elif e.type == pygame.KEYDOWN:    # manual fallback + pick synthetic gaze
                km = {pygame.K_UP:0,pygame.K_DOWN:1,pygame.K_LEFT:2,pygame.K_RIGHT:3}
                if e.key in km:
                    gaze_idx[0] = km[e.key]; game.move(ARROW_DIRS[km[e.key]])
        # periodic EEG decode -> move
        now = time.time()
        if now - last_decode[0] >= win_s:
            last_decode[0] = now
            win = src.window(win_s, gaze_idx[0])
            if win is not None:
                idx, sc = classify(win); scores[0] = sc
                if game.can_move(): game.move(ARROW_DIRS[idx])
                if src.synthetic: gaze_idx[0] = (gaze_idx[0]+1) % 4   # cycle for showcase
        screen.fill((250,248,239))
        title = big.render(f"SSVEP 2048   score {game.score}", True, (119,110,101))
        screen.blit(title, (margin, 30))
        mode = small.render(("SYNTHETIC demo" if src.synthetic else "LIVE Unicorn") +
                            "  |  look at an arrow", True, (140,130,120))
        screen.blit(mode, (margin, 70))
        sc = scores[0]; sct = small.render("CCA: " + "  ".join(
            f"{ARROWS[i]}={sc[i]:.2f}" for i in range(4)), True, (150,140,130))
        screen.blit(sct, (margin, 100))
        draw_board(); draw_flickers(frame/refresh)
        if not game.can_move():
            go = big.render("GAME OVER", True, (200,60,60)); screen.blit(go,(margin,H-40))
        pygame.display.flip(); clock.tick(refresh); frame += 1
    pygame.quit()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--live", action="store_true")
    a = ap.parse_args()
    run(synthetic=not a.live)
