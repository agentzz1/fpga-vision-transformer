"""ssvep_2048_app.py — FLAGSHIP gaming demo: play 2048 with SSVEP.

Self-contained pygame app: renders the 2048 board with 4 flickering arrows around
it, decodes occipital EEG (live Unicorn LSL, or SYNTHETIC fallback so it runs with
no hardware), and applies the chosen move. Zero browser dependency = reliable demo.

    python ssvep_2048_app.py              # live (Unicorn LSL) or auto-synthetic
    python ssvep_2048_app.py --synthetic  # force the no-hardware showcase
    python ssvep_2048_app.py --live --model cal.npy   # calibrated TRCA decode

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
from ssvep_cca import classify, FREQS, ARROWS, FS, synth_ssvep, achievable_freqs

ARROW_DIRS = ["up", "down", "left", "right"]
TILE_COLORS = {
    0:(205,193,180),2:(238,228,218),4:(237,224,200),8:(242,177,121),
    16:(245,149,99),32:(246,124,95),64:(246,94,59),128:(237,207,114),
    256:(237,204,97),512:(237,200,80),1024:(237,197,63),2048:(237,194,46),
}


class EEGSource:
    """Live LSL occipital source, or synthetic generator keyed to a 'gaze' target."""
    def __init__(self, synthetic=True, model=None, notch=50.0):
        self.synthetic = synthetic
        self.model = model            # optional TRCA model dict -> calibrated decode
        self.rng = np.random.default_rng(0)
        self.notch = notch            # mains notch (50 EU / 60 US); None to disable
        self.bad = []                 # last live window's bad-channel reasons (per occ ch)
        self._occ_names = ["Oz", "PO7", "PO8", "Pz"]
        self.live_failed = False       # True if --live was requested but LSL fell back
        self._acq = None; self._occ = None
        if not synthetic:
            try:
                from acquire import LSLAcquirer
                from ssvep_online import _occipital_idx
                self._acq = LSLAcquirer().start(); self._occ = _occipital_idx()
            except Exception as e:
                print("=" * 60)
                print(f"[EEG] !!! LIVE REQUESTED BUT LSL IS DOWN ({e}) -> SYNTHETIC !!!")
                print("[EEG] The board will move WITHOUT a headset. Start 'Unicorn LSL'.")
                print("=" * 60)
                self.synthetic = True; self.live_failed = True

    def window(self, win_s, gaze_idx=None, freqs=FREQS):
        if not self.synthetic and self._acq is not None:
            data, _ = self._acq.get_data(seconds=win_s)
            need = int(win_s*FS)
            if data.shape[1] < need:
                return None
            win = data[self._occ, -need:]
            from acquire import channel_quality, notch_filter
            if not getattr(self, "_unit_checked", False):   # one-time unit sanity check
                self._unit_checked = True
                med = float(np.median(win.std(axis=1)))
                if not (0.3 < med < 500):
                    print(f"[EEG] WARNING: median channel std={med:.2f} is outside the plausible "
                          f"uV range — the LSL stream may be in ADC counts/volts, not uV. The "
                          f"flat/railed gate (tuned in uV) may misfire; rescale the Unicorn LSL output.")
            ok, reasons = channel_quality(win)          # electrode-contact gate
            self.bad = [self._occ_names[i] for i, good in enumerate(ok) if not good]
            if self.notch:
                win = notch_filter(win, fs=FS, freq=self.notch)   # kill 50/60Hz mains
            return win
        # synthetic: emit SSVEP at the gazed arrow's ACTUAL (refresh-locked) frequency
        self.bad = []
        f = freqs[gaze_idx if gaze_idx is not None else self.rng.integers(len(freqs))]
        return synth_ssvep(f, win_s, n_ch=4, snr=0.5, rng=self.rng)


def run(synthetic=True, win_s=2.0, model_path=None, notch=50.0, refresh=None):
    import pygame
    pygame.init()
    W = 560; H = 760
    try:
        screen = pygame.display.set_mode((W, H), vsync=1)
    except Exception:
        screen = pygame.display.set_mode((W, H))
    if refresh is None:                       # --refresh override beats auto-detect
        try:
            refresh = int(round(pygame.display.get_current_refresh_rate()))
        except Exception:
            refresh = 60
    pygame.display.set_caption("SSVEP 2048 — control with your brain")
    font = pygame.font.SysFont("arial", 40, bold=True)
    big = pygame.font.SysFont("arial", 28, bold=True)
    small = pygame.font.SysFont("arial", 20)
    clock = pygame.time.Clock()
    if not refresh: refresh = 60
    # Refresh-LOCKED frequencies: exact on THIS monitor, shared by render + decode.
    freqs, halves = achievable_freqs(refresh, n=4)
    print(f"[SSVEP] display refresh={refresh}Hz -> flicker freqs (Hz): "
          + ", ".join(f"{ARROWS[i]}={freqs[i]:.3f}(every {halves[i]}f)" for i in range(4)))
    assert len(set(round(f, 4) for f in freqs)) == 4, "flicker frequencies collide!"
    game = Game2048()
    model = None
    if model_path:
        import numpy as _np
        model = dict(_np.load(model_path, allow_pickle=True).item()) if model_path.endswith('.npy') else None
    src = EEGSource(synthetic=synthetic, model=model, notch=notch)

    board_px, margin, top = 480, 40, 200
    cell = board_px // SIZE
    from collections import deque
    gaze_idx = [0]          # which arrow the user is "looking at" (synthetic demo: cycle)
    dwell = deque(maxlen=2)  # LIVE: require 2 consecutive confident agreeing windows
    last_decode = [0.0]; scores = [np.zeros(4)]
    step_s = 0.5            # overlapping re-decode cadence (window stays win_s long)
    frame = 0
    from collections import deque as _dq
    dts = _dq(maxlen=refresh)   # ~1s of per-frame dt for a frame-drop / refresh-mismatch check
    drop_warn = [""]
    state = ["listening…"]      # per-decode operator feedback (abstain/confidence/lock)
    last_lock = [0.0]           # wall time of last successful move (for the no-lock hint)

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

    def draw_flickers(frame):
        # 4 flicker bars at the edges, frequency per arrow
        bars = {0:(W//2-40,10,80,30), 1:(W//2-40,H-40,80,30),
                2:(10,top+board_px//2-40,30,80), 3:(W-40,top+board_px//2-40,30,80)}
        for i,rect in enumerate([bars[0],bars[1],bars[2],bars[3]]):
            half = halves[i]                                # exact integer half-cycle for this monitor
            on = (frame // half) % 2 == 0
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
        # EEG decode on OVERLAPPING windows: re-decode every step_s using the last
        # win_s of data, so the dwell gate accumulates evidence continuously and a
        # move lands in ~1-2s, not ~4-6s (non-overlapping). Same gate for live + synth.
        now = time.time()
        if now - last_decode[0] >= step_s:
            last_decode[0] = now
            win = src.window(win_s, gaze_idx[0], freqs=freqs)
            if win is not None and src.bad:
                dwell.clear()                 # bad electrode contact -> refuse to decode
                state[0] = "BAD CONTACT"
            elif win is not None:
                if src.model is not None:
                    from ssvep_trca import classify_trca
                    idx, sc = classify_trca(win, src.model)
                else:
                    idx, sc = classify(win, freqs=freqs)
                scores[0] = sc
                order = np.argsort(sc)[::-1]
                margin_ok = (sc[order[0]] - sc[order[1]]) >= 0.15 * (abs(sc[order[0]]) + 1e-9)
                # IDENTICAL decision logic for synthetic + live: the DECODER decides.
                # (synthetic injects the gazed target, but it must still be decoded
                #  through the confidence+dwell gate, so the showcase can genuinely miss.)
                dwell.append(idx if margin_ok else -1)
                if margin_ok and len(dwell) == dwell.maxlen and len(set(dwell)) == 1:
                    if game.can_move(): game.move(ARROW_DIRS[idx])
                    dwell.clear(); last_lock[0] = now
                    state[0] = f"LOCKED: {ARROWS[idx].upper()}"
                    if src.synthetic:        # advance to the next target after a real hit
                        gaze_idx[0] = (gaze_idx[0] + 1) % 4
                elif margin_ok:
                    state[0] = f"listening… ({ARROWS[idx]}?)"
                else:
                    state[0] = "low confidence"
        screen.fill((250,248,239))
        title = big.render(f"SSVEP 2048   score {game.score}", True, (119,110,101))
        screen.blit(title, (margin, 30))
        mode = small.render(("SYNTHETIC demo" if src.synthetic else "LIVE Unicorn") +
                            f"  |  {state[0]}", True, (140,130,120))
        screen.blit(mode, (margin, 70))
        sc = scores[0]; sct = small.render("CCA: " + "  ".join(
            f"{ARROWS[i]}={sc[i]:.2f}" for i in range(4)), True, (150,140,130))
        screen.blit(sct, (margin, 100))
        if src.synthetic:                    # UNMISTAKABLE no-headset watermark
            wm = big.render("SYNTHETIC — NO HEADSET", True, (210,150,150))
            screen.blit(wm, wm.get_rect(center=(W//2, H-22)))
        if src.live_failed:                  # --live requested but LSL was down
            lf = small.render("LIVE REQUESTED BUT LSL DOWN → SYNTHETIC", True, (200,60,60))
            screen.blit(lf, (margin, H-46))
        if not src.synthetic and (time.time() - last_lock[0]) > 8.0 and not src.bad:
            hint = small.render("weak response — try TRCA (--model), widen window, or blink less",
                                True, (200,120,60)); screen.blit(hint, (margin, 160))
        draw_board(); draw_flickers(frame)   # integer frame count -> refresh-locked parity
        if src.bad:                          # live electrode-contact warning
            warn = big.render("CHECK ELECTRODES: " + ",".join(src.bad), True, (200,60,60))
            screen.blit(warn, (margin, 130))
        if drop_warn[0]:                     # frame-drop / refresh-mismatch warning
            screen.blit(small.render(drop_warn[0], True, (200,60,60)), (margin, 160))
        if not game.can_move():
            go = big.render("GAME OVER", True, (200,60,60)); screen.blit(go,(margin,H-40))
        pygame.display.flip()
        dt = clock.tick(refresh) / 1000.0; frame += 1
        dts.append(dt)
        if len(dts) == dts.maxlen:           # check achieved cadence vs assumed refresh
            measured = 1.0 / (sum(dts) / len(dts))
            drops = sum(1 for d in dts if d > 1.5 / refresh) / len(dts)
            if abs(measured - refresh) > 0.5 or drops > 0.1:
                drop_warn[0] = (f"FRAME ISSUE: assumed {refresh}Hz, measured {measured:.1f}Hz, "
                                f"{drops*100:.0f}% late — flicker unreliable; pass --refresh / close apps")
            else:
                drop_warn[0] = ""
    pygame.quit()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--model", default=None, help="TRCA calibration .npy (calibrated decode)")
    ap.add_argument("--notch", type=float, default=50.0, help="mains notch Hz (50 EU / 60 US; 0=off)")
    ap.add_argument("--refresh", type=int, default=None, help="monitor refresh Hz override (else auto-detect)")
    a = ap.parse_args()
    run(synthetic=not a.live, model_path=a.model, notch=(a.notch or None), refresh=a.refresh)
