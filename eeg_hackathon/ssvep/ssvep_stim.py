"""ssvep_stim.py — flickering arrow stimuli for SSVEP control of 2048.

Four squares flicker at REFRESH-LOCKED frequencies. A flicker is only exact when its
half-period refresh/(2f) is a whole number of frames, so we do NOT hard-code 8.57/10/
12/15 (those only round exactly on a 120 Hz display). Instead we detect the monitor's
refresh and call achievable_freqs() to get four exact, distinct frequencies, and toggle
each box on an integer frame schedule. Pass the SAME freqs to the decoder.

    python ssvep_stim.py
Requires pygame.
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ssvep_cca import achievable_freqs

ARROWS = ["UP", "DOWN", "LEFT", "RIGHT"]


def run(width=900, height=700, refresh=None):
    import pygame
    pygame.init()
    try:
        screen = pygame.display.set_mode((width, height), vsync=1)
    except Exception:
        screen = pygame.display.set_mode((width, height))
    if refresh is None:
        try:
            refresh = int(round(pygame.display.get_current_refresh_rate())) or 60
        except Exception:
            refresh = 60
    freqs, halves = achievable_freqs(refresh, n=4)
    print(f"[stim] refresh={refresh}Hz -> exact flicker freqs (Hz): "
          + ", ".join(f"{ARROWS[i]}={freqs[i]:.3f}(every {halves[i]}f)" for i in range(4)))
    pygame.display.set_caption("SSVEP arrows — look at your target")
    clock = pygame.time.Clock()
    cx, cy, s, gap = width // 2, height // 2, 130, 200
    rects = [pygame.Rect(cx - s // 2, cy - gap - s, s, s),
             pygame.Rect(cx - s // 2, cy + gap, s, s),
             pygame.Rect(cx - gap - s, cy - s // 2, s, s),
             pygame.Rect(cx + gap, cy - s // 2, s, s)]
    font = pygame.font.SysFont(None, 48)
    frame = 0
    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                running = False
        screen.fill((0, 0, 0))
        for i, rect in enumerate(rects):
            on = (frame // halves[i]) % 2 == 0           # refresh-locked square wave
            col = (255, 255, 255) if on else (20, 20, 20)
            pygame.draw.rect(screen, col, rect)
            lab = font.render(ARROWS[i], True, (120, 160, 255))
            screen.blit(lab, lab.get_rect(center=rect.center))
        pygame.display.flip()
        clock.tick(refresh)
        frame += 1
    pygame.quit()


if __name__ == "__main__":
    run()
