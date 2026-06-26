"""ssvep_stim.py — flickering arrow stimuli for SSVEP control of 2048.

Four squares flicker at FREQS (60Hz-monitor sub-harmonics, exact). Look at the
direction you want to move; ssvep_online.py decodes which and presses the key.
Run this in a window next to the 2048 game.

    python ssvep_stim.py
Requires pygame. On a 60 Hz display the per-frame on/off schedule yields the
exact target frequencies (8.57/10/12/15 Hz).
"""
from __future__ import annotations
import math

FREQS = [8.57, 10.0, 12.0, 15.0]
ARROWS = ["UP", "DOWN", "LEFT", "RIGHT"]


def run(width=900, height=700, refresh=60):
    import pygame
    pygame.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("SSVEP arrows — look at your target")
    clock = pygame.time.Clock()
    cx, cy, s, gap = width // 2, height // 2, 130, 200
    boxes = {  # arrow -> (rect, freq)
        "UP":    (pygame.Rect(cx - s // 2, cy - gap - s, s, s), FREQS[0]),
        "DOWN":  (pygame.Rect(cx - s // 2, cy + gap, s, s),     FREQS[1]),
        "LEFT":  (pygame.Rect(cx - gap - s, cy - s // 2, s, s), FREQS[2]),
        "RIGHT": (pygame.Rect(cx + gap, cy - s // 2, s, s),     FREQS[3]),
    }
    font = pygame.font.SysFont(None, 48)
    frame = 0
    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                running = False
        screen.fill((0, 0, 0))
        t = frame / refresh
        for name, (rect, f) in boxes.items():
            on = (math.sin(2 * math.pi * f * t) > 0)     # square-wave flicker
            col = (255, 255, 255) if on else (20, 20, 20)
            pygame.draw.rect(screen, col, rect)
            lab = font.render(name, True, (120, 160, 255))
            screen.blit(lab, lab.get_rect(center=rect.center))
        pygame.display.flip()
        clock.tick(refresh)
        frame += 1
    pygame.quit()


if __name__ == "__main__":
    run()
