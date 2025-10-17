#!/usr/bin/env python3
"""
Minimal Pygame visualizer (safe opener)
- Always opens a window (1280x720) with a lightweight animated pattern.
- If AUDIO_FILE is set or ./audio_file.mp3 exists, it plays it (no analysis needed).
- Esc to quit; also quits on window close.
"""
import os, sys, math, time, random
import pygame
import numpy as np

W, H = 1280, 720
FPS = 60

def init_audio():
    """Try to play any audio just for vibe; analysis is not required here."""
    try:
        pygame.mixer.pre_init(44100, -16, 2, 512)
        pygame.mixer.init()
        audio_path = os.environ.get("AUDIO_FILE") or "audio_file.mp3"
        if os.path.exists(audio_path):
            pygame.mixer.music.load(audio_path)
            pygame.mixer.music.play(-1)
    except Exception:
        # Audio is optional; ignore any failure.
        pass

def draw_lissajous(surface, t):
    surface.fill((10, 10, 14))
    cx, cy = W//2, H//2
    A = min(W, H) * 0.35
    a, b = 3, 2
    delta = t * 0.7
    pts = []
    n = 600
    for i in range(n):
        p = i / (n - 1)
        x = cx + A * math.sin(a * p * 2*math.pi + delta)
        y = cy + A * math.sin(b * p * 2*math.pi)
        pts.append((x, y))
    # fade trail
    pygame.draw.aalines(surface, (200, 220, 255), False, pts)
    # pulsing ring
    r = int(80 + 60 * (0.5 + 0.5 * math.sin(t * 1.6)))
    pygame.draw.circle(surface, (80, 120, 255), (cx, cy), r, 2)

def draw_bars(surface, t):
    surface.fill((8, 10, 12))
    bars = 64
    w = W // bars
    for i in range(bars):
        phase = t * 1.8 + i * 0.23
        h = int((H * 0.3) * (0.55 + 0.45 * math.sin(phase)))
        x = i * w
        y = H - h - 50
        rect = pygame.Rect(x+2, y, w-4, h)
        col = (40 + (i*3) % 160, 140, 220)
        pygame.draw.rect(surface, col, rect, border_radius=4)

def main():
    pygame.init()
    pygame.display.set_caption(os.path.basename(sys.argv[0]) + " — minimal")
    screen = pygame.display.set_mode((W, H))
    clock = pygame.time.Clock()
    init_audio()

    t0 = time.time()
    mode = 0
    next_switch = t0 + 7.0  # alternate between two simple looks

    running = True
    while running:
        t = time.time() - t0
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    running = False
                elif e.key == pygame.K_SPACE:
                    mode = (mode + 1) % 2

        if time.time() > next_switch:
            mode = (mode + 1) % 2
            next_switch = time.time() + 7.0

        if mode == 0:
            draw_lissajous(screen, t)
        else:
            draw_bars(screen, t)

        pygame.display.flip()
        clock.tick(FPS)

    try:
        pygame.mixer.music.stop()
    except Exception:
        pass
    pygame.quit()

if __name__ == "__main__":
    main()
