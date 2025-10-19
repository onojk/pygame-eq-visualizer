#!/usr/bin/env python3
"""
AuroraSunsetGarden — audio-reactive visualizer with fireworks, inversion, and soft edge fixes
"""
import os, math, time, random, colorsys
import numpy as np
import pygame
import pyaudio

from aurora_engine import (
    hsv255, lerp, clamp,
    ColorEngine, preset_ring, preset_starburst, preset_spiral, preset_flower, preset_scaffold,
    MotifEngine, WaterfallSystem, FireworksSystem, NatureDirector, Fireflies,
    draw_background, draw_shape, blit_with_symmetry, TempoTracker, crush_blacks, apply_vignette
)

# ===== Config =====
CHUNK = 512
CHANNELS = 1
RATE = 22050
FPS = 60
GRID_SPACING = 42
DOT_BASE_RADIUS = 3.5
DOT_MAX_BOOST = 6
DEFAULT_DOT_COLOR = (185, 220, 255)
EDGE_MARGIN = 28   # soft border so dots don’t stick to edges

NUM_INFLUENCERS = 14
INFLUENCE_RADIUS = 280.0
BASE_STRENGTH = 1.0

AUDIO_COUPLED = True
PREFERRED_DEVICE_SUBSTR = "pulse"

# ===== Init =====
if "SDL_AUDIODRIVER" not in os.environ:
    os.environ["SDL_AUDIODRIVER"] = "pulse"
pygame.init()
screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
info = pygame.display.Info(); WIDTH, HEIGHT = info.current_w, info.current_h
CENTER = (WIDTH//2, HEIGHT//2)
clock = pygame.time.Clock()

scene_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
trail_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
trail_strengths = [(255,255,255,0), (0,0,0,18), (0,0,0,30), (0,0,0,44)]
trail_ix = 2

# ===== Audio =====
p = pyaudio.PyAudio()
def find_input_index(substr):
    for i in range(p.get_device_count()):
        d = p.get_device_info_by_index(i)
        if d.get("maxInputChannels",0) >= 1 and substr in d.get("name","").lower():
            return i
    return p.get_default_input_device_info()["index"]

stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True,
                input_device_index=find_input_index(PREFERRED_DEVICE_SUBSTR), frames_per_buffer=CHUNK)

# ===== Engines =====
color_engine = ColorEngine()
motifs = MotifEngine(WIDTH, HEIGHT, CENTER)
fireworks = FireworksSystem(WIDTH, HEIGHT)
waterfall = WaterfallSystem(WIDTH, HEIGHT)
nature = NatureDirector(WIDTH, HEIGHT, FPS)
fireflies = Fireflies(WIDTH, HEIGHT, n=140)
tempo = TempoTracker(target_range=(70,180), fps=FPS)

BG_THEMES = ['Sunset','Ocean','Forest','Night']
RENDERERS = ['dots','petals','tri','quad','star','ribbons','soft']
kaleido_sectors = [1,2,4,6]
bg_ix = renderer_ix = kaleido_ix = 0

fireworks_enabled = waterfall_enabled = nature_enabled = fireflies_enabled = True
contrast_mode = 0  # start off for brightness
invert_output = False

# ===== Main loop =====
running = True
while running:
    dt = clock.tick(FPS) / 1000.0
    for e in pygame.event.get():
        if e.type == pygame.QUIT: running = False
        elif e.type == pygame.KEYDOWN:
            if e.key == pygame.K_ESCAPE: running = False
            elif e.key == pygame.K_p: fireworks_enabled = not fireworks_enabled
            elif e.key == pygame.K_w: waterfall_enabled = not waterfall_enabled
            elif e.key == pygame.K_n: nature_enabled = not nature_enabled
            elif e.key == pygame.K_v: fireflies_enabled = not fireflies_enabled
            elif e.key == pygame.K_b: bg_ix = (bg_ix + 1) % len(BG_THEMES)
            elif e.key == pygame.K_c: contrast_mode = (contrast_mode + 1) % 3

    try:
        data = np.frombuffer(stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16)
    except IOError:
        data = np.zeros(CHUNK, dtype=np.int16)
    spectrum = np.abs(np.fft.rfft(data))

    feat_energy = float(np.mean(spectrum)) / (np.max(spectrum) + 1e-9)
    speed_scale = 1.0 + 0.6 * feat_energy

    color_engine.tick({"energy":feat_energy}, {"energy":feat_energy}, dt)
    if fireworks_enabled:
        fireworks.maybe_launch({}, {}, speed_scale)
        if random.random() < 0.25 * dt:
            fireworks.launch(speed_scale=speed_scale)
        fireworks.step(dt, speed_scale)
    if fireflies_enabled: fireflies.step({}, dt)
    if nature_enabled: nature.maybe_step({}, {})

    # --- dots simulation (simple fill) ---
    scene_surf.fill((0,0,0,255))
    draw_background(scene_surf, BG_THEMES[bg_ix], {}, time.time(), WIDTH, HEIGHT)

    # Trails
    if trail_ix == 0: trail_surf.fill((0,0,0,0))
    else: pygame.draw.rect(trail_surf, trail_strengths[trail_ix], (0,0,WIDTH,HEIGHT))

    # invert optional
    if invert_output:
        arr = pygame.surfarray.pixels3d(scene_surf); arr[:] = 255 - arr; del arr

    # contrast modes
    if contrast_mode == 1:
        crush_blacks(scene_surf, amount=0.22); apply_vignette(scene_surf, amount=0.45)
    elif contrast_mode == 2:
        crush_blacks(scene_surf, amount=0.35); apply_vignette(scene_surf, amount=0.65)

    # fireworks drawn on top
    if fireworks_enabled:
        _fw = pygame.Surface((WIDTH,HEIGHT), pygame.SRCALPHA)
        fireworks.draw(_fw)
        scene_surf.blit(_fw,(0,0),special_flags=pygame.BLEND_ADD)

    if fireflies_enabled: fireflies.draw(scene_surf, {})
    if nature_enabled: nature.draw_overlays(scene_surf)

    screen.blit(scene_surf, (0,0))
    pygame.display.flip()

# ===== Cleanup =====
stream.stop_stream(); stream.close(); p.terminate(); pygame.quit()
