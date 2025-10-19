#!/usr/bin/env python3
# lotus_kaleidoscope_visualizer.py
# Full kaleidoscope (blooming lotus) made of audio-reactive string-art petals.

import math, sys, numpy as np
import pygame, pyaudio
from scipy.fftpack import fft

# -------------------- config --------------------
WIDTH, HEIGHT = 1280, 720
FPS = 30
CHUNK = 1024
RATE = 44100
CHANNELS = 1

# Kaleidoscope
NUM_SLICES = 24              # even number works best for mirror pairs
SLICE_ANGLE = 360.0 / NUM_SLICES
WEDGE_RES   = 1024           # internal square resolution for clean rotation
OVERLAY_ALPHA = 235          # 0..255
PETAL_STRINGS = 220          # density of string “threads” per wedge
PETAL_RINGS   = 4            # layers of petals
GLOW_STRENGTH = 2            # 0..3 downscale/upsample glow passes

# Color
PALETTES = [
    [(255, 70, 120), (255, 180, 90), (255, 250, 180)],
    [(90, 200, 255), (130, 140, 255), (230, 200, 255)],
    [(120, 255, 180), (250, 220, 140), (255, 140, 200)],
    [(255, 255, 255), (200, 240, 255), (255, 210, 230)],
]
BG = (0, 0, 0)

# -------------------- audio --------------------
p = pyaudio.PyAudio()
def get_input_device_index():
    try:
        return p.get_default_input_device_info()['index']
    except Exception:
        print("No default input device. If you want system audio, select a monitor/loopback device.")
        sys.exit(1)

def open_audio_stream():
    return p.open(format=pyaudio.paInt16, channels=CHANNELS, rate=RATE, input=True,
                  input_device_index=get_input_device_index(), frames_per_buffer=CHUNK)
stream = open_audio_stream()

# -------------------- pygame --------------------
pygame.init()
pygame.display.set_caption("Lotus Kaleidoscope Visualizer")
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE | pygame.DOUBLEBUF)
clock = pygame.time.Clock()
is_full = False

def toggle_fullscreen():
    global is_full, screen, WIDTH, HEIGHT
    is_full = not is_full
    if is_full:
        screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN | pygame.DOUBLEBUF)
    else:
        screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE | pygame.DOUBLEBUF)

# -------------------- helpers --------------------
def soft_glow(surface, strength=1):
    if strength <= 0: return
    for _ in range(strength):
        w, h = surface.get_size()
        small = pygame.transform.smoothscale(surface, (max(1, w//3), max(1, h//3)))
        large = pygame.transform.smoothscale(small, (w, h))
        surface.blit(large, (0, 0), special_flags=pygame.BLEND_ADD)

def rose_point(theta, k, a, phase=0.0):
    # classic rose: r = a * cos(k*theta + phase) (absolute gives full petals)
    r = a * abs(math.cos(k * theta + phase))
    return r

def mix(a, b, t):
    return a * (1 - t) + b * t

def lerp_color(c1, c2, t):
    return (int(mix(c1[0], c2[0], t)), int(mix(c1[1], c2[1], t)), int(mix(c1[2], c2[2], t)))

def palette_time(pal, t):
    # loop colors over time t in [0..1]
    n = len(pal)
    x = t * n
    i0 = int(x) % n
    i1 = (i0 + 1) % n
    f = x - int(x)
    return lerp_color(pal[i0], pal[i1], f)

def band_energy(spec, lo, hi):
    lo_i = max(0, int(lo * len(spec)))
    hi_i = min(len(spec)-1, int(hi * len(spec)))
    if hi_i <= lo_i: return 0.0
    return float(np.mean(spec[lo_i:hi_i]))

# -------------------- wedge rendering --------------------
def build_wedge_mask(size, deg0, deg1):
    mask = pygame.Surface((size, size), pygame.SRCALPHA)
    cx = cy = size // 2
    r  = size // 2 + 2
    a0, a1 = math.radians(deg0), math.radians(deg1)
    p0 = (int(cx + r * math.cos(a0)), int(cy + r * math.sin(a0)))
    p1 = (int(cx + r * math.cos(a1)), int(cy + r * math.sin(a1)))
    pygame.draw.polygon(mask, (255, 255, 255, 255), [(cx, cy), p0, p1])
    return mask

def draw_lotus_wedge(size, spectrum_smooth, tsec):
    """Render dense string-art petals inside a square surface; mask later into a wedge."""
    surf = pygame.Surface((size, size), pygame.SRCALPHA)
    cx = cy = size // 2
    rad = size // 2 - 6

    # audio bands
    bass  = band_energy(spectrum_smooth, 0.00, 0.10)
    mid   = band_energy(spectrum_smooth, 0.10, 0.45)
    high  = band_energy(spectrum_smooth, 0.45, 0.95)

    # normalize
    bass  = min(1.0, bass * 2.5)
    mid   = min(1.0, mid  * 2.0)
    high  = min(1.0, high * 1.5)

    # dynamic params
    k_base = 3 + int(3 * (0.35 + 0.65 * mid))   # rose multiplier (petal count factor)
    a_base = rad * (0.65 + 0.25 * bass)         # petal size
    phase  = tsec * (0.8 + 1.6 * high)          # rotation of petals

    density = PETAL_STRINGS
    rings   = PETAL_RINGS
    pal = PALETTES[(pygame.time.get_ticks() // 1500) % len(PALETTES)]

    # Draw multiple petal rings with interpolated colors
    for r_i in range(rings):
        t_ring = r_i / max(1, rings - 1)
        k = k_base + r_i % 2            # alternate k slightly per ring
        a = a_base * (0.55 + 0.45 * (1 - t_ring))
        col = palette_time(pal, (t_ring + 0.2 * math.sin(tsec * 0.5)) % 1.0)
        col = (col[0], col[1], col[2], 210)

        # Precompute ring path points
        # We'll map theta in [0..π] to create “string” lines criss-crossing the petal
        thetas = np.linspace(0, math.pi, density, dtype=np.float64)
        pts = []
        for th in thetas:
            r = rose_point(th, k, a, phase)
            x = cx + r * math.cos(th)
            y = cy - r * math.sin(th)
            pts.append((x, y))

        # draw “strings”: connect mirrored indices to create a web
        step = max(1, density // 90)
        for i in range(0, density - 1, step):
            j = density - 1 - i
            x0, y0 = pts[i]
            x1, y1 = pts[j]
            pygame.draw.aaline(surf, col, (x0, y0), (x1, y1))

    return surf

def render_kaleidoscope(dest, center, radius, spectrum_smooth, tsec):
    size = radius * 2
    if size < 64: return

    # 1) build full square petal canvas
    base = draw_lotus_wedge(size, spectrum_smooth, tsec)

    # 2) optional glow
    base_copy = base.copy()
    soft_glow(base_copy, GLOW_STRENGTH)

    # 3) mask into a single wedge
    wedge_mask = build_wedge_mask(size, 0, SLICE_ANGLE)
    wedge = pygame.Surface((size, size), pygame.SRCALPHA)
    wedge.blit(base_copy, (0, 0))
    wedge.blit(wedge_mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)

    # 4) tile wedges around, alternating mirror for classic kaleidoscope
    out = pygame.Surface((size, size), pygame.SRCALPHA)
    for i in range(NUM_SLICES):
        img = wedge
        if i % 2 == 1:
            img = pygame.transform.flip(img, True, False)
        img = pygame.transform.rotate(img, i * SLICE_ANGLE)
        out.blit(img, (0, 0))

    # 5) circular alpha to keep a neat round edge
    circle = pygame.Surface((size, size), pygame.SRCALPHA)
    pygame.draw.circle(circle, (255, 255, 255, OVERLAY_ALPHA), (size//2, size//2), size//2)
    out.blit(circle, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)

    # 6) blit centered
    rect = out.get_rect(center=center)
    dest.blit(out, rect)

# -------------------- main loop --------------------
def main():
    global WIDTH, HEIGHT, screen
    # simple spectral smoothing
    smooth = np.zeros(CHUNK//2, dtype=np.float32)
    alpha = 0.65  # smoothing factor (higher = smoother)

    running = True
    while running:
        dt = clock.tick(FPS) * 0.001
        tsec = pygame.time.get_ticks() * 0.001

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.VIDEORESIZE:
                WIDTH, HEIGHT = e.w, e.h
                screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE | pygame.DOUBLEBUF)
            elif e.type == pygame.KEYDOWN:
                if e.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif e.key in (pygame.K_f, pygame.K_F11):
                    toggle_fullscreen()

        # audio -> spectrum (mono)
        try:
            data = np.frombuffer(stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16)
        except IOError:
            continue
        if CHANNELS == 2:
            data = data.reshape(-1, 2).mean(axis=1).astype(np.int16)

        spec = np.abs(fft(data))[:CHUNK//2].astype(np.float32)
        m = spec.max()
        if m > 1e-6: spec /= m
        smooth = alpha * smooth + (1 - alpha) * spec

        screen.fill(BG)

        # lotus kaleidoscope occupies most of the screen
        radius = max(64, int(0.47 * min(WIDTH, HEIGHT)))
        render_kaleidoscope(screen, (WIDTH//2, HEIGHT//2), radius, smooth, tsec)

        pygame.display.flip()

    # cleanup
    stream.stop_stream(); stream.close(); p.terminate(); pygame.quit()

if __name__ == "__main__":
    main()
