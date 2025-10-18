#!/usr/bin/env python3
# warpfield_offline_4k.py — 4K/30fps offline renderer with concatenated WAV soundtrack

import os, sys, math, random, colorsys, wave, struct, subprocess
import numpy as np
import pygame

# ------------ Output & audio params ------------
WIDTH, HEIGHT = 3840, 2160
FPS = 30
SR = 44100            # must match soundtrack.wav
SAMPLES_PER_FRAME = int(round(SR / FPS))  # ~1470 at 44.1k
N_FFT = 2048          # analysis window (>= SAMPLES_PER_FRAME)
AUDIO_FILE = sys.argv[1] if len(sys.argv) > 1 else "soundtrack.wav"
OUT_FILE   = sys.argv[2] if len(sys.argv) > 2 else "output_4k30.mp4"

# ------------ Field params ------------
GRID_SPACING = 48
DOT_BASE_RADIUS = 4
DOT_MAX_BOOST = 5
NUM_INFLUENCERS = 12
INFLUENCE_RADIUS = 260.0
FALLOFF_POWER = 2.2
BASE_STRENGTH = 1.0
SWIRL_TWIST = 1.4

# disable window; render to surfaces
os.environ["SDL_VIDEODRIVER"] = "dummy"
pygame.init()
screen = pygame.Surface((WIDTH, HEIGHT), flags=pygame.SRCALPHA)
clock = pygame.time.Clock()

CENTER = (WIDTH // 2, HEIGHT // 2)

# ------------ Utils ------------
def lerp(a, b, t): return a + (b - a) * t

def hsv255(h, s, v):
    r, g, b = colorsys.hsv_to_rgb(h % 1.0, max(0.0, min(1.0, s)), max(0.0, min(1.0, v)))
    return (int(r*255), int(g*255), int(b*255))

# ------------ Dots ------------
dots = []
NUM_COLS = WIDTH // GRID_SPACING
NUM_ROWS = HEIGHT // GRID_SPACING
for row in range(NUM_ROWS):
    for col in range(NUM_COLS):
        x = col * GRID_SPACING + GRID_SPACING // 2
        y = row * GRID_SPACING + GRID_SPACING // 2
        dots.append({'pos':[float(x), float(y)], 'home':[float(x), float(y)]})

# ------------ Influencers ------------
class Influencer:
    def __init__(self, x, y, strength=BASE_STRENGTH, radius=INFLUENCE_RADIUS, mode='attract'):
        self.pos = [float(x), float(y)]
        self.strength = strength
        self.radius = radius
        self.mode = mode
    def field(self, px, py):
        dx = self.pos[0] - px
        dy = self.pos[1] - py
        d = math.hypot(dx, dy) + 1e-6
        if d > self.radius: return 0.0, 0.0, 0.0
        t = max(0.0, 1.0 - (d / self.radius) ** FALLOFF_POWER)
        ndx, ndy = dx/d, dy/d
        if self.mode == 'attract':
            fx, fy = ndx, ndy
        elif self.mode == 'repel':
            fx, fy = -ndx, -ndy
        elif self.mode == 'swirl':
            fx, fy = -ndy, ndx
            twist = SWIRL_TWIST * t
            fx = fx * (0.7 + 0.3*twist) + 0.2*ndx
            fy = fy * (0.7 + 0.3*twist) + 0.2*ndy
        else:
            fx, fy = ndx, ndy
        mag = self.strength * t
        return fx*mag, fy*mag, mag

def preset_ring(mode='attract', radius=None):
    r = radius or min(WIDTH, HEIGHT) * 0.32
    return [Influencer(CENTER[0] + r*math.cos(2*math.pi*i/NUM_INFLUENCERS),
                       CENTER[1] + r*math.sin(2*math.pi*i/NUM_INFLUENCERS),
                       mode=mode)
            for i in range(NUM_INFLUENCERS)]

def preset_spiral(mode='attract', turns=1.6):
    infs = []
    r_max = min(WIDTH, HEIGHT) * 0.42
    for i in range(NUM_INFLUENCERS):
        t = i / max(1, NUM_INFLUENCERS-1)
        a = 2 * math.pi * turns * t
        r = lerp(r_max*0.05, r_max, t)
        x = CENTER[0] + r * math.cos(a)
        y = CENTER[1] + r * math.sin(a)
        infs.append(Influencer(x, y, mode=mode))
    return infs

def preset_flower(mode='attract', petals=7):
    R = min(WIDTH, HEIGHT) * 0.28
    infs = []
    for i in range(NUM_INFLUENCERS):
        t = i / NUM_INFLUENCERS
        a = 2 * math.pi * t
        r = R * (1.0 + 0.35 * math.cos(petals * a))
        x = CENTER[0] + r * math.cos(a)
        y = CENTER[1] + r * math.sin(a)
        infs.append(Influencer(x, y, mode=mode))
    return infs

def preset_starburst(mode='attract', arms=6):
    infs = []
    r_inner = min(WIDTH, HEIGHT) * 0.18
    r_outer = min(WIDTH, HEIGHT) * 0.36
    pts = []
    for k in range(arms):
        a = 2*math.pi*k/arms
        pts.append((CENTER[0]+r_outer*math.cos(a), CENTER[1]+r_outer*math.sin(a)))
        b = a + math.pi/arms
        pts.append((CENTER[0]+r_inner*math.cos(b), CENTER[1]+r_inner*math.sin(b)))
    for i in range(NUM_INFLUENCERS):
        x, y = pts[i % len(pts)]
        infs.append(Influencer(x, y, mode=mode))
    return infs

INFLUENCER_MODE = 'attract'
INFLUENCERS = preset_flower(mode='attract')

# ------------ Color Engine ------------
class ColorEngine:
    def __init__(self):
        self.base_h = 0.55
        self.flash = 0.0
        self.hue_drift = 0.0
    def tick(self, feat, params, dt):
        self.hue_drift += dt * (0.06 + 0.25*params["energy"] + 0.35*feat["flux"])
        if feat.get("onset", False):
            self.flash = min(1.0, self.flash + 0.65)
            self.base_h = (self.base_h + 0.08 + 0.12*feat["flux"]) % 1.0
        self.flash *= (0.90 ** (dt*FPS))
    def color_for(self, local_mag, feat, params, t_seconds):
        h = (self.base_h + 0.07*math.sin(t_seconds*0.6) + 0.12*self.hue_drift) % 1.0
        band_push = 0.15*params["low"] - 0.05*params["mid"] + 0.12*params["high"]
        h = (h + band_push) % 1.0
        s = 0.52 + 0.35*params["high"] + 0.30*feat["flux"] + 0.25*self.flash
        v = 0.38 + 0.55*params["energy"] + 0.45*min(1.0, local_mag*0.7) + 0.20*self.flash
        return hsv255(h, s, v)

color_engine = ColorEngine()

# ------------ Analyzer (flux/onset) + Conductor-lite ------------
class Analyzer:
    def __init__(self, n_fft):
        self.prev = np.zeros(n_fft//2+1, dtype=float)
        self.ema, self.var = 0.0, 0.0
    def update(self, sp):
        if sp.max() > 0: sp = sp / sp.max()
        diff = sp - self.prev
        flux = float(np.sum(np.clip(diff, 0, None)))
        # ema/var-ish
        k1, k2 = 0.25, 0.15
        self.ema = (1-k1)*self.ema + k1*flux
        d = flux - self.ema
        self.var = (1-k2)*self.var + k2*(d*d)
        thresh = self.ema + 0.9*math.sqrt(max(1e-6, self.var))
        onset = flux > thresh
        n = len(sp)
        low  = float(np.mean(sp[:max(2,n//8)]))
        mid  = float(np.mean(sp[n//8:n//3]))
        high = float(np.mean(sp[-n//6:]))
        energy = float(np.mean(sp))
        self.prev = sp
        return {"sp":sp, "energy":energy, "low":low, "mid":mid, "high":high, "flux":flux, "onset":onset}

an = Analyzer(N_FFT)

# very light “conductor”: choose a broad goal from band energy
def choose_goal(feat, last_goal):
    if feat["onset"]:
        return "starburst"
    if feat["low"] > 0.22 and feat["energy"] > 0.18:
        return "spiral"
    if feat["high"] > 0.20 and feat["energy"] > 0.14:
        return "lace"
    return "bloom"

# ------------ WAV reader ------------
wf = wave.open(AUDIO_FILE, "rb")
if wf.getframerate() != SR:
    print(f"[!] soundtrack sample rate {wf.getframerate()} != {SR}. Re-make soundtrack at {SR} Hz.", file=sys.stderr); sys.exit(1)
if wf.getnchannels() not in (1,2):
    print(f"[!] soundtrack channels must be 1 or 2, got {wf.getnchannels()}", file=sys.stderr); sys.exit(1)
channels = wf.getnchannels()

def read_samples(n):
    raw = wf.readframes(n)
    if not raw:
        return None
    fmt = "<" + "h"*(len(raw)//2)
    ints = np.frombuffer(raw, dtype=np.int16)
    if channels == 2:
        ints = ints.reshape(-1,2).mean(axis=1).astype(np.int16)
    return ints.astype(np.float32) / 32768.0

# rolling window for FFT
fft_buf = np.zeros(N_FFT, dtype=np.float32)

# ------------ ffmpeg writer ------------
ffmpeg_cmd = [
    "ffmpeg","-y",
    "-f","rawvideo","-pix_fmt","rgb24","-s",f"{WIDTH}x{HEIGHT}","-r",str(FPS),"-i","pipe:0",
    "-i", AUDIO_FILE,
    "-map","0:v:0","-map","1:a:0",
    "-c:v","libx264","-preset","veryfast","-crf","18","-pix_fmt","yuv420p",
    "-c:a","aac","-b:a","320k",
    "-shortest",
    OUT_FILE
]
proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

# ------------ Main render loop ------------
goal = "bloom"
frame_idx = 0
while True:
    sam = read_samples(SAMPLES_PER_FRAME)
    if sam is None:
        break
    # update FFT buffer (simple overlap-add / sliding window)
    fft_buf = np.roll(fft_buf, -len(sam))
    fft_buf[-len(sam):] = sam
    spectrum = np.abs(np.fft.rfft(fft_buf))

    feat = an.update(spectrum)
    goal = choose_goal(feat, goal)

    # morph influencers per goal
    if goal == "bloom":
        INFLUENCER_MODE = "attract"
        base_r = min(WIDTH, HEIGHT) * (0.26 + 0.09*feat["low"])
        INFLUENCERS = preset_ring(mode=INFLUENCER_MODE, radius=base_r)
    elif goal == "starburst":
        INFLUENCER_MODE = "repel"
        INFLUENCERS = preset_starburst(mode=INFLUENCER_MODE, arms=6)
    elif goal == "spiral":
        INFLUENCER_MODE = "attract"
        INFLUENCERS = preset_spiral(mode=INFLUENCER_MODE, turns=1.6)
        ang = 0.008 * frame_idx  # gentle spin
        ca, sa = math.cos(ang), math.sin(ang)
        for inf in INFLUENCERS:
            vx, vy = inf.pos[0]-CENTER[0], inf.pos[1]-CENTER[1]
            inf.pos[0] = CENTER[0] + vx*ca - vy*sa
            inf.pos[1] = CENTER[1] + vx*sa + vy*ca
    elif goal == "lace":
        INFLUENCER_MODE = "swirl"
        INFLUENCERS = preset_flower(mode=INFLUENCER_MODE, petals=7)
        SWIRL_TWIST = 1.2 + 1.0*(1.2*feat["high"])

    # audio-driven boosts
    strength_boost = 1.0 + 0.9*feat["energy"] + 0.8*feat["low"] + 0.8*feat["flux"]
    size_boost     = 1.0 + 0.8*feat["low"]   + 0.6*feat["flux"]

    for inf in INFLUENCERS:
        inf.strength = BASE_STRENGTH * strength_boost
        inf.radius   = INFLUENCE_RADIUS * (0.85 + 0.25*feat["mid"])

    # palette tick
    color_engine.tick({"flux":feat["flux"], "onset":feat["onset"]},
                      {"energy":feat["energy"], "low":feat["low"], "mid":feat["mid"], "high":feat["high"]},
                      1.0/FPS)

    # --------- draw frame ---------
    screen.fill((0,0,0,255))
    energy = feat["energy"]

    for d in dots:
        px, py = d['pos']
        fx_sum = fy_sum = mag_sum = 0.0
        for inf in INFLUENCERS:
            fx, fy, mag = inf.field(px, py)
            fx_sum += fx; fy_sum += fy; mag_sum += mag
        speed = 90.0 + 150.0 * energy + 220.0 * feat["flux"]
        px += fx_sum * speed / FPS
        py += fy_sum * speed / FPS
        # soft home spring
        hx, hy = d['home'][0]-px, d['home'][1]-py
        px += hx * 0.018
        py += hy * 0.018
        # clamp
        px = min(max(px, 6), WIDTH-6); py = min(max(py, 6), HEIGHT-6)
        d['pos'][0], d['pos'][1] = px, py

        r = DOT_BASE_RADIUS + min(DOT_MAX_BOOST, mag_sum * 2.2) * size_boost
        tsec = frame_idx / FPS
        col = color_engine.color_for(mag_sum, {"flux":feat["flux"]},{"energy":feat["energy"],"low":feat["low"],"mid":feat["mid"],"high":feat["high"]}, tsec)
        pygame.draw.circle(screen, col, (int(px), int(py)), int(r))

    # write raw frame to ffmpeg
    frame_bytes = pygame.image.tostring(screen, "RGB")
    proc.stdin.write(frame_bytes)

    frame_idx += 1
    # (no real-time sleep; we render as fast as CPU allows)

# finalize
proc.stdin.close()
proc.wait()
wf.close()
print(f"✅ Rendered {frame_idx} frames to {OUT_FILE}")
