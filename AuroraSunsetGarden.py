#!/usr/bin/env python3
"""
AuroraSunsetGarden — an audio‑reactive, abstract nature‑inspired visualizer.

Upgrades over the original warpfield_plotter:
- Rich backgrounds themed after natural phenomena (Sunset, Ocean, Forest, Night Sky) with
  smooth gradients and horizon bloom that breathe with low frequencies.
- Multiple geometry renderers: dots, petals, triangles, quads, star‑shards, ribbon lines,
  and soft sprites (metaball‑ish discs). Cycle live.
- Kaleidoscope symmetry (1, 2, 4, 6 sectors) for mandala looks.
- Layered persistence: independent afterimage fade for scene, particles, and UI decals.
- Higher‑resolution audio responsiveness: sub‑bass, bass, low‑mid, high‑mid, presence, air
  bands + per‑band routing to different visual subsystems.
- Sparkle bursts and comet trails on high‑band onsets with varying lifetimes.
- Field presets expanded and smoothed rotation; time‑varying flower/spiral morphs.
- Screenshot key, HUD improvements, and quality‑of‑life toggles.

Keys (while running):
  Esc    — Quit
  f      — Fullscreen
  i      — Toggle influencer overlays
  a      — Toggle audio coupling on/off
  m      — Cycle influencer mode (attract→repel→swirl)
  1..5   — Base layout preset (ring/star/spiral/flower/scaffold)
  g      — Cycle geometry renderer (dots→petals→tri→quad→star→ribbons→soft)
  k      — Cycle kaleidoscope sectors (1→2→4→6)
  b      — Cycle background theme (Sunset→Ocean→Forest→Night)
  t      — Cycle trail strength (off→light→medium→long)
  s      — Save screenshot to ./screenshots

Requires: pygame, numpy, pyaudio
Run tip: SDL_AUDIODRIVER=pulse python3 aurora_sunset_garden.py
"""

import pygame
import numpy as np
import pyaudio
import math
import random
import colorsys
import time
import os
from collections import deque

# ===== Config =====
CHUNK = 512
CHANNELS = 1
RATE = 22050
FPS = 60

GRID_SPACING = 42
DOT_BASE_RADIUS = 3.5
DOT_MAX_BOOST = 6
DEFAULT_DOT_COLOR = (185, 220, 255)

NUM_INFLUENCERS = 14
INFLUENCE_RADIUS = 280.0
FALLOFF_POWER = 2.15
BASE_STRENGTH = 1.0
SWIRL_TWIST = 1.4

AUDIO_COUPLED = True
SHOW_INFLUENCERS = False

# Prefer Pulse/Default device to avoid ALSA/JACK spam
PREFERRED_DEVICE_SUBSTR = "pulse"   # set to part of your desired device name, or "" for default

# ===== Init =====
if "SDL_AUDIODRIVER" not in os.environ:
    os.environ["SDL_AUDIODRIVER"] = "pulse"

pygame.init()
screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
info = pygame.display.Info()
WIDTH, HEIGHT = info.current_w, info.current_h
CENTER = (WIDTH // 2, HEIGHT // 2)
clock = pygame.time.Clock()

# Layered surfaces for afterimages/persistence
scene_surf = pygame.Surface((WIDTH, HEIGHT)).convert_alpha()
scene_surf.fill((0,0,0,255))
trail_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
trail_strengths = [(255,255,255,0), (0,0,0,15), (0,0,0,28), (0,0,0,42)]  # clear, light, med, long
trail_ix = 2  # start with medium

# ===== Audio =====
p = pyaudio.PyAudio()

def find_preferred_input_index(substr):
    substr = (substr or "").lower()
    best = None
    for i in range(p.get_device_count()):
        d = p.get_device_info_by_index(i)
        if d.get("maxInputChannels", 0) < 1:
            continue
        name = (d.get("name") or "").lower()
        if substr and substr in name:
            return i
        if best is None:
            best = i
    return best

def open_audio_stream():
    idx = find_preferred_input_index(PREFERRED_DEVICE_SUBSTR)
    if idx is None:
        idx = p.get_default_input_device_info()["index"]
    return p.open(format=pyaudio.paInt16,
                  channels=CHANNELS,
                  rate=RATE,
                  input=True,
                  input_device_index=idx,
                  frames_per_buffer=CHUNK)

stream = open_audio_stream()

# ===== Utils =====

def hsv255(h, s, v):
    r, g, b = colorsys.hsv_to_rgb(h % 1.0, max(0.0, min(1.0, s)), max(0.0, min(1.0, v)))
    return (int(r*255), int(g*255), int(b*255))

def lerp(a, b, t):
    return a + (b - a) * t

def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x

# ===== Color engine (audio-reactive palette) =====
class ColorEngine:
    """Nature‑tinted palette that follows energy/flux and theme tints."""
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

    def color_for(self, local_mag, feat, params, t_seconds, theme_hue_push=0.0):
        h = (self.base_h + 0.07*math.sin(t_seconds*0.6) + 0.12*self.hue_drift + theme_hue_push) % 1.0
        band_push = 0.12*params["bass"] - 0.04*params["lowmid"] + 0.10*params["air"]
        h = (h + band_push) % 1.0
        s = 0.50 + 0.33*params["presence"] + 0.28*feat["flux"] + 0.22*self.flash
        v = 0.36 + 0.56*params["energy"] + 0.46*min(1.0, local_mag*0.7) + 0.22*self.flash
        return hsv255(h, s, v)

# ===== Dots (grid anchors) =====
dots = []
NUM_COLS = WIDTH // GRID_SPACING
NUM_ROWS = HEIGHT // GRID_SPACING
for row in range(NUM_ROWS):
    for col in range(NUM_COLS):
        x = col * GRID_SPACING + GRID_SPACING // 2
        y = row * GRID_SPACING + GRID_SPACING // 2
        dots.append({
            'pos': [float(x), float(y)],
            'home': [float(x), float(y)],
            'color': DEFAULT_DOT_COLOR,
        })

# ===== Influencers =====
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
        if d > self.radius:
            return 0.0, 0.0, 0.0
        t = max(0.0, 1.0 - (d / self.radius) ** FALLOFF_POWER)
        ndx, ndy = dx / d, dy / d
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
        return fx * mag, fy * mag, mag

# ===== Presets =====
def preset_ring(mode='attract', radius=None):
    r = radius or min(WIDTH, HEIGHT) * 0.34
    return [Influencer(CENTER[0] + r*math.cos(2*math.pi*i/NUM_INFLUENCERS),
                       CENTER[1] + r*math.sin(2*math.pi*i/NUM_INFLUENCERS),
                       strength=BASE_STRENGTH, mode=mode)
            for i in range(NUM_INFLUENCERS)]

def preset_starburst(mode='attract', arms=6):
    infs = []
    r_inner = min(WIDTH, HEIGHT) * 0.20
    r_outer = min(WIDTH, HEIGHT) * 0.38
    pts = []
    for k in range(arms):
        a = 2*math.pi*k/arms
        pts.append((CENTER[0]+r_outer*math.cos(a), CENTER[1]+r_outer*math.sin(a)))
        b = a + math.pi/arms
        pts.append((CENTER[0]+r_inner*math.cos(b), CENTER[1]+r_inner*math.sin(b)))
    for i in range(NUM_INFLUENCERS):
        x, y = pts[i % len(pts)]
        infs.append(Influencer(x, y, strength=BASE_STRENGTH, mode=mode))
    return infs

def preset_spiral(mode='attract', turns=1.9):
    infs = []
    r_max = min(WIDTH, HEIGHT) * 0.44
    for i in range(NUM_INFLUENCERS):
        t = i / max(1, NUM_INFLUENCERS - 1)
        a = 2 * math.pi * turns * t
        r = lerp(r_max*0.05, r_max, t)
        x = CENTER[0] + r * math.cos(a)
        y = CENTER[1] + r * math.sin(a)
        infs.append(Influencer(x, y, strength=BASE_STRENGTH, mode=mode))
    return infs

def preset_flower(mode='attract', petals=8, wobble=0.0):
    R = min(WIDTH, HEIGHT) * 0.30
    infs = []
    for i in range(NUM_INFLUENCERS):
        t = i / NUM_INFLUENCERS
        a = 2 * math.pi * t
        r = R * (1.0 + 0.34 * math.cos(petals * a + wobble))
        x = CENTER[0] + r * math.cos(a)
        y = CENTER[1] + r * math.sin(a)
        infs.append(Influencer(x, y, strength=BASE_STRENGTH, mode=mode))
    return infs

def preset_scaffold(mode='attract'):
    return [Influencer(random.uniform(WIDTH*0.2, WIDTH*0.8),
                       random.uniform(HEIGHT*0.2, HEIGHT*0.8),
                       strength=BASE_STRENGTH, mode=mode)
            for _ in range(NUM_INFLUENCERS)]

def build_preset(name, mode, **kw):
    return {
        'ring':     lambda: preset_ring(mode, **kw),
        'star':     lambda: preset_starburst(mode, **kw),
        'spiral':   lambda: preset_spiral(mode, **kw),
        'flower':   lambda: preset_flower(mode, **kw),
        'scaffold': lambda: preset_scaffold(mode),
    }.get(name, lambda: preset_ring(mode))()

INFLUENCER_MODE = 'attract'
current_preset = 'flower'
INFLUENCERS = build_preset(current_preset, INFLUENCER_MODE)

# ===== Conductor =====
class Conductor:
    def __init__(self):
        self.energy_slow = 0.0
        self.energy_fast = 0.0
        self.band_slow = {k:0.0 for k in ['sub','bass','lowmid','mid','highmid','presence','air']}
        self.last_preset_switch = time.time()
        self.goal = 'bloom'
        self.rotate = 0.0
        self.flower_phase = 0.0

    def bandpack(self, sp):
        n = len(sp)
        # Split rFFT bins into psycho-ish regions
        # indices proportional to logarithmic-ish scale
        i = lambda f: int(clamp(f/(RATE/2) * (n-1), 0, n-1))
        bands = {
            'sub':      float(np.mean(sp[i(20):i(60)+1])),
            'bass':     float(np.mean(sp[i(60):i(140)+1])),
            'lowmid':   float(np.mean(sp[i(140):i(400)+1])),
            'mid':      float(np.mean(sp[i(400):i(1000)+1])),
            'highmid':  float(np.mean(sp[i(1000):i(2500)+1])),
            'presence': float(np.mean(sp[i(2500):i(6000)+1])),
            'air':      float(np.mean(sp[i(6000):]))
        }
        bands['energy'] = float(np.mean(sp))
        return bands

    def update(self, spectrum):
        sp = spectrum
        if sp.max() > 0:
            sp = sp / sp.max()
        bands = self.bandpack(sp)

        # Smooth fast/slow and bands
        self.energy_fast = lerp(self.energy_fast, bands['energy'], 0.35)
        self.energy_slow = lerp(self.energy_slow, bands['energy'], 0.05)
        for k in self.band_slow:
            self.band_slow[k] = lerp(self.band_slow[k], bands[k], 0.10)

        # Burst detection
        burst = (self.energy_fast - self.energy_slow > 0.08) or (bands['mid'] - self.band_slow['mid'] > 0.07)

        # Goal selection
        now = time.time()
        if burst:
            self.goal = 'starburst'
            self.last_preset_switch = now
        else:
            if self.band_slow['bass'] > 0.22 and self.energy_slow > 0.18:
                self.goal = 'spiral'
            elif self.band_slow['air'] > 0.20 and self.energy_slow > 0.14:
                self.goal = 'lace'
            else:
                self.goal = 'bloom'

        self.rotate += 0.12 * (0.4 + self.energy_slow)
        self.flower_phase += 0.015 * (1.0 + 0.6*self.band_slow['presence'])

        if now - self.last_preset_switch > 12.0 and self.goal != 'starburst':
            self.last_preset_switch = now
            cycle = ['ring','flower','spiral','star']
            global current_preset, INFLUENCERS
            current_preset = cycle[(cycle.index(current_preset)+1) % len(cycle)]
            INFLUENCERS = build_preset(current_preset, INFLUENCER_MODE)

        out = {**self.band_slow}
        out.update({'energy': self.energy_slow, 'burst': burst, 'goal': self.goal, 'rotate': self.rotate, 'flower_phase': self.flower_phase})
        return out

# ===== Audio change analyzer (flux + onsets) =====
class AudioAnalyzer:
    def __init__(self, chunk):
        self.prev = np.zeros(chunk//2+1, dtype=float)
        self.flux_ema = 0.0
        self.flux_var = 0.0
        self.k1 = 0.25
        self.k2 = 0.15
        self.onset_cool = 0.0

    def update(self, spectrum):
        sp = spectrum.astype(float)
        if sp.max() > 0: sp = sp / sp.max()
        diff = sp - self.prev
        flux = float(np.sum(np.clip(diff, 0, None)))
        self.flux_ema = (1-self.k1)*self.flux_ema + self.k1*flux
        d = flux - self.flux_ema
        self.flux_var = (1-self.k2)*self.flux_var + self.k2*(d*d)
        thresh = self.flux_ema + 0.9*math.sqrt(max(1e-6, self.flux_var))
        onset = flux > thresh and self.onset_cool <= 0.0
        if onset: self.onset_cool = 0.10
        else: self.onset_cool = max(0.0, self.onset_cool - 1.0/FPS)
        self.prev = sp

        return {"sp":sp, "flux":flux, "onset":onset}

# ===== Transient Influencers (sparks/comets) =====
class TransientInfluencer(Influencer):
    def __init__(self, x, y, life=0.6, strength=1.5, radius=220, mode='repel'):
        super().__init__(x, y, strength=strength, radius=radius, mode=mode)
        self.life = life
        self.age = 0.0

    def step(self, dt):
        self.life -= dt
        self.age += dt
        return self.life > 0.0

# ===== Motif engine =====
class MotifEngine:
    def __init__(self):
        self.mode_ix = 0
        self.modes = ['ripple','starburst','swirlstorm']
        self.cool = 0.0
        self.transients = []
        self.sparkles = deque(maxlen=2000)

    def trigger(self, kind, center, band_boost):
        cx, cy = center
        if kind == 'ripple':
            r = min(WIDTH,HEIGHT)*(0.18 + 0.15*band_boost)
            for i in range(12):
                a = 2*math.pi*i/12.0
                x = cx + r*math.cos(a); y = cy + r*math.sin(a)
                self.transients.append(TransientInfluencer(x,y,life=0.7, strength=1.2+0.8*band_boost, radius=200, mode='attract'))
        elif kind == 'starburst':
            arms = 10
            r = min(WIDTH,HEIGHT)*(0.25 + 0.2*band_boost)
            for k in range(arms):
                a = 2*math.pi*k/arms
                x = cx + r*math.cos(a); y = cy + r*math.sin(a)
                self.transients.append(TransientInfluencer(x,y,life=0.5, strength=1.6+1.0*band_boost, radius=240, mode='repel'))
        elif kind == 'swirlstorm':
            for i in range(14):
                a = 2*math.pi*i/14.0
                r = 90 + 90*band_boost
                x = cx + r*math.cos(a); y = cy + r*math.sin(a)
                t = TransientInfluencer(x,y,life=0.85, strength=1.4+0.9*band_boost, radius=220, mode='swirl')
                self.transients.append(t)
        # seed sparkles
        for _ in range(80 + int(160*band_boost)):
            ang = random.uniform(0, 2*math.pi)
            spd = random.uniform(30, 280) * (0.5 + band_boost)
            life = random.uniform(0.25, 1.2)
            self.sparkles.append({
                'x': cx, 'y': cy,
                'vx': math.cos(ang)*spd, 'vy': math.sin(ang)*spd,
                'life': life, 'age': 0.0
            })
        self.cool = 0.12

    def maybe_trigger_from_features(self, feat, bands):
        if self.cool > 0:
            self.cool -= 1.0/FPS
            return
        if feat['onset']:
            kind = self.modes[self.mode_ix % len(self.modes)]
            self.mode_ix += 1
            band_boost = clamp(0.5*bands['bass'] + 0.4*bands['presence'] + 0.3*bands['air'], 0.0, 1.0)
            self.trigger(kind, CENTER, band_boost)

    def step(self, dt):
        self.transients = [t for t in self.transients if t.step(dt)]
        # update sparkles
        alive = deque(maxlen=self.sparkles.maxlen)
        for s in self.sparkles:
            s['age'] += dt
            if s['age'] < s['life']:
                s['x'] += s['vx']*dt
                s['y'] += s['vy']*dt
                s['vy'] += 12*dt  # gentle gravity
                if -50 <= s['x'] <= WIDTH+50 and -50 <= s['y'] <= HEIGHT+50:
                    alive.append(s)
        self.sparkles = alive

conductor = Conductor()
analyzer = AudioAnalyzer(CHUNK)
motifs = MotifEngine()
color_engine = ColorEngine()

# ===== Background themes (nature inspired) =====
BG_THEMES = [
    'Sunset',  # warm horizon glow
    'Ocean',   # teal‑blue with foam glints
    'Forest',  # deep greens with firefly hints
    'Night'    # indigo to violet with starlight
]
bg_ix = 0


def draw_background(surface, theme, bands, t):
    # low frequencies breathe the horizon; presence/air add glints
    sub = bands['sub']; bass = bands['bass']; air = bands['air']; presence = bands['presence']
    if theme == 'Sunset':
        top = hsv255(0.60 + 0.02*math.sin(t*0.2), 0.35, 0.12 + 0.18*air)
        bot = hsv255(0.05 + 0.03*math.sin(t*0.15), 0.85, 0.46 + 0.42*(0.5*bass+0.5*sub))
    elif theme == 'Ocean':
        top = hsv255(0.50, 0.30, 0.10 + 0.20*air)
        bot = hsv255(0.52, 0.75, 0.38 + 0.45*(0.4*bass+0.6*presence))
    elif theme == 'Forest':
        top = hsv255(0.33, 0.40, 0.12 + 0.16*air)
        bot = hsv255(0.33, 0.80, 0.36 + 0.45*(0.5*bass+0.5*sub))
    else:  # Night
        top = hsv255(0.70, 0.25, 0.10 + 0.25*air)
        bot = hsv255(0.75, 0.65, 0.28 + 0.40*(0.3*bass+0.7*presence))

    # vertical gradient
    for y in range(0, HEIGHT, 4):
        k = y / HEIGHT
        c = (
            int(top[0]*(1-k) + bot[0]*k),
            int(top[1]*(1-k) + bot[1]*k),
            int(top[2]*(1-k) + bot[2]*k)
        )
        pygame.draw.rect(surface, c, (0, y, WIDTH, 4))

    # soft horizon bloom reacting to bass
    horizon_y = int(HEIGHT*0.62)
    bloom = int(40 + 220*(0.4*bass+0.6*sub))
    surf = pygame.Surface((WIDTH, bloom), pygame.SRCALPHA)
    g = int(70 + 150*(0.5*bass+0.5*sub))
    pygame.draw.rect(surf, (255, 255, 255, g), (0, 0, WIDTH, bloom))
    surface.blit(surf, (0, horizon_y - bloom//2), special_flags=pygame.BLEND_PREMULTIPLIED)

# ===== Waterfall + Fireworks Systems (bedazzling overlay) =====
class WaterfallSystem:
    def __init__(self):
        self.offset = 0.0
        # precompute columns
        self.columns = []
        col_w = 8
        for x in range(0, WIDTH, col_w):
            phase = random.random()*2*math.pi
            width = col_w + random.randint(0, 6)
            alpha = random.randint(20, 45)
            self.columns.append({'x': x, 'w': width, 'phase': phase, 'alpha': alpha})
        self.mist = pygame.Surface((WIDTH, int(HEIGHT*0.25)), pygame.SRCALPHA)

    def draw(self, surface, bands, dt):
        # strengthen with presence/air for shiny spray; bass/sub increase flow speed
        speed = 80 + 220*(0.5*bands['bass']+0.5*bands['sub'])
        self.offset = (self.offset + speed*dt) % HEIGHT
        veil = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        for c in self.columns:
            amp = 18 + 34*bands['presence']
            sway = math.sin(self.offset*0.005 + c['phase'])*amp
            a = int(c['alpha'] + 80*bands['air'])
            x = int(c['x'] + sway)
            pygame.draw.rect(veil, (200, 220, 255, clamp(a, 10, 180)), (x, 0, c['w'], HEIGHT))
        # gentle gradient for the veil intensity
        surface.blit(veil, (0,0), special_flags=pygame.BLEND_PREMULTIPLIED)
        # base mist near bottom
        self.mist.fill((0,0,0,0))
        fog_a = int(30 + 120*(0.4*bands['presence']+0.6*bands['air']))
        pygame.draw.rect(self.mist, (230, 235, 255, clamp(fog_a, 20, 160)), (0, 0, WIDTH, self.mist.get_height()))
        surface.blit(self.mist, (0, int(HEIGHT*0.75)), special_flags=pygame.BLEND_PREMULTIPLIED)

class FireworksSystem:
    def __init__(self):
        self.shells = []
        self.sparks = []
        self.glow = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)

    def launch(self, x=None):
        x = x if x is not None else random.randint(int(WIDTH*0.2), int(WIDTH*0.8))
        y = random.randint(int(HEIGHT*0.12), int(HEIGHT*0.35))
        vy = -random.uniform(180, 260)
        color = hsv255(random.random(), 0.8, 1.0)
        self.shells.append({'x':x, 'y':HEIGHT-10, 'vx':random.uniform(-40,40), 'vy':vy, 't':0.0, 'color':color, 'exploded':False})

    def maybe_launch(self, feat, bands):
        # on onset or strong presence/air, launch 1-3 shells
        if feat['onset'] or (bands['presence']+bands['air'] > 0.45):
            for _ in range(1, 1+int(2+3*(bands['presence']+bands['air']))):
                self.launch()

    def step(self, dt):
        # update shells
        new_shells = []
        for s in self.shells:
            s['t'] += dt
            s['x'] += s['vx']*dt
            s['y'] += s['vy']*dt
            s['vy'] += 140*dt  # gravity
            if s['vy'] > -20 or s['t']>1.2:  # explode near apex or timeout
                self.explode(s)
            else:
                new_shells.append(s)
        self.shells = new_shells
        # update sparks
        alive = []
        for sp in self.sparks:
            sp['age'] += dt
            if sp['age'] >= sp['life']:
                continue
            sp['x'] += sp['vx']*dt
            sp['y'] += sp['vy']*dt
            sp['vy'] += 220*dt  # pull downward cascade
            sp['vx'] *= (0.985)  # light air drag
            alive.append(sp)
        self.sparks = alive

    def explode(self, shell):
        cx, cy = shell['x'], shell['y']
        base_col = shell['color']
        count = random.randint(80, 140)
        for i in range(count):
            ang = random.random()*2*math.pi
            spd = random.uniform(60, 360)
            life = random.uniform(0.6, 1.8)
            hue_jitter = (random.random()-0.5)*0.06
            r,g,b = base_col
            h,s,v = colorsys.rgb_to_hsv(r/255.0,g/255.0,b/255.0)
            col = hsv255(h+hue_jitter, min(1.0,s*1.1), v)
            self.sparks.append({'x':cx,'y':cy,'vx':math.cos(ang)*spd,'vy':math.sin(ang)*spd,'life':life,'age':0.0,'color':col})
        shell['exploded']=True

    def draw(self, surface):
        # subtle bloom
        self.glow.fill((0,0,0,0))
        for sp in self.sparks:
            k = 1.0 - sp['age']/sp['life']
            a = int(200*k)
            size = max(1, int(2 + 2*k))
            c = (*sp['color'], a)
            pygame.draw.circle(self.glow, c, (int(sp['x']), int(sp['y'])), size)
        surface.blit(self.glow, (0,0), special_flags=pygame.BLEND_PREMULTIPLIED)

# ===== Nature Director (randomized nature-inspired changes) =====
class NatureDirector:
    """Occasional wind gusts, seasonal tint drift, lightning flashes, and migrating flocks."""
    def __init__(self):
        self.running = True
        self.wind_phase = random.random()*1000
        self.wind_dir = random.random()*2*math.pi
        self.wind_gust = 0.0  # 0..1
        self.season_hue = random.uniform(-0.04, 0.04)
        self.lightning = 0.0  # flash alpha 0..1
        self.next_event = time.time() + random.uniform(3,8)
        self.flock = []  # list of birds

    def maybe_step(self, feat, bands):
        if not self.running: return
        now = time.time()
        if now >= self.next_event:
            choice = random.random()
            if choice < 0.4:  # wind gust
                self.wind_dir = random.random()*2*math.pi
                self.wind_gust = min(1.0, 0.4 + 0.8*(0.5*bands['presence']+0.5*bands['air']))
            elif choice < 0.6:  # seasonal hue nudge
                self.season_hue = clamp(self.season_hue + random.uniform(-0.03,0.03), -0.12, 0.12)
            elif choice < 0.8 and (feat['onset'] or bands['presence']>0.35):  # lightning
                self.lightning = 1.0
            else:  # flock
                self.spawn_flock(bands)
            self.next_event = now + random.uniform(2.5, 7.5)
        # decay lightning
        self.lightning *= 0.88
        # ease gust
        self.wind_gust *= 0.985

    def wind(self, x, y, t):
        # gentle spatially-coherent wind using time-varying sines
        base = 40 * self.wind_gust
        kx = math.sin(0.07*t + 0.0008*y)
        ky = math.sin(0.09*t + 0.0008*x)
        dirx = math.cos(self.wind_dir); diry = math.sin(self.wind_dir)
        return base*(dirx*0.6 + 0.4*kx), base*(diry*0.6 + 0.4*ky)

    def spawn_flock(self, bands):
        # spawn simple V-birds crossing the sky
        count = random.randint(6, 12)
        y = random.randint(int(HEIGHT*0.10), int(HEIGHT*0.35))
        speed = 120 + 140*bands['air']
        left_to_right = random.random() < 0.5
        x0 = -60 if left_to_right else WIDTH+60
        vx = speed if left_to_right else -speed
        self.flock = [{'x':x0 + i*20, 'y':y + (i%5-2)*6, 'vx':vx} for i in range(count)]

    def draw_overlays(self, surface):
        # lightning screen flash
        if self.lightning > 0.02:
            a = int(160*self.lightning)
            flash = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            flash.fill((255,255,255,a))
            surface.blit(flash, (0,0), special_flags=pygame.BLEND_PREMULTIPLIED)
        # draw flock silhouettes
        if self.flock:
            birds_alive = []
            for b in self.flock:
                b['x'] += b['vx']/FPS
                if -100 <= b['x'] <= WIDTH+100:
                    birds_alive.append(b)
                    # small V shape
                    p1 = (int(b['x']), int(b['y']))
                    p2 = (int(b['x']-6), int(b['y']+4))
                    p3 = (int(b['x']+6), int(b['y']+4))
                    pygame.draw.line(surface, (20,20,25), p1, p2, 2)
                    pygame.draw.line(surface, (20,20,25), p1, p3, 2)
            self.flock = birds_alive

# ===== Fireflies =====
class Fireflies:
    def __init__(self, n=180):
        self.n = n
        self.points = [{'x':random.uniform(0,WIDTH),'y':random.uniform(HEIGHT*0.55,HEIGHT*0.95),'a':random.random(),'r':random.uniform(1.5,3.0)} for _ in range(n)]
        self.surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)

    def step(self, bands, dt):
        # presence/air brighten; bass causes gentle drift
        for p in self.points:
            p['a'] += (random.uniform(-0.6,0.6) + 0.6*bands['air'])*dt
            p['x'] += (random.uniform(-12,12) + 18*(bands['bass']))*dt
            p['y'] += random.uniform(-10,10)*dt
            p['x'] = clamp(p['x'], 0, WIDTH)
            p['y'] = clamp(p['y'], HEIGHT*0.50, HEIGHT-5)

    def draw(self, surface, bands):
        self.surf.fill((0,0,0,0))
        for p in self.points:
            k = (0.5 + 0.5*math.sin(p['a']*6)) * (0.4 + 0.6*(0.6*bands['presence']+0.4*bands['air']))
            c = hsv255(0.17+0.05*random.random(), 0.6, 0.6+0.4*k)
            pygame.draw.circle(self.surf, (*c, int(80+140*k)), (int(p['x']), int(p['y'])), int(p['r']))
        surface.blit(self.surf, (0,0), special_flags=pygame.BLEND_PREMULTIPLIED)

# ===== Geometry renderers =====
RENDERERS = ['dots','petals','tri','quad','star','ribbons','soft']
renderer_ix = 0


def draw_shape(surface, name, pos, size, rot, color):
    x, y = pos
    if name == 'dots':
        pygame.draw.circle(surface, color, (int(x), int(y)), int(size))
    elif name == 'soft':
        # soft sprite disc
        r = int(size*1.7)
        s = pygame.Surface((r*2, r*2), pygame.SRCALPHA)
        pygame.draw.circle(s, (*color, 110), (r, r), r)
        pygame.draw.circle(s, (*color, 200), (r, r), int(r*0.66))
        surface.blit(s, (int(x-r), int(y-r)), special_flags=pygame.BLEND_PREMULTIPLIED)
    else:
        points = []
        if name == 'petals':
            k = 5
            for i in range(k):
                a = rot + 2*math.pi*i/k
                r = size*(1.4 if i%2==0 else 0.7)
                points.append((x + r*math.cos(a), y + r*math.sin(a)))
        elif name == 'tri':
            k = 3
            for i in range(k):
                a = rot + 2*math.pi*i/k
                points.append((x + size*1.2*math.cos(a), y + size*1.2*math.sin(a)))
        elif name == 'quad':
            k = 4
            for i in range(k):
                a = rot + math.pi/4 + 2*math.pi*i/k
                points.append((x + size*1.1*math.cos(a), y + size*1.1*math.sin(a)))
        elif name == 'star':
            k = 5
            for i in range(k*2):
                a = rot + 2*math.pi*i/(k*2)
                r = size*(1.6 if i%2==0 else 0.6)
                points.append((x + r*math.cos(a), y + r*math.sin(a)))
        elif name == 'ribbons':
            # short ribbon segment in motion direction
            length = size*3
            dx = math.cos(rot); dy = math.sin(rot)
            p1 = (x - dx*length*0.5, y - dy*length*0.5)
            p2 = (x + dx*length*0.5, y + dy*length*0.5)
            pygame.draw.line(surface, color, p1, p2, max(1, int(size*0.8)))
            return
        if points:
            pygame.draw.polygon(surface, color, points)

# ===== Kaleidoscope symmetry =====
kaleido_sectors = [1, 2, 4, 6]
kaleido_ix = 0

def blit_with_symmetry(base, dest):
    sectors = kaleido_sectors[kaleido_ix]
    if sectors == 1:
        dest.blit(base, (0,0))
        return
    # draw in a wedge, then rotate/mirror
    # For simplicity, just rotate the whole frame copies
    for s in range(sectors):
        angle = (360/sectors)*s
        rotated = pygame.transform.rotozoom(base, angle, 1.0)
        if s % 2 == 1:
            rotated = pygame.transform.flip(rotated, True, False)
        dest.blit(rotated, (0,0), special_flags=pygame.BLEND_ADD)

# ===== Renderer state =====
last_positions = {}  # for ribbons orientation

# Waterfall / Fireworks toggles
fireworks_enabled = True
waterfall_enabled = True
haze_enabled = True

fireworks = FireworksSystem()
waterfall = WaterfallSystem()

# Nature systems
nature_enabled = True
fireflies_enabled = True

nature = NatureDirector()
fireflies = Fireflies(n=140)

# ===== HUD font =====
hud_font = pygame.font.SysFont(None, 24)

hud_font = pygame.font.SysFont(None, 24)

# ===== Main objects =====
conductor = Conductor()
analyzer = AudioAnalyzer(CHUNK)
motifs = MotifEngine()
color_engine = ColorEngine()
hue_inverted = False

running = True
while running:
    dt = clock.tick(FPS) / 1000.0

    # Events/controls
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            elif event.key == pygame.K_f:
                screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
            elif event.key == pygame.K_i:
                SHOW_INFLUENCERS = not SHOW_INFLUENCERS
            elif event.key == pygame.K_a:
                AUDIO_COUPLED = not AUDIO_COUPLED
            elif event.key == pygame.K_r:
                INFLUENCERS = preset_scaffold(mode=INFLUENCER_MODE)
                current_preset = 'scaffold'
            elif event.key == pygame.K_m:
                INFLUENCER_MODE = {'attract':'repel','repel':'swirl','swirl':'attract'}[INFLUENCER_MODE]
                for inf in INFLUENCERS:
                    inf.mode = INFLUENCER_MODE
            elif event.key == pygame.K_1:
                current_preset = 'ring'; INFLUENCERS = build_preset('ring', INFLUENCER_MODE)
            elif event.key == pygame.K_2:
                current_preset = 'star'; INFLUENCERS = build_preset('star', INFLUENCER_MODE)
            elif event.key == pygame.K_3:
                current_preset = 'spiral'; INFLUENCERS = build_preset('spiral', INFLUENCER_MODE)
            elif event.key == pygame.K_4:
                current_preset = 'flower'; INFLUENCERS = build_preset('flower', INFLUENCER_MODE)
            elif event.key == pygame.K_5:
                current_preset = 'scaffold'; INFLUENCERS = build_preset('scaffold', INFLUENCER_MODE)
            elif event.key == pygame.K_g:
                renderer_ix = (renderer_ix + 1) % len(RENDERERS)
            elif event.key == pygame.K_k:
                kaleido_ix = (kaleido_ix + 1) % len(kaleido_sectors)
            elif event.key == pygame.K_b:
                bg_ix = (bg_ix + 1) % len(BG_THEMES)
            elif event.key == pygame.K_t:
                trail_ix = (trail_ix + 1) % len(trail_strengths)
            elif event.key == pygame.K_s:
                os.makedirs('screenshots', exist_ok=True)
                path = time.strftime('screenshots/aurorasunset_%Y%m%d_%H%M%S.png')
                pygame.image.save(screen, path)
                print('Saved', path)
            elif event.key == pygame.K_p:
                fireworks_enabled = not fireworks_enabled
            elif event.key == pygame.K_w:
                waterfall_enabled = not waterfall_enabled
            elif event.key == pygame.K_h:
                haze_enabled = not haze_enabled
            elif event.key == pygame.K_n:
                nature_enabled = not nature_enabled
                nature.running = nature_enabled
            elif event.key == pygame.K_v:
                fireflies_enabled = not fireflies_enabled
            elif event.key == pygame.K_p:
                fireworks_enabled = not fireworks_enabled
            elif event.key == pygame.K_w:
                waterfall_enabled = not waterfall_enabled
            elif event.key == pygame.K_h:
                haze_enabled = not haze_enabled

    # Audio
    try:
        data = np.frombuffer(stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16)
    except IOError:
        data = np.zeros(CHUNK, dtype=np.int16)

    spectrum = np.abs(np.fft.rfft(data))  # length CHUNK//2+1

    # Analyze change (flux/onsets) and conduct
    feat = analyzer.update(spectrum)
    params = conductor.update(spectrum)
    goal = params['goal']

    # drive palette
    color_engine.tick(feat, params, dt)

    # fireworks launch hooks
    if fireworks_enabled:
        fireworks.maybe_launch(feat, params)

    # nature director random events
    if nature_enabled:
        nature.maybe_step(feat, params)

    # fireworks launch hooks
    if fireworks_enabled:
        fireworks.maybe_launch(feat, params)

    # Motion magnitudes
    if AUDIO_COUPLED:
        strength_boost = 1.0 + 0.9*params['energy'] + 0.9*params['bass'] + 0.8*feat['flux']
        size_boost     = 1.0 + 0.8*params['lowmid'] + 0.6*feat['flux'] + 0.4*params['presence']
        swirl_boost    = 1.0 + 1.1*params['air'] + 0.7*feat['flux']
    else:
        strength_boost = size_boost = swirl_boost = 1.0

    # Trigger visual motifs
    motifs.maybe_trigger_from_features(feat, params)

    # Morph influencers based on goal
    if goal == 'bloom':
        INFLUENCER_MODE = 'attract'
        base_r = min(WIDTH, HEIGHT) * (0.26 + 0.10*params['bass'])
        INFLUENCERS = preset_ring(mode=INFLUENCER_MODE, radius=base_r)
    elif goal == 'starburst':
        INFLUENCER_MODE = 'repel'
        INFLUENCERS = preset_starburst(mode=INFLUENCER_MODE, arms=6)
    elif goal == 'spiral':
        INFLUENCER_MODE = 'attract'
        INFLUENCERS = preset_spiral(mode=INFLUENCER_MODE, turns=1.8)
        ang = 0.015 * params['rotate']
        ca, sa = math.cos(ang), math.sin(ang)
        for inf in INFLUENCERS:
            vx, vy = inf.pos[0]-CENTER[0], inf.pos[1]-CENTER[1]
            inf.pos[0] = CENTER[0] + vx*ca - vy*sa
            inf.pos[1] = CENTER[1] + vx*sa + vy*ca
    elif goal == 'lace':
        INFLUENCER_MODE = 'swirl'
        INFLUENCERS = preset_flower(mode=INFLUENCER_MODE, petals=7, wobble=params['flower_phase'])

    # Apply audio-based magnitudes
    for inf in INFLUENCERS:
        inf.strength = BASE_STRENGTH * strength_boost
        base = INFLUENCE_RADIUS * (0.85 + 0.25*params['mid'])
        inf.radius = base
        if INFLUENCER_MODE == 'swirl':
            globals()['SWIRL_TWIST'] = 1.2 + 1.0*(swirl_boost-1.0)

    # Hue flip on bigger swells
    energy = params['energy']
    if energy > 0.35 and not hue_inverted:
        for d in dots:
            r,g,b = d['color']
            h,s,v = colorsys.rgb_to_hsv(r/255.0,g/255.0,b/255.0)
            h=(h+0.5)%1.0
            rr,gg,bb = colorsys.hsv_to_rgb(h,s,v)
            d['color']=(int(rr*255),int(gg*255),int(bb*255))
        hue_inverted = True
    elif energy <= 0.35:
        hue_inverted = False

    # ==== Render ====
    tsec = pygame.time.get_ticks()/1000.0

    # Background breathing
    scene_surf.fill((0,0,0,0))
    # season hue influences palette subtly by shifting theme hue push via params proxy (already reflected in color engine hue drift)
    draw_background(scene_surf, BG_THEMES[bg_ix], params, tsec)

    # Waterfall veil (before trails) for depth
    if waterfall_enabled:
        waterfall.draw(scene_surf, params, dt)

    # Trails: fade previous frame based on setting
    if trail_ix == 0:
        trail_surf.fill((0,0,0,0))  # off
    else:
        pygame.draw.rect(trail_surf, trail_strengths[trail_ix], (0,0,WIDTH,HEIGHT))

    # Geometry selection
    geom = RENDERERS[renderer_ix]

    # Update motifs & fireworks
    motifs.step(dt)
    if fireworks_enabled:
        fireworks.step(dt)
    if fireflies_enabled:
        fireflies.step(params, dt)

    # Per-dot movement & draw to trail surface
    for d in dots:
        px, py = d['pos']
        fx_sum = 0.0
        fy_sum = 0.0
        mag_sum = 0.0
        for inf in INFLUENCERS:
            fx, fy, mag = inf.field(px, py)
            fx_sum += fx; fy_sum += fy; mag_sum += mag
        for tinf in motifs.transients:
            fx, fy, mag = tinf.field(px, py)
            fx_sum += fx; fy_sum += fy; mag_sum += 0.6*mag

        speed = 86.0 + 155.0 * energy + 230.0 * feat['flux'] + 90.0*params['presence']
        # add wind influence from nature
        wx, wy = (nature.wind(px, py, tsec) if nature_enabled else (0.0, 0.0))
        nx = px + (fx_sum * speed + wx) * dt
        ny = py + (fy_sum * speed + wy) * dt

        # home spring
        hx, hy = d['home'][0]-nx, d['home'][1]-ny
        nx += hx * 0.016
        ny += hy * 0.016

        # clamp
        nx = clamp(nx, 6, WIDTH-6)
        ny = clamp(ny, 6, HEIGHT-6)
        d['pos'][0], d['pos'][1] = nx, ny

        # size & color
        size = DOT_BASE_RADIUS + min(DOT_MAX_BOOST, mag_sum * 2.2) * (1.0 + 0.8*params['lowmid'] + 0.5*feat['flux'])
        theme_hue_push = 0.04 if BG_THEMES[bg_ix]=='Sunset' else (0.10 if BG_THEMES[bg_ix]=='Ocean' else (0.28 if BG_THEMES[bg_ix]=='Night' else 0.18))
        col = color_engine.color_for(mag_sum, feat, params, tsec, theme_hue_push=theme_hue_push)

        # orientation from last position (for ribbons)
        pid = id(d)
        last = last_positions.get(pid, (nx, ny))
        rot = math.atan2(ny-last[1], nx-last[0]) if (nx!=last[0] or ny!=last[1]) else random.random()*2*math.pi
        last_positions[pid] = (nx, ny)

        draw_shape(trail_surf, geom, (nx, ny), size, rot, col)

    # Sparkles layer
    for s in motifs.sparkles:
        k = 1.0 - s['age']/s['life']
        c = hsv255(0.12 + 0.55*k, 0.6, 0.6 + 0.4*k)
        pygame.draw.circle(trail_surf, c, (int(s['x']), int(s['y'])), max(1, int(2 + 3*k)))

    # Compose: symmetry of trail_surf onto scene, then blit to screen
    composed = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    blit_with_symmetry(trail_surf, composed)
    scene_surf.blit(composed, (0,0), special_flags=pygame.BLEND_PREMULTIPLIED)

    # Fireworks on top with soft additive glow
    if fireworks_enabled:
        fireworks.draw(scene_surf)

    # Fireflies glow near ground
    if fireflies_enabled:
        fireflies.draw(scene_surf, params)

    # Nature overlays (lightning, flock)
    if nature_enabled:
        nature.draw_overlays(scene_surf)

    # Fireworks on top with soft additive glow
    if fireworks_enabled:
        fireworks.draw(scene_surf)

    # Influencers overlay
    if SHOW_INFLUENCERS:
        for inf in INFLUENCERS:
            col = {'attract': (120,255,140), 'repel': (255,130,130), 'swirl': (140,180,255)}[inf.mode]
            pygame.draw.circle(scene_surf, col, (int(inf.pos[0]), int(inf.pos[1])), 6)
            pygame.draw.circle(scene_surf, (60,60,60), (int(inf.pos[0]), int(inf.pos[1])), int(inf.radius), 1)
        for tinf in motifs.transients:
            col = {'attract': (90,200,110), 'repel': (220,100,100), 'swirl': (110,150,230)}[tinf.mode]
            pygame.draw.circle(scene_surf, col, (int(tinf.pos[0]), int(tinf.pos[1])), 4)

    # HUD
    txt = f"AuroraSunsetGarden | {BG_THEMES[bg_ix]} | geom:{RENDERERS[renderer_ix]} | kaleido:{kaleido_sectors[kaleido_ix]} | fx: fireworks:{'on' if fireworks_enabled else 'off'} waterfall:{'on' if waterfall_enabled else 'off'} fireflies:{'on' if fireflies_enabled else 'off'} nature:{'on' if nature_enabled else 'off'} | {current_preset} goal:{goal} mode:{INFLUENCER_MODE} audio:{'on' if AUDIO_COUPLED else 'off'}"
    img = hud_font.render(txt, True, (90, 120, 150))

    # Final blit
    screen.blit(scene_surf, (0,0))
    screen.blit(img, (18, 14))
    pygame.display.flip()

# Cleanup
stream.stop_stream(); stream.close(); p.terminate(); pygame.quit()
