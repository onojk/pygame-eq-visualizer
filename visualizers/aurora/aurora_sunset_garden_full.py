#!/usr/bin/env python3
"""
AuroraSunsetGarden — an audio-reactive, abstract nature-inspired visualizer.

Audio source: PipeWire monitor captured via pw-record subprocess.
  Discovers the monitor source automatically using pactl; no pyaudio needed.
  Override target with AUDIO_INPUT_DEVICE env var.

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
  p      — Toggle fireworks
  w      — Toggle waterfall
  h      — Toggle haze
  n      — Toggle nature director
  v      — Toggle fireflies

Requires: pygame, numpy (no pyaudio)
Run tip: python3 aurora_sunset_garden_full.py
"""

import colorsys
import fcntl
import math
import os
import random
import shutil
import subprocess
import sys
import threading
import time
from collections import deque

import numpy as np
import pygame

# ===== Config =====
CHUNK = 1024   # matches PipeWire quantum
RATE  = 48000  # matches PipeWire default clock; no resampling
FPS   = 60

GRID_SPACING    = 42
DOT_BASE_RADIUS = 3.5
DOT_MAX_BOOST   = 6
DEFAULT_DOT_COLOR = (185, 220, 255)

NUM_INFLUENCERS  = 14
INFLUENCE_RADIUS = 280.0
FALLOFF_POWER    = 2.15
BASE_STRENGTH    = 1.0
SWIRL_TWIST      = 1.4

# Placeholder globals set by main() after display init.
WIDTH  = 0
HEIGHT = 0
CENTER = (0, 0)

# Written by Conductor.update() and key handlers inside main().
current_preset  = 'flower'
INFLUENCER_MODE = 'attract'
INFLUENCERS: list = []

# ===== Audio shared state =====
_audio_lock   = threading.Lock()
_latest_raw   = bytes(CHUNK * 4)   # zeroed silence until first chunk arrives
_chunks_total = 0
_latest_rms   = 0.0
_quit_event   = threading.Event()

# ===== pw-record helpers (pattern from 13_video_warp.py) =====

def _ts() -> str:
    t = time.time()
    return time.strftime("%H:%M:%S", time.localtime(t)) + f".{int(t*1000)%1000:03d}"


def _log(logfile, msg: str) -> None:
    line = f"[{_ts()}] {msg}"
    print(line, flush=True)
    if logfile is not None:
        try:
            logfile.write(line + "\n")
            logfile.flush()
        except (ValueError, OSError):
            pass


def find_monitor_source() -> str:
    override = os.environ.get('AUDIO_INPUT_DEVICE')
    if override is not None:
        return override
    try:
        out = subprocess.check_output(
            ['pactl', 'list', 'sources', 'short'],
            text=True, stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        print(f"[audio] ERROR: pactl failed: {exc}", file=sys.stderr)
        sys.exit(1)
    running: list[str] = []
    fallback: list[str] = []
    for line in out.splitlines():
        cols = line.split()
        if len(cols) < 2:
            continue
        name = cols[1]
        if 'monitor' not in name.lower():
            continue
        (running if cols[-1] == 'RUNNING' else fallback).append(name)
    candidates = running or fallback
    if not candidates:
        print(
            "[audio] ERROR: No monitor source found via pactl.\n"
            "  pactl output:\n" + "\n".join(f"    {l}" for l in out.splitlines()),
            file=sys.stderr,
        )
        sys.exit(1)
    def _score(name: str) -> int:
        n = name.lower()
        if 'speaker' in n:   return 0
        if 'hdmi' in n or 'headphone' in n: return 2
        return 1
    candidates.sort(key=_score)
    return candidates[0]


def start_pw_record(monitor_name: str):
    global _latest_raw, _chunks_total
    if not shutil.which('pw-record'):
        print("[audio] ERROR: pw-record not found. Install: sudo apt install pipewire-bin",
              file=sys.stderr)
        sys.exit(1)
    proc = subprocess.Popen(
        ['pw-record', '--raw', f'--target={monitor_name}',
         '--format=f32', '--rate=48000', '--channels=1',
         '--media-category=Capture', '--media-role=Production', '-'],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0,
    )
    print(f"[audio] Capturing monitor source: {monitor_name}")
    print(f"[audio] pw-record PID: {proc.pid}")

    def _reader():
        global _latest_raw, _chunks_total
        frame_bytes = CHUNK * 4
        fd = proc.stdout.fileno()
        fcntl.fcntl(fd, fcntl.F_SETFL, fcntl.fcntl(fd, fcntl.F_GETFL) | os.O_NONBLOCK)
        buf = bytearray()
        while True:
            try:
                data = os.read(fd, 65536)
                if not data:
                    break
                buf.extend(data)
                while len(buf) >= frame_bytes:
                    with _audio_lock:
                        _latest_raw    = bytes(buf[:frame_bytes])
                        _chunks_total += 1
                    del buf[:frame_bytes]
            except BlockingIOError:
                time.sleep(0.001)

    def _stderr_logger():
        for line in proc.stderr:
            print(f"[pw-record] {line.decode(errors='replace').rstrip()}", flush=True)

    reader_thread = threading.Thread(target=_reader, daemon=True)
    reader_thread.start()
    threading.Thread(target=_stderr_logger, daemon=True).start()
    return proc, reader_thread


def _watchdog_loop(logfile, proc_box: list, reader_box: list, monitor_name: str) -> None:
    last_chunks = 0; stall_ticks = 0; had_audio = False; restart_count = 0
    prev_rms_wd = None; wd_unchanged_ticks = 0; had_nonzero_rms = False; freeze_dumped = False
    MAX_RESTARTS = 50
    while True:
        time.sleep(1.0)
        with _audio_lock:
            rms    = _latest_rms
            chunks = _chunks_total
        if logfile is not None:
            _log(logfile, f"heartbeat: rms={rms:.4f} chunks={chunks} restarts={restart_count}")
        if chunks > last_chunks:
            had_audio = True; stall_ticks = 0
        elif had_audio:
            stall_ticks += 1
        last_chunks = chunks
        if stall_ticks >= 2:
            if restart_count >= MAX_RESTARTS:
                _log(logfile, f"STALL: max restarts ({MAX_RESTARTS}) reached, giving up.")
                _quit_event.set(); return
            restart_count += 1
            _log(logfile, f"STALL detected, recovering... (restart #{restart_count})")
            old_proc = proc_box[0]; old_proc.terminate()
            _log(logfile, "  suspending monitor source")
            suspend = None
            try:
                suspend = subprocess.run(
                    ['pactl', 'suspend-source', monitor_name, '1'], check=False)
            except subprocess.TimeoutExpired:
                _log(logfile, "  suspend timed out; skipping resume")
            try:
                old_proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                old_proc.kill(); old_proc.wait()
            if suspend is None or suspend.returncode != 0:
                _log(logfile, "  suspend failed (skipping resume)")
            else:
                time.sleep(0.1); _log(logfile, "  resuming monitor source")
                try:
                    subprocess.run(['pactl', 'suspend-source', monitor_name, '0'], check=False)
                except subprocess.TimeoutExpired:
                    _log(logfile, "  resume timed out; continuing anyway")
            _log(logfile, "  spawning fresh pw-record")
            new_proc, new_reader = start_pw_record(monitor_name)
            proc_box[0] = new_proc; reader_box[0] = new_reader
            _log(logfile, f"  new PID: {new_proc.pid}")
            stall_ticks = 0; prev_rms_wd = None; wd_unchanged_ticks = 0; freeze_dumped = False
        if rms != prev_rms_wd:
            if rms > 0.0: had_nonzero_rms = True
            prev_rms_wd = rms; wd_unchanged_ticks = 0; freeze_dumped = False
        elif had_nonzero_rms:
            wd_unchanged_ticks += 1
        if wd_unchanged_ticks >= 3 and not freeze_dumped:
            freeze_dumped = True
            _log(logfile, "WATCHDOG: RMS frozen for 3s — possible stall")


# ===== Utils =====

def hsv255(h, s, v):
    r, g, b = colorsys.hsv_to_rgb(h % 1.0, max(0.0, min(1.0, s)), max(0.0, min(1.0, v)))
    return (int(r*255), int(g*255), int(b*255))

def lerp(a, b, t):   return a + (b - a) * t
def clamp(x, lo, hi): return lo if x < lo else hi if x > hi else x


# ===== Color engine =====

class ColorEngine:
    def __init__(self):
        self.base_h = 0.55; self.flash = 0.0; self.hue_drift = 0.0

    def tick(self, feat, params, dt):
        self.hue_drift += dt * (0.06 + 0.25*params["energy"] + 0.35*feat["flux"])
        if feat.get("onset", False):
            self.flash = min(1.0, self.flash + 0.65)
            self.base_h = (self.base_h + 0.08 + 0.12*feat["flux"]) % 1.0
        self.flash *= (0.90 ** (dt*FPS))

    def color_for(self, local_mag, feat, params, t_seconds, theme_hue_push=0.0):
        h = (self.base_h + 0.07*math.sin(t_seconds*0.6) + 0.12*self.hue_drift + theme_hue_push) % 1.0
        h = (h + 0.12*params["bass"] - 0.04*params["lowmid"] + 0.10*params["air"]) % 1.0
        s = 0.50 + 0.33*params["presence"] + 0.28*feat["flux"] + 0.22*self.flash
        v = 0.36 + 0.56*params["energy"] + 0.46*min(1.0, local_mag*0.7) + 0.22*self.flash
        return hsv255(h, s, v)


# ===== Influencers =====

class Influencer:
    def __init__(self, x, y, strength=BASE_STRENGTH, radius=INFLUENCE_RADIUS, mode='attract'):
        self.pos = [float(x), float(y)]
        self.strength = strength; self.radius = radius; self.mode = mode

    def field(self, px, py):
        dx = self.pos[0] - px; dy = self.pos[1] - py
        d  = math.hypot(dx, dy) + 1e-6
        if d > self.radius: return 0.0, 0.0, 0.0
        t   = max(0.0, 1.0 - (d / self.radius) ** FALLOFF_POWER)
        ndx, ndy = dx/d, dy/d
        if self.mode == 'attract':
            fx, fy = ndx, ndy
        elif self.mode == 'repel':
            fx, fy = -ndx, -ndy
        elif self.mode == 'swirl':
            fx, fy = -ndy, ndx
            twist  = SWIRL_TWIST * t
            fx = fx*(0.7 + 0.3*twist) + 0.2*ndx
            fy = fy*(0.7 + 0.3*twist) + 0.2*ndy
        else:
            fx, fy = ndx, ndy
        mag = self.strength * t
        return fx*mag, fy*mag, mag


# ===== Presets =====

def preset_ring(mode='attract', radius=None):
    r = radius or min(WIDTH, HEIGHT) * 0.34
    return [Influencer(CENTER[0] + r*math.cos(2*math.pi*i/NUM_INFLUENCERS),
                       CENTER[1] + r*math.sin(2*math.pi*i/NUM_INFLUENCERS),
                       strength=BASE_STRENGTH, mode=mode)
            for i in range(NUM_INFLUENCERS)]

def preset_starburst(mode='attract', arms=6):
    infs = []; r_inner = min(WIDTH,HEIGHT)*0.20; r_outer = min(WIDTH,HEIGHT)*0.38; pts = []
    for k in range(arms):
        a = 2*math.pi*k/arms
        pts.append((CENTER[0]+r_outer*math.cos(a), CENTER[1]+r_outer*math.sin(a)))
        pts.append((CENTER[0]+r_inner*math.cos(a+math.pi/arms), CENTER[1]+r_inner*math.sin(a+math.pi/arms)))
    for i in range(NUM_INFLUENCERS):
        x, y = pts[i % len(pts)]
        infs.append(Influencer(x, y, strength=BASE_STRENGTH, mode=mode))
    return infs

def preset_spiral(mode='attract', turns=1.9):
    infs = []; r_max = min(WIDTH,HEIGHT)*0.44
    for i in range(NUM_INFLUENCERS):
        t = i/max(1, NUM_INFLUENCERS-1); a = 2*math.pi*turns*t; r = lerp(r_max*0.05, r_max, t)
        infs.append(Influencer(CENTER[0]+r*math.cos(a), CENTER[1]+r*math.sin(a), strength=BASE_STRENGTH, mode=mode))
    return infs

def preset_flower(mode='attract', petals=8, wobble=0.0):
    R = min(WIDTH,HEIGHT)*0.30; infs = []
    for i in range(NUM_INFLUENCERS):
        t = i/NUM_INFLUENCERS; a = 2*math.pi*t; r = R*(1.0+0.34*math.cos(petals*a+wobble))
        infs.append(Influencer(CENTER[0]+r*math.cos(a), CENTER[1]+r*math.sin(a), strength=BASE_STRENGTH, mode=mode))
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


# ===== Conductor =====

class Conductor:
    def __init__(self):
        self.energy_slow = 0.0; self.energy_fast = 0.0
        self.band_slow = {k: 0.0 for k in ['sub','bass','lowmid','mid','highmid','presence','air']}
        self.last_preset_switch = time.time()
        self.goal = 'bloom'; self.rotate = 0.0; self.flower_phase = 0.0

    def bandpack(self, sp):
        n = len(sp)
        idx = lambda f: int(clamp(f / (RATE/2) * (n-1), 0, n-1))
        bands = {
            'sub':      float(np.mean(sp[idx(20):idx(60)+1])),
            'bass':     float(np.mean(sp[idx(60):idx(140)+1])),
            'lowmid':   float(np.mean(sp[idx(140):idx(400)+1])),
            'mid':      float(np.mean(sp[idx(400):idx(1000)+1])),
            'highmid':  float(np.mean(sp[idx(1000):idx(2500)+1])),
            'presence': float(np.mean(sp[idx(2500):idx(6000)+1])),
            'air':      float(np.mean(sp[idx(6000):])),
        }
        bands['energy'] = float(np.mean(sp))
        return bands

    def update(self, spectrum):
        global current_preset, INFLUENCERS
        sp = spectrum.copy()
        if sp.max() > 0: sp /= sp.max()
        bands = self.bandpack(sp)
        self.energy_fast = lerp(self.energy_fast, bands['energy'], 0.35)
        self.energy_slow = lerp(self.energy_slow, bands['energy'], 0.05)
        for k in self.band_slow:
            self.band_slow[k] = lerp(self.band_slow[k], bands[k], 0.10)
        burst = ((self.energy_fast - self.energy_slow > 0.08) or
                 (bands['mid'] - self.band_slow['mid'] > 0.07))
        now = time.time()
        if burst:
            self.goal = 'starburst'; self.last_preset_switch = now
        elif self.band_slow['bass'] > 0.22 and self.energy_slow > 0.18:
            self.goal = 'spiral'
        elif self.band_slow['air'] > 0.20 and self.energy_slow > 0.14:
            self.goal = 'lace'
        else:
            self.goal = 'bloom'
        self.rotate       += 0.12 * (0.4 + self.energy_slow)
        self.flower_phase += 0.015 * (1.0 + 0.6*self.band_slow['presence'])
        if now - self.last_preset_switch > 12.0 and self.goal != 'starburst':
            self.last_preset_switch = now
            cycle = ['ring','flower','spiral','star']
            current_preset = cycle[(cycle.index(current_preset)+1) % len(cycle)]
            INFLUENCERS = build_preset(current_preset, INFLUENCER_MODE)
        out = dict(self.band_slow)
        out.update(energy=self.energy_slow, burst=burst, goal=self.goal,
                   rotate=self.rotate, flower_phase=self.flower_phase)
        return out


# ===== AudioAnalyzer =====

class AudioAnalyzer:
    def __init__(self, chunk):
        self.prev = np.zeros(chunk//2+1, dtype=float)
        self.flux_ema = 0.0; self.flux_var = 0.0
        self.k1 = 0.25; self.k2 = 0.15; self.onset_cool = 0.0

    def update(self, spectrum):
        sp = spectrum.astype(float)
        if sp.max() > 0: sp /= sp.max()
        diff = sp - self.prev
        flux = float(np.sum(np.clip(diff, 0, None)))
        self.flux_ema = (1-self.k1)*self.flux_ema + self.k1*flux
        d = flux - self.flux_ema
        self.flux_var = (1-self.k2)*self.flux_var + self.k2*(d*d)
        thresh = self.flux_ema + 0.9*math.sqrt(max(1e-6, self.flux_var))
        onset  = flux > thresh and self.onset_cool <= 0.0
        if onset: self.onset_cool = 0.10
        else:     self.onset_cool = max(0.0, self.onset_cool - 1.0/FPS)
        self.prev = sp
        return {"sp": sp, "flux": flux, "onset": onset}


# ===== Transient influencers =====

class TransientInfluencer(Influencer):
    def __init__(self, x, y, life=0.6, strength=1.5, radius=220, mode='repel'):
        super().__init__(x, y, strength=strength, radius=radius, mode=mode)
        self.life = life; self.age = 0.0

    def step(self, dt):
        self.life -= dt; self.age += dt
        return self.life > 0.0


# ===== Motif engine =====

class MotifEngine:
    def __init__(self):
        self.mode_ix = 0; self.modes = ['ripple','starburst','swirlstorm']
        self.cool = 0.0; self.transients = []; self.sparkles = deque(maxlen=2000)

    def trigger(self, kind, center, band_boost):
        cx, cy = center
        if kind == 'ripple':
            r = min(WIDTH,HEIGHT)*(0.18 + 0.15*band_boost)
            for i in range(12):
                a = 2*math.pi*i/12
                self.transients.append(TransientInfluencer(
                    cx+r*math.cos(a), cy+r*math.sin(a),
                    life=0.7, strength=1.2+0.8*band_boost, radius=200, mode='attract'))
        elif kind == 'starburst':
            r = min(WIDTH,HEIGHT)*(0.25 + 0.2*band_boost)
            for k in range(10):
                a = 2*math.pi*k/10
                self.transients.append(TransientInfluencer(
                    cx+r*math.cos(a), cy+r*math.sin(a),
                    life=0.5, strength=1.6+1.0*band_boost, radius=240, mode='repel'))
        elif kind == 'swirlstorm':
            for i in range(14):
                a = 2*math.pi*i/14; r = 90 + 90*band_boost
                self.transients.append(TransientInfluencer(
                    cx+r*math.cos(a), cy+r*math.sin(a),
                    life=0.85, strength=1.4+0.9*band_boost, radius=220, mode='swirl'))
        for _ in range(80 + int(160*band_boost)):
            ang = random.uniform(0, 2*math.pi); spd = random.uniform(30, 280)*(0.5+band_boost)
            self.sparkles.append({'x':cx,'y':cy,'vx':math.cos(ang)*spd,'vy':math.sin(ang)*spd,
                                   'life':random.uniform(0.25,1.2),'age':0.0})
        self.cool = 0.12

    def maybe_trigger_from_features(self, feat, bands):
        if self.cool > 0: self.cool -= 1.0/FPS; return
        if feat['onset']:
            kind = self.modes[self.mode_ix % len(self.modes)]; self.mode_ix += 1
            band_boost = clamp(0.5*bands['bass']+0.4*bands['presence']+0.3*bands['air'], 0.0, 1.0)
            self.trigger(kind, CENTER, band_boost)

    def step(self, dt):
        self.transients = [t for t in self.transients if t.step(dt)]
        alive = deque(maxlen=self.sparkles.maxlen)
        for s in self.sparkles:
            s['age'] += dt
            if s['age'] < s['life']:
                s['x'] += s['vx']*dt; s['y'] += s['vy']*dt; s['vy'] += 12*dt
                if -50 <= s['x'] <= WIDTH+50 and -50 <= s['y'] <= HEIGHT+50:
                    alive.append(s)
        self.sparkles = alive


# ===== Background =====

BG_THEMES = ['Sunset', 'Ocean', 'Forest', 'Night']

def draw_background(surface, theme, bands, t):
    sub = bands['sub']; bass = bands['bass']; air = bands['air']; presence = bands['presence']
    if theme == 'Sunset':
        top = hsv255(0.60+0.02*math.sin(t*0.2),  0.35, 0.12+0.18*air)
        bot = hsv255(0.05+0.03*math.sin(t*0.15), 0.85, 0.46+0.42*(0.5*bass+0.5*sub))
    elif theme == 'Ocean':
        top = hsv255(0.50, 0.30, 0.10+0.20*air)
        bot = hsv255(0.52, 0.75, 0.38+0.45*(0.4*bass+0.6*presence))
    elif theme == 'Forest':
        top = hsv255(0.33, 0.40, 0.12+0.16*air)
        bot = hsv255(0.33, 0.80, 0.36+0.45*(0.5*bass+0.5*sub))
    else:
        top = hsv255(0.70, 0.25, 0.10+0.25*air)
        bot = hsv255(0.75, 0.65, 0.28+0.40*(0.3*bass+0.7*presence))
    for y in range(0, HEIGHT, 4):
        k = y/HEIGHT
        c = (int(top[0]*(1-k)+bot[0]*k), int(top[1]*(1-k)+bot[1]*k), int(top[2]*(1-k)+bot[2]*k))
        pygame.draw.rect(surface, c, (0, y, WIDTH, 4))
    horizon_y = int(HEIGHT*0.62); bloom = int(40+220*(0.4*bass+0.6*sub))
    s = pygame.Surface((WIDTH, bloom), pygame.SRCALPHA)
    pygame.draw.rect(s, (255,255,255, int(70+150*(0.5*bass+0.5*sub))), (0,0,WIDTH,bloom))
    surface.blit(s, (0, horizon_y-bloom//2), special_flags=pygame.BLEND_PREMULTIPLIED)


# ===== WaterfallSystem =====

class WaterfallSystem:
    def __init__(self):
        self.offset = 0.0; self.columns = []
        for x in range(0, WIDTH, 8):
            self.columns.append({'x':x,'w':8+random.randint(0,6),
                                 'phase':random.random()*2*math.pi,'alpha':random.randint(20,45)})
        self.mist = pygame.Surface((WIDTH, int(HEIGHT*0.25)), pygame.SRCALPHA)

    def draw(self, surface, bands, dt):
        self.offset = (self.offset + (80+220*(0.5*bands['bass']+0.5*bands['sub']))*dt) % HEIGHT
        veil = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        for c in self.columns:
            sway = math.sin(self.offset*0.005+c['phase'])*(18+34*bands['presence'])
            a    = int(c['alpha']+80*bands['air'])
            pygame.draw.rect(veil, (200,220,255,clamp(a,10,180)), (int(c['x']+sway),0,c['w'],HEIGHT))
        surface.blit(veil, (0,0), special_flags=pygame.BLEND_PREMULTIPLIED)
        self.mist.fill((0,0,0,0))
        fog_a = int(30+120*(0.4*bands['presence']+0.6*bands['air']))
        pygame.draw.rect(self.mist,(230,235,255,clamp(fog_a,20,160)),(0,0,WIDTH,self.mist.get_height()))
        surface.blit(self.mist, (0,int(HEIGHT*0.75)), special_flags=pygame.BLEND_PREMULTIPLIED)


# ===== FireworksSystem =====

class FireworksSystem:
    def __init__(self):
        self.shells = []; self.sparks = []
        self.glow = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)

    def launch(self, x=None):
        x = x if x is not None else random.randint(int(WIDTH*0.2), int(WIDTH*0.8))
        self.shells.append({
            'x': float(x), 'y': float(HEIGHT-10),
            'vx': random.uniform(-40,40), 'vy': -random.uniform(180,260),
            't': 0.0, 'color': hsv255(random.random(),0.8,1.0), 'exploded': False,
        })

    def maybe_launch(self, feat, bands):
        if feat['onset'] or (bands['presence']+bands['air'] > 0.45):
            for _ in range(1, 1+int(2+3*(bands['presence']+bands['air']))):
                self.launch()

    def step(self, dt):
        new_shells = []
        for s in self.shells:
            s['t'] += dt; s['x'] += s['vx']*dt; s['y'] += s['vy']*dt; s['vy'] += 140*dt
            if s['vy'] > -20 or s['t'] > 1.2: self.explode(s)
            else: new_shells.append(s)
        self.shells = new_shells
        alive = []
        for sp in self.sparks:
            sp['age'] += dt
            if sp['age'] < sp['life']:
                sp['x'] += sp['vx']*dt; sp['y'] += sp['vy']*dt
                sp['vy'] += 220*dt; sp['vx'] *= 0.985
                alive.append(sp)
        self.sparks = alive

    def explode(self, shell):
        cx, cy = shell['x'], shell['y']
        r0,g0,b0 = shell['color']
        h0,s0,v0 = colorsys.rgb_to_hsv(r0/255.,g0/255.,b0/255.)
        for _ in range(random.randint(80,140)):
            ang = random.random()*2*math.pi; spd = random.uniform(60,360)
            col = hsv255(h0+(random.random()-0.5)*0.06, min(1.,s0*1.1), v0)
            self.sparks.append({'x':cx,'y':cy,'vx':math.cos(ang)*spd,'vy':math.sin(ang)*spd,
                                 'life':random.uniform(0.6,1.8),'age':0.0,'color':col})
        shell['exploded'] = True

    def draw(self, surface):
        self.glow.fill((0,0,0,0))
        for sp in self.sparks:
            k = 1.0 - sp['age']/sp['life']
            pygame.draw.circle(self.glow, (*sp['color'], int(200*k)),
                                (int(sp['x']),int(sp['y'])), max(1,int(2+2*k)))
        surface.blit(self.glow, (0,0), special_flags=pygame.BLEND_PREMULTIPLIED)


# ===== NatureDirector =====

class NatureDirector:
    def __init__(self):
        self.running   = True
        self.wind_dir  = random.random()*2*math.pi; self.wind_gust = 0.0
        self.season_hue = random.uniform(-0.04,0.04); self.lightning = 0.0
        self.next_event = time.time()+random.uniform(3,8); self.flock = []

    def maybe_step(self, feat, bands):
        if not self.running: return
        now = time.time()
        if now >= self.next_event:
            choice = random.random()
            if choice < 0.4:
                self.wind_dir  = random.random()*2*math.pi
                self.wind_gust = min(1.0, 0.4+0.8*(0.5*bands['presence']+0.5*bands['air']))
            elif choice < 0.6:
                self.season_hue = clamp(self.season_hue+random.uniform(-0.03,0.03),-0.12,0.12)
            elif choice < 0.8 and (feat['onset'] or bands['presence']>0.35):
                self.lightning = 1.0
            else:
                self._spawn_flock(bands)
            self.next_event = now+random.uniform(2.5,7.5)
        self.lightning *= 0.88; self.wind_gust *= 0.985

    def wind(self, x, y, t):
        base = 40*self.wind_gust
        return (base*(math.cos(self.wind_dir)*0.6 + 0.4*math.sin(0.07*t+0.0008*y)),
                base*(math.sin(self.wind_dir)*0.6 + 0.4*math.sin(0.09*t+0.0008*x)))

    def _spawn_flock(self, bands):
        count = random.randint(6,12); y = random.randint(int(HEIGHT*0.10),int(HEIGHT*0.35))
        speed = 120+140*bands['air']; ltr = random.random()<0.5
        x0 = -60 if ltr else WIDTH+60; vx = speed if ltr else -speed
        self.flock = [{'x':x0+i*20,'y':y+(i%5-2)*6,'vx':vx} for i in range(count)]

    def draw_overlays(self, surface):
        if self.lightning > 0.02:
            fl = pygame.Surface((WIDTH,HEIGHT),pygame.SRCALPHA)
            fl.fill((255,255,255,int(160*self.lightning)))
            surface.blit(fl,(0,0),special_flags=pygame.BLEND_PREMULTIPLIED)
        if self.flock:
            alive = []
            for b in self.flock:
                b['x'] += b['vx']/FPS
                if -100 <= b['x'] <= WIDTH+100:
                    alive.append(b)
                    p1=(int(b['x']),int(b['y']))
                    pygame.draw.line(surface,(20,20,25),p1,(p1[0]-6,p1[1]+4),2)
                    pygame.draw.line(surface,(20,20,25),p1,(p1[0]+6,p1[1]+4),2)
            self.flock = alive


# ===== Fireflies =====

class Fireflies:
    def __init__(self, n=180):
        self.points = [{'x':random.uniform(0,WIDTH),'y':random.uniform(HEIGHT*0.55,HEIGHT*0.95),
                        'a':random.random(),'r':random.uniform(1.5,3.0)} for _ in range(n)]
        self.surf = pygame.Surface((WIDTH,HEIGHT), pygame.SRCALPHA)

    def step(self, bands, dt):
        for p in self.points:
            p['a'] += (random.uniform(-0.6,0.6)+0.6*bands['air'])*dt
            p['x'] += (random.uniform(-12,12)+18*bands['bass'])*dt
            p['y'] += random.uniform(-10,10)*dt
            p['x'] = clamp(p['x'],0,WIDTH); p['y'] = clamp(p['y'],HEIGHT*0.50,HEIGHT-5)

    def draw(self, surface, bands):
        self.surf.fill((0,0,0,0))
        for p in self.points:
            k = (0.5+0.5*math.sin(p['a']*6))*(0.4+0.6*(0.6*bands['presence']+0.4*bands['air']))
            c = hsv255(0.17+0.05*random.random(), 0.6, 0.6+0.4*k)
            pygame.draw.circle(self.surf, (*c,int(80+140*k)), (int(p['x']),int(p['y'])), int(p['r']))
        surface.blit(self.surf,(0,0),special_flags=pygame.BLEND_PREMULTIPLIED)


# ===== Geometry =====

RENDERERS       = ['dots','petals','tri','quad','star','ribbons','soft']
kaleido_sectors = [1, 2, 4, 6]

def draw_shape(surface, name, pos, size, rot, color):
    x, y = pos
    if name == 'dots':
        pygame.draw.circle(surface, color, (int(x),int(y)), max(1,int(size))); return
    if name == 'soft':
        r = max(1,int(size*1.7)); s = pygame.Surface((r*2,r*2), pygame.SRCALPHA)
        pygame.draw.circle(s, (*color,110), (r,r), r)
        pygame.draw.circle(s, (*color,200), (r,r), int(r*0.66))
        surface.blit(s,(int(x-r),int(y-r)),special_flags=pygame.BLEND_PREMULTIPLIED); return
    if name == 'ribbons':
        length = size*3; dx = math.cos(rot); dy = math.sin(rot)
        pygame.draw.line(surface, color, (x-dx*length*0.5,y-dy*length*0.5),
                         (x+dx*length*0.5,y+dy*length*0.5), max(1,int(size*0.8))); return
    pts = []
    if name == 'petals':
        for i in range(5):
            a = rot+2*math.pi*i/5; r = size*(1.4 if i%2==0 else 0.7)
            pts.append((x+r*math.cos(a), y+r*math.sin(a)))
    elif name == 'tri':
        for i in range(3): a = rot+2*math.pi*i/3; pts.append((x+size*1.2*math.cos(a),y+size*1.2*math.sin(a)))
    elif name == 'quad':
        for i in range(4): a = rot+math.pi/4+2*math.pi*i/4; pts.append((x+size*1.1*math.cos(a),y+size*1.1*math.sin(a)))
    elif name == 'star':
        for i in range(10):
            a = rot+2*math.pi*i/10; r = size*(1.6 if i%2==0 else 0.6)
            pts.append((x+r*math.cos(a), y+r*math.sin(a)))
    if pts: pygame.draw.polygon(surface, color, pts)


def blit_with_symmetry(base, dest, sectors):
    if sectors == 1:
        dest.blit(base, (0,0)); return
    for s in range(sectors):
        rotated = pygame.transform.rotozoom(base, (360/sectors)*s, 1.0)
        if s % 2 == 1: rotated = pygame.transform.flip(rotated, True, False)
        dest.blit(rotated, (0,0), special_flags=pygame.BLEND_ADD)


# ===== Main =====

def main() -> None:
    global WIDTH, HEIGHT, CENTER
    global current_preset, INFLUENCER_MODE, INFLUENCERS
    global SWIRL_TWIST, _latest_rms

    # Audio must start before pygame so we fail fast if pactl is missing.
    monitor_name = find_monitor_source()
    proc, reader_thread = start_pw_record(monitor_name)
    proc_box   = [proc]
    reader_box = [reader_thread]
    threading.Thread(
        target=_watchdog_loop,
        args=(None, proc_box, reader_box, monitor_name),
        daemon=True,
    ).start()

    if 'SDL_AUDIODRIVER' not in os.environ:
        os.environ['SDL_AUDIODRIVER'] = 'pulse'
    pygame.init()
    pygame.display.set_caption("AuroraSunsetGarden")
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    info   = pygame.display.Info()
    WIDTH, HEIGHT = info.current_w, info.current_h
    CENTER        = (WIDTH // 2, HEIGHT // 2)
    clock         = pygame.time.Clock()

    scene_surf = pygame.Surface((WIDTH, HEIGHT)).convert_alpha()
    scene_surf.fill((0,0,0,255))
    trail_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    trail_strengths = [(255,255,255,0),(0,0,0,15),(0,0,0,28),(0,0,0,42)]
    trail_ix    = 2
    bg_ix       = 0
    renderer_ix = 0
    kaleido_ix  = 0

    num_cols = WIDTH  // GRID_SPACING
    num_rows = HEIGHT // GRID_SPACING
    dots = []
    for row in range(num_rows):
        for col in range(num_cols):
            x = col*GRID_SPACING + GRID_SPACING//2
            y = row*GRID_SPACING + GRID_SPACING//2
            dots.append({'pos':[float(x),float(y)],'home':[float(x),float(y)],'color':DEFAULT_DOT_COLOR})

    current_preset  = 'flower'
    INFLUENCER_MODE = 'attract'
    INFLUENCERS     = build_preset(current_preset, INFLUENCER_MODE)

    conductor    = Conductor()
    analyzer     = AudioAnalyzer(CHUNK)
    motifs       = MotifEngine()
    color_engine = ColorEngine()
    fireworks    = FireworksSystem()
    waterfall    = WaterfallSystem()
    nature       = NatureDirector()
    fireflies    = Fireflies(n=140)
    hud_font     = pygame.font.SysFont(None, 24)

    fireworks_enabled = True
    waterfall_enabled = True
    haze_enabled      = True
    nature_enabled    = True
    fireflies_enabled = True
    audio_coupled     = True
    show_influencers  = False
    hue_inverted      = False
    last_positions: dict = {}

    try:
        running = True
        while running:
            if _quit_event.is_set():
                break

            dt = clock.tick(FPS) / 1000.0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    k = event.key
                    if   k == pygame.K_ESCAPE: running = False
                    elif k == pygame.K_f:  screen = pygame.display.set_mode((0,0), pygame.FULLSCREEN)
                    elif k == pygame.K_i:  show_influencers = not show_influencers
                    elif k == pygame.K_a:  audio_coupled = not audio_coupled
                    elif k == pygame.K_r:  INFLUENCERS = preset_scaffold(mode=INFLUENCER_MODE); current_preset='scaffold'
                    elif k == pygame.K_m:
                        INFLUENCER_MODE = {'attract':'repel','repel':'swirl','swirl':'attract'}[INFLUENCER_MODE]
                        for inf in INFLUENCERS: inf.mode = INFLUENCER_MODE
                    elif k == pygame.K_1:  current_preset='ring';     INFLUENCERS=build_preset('ring',INFLUENCER_MODE)
                    elif k == pygame.K_2:  current_preset='star';     INFLUENCERS=build_preset('star',INFLUENCER_MODE)
                    elif k == pygame.K_3:  current_preset='spiral';   INFLUENCERS=build_preset('spiral',INFLUENCER_MODE)
                    elif k == pygame.K_4:  current_preset='flower';   INFLUENCERS=build_preset('flower',INFLUENCER_MODE)
                    elif k == pygame.K_5:  current_preset='scaffold'; INFLUENCERS=build_preset('scaffold',INFLUENCER_MODE)
                    elif k == pygame.K_g:  renderer_ix = (renderer_ix+1) % len(RENDERERS)
                    elif k == pygame.K_k:  kaleido_ix  = (kaleido_ix+1)  % len(kaleido_sectors)
                    elif k == pygame.K_b:  bg_ix       = (bg_ix+1)       % len(BG_THEMES)
                    elif k == pygame.K_t:  trail_ix    = (trail_ix+1)    % len(trail_strengths)
                    elif k == pygame.K_s:
                        os.makedirs('screenshots', exist_ok=True)
                        path = time.strftime('screenshots/aurorasunset_%Y%m%d_%H%M%S.png')
                        pygame.image.save(screen, path); print('Saved', path)
                    elif k == pygame.K_p:  fireworks_enabled = not fireworks_enabled
                    elif k == pygame.K_w:  waterfall_enabled = not waterfall_enabled
                    elif k == pygame.K_h:  haze_enabled      = not haze_enabled
                    elif k == pygame.K_n:  nature_enabled = not nature_enabled; nature.running = nature_enabled
                    elif k == pygame.K_v:  fireflies_enabled = not fireflies_enabled

            # Audio — grab latest float32 chunk from reader thread
            with _audio_lock:
                raw = _latest_raw
            mono     = np.frombuffer(raw, dtype=np.float32).copy()
            spectrum = np.abs(np.fft.rfft(mono))
            with _audio_lock:
                _latest_rms = float(np.sqrt(np.mean(mono**2)))

            feat   = analyzer.update(spectrum)
            params = conductor.update(spectrum)
            goal   = params['goal']

            color_engine.tick(feat, params, dt)

            if fireworks_enabled:
                fireworks.maybe_launch(feat, params)
            if nature_enabled:
                nature.maybe_step(feat, params)

            if audio_coupled:
                strength_boost = 1.0 + 0.9*params['energy'] + 0.9*params['bass'] + 0.8*feat['flux']
                size_boost     = 1.0 + 0.8*params['lowmid']  + 0.6*feat['flux']   + 0.4*params['presence']
                swirl_boost    = 1.0 + 1.1*params['air']     + 0.7*feat['flux']
            else:
                strength_boost = size_boost = swirl_boost = 1.0

            motifs.maybe_trigger_from_features(feat, params)

            if goal == 'bloom':
                INFLUENCER_MODE = 'attract'
                INFLUENCERS = preset_ring(mode=INFLUENCER_MODE,
                                          radius=min(WIDTH,HEIGHT)*(0.26+0.10*params['bass']))
            elif goal == 'starburst':
                INFLUENCER_MODE = 'repel'
                INFLUENCERS = preset_starburst(mode=INFLUENCER_MODE, arms=6)
            elif goal == 'spiral':
                INFLUENCER_MODE = 'attract'
                INFLUENCERS = preset_spiral(mode=INFLUENCER_MODE, turns=1.8)
                ang = 0.015*params['rotate']; ca, sa = math.cos(ang), math.sin(ang)
                for inf in INFLUENCERS:
                    vx, vy = inf.pos[0]-CENTER[0], inf.pos[1]-CENTER[1]
                    inf.pos[0] = CENTER[0]+vx*ca-vy*sa
                    inf.pos[1] = CENTER[1]+vx*sa+vy*ca
            elif goal == 'lace':
                INFLUENCER_MODE = 'swirl'
                INFLUENCERS = preset_flower(mode=INFLUENCER_MODE, petals=7, wobble=params['flower_phase'])

            for inf in INFLUENCERS:
                inf.strength = BASE_STRENGTH * strength_boost
                inf.radius   = INFLUENCE_RADIUS * (0.85 + 0.25*params['mid'])
            if INFLUENCER_MODE == 'swirl':
                SWIRL_TWIST = 1.2 + 1.0*(swirl_boost - 1.0)

            energy = params['energy']
            if energy > 0.35 and not hue_inverted:
                for d in dots:
                    r,g,b = d['color']; h,s,v = colorsys.rgb_to_hsv(r/255.,g/255.,b/255.)
                    rr,gg,bb = colorsys.hsv_to_rgb((h+0.5)%1.0,s,v)
                    d['color'] = (int(rr*255),int(gg*255),int(bb*255))
                hue_inverted = True
            elif energy <= 0.35:
                hue_inverted = False

            tsec = pygame.time.get_ticks() / 1000.0
            scene_surf.fill((0,0,0,0))
            draw_background(scene_surf, BG_THEMES[bg_ix], params, tsec)
            if waterfall_enabled:
                waterfall.draw(scene_surf, params, dt)

            if trail_ix == 0:
                trail_surf.fill((0,0,0,0))
            else:
                pygame.draw.rect(trail_surf, trail_strengths[trail_ix], (0,0,WIDTH,HEIGHT))

            geom = RENDERERS[renderer_ix]
            motifs.step(dt)
            if fireworks_enabled: fireworks.step(dt)
            if fireflies_enabled: fireflies.step(params, dt)

            for d in dots:
                px, py = d['pos']
                fx_sum = fy_sum = mag_sum = 0.0
                for inf in INFLUENCERS:
                    fx,fy,mag = inf.field(px,py); fx_sum+=fx; fy_sum+=fy; mag_sum+=mag
                for tinf in motifs.transients:
                    fx,fy,mag = tinf.field(px,py); fx_sum+=fx; fy_sum+=fy; mag_sum+=0.6*mag

                speed = 86.0+155.0*energy+230.0*feat['flux']+90.0*params['presence']
                wx,wy = nature.wind(px,py,tsec) if nature_enabled else (0.0,0.0)
                nx = px+(fx_sum*speed+wx)*dt; ny = py+(fy_sum*speed+wy)*dt
                nx += (d['home'][0]-nx)*0.016;  ny += (d['home'][1]-ny)*0.016
                nx  = clamp(nx,6,WIDTH-6);      ny  = clamp(ny,6,HEIGHT-6)
                d['pos'][0] = nx; d['pos'][1] = ny

                size  = DOT_BASE_RADIUS + min(DOT_MAX_BOOST,mag_sum*2.2)*(1.0+0.8*params['lowmid']+0.5*feat['flux'])
                theme_push = {'Sunset':0.04,'Ocean':0.10,'Night':0.28}.get(BG_THEMES[bg_ix],0.18)
                col   = color_engine.color_for(mag_sum, feat, params, tsec, theme_hue_push=theme_push)

                pid  = id(d); last = last_positions.get(pid,(nx,ny))
                rot  = (math.atan2(ny-last[1],nx-last[0])
                        if (nx!=last[0] or ny!=last[1]) else random.random()*2*math.pi)
                last_positions[pid] = (nx,ny)
                draw_shape(trail_surf, geom, (nx,ny), size, rot, col)

            for s in motifs.sparkles:
                k = 1.0-s['age']/s['life']
                pygame.draw.circle(trail_surf, hsv255(0.12+0.55*k,0.6,0.6+0.4*k),
                                   (int(s['x']),int(s['y'])), max(1,int(2+3*k)))

            composed = pygame.Surface((WIDTH,HEIGHT), pygame.SRCALPHA)
            blit_with_symmetry(trail_surf, composed, kaleido_sectors[kaleido_ix])
            scene_surf.blit(composed,(0,0),special_flags=pygame.BLEND_PREMULTIPLIED)

            if fireworks_enabled:  fireworks.draw(scene_surf)
            if fireflies_enabled:  fireflies.draw(scene_surf, params)
            if nature_enabled:     nature.draw_overlays(scene_surf)

            if show_influencers:
                for inf in INFLUENCERS:
                    col = {'attract':(120,255,140),'repel':(255,130,130),'swirl':(140,180,255)}[inf.mode]
                    pygame.draw.circle(scene_surf, col, (int(inf.pos[0]),int(inf.pos[1])), 6)
                    pygame.draw.circle(scene_surf, (60,60,60), (int(inf.pos[0]),int(inf.pos[1])), int(inf.radius), 1)
                for tinf in motifs.transients:
                    col = {'attract':(90,200,110),'repel':(220,100,100),'swirl':(110,150,230)}[tinf.mode]
                    pygame.draw.circle(scene_surf, col, (int(tinf.pos[0]),int(tinf.pos[1])), 4)

            txt = (f"AuroraSunsetGarden | {BG_THEMES[bg_ix]} | geom:{RENDERERS[renderer_ix]}"
                   f" | kaleido:{kaleido_sectors[kaleido_ix]}"
                   f" | fx: fw:{'on' if fireworks_enabled else 'off'}"
                   f" wf:{'on' if waterfall_enabled else 'off'}"
                   f" ff:{'on' if fireflies_enabled else 'off'}"
                   f" nat:{'on' if nature_enabled else 'off'}"
                   f" | {current_preset} goal:{goal} mode:{INFLUENCER_MODE}"
                   f" audio:{'on' if audio_coupled else 'off'}")
            screen.blit(scene_surf, (0,0))
            screen.blit(hud_font.render(txt, True, (90,120,150)), (18,14))
            pygame.display.flip()

    finally:
        proc_box[0].terminate()
        try:
            proc_box[0].wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc_box[0].kill()
        pygame.quit()


if __name__ == '__main__':
    main()
