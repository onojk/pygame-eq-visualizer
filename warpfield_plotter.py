import pygame
import numpy as np
import pyaudio
import math
import random
import colorsys
import time
import os

# ===== Config =====
CHUNK = 512
CHANNELS = 1
RATE = 22050
FPS = 60

GRID_SPACING = 46
DOT_BASE_RADIUS = 4
DOT_MAX_BOOST = 5
DEFAULT_DOT_COLOR = (185, 220, 255)

NUM_INFLUENCERS = 12
INFLUENCE_RADIUS = 260.0
FALLOFF_POWER = 2.2
BASE_STRENGTH = 1.0
SWIRL_TWIST = 1.4

AUDIO_COUPLED = True
SHOW_INFLUENCERS = True

# Prefer Pulse/Default device to avoid ALSA/JACK spam
PREFERRED_DEVICE_SUBSTR = "pulse"   # set to part of your desired device name, or "" for default

# ===== Init =====
# (You can also run with: SDL_AUDIODRIVER=pulse python3 warpfield_plotter.py)
if "SDL_AUDIODRIVER" not in os.environ:
    os.environ["SDL_AUDIODRIVER"] = "pulse"

pygame.init()
screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
info = pygame.display.Info()
WIDTH, HEIGHT = info.current_w, info.current_h
CENTER = (WIDTH // 2, HEIGHT // 2)
clock = pygame.time.Clock()

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
def invert_color_hue(rgb):
    r, g, b = [x / 255.0 for x in rgb]
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    h = (h + 0.5) % 1.0
    r2, g2, b2 = colorsys.hsv_to_rgb(h, s, v)
    return (int(r2 * 255), int(g2 * 255), int(b2 * 255))

def lerp(a, b, t):
    return a + (b - a) * t

# ===== Color engine (audio-reactive palette) =====
class ColorEngine:
    """
    Hue drifts with energy/flux and band tint.
    Saturation pops with highs/onsets.
    Value blooms with global energy + local field magnitude.
    """
    def __init__(self):
        self.base_h = 0.55
        self.flash = 0.0
        self.hue_drift = 0.0

    @staticmethod
    def hsv255(h, s, v):
        r, g, b = colorsys.hsv_to_rgb(h % 1.0, max(0.0, min(1.0, s)), max(0.0, min(1.0, v)))
        return (int(r*255), int(g*255), int(b*255))

    def tick(self, feat, params, dt):
        # Palette hue drifts with energy & spectral change
        self.hue_drift += dt * (0.06 + 0.25*params["energy"] + 0.35*feat["flux"])
        # Onset flash
        if feat.get("onset", False):
            self.flash = min(1.0, self.flash + 0.65)
            self.base_h = (self.base_h + 0.08 + 0.12*feat["flux"]) % 1.0
        self.flash *= (0.90 ** (dt*FPS))

    def color_for(self, local_mag, feat, params, t_seconds):
        # Base hue: slow drift + small wobble + band tint
        h = (self.base_h + 0.07*math.sin(t_seconds*0.6) + 0.12*self.hue_drift) % 1.0
        band_push = 0.15*params["low"] - 0.05*params["mid"] + 0.12*params["high"]
        h = (h + band_push) % 1.0
        # Saturation: highs + flux + flash
        s = 0.52 + 0.35*params["high"] + 0.30*feat["flux"] + 0.25*self.flash
        # Value: energy + local field magnitude + flash
        v = 0.38 + 0.55*params["energy"] + 0.45*min(1.0, local_mag*0.7) + 0.20*self.flash
        return self.hsv255(h, s, v)

# ===== Dots =====
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
    r = radius or min(WIDTH, HEIGHT) * 0.32
    return [Influencer(CENTER[0] + r*math.cos(2*math.pi*i/NUM_INFLUENCERS),
                       CENTER[1] + r*math.sin(2*math.pi*i/NUM_INFLUENCERS),
                       strength=BASE_STRENGTH, mode=mode)
            for i in range(NUM_INFLUENCERS)]

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
        infs.append(Influencer(x, y, strength=BASE_STRENGTH, mode=mode))
    return infs

def preset_spiral(mode='attract', turns=1.6):
    infs = []
    r_max = min(WIDTH, HEIGHT) * 0.42
    for i in range(NUM_INFLUENCERS):
        t = i / max(1, NUM_INFLUENCERS - 1)
        a = 2 * math.pi * turns * t
        r = lerp(r_max*0.05, r_max, t)
        x = CENTER[0] + r * math.cos(a)
        y = CENTER[1] + r * math.sin(a)
        infs.append(Influencer(x, y, strength=BASE_STRENGTH, mode=mode))
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
        infs.append(Influencer(x, y, strength=BASE_STRENGTH, mode=mode))
    return infs

def preset_scaffold(mode='attract'):
    return [Influencer(random.uniform(WIDTH*0.2, WIDTH*0.8),
                       random.uniform(HEIGHT*0.2, HEIGHT*0.8),
                       strength=BASE_STRENGTH, mode=mode)
            for _ in range(NUM_INFLUENCERS)]

def build_preset(name, mode):
    return {
        'ring':    lambda: preset_ring(mode),
        'star':    lambda: preset_starburst(mode),
        'spiral':  lambda: preset_spiral(mode),
        'flower':  lambda: preset_flower(mode),
        'scaffold':lambda: preset_scaffold(mode),
    }.get(name, lambda: preset_ring(mode))()

INFLUENCER_MODE = 'attract'
current_preset = 'flower'
INFLUENCERS = build_preset(current_preset, INFLUENCER_MODE)

# ===== Conductor (music goals) =====
class Conductor:
    """
    Chooses pretty, abstract 'goals' from audio features and
    smoothly morphs parameters/presets.
    """
    def __init__(self):
        self.energy_slow = 0.0
        self.energy_fast = 0.0
        self.low_slow = 0.0
        self.mid_slow = 0.0
        self.high_slow = 0.0
        self.last_preset_switch = time.time()
        self.goal = 'bloom'   # bloom, starburst, spiral, lace
        self.rotate = 0.0     # used to spin spiral/ring slowly

    def update(self, spectrum):
        # Normalize and band
        if spectrum.max() > 0:
            sp = spectrum / spectrum.max()
        else:
            sp = spectrum
        n = len(sp)
        low  = float(np.mean(sp[:max(2, n//8)]))
        mid  = float(np.mean(sp[n//8:n//3]))
        high = float(np.mean(sp[-n//6:]))

        energy = float(np.mean(sp))
        # Smooths
        self.energy_fast = lerp(self.energy_fast, energy, 0.35)
        self.energy_slow = lerp(self.energy_slow, energy, 0.05)
        self.low_slow  = lerp(self.low_slow,  low,  0.10)
        self.mid_slow  = lerp(self.mid_slow,  mid,  0.10)
        self.high_slow = lerp(self.high_slow, high, 0.10)

        # Burst detection (snare/clap-ish)
        burst = self.energy_fast - self.energy_slow > 0.08 or (mid - self.mid_slow > 0.07)

        # Goal logic
        now = time.time()
        if burst:
            self.goal = 'starburst'
            self.last_preset_switch = now
        else:
            # calm vs energetic
            if self.low_slow > 0.22 and self.energy_slow > 0.18:
                self.goal = 'spiral'  # flowing build
            elif self.high_slow > 0.20 and self.energy_slow > 0.14:
                self.goal = 'lace'    # airy highs
            else:
                self.goal = 'bloom'   # gentle base

        # Slow rotation for hypnotic feel
        self.rotate += 0.12 * (0.4 + self.energy_slow)

        # Periodic auto-cycle (avoid being stuck)
        if now - self.last_preset_switch > 12.0 and self.goal != 'starburst':
            self.last_preset_switch = now
            # nudge to a new base shape to keep it fresh
            cycle = ['ring','flower','spiral','star']
            global current_preset, INFLUENCERS
            current_preset = cycle[(cycle.index(current_preset)+1) % len(cycle)]
            INFLUENCERS = build_preset(current_preset, INFLUENCER_MODE)

        # Expose parameters for renderer
        return {
            'low': self.low_slow,
            'mid': self.mid_slow,
            'high': self.high_slow,
            'energy': self.energy_slow,
            'burst': burst,
            'goal': self.goal,
            'rotate': self.rotate
        }

# ===== Audio change analyzer (spectral flux + onsets) =====
class AudioAnalyzer:
    def __init__(self, chunk):
        self.prev = np.zeros(chunk//2+1, dtype=float)
        self.flux_ema = 0.0
        self.flux_var = 0.0
        self.k1 = 0.25   # EMA gain for mean
        self.k2 = 0.15   # EMA gain for variance-ish
        self.onset_cool = 0.0

    def update(self, spectrum):
        sp = spectrum.astype(float)
        if sp.max() > 0: sp = sp / sp.max()
        # spectral flux (reactivity to change)
        diff = sp - self.prev
        flux = float(np.sum(np.clip(diff, 0, None)))

        # EMA mean/var (rough)
        self.flux_ema = (1-self.k1)*self.flux_ema + self.k1*flux
        d = flux - self.flux_ema
        self.flux_var = (1-self.k2)*self.flux_var + self.k2*(d*d)
        thresh = self.flux_ema + 0.9*math.sqrt(max(1e-6, self.flux_var))
        onset = flux > thresh and self.onset_cool <= 0.0
        if onset: self.onset_cool = 0.10
        else: self.onset_cool = max(0.0, self.onset_cool - 1.0/FPS)

        n = len(sp)
        low  = float(np.mean(sp[:max(2,n//8)]))
        mid  = float(np.mean(sp[n//8:n//3]))
        high = float(np.mean(sp[-n//6:]))

        energy = float(np.mean(sp))

        self.prev = sp

        return {"sp":sp, "energy":energy, "low":low, "mid":mid, "high":high,
                "flux":flux, "onset":onset}

# ===== Transient Influencers (firework sparks) =====
class TransientInfluencer(Influencer):
    def __init__(self, x, y, life=0.6, strength=1.5, radius=220, mode='repel'):
        super().__init__(x, y, strength=strength, radius=radius, mode=mode)
        self.life = life

    def step(self, dt):
        self.life -= dt
        return self.life > 0.0

# ===== Motif engine (alternates burst types) =====
class MotifEngine:
    def __init__(self):
        self.mode_ix = 0
        self.modes = ['ripple','starburst','swirlstorm']
        self.cool = 0.0
        self.transients = []

    def trigger(self, kind, center, band_boost):
        cx, cy = center
        if kind == 'ripple':
            r = min(WIDTH,HEIGHT)*(0.18 + 0.15*band_boost)
            for i in range(10):
                a = 2*math.pi*i/10.0
                x = cx + r*math.cos(a); y = cy + r*math.sin(a)
                self.transients.append(TransientInfluencer(x,y,life=0.7, strength=1.2+0.8*band_boost, radius=200, mode='attract'))
        elif kind == 'starburst':
            arms = 8
            r = min(WIDTH,HEIGHT)*(0.25 + 0.2*band_boost)
            for k in range(arms):
                a = 2*math.pi*k/arms
                x = cx + r*math.cos(a); y = cy + r*math.sin(a)
                self.transients.append(TransientInfluencer(x,y,life=0.5, strength=1.6+1.0*band_boost, radius=230, mode='repel'))
        elif kind == 'swirlstorm':
            for i in range(12):
                a = 2*math.pi*i/12.0
                r = 90 + 80*band_boost
                x = cx + r*math.cos(a); y = cy + r*math.sin(a)
                t = TransientInfluencer(x,y,life=0.8, strength=1.4+0.9*band_boost, radius=210, mode='swirl')
                self.transients.append(t)
        self.cool = 0.12

    def maybe_trigger_from_features(self, feat):
        if self.cool > 0:
            self.cool -= 1.0/FPS
            return
        if feat['onset']:
            kind = self.modes[self.mode_ix % len(self.modes)]
            self.mode_ix += 1
            band_boost = max(0.0, min(1.0, 0.6*feat['low'] + 0.3*feat['mid'] + 0.4*feat['high']))
            self.trigger(kind, CENTER, band_boost)

    def step(self, dt):
        self.transients = [t for t in self.transients if t.step(dt)]

conductor = Conductor()
analyzer = AudioAnalyzer(CHUNK)
motifs = MotifEngine()
color_engine = ColorEngine()
hue_inverted = False

# ===== Main loop =====
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

    # Stronger, proportional motion: base energy plus change (flux)
    if AUDIO_COUPLED:
        strength_boost = 1.0 + 0.9*params['energy'] + 0.8*params['low'] + 0.8*feat['flux']
        size_boost     = 1.0 + 0.8*params['low']   + 0.6*feat['flux']
        swirl_boost    = 1.0 + 1.2*params['high']  + 0.7*feat['flux']
    else:
        strength_boost = size_boost = swirl_boost = 1.0

    # Trigger visual motifs on onsets (alternating)
    motifs.maybe_trigger_from_features(feat)

    # Morph influencers based on goal (no global statements needed)
    if goal == 'bloom':
        INFLUENCER_MODE = 'attract'
        base_r = min(WIDTH, HEIGHT) * (0.26 + 0.09*params['low'])
        INFLUENCERS = preset_ring(mode=INFLUENCER_MODE, radius=base_r)
    elif goal == 'starburst':
        INFLUENCER_MODE = 'repel'
        INFLUENCERS = preset_starburst(mode=INFLUENCER_MODE, arms=6)
    elif goal == 'spiral':
        INFLUENCER_MODE = 'attract'
        INFLUENCERS = preset_spiral(mode=INFLUENCER_MODE, turns=1.6)
        # rotate influencers by small angle
        ang = 0.015 * params['rotate']
        ca, sa = math.cos(ang), math.sin(ang)
        for inf in INFLUENCERS:
            vx, vy = inf.pos[0]-CENTER[0], inf.pos[1]-CENTER[1]
            inf.pos[0] = CENTER[0] + vx*ca - vy*sa
            inf.pos[1] = CENTER[1] + vx*sa + vy*ca
    elif goal == 'lace':
        INFLUENCER_MODE = 'swirl'
        INFLUENCERS = preset_flower(mode=INFLUENCER_MODE, petals=7)

    # Apply audio-based magnitudes
    for inf in INFLUENCERS:
        inf.strength = BASE_STRENGTH * strength_boost
        base = INFLUENCE_RADIUS * (0.85 + 0.25*params['mid'])
        inf.radius = base
        if INFLUENCER_MODE == 'swirl':
            # make swirl tighter/looser with highs
            SWIRL_TWIST = 1.2 + 1.0*(swirl_boost-1.0)

    # Optional: hue flip on bigger swells
    energy = params['energy']
    if energy > 0.35 and not hue_inverted:
        for d in dots:
            d['color'] = invert_color_hue(d['color'])
        hue_inverted = True
    elif energy <= 0.35:
        hue_inverted = False

    # ==== Render ====
    screen.fill((0, 0, 0))
    motifs.step(dt)

    for d in dots:
        px, py = d['pos']
        fx_sum = 0.0
        fy_sum = 0.0
        mag_sum = 0.0
        # base influencers
        for inf in INFLUENCERS:
            fx, fy, mag = inf.field(px, py)
            fx_sum += fx; fy_sum += fy; mag_sum += mag
        # transient fireworks
        for tinf in motifs.transients:
            fx, fy, mag = tinf.field(px, py)
            fx_sum += fx; fy_sum += fy; mag_sum += 0.6*mag

        # stronger, change-aware speed
        speed = 90.0 + 150.0 * energy + 220.0 * feat['flux']
        px += fx_sum * speed * dt
        py += fy_sum * speed * dt

        # soft home spring to avoid drift
        hx, hy = d['home'][0]-px, d['home'][1]-py
        px += hx * 0.018
        py += hy * 0.018

        # clamp
        px = min(max(px, 6), WIDTH-6)
        py = min(max(py, 6), HEIGHT-6)
        d['pos'][0], d['pos'][1] = px, py

        r = DOT_BASE_RADIUS + min(DOT_MAX_BOOST, mag_sum * 2.2) * size_boost
        tsec = pygame.time.get_ticks() / 1000.0
        col = color_engine.color_for(mag_sum, feat, params, tsec)
        pygame.draw.circle(screen, col, (int(px), int(py)), int(r))

    if SHOW_INFLUENCERS:
        for inf in INFLUENCERS:
            col = {'attract': (120,255,140), 'repel': (255,130,130), 'swirl': (140,180,255)}[inf.mode]
            pygame.draw.circle(screen, col, (int(inf.pos[0]), int(inf.pos[1])), 6)
            pygame.draw.circle(screen, (60,60,60), (int(inf.pos[0]), int(inf.pos[1])), int(inf.radius), 1)
        # transient hints
        for tinf in motifs.transients:
            col = {'attract': (90,200,110), 'repel': (220,100,100), 'swirl': (110,150,230)}[tinf.mode]
            pygame.draw.circle(screen, col, (int(tinf.pos[0]), int(tinf.pos[1])), 4)

    hud_font = pygame.font.SysFont(None, 24)
    txt = f"{current_preset} | goal:{goal} | mode:{INFLUENCER_MODE} | audio:{'on' if AUDIO_COUPLED else 'off'}"
    img = hud_font.render(txt, True, (90, 120, 150))
    screen.blit(img, (18, 14))

    pygame.display.flip()

# Cleanup
stream.stop_stream(); stream.close(); p.terminate(); pygame.quit()
