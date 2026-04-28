"""
aurora_engine.py — Shared systems for AuroraSunsetGarden

Contains:
- Utils: hsv255 / lerp / clamp + contrast helpers (luminance, apply_vignette, crush_blacks, draw_dark_halo)
- ColorEngine
- Influencer + presets (ring/star/spiral/flower/scaffold)
- MotifEngine (with transients)
- Background renderer, shape renderer, kaleidoscope blitter
- WaterfallSystem / FireworksSystem / NatureDirector / Fireflies
- TempoTracker (BPM estimate → speed scale)

All rendering respects alpha and is designed for additive/soft compositing.
"""
import math, random, colorsys, time
import pygame
import numpy as np

# ---------- utils ----------
def hsv255(h, s, v):
    r, g, b = colorsys.hsv_to_rgb(h % 1.0, max(0.0, min(1.0, s)), max(0.0, min(1.0, v)))
    return (int(r*255), int(g*255), int(b*255))

def lerp(a, b, t):
    return a + (b - a) * t

def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x

# ---------- contrast & tone helpers ----------
def luminance(rgb):
    r, g, b = rgb
    return (0.2126*r + 0.7152*g + 0.0722*b) / 255.0

_vignette_cache = {}
def apply_vignette(surface, amount=0.6):
    """Darken edges to center (0..1)."""
    if amount <= 0.0:
        return
    W, H = surface.get_size()
    key = (W, H)
    if key not in _vignette_cache:
        mask = pygame.Surface((W, H), pygame.SRCALPHA)
        cx, cy = W/2, H/2
        max_d = ( (cx*cx + cy*cy) ** 0.5 )
        step = 3
        for y in range(0, H, step):
            for x in range(0, W, step):
                d = ((x-cx)**2 + (y-cy)**2) ** 0.5
                k = min(1.0, d / max_d)      # 0 center → 1 edge
                a = int(255 * (k**1.5))      # steeper at the end
                mask.fill((0, 0, 0, a), rect=pygame.Rect(x, y, step, step))
        _vignette_cache[key] = mask
    if amount >= 1.0:
        surface.blit(_vignette_cache[(W, H)], (0, 0), special_flags=pygame.BLEND_PREMULTIPLIED)
    else:
        scaled = _vignette_cache[(W, H)].copy()
        scaled.set_alpha(int(255 * amount))
        surface.blit(scaled, (0, 0), special_flags=pygame.BLEND_PREMULTIPLIED)

def crush_blacks(surface, amount=0.25):
    """Approximate contrast curve by multiplying RGB down (0..1)."""
    if amount <= 0.0:
        return
    W, H = surface.get_size()
    mul = int(255 * (1.0 - 0.30*amount))  # 1.0..0.70
    overlay = pygame.Surface((W, H), pygame.SRCALPHA)
    overlay.fill((mul, mul, mul, 255))
    surface.blit(overlay, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)

def draw_dark_halo(surface, pos, radius, opacity=160):
    """Soft black halo behind a bright element."""
    r = int(max(2, radius))
    s = pygame.Surface((r*2, r*2), pygame.SRCALPHA)
    pygame.draw.circle(s, (0, 0, 0, int(opacity*0.70)), (r, r), r)
    pygame.draw.circle(s, (0, 0, 0, opacity), (r, r), int(r*0.66))
    surface.blit(s, (int(pos[0]-r), int(pos[1]-r)), special_flags=pygame.BLEND_PREMULTIPLIED)

# ---------- color engine ----------
class ColorEngine:
    def __init__(self):
        self.base_h = 0.55
        self.flash = 0.0
        self.hue_drift = 0.0
    def tick(self, feat, params, dt):
        self.hue_drift += dt * (0.06 + 0.25*params.get("energy",0) + 0.35*feat.get("flux",0))
        if feat.get("onset", False):
            self.flash = min(1.0, self.flash + 0.65)
            self.base_h = (self.base_h + 0.08 + 0.12*feat.get("flux",0)) % 1.0
        self.flash *= (0.90 ** (dt*60))
    def color_for(self, local_mag, feat, params, t_seconds, theme_hue_push=0.0):
        h = (self.base_h + 0.07*math.sin(t_seconds*0.6) + 0.12*self.hue_drift + theme_hue_push) % 1.0
        band_push = 0.12*params.get("bass",0) - 0.04*params.get("lowmid",0) + 0.10*params.get("air",0)
        h = (h + band_push) % 1.0
        s = 0.50 + 0.33*params.get("presence",0) + 0.28*feat.get("flux",0) + 0.22*self.flash
        v = 0.36 + 0.56*params.get("energy",0) + 0.46*min(1.0, local_mag*0.7) + 0.22*self.flash
        return hsv255(h, s, v)

# ---------- influencer field ----------
class Influencer:
    FALLOFF_POWER = 2.15
    SWIRL_TWIST = 1.4
    def __init__(self, x, y, strength=1.0, radius=280.0, mode='attract'):
        self.pos = [float(x), float(y)]
        self.strength = strength
        self.radius = radius
        self.mode = mode
    def field(self, px, py):
        dx = self.pos[0] - px; dy = self.pos[1] - py
        d = math.hypot(dx, dy) + 1e-6
        if d > self.radius:
            return 0.0, 0.0, 0.0
        t = max(0.0, 1.0 - (d / self.radius) ** self.FALLOFF_POWER)
        ndx, ndy = dx/d, dy/d
        if self.mode == 'attract': fx, fy = ndx, ndy
        elif self.mode == 'repel': fx, fy = -ndx, -ndy
        elif self.mode == 'swirl':
            fx, fy = -ndy, ndx
            twist = self.SWIRL_TWIST * t
            fx = fx * (0.7 + 0.3*twist) + 0.2*ndx
            fy = fy * (0.7 + 0.3*twist) + 0.2*ndy
        else: fx, fy = ndx, ndy
        mag = self.strength * t
        return fx*mag, fy*mag, mag

# ---------- presets ----------
def preset_ring(mode, WIDTH, HEIGHT, CENTER, NUM_INFLUENCERS, radius=None):
    r = radius or min(WIDTH, HEIGHT) * 0.34
    return [Influencer(CENTER[0] + r*math.cos(2*math.pi*i/NUM_INFLUENCERS),
                       CENTER[1] + r*math.sin(2*math.pi*i/NUM_INFLUENCERS),
                       strength=1.0, mode=mode)
            for i in range(NUM_INFLUENCERS)]

def preset_starburst(mode, WIDTH, HEIGHT, CENTER, arms=6, NUM_INFLUENCERS=12):
    infs=[]; r_inner=min(WIDTH,HEIGHT)*0.20; r_outer=min(WIDTH,HEIGHT)*0.38
    pts=[]
    for k in range(arms):
        a=2*math.pi*k/arms
        pts.append((CENTER[0]+r_outer*math.cos(a), CENTER[1]+r_outer*math.sin(a)))
        b=a+math.pi/arms
        pts.append((CENTER[0]+r_inner*math.cos(b), CENTER[1]+r_inner*math.sin(b)))
    for i in range(NUM_INFLUENCERS):
        x,y=pts[i%len(pts)]; infs.append(Influencer(x,y,strength=1.0,mode=mode))
    return infs

def preset_spiral(mode, WIDTH, HEIGHT, CENTER, turns=1.9, NUM_INFLUENCERS=12):
    infs=[]; r_max=min(WIDTH,HEIGHT)*0.44
    for i in range(NUM_INFLUENCERS):
        t=i/max(1,NUM_INFLUENCERS-1); a=2*math.pi*turns*t; r=lerp(r_max*0.05, r_max, t)
        x=CENTER[0]+r*math.cos(a); y=CENTER[1]+r*math.sin(a)
        infs.append(Influencer(x,y,strength=1.0,mode=mode))
    return infs

def preset_flower(mode, WIDTH, HEIGHT, CENTER, petals=8, wobble=0.0, NUM_INFLUENCERS=12):
    R=min(WIDTH,HEIGHT)*0.30; infs=[]
    for i in range(NUM_INFLUENCERS):
        t=i/NUM_INFLUENCERS; a=2*math.pi*t; r=R*(1.0+0.34*math.cos(petals*a+wobble))
        x=CENTER[0]+r*math.cos(a); y=CENTER[1]+r*math.sin(a)
        infs.append(Influencer(x,y,strength=1.0,mode=mode))
    return infs

def preset_scaffold(mode, WIDTH, HEIGHT, NUM_INFLUENCERS=12):
    return [Influencer(random.uniform(WIDTH*0.2, WIDTH*0.8),
                       random.uniform(HEIGHT*0.2, HEIGHT*0.8),
                       strength=1.0, mode=mode)
            for _ in range(NUM_INFLUENCERS)]

# ---------- motifs ----------
class TransientInfluencer(Influencer):
    def __init__(self, x, y, life=0.6, strength=1.5, radius=220, mode='repel'):
        super().__init__(x, y, strength, radius, mode)
        self.life=life; self.age=0.0
    def step(self, dt):
        self.life-=dt; self.age+=dt
        return self.life>0.0

class MotifEngine:
    def __init__(self, WIDTH, HEIGHT, CENTER):
        from collections import deque
        self.mode_ix=0; self.modes=['ripple','starburst','swirlstorm']
        self.cool=0.0; self.transients=[]; self.sparkles=deque(maxlen=2000)
        self.WIDTH=WIDTH; self.HEIGHT=HEIGHT; self.CENTER=CENTER
    def trigger(self, kind, center, band_boost):
        cx, cy = center; WIDTH, HEIGHT = self.WIDTH, self.HEIGHT
        if kind=='ripple':
            r=min(WIDTH,HEIGHT)*(0.18+0.15*band_boost)
            for i in range(12):
                a=2*math.pi*i/12.0; x=cx+r*math.cos(a); y=cy+r*math.sin(a)
                self.transients.append(TransientInfluencer(x,y,life=0.7,strength=1.2+0.8*band_boost,radius=200,mode='attract'))
        elif kind=='starburst':
            arms=10; r=min(WIDTH,HEIGHT)*(0.25+0.2*band_boost)
            for k in range(arms):
                a=2*math.pi*k/arms; x=cx+r*math.cos(a); y=cy+r*math.sin(a)
                self.transients.append(TransientInfluencer(x,y,life=0.5,strength=1.6+1.0*band_boost,radius=240,mode='repel'))
        elif kind=='swirlstorm':
            for i in range(14):
                a=2*math.pi*i/14.0
                r=90+90*band_boost
                x=cx+r*math.cos(a); y=cy+r*math.sin(a)
                self.transients.append(TransientInfluencer(x,y,life=0.8,strength=1.4+0.9*band_boost,radius=210,mode='swirl'))
        # sparkles
        for _ in range(80 + int(160*band_boost)):
            ang=random.uniform(0,2*math.pi); spd=random.uniform(30,280)*(0.5+band_boost)
            life=random.uniform(0.25,1.2)
            self.sparkles.append({'x':cx,'y':cy,'vx':math.cos(ang)*spd,'vy':math.sin(ang)*spd,'life':life,'age':0.0})
        self.cool=0.12
    def maybe_trigger_from_features(self, feat, bands):
        if self.cool>0: self.cool-=1.0/60; return
        if feat.get('onset',False):
            kind=self.modes[self.mode_ix%len(self.modes)]; self.mode_ix+=1
            band_boost=clamp(0.5*bands.get('bass',0)+0.4*bands.get('presence',0)+0.3*bands.get('air',0),0.0,1.0)
            self.trigger(kind, self.CENTER, band_boost)
    def step(self, dt):
        self.transients=[t for t in self.transients if t.step(dt)]
        alive = type(self.sparkles)(maxlen=self.sparkles.maxlen)
        for s in self.sparkles:
            s['age']+=dt
            if s['age']<s['life']:
                s['x']+=s['vx']*dt; s['y']+=s['vy']*dt; s['vy']+=12*dt
                if -50<=s['x']<=self.WIDTH+50 and -50<=s['y']<=self.HEIGHT+50: alive.append(s)
        self.sparkles=alive

# ---------- background ----------
def draw_background(surface, theme, bands, t, WIDTH, HEIGHT):
    sub=bands.get('sub',0); bass=bands.get('bass',0); air=bands.get('air',0); presence=bands.get('presence',0)
    if theme=='Sunset':
        top=hsv255(0.60+0.02*math.sin(t*0.2),0.35,0.12+0.18*air)
        bot=hsv255(0.05+0.03*math.sin(t*0.15),0.85,0.46+0.42*(0.5*bass+0.5*sub))
    elif theme=='Ocean':
        top=hsv255(0.50,0.30,0.10+0.20*air); bot=hsv255(0.52,0.75,0.38+0.45*(0.4*bass+0.6*presence))
    elif theme=='Forest':
        top=hsv255(0.33,0.40,0.12+0.16*air); bot=hsv255(0.33,0.80,0.36+0.45*(0.5*bass+0.5*sub))
    else:
        top=hsv255(0.70,0.25,0.10+0.25*air); bot=hsv255(0.75,0.65,0.28+0.40*(0.3*bass+0.7*presence))
    for y in range(0, HEIGHT, 4):
        k=y/HEIGHT; c=(int(top[0]*(1-k)+bot[0]*k), int(top[1]*(1-k)+bot[1]*k), int(top[2]*(1-k)+bot[2]*k))
        pygame.draw.rect(surface, c, (0,y,WIDTH,4))
    horizon_y=int(HEIGHT*0.62); bloom=int(40+220*(0.4*bass+0.6*sub))
    surf=pygame.Surface((WIDTH,bloom), pygame.SRCALPHA)
    g=int(70+150*(0.5*bass+0.5*sub)); pygame.draw.rect(surf,(255,255,255,g),(0,0,WIDTH,bloom))
    surface.blit(surf,(0,horizon_y-bloom//2), special_flags=pygame.BLEND_PREMULTIPLIED)
    # base vignette to seat brights on darker ground
    apply_vignette(surface, amount=0.35)

# ---------- geometry renderers ----------
def draw_shape(surface, name, pos, size, rot, color):
    x,y=pos
    # auto-ink for bright fills
    if luminance(color) > 0.72:
        draw_dark_halo(surface, (x,y), radius=size*1.3, opacity=150)

    if name=='dots':
        pygame.draw.circle(surface, color, (int(x),int(y)), int(size))
        return
    if name=='soft':
        r=int(size*1.7); s=pygame.Surface((r*2,r*2), pygame.SRCALPHA)
        pygame.draw.circle(s, (*color,110), (r,r), r)
        pygame.draw.circle(s, (*color,200), (r,r), int(r*0.66))
        surface.blit(s,(int(x-r),int(y-r)), special_flags=pygame.BLEND_PREMULTIPLIED)
        return

    points=[]
    if name=='petals':
        k=5
        for i in range(k):
            a=rot+2*math.pi*i/k; r=size*(1.4 if i%2==0 else 0.7); points.append((x+r*math.cos(a), y+r*math.sin(a)))
    elif name=='tri':
        k=3
        for i in range(k): a=rot+2*math.pi*i/k; points.append((x+size*1.2*math.cos(a), y+size*1.2*math.sin(a)))
    elif name=='quad':
        k=4
        for i in range(k): a=rot+math.pi/4+2*math.pi*i/k; points.append((x+size*1.1*math.cos(a), y+size*1.1*math.sin(a)))
    elif name=='star':
        k=5
        for i in range(k*2): a=rot+2*math.pi*i/(k*2); r=size*(1.6 if i%2==0 else 0.6); points.append((x+r*math.cos(a), y+r*math.sin(a)))
    elif name=='ribbons':
        length=size*3; dx=math.cos(rot); dy=math.sin(rot)
        p1=(x-dx*length*0.5, y-dy*length*0.5); p2=(x+dx*length*0.5, y+dy*length*0.5)
        if luminance(color) > 0.72:
            pygame.draw.line(surface, (0,0,0,180), p1, p2, max(1, int(size*1.2)))
        pygame.draw.line(surface, color, p1, p2, max(1, int(size*0.8)))
        return

    if points:
        # polygon ink underlay for bright fills (slightly larger)
        if luminance(color) > 0.72:
            cx, cy = x, y
            inflated = [ (cx+(px-cx)*1.06, cy+(py-cy)*1.06) for (px,py) in points ]
            pygame.draw.polygon(surface, (0,0,0,160), inflated)
        pygame.draw.polygon(surface, color, points)

# ---------- kaleidoscope ----------
def blit_with_symmetry(base, dest, WIDTH, HEIGHT, sectors=4):
    sectors = sectors or 1
    if sectors == 1:
        dest.blit(base,(0,0)); return
    for s in range(sectors):
        angle=(360/sectors)*s; rotated=pygame.transform.rotozoom(base, angle, 1.0)
        if s%2==1: rotated=pygame.transform.flip(rotated, True, False)
        dest.blit(rotated,(0,0), special_flags=pygame.BLEND_ADD)

# ---------- Waterfall / Fireworks / Nature / Fireflies ----------
class WaterfallSystem:
    def __init__(self, WIDTH, HEIGHT):
        self.WIDTH=WIDTH; self.HEIGHT=HEIGHT; self.offset=0.0
        self.columns=[]; col_w=8
        for x in range(0, WIDTH, col_w):
            phase=random.random()*2*math.pi; width=col_w+random.randint(0,6); alpha=random.randint(20,45)
            self.columns.append({'x':x,'w':width,'phase':phase,'alpha':alpha})
        self.mist=pygame.Surface((WIDTH, int(HEIGHT*0.25)), pygame.SRCALPHA)
    def draw(self, surface, bands, dt, speed_scale=1.0):
        WIDTH, HEIGHT=self.WIDTH, self.HEIGHT
        speed=(80 + 220*(0.5*bands.get('bass',0)+0.5*bands.get('sub',0))) * speed_scale
        self.offset=(self.offset+speed*dt)%HEIGHT
        veil=pygame.Surface((WIDTH,HEIGHT), pygame.SRCALPHA)
        for c in self.columns:
            amp=18+34*bands.get('presence',0); sway=math.sin(self.offset*0.005+c['phase'])*amp
            a=int(c['alpha']+80*bands.get('air',0)); x=int(c['x']+sway)
            pygame.draw.rect(veil, (200,220,255, clamp(a,10,180)), (x,0,c['w'],HEIGHT))
        surface.blit(veil,(0,0), special_flags=pygame.BLEND_PREMULTIPLIED)
        self.mist.fill((0,0,0,0))
        fog_a=int(30+120*(0.4*bands.get('presence',0)+0.6*bands.get('air',0)))
        pygame.draw.rect(self.mist,(230,235,255, clamp(fog_a,20,160)), (0,0,WIDTH,self.mist.get_height()))
        surface.blit(self.mist,(0,int(HEIGHT*0.75)), special_flags=pygame.BLEND_PREMULTIPLIED)

class FireworksSystem:
    """
    Lightweight fireworks with variable brightness/size/intensity/glow,
    randomized peak heights, and cheap ember trails.

    Public API (compatible):
      - maybe_launch(feat, params, speed_scale)
      - launch(speed_scale=1.0)
      - step(dt, speed_scale)
      - draw(surface)
    """
    def __init__(self, W, H):
        import pygame
        self.W, self.H = W, H
        self.shells = []   # active shells
        self.embers = []   # pooled embers
        self.rng = __import__("random")
        self._glow_cache = {}  # radius -> soft disc surface
        self.max_shells = 8
        self.max_embers = 900   # hard cap for perf

    # ---------- utilities ----------
    def _soft_disc(self, r, color, alpha=255):
        import pygame
        key = (r, color, alpha)
        if key in self._glow_cache:
            return self._glow_cache[key]
        d = r*2+2
        surf = pygame.Surface((d,d), pygame.SRCALPHA)
        cx = cy = d//2
        # radial falloff (1 - (d/r)^2)^2 -> nice gaussian-ish
        for y in range(d):
            for x in range(d):
                dx, dy = x-cx, y-cy
                rr = (dx*dx + dy*dy) ** 0.5
                if rr <= r:
                    k = 1.0 - (rr/r)
                    a = max(0, min(255, int(alpha * (k*k))))
                    surf.set_at((x,y), (*color, a))
        self._glow_cache[key] = surf
        return surf

    def _rand_col(self, intensity=1.0):
        # vivid but not pure white; intensity scales v
        import colorsys
        h = self.rng.random()
        s = self.rng.uniform(0.65, 1.0)
        v = min(1.0, 0.8 + 0.2*intensity)
        r,g,b = colorsys.hsv_to_rgb(h, s, v)
        return (int(r*255), int(g*255), int(b*255))

    # ---------- spawners ----------
    def maybe_launch(self, feat, params, speed_scale):
        # Keep your existing bass/energy linkage if desired; here we bias to activity.
        energy = params.get('energy', 0.2)
        bass   = params.get('bass', 0.2)
        p = 0.6*energy + 0.9*bass
        p *= 0.25 * speed_scale  # overall rate
        if len(self.shells) < self.max_shells and self.rng.random() < p:
            self.launch(speed_scale=speed_scale)

    def launch(self, speed_scale=1.0):
        # Random lane (x), target peak height, velocity tuned by speed_scale
        x = self.rng.uniform(self.W*0.08, self.W*0.92)
        base_v = self.rng.uniform(420, 680) * speed_scale
        peak_y = self.rng.uniform(self.H*0.18, self.H*0.78)  # different heights

        # Visual personality
        size = self.rng.uniform(6, 14)              # base shell size (px)
        intensity = self.rng.uniform(0.7, 1.35)     # affects brightness/glow
        glow = self.rng.uniform(0.6, 1.4)           # soft aura strength
        color = self._rand_col(intensity)

        # Shell record
        self.shells.append({
            "pos": [x, self.H + 6.0],
            "vel": [self.rng.uniform(-60, 60), -base_v],
            "peak_y": peak_y,
            "size": size,
            "color": color,
            "glow": glow,
            "intensity": intensity,
            "state": "ascend",         # ascend -> burst -> done
            "life": self.rng.uniform(0.9, 1.3),  # safety life
        })

    # ---------- simulation ----------
    def step(self, dt, speed_scale):
        g = 540.0  # gravity (px/s^2)
        drag = 0.02

        new_shells = []
        for sh in self.shells:
            sh["life"] -= dt
            if sh["state"] == "ascend":
                # integrate
                sh["vel"][1] += g * dt
                sh["vel"][0] *= (1.0 - drag*dt)
                sh["pos"][0] += sh["vel"][0] * dt
                sh["pos"][1] += sh["vel"][1] * dt

                # ember drip while ascending
                if self.rng.random() < 4.0*dt:
                    self._emit_ember(sh, ascend=True)

                # peak reached?
                if sh["pos"][1] <= sh["peak_y"] or sh["vel"][1] >= -30.0 or sh["life"] <= 0.0:
                    sh["state"] = "burst"
                    # spawn a small ring of embers with outward kick
                    num = int(18 + 14*self.rng.random())
                    for _ in range(num):
                        self._emit_ember(sh, ascend=False)

            elif sh["state"] == "burst":
                # brief post-burst glow then done
                sh["life"] -= 1.5*dt
                if self.rng.random() < 14.0*dt:
                    self._emit_ember(sh, ascend=False, small=True)
                if sh["life"] <= 0.0:
                    sh["state"] = "done"

            if sh["state"] != "done" and -40 <= sh["pos"][1] <= self.H+40:
                new_shells.append(sh)

        self.shells = new_shells

        # Update embers (cheap)
        new_embers = []
        for e in self.embers:
            e["v"][1] += (320.0 + 80.0*e["drag"]) * dt
            e["v"][0] *= (1.0 - 0.4*dt*e["drag"])
            e["p"][0] += e["v"][0] * dt
            e["p"][1] += e["v"][1] * dt
            e["life"] -= (0.7 + 0.6*e["drag"]) * dt
            e["a"] = max(0, e["a"] - (120.0 + 80.0*self.rng.random())*dt)
            e["r"] = max(0.6, e["r"] - 20.0*dt)  # shrink
            if e["life"] > 0 and -30 <= e["p"][1] <= self.H+30:
                new_embers.append(e)
        # Hard cap for perf
        if len(new_embers) > self.max_embers:
            new_embers = new_embers[-self.max_embers:]
        self.embers = new_embers

    def _emit_ember(self, sh, ascend=True, small=False):
        # Basic ember with slightly randomized color and velocity
        spd =  self.rng.uniform(60, 180) if ascend else self.rng.uniform(120, 360)
        ang =  self.rng.uniform(-0.9, -2.25) if ascend else self.rng.uniform(0, 6.283)
        vx = spd * math.cos(ang) * (0.6 + 0.8*self.rng.random())
        vy = spd * math.sin(ang) * (0.6 + 0.8*self.rng.random())
        jitter = self.rng.uniform(-10,10)

        base = sh["intensity"]
        col = (
            min(255, int(sh["color"][0] * (0.8 + 0.5*self.rng.random()))),
            min(255, int(sh["color"][1] * (0.8 + 0.5*self.rng.random()))),
            min(255, int(sh["color"][2] * (0.8 + 0.5*self.rng.random()))),
        )
        self.embers.append({
            "p": [sh["pos"][0] + jitter*0.2, sh["pos"][1] + jitter*0.2],
            "v": [vx, vy - self.rng.uniform(20,90)],
            "r": (1.6 if small else 2.6) * (0.7 + 0.6*self.rng.random()),
            "a": int(210 * min(1.0, 0.6 + 0.8*base)),
            "c": col,
            "life": self.rng.uniform(0.45, 1.1 if ascend else 1.5),
            "drag": self.rng.uniform(0.2, 1.0),
        })

    # ---------- draw ----------
    def draw(self, surface):
        import pygame
        # draw embers first (additive-ish)
        for e in self.embers:
            r = int(e["r"])
            if r <= 0: 
                continue
            glow = self._soft_disc(max(2, r), e["c"], alpha=e["a"])
            surface.blit(glow, (int(e["p"][0]-glow.get_width()/2), int(e["p"][1]-glow.get_height()/2)), special_flags=pygame.BLEND_PREMULTIPLIED)

        # draw active shells with glow that varies per shell
        for sh in self.shells:
            size = int(sh["size"])
            c    = sh["color"]
            a    = int(180 * sh["intensity"])
            radius = max(2, int(size * (1.0 if sh["state"]=='ascend' else 1.6)))
            al = max(0, min(255, int(a * sh["glow"]))); glow = self._soft_disc(radius, c, alpha=al)
            surface.blit(glow, (int(sh["pos"][0]-glow.get_width()/2), int(sh["pos"][1]-glow.get_height()/2)), special_flags=pygame.BLEND_PREMULTIPLIED)

class NatureDirector:
    def __init__(self, WIDTH, HEIGHT, FPS=60):
        self.WIDTH=WIDTH; self.HEIGHT=HEIGHT; self.running=True; self.FPS=FPS
        self.wind_phase=random.random()*1000; self.wind_dir=random.random()*2*math.pi; self.wind_gust=0.0
        self.season_hue=random.uniform(-0.04,0.04); self.lightning=0.0; self.next_event=time.time()+random.uniform(3,8)
        self.flock=[]
    def maybe_step(self, feat, bands):
        if not self.running: return
        now=time.time()
        if now>=self.next_event:
            choice=random.random()
            if choice<0.4: self.wind_dir=random.random()*2*math.pi; self.wind_gust=min(1.0, 0.4+0.8*(0.5*bands.get('presence',0)+0.5*bands.get('air',0)))
            elif choice<0.6: self.season_hue=clamp(self.season_hue+random.uniform(-0.03,0.03), -0.12, 0.12)
            elif choice<0.8 and (feat.get('onset',False) or bands.get('presence',0)>0.35): self.lightning=1.0
            else: self.spawn_flock(bands)
            self.next_event=now+random.uniform(2.5,7.5)
        self.lightning*=0.88; self.wind_gust*=0.985
    def wind(self, x, y, t):
        base=40*self.wind_gust; kx=math.sin(0.07*t+0.0008*y); ky=math.sin(0.09*t+0.0008*x)
        dirx=math.cos(self.wind_dir); diry=math.sin(self.wind_dir)
        return base*(dirx*0.6+0.4*kx), base*(diry*0.6+0.4*ky)
    def spawn_flock(self, bands):
        count=random.randint(6,12); y=random.randint(int(self.HEIGHT*0.10), int(self.HEIGHT*0.35))
        speed=120+140*bands.get('air',0); ltr=random.random()<0.5; x0=-60 if ltr else self.WIDTH+60; vx=speed if ltr else -speed
        self.flock=[{'x':x0+i*20,'y':y+(i%5-2)*6,'vx':vx} for i in range(count)]
    def draw_overlays(self, surface):
        if self.lightning>0.02:
            a=int(160*self.lightning); flash=pygame.Surface((self.WIDTH,self.HEIGHT), pygame.SRCALPHA); flash.fill((255,255,255,a))
            surface.blit(flash,(0,0), special_flags=pygame.BLEND_PREMULTIPLIED)
        if self.flock:
            birds=[]
            for b in self.flock:
                b['x']+=b['vx']/self.FPS
                if -100<=b['x']<=self.WIDTH+100:
                    birds.append(b)
                    p1=(int(b['x']),int(b['y'])); p2=(int(b['x']-6),int(b['y']+4)); p3=(int(b['x']+6),int(b['y']+4))
                    pygame.draw.line(surface,(20,20,25),p1,p2,2); pygame.draw.line(surface,(20,20,25),p1,p3,2)
            self.flock=birds

class Fireflies:
    def __init__(self, WIDTH, HEIGHT, n=180):
        self.WIDTH=WIDTH; self.HEIGHT=HEIGHT; self.points=[{'x':random.uniform(0,WIDTH),'y':random.uniform(HEIGHT*0.55,HEIGHT*0.95),'a':random.random(),'r':random.uniform(1.5,3.0)} for _ in range(n)]
        self.surf=pygame.Surface((WIDTH,HEIGHT), pygame.SRCALPHA)
    def step(self, bands, dt):
        for p in self.points:
            p['a'] += (random.uniform(-0.6,0.6) + 0.6*bands.get('air',0))*dt
            p['x'] += (random.uniform(-12,12) + 18*(bands.get('bass',0)))*dt
            p['y'] += random.uniform(-10,10)*dt
            p['x'] = clamp(p['x'], 0, self.WIDTH)
            p['y'] = clamp(p['y'], self.HEIGHT*0.50, self.HEIGHT-5)
    def draw(self, surface, bands):
        self.surf.fill((0,0,0,0))
        for p in self.points:
            k=(0.5+0.5*math.sin(p['a']*6))*(0.4+0.6*(0.6*bands.get('presence',0)+0.4*bands.get('air',0)))
            c=hsv255(0.17+0.05*random.random(),0.6,0.6+0.4*k)
            pygame.draw.circle(self.surf, (*c, int(80+140*k)), (int(p['x']), int(p['y'])), int(p['r']))
        surface.blit(self.surf,(0,0), special_flags=pygame.BLEND_PREMULTIPLIED)

# ---------- tempo tracker ----------
class TempoTracker:
    """Estimate BPM from onsets; provide a smoothed speed scale mapped to [0.8, 1.6]."""
    def __init__(self, target_range=(70,180), fps=60):
        from collections import deque
        self.onsets=deque(maxlen=64)
        self.bpm_est=None
        self.smooth_scale=1.0
        self.fps=fps
        self.min_bpm, self.max_bpm = target_range
    def update(self, feat):
        if feat.get('onset', False):
            self.onsets.append(time.time())
    def estimate_bpm(self):
        if len(self.onsets) < 4:
            return self.bpm_est
        intervals=[t2-t1 for t1,t2 in zip(self.onsets, list(self.onsets)[1:]) if t2>t1]
        if not intervals:
            return self.bpm_est
        intervals.sort()
        mid=intervals[len(intervals)//2]
        if mid<=0: return self.bpm_est
        bpm=60.0/mid
        # fold to target range by octave (×2/÷2)
        while bpm < self.min_bpm: bpm*=2
        while bpm > self.max_bpm: bpm/=2
        self.bpm_est=bpm
        return bpm
    def speed_scale(self):
        bpm=self.estimate_bpm()
        if bpm is None:
            target=1.0
        else:
            # map bpm linearly to [0.8, 1.6]
            t=(bpm-self.min_bpm)/(self.max_bpm-self.min_bpm)
            target=0.8 + 0.8*clamp(t,0.0,1.0)
        self.smooth_scale = lerp(self.smooth_scale, target, 0.08)
        return self.smooth_scale
    def readable_bpm(self):
        return f"~{int(self.bpm_est)} BPM" if self.bpm_est else "~? BPM"
