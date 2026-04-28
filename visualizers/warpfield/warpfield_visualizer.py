import pygame
import numpy as np
import pyaudio
import math
import random
import colorsys
from scipy.fftpack import fft

# === Audio Settings ===
CHUNK = 512
CHANNELS = 1
RATE = 22050

# === Pygame Init ===
pygame.init()
screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
info = pygame.display.Info()
clock = pygame.time.Clock()

WIDTH, HEIGHT = info.current_w, info.current_h
CENTER = (WIDTH // 2, HEIGHT // 2)
FPS = 60

# === Dot Settings ===
DOT_RADIUS = 6
GRID_SPACING = 50
DEFAULT_DOT_COLOR = (180, 220, 255)
ALPHA_BASE_RADIUS = DOT_RADIUS + 2

NUM_COLS = WIDTH // GRID_SPACING
NUM_ROWS = HEIGHT // GRID_SPACING
ALPHA_SPEED = 4.5  # exaggerate alpha speed
NUM_ALPHAS = 7  # more alphas

# === Color Hue Inversion ===
def invert_color_hue(rgb):
    r, g, b = [x / 255.0 for x in rgb]
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    h = (h + 0.5) % 1.0
    r2, g2, b2 = colorsys.hsv_to_rgb(h, s, v)
    return (int(r2 * 255), int(g2 * 255), int(b2 * 255))

# === Dot Field ===
dots = []
for row in range(NUM_ROWS):
    for col in range(NUM_COLS):
        x = col * GRID_SPACING + GRID_SPACING // 2
        y = row * GRID_SPACING + GRID_SPACING // 2
        dots.append({
            'pos': [x, y],
            'home': [x, y],
            'dir': [0, 0],
            'speed_mult': 0.15,
            'scatter_target': None,
            'color': DEFAULT_DOT_COLOR
        })

# === Alpha Gravity Point ===
class Alpha:
    def __init__(self):
        self.pos = [random.randint(WIDTH // 4, WIDTH * 3 // 4),
                    random.randint(HEIGHT // 4, HEIGHT * 3 // 4)]
        self.dir = [random.uniform(-2.0, 2.0), random.uniform(-2.0, 2.0)]
        self.speed = ALPHA_SPEED
        self.spawn_time = pygame.time.get_ticks()

    def update(self, beat_detected=False):
        for i in [0, 1]:
            if random.random() < 0.04:
                self.dir[i] += random.uniform(-0.5, 0.5)
            if beat_detected:
                self.dir[i] += random.uniform(-6.0, 6.0)
            self.pos[i] += self.dir[i] * self.speed
            if self.pos[i] < 0 or self.pos[i] > (WIDTH if i == 0 else HEIGHT):
                self.dir[i] *= -1

alphas = [Alpha() for _ in range(NUM_ALPHAS)]

# === Audio Input ===
p = pyaudio.PyAudio()
def get_input_device_index():
    return p.get_default_input_device_info()['index']

def open_audio_stream():
    return p.open(format=pyaudio.paInt16,
                  channels=CHANNELS,
                  rate=RATE,
                  input=True,
                  input_device_index=get_input_device_index(),
                  frames_per_buffer=CHUNK)

stream = open_audio_stream()

# === State Machine ===
STATE = "formation"
INITIALIZED = False
STATE_START = pygame.time.get_ticks()

def switch_state(new_state):
    global STATE, STATE_START
    STATE = new_state
    STATE_START = pygame.time.get_ticks()
    print(f"🔁 Switched to: {STATE}")

# === Main Loop ===
def start_warping():
    global INITIALIZED
    if not INITIALIZED and STATE == "formation":
        for dot in dots:
            dot['color'] = invert_color_hue(dot['color'])
            dot['inverted'] = True
        switch_state("warping")
        INITIALIZED = True

running = True
start_time = pygame.time.get_ticks()
last_speedup_time = start_time

while running:
    start_warping()
    dt = clock.tick(FPS)
    screen.fill((0, 0, 0))

    try:
        data = np.frombuffer(stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16)
    except IOError:
        continue

    audio_bars = np.abs(fft(data))[:CHUNK // 2]
    audio_bars = audio_bars / np.max(audio_bars) if np.max(audio_bars) != 0 else np.zeros_like(audio_bars)
    avg_amplitude = np.mean(audio_bars)
    is_beat = avg_amplitude > 0.12

    for alpha in alphas:
        alpha.update(beat_detected=is_beat)

    all_close = True
    for dot in dots:
        total_dx = 0
        total_dy = 0
        for alpha in alphas:
            dx = alpha.pos[0] - dot['pos'][0]
            dy = alpha.pos[1] - dot['pos'][1]
            dist = math.hypot(dx, dy)
            if dist > 30:
                all_close = False
            if dist > 1:
                norm_dx = dx / dist
                norm_dy = dy / dist
                force_scale = 32.0 if is_beat else 18.0
                total_dx += norm_dx * force_scale
                total_dy += norm_dy * force_scale

        if STATE == "warping":
            # Apply easing factor to make warping smoother
            easing = 0.16 if is_beat else 0.10
            dot['pos'][0] += total_dx * easing
            dot['pos'][1] += total_dy * easing

        elif STATE == "paused":
            # Return dots to their home positions
            dx = dot['home'][0] - dot['pos'][0]
            dy = dot['home'][1] - dot['pos'][1]
            dist = math.hypot(dx, dy)
            if dist > 1:
                dot['pos'][0] += dx / dist * 2.0
                dot['pos'][1] += dy / dist * 2.0

        elif STATE == "dispersing":
            if dot['scatter_target'] is not None:
                tx, ty = dot['scatter_target']
                sdx = tx - dot['pos'][0]
                sdy = ty - dot['pos'][1]
                sdist = math.hypot(sdx, sdy)
                if sdist > 1:
                    dot['pos'][0] += sdx / sdist * 2.0
                    dot['pos'][1] += sdy / sdist * 2.0
                else:
                    dot['scatter_target'] = None

        pygame.draw.circle(screen, dot['color'], (int(dot['pos'][0]), int(dot['pos'][1])), DOT_RADIUS)

    # State transition checks
    if STATE == "warping":
        total_x = sum(dot['pos'][0] for dot in dots)
        total_y = sum(dot['pos'][1] for dot in dots)
        center_x = total_x / len(dots)
        center_y = total_y / len(dots)

        near_cluster = 0
        for dot in dots:
            dx = dot['pos'][0] - center_x
            dy = dot['pos'][1] - center_y
            if math.hypot(dx, dy) < GRID_SPACING * 0.25:
                near_cluster += 1

        if near_cluster >= len(dots) * 0.8:
            switch_state("coalescing")
            pygame.time.set_timer(pygame.USEREVENT, 1000, True)

    elif STATE == "coalescing":
        # This state is just a timer before switching to paused
        pass

    elif STATE == "paused":
        # Check if all dots have returned home
        all_home = all(
            math.hypot(dot['home'][0] - dot['pos'][0], dot['home'][1] - dot['pos'][1]) < 2
            for dot in dots
        )
        if all_home:
            switch_state("formation")
            INITIALIZED = False

    elif STATE == "dispersing":
        if all(dot['scatter_target'] is None for dot in dots):
            for dot in dots:
                dot['color'] = invert_color_hue(dot['color'])
                dot['inverted'] = not dot.get('inverted', False)
            switch_state("warping")

    # Speed up dots over time
    now = pygame.time.get_ticks()
    if now - last_speedup_time >= 10000:
        last_speedup_time = now
        for dot in dots:
            if dot['speed_mult'] < 0.95:
                dot['speed_mult'] = min(dot['speed_mult'] + 0.1, 0.95)

    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.USEREVENT and STATE == "coalescing":
            for dot in dots:
                dot['color'] = invert_color_hue(dot['color'])
                dot['inverted'] = True
            switch_state("paused")
        elif event.type == pygame.USEREVENT and STATE == "paused":
            for dot in dots:
                angle = random.uniform(0, 2 * math.pi)
                spread = random.uniform(0.10, 0.25)
                tx = dot['home'][0] + math.cos(angle) * WIDTH * spread
                ty = dot['home'][1] + math.sin(angle) * HEIGHT * spread
                dot['scatter_target'] = [tx, ty]
            switch_state("dispersing")
        elif event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            running = False
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_f:
            screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)

    pygame.display.flip()

# Cleanup
stream.stop_stream()
stream.close()
p.terminate()
pygame.quit()
