import pygame
import random
import math
import time
import colorsys  # For hue inversion

pygame.init()
screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
info = pygame.display.Info()
clock = pygame.time.Clock()

WIDTH, HEIGHT = info.current_w, info.current_h
CENTER = (WIDTH // 2, HEIGHT // 2)

DOT_RADIUS = 6
GRID_SPACING = 50
DEFAULT_DOT_COLOR = (180, 220, 255)
ALPHA_COLOR = (255, 100, 255)
ALPHA_BASE_RADIUS = DOT_RADIUS + 2

NUM_COLS = WIDTH // GRID_SPACING
NUM_ROWS = HEIGHT // GRID_SPACING
ALPHA_SPEED = 2.0

# === Hue-based color inversion ===
def invert_color_hue(rgb):
    r, g, b = [x / 255.0 for x in rgb]
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    h = (h + 0.5) % 1.0  # Rotate hue 180°
    r2, g2, b2 = colorsys.hsv_to_rgb(h, s, v)
    return (int(r2 * 255), int(g2 * 255), int(b2 * 255))

# === Dots ===
dots = []
for row in range(NUM_ROWS):
    for col in range(NUM_COLS):
        x = col * GRID_SPACING + GRID_SPACING // 2
        y = row * GRID_SPACING + GRID_SPACING // 2
        dots.append({
            'pos': [x, y],
            'dir': [0, 0],
            'speed_mult': 0.1,
            'scatter_target': None,
            'color': DEFAULT_DOT_COLOR
        })

# === Alpha ===
class Alpha:
    def __init__(self):
        self.pos = [random.randint(WIDTH // 3, WIDTH * 2 // 3),
                    random.randint(HEIGHT // 3, HEIGHT * 2 // 3)]
        self.dir = [random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0)]
        self.speed = ALPHA_SPEED
        self.spawn_time = time.time()

    def update(self):
        for i in [0, 1]:
            if random.random() < 0.02:
                self.dir[i] += random.uniform(-0.3, 0.3)
            self.pos[i] += self.dir[i] * self.speed
            if self.pos[i] < 0 or self.pos[i] > (WIDTH if i == 0 else HEIGHT):
                self.dir[i] *= -1

    def get_pulse_radius(self):
        elapsed = time.time() - self.spawn_time
        pulse = math.sin(elapsed * 2.5)
        scale = 0.95 + 0.85 * ((pulse + 1) / 2)
        return int(ALPHA_BASE_RADIUS * scale)

alpha = Alpha()

# === State Machine ===
STATE = "coalescing"
STATE_START = time.time()

def switch_state(new_state):
    global STATE, STATE_START
    STATE = new_state
    STATE_START = time.time()
    print(f"🔁 Switched to: {STATE}")

    if STATE == "paused":
        # Invert dot colors using hue rotation
        for dot in dots:
            dot['color'] = invert_color_hue(dot['color'])

# === Timing ===
start_time = time.time()
last_speedup_time = start_time

running = True
while running:
    screen.fill((0, 0, 0))
    alpha.update()
    alpha_radius = alpha.get_pulse_radius()

    # === Dots Update ===
    all_close = True
    for dot in dots:
        dx = alpha.pos[0] - dot['pos'][0]
        dy = alpha.pos[1] - dot['pos'][1]
        dist = math.hypot(dx, dy)

        if STATE == "coalescing":
            if dist > 50:
                all_close = False
            norm_dx = dx / max(1, dist)
            norm_dy = dy / max(1, dist)
            speed = ALPHA_SPEED * dot['speed_mult']
            dot['pos'][0] += norm_dx * speed
            dot['pos'][1] += norm_dy * speed

        elif STATE == "paused":
            pass

        elif STATE == "scattering":
            if dot['scatter_target'] is None:
                angle = random.uniform(0, 2 * math.pi)
                spread = random.uniform(0.10, 0.20)
                target_x = dot['pos'][0] + math.cos(angle) * WIDTH * spread
                target_y = dot['pos'][1] + math.sin(angle) * HEIGHT * spread
                dot['scatter_target'] = [target_x, target_y]

            tx, ty = dot['scatter_target']
            sdx = tx - dot['pos'][0]
            sdy = ty - dot['pos'][1]
            sdist = math.hypot(sdx, sdy)
            if sdist > 1:
                dot['pos'][0] += sdx / sdist * 1.5
                dot['pos'][1] += sdy / sdist * 1.5
            else:
                dot['scatter_target'] = None

        pygame.draw.circle(screen, dot['color'], (int(dot['pos'][0]), int(dot['pos'][1])), DOT_RADIUS)

    # === State Transitions ===
    if STATE == "coalescing" and all_close:
        switch_state("paused")

    elif STATE == "paused" and time.time() - STATE_START > 20:
        switch_state("scattering")

    elif STATE == "scattering":
        if all(dot['scatter_target'] is None for dot in dots):
            switch_state("coalescing")

    # === Speed Ramp ===
    now = time.time()
    if now - last_speedup_time >= 10:
        last_speedup_time = now
        for dot in dots:
            if dot['speed_mult'] < 0.95:
                dot['speed_mult'] = min(dot['speed_mult'] + 0.1, 0.95)

    pygame.draw.circle(screen, ALPHA_COLOR, (int(alpha.pos[0]), int(alpha.pos[1])), alpha_radius)

    # === Events ===
    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            running = False
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_f:
            screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
