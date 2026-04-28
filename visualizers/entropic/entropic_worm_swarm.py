import pygame
import random
import math
import time
import colorsys

# === Init ===
pygame.init()
screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
info = pygame.display.Info()
clock = pygame.time.Clock()

WIDTH, HEIGHT = info.current_w, info.current_h
CENTER = (WIDTH // 2, HEIGHT // 2)
DOT_RADIUS = 6
DEFAULT_COLOR = (180, 220, 255)
ALPHA_COLOR = (255, 100, 255)
ALPHA_SPEED = 3.36  # 112% of worm speed (which is now 1.5x original)
MAX_LENGTH = 50  # 250% longer
NUM_WORMS = (WIDTH // 50) * (HEIGHT // 50)

# === Alpha ===
class Alpha:
    def __init__(self):
        self.pos = [random.randint(WIDTH // 3, WIDTH * 2 // 3), random.randint(HEIGHT // 3, HEIGHT * 2 // 3)]
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
        return int(DOT_RADIUS + 2 * scale)

# === Worms ===
def sphere_position(index):
    angle = (index / NUM_WORMS) * 2 * math.pi
    base_radius = min(WIDTH, HEIGHT) // 3
    variance = random.uniform(0.8, 1.2)  # ±20% variation
    radius = base_radius * variance
    x = CENTER[0] + math.cos(angle) * radius
    y = CENTER[1] + math.sin(angle) * radius
    return [x, y]

def invert_color_hue(rgb):
    r, g, b = [x / 255.0 for x in rgb]
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    h = (h + 0.5) % 1.0
    r2, g2, b2 = colorsys.hsv_to_rgb(h, s, v)
    return (int(r2 * 255), int(g2 * 255), int(b2 * 255))

worms = [[sphere_position(i)] for i in range(NUM_WORMS)]
worm_colors = [DEFAULT_COLOR if i % 2 == 0 else (0, 0, 0) for i in range(NUM_WORMS)]
origins = [sphere_position(i) for i in range(NUM_WORMS)]

alpha = Alpha()
STATE = "coalescing"
STATE_START = time.time()

def switch_state(new_state):
    global STATE, STATE_START
    STATE = new_state
    STATE_START = time.time()
    print(f"🔁 Switched to: {STATE}")
    if STATE == "paused":
        for i in range(len(worm_colors)):
            worm_colors[i] = invert_color_hue(worm_colors[i])

running = True
while running:
    screen.fill((0, 0, 0))
    alpha.update()
    alpha_radius = alpha.get_pulse_radius()
    all_close = True

    for i, worm in enumerate(worms):
        head = worm[-1]
        if STATE == "coalescing":
            dx = alpha.pos[0] - head[0]
            dy = alpha.pos[1] - head[1]
            dist = math.hypot(dx, dy)
            if dist > 50:
                all_close = False
            dx /= max(1, dist)
            dy /= max(1, dist)
            lag_factor = random.uniform(0.7, 1.3)  # ±30% lag variation
            wiggle = [math.sin(time.time() * 10 + i) * 0.5, math.cos(time.time() * 10 + i) * 0.5]
            new_head = [head[0] + dx * 3 * lag_factor + wiggle[0], head[1] + dy * 3 * lag_factor + wiggle[1]]
            worm.append(new_head)
            if len(worm) > MAX_LENGTH:
                worm.pop(0)

        elif STATE == "paused":
            pass

        elif STATE == "scattering":
            target = origins[i]
            dx = target[0] - head[0]
            dy = target[1] - head[1]
            dist = math.hypot(dx, dy)
            if dist > 1:
                dx /= dist
                dy /= dist
                lag_factor = random.uniform(0.7, 1.3)
                wiggle = [math.sin(time.time() * 10 + i) * 0.5, math.cos(time.time() * 10 + i) * 0.5]
                new_head = [head[0] + dx * 2 * lag_factor + wiggle[0], head[1] + dy * 2 * lag_factor + wiggle[1]]
                worm.append(new_head)
            if len(worm) > MAX_LENGTH:
                worm.pop(0)

        # Draw worm
        for j in range(len(worm) - 1):
            start = worm[j]
            end = worm[j + 1]
            alpha_line = j / MAX_LENGTH
            color = worm_colors[i]
            pygame.draw.line(screen, color, (int(start[0]), int(start[1])), (int(end[0]), int(end[1])), int(6 * (1 - alpha_line)))

    # === State Transitions ===
    if STATE == "coalescing" and all_close:
        switch_state("paused")
    elif STATE == "paused" and time.time() - STATE_START > 20:
        switch_state("scattering")
    elif STATE == "scattering" and all(math.hypot(w[-1][0] - origins[i][0], w[-1][1] - origins[i][1]) < 3 for i, w in enumerate(worms)):
        worms = [[origins[i]] for i in range(NUM_WORMS)]
        switch_state("coalescing")

    pygame.draw.circle(screen, ALPHA_COLOR, (int(alpha.pos[0]), int(alpha.pos[1])), alpha_radius)

    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            running = False
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_f:
            screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
