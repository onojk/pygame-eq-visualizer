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
ALPHA_SPEED = 3.36
MAX_LENGTH = 50
MAX_DENSITY = 10  # max number of overlapping worms per cell
BODY_RADIUS = 6

# === Alpha ===
class Alpha:
    def __init__(self):
        self.pos = [random.randint(WIDTH // 3, WIDTH * 2 // 3), random.randint(HEIGHT // 3, HEIGHT * 2 // 3)]
        self.dir = [random.uniform(-1.0, 1.0), random.uniform(-0.3, 0.3)]
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

# === Worm Utilities ===
def invert_color_hue(rgb):
    r, g, b = [x / 255.0 for x in rgb]
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    h = (h + 0.5) % 1.0
    r2, g2, b2 = colorsys.hsv_to_rgb(h, s, v)
    return (int(r2 * 255), int(g2 * 255), int(b2 * 255))

def spawn_worm():
    side = random.choice(['top', 'bottom', 'left', 'right'])
    if side == 'top': x, y = random.randint(0, WIDTH), -100
    elif side == 'bottom': x, y = random.randint(0, WIDTH), HEIGHT + 100
    elif side == 'left': x, y = -100, random.randint(0, HEIGHT)
    else: x, y = WIDTH + 100, random.randint(0, HEIGHT)
    color = DEFAULT_COLOR if len(worms) % 2 == 0 else (0, 0, 0)
    return [[x, y]], color

# === Simulation ===
worms = []
worm_colors = []
alpha = Alpha()

# populate grid-like starting worms
grid_spacing = 50
for row in range(HEIGHT // grid_spacing):
    for col in range(WIDTH // grid_spacing):
        x = col * grid_spacing + grid_spacing // 2
        y = row * grid_spacing + grid_spacing // 2
        if (row * (WIDTH // grid_spacing) + col) % 2 == 0:
            worms.append([[x, y]])
            worm_colors.append(DEFAULT_COLOR)
        else:
            worms.append([[x, y]])
            worm_colors.append((0, 0, 0))

font = pygame.font.SysFont(None, 36)
face_mode = '^_^'
disappear_count = 0
face_timer = 0

running = True
while running:
    screen.fill((0, 0, 0))
    alpha.update()
    alpha_radius = alpha.get_pulse_radius()

    # spawn worms continuously
    new_worm, new_color = spawn_worm()
    worms.append(new_worm)
    worm_colors.append(new_color)

    # density check: remove overlapping worms
    positions = {}
    for i, worm in enumerate(worms):
        head = worm[-1]
        key = (int(head[0] // BODY_RADIUS), int(head[1] // BODY_RADIUS))
        if key in positions:
            positions[key].append(i)
        else:
            positions[key] = [i]

    dense_indices = set()
    for indices in positions.values():
        if len(indices) >= MAX_DENSITY:
            dense_indices.update(indices)

    # count and update face
    if dense_indices:
        disappear_count += len(dense_indices)
        if (disappear_count // 10) % 2 == 0:
            face_mode = '^_^'
            face_timer = 0
        else:
            face_mode = '-_-'
            face_timer = time.time()

    worms = [w for i, w in enumerate(worms) if i not in dense_indices]
    worm_colors = [c for i, c in enumerate(worm_colors) if i not in dense_indices]

    # update and draw worms
    for i, worm in enumerate(worms):
        head = worm[-1]
        dx = alpha.pos[0] - head[0]
        dy = alpha.pos[1] - head[1]
        dist = math.hypot(dx, dy)
        dx /= max(1, dist)
        dy /= max(1, dist)
        lag_factor = random.uniform(0.7, 1.3)
        wiggle = [math.sin(time.time() * 10 + i) * 0.5, math.cos(time.time() * 10 + i) * 0.5]
        new_head = [head[0] + dx * 3 * lag_factor + wiggle[0], head[1] + dy * 3 * lag_factor + wiggle[1]]
        worm.append(new_head)
        if len(worm) > MAX_LENGTH:
            worm.pop(0)

        # draw worm
        for j in range(len(worm) - 1):
            start = worm[j]
            end = worm[j + 1]
            alpha_line = j / MAX_LENGTH
            color = worm_colors[i]
            pygame.draw.line(screen, color, (int(start[0]), int(start[1])), (int(end[0]), int(end[1])), int(6 * (1 - alpha_line)))

    # draw alpha dot
    pygame.draw.circle(screen, ALPHA_COLOR, (int(alpha.pos[0]), int(alpha.pos[1])), alpha_radius)

    # draw narrator face
    if face_mode == '-_-' and time.time() - face_timer > 0.5:
        face_mode = '^_^'
    face_surface = font.render(face_mode, True, (255, 255, 255))
    screen.blit(face_surface, (10, 10))

    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            running = False
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_f:
            screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
