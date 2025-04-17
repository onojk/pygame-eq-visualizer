import pygame
import random
import time
import math

pygame.init()
screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
info = pygame.display.Info()
zoom_out_scale = 1.0  # 1:1 rendering scale
pygame.display.set_caption("Text Overlay Example")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 36)

frame_counter = 0
NUM_CREATURES = 10

# === Entropic Dot Creatures ===
creatures = []
leader_creature = {
    'pos': [random.randint(100, 400), random.randint(100, 400)],
    'dir': [random.uniform(-1.5, 1.5), random.uniform(-1.5, 1.5)],
    'color': (255, 255, 255)
}
creatures.append(leader_creature)
for _ in range(NUM_CREATURES):
    base_x, base_y = leader_creature['pos']
    creatures.append({
        'pos': [base_x + random.randint(-30, 30), base_y + random.randint(-30, 30)],
        'dir': [random.uniform(-1.5, 1.5), random.uniform(-1.5, 1.5)],
        'color': (random.randint(50, 254), random.randint(50, 254), random.randint(50, 254))
    })

# === Additional Groups ===
additional_groups = []
speed_multipliers = [0.95, 0.90]
for group_id in range(2):
    group = []
    leader = {
        'pos': [random.randint(300, info.current_w - 300), random.randint(100, info.current_h - 100)],
        'dir': [random.uniform(-1.5, 1.5), random.uniform(-1.5, 1.5)],
        'color': (255, 255, 0) if group_id == 0 else (0, 255, 255)
    }
    group.append(leader)
    for _ in range(NUM_CREATURES):
        base_x, base_y = leader['pos']
        group.append({
            'pos': [base_x + random.randint(-30, 30), base_y + random.randint(-30, 30)],
            'dir': [random.uniform(-1.2, 1.2), random.uniform(-1.2, 1.2)],
            'color': (random.randint(50, 254), random.randint(50, 254), random.randint(50, 254))
        })
    additional_groups.append(group)

# === Extra 2 Swarms ===
for _ in range(2):
    group = []
    leader = {
        'pos': [random.randint(100, info.current_w - 100), random.randint(100, info.current_h - 100)],
        'dir': [random.uniform(-1.3, 1.3), random.uniform(-1.3, 1.3)],
        'color': (random.randint(50, 254), random.randint(50, 254), random.randint(50, 254))
    }
    group.append(leader)
    for _ in range(NUM_CREATURES):
        base_x, base_y = leader['pos']
        group.append({
            'pos': [base_x + random.randint(-30, 30), base_y + random.randint(-30, 30)],
            'dir': [random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0)],
            'color': (random.randint(50, 254), random.randint(50, 254), random.randint(50, 254))
        })
    additional_groups.append(group)

# === Alpha Leader (Super Leader) ===
class MysteriousAlpha:
    def __init__(self):
        self.pos = [random.randint(500, 800), random.randint(300, 600)]
        self.dir = [random.uniform(-2.5, 2.5), random.uniform(-2.5, 2.5)]
        self.color = (255, 0, 255)

    def update(self, screen_center, time_since_return):
        quadrant_targets = [
            (info.current_w * 0.25, info.current_h * 0.25),
            (info.current_w * 0.75, info.current_h * 0.25),
            (info.current_w * 0.75, info.current_h * 0.75),
            (info.current_w * 0.25, info.current_h * 0.75)
        ]
        cycle = int(time.time() / 5) % 4
        target = quadrant_targets[cycle]
        to_target = [target[0] - self.pos[0], target[1] - self.pos[1]]
        to_center = [screen_center[0] - self.pos[0], screen_center[1] - self.pos[1]]
        dist = max(1, math.hypot(*to_target))
        cent_dist = max(1, math.hypot(*to_center))
        to_target[0] /= dist
        to_target[1] /= dist
        to_center[0] /= cent_dist
        to_center[1] /= cent_dist
        # Blend toward center more heavily
        self.dir[0] += (to_target[0] * 0.3 + to_center[0] * 0.7) + random.uniform(-0.5, 0.5)
        self.dir[1] += (to_target[1] * 0.3 + to_center[1] * 0.7) + random.uniform(-0.5, 0.5)
        self.pos[0] += self.dir[0]
        self.pos[1] += self.dir[1]
        self.pos[0] = max(0, min(info.current_w, self.pos[0]))
        self.pos[1] = max(0, min(info.current_h, self.pos[1]))

alpha_leader = MysteriousAlpha()

running = True
alpha_return_timer = time.time()

while running:
    screen.fill((0, 0, 0))

    screen_center = (info.current_w / 2, info.current_h / 2)
    time_since_return = time.time() - alpha_return_timer
    alpha_leader.update(screen_center, time_since_return)

    blink_alpha = (int(time.time() * 4) % 2) == 0
    alpha_pos = (int(alpha_leader.pos[0]), int(alpha_leader.pos[1]))
    radius = 24 if blink_alpha else 32
    pygame.draw.circle(screen, alpha_leader.color, alpha_pos, radius)

    width, height = screen.get_size()

    # === All Groups Including Additional ===
    all_groups = [creatures] + additional_groups
    for group_index, group in enumerate(all_groups):
        speed_factor = 1.0 - group_index * 0.05
        for i, creature in enumerate(group):
            if i == 0:
                to_alpha = [alpha_leader.pos[0] - creature['pos'][0], alpha_leader.pos[1] - creature['pos'][1]]
                dist = max(1, math.hypot(*to_alpha))
                to_alpha[0] /= dist
                to_alpha[1] /= dist
                creature['dir'][0] += to_alpha[0] * 0.4 + random.uniform(-0.6, 0.6)
                creature['dir'][1] += to_alpha[1] * 0.4 + random.uniform(-0.6, 0.6)
                creature['pos'][0] += creature['dir'][0] * speed_factor
                creature['pos'][1] += creature['dir'][1] * speed_factor
            else:
                dx = (group[0]['pos'][0] - creature['pos'][0]) * 0.002
                dy = (group[0]['pos'][1] - creature['pos'][1]) * 0.002
                creature['dir'][0] += dx + random.uniform(-0.4, 0.4)
                creature['dir'][1] += dy + random.uniform(-0.4, 0.4)
                creature['pos'][0] += creature['dir'][0] * (speed_factor - 0.05)
                creature['pos'][1] += creature['dir'][1] * (speed_factor - 0.05)

            creature['pos'][0] = max(0, min(width, creature['pos'][0]))
            creature['pos'][1] = max(0, min(height, creature['pos'][1]))
            pygame.draw.circle(screen, creature['color'], (int(creature['pos'][0]), int(creature['pos'][1])), 6)

    pygame.display.flip()
    clock.tick(60)

    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            running = False
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_F11:
            screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)

pygame.quit()
