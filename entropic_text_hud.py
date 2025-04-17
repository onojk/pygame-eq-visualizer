import pygame
import random
import time
import math

pygame.init()
info = pygame.display.Info()
screen = pygame.display.set_mode((0, 0), pygame.RESIZABLE)
zoom_out_scale = 0.5  # 50% zoom out
pygame.display.set_caption("Text Overlay Example")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 36)

frame_counter = 0

# === Entropic Dot Creatures ===
NUM_CREATURES = 10
creatures = []
leader_creature = {
    'pos': [random.randint(100, 400), random.randint(100, 400)],
    'dir': [random.uniform(-1.5, 1.5), random.uniform(-1.5, 1.5)],
    'color': (255, 255, 255)  # white leader dot
}
creatures.append(leader_creature)
for _ in range(NUM_CREATURES):
    creatures.append({
        'pos': [random.randint(100, 400), random.randint(100, 400)],
        'dir': [random.uniform(-1.5, 1.5), random.uniform(-1.5, 1.5)],
        'color': (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
    })

# === Entropic Dot Creatures Group 2 ===
NUM_GROUP2 = 10
group2 = []
for _ in range(NUM_GROUP2):
    group2.append({
        'pos': [random.randint(800, 1200), random.randint(300, 600)],
        'dir': [random.uniform(-1.5, 1.5), random.uniform(-1.5, 1.5)],
        'color': (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
    })
for _ in range(NUM_CREATURES):
    creatures.append({
        'pos': [random.randint(100, 400), random.randint(100, 400)],
        'dir': [random.uniform(-1.5, 1.5), random.uniform(-1.5, 1.5)],
        'color': (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
    })
running = True

while running:
    screen.fill((0, 0, 0))  # black background
    width, height = screen.get_size()

    # Example live values
    hue_value = ((time.time() * 12) % 360)
    fps = int(clock.get_fps())

    # Draw dots based on dynamic values
    pygame.draw.circle(screen, (0, 0, 0), (100, 100), 5)  # fixed dot
    pygame.draw.circle(screen, (0, 0, 255), (int(width * hue_value / 360), 200), 8)  # hue-driven dot
    pygame.draw.circle(screen, (255, 0, 0), (int(width * (frame_counter % 100) / 100), 300), 10)  # frame-driven dot
    pygame.draw.circle(screen, (0, 200, 0), (int(width * fps / 100), 400), 6)

    # Draw entropic creatures (leading group)
    # === Update leader dot separately ===
    center_vector = [width / 2, height / 2]
    to_center = [center_vector[0] - leader_creature['pos'][0], center_vector[1] - leader_creature['pos'][1]]
    dist = max(1, (to_center[0]**2 + to_center[1]**2) ** 0.5)
    to_center[0] /= dist
    to_center[1] /= dist

    # Momentum + directed wandering
    random_angle = random.uniform(0, 2 * math.pi)
    periodic_bias = math.sin(time.time() * 0.5)  # Oscillating randomness
    wander_force = [math.cos(random_angle + periodic_bias) * 4.0, math.sin(random_angle + periodic_bias) * 4.0]

    boost = 0.01 if int(time.time()) % 5 == 0 else 0.001
    leader_creature['dir'][0] += wander_force[0] * 0.02 + to_center[0] * boost
    leader_creature['dir'][1] += wander_force[1] * 0.02 + to_center[1] * boost

    
    leader_creature['pos'][0] += leader_creature['dir'][0]
    leader_creature['pos'][1] += leader_creature['dir'][1]
    leader_creature['pos'][0] = max(width * -0.04, min(width * 1.04, leader_creature['pos'][0]))
    leader_creature['pos'][1] = max(height * -0.04, min(height * 1.04, leader_creature['pos'][1]))
        # Blinking effect for leader dot
    blink = (int(time.time() * 2) % 2) == 0  # toggles every 0.5 seconds
    if blink:
        pygame.draw.circle(screen, leader_creature['color'], (int(leader_creature['pos'][0] * zoom_out_scale), int(leader_creature['pos'][1] * zoom_out_scale)), 6)
        pygame.draw.circle(screen, leader_creature['color'], (int(leader_creature['pos'][0]), int(leader_creature['pos'][1])), 6)

    # === Update follower creatures ===
    for creature in creatures[1:]:
        # === Entropic Directional Dice Components ===
        center_pull = [(leader_creature['pos'][0] - creature['pos'][0]) * 0.0015,
                       (leader_creature['pos'][1] - creature['pos'][1]) * 0.0015]

        wander = [random.uniform(-1.0, 1.0) * 1.0 if random.random() < 0.1 else 0,
                  random.uniform(-1.0, 1.0) * 0.1 if random.random() < 0.1 else 0]

        bounce = [0, 0]
        if creature['pos'][0] <= 0 or creature['pos'][0] >= width:
            bounce[0] = -creature['dir'][0] * 0.5
        if creature['pos'][1] <= 0 or creature['pos'][1] >= height:
            bounce[1] = -creature['dir'][1] * 0.5

        # Composite entropic direction
        creature['dir'][0] += center_pull[0] + wander[0] + bounce[0]
        creature['dir'][1] += center_pull[1] + wander[1] + bounce[1]

        # Apply movement
        creature['pos'][0] += creature['dir'][0]
        creature['pos'][1] += creature['dir'][1]
        creature['pos'][0] = max(0, min(width, creature['pos'][0]))
        creature['pos'][1] = max(0, min(height, creature['pos'][1]))

        pygame.draw.circle(screen, creature['color'], (int(creature['pos'][0] * zoom_out_scale), int(creature['pos'][1] * zoom_out_scale)), 6)

        # Simple cohesion (center pull)
        creature['dir'][0] += (width/2 - creature['pos'][0]) * 0.0005
        creature['dir'][1] += (height/2 - creature['pos'][1]) * 0.0005

        pygame.draw.circle(screen, creature['color'], (int(creature['pos'][0]), int(creature['pos'][1])), 6)

    # Draw entropic group 2 (tracking group)
    # Find average position of creatures to follow
    if creatures:
        avg_x = sum(c['pos'][0] for c in creatures) / len(creatures)
        avg_y = sum(c['pos'][1] for c in creatures) / len(creatures)

    for dot in group2:
        # Entropic wobble
        if random.random() < 0.05:
            dot['dir'][0] += random.uniform(-0.2, 0.2)
            dot['dir'][1] += random.uniform(-0.2, 0.2)

        # Move slower (95%)
        dot['pos'][0] += dot['dir'][0] * 0.95
        dot['pos'][1] += dot['dir'][1] * 0.95

        # Track leading group's center
        dot['dir'][0] += (avg_x - dot['pos'][0]) * 0.0007
        dot['dir'][1] += (avg_y - dot['pos'][1]) * 0.0007

        dot['pos'][0] = max(0, min(width, dot['pos'][0]))
        dot['pos'][1] = max(0, min(height, dot['pos'][1]))

        pygame.draw.circle(screen, dot['color'], (int(dot['pos'][0] * zoom_out_scale), int(dot['pos'][1] * zoom_out_scale)), 6)  # fps-driven dot

    pygame.display.flip()
    frame_counter += 1

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False

    clock.tick(60)

pygame.quit()
