import pygame
import random
import math

# === ENTROPIC VISUALIZER — Worm-in-a-Terrarium Smooth Panning Camera ===
pygame.init()
screen = pygame.display.set_mode((800, 600), pygame.RESIZABLE)
pygame.display.set_caption("Entropic Dice Visualizer")
clock = pygame.time.Clock()

# Dice direction logic
DIRECTIONS = {
    'x': [(1, 0, 0), (-1, 0, 0)],
    'y': [(0, 1, 0), (0, -1, 0)],
    'z': [(0, 0, 1), (0, 0, -1)]
}

worm = [(0, 0, 0)]
colors = [(255, 255, 255)]
MAX_LENGTH = 60

roll_history = []
MAX_HISTORY = 5
STREAK_CHANCE = 0.6
STREAK_MAX = 3
current_streak = 0
last_direction = None

camera_focus = [0.0, 0.0, 0.0]  # Smoothed camera target
camera_target = [0.0, 0.0, 0.0]  # Target position to move toward
camera_angle = 0.0  # Dynamic panning angle
camera_radius = 12.0


def roll_entropic_dice():
    global roll_history, current_streak, last_direction
    if last_direction and current_streak < STREAK_MAX and random.random() < STREAK_CHANCE:
        current_streak += 1
        roll_history.append((last_direction[0], last_direction[1]))
        roll_history = roll_history[-MAX_HISTORY:]
        return last_direction[1]

    current_streak = 0
    available_axes = list(DIRECTIONS.keys())
    if len(roll_history) >= 2:
        last_axis = roll_history[-1][0]
        if last_axis in available_axes:
            available_axes.remove(last_axis)
    chosen_axis = random.choice(available_axes)
    direction = random.choice(DIRECTIONS[chosen_axis])
    roll_history.append((chosen_axis, direction))
    roll_history = roll_history[-MAX_HISTORY:]
    last_direction = (chosen_axis, direction)
    return direction


def move_worm():
    dx, dy, dz = roll_entropic_dice()
    x, y, z = worm[-1]
    new_pos = (x + dx, y + dy, z + dz)
    worm.append(new_pos)
    color = (100 + z * 30 % 155, 100 + y * 40 % 155, 200 - x * 10 % 100)
    colors.append(color)
    if len(worm) > MAX_LENGTH:
        worm.pop(0)
        colors.pop(0)

    # Update target camera position
    camera_target[0] = new_pos[0]
    camera_target[1] = new_pos[1]
    camera_target[2] = new_pos[2]


def project(point):
    # Smooth camera tracking with +/-20% slack
    global camera_focus, camera_angle
    camera_angle += 0.0001  # Slow smooth pan

    # Interpolate camera focus toward camera_target
    for i in range(3):
        camera_focus[i] += (camera_target[i] - camera_focus[i]) * 0.001

    # Camera orbit motion
    orbit_x = math.cos(camera_angle) * camera_radius
    orbit_y = math.sin(camera_angle) * camera_radius
    orbit_z = 10

    rel_x = point[0] - camera_focus[0] - orbit_x
    rel_y = point[1] - camera_focus[1] - orbit_y
    rel_z = point[2] - camera_focus[2] + orbit_z
    if rel_z < 1:
        rel_z = 1
    scale = 256 / rel_z  # Adjusted to keep worm within ±7% frame slack
    x = int(rel_x * scale + 400)
    y = int(rel_y * scale + 300)
    return x, y


def draw():
    screen.fill((0, 0, 0))
    for i in range(len(worm) - 1):
        start = project(worm[i])
        end = project(worm[i + 1])
        pygame.draw.line(screen, colors[i], start, end, 3)
    pygame.display.flip()


fullscreen = False

# Main loop
running = True
while running:
    move_worm()
    draw()
    clock.tick(60)

    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            running = False
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_F11:
            fullscreen = not fullscreen
            if fullscreen:
                screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
            else:
                screen = pygame.display.set_mode((800, 600), pygame.RESIZABLE)

pygame.quit()
