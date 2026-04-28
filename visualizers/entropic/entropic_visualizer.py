import pygame
import random
import math
import colorsys
import time
import os
import psutil
import sys

# === Initialization ===
pygame.init()
screen = pygame.display.set_mode((800, 600), pygame.RESIZABLE)
pygame.display.set_caption("Entropic Worm Swarm")
pygame.key.set_repeat(10, 10)
clock = pygame.time.Clock()

# === Constants ===
NUM_WORMS = 12
MAX_LENGTH = 30
INITIAL_COLLISION_DELAY = 60
COOLDOWN_AFTER_COLLISION = 120
THICKNESS_MIN = 10
STEP_SCALE_BASE = 0.05
MOMENTUM_MULTIPLIER = 3.0
ZOOM = 512

ANCHOR_DIRECTIONS = [
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1),
    (1, 1, 0), (-1, -1, 0),
    (-1, 1, 0), (1, -1, 0),
    (1, 0, 1), (-1, 0, -1),
    (0, 1, 1), (0, -1, -1),
    (1, 1, 1), (-1, -1, -1)
]

# === Utility Functions ===
def get_memory_usage_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def jitter_vector(vec, amount=0.05):
    jittered = [v + random.uniform(-amount, amount) for v in vec]
    norm = math.sqrt(sum(v**2 for v in jittered)) or 1
    return tuple(v / norm for v in jittered)

def random_start():
    scale = int(MAX_LENGTH * 0.35)
    return tuple(random.randint(-scale, scale) for _ in range(3))

def get_hue(t):
    r, g, b = colorsys.hsv_to_rgb((t + time.time() * 0.05) % 1.0, 1.0, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)

def project(point, offset=(0, 0, 0)):
    tracked_head = worms[camera_target_index][-1]
    for i in range(3):
        if auto_track:
            camera_focus[i] += (tracked_head[i] - camera_focus[i]) * 0.0001
    rel_x = point[0] + offset[0] - camera_focus[0] + camera_offset[0]
    rel_y = point[1] + offset[1] - camera_focus[1] + camera_offset[1]
    rel_z = point[2] + offset[2] - camera_focus[2] + 10
    yaw = math.radians(camera_pivot[0])
    pitch = math.radians(camera_pivot[1])
    x1 = rel_x * math.cos(yaw) - rel_y * math.sin(yaw)
    y1 = rel_x * math.sin(yaw) + rel_y * math.cos(yaw)
    z1 = rel_z
    y2 = y1 * math.cos(pitch) - z1 * math.sin(pitch)
    z2 = y1 * math.sin(pitch) + z1 * math.cos(pitch)
    scale = ZOOM / max(z2, 1)
    return int(x1 * scale + 400), int(y2 * scale + 300)

def draw():
    screen.fill((0, 0, 0))
    hue_value = ((time.time() * 12) % 360) / 360.0
    for w_index, worm in enumerate(worms):
        for i in range(len(worm) - 1):
            start = project(worm[i], offsets[w_index])
            end = project(worm[i + 1], offsets[w_index])
            segment_ratio = i / MAX_LENGTH
            color = get_hue(segment_ratio)
            thickness = int(THICKNESS_MIN * (1 - segment_ratio))
            pygame.draw.line(screen, color, start, end, thickness)

def main():
    global worms, offsets, camera_focus, camera_offset, camera_pivot, camera_target_index, auto_track, global_frame_counter

    worms = [[random_start()] for _ in range(NUM_WORMS)]
    offsets = [jitter_vector((0, 0, 0)) for _ in range(NUM_WORMS)]
    camera_focus = [0.0, 0.0, 0.0]
    camera_offset = [0.0, 0.0, 0.0]
    camera_pivot = [0.0, 0.0]
    camera_target_index = 0
    auto_track = True
    global_frame_counter = 0

    running = True
    while running:
        global_frame_counter += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False

        # Worm updates
        for idx, worm in enumerate(worms):
            last = worm[-1]
            jittered = jitter_vector(random.choice(ANCHOR_DIRECTIONS))
            new_pos = (
                last[0] + jittered[0] * STEP_SCALE_BASE * MOMENTUM_MULTIPLIER,
                last[1] + jittered[1] * STEP_SCALE_BASE * MOMENTUM_MULTIPLIER,
                last[2] + jittered[2] * STEP_SCALE_BASE * MOMENTUM_MULTIPLIER
            )
            worm.append(new_pos)
            if len(worm) > MAX_LENGTH:
                worm.pop(0)

        draw()
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n💥 Crash: {e}")
        pygame.quit()
        sys.exit(1)
