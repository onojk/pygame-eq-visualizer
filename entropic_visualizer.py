import pygame
import random
import math
import colorsys
import time
import os
import psutil
import sys

pygame.init()
screen = pygame.display.set_mode((800, 600), pygame.RESIZABLE)
pygame.display.set_caption("Entropic Worm Swarm")
pygame.key.set_repeat(10, 10)
clock = pygame.time.Clock()

NUM_WORMS = 12
MAX_LENGTH = 30  # even shorter tail for proportional body
INITIAL_COLLISION_DELAY = 60
COOLDOWN_AFTER_COLLISION = 120
THICKNESS_MIN = 10
STEP_SCALE_BASE = 0.05  # much smaller per-segment distance
MOMENTUM_MULTIPLIER = 3.0
DIE_RESOLUTION = 48  # more angles for smoother motion
BIAS_RESOLUTION = 240
MOMENTUM_BIAS_COUNT = 160
MAX_MEMORY_MB = 100

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

def get_memory_usage_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def jitter_vector(vec, amount=0.05):
    jittered = [v + random.uniform(-amount, amount) for v in vec]
    norm = math.sqrt(sum(v**2 for v in jittered)) or 1
    return tuple(v / norm for v in jittered)

def build_weighted_die(partner_vec=None, last_vec=None, flee=False):
    return []  # unused with new steering model

def random_start():
    scale = int(MAX_LENGTH * 0.35)
    return tuple(random.randint(-scale, scale) for _ in range(3))

def shuffle_new_partners(prev_partners):
    worms = list(range(NUM_WORMS))
    new_pairs = {}
    random.shuffle(worms)
    while worms:
        a = worms.pop()
        if not worms:
            new_pairs[a] = a
            break
        b = worms.pop()
        if prev_partners.get(a) != b and prev_partners.get(b) != a:
            new_pairs[a] = b
            new_pairs[b] = a
        else:
            worms.append(b)
            worms.insert(0, a)
    return new_pairs

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
    scale = zoom / max(z2, 1)
    return int(x1 * scale + 400), int(y2 * scale + 300)

def draw_head_emojis():
    return  # removed emoji rendering for simplicity(current_time + blink_offset) % (blink_cycle_duration * 2)
def check_for_collisions(idx, new_pos):
    if global_frame_counter < INITIAL_COLLISION_DELAY:
        return False
    for j, other_worm in enumerate(worms):
        if j == idx:
            continue
        if new_pos in other_worm[-10:]:
            return True
    return False

def draw():
    global hue_value
    hue_value = ((time.time() * 12) % 360) / 360.0  # 12 degrees per second cycling hue
    hue_shift = hue_value / 360.0
    screen.fill((0, 0, 0))  # static black background
    base_hue = (time.time() * 0.0025 + hue_shift) % 1.0

    for w_index, worm in enumerate(worms):
        for i in range(len(worm) - 1):
            start = project(worm[i], offsets[w_index])
            end = project(worm[i + 1], offsets[w_index])
            segment_ratio = i / MAX_LENGTH
            hue = get_hue(segment_ratio)
