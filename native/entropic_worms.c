import pygame
import random
import math
import colorsys
import time
import os
import psutil
import sys

pygame.init()
info = pygame.display.Info()
screen = pygame.display.set_mode((0, 0), pygame.RESIZABLE)
pygame.display.set_caption("Entropic Worm Swarm")
pygame.key.set_repeat(10, 10)
clock = pygame.time.Clock()

NUM_WORMS = 48  # doubled: two flocks, now with 24 each
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

def flock_steer(index, last_vec):
    team = index % 2
    pos = worms[index][-1]
    avg_pos = [0.0, 0.0, 0.0]
    count = 0

    # Influence from all other worms
    for i, other in enumerate(worms):
        if i == index:
            continue
        op = other[-1]
        dist = math.sqrt(sum((pos[j] - op[j])**2 for j in range(3)))
        if 2 < dist < 10:  # consider near neighbors
            for j in range(3):
                avg_pos[j] += op[j]
            count += 1

    if count > 0:
        for j in range(3):
            avg_pos[j] /= count
        steer_vec = [avg_pos[j] - pos[j] for j in range(3)]
        norm = math.sqrt(sum(v**2 for v in steer_vec)) or 1
        steer_vec = [v / norm for v in steer_vec]
    else:
        steer_vec = list(last_vec)

    # Entropic flock influence
    flock_bias = jitter_vector(random.choice(ANCHOR_DIRECTIONS), amount=0.03)
    # Mild entropic individual influence
    weak_jitter = jitter_vector(last_vec, amount=0.01)

    # Weighted combination
    blended = [(steer_vec[i] * 0.7 + flock_bias[i] * 0.2 + weak_jitter[i] * 0.1) for i in range(3)]
    norm = math.sqrt(sum(b**2 for b in blended)) or 1
    return tuple(b / norm for b in blended)

def build_weighted_die(partner_vec=None, last_vec=None, flee=False):
    return []  # no longer used  # unused with new steering model

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
    width, height = screen.get_size()
    return int(x1 * scale + width // 2), int(y2 * scale + height // 2)

def draw_head_emojis():
    return  # removed emoji rendering for simplicity(current_time + blink_offset) % (blink_cycle_duration * 2)
def roll_entropic_dice(state, idx):
    last_vec = state.get('last_vec', (0, 0, 1))
    direction = flock_steer(idx, last_vec)
    state['last_vec'] = direction
    return direction

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
    width, height = screen.get_size()
    screen.fill((0, 0, 0))  # static black background
    base_hue = (time.time() * 0.0025 + hue_shift) % 1.0

    for w_index, worm in enumerate(worms):
        for i in range(len(worm) - 1):
            start = project(worm[i], offsets[w_index])
            end = project(worm[i + 1], offsets[w_index])
            segment_ratio = i / MAX_LENGTH
            hue = get_hue(segment_ratio)
