#!/usr/bin/env python3
import numpy as np
import pygame
import os
import sys
import glob
from scipy.fftpack import fft
import math
import random

# ============== CONFIGURATION ==============
WAV_FOLDER = "/home/onojk123/Downloads/truth2_45884854"
OUTPUT_VIDEO = "/home/onojk123/TRUTH_FULL_ALBUM_kaleidoscope.mp4"
WIDTH, HEIGHT = 1280, 720
FPS = 24
# ===========================================

# Find all tracks
audio_files = sorted(glob.glob(f"{WAV_FOLDER}/Number*.wav"))
if len(audio_files) == 0:
    print("No Number*.wav files found in folder!")
    sys.exit(1)

print(f"Found {len(audio_files)} tracks. Loading audio...")

# Initialize pygame mixer properly
pygame.mixer.pre_init(frequency=22050, size=-16, channels=2, buffer=512)
pygame.init()

# Load and concatenate ALL audio into one giant mono numpy array
full_audio = np.array([], dtype=np.int16)

for i, wav in enumerate(audio_files):
    print(f"  Loading {i+1}/{len(audio_files)}: {os.path.basename(wav)}")
    sound = pygame.mixer.Sound(wav)
    samples = pygame.sndarray.samples(sound)        # This returns numpy array
    # Convert stereo → mono if needed
    if samples.ndim == 2:
        samples = samples.mean(axis=1).astype(np.int16)
    else:
        samples = samples.astype(np.int16)
    full_audio = np.concatenate((full_audio, samples))

print(f"Total audio length: {len(full_audio)/22050/60:.1f} minutes")

# Constants
CHUNK = 512
NUM_SLICES = 16
SLICE_ANGLE = 360 / NUM_SLICES
show_black_lines = True
FRAME_DIR = "/tmp/kaleidoscope_frames"
os.makedirs(FRAME_DIR, exist_ok=True)

screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

# Color sets
COLOR_SETS = [
    [(0,123,255), (255,0,255), (0,255,0), (255,255,0)],
    [(226,114,91), (128,128,0), (92,64,51), (244,164,96)],
    [(152,255,152), (230,230,250), (179,229,252), (255,218,185)],
    [(192,192,192), (255,215,0), (205,127,50), (47,79,79)],
]

def generate_simple_lines(surf, w, h, audio_bars, color_set, idx):
    cx = w // 2
    cy = h // 2
    for i in range(min(1, len(audio_bars))):
        amp = audio_bars[i] * h * 0.2
        freq = 0.01 * (i + 1)
        phase = pygame.time.get_ticks() * 0.001
        color = color_set[(idx + i) % len(color_set)]
        points = []
        step = 60
        for x in range(0, w//2, step):
            y_offset = amp * math.sin(freq * x + phase)
            y = cy + y_offset
            points.append((cx + x, int(y)))
        mirrored = [(cx - (x - cx), y) for x, y in points[::-1]]
        pygame.draw.lines(surf, color, False, points + mirrored, 2)

def draw_broken_black_wave(surf, w, h, idx):
    if not show_black_lines: return
    cx, cy = w//2, h//2
    amp = h * 0.15
    freq = 0.03 * (idx + 1)
    phase = pygame.time.get_ticks() * 0.001
    points = []
    for x in range(0, int(w * 0.8), 60):
        y = cy + amp * math.sin(freq * x + phase)
        points.append((cx + x, int(y)))
    mirrored = [(cx - (x - cx), y) for x, y in points[::-1]]
    pygame.draw.lines(surf, (0,0,0), False, points + mirrored, random.randint(2,4))

def draw_kaleidoscope(surface, center, radius, audio_bars):
    color_set = COLOR_SETS[int(pygame.time.get_ticks() / 1000) % len(COLOR_SETS)]
    for i in range(NUM_SLICES):
        s = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
        generate_simple_lines(s, radius*2, radius*2, audio_bars, color_set, i)
        draw_broken_black_wave(s, radius*2, radius*2, i)
        rotated = pygame.transform.rotate(s, i * SLICE_ANGLE)
        rect = rotated.get_rect(center=center)
        surface.blit(rotated, rect)

# ============== RENDER ALL FRAMES ==============
RATE = 22050
samples_per_frame = RATE // FPS
total_frames = len(full_audio) // samples_per_frame + 1

print(f"Starting render of {total_frames:,} frames... (this will take 15-40 minutes)")

frame_idx = 0
while frame_idx < total_frames:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            print("\nCancelled by user")
            sys.exit()

    start = frame_idx * samples_per_frame
    end = start + samples_per_frame
    chunk = full_audio[start:end]

    if len(chunk) < samples_per_frame:
        chunk = np.pad(chunk, (0, samples_per_frame - len(chunk)))

    # FFT
    fft_vals = np.abs(fft(chunk))[:CHUNK//2]
    if len(fft_vals) > 0 and fft_vals.max() > 0:
        audio_bars = fft_vals / fft_vals.max()
    else:
        audio_bars = np.zeros(CHUNK//2)

    screen.fill((0, 0, 0))
    draw_kaleidoscope(screen, (WIDTH//2, HEIGHT//2), min(WIDTH, HEIGHT)//2 - 50, audio_bars)
    pygame.display.flip()

    frame_path = f"{FRAME_DIR}/frame_{frame_idx:08d}.png"
    pygame.image.save(screen, frame_path)

    frame_idx += 1
    if frame_idx % 50 == 0:
        mins_left = (total_frames - frame_idx) / FPS / 60
        print(f"\rFrame {frame_idx}/{total_frames} | ~{mins_left:.1f} min left", end="")

print(f"\nRendering complete! Encoding video...")

# ============== FFMPEG: Concat audio + video ==============
print("Concatenating original audio tracks...")
with open("audio_list.txt", "w") as f:
    for wav in audio_files:
        f.write(f"file '{wav}'\n")

os.system('ffmpeg -y -f concat -safe 0 -i audio_list.txt -c:a aac -b:a 320k /tmp/album_audio.aac')

print("Final encoding...")
os.system(f'''
ffmpeg -y -r {FPS} -i {FRAME_DIR}/frame_%08d.png \
       -i /tmp/album_audio.aac \
       -c:v libx264 -preset medium -crf 17 -pix_fmt yuv420p \
       -c:a copy -shortest "{OUTPUT_VIDEO}"
''')

# Cleanup
os.system("rm -rf /tmp/kaleidoscope_frames audio_list.txt /tmp/album_audio.aac")
print(f"\nYOUR FULL ALBUM VIDEO IS READY!")
print(f"→ {OUTPUT_VIDEO}")
pygame.quit()
