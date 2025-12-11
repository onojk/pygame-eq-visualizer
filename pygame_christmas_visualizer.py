#!/usr/bin/env python3
import numpy as np
import pygame
import os
import sys
import glob
import math
from scipy.fftpack import fft

# ========================= CONFIG =========================
WAV_FOLDER   = "/home/onojk123/Downloads/MeganChristmas/old"
OUTPUT_VIDEO = "/home/onojk123/MEGAN_CHRISTMAS_ALBUM_2025.mp4"
WIDTH, HEIGHT = 1280, 720
FPS = 24
RATE = 22050
FRAME_DIR = "/tmp/christmas_frames"
# =========================================================

audio_files = sorted(glob.glob(f"{WAV_FOLDER}/*.wav"))
if len(audio_files) == 0:
    print("No .wav files found in the folder!")
    sys.exit(1)

print(f"Found {len(audio_files)} Christmas songs – preparing the magic...")

pygame.mixer.pre_init(frequency=RATE, size=-16, channels=2, buffer=512)
pygame.init()

# Load all songs into one giant mono stream
full_audio = np.array([], dtype=np.int16)
for i, path in enumerate(audio_files, 1):
    name = os.path.basename(path).replace(".wav", "")
    print(f"  {i}/{len(audio_files)} → {name}")
    snd = pygame.mixer.Sound(path)
    samples = pygame.sndarray.samples(snd)
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    full_audio = np.concatenate((full_audio, samples.astype(np.int16)))

print(f"\nTotal length ≈ {len(full_audio)/RATE/60:.1f} minutes of Christmas joy\n")

os.makedirs(FRAME_DIR, exist_ok=True)
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Megan’s Christmas 2025 – Rendering...")

# Christmas color palette – warm lights, snow, gold, deep green & ruby red
CHRISTMAS_COLORS = [
    (255, 240, 200),  # warm white
    (255, 200, 100),  # golden glow
    (220, 20, 60),    # crimson red
    (0, 100, 80),     # deep pine green
    (255, 180, 180),  # soft pink
    (200, 255, 255),  # icy blue-white
]

def draw_snowflake(surface, cx, cy, size, alpha=180):
    points = []
    for i in range(0, 360, 60):
        x1 = cx + size * math.cos(math.radians(i))
        y1 = cy + size * math.sin(math.radians(i))
        x2 = cx + (size * 1.4) * math.cos(math.radians(i + 30))
        y2 = cy + (size * 1.4) * math.sin(math.radians(i + 30))
        points.extend([(cx, cy), (x1, y1), (x2, y2)])
    pygame.draw.lines(surface, (255, 255, 255, alpha), False, points, 2)

def draw_christmas_visualizer(surface, cx, cy, radius, bars, t):
    # Deep night sky with a hint of snow
    surface.fill((10, 15, 40))

    # Falling snow (very light)
    for i in range(80):
        x = (i * 73 + int(t * 20)) % WIDTH
        y = (i * 97 + int(t * 30)) % (HEIGHT + 50) - 50
        pygame.draw.circle(surface, (255, 255, 255, 60), (x, y), 1)

    # Warm pulsing Christmas tree star + glow
    star_glow = 100 + 80 * np.mean(bars[:10])
    for r in [60, 90, 120]:
        alpha = int(60 + 40 * math.sin(t * 3))
        temp = pygame.Surface((r * 4, r * 4), pygame.SRCALPHA)
        pygame.draw.circle(temp, (*CHRISTMAS_COLORS[1], alpha), (r * 2, r * 2), r)
        surface.blit(temp, (cx - r * 2, cy - HEIGHT // 2 + 60))

    # Rotating golden ornaments + colored lights (12-fold like a clock)
    for seg in range(12):
        angle = seg * 30 + t * 8
        part = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)

        # String lights
        for x in range(30, radius, 20):
            intensity = bars[x % len(bars)]
            col_idx = int(x / 50 + t * 2) % len(CHRISTMAS_COLORS)
            col = CHRISTMAS_COLORS[col_idx]
            size = 8 + int(20 * intensity)
            alpha = int(120 + 135 * intensity)
            px = x
            py = radius // 2 + 60 * math.sin(t * 0.5 + x * 0.03 + seg)
            temp = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
            pygame.draw.circle(temp, (*col, alpha), (size, size), size)
            part.blit(temp, (int(px - size), int(py - size)))

        # Tiny snowflake at the tips
        tip_x = radius - 20
        tip_y = radius // 2
        draw_snowflake(part, int(tip_x), int(tip_y), 15, 150)

        rotated = pygame.transform.rotate(part, angle)
        rect = rotated.get_rect(center=(cx, cy))
        surface.blit(rotated, rect)

# ========================= RENDER =========================
samples_per_frame = RATE // FPS
total_frames = len(full_audio) // samples_per_frame + 10

print(f"Rendering {total_frames:,} magical Christmas frames…\n")

for frame_idx in range(total_frames):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()

    start = frame_idx * samples_per_frame
    chunk = full_audio[start:start + samples_per_frame]
    if len(chunk) < samples_per_frame:
        chunk = np.pad(chunk, (0, samples_per_frame - len(chunk)))

    fft_vals = np.abs(fft(chunk))[:512]
    bars = fft_vals / (fft_vals.max() + 1e-8) if fft_vals.size > 0 else np.zeros(512)

    t = pygame.time.get_ticks() * 0.001

    draw_christmas_visualizer(
        screen,
        WIDTH // 2,
        HEIGHT // 2,
        min(WIDTH, HEIGHT) // 2 - 40,
        bars,
        t
    )

    pygame.display.flip()
    pygame.image.save(screen, f"{FRAME_DIR}/frame_{frame_idx:08d}.png")

    if frame_idx % 80 == 0:
        print(f"\rFrame {frame_idx+1}/{total_frames} – Merry Christmas!", end="", flush=True)

# ========================= FFMPEG =========================
print("\n\nWrapping your Christmas gift with ffmpeg…")
with open("/tmp/xmas_list.txt", "w") as f:
    for w in audio_files:
        f.write(f"file '{w}'\n")

os.system("ffmpeg -y -f concat -safe 0 -i /tmp/xmas_list.txt -c:a aac -b:a 320k /tmp/xmas_audio.aac")
os.system(f'''
ffmpeg -y -r {FPS} -i {FRAME_DIR}/frame_%08d.png -i /tmp/xmas_audio.aac \
       -c:v libx264 -preset veryslow -crf 14 -pix_fmt yuv420p -c:a copy -shortest \
       "{OUTPUT_VIDEO}"
''')

os.system("rm -rf /tmp/christmas_frames /tmp/xmas_list.txt /tmp/xmas_audio.aac")
print(f"\nMERRY CHRISTMAS! Your beautiful video is ready!")
print(f"   {OUTPUT_VIDEO}")
pygame.quit()
