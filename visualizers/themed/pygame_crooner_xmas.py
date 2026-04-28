#!/usr/bin/env python3
import numpy as np
import pygame
import os
import sys
import glob
import math
from scipy.fftpack import fft

# ========================= CONFIG =========================
WAV_FOLDER   = "/home/onojk123/Downloads/xmas48888888/old"
OUTPUT_VIDEO = "/home/onojk123/CHRISTMAS_CROONER_MASTERPIECE_2025.mp4"
WIDTH, HEIGHT = 1280, 720
FPS = 24
RATE = 22050
FRAME_DIR = "/tmp/crooner_frames"
# =========================================================

# Collect WAVs (skip stray verse file)
audio_files = sorted(glob.glob(f"{WAV_FOLDER}/*.wav"))
audio_files = [f for f in audio_files if "Verse 1.wav" not in f]

print(f"Found {len(audio_files)} crooner gems")

if not audio_files:
    print("No WAV files found – exiting.")
    sys.exit(1)

pygame.mixer.pre_init(frequency=RATE, size=-16, channels=2, buffer=512)
pygame.init()

# Load all songs into one giant mono stream
full_audio = np.array([], dtype=np.int16)
for i, path in enumerate(audio_files, 1):
    name = os.path.basename(path)
    print(f"  {i:2d}/{len(audio_files)}  🎄 {name}")
    snd = pygame.mixer.Sound(path)
    arr = pygame.sndarray.samples(snd)
    if arr.ndim == 2:
        arr = arr.mean(axis=1)
    full_audio = np.concatenate((full_audio, arr.astype(np.int16)))

print(f"\nTotal ≈ {len(full_audio) / RATE / 60:.1f} minutes of pure velvet Christmas\n")

os.makedirs(FRAME_DIR, exist_ok=True)
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Crooner Christmas 2025 – Rendering...")

# Crooner palette – cigarette smoke, whisky, candlelight
C = [
    (220, 180, 100),  # warm amber
    (180, 120, 60),   # whisky
    (200, 160, 120),  # vintage paper
    (255, 220, 180),  # soft spotlight
    (100, 140, 180),  # midnight blue
    (255, 240, 220),  # champagne
]


def circle_alpha(surf, color, center, radius, alpha=255):
    """Draw a soft, alpha-blended circle safely (clamped 0–255)."""
    radius = int(radius)
    if radius < 1:
        return

    # Clamp alpha and unpack color
    a = max(0, min(255, int(alpha)))
    r, g, b = color

    temp = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
    pygame.draw.circle(temp, (r, g, b, a), (radius, radius), radius)
    surf.blit(temp, (int(center[0] - radius), int(center[1] - radius)))


def draw_crooner_scene(surface, cx, cy, radius, bars, t):
    # Deep smoky lounge background
    surface.fill((15, 10, 25))

    # Slow smoke curls
    for i in range(15):
        x = cx + 300 * math.sin(t * 0.3 + i)
        y = cy - 200 + i * 40 + 80 * math.sin(t * 0.4 + i * 1.7)
        alpha = 15 + 10 * math.sin(t + i)
        circle_alpha(surface, (200, 200, 220), (int(x), int(y)), 60 + i * 8, alpha)

    # Vintage microphone in the center (classic 1950s stand mic)
    mic_color = (180, 180, 190)
    pygame.draw.circle(surface, mic_color, (cx, cy - 80), 45)         # head
    pygame.draw.rect(surface, mic_color, (cx - 8, cy - 80, 16, 180))  # stand
    pygame.draw.circle(surface, (80, 80, 80), (cx, cy + 100), 60)     # base

    # Warm spotlight glow that breathes with the voice
    voice = float(np.mean(bars[5:40])) if bars.size else 0.0  # crooner mids
    for r in range(100, 420, 40):
        alpha = 25 + 45 * voice + 30 * math.sin(t * 2 + r / 100)
        circle_alpha(surface, C[3], (cx, cy), r, alpha)

    # Champagne bubbles rising slowly
    for i in range(40):
        angle = i * 9
        dist = (t * 15 + i * 37) % 500
        x = cx + dist * math.cos(math.radians(angle))
        y = cy + 200 - dist * 1.8
        if 0 < y < HEIGHT:
            sz = 3 + int(8 * voice)
            circle_alpha(surface, C[5], (int(x), int(y)), sz, 180)

    # Velvet snowflakes – big, slow, luxurious
    for i in range(18):
        angle = i * 20 + t * 5
        dist = radius * 0.9
        x = cx + dist * math.cos(math.radians(angle))
        y = cy + dist * math.sin(math.radians(angle))
        rot = t * 10 + i * 20
        size = 20 + 15 * voice

        points = []
        for j in range(6):
            a = math.radians(rot + j * 60)
            points.append((x + size * math.cos(a), y + size * math.sin(a)))
            a2 = math.radians(rot + j * 60 + 30)
            points.append((x + size * 0.5 * math.cos(a2), y + size * 0.5 * math.sin(a2)))

        # alpha in color tuple is ignored on display surface, but fine
        pygame.draw.lines(surface, (255, 255, 255, 140), True, points, 3)

    # Title in classic cursive style (appears gently)
    font = pygame.font.SysFont("times new roman", 48, bold=True)
    text = font.render("Christmas Crooners 2025", True, C[0])
    alpha = int(80 + 40 * math.sin(t * 1.5))
    alpha = max(0, min(255, alpha))
    text.set_alpha(alpha)
    surface.blit(text, (cx - text.get_width() // 2, 80))


# ========================= RENDER LOOP =========================
samples_per_frame = RATE // FPS
total_frames = len(full_audio) // samples_per_frame + 10

print(f"Rendering {total_frames:,} velvety frames…\n")

for frame_idx in range(total_frames):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()

    start = frame_idx * samples_per_frame
    chunk = full_audio[start:start + samples_per_frame]
    if len(chunk) < samples_per_frame:
        chunk = np.pad(chunk, (0, samples_per_frame - len(chunk)))

    fft_vals = np.abs(fft(chunk))[:512]
    if fft_vals.size > 0:
        bars = fft_vals / (fft_vals.max() + 1e-8)
    else:
        bars = np.zeros(512)

    t = pygame.time.get_ticks() * 0.001

    draw_crooner_scene(
        screen,
        WIDTH // 2,
        HEIGHT // 2,
        min(WIDTH, HEIGHT) // 2 - 50,
        bars,
        t
    )

    pygame.display.flip()
    pygame.image.save(screen, f"{FRAME_DIR}/frame_{frame_idx:08d}.png")

    if frame_idx % 80 == 0:
        print(f"\rFrame {frame_idx+1}/{total_frames} – smooth as Sinatra", end="", flush=True)

# ========================= FFMPEG =========================
print("\n\nPouring the final whisky… encoding video")

with open("/tmp/crooner_list.txt", "w") as f:
    for w in audio_files:
        f.write(f"file '{w}'\n")

os.system(
    "ffmpeg -y -f concat -safe 0 -i /tmp/crooner_list.txt "
    "-c:a aac -b:a 320k /tmp/crooner_audio.aac"
)

os.system(f'''
ffmpeg -y -r {FPS} -i {FRAME_DIR}/frame_%08d.png -i /tmp/crooner_audio.aac \
       -c:v libx264 -preset veryslow -crf 14 -pix_fmt yuv420p -c:a copy -shortest \
       "{OUTPUT_VIDEO}"
''')

os.system("rm -rf /tmp/crooner_frames /tmp/crooner_list.txt /tmp/crooner_audio.aac")
print(f"\nYour crooner Christmas masterpiece is ready!")
print(f"   {OUTPUT_VIDEO}")
pygame.quit()
