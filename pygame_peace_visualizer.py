#!/usr/bin/env python3
import numpy as np
import pygame
import os
import sys
import glob
from scipy.fftpack import fft
import math

# ========================= CONFIG =========================
WAV_FOLDER   = "/home/onojk123/Downloads/fullofpeace2342343243"
OUTPUT_VIDEO = "/home/onojk123/FULL_OF_PEACE_kaleidoscope.mp4"
WIDTH, HEIGHT = 1280, 720
FPS = 24
RATE = 22050  # must match mixer frequency
FRAME_DIR = "/tmp/peace_frames"
# =========================================================


# ========================= AUDIO LOAD =========================
audio_files = sorted(glob.glob(f"{WAV_FOLDER}/Number*.wav"))
if not audio_files:
    print("No tracks found in:", WAV_FOLDER)
    sys.exit(1)

print(f"Loading {len(audio_files)} peaceful tracks...")
pygame.mixer.pre_init(frequency=RATE, size=-16, channels=2, buffer=512)
pygame.init()

full_audio = np.array([], dtype=np.int16)

for i, wav in enumerate(audio_files, 1):
    print(f"  {i}/{len(audio_files)} → {os.path.basename(wav)}")
    snd = pygame.mixer.Sound(wav)
    arr = pygame.sndarray.samples(snd)
    # stereo → mono
    if arr.ndim == 2:
        arr = arr.mean(axis=1)
    full_audio = np.concatenate((full_audio, arr.astype(np.int16)))

total_seconds = len(full_audio) / RATE
print(f"\nTotal ≈ {total_seconds/60:.1f} minutes of pure peace\n")


# ========================= VISUAL SETUP =========================
PALETTES = [
    [(255, 200, 220), (200, 255, 255), (220, 180, 255), (255, 240, 200)],
    [(180, 255, 200), (255, 220, 180), (220, 255, 255), (255, 200, 180)],
    [(255, 220, 180), (180, 220, 255), (220, 255, 200), (255, 180, 240)],
    [(250, 230, 180), (180, 250, 230), (230, 180, 250), (255, 245, 200)],
]

os.makedirs(FRAME_DIR, exist_ok=True)
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Peaceful Kaleidoscope Visualizer")


def circle_alpha(surface, color, center, radius, alpha=255):
    """
    Draw a circle with per-pixel alpha in a way that works on pygame 2.5.2.
    Alpha is safely clamped into [0, 255].
    """
    cx, cy = center
    a = max(0, min(255, int(alpha)))
    if a <= 0 or radius <= 0:
        return

    temp = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
    pygame.draw.circle(temp, (*color, a), (radius, radius), radius)
    surface.blit(temp, (cx - radius, cy - radius))


def draw_peace_kaleidoscope(surface, cx, cy, radius, bars):
    t = pygame.time.get_ticks() * 0.001
    palette = PALETTES[int(t / 12) % len(PALETTES)]

    # Low bass breathing for subtle pulsing (not strictly needed, but nice)
    bass = np.mean(bars[:20]) if len(bars) >= 20 else 0

    # -------- Glow rings --------
    for i in range(9):
        base_r = 40 + i * 50
        # Add a tiny bass wobble
        r = int(base_r * (1.0 + 0.08 * bass))

        alpha = 40 + 80 * math.sin(t * 1.2 + i)
        col = palette[i % len(palette)]
        circle_alpha(surface, col, (cx, cy), r, alpha)

    # -------- 12-fold particles --------
    for seg in range(12):
        angle = seg * 30
        part = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)

        for x in range(40, radius, 14):
            if len(bars) == 0:
                intensity = 0.0
            else:
                intensity = float(bars[x % len(bars)])

            size = 6 + int(40 * intensity)
            alpha = 90 + 165 * intensity
            col = palette[int(x / 80) % len(palette)]

            py = radius // 2 + 50 * math.sin(t * 0.7 + x * 0.02 + seg)
            circle_alpha(part, col, (int(x), int(py)), size, alpha)

        rotated = pygame.transform.rotate(part, angle)
        rect = rotated.get_rect(center=(cx, cy))
        surface.blit(rotated, rect)


# ========================= RENDER LOOP =========================
samples_per_frame = RATE // FPS
total_frames = len(full_audio) // samples_per_frame + 10

print(f"Rendering {total_frames:,} dreamy frames…\n")

for frame_idx in range(total_frames):
    # Allow closing the window
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            print("\nUser requested quit. Stopping early.")
            pygame.quit()
            sys.exit(0)

    start = frame_idx * samples_per_frame
    chunk = full_audio[start:start + samples_per_frame]
    if len(chunk) == 0:
        # no more audio data; stop rendering extra frames
        break
    if len(chunk) < samples_per_frame:
        chunk = np.pad(chunk, (0, samples_per_frame - len(chunk)))

    fft_vals = np.abs(fft(chunk))[:512]
    if fft_vals.size > 0:
        fft_max = fft_vals.max()
        if fft_max > 0:
            bars = fft_vals / fft_max
        else:
            bars = np.zeros(512)
    else:
        bars = np.zeros(512)

    # Background
    screen.fill((8, 4, 22))

    # Kaleidoscope
    draw_peace_kaleidoscope(
        screen,
        WIDTH // 2,
        HEIGHT // 2,
        min(WIDTH, HEIGHT) // 2 - 30,
        bars,
    )

    # Update the visible window so you can see it live
    pygame.display.flip()

    # Save this frame
    pygame.image.save(screen, f"{FRAME_DIR}/frame_{frame_idx:08d}.png")

    if frame_idx % 100 == 0 or frame_idx == total_frames - 1:
        print(f"\rFrame {frame_idx+1}/{total_frames} – pure bliss", end="", flush=True)

print("\n\nFinished rendering frames.")


# ========================= ENCODE WITH FFMPEG =========================
print("Encoding your masterpiece…")

# Build audio concat list
list_path = "/tmp/peace_list.txt"
with open(list_path, "w") as f:
    for w in audio_files:
        f.write(f"file '{w}'\n")

# High-quality AAC audio
os.system(
    f"ffmpeg -y -f concat -safe 0 -i {list_path} "
    f"-c:a aac -b:a 320k /tmp/peace_audio.aac"
)

# Combine frames + audio into final MP4
os.system(
    f'ffmpeg -y -r {FPS} -i {FRAME_DIR}/frame_%08d.png -i /tmp/peace_audio.aac '
    f'-c:v libx264 -preset slow -crf 15 -pix_fmt yuv420p -c:a copy -shortest '
    f'"{OUTPUT_VIDEO}"'
)

# Cleanup
os.system(f"rm -rf {FRAME_DIR} {list_path} /tmp/peace_audio.aac")

print("\nFINISHED! Your peaceful album video is ready:")
print(f"   {OUTPUT_VIDEO}")

pygame.quit()
