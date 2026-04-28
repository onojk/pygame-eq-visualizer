import numpy as np
import pygame
import pygame.sndarray
from scipy.fftpack import fft
import math
import random
import sys
import os
from pygame import surfarray

# ------------------- CONFIGURATION -------------------
# Change these two lines only
AUDIO_FILE = "/home/onojk123/Downloads/truth2_45884854/Number01.wav"   # <-- put your file here
OUTPUT_VIDEO = "/home/onojk123/Number01_kaleidoscope.mp4"             # final video

WIDTH, HEIGHT = 1280, 720
FPS = 24
# -----------------------------------------------------

# Same constants as your original script
CHUNK = 512
NUM_SLICES = 16
SLICE_ANGLE = 360 / NUM_SLICES
RAINBOW_SPEED = 0.002
LINE_DENSITY = 1
GLOBAL_ALPHA = 255
BROKEN_WAVE_FRACTION = 0.8
show_black_lines = True

# Create temp folder for frames
FRAME_DIR = "temp_frames"
os.makedirs(FRAME_DIR, exist_ok=True)

pygame.init()
pygame.display.set_caption("Kaleidoscope → Video Render")
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

# Load the audio file with pygame.mixer (so we get exact length and perfect sync)
pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=512)
sound = pygame.mixer.Sound(AUDIO_FILE)
channel = sound.play()
total_frames = int(sound.get_length() * FPS)

# Pre-load entire audio into numpy array (so FFT is perfectly in sync)
raw_audio = pygame.sndarray.array(sound)
RATE = 22050
samples_per_frame = RATE // FPS
total_samples = len(raw_audio)

# Color sets (same as yours)
COLOR_SETS = [
    [(0, 123, 255, GLOBAL_ALPHA), (255, 0, 255, GLOBAL_ALPHA), (0, 255, 0, GLOBAL_ALPHA), (255, 255, 0, GLOBAL_ALPHA)],
    [(226, 114, 91, GLOBAL_ALPHA), (128, 128, 0, GLOBAL_ALPHA), (92, 64, 51, GLOBAL_ALPHA), (244, 164, 96, GLOBAL_ALPHA)],
    [(152, 255, 152, GLOBAL_ALPHA), (230, 230, 250, GLOBAL_ALPHA), (179, 229, 252, GLOBAL_ALPHA), (255, 218, 185, GLOBAL_ALPHA)],
    [(192, 192, 192, GLOBAL_ALPHA), (255, 215, 0, GLOBAL_ALPHA), (205, 127, 50, GLOBAL_ALPHA), (47, 79, 79, GLOBAL_ALPHA)],
]

# ---- Your original drawing functions (unchanged, only tiny fixes) ----
def generate_simple_lines(surface, width, height, audio_bars, color_set, slice_index):
    num_lines = min(LINE_DENSITY, len(audio_bars))
    center_x, center_y = width // 2, height // 2
    for i in range(num_lines):
        bar = audio_bars[i]
        color = color_set[(slice_index + i) % len(color_set)]
        amplitude = bar * height * 0.2
        frequency = 0.01 * (i + 1)
        phase_shift = pygame.time.get_ticks() * 0.001
        thickness = 2
        points = []
        for x in range(0, width // 2, 60):
            y = center_y + amplitude * math.sin(frequency * x + phase_shift)
            points.append((center_x + x, int(y)))
        mirrored_points = [(center_x - (x - center_x), y) for x, y in points[::-1]]
        pygame.draw.lines(surface, color[:3], False, points + mirrored_points, thickness)

def draw_broken_black_wave(surface, width, height, slice_index):
    if not show_black_lines:
        return
    center_x, center_y = width // 2, height // 2
    amplitude = height * 0.15
    frequency = 0.03 * (slice_index + 1)
    phase_shift = pygame.time.get_ticks() * 0.001
    thickness = random.randint(2, 4)
    points = []
    for x in range(0, int(width * BROKEN_WAVE_FRACTION), 60):
        y = center_y + amplitude * math.sin(frequency * x + phase_shift)
        points.append((center_x + x, int(y)))
    mirrored_points = [(center_x - (x - center_x), y) for x, y in points[::-1]]
    pygame.draw.lines(surface, (0, 0, 0), False, points + mirrored_points, thickness)

def draw_kaleidoscope(surface, center, radius, audio_bars, color_offset):
    color_set_index = int(pygame.time.get_ticks() / 1000) % len(COLOR_SETS)
    current_color_set = COLOR_SETS[color_set_index]
    for i in range(NUM_SLICES):
        start_angle = i * SLICE_ANGLE
        slice_surface = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        generate_simple_lines(slice_surface, radius * 2, radius * 2, audio_bars, current_color_set, i)
        draw_broken_black_wave(slice_surface, radius * 2, radius * 2, i)
        slice_rotated = pygame.transform.rotate(slice_surface, start_angle)
        slice_rect = slice_rotated.get_rect(center=center)
        surface.blit(slice_rotated, slice_rect)

# ------------------- MAIN RENDER LOOP -------------------
print(f"Rendering {total_frames} frames → {OUTPUT_VIDEO}")
frame_idx = 0

while channel.get_busy() or frame_idx < total_frames:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    # Calculate exact sample range for this frame
    start_sample = int(frame_idx * samples_per_frame)
    end_sample = min(start_sample + samples_per_frame, total_samples)
    frame_data = raw_audio[start_sample:end_sample]

    # Pad if last frame is short
    if len(frame_data) < samples_per_frame:
        frame_data = np.pad(frame_data, (0, samples_per_frame - len(frame_data)), mode='constant')

    # FFT exactly like your original code
    fft_vals = np.abs(fft(frame_data))[:CHUNK//2]
    max_val = np.max(fft_vals)
    if max_val > 0:
        audio_bars = fft_vals / max_val
    else:
        audio_bars = np.zeros(CHUNK//2)

    screen.fill((0, 0, 0))
    radius = min(WIDTH, HEIGHT) // 2 - 50
    draw_kaleidoscope(screen, (WIDTH//2, HEIGHT//2), radius, audio_bars, 0)

    # Save frame
    filename = f"{FRAME_DIR}/frame_{frame_idx:06d}.png"
    pygame.image.save(screen, filename)

    pygame.display.flip()
    clock.tick(FPS)
    frame_idx += 1

    print(f"\rRendered frame {frame_idx}/{total_frames}", end="")

print("\nRendering complete! Now encoding with ffmpeg...")

# ------------------- FFMPEG ENCODING -------------------
# High quality, fast, widely compatible
ffmpeg_cmd = (
    f'ffmpeg -y -r {FPS} -i {FRAME_DIR}/frame_%06d.png '
    f'-i "{AUDIO_FILE}" '
    f'-c:v libx264 -preset slow -crf 18 -pix_fmt yuv420p '
    f'-c:a aac -b:a 320k -shortest "{OUTPUT_VIDEO}"'
)

os.system(ffmpeg_cmd)

# Clean up frames (optional)
import shutil
shutil.rmtree(FRAME_DIR)
print(f"Video saved → {OUTPUT_VIDEO}")
pygame.quit()
