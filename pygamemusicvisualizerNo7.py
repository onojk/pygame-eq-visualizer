import numpy as np
import pyaudio
import pygame
from scipy.fftpack import fft
import math
import random
import sys

# Initialize Pygame
pygame.init()

# Constants
CHUNK = 512
CHANNELS = 1
RATE = 22050
NUM_SLICES = 16
SLICE_ANGLE = 360 / NUM_SLICES
WINDOW_WIDTH, WINDOW_HEIGHT = 1280, 720
RAINBOW_SPEED = 0.002
LINE_DENSITY = 2  # Reduced line density for 50% fewer lines
COLOR_ROTATION_SPEED = 1  # Rotate colors every frame
GLOBAL_ALPHA = 255

# Color Sets
COLOR_SETS = [
    [(0, 123, 255, GLOBAL_ALPHA), (255, 0, 255, GLOBAL_ALPHA), (0, 255, 0, GLOBAL_ALPHA), (255, 255, 0, GLOBAL_ALPHA)],  # Vibrant Neon
    [(226, 114, 91, GLOBAL_ALPHA), (128, 128, 0, GLOBAL_ALPHA), (92, 64, 51, GLOBAL_ALPHA), (244, 164, 96, GLOBAL_ALPHA)],  # Warm Earth
    [(152, 255, 152, GLOBAL_ALPHA), (230, 230, 250, GLOBAL_ALPHA), (179, 229, 252, GLOBAL_ALPHA), (255, 218, 185, GLOBAL_ALPHA)],  # Pastel
    [(192, 192, 192, GLOBAL_ALPHA), (255, 215, 0, GLOBAL_ALPHA), (205, 127, 50, GLOBAL_ALPHA), (47, 79, 79, GLOBAL_ALPHA)],  # Metallic
]

# Audio Input Setup
p = pyaudio.PyAudio()

def get_input_device_index():
    try:
        return p.get_default_input_device_info()['index']
    except IOError:
        print("No default input device found.")
        sys.exit(1)

input_device_index = get_input_device_index()

def open_audio_stream():
    stream = p.open(format=pyaudio.paInt16,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    input_device_index=input_device_index,
                    frames_per_buffer=CHUNK)
    return stream

stream = open_audio_stream()

pygame.display.set_caption("Kaleidoscope Visualizer")
window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.RESIZABLE | pygame.DOUBLEBUF)

def generate_simple_lines(surface, width, height, audio_bars, color_set, slice_index):
    """
    Draws fewer lines, dynamically selecting colors from the current color set.
    """
    num_lines = min(LINE_DENSITY, len(audio_bars))
    center_x, center_y = width // 2, height // 2

    for i in range(num_lines):
        bar = audio_bars[i]
        color = color_set[(slice_index + i) % len(color_set)]

        # Line properties
        amplitude = bar * height * 0.2
        frequency = 0.01 * (i + 1)
        phase_shift = pygame.time.get_ticks() * 0.001
        thickness = 2

        # Wavy line
        points = []
        for x in range(0, width // 2, 40):  # Reduced points for performance
            y = center_y + amplitude * math.sin(frequency * x + phase_shift)
            points.append((center_x + x, int(y)))

        # Mirror the points
        mirrored_points = [(center_x - (x - center_x), y) for x, y in points[::-1]]

        # Draw both lines
        pygame.draw.lines(surface, color, False, points + mirrored_points, thickness)

def draw_kaleidoscope(surface, center, radius, audio_bars, color_offset):
    """
    Draws the kaleidoscope visualization with reduced line density and rotating colors.
    """
    color_set_index = int(pygame.time.get_ticks() / 1000) % len(COLOR_SETS)  # Rotate color sets
    current_color_set = COLOR_SETS[color_set_index]

    for i in range(NUM_SLICES):
        start_angle = i * SLICE_ANGLE

        # Create a temporary surface for the slice
        slice_surface = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        generate_simple_lines(slice_surface, radius * 2, radius * 2, audio_bars, current_color_set, i)

        # Rotate and position the slice
        slice_rotated = pygame.transform.rotate(slice_surface, start_angle)
        slice_rect = slice_rotated.get_rect(center=center)
        surface.blit(slice_rotated, slice_rect)

def main():
    clock = pygame.time.Clock()
    running = True
    color_offset = 0

    while running:
        dt = clock.tick(30) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        window.fill((0, 0, 0))

        try:
            data = np.frombuffer(stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16)
        except IOError:
            continue

        if CHANNELS == 2:
            data = data.reshape(-1, 2).mean(axis=1).astype(np.int16)

        audio_bars = np.abs(fft(data))[:CHUNK // 2]
        audio_bars = audio_bars / np.max(audio_bars)  # Normalize

        # Draw the kaleidoscope
        draw_kaleidoscope(window, (WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2), min(WINDOW_WIDTH, WINDOW_HEIGHT) // 2 - 50, audio_bars, color_offset)
        pygame.display.flip()
        color_offset += RAINBOW_SPEED

    stream.stop_stream()
    stream.close()
    p.terminate()
    pygame.quit()

if __name__ == "__main__":
    main()

