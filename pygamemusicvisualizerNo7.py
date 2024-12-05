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
DEFAULT_WIDTH, DEFAULT_HEIGHT = 1280, 720
RAINBOW_SPEED = 0.002
LINE_DENSITY = 1  # Reduced line density for better performance
GLOBAL_ALPHA = 255
BROKEN_WAVE_FRACTION = 0.8
FPS = 24  # Lower frame rate for smoother fullscreen performance

# Global Toggle
show_black_lines = True

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
window = pygame.display.set_mode((DEFAULT_WIDTH, DEFAULT_HEIGHT), pygame.RESIZABLE | pygame.DOUBLEBUF)

is_fullscreen = False

COLOR_SETS = [
    [(0, 123, 255, GLOBAL_ALPHA), (255, 0, 255, GLOBAL_ALPHA), (0, 255, 0, GLOBAL_ALPHA), (255, 255, 0, GLOBAL_ALPHA)],
    [(226, 114, 91, GLOBAL_ALPHA), (128, 128, 0, GLOBAL_ALPHA), (92, 64, 51, GLOBAL_ALPHA), (244, 164, 96, GLOBAL_ALPHA)],
    [(152, 255, 152, GLOBAL_ALPHA), (230, 230, 250, GLOBAL_ALPHA), (179, 229, 252, GLOBAL_ALPHA), (255, 218, 185, GLOBAL_ALPHA)],
    [(192, 192, 192, GLOBAL_ALPHA), (255, 215, 0, GLOBAL_ALPHA), (205, 127, 50, GLOBAL_ALPHA), (47, 79, 79, GLOBAL_ALPHA)],
]

def toggle_fullscreen():
    global is_fullscreen, window, WINDOW_WIDTH, WINDOW_HEIGHT
    is_fullscreen = not is_fullscreen
    if is_fullscreen:
        window = pygame.display.set_mode((0, 0), pygame.FULLSCREEN | pygame.DOUBLEBUF)
    else:
        window = pygame.display.set_mode((DEFAULT_WIDTH, DEFAULT_HEIGHT), pygame.RESIZABLE | pygame.DOUBLEBUF)
    WINDOW_WIDTH, WINDOW_HEIGHT = window.get_size()

WINDOW_WIDTH, WINDOW_HEIGHT = DEFAULT_WIDTH, DEFAULT_HEIGHT

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
        for x in range(0, width // 2, 60):  # Increased step size for fewer points
            y = center_y + amplitude * math.sin(frequency * x + phase_shift)
            points.append((center_x + x, int(y)))

        mirrored_points = [(center_x - (x - center_x), y) for x, y in points[::-1]]
        pygame.draw.lines(surface, color, False, points + mirrored_points, thickness)

def draw_broken_black_wave(surface, width, height, slice_index):
    if not show_black_lines:
        return
    center_x, center_y = width // 2, height // 2
    amplitude = height * 0.15
    frequency = 0.03 * (slice_index + 1)
    phase_shift = pygame.time.get_ticks() * 0.001

    thickness = random.randint(2, 4)  # Thinner lines for performance

    points = []
    for x in range(0, int(width * BROKEN_WAVE_FRACTION), 60):  # Increased step size
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

def main():
    global show_black_lines
    clock = pygame.time.Clock()
    running = True
    color_offset = 0

    while running:
        dt = clock.tick(FPS) / 1000.0  # Limit FPS
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_f or event.key == pygame.K_F11:
                    toggle_fullscreen()
                elif event.key == pygame.K_b:
                    show_black_lines = not show_black_lines

        window.fill((0, 0, 0))

        try:
            data = np.frombuffer(stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16)
        except IOError:
            continue

        if CHANNELS == 2:
            data = data.reshape(-1, 2).mean(axis=1).astype(np.int16)

        audio_bars = np.abs(fft(data))[:CHUNK // 2]
        max_value = np.max(audio_bars)
        if max_value > 0:
            audio_bars = audio_bars / max_value
        else:
            audio_bars = np.zeros_like(audio_bars)

        draw_kaleidoscope(window, (WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2), min(WINDOW_WIDTH, WINDOW_HEIGHT) // 2 - 50, audio_bars, color_offset)
        pygame.display.flip()
        color_offset += RAINBOW_SPEED

    stream.stop_stream()
    stream.close()
    p.terminate()
    pygame.quit()

if __name__ == "__main__":
    main()

