import pygame
import numpy as np
from scipy.fftpack import fft
import math
import random
import pyaudio
from PIL import Image
from scipy.spatial import KDTree

# Initialize Pygame
pygame.init()

# Constants
CHUNK = 256  # Reduced for better performance
CHANNELS = 1
RATE = 16000  # Lower sample rate for improved performance
NUM_SLICES = 8  # Fewer slices for reduced computational load
SLICE_ANGLE = 360 / NUM_SLICES
DEFAULT_WIDTH, DEFAULT_HEIGHT = 480, 480  # Smaller window size
RAINBOW_SPEED = 0.005  # Faster rainbow effect
LINE_DENSITY = 1
GLOBAL_ALPHA = 255
BROKEN_WAVE_FRACTION = 0.5  # Reduced wave coverage for performance
FPS = 30  # Target FPS for smoother performance

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
    return p.open(format=pyaudio.paInt16,
                  channels=CHANNELS,
                  rate=RATE,
                  input=True,
                  input_device_index=input_device_index,
                  frames_per_buffer=CHUNK)

stream = open_audio_stream()

# Pygame Display Setup
pygame.display.set_caption("Kaleidoscope Visualizer")
window = pygame.display.set_mode((DEFAULT_WIDTH, DEFAULT_HEIGHT), pygame.RESIZABLE | pygame.DOUBLEBUF)
is_fullscreen = False

# Color Sets
COLOR_SETS = [
    [(255, 69, 0, GLOBAL_ALPHA), (255, 140, 0, GLOBAL_ALPHA), (255, 215, 0, GLOBAL_ALPHA), (50, 205, 50, GLOBAL_ALPHA)],
    [(30, 144, 255, GLOBAL_ALPHA), (138, 43, 226, GLOBAL_ALPHA), (75, 0, 130, GLOBAL_ALPHA), (220, 20, 60, GLOBAL_ALPHA)],
    [(255, 20, 147, GLOBAL_ALPHA), (0, 255, 255, GLOBAL_ALPHA), (255, 99, 71, GLOBAL_ALPHA), (144, 238, 144, GLOBAL_ALPHA)],
    [(255, 105, 180, GLOBAL_ALPHA), (186, 85, 211, GLOBAL_ALPHA), (123, 104, 238, GLOBAL_ALPHA), (60, 179, 113, GLOBAL_ALPHA)]
]

WINDOW_WIDTH, WINDOW_HEIGHT = DEFAULT_WIDTH, DEFAULT_HEIGHT

def toggle_fullscreen():
    global is_fullscreen, window, WINDOW_WIDTH, WINDOW_HEIGHT
    is_fullscreen = not is_fullscreen
    if is_fullscreen:
        window = pygame.display.set_mode((0, 0), pygame.FULLSCREEN | pygame.DOUBLEBUF)
    else:
        window = pygame.display.set_mode((DEFAULT_WIDTH, DEFAULT_HEIGHT), pygame.RESIZABLE | pygame.DOUBLEBUF)
    WINDOW_WIDTH, WINDOW_HEIGHT = window.get_size()

# White Line Extraction and Path Creation
def extract_white_lines(image):
    width, height = image.size
    white_points = [(x, y) for x in range(width) for y in range(height) if image.getpixel((x, y))[:3] == (255, 255, 255)]
    print(f"Extracted {len(white_points)} white points.")
    return np.array(white_points).reshape(-1, 2)  # Ensure 2D shape even if empty

def create_smooth_paths(white_points):
    if len(white_points) == 0:
        print("No white points detected. Skipping path creation.")
        return []

    tree = KDTree(white_points)
    visited = set()
    paths = []

    for point in map(tuple, white_points):
        if point not in visited:
            path = []
            current = point

            while current not in visited:
                visited.add(current)
                path.append(current)
                neighbors = tree.query_ball_point(current, 2)
                next_point = next((tuple(white_points[idx]) for idx in neighbors if tuple(white_points[idx]) not in visited), None)
                if next_point is None:
                    break
                current = next_point

            if path:
                paths.append(path)

    print(f"Created {len(paths)} paths.")
    return paths

# Image Scaling
def scale_and_center_image(image, window_width, window_height):
    img_width, img_height = image.size
    scale_factor = min(window_width / img_width, window_height / img_height)
    new_size = (int(img_width * scale_factor), int(img_height * scale_factor))
    scaled_image = image.resize(new_size, Image.LANCZOS)
    offset_x = (window_width - new_size[0]) // 2
    offset_y = (window_height - new_size[1]) // 2
    return scaled_image, offset_x, offset_y

# Kaleidoscope Drawing
def generate_simple_lines(surface, width, height, audio_bars, color_set, slice_index):
    num_lines = min(LINE_DENSITY, len(audio_bars))
    center_x, center_y = width // 2, height // 2

    for i in range(num_lines):
        bar = audio_bars[i]
        color = color_set[(slice_index + i) % len(color_set)]

        amplitude = bar * height * 0.15
        frequency = 0.02 * (i + 1)
        phase_shift = pygame.time.get_ticks() * 0.001
        thickness = 1

        points = [(center_x + x, int(center_y + amplitude * math.sin(frequency * x + phase_shift)))
                  for x in range(0, width // 2, 80)]
        mirrored_points = [(center_x - (x - center_x), y) for x, y in points[::-1]]

        pygame.draw.lines(surface, color, False, points + mirrored_points, thickness)

def draw_kaleidoscope(surface, center, radius, audio_bars, color_offset, paths, offset_x, offset_y):
    color_set_index = int(pygame.time.get_ticks() / 1000) % len(COLOR_SETS)
    current_color_set = COLOR_SETS[color_set_index]

    for i in range(NUM_SLICES):
        start_angle = i * SLICE_ANGLE

        slice_surface = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        generate_simple_lines(slice_surface, radius * 2, radius * 2, audio_bars, current_color_set, i)

        slice_rotated = pygame.transform.rotate(slice_surface, start_angle)
        slice_rect = slice_rotated.get_rect(center=center)
        surface.blit(slice_rotated, slice_rect)

def apply_decay(surface, decay_rate):
    decay_overlay = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
    decay_overlay.fill((0, 0, 0, decay_rate))
    surface.blit(decay_overlay, (0, 0), special_flags=pygame.BLEND_RGBA_SUB)

# Add visual shake effect based on audio
def apply_shake(draw_surface, audio_bars):
    max_amplitude = max(audio_bars) if len(audio_bars) > 0 else 0
    shake_intensity = int(max_amplitude * 10)  # Adjust shake intensity based on amplitude

    shake_x = random.randint(-shake_intensity, shake_intensity)
    shake_y = random.randint(-shake_intensity, shake_intensity)

    draw_surface.scroll(dx=shake_x, dy=shake_y)

# Main Loop
def main():
    global show_black_lines
    clock = pygame.time.Clock()
    running = True
    color_offset = 0

    # Load and process the image for white line tracing
    image_path = "/home/onojk123/Documents/dasfasdfwqere.jpg"
    image = Image.open(image_path).convert("RGBA")
    scaled_image, offset_x, offset_y = scale_and_center_image(image, WINDOW_WIDTH, WINDOW_HEIGHT)
    white_points = extract_white_lines(scaled_image)
    paths = create_smooth_paths(white_points)

    draw_surface = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)

    while running:
        dt = clock.tick(FPS) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key in {pygame.K_f, pygame.K_F11}:
                    toggle_fullscreen()
                elif event.key == pygame.K_b:
                    show_black_lines = not show_black_lines

        draw_surface.fill((0, 0, 0, 0))

        try:
            data = np.frombuffer(stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16)
        except IOError:
            continue

        if CHANNELS == 2:
            data = data.reshape(-1, 2).mean(axis=1).astype(np.int16)

        audio_bars = np.abs(fft(data))[:CHUNK // 2]
        max_value = np.max(audio_bars)
        audio_bars = audio_bars / max_value if max_value > 0 else np.zeros_like(audio_bars)

        draw_kaleidoscope(
            draw_surface,
            (WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2),
            min(WINDOW_WIDTH, WINDOW_HEIGHT) // 2 - 50,
            audio_bars,
            color_offset,
            paths,
            offset_x,
            offset_y,
        )

        apply_shake(draw_surface, audio_bars)  # Add shake effect
        apply_decay(draw_surface, 10)

        window.blit(draw_surface, (0, 0))
        pygame.display.flip()
        color_offset += RAINBOW_SPEED

    stream.stop_stream()
    stream.close()
    p.terminate()
    pygame.quit()

if __name__ == "__main__":
    main()

