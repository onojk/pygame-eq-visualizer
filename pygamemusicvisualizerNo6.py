import numpy as np
import pyaudio
import pygame
from scipy.fftpack import fft
import math
import sys
from collections import deque

# Initialize Pygame
pygame.init()

# Constants
CHUNK = 1024
RATE = 44100
CHANNELS = 2
MAX_CIRCLES = 15
BEAT_THRESHOLD_MULTIPLIER = 1.3
ENERGY_HISTORY = deque(maxlen=43)

# Global Variables
rainbow_offset = 0
paused = False
circle_list = []
particles = []
prev_bass_amplitude = 0
rotation_angle = 0
rotation_speed = 0.5
is_fullscreen = False
layer_offset = 0
current_color_mode = 0
layers = ["spokes", "circles", "particles"]  # Layer order
bar_scaling_factor = 1.5  # Scale bars for larger spokes
circle_growth_scaling = 2.0  # Scale circle radius growth

# PyAudio Setup
p = pyaudio.PyAudio()

def get_input_device_index(desired_input_device=None):
    if desired_input_device is None:
        return p.get_default_input_device_info()['index']
    for i in range(p.get_device_count()):
        device_info = p.get_device_info_by_index(i)
        if desired_input_device.lower() in device_info['name'].lower():
            return i
    return None

# Input device can be None to use the default
input_device_index = get_input_device_index()

# Try multiple sample rates to open the audio stream
sample_rates_to_try = [44100, 48000, 22050, 16000, 11025, 8000]

for rate in sample_rates_to_try:
    try:
        stream = p.open(format=pyaudio.paInt16,
                        channels=CHANNELS,
                        rate=rate,
                        input=True,
                        input_device_index=input_device_index,
                        frames_per_buffer=CHUNK)
        RATE = rate  # Set RATE to the successfully opened sample rate
        print(f"Opened audio stream with sample rate: {RATE}")
        break
    except Exception as e:
        print(f"Could not open audio stream with sample rate {rate}: {e}")
else:
    print("Failed to open audio stream. Exiting.")
    p.terminate()
    sys.exit(1)

# Color Modes
def rainbow_color(index, total_bars, offset):
    hue = (index / total_bars + offset) % 1.0
    color = pygame.Color(0)
    color.hsva = (hue * 360, 100, 100, 100)
    return (color.r, color.g, color.b)

def grayscale_color(index, total_bars, offset):
    intensity = int(255 * (((index / total_bars) + offset) % 1.0))
    return (intensity, intensity, intensity)

def fire_color(index, total_bars, offset):
    ratio = ((index / total_bars) + offset) % 1.0
    if ratio < 0.5:
        r = 255
        g = int(255 * (ratio / 0.5))
        b = 0
    else:
        r = 255
        g = 255
        b = int(255 * ((ratio - 0.5) / 0.5))
    return (r, g, b)

color_modes = [rainbow_color, grayscale_color, fire_color]

# Frequency Bars
def get_frequency_bars(data, num_bars, window_height):
    fft_data = fft(data)
    fft_magnitude = np.abs(fft_data[:CHUNK // 2])
    bar_heights = np.interp(
        np.linspace(0, len(fft_magnitude), num_bars),
        np.arange(len(fft_magnitude)),
        fft_magnitude
    )
    return (bar_heights / np.max(bar_heights)) * window_height * bar_scaling_factor

# Visual Elements
def draw_rotating_spokes(surface, bars, center, max_radius, num_spokes=40, rotation_angle=0):
    max_radius *= 1.5  # Extend spokes beyond the screen
    angle_between_spokes = 360 / num_spokes
    for i in range(num_spokes):
        angle = i * angle_between_spokes + rotation_angle
        bar_index = i % len(bars)
        bar_height = bars[bar_index]
        spoke_length = max_radius * (bar_height / (max(bars) + 1e-6))
        rad = math.radians(angle)
        end_x = center[0] + spoke_length * math.cos(rad)
        end_y = center[1] - spoke_length * math.sin(rad)
        color = color_modes[current_color_mode](bar_index, len(bars), rainbow_offset)
        pygame.draw.line(surface, color, center, (end_x, end_y), 2)

def draw_expanding_circles(surface, circles, center):
    max_radius = max(WINDOW_WIDTH, WINDOW_HEIGHT)  # Adjust circles to cover the entire screen
    for circle in circles:
        pygame.draw.circle(surface, circle['color'], center, min(int(circle['radius']), max_radius), 2)
        circle['radius'] += circle['growth_rate'] * circle_growth_scaling  # Scale circle growth
        circle['alpha'] -= circle['decay_rate']
        circle['color'] = (*circle['base_color'], max(int(circle['alpha']), 0))
    circles[:] = [c for c in circles if c['alpha'] > 0]

def draw_particles(surface):
    for particle in particles:
        particle['position'][0] += particle['velocity'][0]
        particle['position'][1] += particle['velocity'][1]
        particle['lifespan'] -= 1
        pygame.draw.circle(surface, particle['color'], particle['position'], particle['size'])
    particles[:] = [p for p in particles if p['lifespan'] > 0]

# Layer Rendering
def draw_layers(surface, layers, center, bars):
    for layer in layers:
        if layer == "spokes":
            draw_rotating_spokes(surface, bars, center, min(WINDOW_WIDTH, WINDOW_HEIGHT) // 2 - 50)
        elif layer == "circles":
            draw_expanding_circles(circle_surface, circle_list, center)
            surface.blit(circle_surface, (0, 0))
        elif layer == "particles":
            draw_particles(surface)

# Toggle Fullscreen
def toggle_fullscreen(window, is_fullscreen):
    global WINDOW_WIDTH, WINDOW_HEIGHT, circle_surface
    if is_fullscreen:
        WINDOW_WIDTH, WINDOW_HEIGHT = pygame.display.get_desktop_sizes()[0]  # Full desktop resolution
        window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.FULLSCREEN | pygame.SRCALPHA)
    else:
        WINDOW_WIDTH, WINDOW_HEIGHT = 1200, 900  # Default window size
        window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.RESIZABLE | pygame.SRCALPHA)

    circle_surface = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
    return window, not is_fullscreen

# Main Loop
def main():
    global layers, current_color_mode, rainbow_offset, paused, rotation_angle, rotation_speed, is_fullscreen, circle_surface, WINDOW_WIDTH, WINDOW_HEIGHT

    # Window Setup
    WINDOW_WIDTH, WINDOW_HEIGHT = 1200, 900
    window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.RESIZABLE | pygame.SRCALPHA)
    pygame.display.set_caption("Audio Visualizer")
    clock = pygame.time.Clock()
    running = True

    circle_surface = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)

    try:
        while running:
            dt = clock.tick(60) / 1000.0

            # Event Handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RIGHT:
                        current_color_mode = (current_color_mode + 1) % len(color_modes)
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key == pygame.K_UP:
                        layers.insert(0, layers.pop())
                    elif event.key == pygame.K_DOWN:
                        layers.append(layers.pop(0))
                    elif event.key == pygame.K_f or event.key == pygame.K_F11:
                        window, is_fullscreen = toggle_fullscreen(window, is_fullscreen)

            window.fill((0, 0, 0))

            if not paused:
                data = np.frombuffer(stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16)
                if CHANNELS == 2:
                    data = data.reshape(-1, 2).mean(axis=1).astype(np.int16)

                bars = get_frequency_bars(data, max(10, WINDOW_WIDTH // 40), WINDOW_HEIGHT // 2)
                center = (WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2)  # Center visuals
                draw_layers(window, layers, center, bars)

                # Beat Detection
                bass_energy = np.sum(data[:CHUNK // 8]**2)
                ENERGY_HISTORY.append(bass_energy)
                threshold = BEAT_THRESHOLD_MULTIPLIER * np.mean(ENERGY_HISTORY)

                if bass_energy > threshold:
                    base_col = color_modes[current_color_mode](0, len(bars), rainbow_offset)
                    circle_list.append({
                        'radius': 10,
                        'growth_rate': 4,
                        'decay_rate': 3,
                        'alpha': 255,
                        'base_color': base_col,
                        'color': (*base_col, 255)
                    })

            rainbow_offset += 0.002
            pygame.display.flip()

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
        pygame.quit()

if __name__ == "__main__":
    main()

