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
BEAT_THRESHOLD_MULTIPLIER = 1.3  # Multiplier for dynamic threshold
ENERGY_HISTORY = deque(maxlen=43)  # Approximately 1 second of history at ~43 FPS

# Global variables
rainbow_offset = 0
paused = False
circle_list = []
particles = []
prev_bass_amplitude = 0
rotation_angle = 0
rotation_speed = 0.5  # Base rotation speed
is_fullscreen = False  # Track full-screen state

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

# Define color modes
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
current_color_mode = 0

# Get frequency bars from audio data
def get_frequency_bars(data, num_bars, window_height):
    fft_data = fft(data)
    fft_magnitude = np.abs(fft_data)[:CHUNK//2]
    freq_bins = np.linspace(0, RATE/2, CHUNK//2)
    freq_band_limits = np.logspace(np.log10(20), np.log10(RATE/2), num_bars + 1)

    bar_heights = []
    for i in range(num_bars):
        idx = np.where((freq_bins >= freq_band_limits[i]) & (freq_bins < freq_band_limits[i+1]))[0]
        if len(idx) > 0:
            avg_magnitude = np.mean(fft_magnitude[idx])
            bar_height = int((avg_magnitude / (np.max(fft_magnitude) + 1e-6)) * window_height)
            bar_heights.append(bar_height)
        else:
            bar_heights.append(0)
    return bar_heights

# Draw rotating spokes
def draw_rotating_spokes(surface, bars, center, max_radius, num_spokes=240, rotation_angle=0):
    angle_between_spokes = 360 / num_spokes
    for i in range(num_spokes):
        angle = i * angle_between_spokes + rotation_angle
        bar_index = i % len(bars)
        bar_height = bars[bar_index]
        spoke_length = max_radius * (bar_height / (max(bars) + 1e-6))
        spoke_length = min(spoke_length, max_radius)
        rad = math.radians(angle)
        end_x = center[0] + spoke_length * math.cos(rad)
        end_y = center[1] - spoke_length * math.sin(rad)
        color = color_modes[current_color_mode](bar_index, len(bars), rainbow_offset)
        pygame.draw.line(surface, color, center, (end_x, end_y), width=2)

# Draw expanding circles on beat
def draw_expanding_circles(surface, circles, center):
    for circle in circles:
        try:
            pygame.draw.circle(surface, circle['color'], center, int(circle['radius']), 2)
            circle['radius'] += circle['growth_rate'] + 3 * math.sin(pygame.time.get_ticks() / 100.0)  # Pulsating effect
            circle['alpha'] -= circle['decay_rate']
            circle['color'] = (*circle['base_color'], max(int(circle['alpha']), 0))
        except KeyError as e:
            print(f"Missing key in circle dictionary: {e}")

    circles[:] = [c for c in circles if c['alpha'] > 0]

# Draw particles on beat
def draw_particles(surface):
    for particle in particles:
        particle['position'][0] += particle['velocity'][0]
        particle['position'][1] += particle['velocity'][1]
        particle['lifespan'] -= 1
        pygame.draw.circle(surface, particle['color'], particle['position'], particle['size'])
    particles[:] = [p for p in particles if p['lifespan'] > 0]

# Function to toggle full-screen mode
def toggle_fullscreen(window, is_fullscreen):
    if is_fullscreen:
        window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.RESIZABLE | pygame.SRCALPHA | pygame.DOUBLEBUF)
    else:
        window = pygame.display.set_mode((0, 0), pygame.FULLSCREEN | pygame.SRCALPHA | pygame.DOUBLEBUF)
    return window

# Main loop
def main():
    global current_color_mode, rainbow_offset, paused, prev_bass_amplitude, rotation_angle, rotation_speed, is_fullscreen, WINDOW_WIDTH, WINDOW_HEIGHT, circle_surface

    WINDOW_WIDTH, WINDOW_HEIGHT = 1200, 900
    window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.RESIZABLE | pygame.SRCALPHA | pygame.DOUBLEBUF)
    pygame.display.set_caption("Audio Visualizer")
    NUM_BARS = max(10, WINDOW_WIDTH // 40)
    NUM_BARS = min(NUM_BARS, 40)

    clock = pygame.time.Clock()
    running = True
    circle_surface = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)

    try:
        while running:
            dt = clock.tick(60) / 1000.0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RIGHT:
                        current_color_mode = (current_color_mode + 1) % len(color_modes)
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_f or event.key == pygame.K_F11:
                        is_fullscreen = not is_fullscreen
                        if is_fullscreen:
                            window = pygame.display.set_mode((0, 0), pygame.FULLSCREEN | pygame.SRCALPHA | pygame.DOUBLEBUF)
                            WINDOW_WIDTH, WINDOW_HEIGHT = window.get_size()
                        else:
                            WINDOW_WIDTH, WINDOW_HEIGHT = 1200, 900
                            window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.RESIZABLE | pygame.SRCALPHA | pygame.DOUBLEBUF)
                        circle_surface = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
                        NUM_BARS = max(10, WINDOW_WIDTH // 40)
                        NUM_BARS = min(NUM_BARS, 40)
                elif event.type == pygame.VIDEORESIZE and not is_fullscreen:
                    WINDOW_WIDTH, WINDOW_HEIGHT = event.w, event.h
                    window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.RESIZABLE | pygame.SRCALPHA | pygame.DOUBLEBUF)
                    circle_surface = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
                    NUM_BARS = max(10, WINDOW_WIDTH // 40)
                    NUM_BARS = min(NUM_BARS, 40)

            window.fill((0, 0, 0))

            if not paused:
                try:
                    data = np.frombuffer(stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16)
                except IOError as e:
                    print(f"Audio buffer overflow: {e}. Skipping frame.")
                    continue

                if CHANNELS == 2:
                    data = data.reshape(-1, 2).mean(axis=1).astype(np.int16)

                bars = get_frequency_bars(data, NUM_BARS, max(WINDOW_WIDTH, WINDOW_HEIGHT) // 2)
                center = (WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2)
                draw_rotating_spokes(window, bars, center, min(WINDOW_WIDTH, WINDOW_HEIGHT) // 2 - 50, rotation_angle=rotation_angle)

                bass_energy = np.sum(data[:CHUNK//8]**2)
                ENERGY_HISTORY.append(bass_energy)
                average_energy = np.mean(ENERGY_HISTORY) if ENERGY_HISTORY else 1
                threshold = BEAT_THRESHOLD_MULTIPLIER * average_energy

                if bass_energy > threshold and len(circle_list) < MAX_CIRCLES:
                    base_col = color_modes[current_color_mode](0, NUM_BARS, rainbow_offset)
                    new_circle = {
                        'radius': 10,
                        'growth_rate': 5,
                        'decay_rate': 3,
                        'alpha': 255,
                        'base_color': base_col,
                        'color': (*base_col, 255)
                    }
                    circle_list.append(new_circle)

                    for _ in range(5):
                        particles.append({
                            'position': list(center),
                            'velocity': [np.random.uniform(-5, 5), np.random.uniform(-5, 5)],
                            'color': base_col,
                            'size': np.random.randint(2, 6),
                            'lifespan': 100
                        })

                    rotation_speed = 2.0
                else:
                    rotation_speed += (0.5 - rotation_speed) * 0.05

                rotation_angle += rotation_speed * dt * 60
                draw_expanding_circles(circle_surface, circle_list, center)
                window.blit(circle_surface, (0, 0))
                draw_particles(window)

            pygame.display.flip()
            rainbow_offset += 0.001

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
        pygame.quit()

if __name__ == "__main__":
    main()
