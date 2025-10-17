import numpy as np
import pyaudio
import pygame
from scipy.fftpack import fft
import math
import sys
import random
from collections import deque

# Initialize Pygame
pygame.init()

# Constants
CHUNK = 1024
RATE = 44100
CHANNELS = 2
WINDOW_WIDTH, WINDOW_HEIGHT = 1200, 900
MAX_SPOKES_BOTTOM = 240
MAX_SPOKES_TOP = 180
BASE_SPOKE_SIZE_BOTTOM = 10
BASE_SPOKE_SIZE_TOP = 5
NUM_RANDOM_LINES = 100
LINE_WIDTH = 2
BEAT_THRESHOLD_MULTIPLIER = 1.2
FREQUENCY_SHIFT_THRESHOLD = 2.5
ENERGY_HISTORY = deque(maxlen=43)
SPOKE_SCALE_FACTOR = 1.5
ROTATION_SPEED_BASE = 0.5
ROTATION_SPEED_BOOST = 2.0
BURST_DURATION = 0.5
BURST_SCALE_FACTOR = 5.0
BURST_COLOR_CHANGE = True
WHITE_FLASH_DURATION = 0.2  # Flash duration in seconds
WHITE_FLASH_COOLDOWN = 5.0  # Cooldown period to prevent frequent flashes
HIGH_FREQUENCY_THRESHOLD = 7.0 #Threshold for high frequency event

rainbow_offset = 0
rotation_angle = 0.0
rotation_speed = ROTATION_SPEED_BASE
is_fullscreen = False
paused = False
burst_active = False
burst_timer = 0
flash_active = False
flash_timer = 0
flash_cooldown_timer = 0

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

# Try opening the audio stream with multiple sample rates
sample_rates_to_try = [44100, 48000, 22050]
for rate in sample_rates_to_try:
    try:
        stream = p.open(format=pyaudio.paInt16,
                        channels=CHANNELS,
                        rate=rate,
                        input=True,
                        input_device_index=input_device_index,
                        frames_per_buffer=CHUNK)
        RATE = rate
        print(f"Opened audio stream with sample rate: {RATE}")
        break
    except Exception as e:
        print(f"Could not open audio stream with sample rate {rate}: {e}")
else:
    print("Failed to open audio stream. Exiting.")
    p.terminate()
    sys.exit(1)

# Define vibrant color mode with full saturation and brightness
def vibrant_rainbow_color(index, total_bars, offset):
    hue = (index / total_bars + offset) % 1.0
    color = pygame.Color(0)
    color.hsva = (hue * 360, 100, 100)
    return (color.r, color.g, color.b)

def adjust_opacity(color, opacity):
    return (*color, opacity)

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

# Detect beats by analyzing bass frequencies
def detect_beat(data, threshold_multiplier):
    bass_energy = np.sum(data[:CHUNK//8]**2)
    ENERGY_HISTORY.append(bass_energy)
    average_energy = np.mean(ENERGY_HISTORY) if ENERGY_HISTORY else 1
    threshold = threshold_multiplier * average_energy

    return bass_energy > threshold

# Detect large shifts in energy to trigger color bursts
def detect_large_frequency_shift(data):
    current_energy = np.sum(data**2)
    if len(ENERGY_HISTORY) > 1:
        energy_change = abs(current_energy - ENERGY_HISTORY[-1])
        average_energy = np.mean(ENERGY_HISTORY) if ENERGY_HISTORY else 1
        return energy_change / average_energy > FREQUENCY_SHIFT_THRESHOLD
    return False

# Detect high-frequency spike for triggering white flash
def detect_high_frequency_event(bars):
    high_freq_energy = np.sum(bars[-len(bars)//4:]**2)  # Check the top quarter of the frequency range
    avg_energy = np.mean(bars)
    return high_freq_energy / avg_energy > HIGH_FREQUENCY_THRESHOLD

# Draw rotating spokes with vibrant colors, and scale on beats
def draw_rotating_spokes(surface, bars, center, max_radius, num_spokes, rotation_angle, spoke_size, opacity, scale_factor=1.0):
    angle_between_spokes = 360 / num_spokes

    for i in range(num_spokes):
        angle = i * angle_between_spokes + rotation_angle
        bar_index = i % len(bars)
        bar_height = bars[bar_index]
        spoke_length = max_radius * (bar_height / (max(bars) + 1e-6)) * scale_factor
        spoke_length = min(spoke_length, max_radius)

        # Get vibrant color for the spokes
        color = vibrant_rainbow_color(bar_index, len(bars), rainbow_offset)
        color_with_opacity = adjust_opacity(color, opacity)
        pygame.draw.line(surface, color_with_opacity, center, 
                         (center[0] + spoke_length * math.cos(math.radians(angle)),
                          center[1] - spoke_length * math.sin(math.radians(angle))), width=int(spoke_size * scale_factor))

# Draw random black lines around the center on the top layer
def draw_random_black_lines(surface, center, num_lines, max_length, min_length):
    for _ in range(num_lines):
        angle = random.uniform(0, 360)
        length = random.uniform(min_length, max_length)
        rad = math.radians(angle)
        end_x = center[0] + length * math.cos(rad)
        end_y = center[1] - length * math.sin(rad)

        # Draw black lines
        pygame.draw.line(surface, (0, 0, 0), center, (end_x, end_y), width=LINE_WIDTH)

def main():
    global paused, is_fullscreen, WINDOW_WIDTH, WINDOW_HEIGHT, rainbow_offset, rotation_angle, rotation_speed, burst_active, burst_timer, flash_active, flash_timer, flash_cooldown_timer

    # Initialize Pygame window
    window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.RESIZABLE | pygame.SRCALPHA | pygame.DOUBLEBUF)
    pygame.display.set_caption("Dance Rhythm Audio Visualizer with Bursts and Flash")
    NUM_BARS = max(10, WINDOW_WIDTH // 40)
    NUM_BARS = min(NUM_BARS, 40)

    clock = pygame.time.Clock()
    running = True
    circle_surface = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)

    try:
        while running:
            dt = clock.tick(60) / 1000.0  # Delta time in seconds

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
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

            window.fill((0, 0, 0))  # Clear screen with black

            if not paused:
                try:
                    data = np.frombuffer(stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16)
                except IOError as e:
                    print(f"Audio buffer overflow: {e}. Skipping frame.")
                    continue

                if CHANNELS == 2:
                    data = data.reshape(-1, 2).mean(axis=1).astype(np.int16)

                # Frequency bars based on audio input
                bars = get_frequency_bars(data, NUM_BARS, max(WINDOW_WIDTH, WINDOW_HEIGHT) // 2)
                center = (WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2)

                # Detect beat and adjust visual effects accordingly
                if detect_beat(data, BEAT_THRESHOLD_MULTIPLIER):
                    rotation_speed = ROTATION_SPEED_BOOST
                    scale_factor = SPOKE_SCALE_FACTOR
                else:
                    rotation_speed += (ROTATION_SPEED_BASE - rotation_speed) * 0.1  # Gradually return to base speed
                    scale_factor = 1.0

                # Detect large frequency shifts for color burst
                if detect_large_frequency_shift(data):
                    burst_active = True
                    burst_timer = BURST_DURATION
                    rotation_angle += random.uniform(-30, 30)  # Random shift in rotation

                # If burst is active, intensify colors and scale
                if burst_active:
                    scale_factor = BURST_SCALE_FACTOR
                    if BURST_COLOR_CHANGE:
                        rainbow_offset += 0.1  # Rapid color shifts during burst

                    burst_timer -= dt
                    if burst_timer <= 0:
                        burst_active = False

                # Detect high frequency event and trigger white flash
                if detect_high_frequency_event(bars) and not flash_active and flash_cooldown_timer <= 0:
                    flash_active = True
                    flash_timer = WHITE_FLASH_DURATION
                    flash_cooldown_timer = WHITE_FLASH_COOLDOWN

                # Decrease flash timer and cooldown
                if flash_active:
                    flash_timer -= dt
                    if flash_timer <= 0:
                        flash_active = False

                flash_cooldown_timer = max(0, flash_cooldown_timer - dt)

                # Draw bottom layer of spokes (thicker, slower)
                draw_rotating_spokes(circle_surface, bars, center, min(WINDOW_WIDTH, WINDOW_HEIGHT) // 2 - 50, 
                                     MAX_SPOKES_BOTTOM, rotation_angle, BASE_SPOKE_SIZE_BOTTOM, 255, scale_factor)

                # Draw top layer of spokes (thinner, faster)
                draw_rotating_spokes(circle_surface, bars, center, min(WINDOW_WIDTH, WINDOW_HEIGHT) // 2 - 50, 
                                     MAX_SPOKES_TOP, rotation_angle * 1.2, BASE_SPOKE_SIZE_TOP, 180, scale_factor)

                # Draw random black lines on top layer
                draw_random_black_lines(circle_surface, center, NUM_RANDOM_LINES, max(WINDOW_WIDTH, WINDOW_HEIGHT) // 3, 50)

                # Apply white flash if active
                if flash_active:
                    flash_surface = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
                    flash_surface.fill((255, 255, 255))
                    flash_surface.set_alpha(200)  # Adjust intensity of the flash
                    window.blit(flash_surface, (0, 0))

                # Blit the surface onto the window
                window.blit(circle_surface, (0, 0))

                # Update rotation angle based on speed
                rotation_angle += rotation_speed * dt * 60

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

