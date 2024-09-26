import numpy as np
import pyaudio
import pygame
from scipy.fftpack import fft
import sys
import random
import math
import colorsys

# Initialize Pygame
pygame.init()

# Constants
CHUNK = 1024  # Adjusted for balance between responsiveness and buffer management
RATE = 44100  # Sample rate
CHANNELS = 2  # Stereo input
SILENCE_THRESHOLD = 20  # Threshold to detect silence
LOW_FREQ_BOOST = 1.0
MID_FREQ_BOOST = 1.0
HIGH_FREQ_BOOST = 1.0
MOVEMENT_SCALING = 5.0
RMS_DECAY = 0.2  # Faster decay for bar heights
MAX_CIRCLES = 15  # Maximum number of active circles
BEAT_THRESHOLD = 1000  # Threshold to trigger new circles

# Global variables
rainbow_offset = 0
paused = False
silence_counter = 0
required_silence_frames = 5  # Number of frames to consider as silence
rms_smooth = 7  # Smoothed RMS value
previous_bar_heights = np.zeros(10)  # Will adjust dynamically
circle_list = []  # List to hold active circles
prev_bass_amplitude = 0  # For smoothing bass amplitude

# PyAudio Setup
p = pyaudio.PyAudio()

def get_input_device_index(desired_device_name=None):
    """
    Find the index of the desired input device.
    If desired_device_name is None, return the first available input device.
    """
    info = p.get_host_api_info_by_index(0)
    num_devices = info.get('deviceCount')
    for i in range(num_devices):
        device_info = p.get_device_info_by_host_api_device_index(0, i)
        if device_info.get('maxInputChannels') > 0:
            if desired_device_name:
                if desired_device_name.lower() in device_info.get('name').lower():
                    print(f"Selected input device: {device_info.get('name')}")
                    return i
            else:
                print(f"Selected input device: {device_info.get('name')}")
                return i
    return None

# You can specify your desired input device name here
desired_input_device = None  # e.g., "ALC295 Analog"

input_device_index = get_input_device_index(desired_input_device)

# Open audio stream with specified method
try:
    stream = p.open(format=pyaudio.paInt16,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    input_device_index=input_device_index,
                    frames_per_buffer=CHUNK)
    print(f"Opened audio stream with sample rate {RATE} and {CHANNELS} channel(s).")
except Exception as e:
    print(f"Could not open audio stream: {e}")
    p.terminate()
    sys.exit()

# Define Color Mode Functions for Frequency Bars
def rainbow_color(index, total_bars, offset):
    hue = (index / total_bars + offset) % 1.0
    color = pygame.Color(0)
    color.hsva = (hue * 360, 100, 100, 100)
    return (color.r, color.g, color.b)

def grayscale_color(index, total_bars, offset):
    intensity = int(255 * ((index / total_bars) + offset) % 1.0)
    return (intensity, intensity, intensity)

def random_color(index, total_bars, offset):
    random.seed(index + int(offset * 100))
    return (random.randint(0,255), random.randint(0,255), random.randint(0,255))

# Add more color modes as needed
color_modes = [rainbow_color, grayscale_color, random_color]
current_color_mode = 0  # Initialize to the first color mode

# Function to get frequency bars from audio data
def get_frequency_bars(data, num_bars, window_height):
    """
    Processes the audio data and returns a list of bar heights corresponding to frequency bands.
    """
    # Apply FFT
    fft_data = fft(data)
    fft_magnitude = np.abs(fft_data)[:CHUNK//2]

    # Frequency resolution
    freq_bins = np.linspace(0, RATE/2, CHUNK//2)

    # Define frequency bands (e.g., logarithmic scale)
    freq_band_limits = np.logspace(np.log10(20), np.log10(RATE/2), num_bars + 1)
    bar_heights = []

    for i in range(num_bars):
        # Find indices for the current band
        idx = np.where((freq_bins >= freq_band_limits[i]) & (freq_bins < freq_band_limits[i+1]))[0]
        if len(idx) > 0:
            # Compute the average magnitude for the band
            avg_magnitude = np.mean(fft_magnitude[idx])
            # Normalize and scale the magnitude to fit the window height
            bar_height = int((avg_magnitude / np.max(fft_magnitude)) * window_height)
            bar_heights.append(bar_height)
        else:
            bar_heights.append(0)
    return bar_heights

# Function to draw a notched bar
def draw_notched_bar(surface, x, base_y, width, height, color):
    """
    Draws a bar with notches for visual effect.
    """
    notch_height = 10
    notch_spacing = 20
    num_notches = height // notch_spacing
    for i in range(num_notches):
        notch_y = base_y - (i + 1) * notch_spacing
        pygame.draw.rect(surface, color, (x, notch_y, width, notch_height))
    # Draw the main bar
    pygame.draw.rect(surface, color, (x, base_y - height, width, height % notch_spacing))

# Function to draw circles based on frequency data
def draw_circles(surface, bars):
    """
    Draws circles that react to the bass (low-frequency) amplitude.
    """
    global circle_list, prev_bass_amplitude

    # Assuming the first bar corresponds to the lowest frequencies (bass)
    bass_amplitude = bars[0]

    # Smooth the bass amplitude
    smoothed_bass = prev_bass_amplitude * 0.9 + bass_amplitude * 0.1
    prev_bass_amplitude = smoothed_bass

    # Add new circles based on beat threshold
    if smoothed_bass > BEAT_THRESHOLD and len(circle_list) < MAX_CIRCLES:
        circle = {
            'position': [random.randint(0, surface.get_width()), random.randint(0, surface.get_height())],
            'radius': 0,
            'max_radius': smoothed_bass / 10,
            'color': random.choice(color_modes)(0, len(bars), rainbow_offset)
        }
        circle_list.append(circle)

    # Update and draw circles
    for circle in circle_list[:]:
        circle['radius'] += 2  # Expand the circle
        if circle['radius'] > circle['max_radius']:
            circle_list.remove(circle)
            continue
        # Fade the circle color
        faded_color = tuple(max(0, min(255, c - 2)) for c in circle['color'])
        pygame.draw.circle(surface, faded_color, circle['position'], int(circle['radius']), 2)

# Main loop
def main():
    global current_color_mode, rainbow_offset, paused, silence_counter, rms_smooth, previous_bar_heights, circle_list, prev_bass_amplitude

    # Initialize the window size
    WINDOW_WIDTH, WINDOW_HEIGHT = 800, 600
    window = pygame.display.set_mode(
        (WINDOW_WIDTH, WINDOW_HEIGHT), pygame.RESIZABLE | pygame.NOFRAME)

    # Initialize number of bars based on window size
    NUM_BARS = max(5, WINDOW_WIDTH // 80)  # At least 5 bars
    NUM_BARS = min(NUM_BARS, 20)           # Limit to 20 bars
    previous_bar_heights = np.zeros(NUM_BARS)

    clock = pygame.time.Clock()
    running = True

    print("Starting main loop.")

    try:
        while running:
            dt = clock.tick(60) / 1000.0  # FPS capped at 60

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                # Handle key presses
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_F11:
                        pygame.display.toggle_fullscreen()
                    elif event.key == pygame.K_RIGHT:
                        current_color_mode = (current_color_mode + 1) % len(color_modes)
                    elif event.key == pygame.K_LEFT:
                        current_color_mode = (current_color_mode - 1) % len(color_modes)
                    elif event.key == pygame.K_SPACE:
                        paused = not paused

                elif event.type == pygame.VIDEORESIZE:
                    WINDOW_WIDTH, WINDOW_HEIGHT = event.size
                    window = pygame.display.set_mode(
                        (WINDOW_WIDTH, WINDOW_HEIGHT), pygame.RESIZABLE | pygame.NOFRAME)
                    NUM_BARS = max(5, WINDOW_WIDTH // 80)
                    NUM_BARS = min(NUM_BARS, 20)
                    previous_bar_heights = np.zeros(NUM_BARS)

            if not paused:
                # Read audio data
                try:
                    data = np.frombuffer(stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16)
                except IOError as e:
                    print(f"Audio buffer overflow: {e}. Skipping this frame.")
                    continue

                # Convert stereo to mono if necessary
                if CHANNELS == 2:
                    data = data.reshape(-1, 2)
                    data = data.mean(axis=1).astype(np.int16)

                # Calculate frequency bars
                bars = get_frequency_bars(data, NUM_BARS, WINDOW_HEIGHT)

                # Clear the window with a black background
                window.fill((0, 0, 0))

                # Draw the bars with the selected color scheme
                bar_width = WINDOW_WIDTH / NUM_BARS
                for i, bar_height in enumerate(bars):
                    x = int(i * bar_width)
                    color = color_modes[current_color_mode](i, NUM_BARS, rainbow_offset)
                    draw_notched_bar(window, x, WINDOW_HEIGHT, int(bar_width - 2), int(bar_height), color)

                # Draw all circles
                draw_circles(window, bars)

            pygame.display.flip()
            rainbow_offset += 0.01

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        # Cleanup on exit
        stream.stop_stream()
        stream.close()
        p.terminate()
        pygame.quit()
        print("Audio stream closed and Pygame quit.")

if __name__ == "__main__":
    main()

