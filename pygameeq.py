import numpy as np
import pyaudio
import pygame
from scipy.fftpack import fft
import sys
import random

# Initialize Pygame
pygame.init()

# Constants
CHUNK = 2048
RATE = 44100  # Sample rate
CHANNELS = 2  # Stereo input
SILENCE_THRESHOLD = 20  # Threshold to detect silence
LOW_FREQ_BOOST = 1.0
MID_FREQ_BOOST = 1.0
HIGH_FREQ_BOOST = 1.0
MOVEMENT_SCALING = 5.0
RMS_DECAY = 0.2  # Faster decay for bar heights

# Global variables
rainbow_offset = 0
paused = False
silence_counter = 0
required_silence_frames = 5  # Number of frames to consider as silence
rms_smooth = 7  # Smoothed RMS value
previous_bar_heights = np.zeros(10)  # Will adjust dynamically

# PyAudio Setup
p = pyaudio.PyAudio()

try:
    stream = p.open(format=pyaudio.paInt16,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    print(f"Opened audio stream with sample rate {RATE} and {CHANNELS} channel(s).")
except Exception as e:
    print(f"Could not open audio stream: {e}")
    p.terminate()
    sys.exit()

# Define Color Mode Functions
def rainbow_color(index, total_bars, offset):
    hue = (index / total_bars + offset) % 1.0
    color = pygame.Color(0)
    color.hsva = (hue * 360, 100, 100)
    return (color.r, color.g, color.b)

def monochrome_color(index, total_bars, offset):
    return (255, 255, 255)

def blue_to_red_gradient(index, total_bars, offset):
    ratio = index / total_bars
    return (int(255 * ratio), 0, int(255 * (1 - ratio)))

def green_spectrum_color(index, total_bars, offset):
    return (0, 255, 0)

def red_spectrum_color(index, total_bars, offset):
    return (255, 0, 0)

def blue_spectrum_color(index, total_bars, offset):
    return (0, 0, 255)

def orange_purple_alternating(index, total_bars, offset):
    colors = [(255, 165, 0), (128, 0, 128)]  # Orange and Purple
    return colors[index % 2]

def random_color(index, total_bars, offset):
    random.seed(index + offset)
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

def custom_color_mode(index, total_bars, offset):
    # Define your custom colors here
    return (255, 165, 0)  # Bright Orange

# List of color modes
color_modes = [
    rainbow_color,
    monochrome_color,
    blue_to_red_gradient,
    green_spectrum_color,
    red_spectrum_color,
    blue_spectrum_color,
    orange_purple_alternating,
    random_color,
    custom_color_mode,
]

color_mode_names = [
    "Rainbow",
    "Monochrome",
    "Blue to Red Gradient",
    "Green Spectrum",
    "Red Spectrum",
    "Blue Spectrum",
    "Orange and Purple Alternating",
    "Random Colors",
    "Custom Color Mode",
]

current_color_mode = 0

# Function to calculate frequency bars from audio data
def get_frequency_bars(data, num_bars, max_height):
    global previous_bar_heights, silence_counter, rms_smooth

    # Calculate RMS and smooth it
    rms = np.sqrt(np.mean(np.square(data)))
    rms_smooth = (rms_smooth * 0.8) + (rms * 0.2)  # Faster response to changes

    # Silence detection based on adjusted rms_smooth
    if rms_smooth < SILENCE_THRESHOLD:
        silence_counter += 1
        if silence_counter >= required_silence_frames:
            previous_bar_heights = np.zeros_like(previous_bar_heights)
            return previous_bar_heights
    else:
        silence_counter = 0

    # FFT Processing
    fft_data = np.abs(fft(data))[:CHUNK // 2]
    bar_heights = np.zeros(num_bars)

    # Logarithmic frequency bins
    log_bins = np.logspace(np.log10(20), np.log10(RATE / 2), num=num_bars + 1)
    bin_indices = np.clip(
        np.floor((log_bins / (RATE / 2)) * (CHUNK // 2)).astype(int), 0, CHUNK // 2)

    for i in range(num_bars):
        avg_value = np.mean(fft_data[bin_indices[i]:bin_indices[i + 1]])
        center_freq = np.sqrt(log_bins[i] * log_bins[i + 1])  # Geometric mean
        if center_freq < 200:
            sensitivity_boost = LOW_FREQ_BOOST
        elif center_freq <= 600:
            sensitivity_boost = MID_FREQ_BOOST
        else:
            sensitivity_boost = HIGH_FREQ_BOOST
        bar_heights[i] = avg_value * sensitivity_boost

    # Non-linear scaling and normalization
    bar_heights = np.power(bar_heights, MOVEMENT_SCALING)
    max_value = np.max(bar_heights)
    if max_value > 0:
        bar_heights = np.clip(
            bar_heights / max_value * max_height, 0, max_height)
    else:
        bar_heights = np.zeros(num_bars)

    # Apply smoothing to bar heights (ghosting effect)
    smoothed_bar_heights = (0.6 * previous_bar_heights + 0.4 * bar_heights)
    previous_bar_heights = smoothed_bar_heights

    return smoothed_bar_heights

# Function to draw notched bars
def draw_notched_bar(surface, x, y, width, height, color, notch_height=5):
    y = int(y)
    height = int(height)
    notch_height = int(notch_height)
    width = int(width)
    for notch_y in range(y, y - height, -notch_height - 2):
        pygame.draw.rect(surface, color, (x, notch_y - notch_height, width, notch_height))

# Main loop
def main():
    global current_color_mode, rainbow_offset, paused, silence_counter, rms_smooth, previous_bar_heights

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

    font = pygame.font.SysFont(None, 24)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # Handle key presses
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_F11:
                    pygame.display.toggle_fullscreen()
                elif event.key == pygame.K_RIGHT:
                    current_color_mode = (
                        current_color_mode + 1) % len(color_modes)
                elif event.key == pygame.K_LEFT:
                    current_color_mode = (
                        current_color_mode - 1) % len(color_modes)
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
            data = np.frombuffer(stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16)

            # Convert stereo to mono if necessary
            if CHANNELS == 2:
                data = data.reshape(-1, 2)
                data = data.mean(axis=1).astype(np.int16)

            bars = get_frequency_bars(data, NUM_BARS, WINDOW_HEIGHT)

            # Clear the window with a black background
            window.fill((0, 0, 0))

            # Draw the bars with the selected color scheme
            bar_width = WINDOW_WIDTH / NUM_BARS
            for i, bar_height in enumerate(bars):
                x = int(i * bar_width)
                color = color_modes[current_color_mode](
                    i, NUM_BARS, rainbow_offset)
                draw_notched_bar(window, x, WINDOW_HEIGHT,
                                 int(bar_width - 2), int(bar_height), color)

            # Display the current color mode name
            #color_mode_text = font.render(
               #f"Mode: {color_mode_names[current_color_mode]}", True, (255, 255, 255))
            #window.blit(color_mode_text, (10, 10))

        pygame.display.flip()
        rainbow_offset += 0.01
        clock.tick(60)

    # Cleanup on exit
    stream.stop_stream()
    stream.close()
    p.terminate()
    pygame.quit()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # Ensure resources are cleaned up on exit
        stream.stop_stream()
        stream.close()
        p.terminate()
        pygame.quit()
        sys.exit()

