import numpy as np
import pyaudio
import pygame
from scipy.fftpack import fft

# Pygame setup
pygame.init()

# Constants
CHUNK = 1024
RATE = 44100
NUM_BARS = 10
BAR_SEGMENTS = 20
GHOST_DECAY = 0.03
SEGMENT_HEIGHT_RATIO = 0.9
AMPLITUDE_SCALING = 0.6

# Sensitivity Controls
LOW_FREQ_BOOST = 1.2
MID_FREQ_BOOST = 1.5
HIGH_FREQ_BOOST = 2.2
MOVEMENT_SCALING = 2.0
SILENCE_THRESHOLD = 15

# PyAudio setup
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16,
                channels=2,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

# Initialize previous bar heights for ghosting effect
previous_bar_heights = np.zeros(NUM_BARS)
ghost_heights = np.zeros(NUM_BARS)

# Color schemes based on the provided image
def get_gradient_color_1(position, max_height):
    ratio = position / max_height
    if ratio < 0.5:
        return (255, int(255 * ratio * 2), 0)
    else:
        return (int(255 * (1 - ratio) * 2), 0, int(255 * (ratio - 0.5) * 2))

def get_gradient_color_2(position, max_height):
    ratio = position / max_height
    return (int(255 * (1 - ratio)), 255, 255)

def get_rainbow_color(position, max_height):
    hue = (position / max_height) * 360
    color = pygame.Color(0)
    color.hsva = (hue, 100, 100)
    return (color.r, color.g, color.b)

def get_solid_orange_color():
    return (255, 165, 0)

# Function to convert FFT data to bar heights with sensitivity adjustments
def get_frequency_bars(data, num_bars, max_height):
    fft_data = np.abs(fft(data))[:CHUNK // 2]  # FFT and keep only positive frequencies
    bar_heights = np.zeros(num_bars)

    # Apply logarithmic scaling to distribute frequencies across the bars more evenly
    freq_bins = np.logspace(0, np.log10(len(fft_data)), num_bars, dtype=int)

    # Return no movement if no sound is detected
    if np.mean(np.abs(data)) < SILENCE_THRESHOLD:
        return np.zeros(num_bars, dtype=int)

    for i in range(num_bars):
        if i == 0:
            avg_value = np.mean(fft_data[:freq_bins[i]])  # For the first bar
        else:
            avg_value = np.mean(fft_data[freq_bins[i-1]:freq_bins[i]])

        # Apply sensitivity boost based on the frequency range
        if i < num_bars // 3:  # Low frequencies (bass)
            sensitivity_boost = LOW_FREQ_BOOST
        elif i < 2 * num_bars // 3:  # Mid frequencies
            sensitivity_boost = MID_FREQ_BOOST
        else:  # High frequencies (treble)
            sensitivity_boost = HIGH_FREQ_BOOST

        # Adjust bar height by sensitivity boost and overall movement scaling
        bar_heights[i] = avg_value * sensitivity_boost * AMPLITUDE_SCALING

    # Apply non-linear scaling for enhanced movement
    bar_heights = np.power(bar_heights, MOVEMENT_SCALING)

    # Normalize bar heights to fit within the window
    max_value = np.max(bar_heights)
    if max_value > 0:
        bar_heights = np.clip(bar_heights / max_value * max_height, 0, max_height).astype(int)
    else:
        bar_heights = np.zeros(num_bars, dtype=int)

    # Apply smoothing to bar heights (ghosting effect)
    global previous_bar_heights
    smoothed_bar_heights = (0.6 * previous_bar_heights + 0.4 * bar_heights)
    previous_bar_heights = smoothed_bar_heights

    return smoothed_bar_heights.astype(int)

# Main loop for real-time visualization with ghosting effect
def main():
    global ghost_heights
    global WINDOW_WIDTH, WINDOW_HEIGHT
    WINDOW_WIDTH, WINDOW_HEIGHT = 800, 600
    window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.RESIZABLE)
    pygame.display.set_caption("Sound Visualizer")
    rainbow_offset = 0
    fullscreen = False

    # Available color modes
    color_modes = ['gradient_1', 'gradient_2', 'rainbow', 'solid_orange']
    current_color_mode = 0

    try:
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.VIDEORESIZE:
                    WINDOW_WIDTH, WINDOW_HEIGHT = event.w, event.h
                    window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.RESIZABLE)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_F11:
                        fullscreen = not fullscreen
                        if fullscreen:
                            info = pygame.display.Info()  # Get the display resolution
                            WINDOW_WIDTH, WINDOW_HEIGHT = info.current_w, info.current_h
                            window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.FULLSCREEN)
                        else:
                            # Reset back to maximized mode from full screen
                            WINDOW_WIDTH, WINDOW_HEIGHT = 800, 600
                            window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.RESIZABLE)
                            pygame.display.toggle_fullscreen()  # Ensures smooth toggle when leaving maximized
                    elif event.key == pygame.K_LEFT:
                        current_color_mode = (current_color_mode - 1) % len(color_modes)
                    elif event.key == pygame.K_RIGHT:
                        current_color_mode = (current_color_mode + 1) % len(color_modes)

            # Ensure that exactly 10 bars are always displayed
            BAR_WIDTH = WINDOW_WIDTH // NUM_BARS

            global ghost_heights, previous_bar_heights
            if len(ghost_heights) != NUM_BARS:
                ghost_heights = np.zeros(NUM_BARS)
                previous_bar_heights = np.zeros(NUM_BARS)

            # Clear the window with a black background
            window.fill((0, 0, 0))

            # Read audio data from PyAudio stream
            data = np.frombuffer(stream.read(CHUNK), dtype=np.int16)

            # Get frequency bars
            bars = get_frequency_bars(data, NUM_BARS, WINDOW_HEIGHT)

            # Update the ghost effect
            ghost_heights = np.maximum(ghost_heights - GHOST_DECAY * WINDOW_HEIGHT, bars)

            # Draw the bars with selected color scheme
            for i, bar_height in enumerate(bars):
                x = i * BAR_WIDTH
                if color_modes[current_color_mode] == 'gradient_1':
                    color = get_gradient_color_1(bar_height, WINDOW_HEIGHT)
                elif color_modes[current_color_mode] == 'gradient_2':
                    color = get_gradient_color_2(bar_height, WINDOW_HEIGHT)
                elif color_modes[current_color_mode] == 'rainbow':
                    color = get_rainbow_color(bar_height, WINDOW_HEIGHT)
                else:
                    color = get_solid_orange_color()

                # Ensure valid color tuple
                color = tuple(max(0, min(255, c)) for c in color)

                # Draw the segments of the bar (with ghosting effect)
                segment_height = WINDOW_HEIGHT // BAR_SEGMENTS
                for j in range(BAR_SEGMENTS):
                    segment_top = WINDOW_HEIGHT - j * segment_height
                    segment_bottom = segment_top - int(segment_height * SEGMENT_HEIGHT_RATIO)

                    if segment_bottom < WINDOW_HEIGHT - ghost_heights[i]:
                        break

                    pygame.draw.rect(window, color, (x, segment_bottom, BAR_WIDTH - 2, segment_height * SEGMENT_HEIGHT_RATIO))

            # Update the display
            pygame.display.flip()

            # Control the frame rate
            pygame.time.Clock().tick(60)

    except KeyboardInterrupt:
        # Cleanup
        stream.stop_stream()
        stream.close()
        p.terminate()
        pygame.quit()

if __name__ == "__main__":
    main()

