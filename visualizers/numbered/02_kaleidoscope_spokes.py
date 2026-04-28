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
            if np.max(fft_magnitude) != 0:
                bar_height = int((avg_magnitude / np.max(fft_magnitude)) * window_height)
            else:
                bar_height = 0
            bar_heights.append(bar_height)
        else:
            bar_heights.append(0)
    return bar_heights

# Function to draw notched bars within the kaleidoscope (Retained for optional use)
def draw_notched_bar(surface, center, angle, length, width, color, num_notches=5, notch_size=5):
    """
    Draws a notched bar (segment) in a specific direction within the kaleidoscope.

    Parameters:
        surface (pygame.Surface): The main surface to draw on.
        center (tuple): The (x, y) coordinates of the center of the kaleidoscope.
        angle (float): The angle in degrees where the bar is drawn.
        length (int): The length of the bar.
        width (int): The thickness of the bar.
        color (tuple): The RGB color of the bar.
        num_notches (int): Number of notches along the bar.
        notch_size (int): Size of each notch.
    """
    # Calculate end position
    rad = math.radians(angle)
    end_x = center[0] + length * math.cos(rad)
    end_y = center[1] - length * math.sin(rad)

    # Calculate notches
    for i in range(num_notches):
        # Position of the notch
        interp = (i + 1) / (num_notches + 1)
        notch_x = center[0] + (length * interp) * math.cos(rad)
        notch_y = center[1] - (length * interp) * math.sin(rad)

        # Draw notch as a small line perpendicular to the bar
        perp_rad = rad + math.pi / 2
        notch_end_x = notch_x + notch_size * math.cos(perp_rad)
        notch_end_y = notch_y - notch_size * math.sin(perp_rad)

        pygame.draw.line(surface, color, (notch_x, notch_y), (notch_end_x, notch_end_y), width=1)

# Function to draw the rotating spokes
def draw_rotating_spokes(surface, bars, center, max_radius, num_spokes=120, rotation_angle=0):
    """
    Draws numerous rotating spokes emanating from the center based on frequency bars.

    Parameters:
        surface (pygame.Surface): The main surface to draw on.
        bars (list): List of frequency bar heights.
        center (tuple): The (x, y) coordinates of the center of the visualization.
        max_radius (int): The maximum radius of the visualization.
        num_spokes (int): Total number of spokes to draw.
        rotation_angle (float): The rotation angle in degrees.
    """
    angle_between_spokes = 360 / num_spokes  # Angle between each spoke

    for i in range(num_spokes):
        # Calculate the current angle for this spoke
        angle = i * angle_between_spokes + rotation_angle

        # Map spoke index to frequency band
        bar_index = i % len(bars)
        bar_height = bars[bar_index]

        # Scale the length of the spoke based on the frequency magnitude
        spoke_length = max_radius * (bar_height / max(bars)) if max(bars) != 0 else 0
        spoke_length = min(spoke_length, max_radius)

        # Determine the end position of the spoke
        rad = math.radians(angle)
        end_x = center[0] + spoke_length * math.cos(rad)
        end_y = center[1] - spoke_length * math.sin(rad)

        # Choose color based on color mode
        color = color_modes[current_color_mode](bar_index, len(bars), rainbow_offset)

        # Draw the spoke as a line
        pygame.draw.line(surface, color, center, (end_x, end_y), width=1)

# Function to draw circles based on frequency data (Optional Feature)
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
        pygame.draw.circle(surface, faded_color, circle['position'], int(circle['radius']), width=2)

# Main loop
def main():
    global current_color_mode, rainbow_offset, paused, silence_counter, rms_smooth, previous_bar_heights, circle_list, prev_bass_amplitude

    # Initialize the window size
    WINDOW_WIDTH, WINDOW_HEIGHT = 1200, 900  # Increased window size for a larger visualization
    window = pygame.display.set_mode(
        (WINDOW_WIDTH, WINDOW_HEIGHT), pygame.RESIZABLE | pygame.NOFRAME)

    # Initialize number of bars based on window size
    NUM_BARS = max(10, WINDOW_WIDTH // 40)  # Increased minimum bars to 10
    NUM_BARS = min(NUM_BARS, 40)           # Increased maximum bars to 40
    previous_bar_heights = np.zeros(NUM_BARS)

    # Create a semi-transparent surface for ghosting effect
    ghost_surface = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
    ghost_surface = ghost_surface.convert_alpha()
    ghost_surface.fill((0, 0, 0, 25))  # Adjust alpha for fade speed (0-255)

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
                    NUM_BARS = max(10, WINDOW_WIDTH // 40)  # Adjusted for more bars
                    NUM_BARS = min(NUM_BARS, 40)           # Increased maximum bars
                    previous_bar_heights = np.zeros(NUM_BARS)

                    # Resize ghost_surface
                    ghost_surface = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
                    ghost_surface = ghost_surface.convert_alpha()
                    ghost_surface.fill((0, 0, 0, 25))  # Adjust alpha as needed

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
                bars = get_frequency_bars(data, NUM_BARS, max(WINDOW_WIDTH, WINDOW_HEIGHT) // 2)  # Increased max_radius

                # ----------------------------------------------
                # Apply Ghosting Effect
                # ----------------------------------------------
                # Blit the ghost_surface onto the main window to fade previous drawings
                window.blit(ghost_surface, (0, 0))
                # ----------------------------------------------

                # ----------------------------------------------
                # Draw the Rotating Spokes Effect
                # ----------------------------------------------
                # Define center and maximum radius
                center = (WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2)
                max_radius = min(WINDOW_WIDTH, WINDOW_HEIGHT) // 2  # Increased radius to take up more screen

                # Define number of spokes
                num_spokes = 120  # Increased number of spokes for density

                # Define rotation angle (dynamic rotation)
                rotation_angle = rainbow_offset * 10  # Adjust speed as desired

                # Draw the rotating spokes
                draw_rotating_spokes(window, bars, center, max_radius, num_spokes, rotation_angle)
                # ----------------------------------------------

                # Optionally, draw circles (Retain if desired)
                # draw_circles(window, bars)
            else:
                # If paused, overlay a translucent surface to indicate pause
                pause_overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
                pause_overlay.fill((0, 0, 0, 100))  # Semi-transparent black
                window.blit(pause_overlay, (0, 0))

            # Update the ghost_surface by filling it with a semi-transparent black
            # to create a fading trail effect
            ghost_surface.fill((0, 0, 0, 25))  # Adjust alpha (0-255) for fade speed

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

