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
CHANNELS = 2
MAX_CIRCLES = 15
BEAT_THRESHOLD_MULTIPLIER = 1.3  # Multiplier for dynamic threshold
ENERGY_HISTORY = deque(maxlen=43)  # Approximately 1 second of history at ~43 FPS

# Global variables
rainbow_offset = 0
paused = False
circle_list = []
rotation_angle = 0
rotation_speed = 0.5  # Base rotation speed
is_fullscreen = False  # Track full-screen state

# PyAudio Setup
p = pyaudio.PyAudio()

def list_audio_devices():
    """Lists all available audio input devices."""
    print("Available audio input devices:")
    for i in range(p.get_device_count()):
        device_info = p.get_device_info_by_index(i)
        if device_info['maxInputChannels'] > 0:
            print(f"{i}: {device_info['name']}")

def get_input_device_index(desired_input_device=None):
    """
    Retrieves the index of the desired audio input device.
    If none is specified, returns the default input device index.
    """
    if desired_input_device is None:
        try:
            return p.get_default_input_device_info()['index']
        except IOError:
            print("No default input device found.")
            sys.exit(1)
    for i in range(p.get_device_count()):
        device_info = p.get_device_info_by_index(i)
        if desired_input_device.lower() in device_info['name'].lower():
            return i
    print(f"Desired input device '{desired_input_device}' not found.")
    sys.exit(1)

# Optional: Uncomment to list audio devices
# list_audio_devices()

# Specify your desired input device name here (replace with actual device name)
# desired_input_device_name = "Your Device Name Here"
# input_device_index = get_input_device_index(desired_input_device=desired_input_device_name)

# Input device can be None to use the default
input_device_index = get_input_device_index()

# Try multiple sample rates to open the audio stream
def open_audio_stream():
    sample_rates_to_try = [44100, 48000, 22050, 16000, 11025, 8000]
    for rate in sample_rates_to_try:
        try:
            stream = p.open(format=pyaudio.paInt16,
                            channels=CHANNELS,
                            rate=rate,
                            input=True,
                            input_device_index=input_device_index,
                            frames_per_buffer=CHUNK)
            print(f"Opened audio stream with sample rate: {rate}")
            return stream, rate
        except Exception as e:
            print(f"Could not open audio stream with sample rate {rate}: {e}")
    print("Failed to open audio stream. Exiting.")
    p.terminate()
    sys.exit(1)

stream, RATE = open_audio_stream()

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

# Fonts
pygame.font.init()
FONT_SMALL = pygame.font.SysFont('Arial', 16)
FONT_LARGE = pygame.font.SysFont('Arial', 24)

# Function to create a triangular mask
def create_triangular_mask(slice_angle, radius):
    """
    Creates a triangular mask surface for a given slice angle and radius.
    """
    # Create a transparent surface
    mask_surface = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
    # Define the triangle points
    center = (radius, radius)
    start_rad = math.radians(slice_angle)
    end_rad = math.radians(slice_angle + 22.5)
    x1 = center[0] + radius * math.cos(start_rad)
    y1 = center[1] + radius * math.sin(start_rad)
    x2 = center[0] + radius * math.cos(end_rad)
    y2 = center[1] + radius * math.sin(end_rad)
    pygame.draw.polygon(mask_surface, (255, 255, 255, 255), [center, (x1, y1), (x2, y2)])
    return mask_surface

# Function to load pattern images
def load_pattern_images():
    """
    Loads the grey_slice.png and dark_slice.png images.
    dark_slice.png is a horizontally mirrored version of grey_slice.png.
    """
    try:
        grey_slice_img = pygame.image.load('grey_slice.png').convert_alpha()
        dark_slice_img = pygame.transform.flip(grey_slice_img, True, False)  # Horizontally mirror
        return grey_slice_img, dark_slice_img
    except pygame.error as e:
        print(f"Error loading pattern images: {e}")
        pygame.quit()
        sys.exit(1)

# Define color mapping functions
def get_frequency_bars(data, num_bars, window_radius):
    """
    Processes audio data to extract frequency bars using FFT.
    """
    fft_data = fft(data)
    fft_magnitude = np.abs(fft_data)[:CHUNK//2]
    freq_bins = np.linspace(0, RATE/2, CHUNK//2)
    freq_band_limits = np.logspace(np.log10(20), np.log10(RATE/2), num_bars + 1)

    bar_heights = []
    for i in range(num_bars):
        idx = np.where((freq_bins >= freq_band_limits[i]) & (freq_bins < freq_band_limits[i+1]))[0]
        if len(idx) > 0:
            avg_magnitude = np.mean(fft_magnitude[idx])
            bar_height = avg_magnitude
            bar_heights.append(bar_height)
        else:
            bar_heights.append(0)
    return bar_heights

# Function to map frequency bars to slices
def map_bars_to_slices(bars, num_slices):
    """
    Maps frequency bars to kaleidoscope slices.
    If there are fewer bars than slices, it repeats the bars.
    """
    slice_values = []
    for i in range(num_slices):
        slice_values.append(bars[i % len(bars)])
    return slice_values

# Draw kaleidoscope slices with patterns
def draw_kaleidoscope(surface, center, radius, slice_values, grey_img, dark_img):
    """
    Draws the kaleidoscope visualization with patterned triangular slices.
    """
    num_slices = 16
    slice_angle = 360 / num_slices  # 22.5 degrees

    for i in range(num_slices):
        start_angle = i * slice_angle
        end_angle = (i + 1) * slice_angle

        # Determine transformation type and corresponding image
        if i % 2 == 0:
            # Grey Slice
            pattern_img = grey_img
            transformation = "Mirror 22.5°"
        else:
            # Dark Slice
            pattern_img = dark_img
            transformation = "Rotation 22.5°"

        # Keep transformation_angle fixed to maintain stability
        transformation_angle = 0  # No dynamic transformation

        # Calculate the points of the triangle
        start_rad = math.radians(start_angle + transformation_angle)
        end_rad = math.radians(end_angle + transformation_angle)

        # Calculate the triangle points
        x1 = center[0] + radius * math.cos(start_rad)
        y1 = center[1] + radius * math.sin(start_rad)
        x2 = center[0] + radius * math.cos(end_rad)
        y2 = center[1] + radius * math.sin(end_rad)

        # Define the triangle
        triangle = [center, (x1, y1), (x2, y2)]

        # Create a mask for the slice
        mask = create_triangular_mask(start_angle + transformation_angle, radius)
        mask_rect = mask.get_rect(center=(center[0], center[1]))

        # Blit the pattern image onto the mask
        pattern_scaled = pygame.transform.scale(pattern_img, (radius*2, radius*2))
        masked_pattern = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
        masked_pattern.blit(pattern_scaled, (0, 0))
        masked_pattern.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)

        # Blit the masked pattern onto the main surface
        surface.blit(masked_pattern, (center[0] - radius, center[1] - radius), special_flags=pygame.BLEND_RGBA_ADD)

        # Draw the triangle outline (optional for better visibility)
        pygame.draw.polygon(surface, (50, 50, 50), triangle, 1)

        # Draw the label
        mid_angle = (start_angle + end_angle) / 2 + transformation_angle
        mid_rad = math.radians(mid_angle)
        label_radius = radius * 0.6
        label_x = center[0] + label_radius * math.cos(mid_rad)
        label_y = center[1] + label_radius * math.sin(mid_rad)

        label_text = FONT_SMALL.render(transformation, True, (255, 255, 255))
        label_rect = label_text.get_rect(center=(label_x, label_y))
        surface.blit(label_text, label_rect)

# Draw the mathematical equation at the bottom
def draw_equation(surface, center, width, height):
    """
    Renders a mathematical equation at the bottom center of the window.
    """
    equation = "22.5° X 2 = 45°, 45° X 8 = 360°"
    eq_text = FONT_LARGE.render(equation, True, (255, 255, 255))
    eq_rect = eq_text.get_rect(center=(center[0], height - 50))
    surface.blit(eq_text, eq_rect)

# Draw expanding circles on beat
def draw_expanding_circles(surface, circles, center):
    """
    Draws and updates expanding circles on the surface based on detected beats.
    """
    for circle in circles:
        try:
            # Draw the circle with current color (includes alpha)
            pygame.draw.circle(surface, circle['color'], center, int(circle['radius']), 2)
            # Update circle properties
            circle['radius'] += circle['growth_rate']
            circle['alpha'] -= circle['decay_rate']
            # Update the color with new alpha
            circle['color'] = (*circle['base_color'], max(int(circle['alpha']), 0))
        except KeyError as e:
            print(f"Missing key in circle dictionary: {e}")

    # Remove circles that are no longer visible
    circles[:] = [c for c in circles if c['alpha'] > 0]

# Function to toggle full-screen mode
def toggle_fullscreen(current_window, is_fullscreen, WINDOW_WIDTH, WINDOW_HEIGHT):
    """
    Toggles between full-screen and windowed mode.
    Returns the updated window and its dimensions.
    """
    if is_fullscreen:
        # Switch to windowed mode with 1920x1080 resolution
        WINDOW_WIDTH, WINDOW_HEIGHT = 1920, 1080
        current_window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.RESIZABLE | pygame.DOUBLEBUF)
    else:
        # Switch to full-screen mode
        current_window = pygame.display.set_mode((0, 0), pygame.FULLSCREEN | pygame.DOUBLEBUF)
        WINDOW_WIDTH, WINDOW_HEIGHT = current_window.get_size()
    return current_window, WINDOW_WIDTH, WINDOW_HEIGHT

# Function to initialize the display
def initialize_display():
    global WINDOW_WIDTH, WINDOW_HEIGHT, window, circle_surface
    WINDOW_WIDTH, WINDOW_HEIGHT = 1920, 1080  # 1080p resolution
    window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.RESIZABLE | pygame.DOUBLEBUF)
    pygame.display.set_caption("Kaleidoscope Audio Visualizer")
    # Surface for circles with per-pixel alpha
    circle_surface = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)

# Main loop
def main():
    global current_color_mode, rainbow_offset, paused, rotation_angle, rotation_speed, is_fullscreen, circle_surface

    # Initialize display before loading images
    initialize_display()

    # Load pattern images **after** setting up the display
    grey_slice_img, dark_slice_img = load_pattern_images()

    clock = pygame.time.Clock()
    running = True
    NUM_BARS = 16  # One bar per slice

    try:
        while running:
            dt = clock.tick(60) / 1000.0  # Delta time in seconds

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
                        # Toggle full-screen mode
                        is_fullscreen = not is_fullscreen
                        window, WINDOW_WIDTH, WINDOW_HEIGHT = toggle_fullscreen(window, is_fullscreen, WINDOW_WIDTH, WINDOW_HEIGHT)
                        # Reinitialize the circle surface to match new window size
                        circle_surface = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
                elif event.type == pygame.VIDEORESIZE and not is_fullscreen:
                    WINDOW_WIDTH, WINDOW_HEIGHT = event.w, event.h
                    window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.RESIZABLE | pygame.DOUBLEBUF)
                    circle_surface = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
                    # Recalculate number of bars based on new window size
                    NUM_BARS = max(10, WINDOW_WIDTH // 40)
                    NUM_BARS = min(NUM_BARS, 40)

            window.fill((30, 30, 30))  # Dark background

            if not paused:
                try:
                    data = np.frombuffer(stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16)
                except IOError as e:
                    print(f"Audio buffer overflow: {e}. Skipping frame.")
                    continue

                if CHANNELS == 2:
                    data = data.reshape(-1, 2).mean(axis=1).astype(np.int16)

                # Get frequency bars
                bars = get_frequency_bars(data, NUM_BARS, min(WINDOW_WIDTH, WINDOW_HEIGHT) // 2 - 50)
                # Normalize bars
                max_bar = max(bars) if max(bars) > 0 else 1
                normalized_bars = [bar / max_bar for bar in bars]
                # Map bars to slices
                slice_values = map_bars_to_slices(normalized_bars, 16)
                center = (WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2)
                radius = min(WINDOW_WIDTH, WINDOW_HEIGHT) // 2 - 100  # Adjusted for padding

                # Draw the kaleidoscope with patterns
                draw_kaleidoscope(window, center, radius, slice_values, grey_slice_img, dark_slice_img)

                # Draw the mathematical equation
                draw_equation(window, center, WINDOW_WIDTH, WINDOW_HEIGHT)

                # Beat Detection
                # Calculate energy in low frequencies (e.g., first 1/8th of the chunk)
                bass_energy = np.sum(data[:CHUNK//8]**2)  # Adjust CHUNK//8 for bass range
                ENERGY_HISTORY.append(bass_energy)
                average_energy = np.mean(ENERGY_HISTORY) if ENERGY_HISTORY else 1
                threshold = BEAT_THRESHOLD_MULTIPLIER * average_energy

                if bass_energy > threshold and len(circle_list) < MAX_CIRCLES:
                    # On beat, add a new expanding circle with initialized 'color'
                    base_col = color_modes[current_color_mode](0, NUM_BARS, rainbow_offset)
                    new_circle = {
                        'radius': 10,
                        'growth_rate': 5,
                        'decay_rate': 3,
                        'alpha': 255,
                        'base_color': base_col,
                        'color': (*base_col, 255)  # Initialize 'color' with full alpha
                    }
                    circle_list.append(new_circle)

                    # Optionally, adjust rotation speed based on beat intensity
                    rotation_speed = 2.0  # Increase rotation speed on beat

                else:
                    # Gradually return to base rotation speed
                    rotation_speed += (0.5 - rotation_speed) * 0.05

                rotation_angle += rotation_speed * dt * 60  # Adjust rotation based on speed and frame rate

                # Draw expanding circles
                draw_expanding_circles(circle_surface, circle_list, center)
                window.blit(circle_surface, (0, 0))

            # Update the display
            pygame.display.flip()
            rainbow_offset += 0.001  # Slower offset for smoother color transitions

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
        pygame.quit()
        sys.exit(0)

if __name__ == "__main__":
    main()

