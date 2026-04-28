import numpy as np
import pyaudio
import pygame  # Ensure pygame is imported here
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

def fire_color(index, total_bars, offset):
    """
    Generates fiery colors transitioning from red to yellow to white.
    """
    ratio = (index / total_bars + offset) % 1.0
    if ratio < 0.5:
        # Red to yellow
        r = 255
        g = int(255 * (ratio / 0.5))
        b = 0
    else:
        # Yellow to white
        r = 255
        g = 255
        b = int(255 * ((ratio - 0.5) / 0.5))
    return (r, g, b)

# Add more color modes as needed
color_modes = [rainbow_color, grayscale_color, random_color, fire_color]
current_color_mode = 0  # Initialize to the first color mode

# Add a list of visualizer modes
visualizer_modes = ['Rotating Spokes', 'Particle Wave']  # Existing and new modes
current_visualizer_mode = 0  # Initialize to the first visualizer mode

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

# Function to draw the rotating spokes with angle offsets
def draw_rotating_spokes(surface, bars, center, max_radius, num_spokes=240, rotation_angle=0, spoke_offsets=None):
    """
    Draws numerous rotating spokes emanating from the center based on frequency bars.

    Parameters:
        surface (pygame.Surface): The main surface to draw on.
        bars (list): List of frequency bar heights.
        center (tuple): The (x, y) coordinates of the center of the visualization.
        max_radius (int): The maximum radius of the visualization.
        num_spokes (int): Total number of spokes to draw.
        rotation_angle (float): The rotation angle in degrees.
        spoke_offsets (list): List of angle offsets for each spoke.
    """
    if spoke_offsets is None:
        spoke_offsets = [0] * num_spokes  # Default to no offset

    angle_between_spokes = 360 / num_spokes  # Angle between each spoke

    for i in range(num_spokes):
        # Calculate the current angle for this spoke with its offset
        angle = i * angle_between_spokes + rotation_angle + spoke_offsets[i]

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

# Particle Class Definition
class Particle:
    def __init__(self, position, velocity, size, color, lifespan):
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.size = size
        self.color = color
        self.lifespan = lifespan  # Total lifespan in frames
        self.age = 0  # Current age in frames

    def update(self):
        # Update position based on velocity
        self.position += self.velocity
        # Increment age
        self.age += 1

    def draw(self, surface):
        # Calculate fade based on lifespan
        fade = max(0, 255 * (1 - self.age / self.lifespan))
        if fade > 0:
            color = (*self.color[:3], int(fade))  # Add alpha for fading
            pygame.draw.circle(surface, color, self.position.astype(int), self.size)

# Flare Class Definition
class Flare:
    def __init__(self, position, max_radius, color, lifespan):
        self.position = np.array(position, dtype=float)
        self.max_radius = max_radius
        self.current_radius = 0
        self.color = color
        self.lifespan = lifespan
        self.age = 0

    def update(self):
        self.current_radius += self.max_radius / self.lifespan
        self.age += 1

    def draw(self, surface):
        fade = max(0, 255 * (1 - self.age / self.lifespan))
        if fade > 0:
            flare_color = (*self.color[:3], int(fade * 0.5))  # Semi-transparent
            pygame.draw.circle(surface, flare_color, self.position.astype(int), int(self.current_radius), width=2)

# Function to draw Particle Wave
def draw_particle_wave(surface, bars, particles, center):
    """
    Draws and updates particles based on frequency data.

    Parameters:
        surface (pygame.Surface): The main surface to draw on.
        bars (list): List of frequency bar heights.
        particles (list): List to hold active Particle instances.
        center (tuple): The (x, y) coordinates of the center of the visualization.
    """
    # Parameters for particle behavior
    PARTICLE_LIFESPAN = 60  # Frames
    PARTICLES_PER_FRAME = 2  # Number of particles to emit each frame

    # Emit new particles based on certain frequency bands
    for i, bar in enumerate(bars[:5]):  # Use the first 5 bars (low frequencies) for particle emission
        for _ in range(PARTICLES_PER_FRAME):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 3) * (bar / max(bars) if max(bars) != 0 else 1)
            velocity = (math.cos(angle) * speed, math.sin(angle) * speed)
            size = random.randint(2, 5)
            color = color_modes[current_color_mode](i, len(bars), rainbow_offset)
            lifespan = PARTICLE_LIFESPAN
            particle = Particle(position=center, velocity=velocity, size=size, color=color, lifespan=lifespan)
            particles.append(particle)

    # Update and draw particles
    for particle in particles[:]:
        particle.update()
        particle.draw(surface)
        if particle.age >= particle.lifespan:
            particles.remove(particle)

    # Limit the number of particles to prevent performance issues
    MAX_PARTICLES = 500  # Adjust as needed
    while len(particles) > MAX_PARTICLES:
        particles.pop(0)  # Remove the oldest particle

# Function to draw flares
def draw_flares(surface, flares):
    for flare in flares[:]:
        flare.update()
        flare.draw(surface)
        if flare.age >= flare.lifespan:
            flares.remove(flare)

# Function to detect and create flares based on beats
def handle_flares(bass_amplitude, center, flares):
    FLARE_THRESHOLD = 1000  # Define appropriately
    MAX_FLARES = 10
    FLARE_LIFESPAN = 60  # Frames

    if bass_amplitude > FLARE_THRESHOLD and len(flares) < MAX_FLARES:
        flare_color = color_modes[current_color_mode](0, len(bars), rainbow_offset)
        flare = Flare(position=center,
                      max_radius=100,  # Adjust as needed
                      color=flare_color,
                      lifespan=FLARE_LIFESPAN)
        flares.append(flare)

# Main loop
def main():
    global current_color_mode, rainbow_offset, paused, silence_counter, rms_smooth, previous_bar_heights, circle_list, prev_bass_amplitude, current_visualizer_mode

    # Initialize the window size
    WINDOW_WIDTH, WINDOW_HEIGHT = 1200, 900  # Increased window size for a larger visualization
    window = pygame.display.set_mode(
        (WINDOW_WIDTH, WINDOW_HEIGHT), pygame.RESIZABLE | pygame.NOFRAME)

    # Initialize number of bars based on window size
    NUM_BARS = max(10, WINDOW_WIDTH // 40)  # Increased minimum bars to 10
    NUM_BARS = min(NUM_BARS, 40)           # Increased maximum bars to 40
    previous_bar_heights = np.zeros(NUM_BARS)

    # Initialize number of spokes
    num_spokes = 240  # As defined earlier

    # Initialize angle offsets for each spoke
    spoke_offsets = [random.uniform(-5, 5) for _ in range(num_spokes)]

    # Create a semi-transparent surface for ghosting effect
    ghost_surface = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
    ghost_surface = ghost_surface.convert_alpha()
    ghost_surface.fill((0, 0, 0, 10))  # Lower alpha for stronger ghosting

    # Initialize particle list for Particle Wave mode
    particle_list = []

    # Initialize flare list for flaring effect
    flare_list = []

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
                        print(f"Switched to color mode: {color_modes[current_color_mode].__name__}")
                    elif event.key == pygame.K_LEFT:
                        current_color_mode = (current_color_mode - 1) % len(color_modes)
                        print(f"Switched to color mode: {color_modes[current_color_mode].__name__}")
                    elif event.key == pygame.K_UP:
                        current_visualizer_mode = (current_visualizer_mode + 1) % len(visualizer_modes)
                        print(f"Switched to visualizer mode: {visualizer_modes[current_visualizer_mode]}")
                    elif event.key == pygame.K_DOWN:
                        current_visualizer_mode = (current_visualizer_mode - 1) % len(visualizer_modes)
                        print(f"Switched to visualizer mode: {visualizer_modes[current_visualizer_mode]}")
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
                    ghost_surface.fill((0, 0, 0, 10))  # Lower alpha for stronger ghosting

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
                # Update Angle Offsets for Spokes
                # ----------------------------------------------
                for i in range(num_spokes):
                    # Change the offset by a small random amount each frame
                    spoke_offsets[i] += random.uniform(-1, 1)

                    # Optional: Clamp the offset to prevent excessive rotation
                    if spoke_offsets[i] > 10:
                        spoke_offsets[i] = 10
                    elif spoke_offsets[i] < -10:
                        spoke_offsets[i] = -10
                # ----------------------------------------------

                # ----------------------------------------------
                # Draw the Selected Visualizer Mode
                # ----------------------------------------------
                # Define center
                center = (WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2)

                if visualizer_modes[current_visualizer_mode] == 'Rotating Spokes':
                    # Existing Rotating Spokes drawing code with enhanced parameters
                    max_radius = min(WINDOW_WIDTH, WINDOW_HEIGHT) // 2
                    rotation_angle = rainbow_offset * 10
                    draw_rotating_spokes(
                        window,
                        bars,
                        center,
                        max_radius,
                        num_spokes=num_spokes,
                        rotation_angle=rotation_angle,
                        spoke_offsets=spoke_offsets  # Pass the offsets here
                    )

                    # ----------------------------------------------
                    # Handle Beat Detection and Flaring
                    # ----------------------------------------------
                    # Assuming the first bar corresponds to the lowest frequencies (bass)
                    bass_amplitude = bars[0]

                    # Smooth the bass amplitude
                    smoothed_bass = prev_bass_amplitude * 0.9 + bass_amplitude * 0.1
                    prev_bass_amplitude = smoothed_bass

                    # Detect beat and create flare
                    handle_flares(smoothed_bass, center, flare_list)

                    # Draw active flares
                    draw_flares(window, flare_list)
                    # ----------------------------------------------

                elif visualizer_modes[current_visualizer_mode] == 'Particle Wave':
                    # Draw Particle Wave
                    draw_particle_wave(window, bars, particle_list, center)
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
            ghost_surface.fill((0, 0, 0, 10))  # Lower alpha for stronger ghosting

            # Display current mode information
            font = pygame.font.SysFont(None, 24)
            mode_text = f"Visualizer Mode: {visualizer_modes[current_visualizer_mode]}"
            color_text = f"Color Mode: {color_modes[current_color_mode].__name__}"
            text_surface = font.render(f"{mode_text} | {color_text}", True, (255, 255, 255))
            window.blit(text_surface, (10, 10))

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

