import numpy as np
import pyaudio
import pygame
from scipy.fftpack import fft
import math
import colorsys
import os
import sys

# Initialize Pygame
pygame.init()

# Constants
CHUNK = 512
CHANNELS = 1
RATE = 22050
NUM_SLICES = 16
SLICE_ANGLE = 360 / NUM_SLICES
DEFAULT_WIDTH, DEFAULT_HEIGHT = 1280, 720
RAINBOW_SPEED = 0.002
LINE_DENSITY = 1
GLOBAL_ALPHA = 255
FPS = 24
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# File for recording data
home_directory = os.path.expanduser("~")
output_file_path = os.path.join(home_directory, "spectrum_data.txt")

# Audio Input Setup
p = pyaudio.PyAudio()

def get_input_device_index():
    """Get the default audio input device index."""
    try:
        return p.get_default_input_device_info()['index']
    except IOError:
        print("No default input device found.")
        sys.exit(1)

input_device_index = get_input_device_index()

def open_audio_stream():
    """Open the audio input stream."""
    stream = p.open(format=pyaudio.paInt16,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    input_device_index=input_device_index,
                    frames_per_buffer=CHUNK)
    return stream

stream = open_audio_stream()

# Screen Setup
screen = pygame.display.set_mode((DEFAULT_WIDTH, DEFAULT_HEIGHT))
pygame.display.set_caption("Audio Visualizer with Recording")

# Clock
clock = pygame.time.Clock()

# Recording State
recording = False

def audio_to_visual(data):
    """Process audio data into visual slices."""
    fft_data = fft(np.frombuffer(data, dtype=np.int16))[:CHUNK // 2]
    magnitude = np.abs(fft_data)
    normalized = np.clip(magnitude / (max(magnitude) + 1e-10), 0.0, 1.0)  # Avoid divide-by-zero
    return normalized

def draw_visual(spectrum):
    """Draw visual based on the audio spectrum."""
    screen.fill(BLACK)
    center_x, center_y = DEFAULT_WIDTH // 2, DEFAULT_HEIGHT // 2
    radius = 200

    for i, value in enumerate(spectrum[:NUM_SLICES]):
        value = np.clip(value, 0.0, 1.0)  # Ensure value is clamped
        # Calculate HSV and convert to RGB
        hue = value  # Hue is proportional to the normalized value
        saturation = 1.0  # Full saturation
        brightness = value  # Brightness proportional to value
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, brightness)
        r, g, b = int(r * 255), int(g * 255), int(b * 255)

        # Determine line endpoints
        angle = math.radians(i * SLICE_ANGLE)
        end_x = center_x + int(radius * math.cos(angle) * (1 + value))
        end_y = center_y + int(radius * math.sin(angle) * (1 + value))

        # Create the color and draw the line
        color = (r, g, b)
        pygame.draw.line(screen, color, (center_x, center_y), (end_x, end_y), 2)

    # Display current spectrum data
    font = pygame.font.Font(None, 36)
    spectrum_text = font.render(f"Spectrum: {spectrum[:5]}", True, WHITE)
    screen.blit(spectrum_text, (10, 10))

def draw_button():
    """Draw the start/stop recording button."""
    font = pygame.font.Font(None, 36)
    button_text = "Stop Recording" if recording else "Start Recording"
    button_color = (255, 0, 0) if recording else (0, 255, 0)
    button_rect = pygame.Rect(DEFAULT_WIDTH - 200, 10, 180, 50)
    pygame.draw.rect(screen, button_color, button_rect)
    text = font.render(button_text, True, WHITE)
    screen.blit(text, (DEFAULT_WIDTH - 190, 20))
    return button_rect

def toggle_recording():
    """Toggle the recording state."""
    global recording
    recording = not recording
    if recording:
        print("Recording started.")
        with open(output_file_path, "w") as f:
            f.write("Spectrum Data Recording Started\n")
    else:
        print("Recording stopped.")
        with open(output_file_path, "a") as f:
            f.write("Spectrum Data Recording Stopped\n")

# Main Loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            # Check if the button is clicked
            if draw_button().collidepoint(event.pos):
                toggle_recording()

    # Read audio stream
    data = stream.read(CHUNK, exception_on_overflow=False)
    spectrum = audio_to_visual(data)

    # Record data if recording is active
    if recording:
        with open(output_file_path, "a") as f:
            f.write(",".join(map(str, spectrum)) + "\n")

    # Draw visual and button
    draw_visual(spectrum)
    draw_button()

    # Update screen
    pygame.display.flip()
    clock.tick(FPS)

# Clean up
stream.stop_stream()
stream.close()
p.terminate()
pygame.quit()
sys.exit()

