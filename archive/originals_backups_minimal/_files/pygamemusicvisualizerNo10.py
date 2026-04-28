import numpy as np
import pyaudio
import pygame
from scipy.fftpack import fft
import math
from PIL import Image
import os

# Initialize Pygame
pygame.init()

# Constants
CHUNK = 512  # Audio chunk size
CHANNELS = 1  # Mono audio
RATE = 22050  # Audio sample rate
RENDER_WIDTH, RENDER_HEIGHT = 3840, 2160  # 4K Resolution
WINDOW_WIDTH, WINDOW_HEIGHT = 1280, 720  # Window size for preview
AMPLITUDE_SCALE = 1.0  # Scaling for motion
ROTATION_SPEED = 1.0  # Speed of rotation
SCALE_MIN = 0.5  # Minimum scaling factor
SCALE_MAX = 2.0  # Maximum scaling factor

# Path to your image
IMAGE_PATH = './orbit_of_reflections_by_onojk123_dikj084.jpg'

# Folder to save frames for recording
OUTPUT_FOLDER = './frames'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Audio Input Setup
import sys
import pyaudio

p = pyaudio.PyAudio()

# Get the default input device index
def get_input_device_index():
    try:
        return p.get_default_input_device_info()['index']
    except IOError:
        print("No default input device found.")
        sys.exit(1)

# Open an audio stream
def open_audio_stream():
    stream = p.open(format=pyaudio.paInt16,  # 16-bit audio
                    channels=CHANNELS,  # Mono
                    rate=RATE,  # Sample rate
                    input=True,  # Input mode
                    input_device_index=get_input_device_index(),  # Use default input device
                    frames_per_buffer=CHUNK)  # Buffer size
    return stream

# Initialize the audio stream
stream = open_audio_stream()

# Pygame Setup
pygame.display.set_caption("4K Visualizer")
window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.RESIZABLE | pygame.DOUBLEBUF)

# Load and process image
def load_image(filepath):
    """
    Load and resize the image to fit the 4K rendering size.
    """
    image = Image.open(filepath)
    image = image.resize((RENDER_WIDTH, RENDER_HEIGHT))  # Fit image to 4K size
    return pygame.image.fromstring(image.tobytes(), image.size, image.mode)

# Apply transformations
def apply_transformations(surface, image, audio_bars, dt):
    """
    Apply scaling, rotation, and motion to the image based on audio.
    """
    center_x, center_y = RENDER_WIDTH // 2, RENDER_HEIGHT // 2

    # Compute transformations
    avg_amplitude = np.mean(audio_bars)
    scale_factor = SCALE_MIN + (avg_amplitude * (SCALE_MAX - SCALE_MIN)) * 2
    rotation_angle = (avg_amplitude * 360 * ROTATION_SPEED * 2) % 360
    offset_x = int(math.sin(pygame.time.get_ticks() * 0.005) * avg_amplitude * 300)
    offset_y = int(math.cos(pygame.time.get_ticks() * 0.005) * avg_amplitude * 300)

    # Scale and rotate the image
    scaled_image = pygame.transform.scale(image, (
        int(RENDER_WIDTH * scale_factor),
        int(RENDER_HEIGHT * scale_factor)
    ))
    rotated_image = pygame.transform.rotate(scaled_image, rotation_angle)

    # Get the image rect and apply offsets
    rect = rotated_image.get_rect(center=(center_x + offset_x, center_y + offset_y))
    surface.blit(rotated_image, rect)

# Main loop
def main():
    clock = pygame.time.Clock()
    running = True
    recording = False
    frame_count = 0

    image = load_image(IMAGE_PATH)
    render_surface = pygame.Surface((RENDER_WIDTH, RENDER_HEIGHT))  # Off-screen 4K rendering surface

    while running:
        dt = clock.tick(30) / 1000.0  # Limit to 30 FPS
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:  # Start/Stop recording
                    recording = not recording
                    print("Recording started." if recording else "Recording stopped.")

        # Clear the screen
        render_surface.fill((0, 0, 0))

        try:
            # Read audio data
            data = np.frombuffer(stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16)
        except IOError:
            continue

        # Handle stereo audio
        if CHANNELS == 2:
            data = data.reshape(-1, 2).mean(axis=1).astype(np.int16)

        # Compute FFT
        audio_bars = np.abs(fft(data))[:CHUNK // 2]
        if np.max(audio_bars) == 0:
            audio_bars = np.zeros_like(audio_bars)
        else:
            audio_bars = audio_bars / np.max(audio_bars)

        # Apply transformations
        apply_transformations(render_surface, image, audio_bars, dt)

        # Display on the preview window
        pygame.transform.scale(render_surface, (WINDOW_WIDTH, WINDOW_HEIGHT), window)
        pygame.display.flip()

        # Save frame for recording
        if recording:
            frame_filename = os.path.join(OUTPUT_FOLDER, f"frame_{frame_count:05d}.png")
            pygame.image.save(render_surface, frame_filename)
            frame_count += 1

    # Cleanup
    stream.stop_stream()
    stream.close()
    p.terminate()
    pygame.quit()

if __name__ == "__main__":
    main()
