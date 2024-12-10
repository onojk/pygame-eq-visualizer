import numpy as np
import pyaudio
import pygame
from scipy.fftpack import fft
import math
import os
from moviepy.editor import VideoFileClip

# Set FFmpeg path for moviepy
os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"

# Initialize Pygame
pygame.init()

# Constants
CHUNK = 512  # Audio chunk size
CHANNELS = 1  # Mono audio
RATE = 22050  # Audio sample rate
RENDER_WIDTH, RENDER_HEIGHT = 1280, 720  # Rendering resolution
AMPLITUDE_SCALE = 1.0  # Scaling for motion
ROTATION_SPEED = 1.0  # Speed of rotation
SCALE_MIN = 0.8  # Minimum scaling factor
SCALE_MAX = 1.2  # Maximum scaling factor
FPS = 30  # Frames per second for video playback

# Path to your video
VIDEO_PATH = "/home/onojk123/Downloads/Lotus Dream.mp4"

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
pygame.display.set_caption("Music-Responsive Video")
window = pygame.display.set_mode((RENDER_WIDTH, RENDER_HEIGHT), pygame.RESIZABLE | pygame.DOUBLEBUF)

# Load video with MoviePy
def load_video(filepath):
    try:
        clip = VideoFileClip(filepath)
        return clip
    except Exception as e:
        print(f"Error loading video: {e}")
        sys.exit(1)

# Apply transformations
def apply_transformations(surface, video_frame, audio_bars):
    """
    Apply scaling, rotation, and motion to the video frame based on audio.
    """
    center_x, center_y = RENDER_WIDTH // 2, RENDER_HEIGHT // 2

    # Compute transformations
    avg_amplitude = np.mean(audio_bars)
    scale_factor = SCALE_MIN + avg_amplitude * (SCALE_MAX - SCALE_MIN)
    rotation_angle = avg_amplitude * 360 * ROTATION_SPEED
    offset_x = int(math.sin(pygame.time.get_ticks() * 0.005) * avg_amplitude * 50)
    offset_y = int(math.cos(pygame.time.get_ticks() * 0.005) * avg_amplitude * 50)

    # Scale and rotate the video frame
    scaled_frame = pygame.transform.scale(video_frame, (
        int(RENDER_WIDTH * scale_factor),
        int(RENDER_HEIGHT * scale_factor)
    ))
    rotated_frame = pygame.transform.rotate(scaled_frame, rotation_angle)

    # Get the frame rect and apply offsets
    rect = rotated_frame.get_rect(center=(center_x + offset_x, center_y + offset_y))
    surface.blit(rotated_frame, rect)

# Main loop
def main():
    clock = pygame.time.Clock()
    running = True
    video_clip = load_video(VIDEO_PATH)
    video_surface = pygame.Surface((RENDER_WIDTH, RENDER_HEIGHT))  # Off-screen rendering surface
    video_duration = video_clip.duration
    video_time = 0

    while running:
        dt = clock.tick(FPS) / 1000.0  # Limit to FPS and get delta time
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False

        # Clear the screen
        video_surface.fill((0, 0, 0))

        # Read audio data
        try:
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

        # Get the current video frame
        if video_time < video_duration:
            frame_array = video_clip.get_frame(video_time)
            frame_surface = pygame.surfarray.make_surface(np.rot90(frame_array))

            # Apply transformations to the video frame
            apply_transformations(video_surface, frame_surface, audio_bars)

            video_time += dt  # Update video time based on delta time

        # Display on the preview window
        pygame.transform.scale(video_surface, (RENDER_WIDTH, RENDER_HEIGHT), window)
        pygame.display.flip()

    # Cleanup
    video_clip.close()
    stream.stop_stream()
    stream.close()
    p.terminate()
    pygame.quit()

if __name__ == "__main__":
    main()

