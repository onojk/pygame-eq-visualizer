import numpy as np
import pyaudio
import pygame
import math

# Initialize Pygame
pygame.init()

# Constants
CHUNK = 1024
RATE = 44100  # Sampling rate
CHANNELS = 1
FPS = 60
WINDOW_WIDTH, WINDOW_HEIGHT = 1920, 1080  # Default HD resolution

# Frequency range for human hearing (20 Hz - 20 kHz)
MIN_FREQ = 20
MAX_FREQ = 20000

# Audio Input Setup
p = pyaudio.PyAudio()
def get_default_input_device():
    for i in range(p.get_device_count()):
        dev_info = p.get_device_info_by_index(i)
        if dev_info['maxInputChannels'] > 0:
            return i
    return None

input_device_index = get_default_input_device()
if input_device_index is None:
    raise ValueError("No input audio device found.")

stream = p.open(format=pyaudio.paInt16,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                input_device_index=input_device_index,
                frames_per_buffer=CHUNK)

# Pygame setup
window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.RESIZABLE)
pygame.display.set_caption("Full-Screen Audio Spectrum Visualizer")
clock = pygame.time.Clock()
font = pygame.font.Font(None, 36)

def draw_frequency_spectrum(surface, audio_data):
    """Draw the real-time frequency spectrum."""
    # Perform FFT on audio data
    fft_data = np.abs(np.fft.rfft(audio_data))  # Positive frequencies only
    fft_frequencies = np.fft.rfftfreq(len(audio_data), d=1.0 / RATE)

    # Filter frequencies within human hearing range
    valid_indices = np.where((fft_frequencies >= MIN_FREQ) & (fft_frequencies <= MAX_FREQ))
    fft_data = fft_data[valid_indices]
    fft_frequencies = fft_frequencies[valid_indices]

    # Normalize FFT data for amplitude
    if np.max(fft_data) > 0:
        fft_data = fft_data / np.max(fft_data)

    # Map frequencies to screen width
    num_bars = len(fft_data)
    bar_width = max(1, WINDOW_WIDTH // num_bars)

    # Draw bars for each frequency bin
    for i, amplitude in enumerate(fft_data):
        # Height of the bar, scaled to exceed the window height for clipping
        bar_height = int(amplitude * WINDOW_HEIGHT * 1.5)  # Scale up to exceed full height
        x = i * bar_width
        y = WINDOW_HEIGHT - bar_height

        # Color gradient from blue to magenta
        color = (
            max(0, min(255, int(128 + 127 * math.sin(2 * math.pi * i / num_bars)))),
            max(0, min(255, int(64 + 191 * math.cos(2 * math.pi * i / num_bars)))),
            max(0, min(255, int(128 + 127 * math.sin(2 * math.pi * i / num_bars + 3)))),
        )

        # Draw the bar
        pygame.draw.rect(surface, color, (x, y, bar_width - 1, bar_height))  # Thin gaps between bars

def main():
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.VIDEORESIZE:
                global WINDOW_WIDTH, WINDOW_HEIGHT
                WINDOW_WIDTH, WINDOW_HEIGHT = event.w, event.h
                window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.RESIZABLE)

        # Clear screen
        window.fill((0, 0, 0))

        # Read audio data
        try:
            data = np.frombuffer(stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16)
        except IOError:
            data = np.zeros(CHUNK, dtype=np.int16)

        # Draw the real-time frequency spectrum
        draw_frequency_spectrum(window, data)

        # Display FPS
        fps_text = font.render(f"FPS: {int(clock.get_fps())}", True, (255, 255, 255))
        window.blit(fps_text, (10, 10))

        # Update display
        pygame.display.flip()
        clock.tick(FPS)

    # Cleanup
    stream.stop_stream()
    stream.close()
    p.terminate()
    pygame.quit()

if __name__ == "__main__":
    main()

