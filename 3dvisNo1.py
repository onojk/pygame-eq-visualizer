import numpy as np
import pyaudio
import pygame
from scipy.fftpack import fft
import random
import colorsys

# Initialize Pygame
pygame.init()

# Constants
CHUNK = 1024
RATE = 44100
CHANNELS = 2
FPS = 30
WINDOW_WIDTH, WINDOW_HEIGHT = 800, 600
NUM_CREATURES = 10
TAIL_LENGTH = 2000
PIPE_RADIUS = 40
SQUARE_SIZE = 40
SPEED_MULTIPLIER = 200
AMPLIFY_MUSIC_RESPONSE = 3
TAIL_SPACING = 14  # Adjustable spacing between tail segments
BACKGROUND_COLOR = (0, 0, 0)

# PyAudio setup
p = pyaudio.PyAudio()
input_device_index = None  # Automatically select the first input device

# Open audio stream
try:
    stream = p.open(format=pyaudio.paInt16,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    input_device_index=input_device_index,
                    frames_per_buffer=CHUNK)
    print(f"Audio stream opened with sample rate {RATE}")
except Exception as e:
    print(f"Could not open audio stream: {e}")
    p.terminate()
    exit()

def hsv_to_rgb(h, s, v):
    """Convert HSV color to RGB."""
    return tuple(int(c * 255) for c in colorsys.hsv_to_rgb(h, s, v))

class Creature:
    def __init__(self, freq_band_index):
        self.position = np.array([random.randint(PIPE_RADIUS, WINDOW_WIDTH - PIPE_RADIUS),
                                  random.randint(PIPE_RADIUS, WINDOW_HEIGHT - PIPE_RADIUS)], dtype=float)
        self.velocity = np.random.uniform(-1, 1, 2)
        self.velocity /= np.linalg.norm(self.velocity)
        self.velocity *= SPEED_MULTIPLIER / FPS
        self.freq_band_index = freq_band_index
        self.tail = [{"pos": self.position.copy(), "brightness": 0.3} for _ in range(TAIL_LENGTH)]
        self.hue = random.random()

    def update(self, dt, bar_value):
        # Update position
        self.position += self.velocity * (dt / 1000)

        # Bounce off walls
        if self.position[0] - PIPE_RADIUS < 0 or self.position[0] + PIPE_RADIUS > WINDOW_WIDTH:
            self.velocity[0] *= -1
        if self.position[1] - PIPE_RADIUS < 0 or self.position[1] + PIPE_RADIUS > WINDOW_HEIGHT:
            self.velocity[1] *= -1

        # Update tail positions
        previous_position = self.position.copy()
        for segment in self.tail:
            current_position = segment["pos"].copy()
            segment["pos"] = previous_position
            previous_position = current_position

        # Apply brightness based on bar value
        num_active_segments = int(bar_value * TAIL_LENGTH * AMPLIFY_MUSIC_RESPONSE)
        for i in range(TAIL_LENGTH):
            if i < num_active_segments:
                self.tail[i]["brightness"] = 1.0  # Fully lit
            else:
                self.tail[i]["brightness"] *= 0.8  # Decay brightness more gradually

    def draw(self, screen):
        for i, segment in enumerate(reversed(self.tail)):
            pos = segment["pos"]
            brightness = segment["brightness"]

            if i % 5 == 0:  # Black notches
                color = (30, 30, 30)  # Dark gray instead of pure black
            else:
                color = hsv_to_rgb(self.hue, 1.0, max(brightness, 0.3))  # Ensure non-lit segments are visible

            pygame.draw.rect(screen, color, (pos[0] - SQUARE_SIZE // 2, pos[1] - SQUARE_SIZE // 2, SQUARE_SIZE, SQUARE_SIZE), border_radius=3)

        # Draw the transparent head
        head_surface = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
        transparent_color = (255, 255, 255, 128)  # Semi-transparent white
        pygame.draw.rect(head_surface, transparent_color, (0, 0, SQUARE_SIZE, SQUARE_SIZE))
        screen.blit(head_surface, (self.position[0] - SQUARE_SIZE // 2, self.position[1] - SQUARE_SIZE // 2))

def get_frequency_bars(data, num_bands):
    fft_data = fft(data)
    fft_magnitude = np.abs(fft_data)[:CHUNK // 2]
    freq_bins = np.linspace(0, RATE / 2, CHUNK // 2)
    band_limits = np.logspace(np.log10(20), np.log10(RATE / 2), num_bands + 1)
    bar_values = []

    for i in range(num_bands):
        idx = np.where((freq_bins >= band_limits[i]) & (freq_bins < band_limits[i + 1]))[0]
        if len(idx) > 0:
            avg_magnitude = np.mean(fft_magnitude[idx])
            max_magnitude = np.max(fft_magnitude)
            if max_magnitude > 0:
                bar_values.append(avg_magnitude / max_magnitude)
            else:
                bar_values.append(0)
        else:
            bar_values.append(0)
    return bar_values

def main():
    global WINDOW_WIDTH, WINDOW_HEIGHT
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Music Reactive Creatures with Visible Tails")
    clock = pygame.time.Clock()
    creatures = [Creature(i) for i in range(NUM_CREATURES)]
    running = True

    try:
        while running:
            dt = clock.tick(FPS)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.VIDEORESIZE:
                    WINDOW_WIDTH, WINDOW_HEIGHT = event.size
                    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.RESIZABLE)

            try:
                data = np.frombuffer(stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16)
                if CHANNELS == 2:
                    data = data.reshape(-1, 2).mean(axis=1).astype(np.int16)
            except IOError as e:
                print(f"Audio buffer overflow: {e}")
                continue

            bar_values = get_frequency_bars(data, NUM_CREATURES)

            screen.fill(BACKGROUND_COLOR)
            for i, creature in enumerate(creatures):
                creature.update(dt, bar_values[i])
                creature.draw(screen)

            pygame.display.flip()

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
        pygame.quit()

if __name__ == "__main__":
    main()

