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
TAIL_LENGTH = 300
PIPE_RADIUS = 60
SQUARE_SIZE = 60
SPEED_MULTIPLIER = 300
AMPLIFY_MUSIC_RESPONSE = 5  # Amplify the pulse effect
BACKGROUND_COLOR = (0, 0, 0)
rainbow_offset = 0

# PyAudio Setup
p = pyaudio.PyAudio()

def get_input_device_index(desired_device_name=None):
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
    print("No suitable input device found.")
    return None

# Initialize input_device_index
input_device_index = get_input_device_index()

# Check if input_device_index is valid
if input_device_index is None:
    print("Error: No valid input device found.")
    p.terminate()
    exit()

# Open audio stream
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
    exit()

# Helper Functions
def rainbow_color(index, total_bars, offset):
    """Generate a rainbow color."""
    hue = (index / total_bars + offset) % 1.0
    rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
    return tuple(int(c * 255) for c in rgb)

def get_frequency_bars(data, num_bars):
    """Process audio data and return bar values."""
    fft_data = fft(data)
    fft_magnitude = np.abs(fft_data)[:CHUNK // 2]
    freq_bins = np.linspace(0, RATE / 2, CHUNK // 2)
    freq_band_limits = np.logspace(np.log10(20), np.log10(RATE / 2), num_bars + 1)
    bar_values = []

    for i in range(num_bars):
        indices = np.where((freq_bins >= freq_band_limits[i]) & (freq_bins < freq_band_limits[i + 1]))[0]
        if len(indices) > 0:
            avg_magnitude = np.mean(fft_magnitude[indices])
            bar_values.append(min(avg_magnitude / 5000, 1.0))  # Normalize
        else:
            bar_values.append(0.0)
    return bar_values

# Creature Class
class Creature:
    def __init__(self, freq_band_index):
        self.position = np.array([random.randint(PIPE_RADIUS, WINDOW_WIDTH - PIPE_RADIUS),
                                  random.randint(PIPE_RADIUS, WINDOW_HEIGHT - PIPE_RADIUS)], dtype=float)
        self.velocity = np.random.uniform(-1, 1, 2)
        self.velocity /= np.linalg.norm(self.velocity)
        self.velocity *= SPEED_MULTIPLIER / FPS
        self.freq_band_index = freq_band_index
        self.tail = [{"pos": self.position.copy(), "color": rainbow_color(i, TAIL_LENGTH, rainbow_offset)} for i in range(TAIL_LENGTH)]
        self.rainbow_index = 0

    def update(self, dt, bar_value):
        self.position += self.velocity * (dt / 1000)
        if self.position[0] - PIPE_RADIUS < 0 or self.position[0] + PIPE_RADIUS > WINDOW_WIDTH:
            self.velocity[0] *= -1
        if self.position[1] - PIPE_RADIUS < 0 or self.position[1] + PIPE_RADIUS > WINDOW_HEIGHT:
            self.velocity[1] *= -1

        # Update tail
        self.tail.pop(0)
        active_color = rainbow_color(self.rainbow_index, TAIL_LENGTH, rainbow_offset)
        self.tail.append({"pos": self.position.copy(), "color": active_color})
        self.rainbow_index = (self.rainbow_index + 1) % TAIL_LENGTH

        # Add pulse effect
        num_active_segments = int(bar_value * TAIL_LENGTH * AMPLIFY_MUSIC_RESPONSE)
        for i in range(min(num_active_segments, len(self.tail))):
            self.tail[i]["color"] = (255, 255, 255)  # White for the pulse effect

    def draw(self, screen):
        for i, segment in enumerate(reversed(self.tail)):
            pos = segment["pos"]
            color = segment["color"]
            pygame.draw.rect(screen, color, (pos[0] - SQUARE_SIZE // 2, pos[1] - SQUARE_SIZE // 2, SQUARE_SIZE, SQUARE_SIZE), border_radius=3)
        pygame.draw.rect(screen, (255, 255, 255), (self.position[0] - SQUARE_SIZE // 2, self.position[1] - SQUARE_SIZE // 2, SQUARE_SIZE, SQUARE_SIZE), border_radius=3)

# Main Function
def main():
    global rainbow_offset
    global WINDOW_WIDTH, WINDOW_HEIGHT

    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.RESIZABLE)
    pygame.display.set_caption("Dynamic Music Creatures with Reverse Pulse")
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
            rainbow_offset += 0.01

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
        pygame.quit()
        print("Visualizer closed.")

if __name__ == "__main__":
    main()

