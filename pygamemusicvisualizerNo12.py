import numpy as np
import pyaudio
import pygame
from scipy.fftpack import fft
import random

# Initialize Pygame
pygame.init()

# Constants
CHUNK = 512  # Audio chunk size
CHANNELS = 1  # Mono audio
RATE = 22050  # Audio sample rate
WINDOW_WIDTH, WINDOW_HEIGHT = 1280, 720  # Window size for preview
AMPLITUDE_THRESHOLD = 150000  # Threshold to trigger lightning bolts
MAX_DEPTH = 6  # Maximum recursion depth for fractal branches
BRANCH_ANGLE_RANGE = 35  # Maximum angle variation for branches
LIGHTNING_COLORS = [
    (255, 255, 255),  # White
    (0, 191, 255),    # Deep Sky Blue
    (255, 69, 0),     # Orange Red
    (75, 0, 130),     # Indigo
    (238, 130, 238),  # Violet
    (50, 205, 50),    # Lime Green
    (255, 215, 0)     # Gold
]

# Audio Input Setup
p = pyaudio.PyAudio()

def get_input_device_index():
    try:
        return p.get_default_input_device_info()['index']
    except IOError:
        print("No default input device found.")
        sys.exit(1)

def open_audio_stream():
    stream = p.open(format=pyaudio.paInt16,  # 16-bit audio
                    channels=CHANNELS,  # Mono
                    rate=RATE,  # Sample rate
                    input=True,  # Input mode
                    input_device_index=get_input_device_index(),  # Use default input device
                    frames_per_buffer=CHUNK)  # Buffer size
    return stream

stream = open_audio_stream()

# Pygame Setup
pygame.display.set_caption("Jagged and Bold Lightning Symphony")
window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.RESIZABLE | pygame.DOUBLEBUF)
clock = pygame.time.Clock()

# Recursive function to draw fractal lightning
def draw_fractal_lightning(surface, start_x, start_y, length, angle, depth, color, thickness):
    if depth == 0:
        return

    # Calculate the end point of the current segment
    end_x = start_x + length * np.cos(np.radians(angle))
    end_y = start_y + length * np.sin(np.radians(angle))

    # Draw the line segment
    pygame.draw.line(surface, color, (start_x, start_y), (end_x, end_y), max(1, int(thickness)))

    # Randomize branching
    branch_count = random.randint(1, 3)  # Number of branches from this point
    for _ in range(branch_count):
        new_length = length * random.uniform(0.5, 0.8)  # Reduce length for branches
        new_angle = angle + random.uniform(-BRANCH_ANGLE_RANGE, BRANCH_ANGLE_RANGE)  # Randomize branch angle
        new_thickness = max(1, int(thickness * random.uniform(0.5, 1.0)))  # Randomize thickness for branches
        draw_fractal_lightning(surface, end_x, end_y, new_length, new_angle, depth - 1, color, new_thickness)

# Main Loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Read audio data
    data = np.frombuffer(stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16)
    amplitude = np.abs(data).sum()  # Compute total amplitude

    # Clear screen
    window.fill((0, 0, 0))

    # Trigger multiple lightning bolts if amplitude exceeds threshold
    if amplitude > AMPLITUDE_THRESHOLD:
        num_bolts = int(amplitude / AMPLITUDE_THRESHOLD)  # Scale the number of bolts to amplitude
        for _ in range(min(num_bolts, 20)):  # Limit the maximum number of bolts
            start_x = random.randint(0, WINDOW_WIDTH)  # Random x position for the bolt
            start_y = 0  # Lightning starts at the top of the screen
            initial_length = random.randint(WINDOW_HEIGHT // 4, WINDOW_HEIGHT)  # Random length
            initial_angle = random.uniform(70, 110)  # Slightly randomized downward angle
            color = random.choice(LIGHTNING_COLORS)  # Choose a random color for the bolt
            initial_thickness = random.randint(2, 10)  # Randomize the initial thickness
            draw_fractal_lightning(window, start_x, start_y, initial_length, initial_angle, MAX_DEPTH, color, initial_thickness)

    # Update the display
    pygame.display.flip()
    clock.tick(30)

# Cleanup
stream.stop_stream()
stream.close()
p.terminate()
pygame.quit()

