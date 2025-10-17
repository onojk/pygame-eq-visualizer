import numpy as np
import pyaudio
import pygame
from scipy.fftpack import fft
import sys
from collections import deque

# Initialize Pygame
pygame.init()

# Constants
CHUNK = 1024               # Number of audio samples per frame
CHANNELS = 1               # Number of audio channels (mono for simplicity)
FORMAT = pyaudio.paInt16   # Audio format
RATE = 44100               # Sampling rate
MAX_FPS = 60               # Maximum Frames Per Second
BEAT_THRESHOLD_MULTIPLIER = 3.3  # Multiplier for dynamic threshold
ENERGY_HISTORY = deque(maxlen=43)  # Approximately 0.73 seconds of history at ~60 FPS
NUM_RINGS = 8             # Number of concentric rings per speaker
RING_SPACING = 2           # Spacing between rings in pixels (smaller for more rings)
INVERSION_DURATION = 0.3   # Duration of inversion in seconds
SCALE_SPEED = 0.1          # Speed at which the speaker cone scales
NUM_SPEAKERS = 64          # Number of speakers in the grid (8x8)
TRAIL_LENGTH = 30         # Number of frames to keep as ghost trail
TRAIL_ALPHA_DECAY = 60     # How much to reduce alpha for each ghost frame (0 to 255)

# Global variables
paused = False
is_fullscreen = False

# PyAudio Setup
p = pyaudio.PyAudio()

# Function to select the audio input device
def get_input_device_index(desired_input_device=None):
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

# Open the audio stream
def open_audio_stream():
    try:
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)
        print(f"Opened audio stream with sample rate: {RATE}")
        return stream
    except Exception as e:
        print(f"Could not open audio stream: {e}")
        p.terminate()
        sys.exit(1)

stream = open_audio_stream()

# Function to compute energy of the audio signal
def compute_energy(audio_data):
    """Computes the energy of the audio signal."""
    return np.sum(np.abs(audio_data)) / len(audio_data)

# Function to split FFT data into 64 frequency bands using logarithmic spacing
def get_frequency_bands(fft_data, rate, num_bands=64):
    freq_min = 20
    freq_max = 20000
    bands = []
    for i in range(num_bands + 1):
        low = freq_min * (freq_max / freq_min) ** (i / num_bands)
        bands.append(low)
    
    band_energies = []
    n = len(fft_data)
    freq_resolution = rate / n
    
    for i in range(num_bands):
        low = bands[i]
        high = bands[i + 1]
        low_bin = int(low / freq_resolution)
        high_bin = int(high / freq_resolution)
        
        low_bin = max(low_bin, 0)
        high_bin = min(high_bin, n)
        
        band_energy = np.mean(np.abs(fft_data[low_bin:high_bin]))
        band_energies.append(band_energy)
    
    return band_energies

# Function to draw a speaker-like visualization with ghost trails
def draw_speaker(window, position, frequency_energy, scale, inverted, rings_inverted, trail_history):
    center_x, center_y = position
    base_cone_radius = 10  # Base radius for the speaker cone
    cone_radius = int(base_cone_radius * scale)  # Scaled radius
    
    if inverted:
        cone_color = (255, 255, 255)  # White
        frame_color = (0, 0, 0)        # Black
    else:
        cone_color = (0, 0, 0)        # Black
        frame_color = (255, 255, 255)  # White
    
    for frame_idx, past_scale in enumerate(trail_history):
        trail_alpha = max(0, 255 - (TRAIL_ALPHA_DECAY * frame_idx))
        ghost_cone_color = (*cone_color[:3], trail_alpha)
        ghost_frame_color = (*frame_color[:3], trail_alpha)
        ghost_radius = int(base_cone_radius * past_scale)
        
        pygame.draw.circle(window, ghost_cone_color, (center_x, center_y), ghost_radius)
        
        for i in range(1, NUM_RINGS + 1):
            ring_radius = ghost_radius + i * RING_SPACING
            if rings_inverted:
                ring_color = (255, 255, 255, trail_alpha) if i % 2 == 0 else (0, 0, 0, trail_alpha)
            else:
                ring_color = (0, 0, 0, trail_alpha) if i % 2 == 0 else (255, 255, 255, trail_alpha)
            pygame.draw.circle(window, ring_color, (center_x, center_y), ring_radius, 2)
    
    pygame.draw.circle(window, cone_color, (center_x, center_y), cone_radius)
    
    frame_thickness = 4
    frame_radius = base_cone_radius * 2
    pygame.draw.circle(window, frame_color, (center_x, center_y), frame_radius, frame_thickness)
    
    for i in range(1, NUM_RINGS + 1):
        ring_radius = frame_radius + i * RING_SPACING
        if rings_inverted:
            ring_color = (255, 255, 255) if i % 2 == 0 else (0, 0, 0)
        else:
            ring_color = (0, 0, 0) if i % 2 == 0 else (255, 255, 255)
        pygame.draw.circle(window, ring_color, (center_x, center_y), ring_radius, 2)

# Calculate positions for a grid of speakers
def calculate_grid_positions(grid_size, window_size):
    cols, rows = grid_size
    window_width, window_height = window_size
    margin_x = window_width // (cols + 1)
    margin_y = window_height // (rows + 1)
    positions = []
    for row in range(1, rows + 1):
        for col in range(1, cols + 1):
            x = col * margin_x
            y = row * margin_y
            positions.append((x, y))
    return positions

# Main function
def main():
    global paused, is_fullscreen
    
    WINDOW_WIDTH, WINDOW_HEIGHT = 1200, 1200
    window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.RESIZABLE | pygame.DOUBLEBUF)
    pygame.display.set_caption("8x8 Speaker-Like Audio Visualizer (64 speakers)")
    
    GRID_COLS, GRID_ROWS = 8, 8  # 8x8 grid for 64 speakers
    grid_size = (GRID_COLS, GRID_ROWS)
    speaker_positions = calculate_grid_positions(grid_size, (WINDOW_WIDTH, WINDOW_HEIGHT))
    
    scaling_factors = {i: 1.0 for i in range(NUM_SPEAKERS)}
    target_scales = {i: 1.0 for i in range(NUM_SPEAKERS)}
    
    inversion_states = {i: False for i in range(NUM_SPEAKERS)}
    inversion_timers = {i: 0 for i in range(NUM_SPEAKERS)}
    rings_inversion_states = {i: False for i in range(NUM_SPEAKERS)}
    rings_inversion_timers = {i: 0 for i in range(NUM_SPEAKERS)}
    
    clock = pygame.time.Clock()
    
    FONT_SMALL = pygame.font.SysFont('Arial', 16)
    FONT_LARGE = pygame.font.SysFont('Arial', 24)
    
    try:
        while True:
            dt = clock.tick(MAX_FPS) / 1000.0
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key == pygame.K_ESCAPE:
                        raise KeyboardInterrupt
                    elif event.key == pygame.K_f or event.key == pygame.K_F11:
                        is_fullscreen = not is_fullscreen
                        if is_fullscreen:
                            window = pygame.display.set_mode((0, 0), pygame.FULLSCREEN | pygame.DOUBLEBUF)
                        else:
                            window = pygame.display.set_mode((1200, 1200), pygame.RESIZABLE | pygame.DOUBLEBUF)
                        WINDOW_WIDTH, WINDOW_HEIGHT = window.get_size()
                        speaker_positions = calculate_grid_positions(grid_size, (WINDOW_WIDTH, WINDOW_HEIGHT))
            
            window.fill((50, 50, 50))
            
            if not paused:
                try:
                    data = np.frombuffer(stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16)
                except IOError as e:
                    print(f"Audio buffer overflow: {e}. Skipping frame.")
                    continue
                
                fft_data = fft(data)
                fft_data = fft_data[:len(fft_data)//2]
                
                band_energies = get_frequency_bands(fft_data, RATE, num_bands=NUM_SPEAKERS)
                
                ENERGY_HISTORY.append(band_energies)
                avg_energies = np.mean(ENERGY_HISTORY, axis=0)
                
                thresholds = BEAT_THRESHOLD_MULTIPLIER * avg_energies
                
                for i in range(NUM_SPEAKERS):
                    energy = band_energies[i]
                    if energy > thresholds[i] and energy > 1000:
                        inversion_states[i] = True
                        inversion_timers[i] = INVERSION_DURATION
                        rings_inversion_states[i] = True
                        rings_inversion_timers[i] = INVERSION_DURATION
                        target_scales[i] = min(2.0, 1.0 + (energy / 5000))
                    else:
                        target_scales[i] = max(1.0, target_scales[i] - 0.05)
                
                for i in range(NUM_SPEAKERS):
                    if inversion_timers[i] > 0:
                        inversion_timers[i] -= dt
                        if inversion_timers[i] <= 0:
                            inversion_states[i] = False
                    if rings_inversion_timers[i] > 0:
                        rings_inversion_timers[i] -= dt
                        if rings_inversion_timers[i] <= 0:
                            rings_inversion_states[i] = False
                
                for i in range(NUM_SPEAKERS):
                    if scaling_factors[i] < target_scales[i]:
                        scaling_factors[i] += SCALE_SPEED
                        if scaling_factors[i] > target_scales[i]:
                            scaling_factors[i] = target_scales[i]
                    elif scaling_factors[i] > target_scales[i]:
                        scaling_factors[i] -= SCALE_SPEED
                        if scaling_factors[i] < target_scales[i]:
                            scaling_factors[i] = target_scales[i]
            
            for i in range(NUM_SPEAKERS):
                position = speaker_positions[i]
                frequency_energy = band_energies[i] if not paused else 0
                scale = scaling_factors[i]
                inverted = inversion_states[i]
                rings_inverted = rings_inversion_states[i]
                draw_speaker(window, position, frequency_energy, scale, inverted, rings_inverted, [scale]*TRAIL_LENGTH)
            
            fps_text = FONT_SMALL.render(f"FPS: {int(clock.get_fps())}", True, (255, 255, 255))
            instructions = ["Space: Pause/Resume", "Esc: Quit", "F11: Toggle Fullscreen"]
            window.blit(fps_text, (10, 10))
            for idx, text in enumerate(instructions):
                instr_text = FONT_SMALL.render(text, True, (255, 255, 255))
                window.blit(instr_text, (10, 30 + idx * 20))
            
            if paused:
                pause_text = FONT_LARGE.render("Paused", True, (255, 0, 0))
                pause_rect = pause_text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2))
                window.blit(pause_text, pause_rect)
            
            pygame.display.flip()

    except KeyboardInterrupt:
        print("Program exited by user.")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    main()

