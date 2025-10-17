import numpy as np
import pygame
import random

# Initialize Pygame
pygame.init()

# Constants
WINDOW_WIDTH, WINDOW_HEIGHT = 1280, 720  # 720p Window size
GRID_SIZE = 100  # Grid resolution for potential field
WINDOW_PORT_IMAGE = "/home/onojk123/pygame-eq-visualizer/window.png"
MAIN_BOLT_COLOR = (255, 255, 255)
COOLDOWN_TIME = 0.5  # Lightning cooldown time in seconds
AMPLITUDE_MULTIPLIER = 1.2  # Sensitivity multiplier for amplitude detection
MAX_RECURSION_DEPTH = 10  # Prevent excessive recursion
CENTER_THIRD_START = GRID_SIZE // 3  # Start of center third
CENTER_THIRD_END = 2 * (GRID_SIZE // 3)  # End of center third

# Setup Pygame
window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Exaggerated Lightning Visualizer")
clock = pygame.time.Clock()

def trace_lightning(start_x, max_depth=50, branching_prob=0.3, recursion_depth=0):
    """
    Trace a lightning bolt from the top to somewhere in the air, with branching.
    Bolts can extend off-screen with added variance.
    """
    if recursion_depth > MAX_RECURSION_DEPTH:
        return [], []  # Stop recursion if depth is exceeded

    path = [(start_x, 0)]  # Start at the top
    branches = []
    x, y = start_x, 0
    depth = 0

    while y < GRID_SIZE - 1 and depth < max_depth:
        neighbors = [
            (x, y + 1),  # Down
            (x - 1, y + 1),  # Down-left
            (x + 1, y + 1),  # Down-right
        ]
        neighbors = [(nx, ny) for nx, ny in neighbors if 0 <= nx < GRID_SIZE]

        if not neighbors:
            break

        # Random jaggedness for visual realism
        next_x, next_y = random.choice(neighbors)
        x = max(0, min(next_x + random.randint(-3, 3), GRID_SIZE - 1))  # Exaggerate randomness
        y = next_y + random.randint(0, int(GRID_SIZE * 0.15))  # Allow bolts to extend off-screen by 15%
        path.append((x, y))
        depth += 1

        # Random branching
        if random.random() < branching_prob:
            branch_start = (x + random.randint(-5, 5), y)
            branch_path, _ = trace_lightning(branch_start[0], max_depth=10, branching_prob=0.2, recursion_depth=recursion_depth + 1)
            branches.append(branch_path)

    return path, branches

def draw_lightning(surface, path, branches):
    """
    Draw a lightning bolt and its branches on the Pygame surface with varying widths.
    """
    scale_x = WINDOW_WIDTH / GRID_SIZE
    scale_y = WINDOW_HEIGHT / GRID_SIZE

    # Draw main bolt
    for i in range(1, len(path)):
        start = (path[i - 1][0] * scale_x, path[i - 1][1] * scale_y)
        end = (path[i][0] * scale_x, path[i][1] * scale_y)
        thickness = random.randint(3, 10)  # Vary main bolt thickness
        pygame.draw.line(surface, MAIN_BOLT_COLOR, start, end, thickness)

    # Draw branches
    for branch in branches:
        for i in range(1, len(branch)):
            start = (branch[i - 1][0] * scale_x, branch[i - 1][1] * scale_y)
            end = (branch[i][0] * scale_x, branch[i][1] * scale_y)
            thickness = random.randint(1, 4)  # Thinner branches
            pygame.draw.line(surface, MAIN_BOLT_COLOR, start, end, thickness)

def main():
    # Load the window overlay image
    try:
        overlay_image = pygame.image.load(WINDOW_PORT_IMAGE).convert_alpha()
        overlay_image = pygame.transform.scale(overlay_image, (WINDOW_WIDTH, WINDOW_HEIGHT))
    except pygame.error as e:
        print(f"Error loading overlay image: {e}")
        overlay_image = None

    running = True
    last_bolt_time = pygame.time.get_ticks() / 1000  # Current time in seconds

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Clear screen for next frame
        window.fill((0, 0, 0))

        # Trigger lightning bolts
        current_time = pygame.time.get_ticks() / 1000
        if current_time - last_bolt_time > COOLDOWN_TIME:
            start_x = random.randint(CENTER_THIRD_START, CENTER_THIRD_END)
            lightning_path, branches = trace_lightning(start_x)
            draw_lightning(window, lightning_path, branches)
            last_bolt_time = current_time

        # Overlay the window PNG if loaded
        if overlay_image:
            window.blit(overlay_image, (0, 0))

        # Update display
        pygame.display.flip()
        clock.tick(30)

    # Cleanup
    pygame.quit()

if __name__ == "__main__":
    main()

