import pygame
import numpy as np
import random

# Initialize Pygame
pygame.init()

# Screen dimensions
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
BACKGROUND_COLOR = (10, 10, 30)
PIPE_RADIUS = 30  # Radius for both the circle and the face size
PIPE_SPEED = 300  # Speed multiplier for pipe movement
FPS = 60

# Load emoji images
EMOJI_IMAGES = {
    "happy": pygame.image.load("happy.png"),
    "distressed": pygame.image.load("distressed.png"),
    "collision": pygame.image.load("collision.png"),
}

# Scale emoji images to match the circle size
for key in EMOJI_IMAGES:
    EMOJI_IMAGES[key] = pygame.transform.smoothscale(
        EMOJI_IMAGES[key], (PIPE_RADIUS * 2, PIPE_RADIUS * 2)
    )


class Pipe:
    def __init__(self):
        self.position = np.array(
            [random.randint(PIPE_RADIUS, SCREEN_WIDTH - PIPE_RADIUS),
             random.randint(PIPE_RADIUS, SCREEN_HEIGHT - PIPE_RADIUS)], dtype=float)
        self.velocity = np.random.uniform(-1, 1, 2)
        self.velocity /= np.linalg.norm(self.velocity)
        self.velocity *= PIPE_SPEED / FPS
        self.color = [random.randint(50, 255) for _ in range(3)]
        self.face = "happy"  # Default face
        self.emojis = EMOJI_IMAGES

    def update(self, dt):
        # Update position
        self.position += self.velocity * (dt / 1000)

        # Bounce off walls
        if self.position[0] - PIPE_RADIUS < 0 or self.position[0] + PIPE_RADIUS > SCREEN_WIDTH:
            self.velocity[0] *= -1
            self.change_face("distressed")
        if self.position[1] - PIPE_RADIUS < 0 or self.position[1] + PIPE_RADIUS > SCREEN_HEIGHT:
            self.velocity[1] *= -1
            self.change_face("distressed")

        # Gradually return to the happy face
        if self.face == "distressed":
            self.face = "happy"

    def change_face(self, new_face):
        self.face = new_face

    def draw(self, screen):
        # Draw the circle
        pygame.draw.circle(screen, self.color, self.position.astype(int), PIPE_RADIUS)

        # Draw emoji face, aligned to match the circle
        face_image = self.emojis[self.face]
        face_rect = face_image.get_rect(center=self.position.astype(int))  # Center the emoji on the circle
        screen.blit(face_image, face_rect)

        # Debugging rectangle around the emoji face (optional)
        pygame.draw.rect(screen, (0, 255, 0), face_rect, 1)  # Green bounding box


def main():
    # Initialize screen
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Pipe Visualizer with Emoji Faces")
    clock = pygame.time.Clock()

    # Create pipes
    pipes = [Pipe() for _ in range(10)]

    running = True
    while running:
        dt = clock.tick(FPS)

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Update pipes
        for pipe in pipes:
            pipe.update(dt)

        # Draw everything
        screen.fill(BACKGROUND_COLOR)
        for pipe in pipes:
            pipe.draw(screen)
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()

