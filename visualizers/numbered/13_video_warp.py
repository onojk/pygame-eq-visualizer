"""
13_video_warp.py  —  Live procedural plasma, audio-warped.

No video file required. Generates a flowing plasma background each frame
and distorts it in real time using live audio FFT bands.

Audio source: PulseAudio monitor (system audio). See commit 4 for
  full monitor detection; this commit uses the default input device.
"""

from __future__ import annotations

import math
import sys

import numpy as np
import pyaudio
import pygame
from scipy.fftpack import fft

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
CHUNK   = 1024   # audio buffer (samples)
RATE    = 22050  # sample rate (Hz)
FPS     = 30     # target frame rate

WIDTH,  HEIGHT  = 1280, 720   # display resolution
PLAS_W, PLAS_H  = 640,  360   # internal plasma resolution (scaled up 2×)

TAU = math.tau  # 2π

# ---------------------------------------------------------------------------
# Coordinate grids — computed once, reused every frame.
# _px is (1, PLAS_W) and _py is (PLAS_H, 1) so arithmetic between them
# broadcasts to (PLAS_H, PLAS_W) without an explicit meshgrid.
# ---------------------------------------------------------------------------
_px = np.linspace(0.0, TAU, PLAS_W, dtype=np.float32).reshape(1, PLAS_W)
_py = np.linspace(0.0, TAU, PLAS_H, dtype=np.float32).reshape(PLAS_H, 1)


# ---------------------------------------------------------------------------
# Plasma generator
# ---------------------------------------------------------------------------

def generate_plasma(t: float) -> np.ndarray:
    """
    Return (PLAS_H, PLAS_W, 3) uint8 RGB array.
    Three independent sine-field layers give R, G, B channels.
    Values cycle smoothly with t (time in seconds).
    """
    r = np.sin(_px * 1.30 + t * 0.71) + np.sin(_py * 0.90 + t * 1.13)
    g = np.sin(_px * 0.70 + t * 0.93) + np.cos(_py * 1.10 + t * 0.67)
    b = np.cos(_px * 1.10 + t * 0.53) + np.sin(_py * 0.80 + t * 1.31)
    # Each array is (PLAS_H, PLAS_W), values in [-2, 2]; map to [0, 255]
    k = 255.0 / 4.0
    return np.stack([
        ((r + 2.0) * k).astype(np.uint8),
        ((g + 2.0) * k).astype(np.uint8),
        ((b + 2.0) * k).astype(np.uint8),
    ], axis=2)  # (PLAS_H, PLAS_W, 3)


def array_to_surface(arr: np.ndarray) -> pygame.Surface:
    """Convert (H, W, 3) uint8 ndarray → pygame Surface."""
    # surfarray.make_surface expects (W, H, 3)
    return pygame.surfarray.make_surface(np.ascontiguousarray(arr.swapaxes(0, 1)))


# ---------------------------------------------------------------------------
# Audio helpers (basic — monitor detection added in commit 4)
# ---------------------------------------------------------------------------

def open_audio_stream(pa: pyaudio.PyAudio) -> pyaudio.Stream | None:
    try:
        return pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
        )
    except OSError as exc:
        print(f"[audio] Could not open stream: {exc}", file=sys.stderr)
        return None


def drain_audio(stream: pyaudio.Stream | None) -> None:
    """Read and discard one chunk to keep the buffer from backing up."""
    if stream is None:
        return
    try:
        stream.read(CHUNK, exception_on_overflow=False)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    pygame.init()
    pygame.display.set_caption("Plasma Warp")
    window = pygame.display.set_mode((WIDTH, HEIGHT), pygame.DOUBLEBUF)
    clock  = pygame.time.Clock()

    pa     = pyaudio.PyAudio()
    stream = open_audio_stream(pa)

    t       = 0.0
    running = True

    while running:
        dt = clock.tick(FPS) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        drain_audio(stream)

        plasma = generate_plasma(t)
        surf   = array_to_surface(plasma)
        scaled = pygame.transform.scale(surf, (WIDTH, HEIGHT))
        window.blit(scaled, (0, 0))
        pygame.display.flip()

        t += dt

    if stream:
        stream.stop_stream()
        stream.close()
    pa.terminate()
    pygame.quit()


if __name__ == "__main__":
    main()
