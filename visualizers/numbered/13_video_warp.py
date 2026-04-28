"""
13_video_warp.py  —  Live procedural plasma, audio-warped.

No video file required. Generates a flowing plasma background each frame
and distorts it in real time using live audio FFT bands.

Audio source: PulseAudio monitor (system audio — whatever is playing
  through the speakers). Requires a monitor source to be visible to
  PyAudio; exits with an error and instructions if none is found.
  Override with AUDIO_INPUT_DEVICE env var (device index or name substring).
"""

from __future__ import annotations

import math
import os
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
# Audio — monitor detection (fail loud)
# ---------------------------------------------------------------------------

def find_monitor_device(pa: pyaudio.PyAudio) -> int:
    """
    Return the PyAudio device index of a PulseAudio monitor source.

    Resolution order:
      1. AUDIO_INPUT_DEVICE env var — numeric index takes priority,
         otherwise treated as a case-insensitive name substring.
      2. First device whose name contains 'monitor' (case-insensitive).
      3. Neither found → print instructions and sys.exit(1).
    """
    override = os.environ.get("AUDIO_INPUT_DEVICE")
    if override is not None:
        if override.isdigit():
            idx = int(override)
            name = pa.get_device_info_by_index(idx)["name"]
            print(f"[audio] AUDIO_INPUT_DEVICE override: [{idx}] {name}")
            return idx
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if override.lower() in info["name"].lower() and info["maxInputChannels"] > 0:
                print(f"[audio] AUDIO_INPUT_DEVICE override: [{i}] {info['name']}")
                return i
        print(
            f"[audio] AUDIO_INPUT_DEVICE={override!r} matched no device — "
            "falling through to monitor scan.",
            file=sys.stderr,
        )

    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if "monitor" in info["name"].lower() and info["maxInputChannels"] > 0:
            print(f"[audio] Monitor source: [{i}] {info['name']}")
            return i

    print(
        "\n[audio] ERROR: No PulseAudio monitor source found.\n"
        "  To capture system audio, run:\n"
        "      pactl load-module module-loopback latency_msec=1\n"
        "  Or open pavucontrol → Recording tab and set this app's source to\n"
        "  'Monitor of <your output device>'.\n"
        "  Then re-run this script.\n",
        file=sys.stderr,
    )
    pa.terminate()
    sys.exit(1)


def open_audio_stream(pa: pyaudio.PyAudio, device_index: int) -> pyaudio.Stream:
    return pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=RATE,
        input=True,
        input_device_index=device_index,
        frames_per_buffer=CHUNK,
    )


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    # Detect monitor source before opening a window — fail loud if absent.
    pa           = pyaudio.PyAudio()
    monitor_idx  = find_monitor_device(pa)   # exits 1 if no monitor found
    # Stream opened in next commit when FFT warp is wired up.
    pa.terminate()

    pygame.init()
    pygame.display.set_caption("Plasma Warp")
    window = pygame.display.set_mode((WIDTH, HEIGHT), pygame.DOUBLEBUF)
    clock  = pygame.time.Clock()

    t       = 0.0
    running = True

    while running:
        dt = clock.tick(FPS) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        plasma = generate_plasma(t)
        surf   = array_to_surface(plasma)
        scaled = pygame.transform.scale(surf, (WIDTH, HEIGHT))
        window.blit(scaled, (0, 0))
        pygame.display.flip()

        t += dt

    pygame.quit()


if __name__ == "__main__":
    main()
