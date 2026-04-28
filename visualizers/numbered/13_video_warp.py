"""
13_video_warp.py  —  Live procedural plasma, audio-warped.

No video file required. Generates a flowing plasma background each frame
and distorts it in real time using live audio FFT bands.

Audio source: PulseAudio monitor (system audio — whatever is playing
  through the speakers). Requires a monitor source to be visible to
  sounddevice; exits with an error and instructions if none is found.
  Override with AUDIO_INPUT_DEVICE env var (device index or name substring).
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pygame
import sounddevice as sd
from scipy.fftpack import fft

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
CHUNK         = 1024   # audio buffer (samples)
RATE_FALLBACK = 48000  # used when device native rate is unavailable
FPS           = 30     # target frame rate
_ALPHA        = 0.3    # exponential smoothing for FFT bands (0=frozen, 1=raw)

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

def generate_plasma(
    t: float,
    bass: float = 0.0,
    mid: float = 0.0,
    high: float = 0.0,
) -> np.ndarray:
    """Return (PLAS_H, PLAS_W, 3) uint8 RGB array, warped by audio bands.

    bass → large slow displacement of both axes
    mid  → cross-coupled swirl
    high → fine fast jitter
    """
    px = _px + bass * 6.0 * np.sin(_py * 0.5 + t * 0.4)
    py = _py + bass * 6.0 * np.cos(_px * 0.5 + t * 0.4)
    px = px  + mid  * 2.5 * np.cos(py + t * 0.7)
    py = py  + mid  * 2.5 * np.sin(px + t * 0.7)
    px = px  + high * 0.8 * np.sin(px * 5.0 + t * 8.0)
    py = py  + high * 0.8 * np.cos(py * 5.0 + t * 8.0)

    r = np.sin(px * 1.30 + t * 0.71) + np.sin(py * 0.90 + t * 1.13)
    g = np.sin(px * 0.70 + t * 0.93) + np.cos(py * 1.10 + t * 0.67)
    b = np.cos(px * 1.10 + t * 0.53) + np.sin(py * 0.80 + t * 1.31)
    k = 255.0 / 4.0
    return np.stack([
        ((r + 2.0) * k).astype(np.uint8),
        ((g + 2.0) * k).astype(np.uint8),
        ((b + 2.0) * k).astype(np.uint8),
    ], axis=2)


def array_to_surface(arr: np.ndarray) -> pygame.Surface:
    """Convert (H, W, 3) uint8 ndarray → pygame Surface."""
    # surfarray.make_surface expects (W, H, 3)
    return pygame.surfarray.make_surface(np.ascontiguousarray(arr.swapaxes(0, 1)))


# ---------------------------------------------------------------------------
# Audio — monitor detection (fail loud)
# ---------------------------------------------------------------------------

def _device_rate(info: dict) -> int:
    """Extract native sample rate from device info; fall back to RATE_FALLBACK."""
    try:
        r = int(info['default_samplerate'])
        return r if r > 0 else RATE_FALLBACK
    except (KeyError, ValueError, TypeError):
        return RATE_FALLBACK


def find_monitor_device() -> tuple[int, dict]:
    """
    Return (device_index, device_info) for system-audio capture.

    Resolution order:
      1. AUDIO_INPUT_DEVICE env var — checked first, skips scanning entirely.
         - Digit string  → validate that index, fail loud if invalid.
         - Other string  → case-insensitive substring match across all devices.
      2. Scan all devices for a name containing 'monitor' or 'pulse'.
         'pulse' is the PipeWire→PulseAudio bridge; routing is done via
         pavucontrol's Recording tab after the stream is open.
      3. Nothing found → print instructions and sys.exit(1).
    """
    override = os.environ.get("AUDIO_INPUT_DEVICE")

    # --- env var path ---
    if override is not None:
        if override.isdigit():
            idx = int(override)
            try:
                info = dict(sd.query_devices(idx))
                print(f"[audio] AUDIO_INPUT_DEVICE={override}: [{idx}] {info['name']}")
                return idx, info
            except (ValueError, sd.PortAudioError):
                print(
                    f"[audio] ERROR: AUDIO_INPUT_DEVICE={override} is not a "
                    "valid device index on this system.",
                    file=sys.stderr,
                )
                sys.exit(1)

        # substring match
        for i, info in enumerate(sd.query_devices()):
            if override.lower() in info['name'].lower() and info['max_input_channels'] > 0:
                print(f"[audio] AUDIO_INPUT_DEVICE={override!r}: [{i}] {info['name']}")
                return i, dict(info)
        print(
            f"[audio] ERROR: AUDIO_INPUT_DEVICE={override!r} matched no device.\n"
            f"  Run with no AUDIO_INPUT_DEVICE set to see available devices.",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- auto-scan path ---
    for i, info in enumerate(sd.query_devices()):
        name = info['name'].lower()
        if info['max_input_channels'] > 0 and (name == 'pulse' or 'monitor' in name):
            print(f"[audio] System-audio source: [{i}] {info['name']}")
            return i, dict(info)

    # Print available devices to help the user pick one manually
    print("\n[audio] Available input devices:", file=sys.stderr)
    for i, info in enumerate(sd.query_devices()):
        if info['max_input_channels'] > 0:
            print(f"  [{i}] {info['name']}", file=sys.stderr)

    print(
        "\n[audio] ERROR: No monitor or pulse source found.\n"
        "  To capture system audio:\n"
        "      pactl load-module module-loopback latency_msec=1\n"
        "  Or open pavucontrol → Recording tab and route this app's input to\n"
        "  'Monitor of <your output device>'.\n"
        "  On PipeWire, try: AUDIO_INPUT_DEVICE=pulse python3 main.py --run 13_video_warp.py\n"
        "  Or set AUDIO_INPUT_DEVICE=<index> using one of the indices listed above.\n",
        file=sys.stderr,
    )
    sys.exit(1)


def open_audio_stream(device_index: int, rate: int) -> sd.InputStream:
    stream = sd.InputStream(
        device=device_index,
        samplerate=rate,
        channels=1,
        blocksize=CHUNK,
        dtype='float32',
    )
    stream.start()
    return stream


def _smooth_bands(
    chunk: np.ndarray,
    rate: int,
    bass_s: float,
    mid_s: float,
    high_s: float,
) -> tuple[float, float, float]:
    """FFT the mono float32 chunk; return exponentially smoothed bass/mid/high."""
    N        = len(chunk)
    spectrum = np.abs(fft(chunk)[:N // 2]) * (2.0 / N)
    freqs    = np.arange(N // 2) * (rate / N)

    bass_raw = float(spectrum[freqs <  200].mean()) if (freqs <  200).any() else 0.0
    mid_mask = (freqs >= 200) & (freqs < 2000)
    mid_raw  = float(spectrum[mid_mask].mean())     if mid_mask.any()       else 0.0
    high_raw = float(spectrum[freqs >= 2000].mean()) if (freqs >= 2000).any() else 0.0

    return (
        bass_s + _ALPHA * (bass_raw - bass_s),
        mid_s  + _ALPHA * (mid_raw  - mid_s),
        high_s + _ALPHA * (high_raw - high_s),
    )


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    monitor_idx, dev_info = find_monitor_device()   # exits 1 if no monitor found
    rate                  = _device_rate(dev_info)
    print(f"[audio] Using device [{monitor_idx}] '{dev_info['name']}' at {rate} Hz")
    stream = open_audio_stream(monitor_idx, rate)

    pygame.init()
    pygame.display.set_caption("Plasma Warp")
    window = pygame.display.set_mode((WIDTH, HEIGHT), pygame.DOUBLEBUF)
    clock  = pygame.time.Clock()

    t      = 0.0
    bass_s = mid_s = high_s = 0.0
    running = True

    try:
        while running:
            dt = clock.tick(FPS) / 1000.0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False

            audio_chunk, _ = stream.read(CHUNK)
            bass_s, mid_s, high_s = _smooth_bands(
                audio_chunk[:, 0], rate, bass_s, mid_s, high_s,
            )

            plasma = generate_plasma(t, bass_s, mid_s, high_s)
            surf   = array_to_surface(plasma)
            scaled = pygame.transform.scale(surf, (WIDTH, HEIGHT))
            window.blit(scaled, (0, 0))
            pygame.display.flip()

            t += dt
    finally:
        stream.stop()
        stream.close()
        pygame.quit()


if __name__ == "__main__":
    main()
