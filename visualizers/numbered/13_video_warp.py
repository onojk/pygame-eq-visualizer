"""
13_video_warp.py  —  Live procedural plasma, audio-warped.

No video file required. Generates a flowing plasma background each frame
and distorts it in real time using live audio FFT bands.

Audio source: PipeWire monitor captured via pw-record subprocess.
  Discovers the monitor source automatically using pactl; starts pw-record
  as a child process and reads raw float32 PCM from its stdout on a daemon
  thread. No PortAudio / sounddevice in the hot path — zero playback interference.
  Override target with AUDIO_INPUT_DEVICE env var (passed directly to pw-record --target).
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import sys
import threading

import numpy as np
import pygame
from scipy.fftpack import fft

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
CHUNK         = 1024   # audio buffer (samples)
RATE_FALLBACK = 48000  # used when device native rate is unavailable
FPS           = 30     # target frame rate
_ALPHA        = 0.3    # exponential smoothing for FFT bands (0=frozen, 1=raw)

# Warp scale — applied after sqrt() compression of smoothed band values.
# sqrt(band_mean≈0.01) ≈ 0.1; BASS_SCALE=50 → displacement ≈ 5 rad on 2π grid.
BASS_SCALE = 50.0
MID_SCALE  = 25.0
HIGH_SCALE = 12.0

WIDTH,  HEIGHT  = 1280, 720   # display resolution
PLAS_W, PLAS_H  = 640,  360   # internal plasma resolution (scaled up 2×)

TAU = math.tau  # 2π

# ---------------------------------------------------------------------------
# Overlay — amplitude bars drawn on every frame for visual diagnostics
# ---------------------------------------------------------------------------
_OV_MAX_W    = int(WIDTH * 0.8)           # max bar width in pixels
_OV_X0       = (WIDTH - _OV_MAX_W) // 2  # left edge of all bars
_OV_BAR_Y    = HEIGHT - 40               # top of the main amplitude bar
_OV_BAR_H    = 20                        # main bar height
_OV_BAND_H   = 8                         # per-band bar height
_OV_GAP      = 3                         # gap between bars
# Scales below apply to sqrt-compressed values; sqrt(0.01)≈0.1 is the baseline.
_OV_AMP_SCL  = 8.0   # sqrt(RMS) → fill  (RMS≈0.01 → sqrt≈0.1 → 80% at scale 8)
_OV_BAND_SCL = 5.0   # sqrt(band) → fill (band≈0.01 → sqrt≈0.1 → 50% at scale 5)

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
    px = _px + bass * np.sin(_py * 0.5 + t * 0.4)
    py = _py + bass * np.cos(_px * 0.5 + t * 0.4)
    px = px  + mid  * np.cos(py + t * 0.7)
    py = py  + mid  * np.sin(px + t * 0.7)
    px = px  + high * np.sin(px * 5.0 + t * 8.0)
    py = py  + high * np.cos(py * 5.0 + t * 8.0)

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


def draw_overlay(
    surface: pygame.Surface,
    rms: float,
    bass: float,
    mid: float,
    high: float,
    font: pygame.font.Font,
) -> None:
    """Draw amplitude bars and RMS readout at the bottom of the window."""
    # Main bar (cyan) — overall amplitude, sqrt-compressed
    amp_w = int(_OV_MAX_W * min(math.sqrt(max(rms, 0.0)) * _OV_AMP_SCL, 1.0))
    pygame.draw.rect(surface, (0, 255, 255),
                     (_OV_X0, _OV_BAR_Y, max(amp_w, 1), _OV_BAR_H))

    # Band bars stacked above main bar: bass top, mid middle, high bottom
    y = _OV_BAR_Y - _OV_GAP
    for value, color in (
        (high, (60,  120, 255)),   # blue  — high
        (mid,  (0,   220,   0)),   # green — mid
        (bass, (255,  60,  60)),   # red   — bass
    ):
        y -= _OV_BAND_H
        bw = int(_OV_MAX_W * min(math.sqrt(max(value, 0.0)) * _OV_BAND_SCL, 1.0))
        pygame.draw.rect(surface, color, (_OV_X0, y, max(bw, 1), _OV_BAND_H))
        y -= _OV_GAP

    # RMS numeric readout to the right of the bar area
    label = font.render(f"RMS: {rms:.4f}", True, (255, 255, 255))
    surface.blit(label, (_OV_X0 + _OV_MAX_W + 8, _OV_BAR_Y + 2))


# ---------------------------------------------------------------------------
# Shared audio state — written by the reader thread, read by the render loop.
# _latest_chunk always holds the most recent CHUNK mono float32 samples.
# ---------------------------------------------------------------------------
_audio_lock   = threading.Lock()
_latest_chunk = np.zeros(CHUNK, dtype='float32')


# ---------------------------------------------------------------------------
# Audio — pw-record subprocess capture
# ---------------------------------------------------------------------------

def find_monitor_source() -> str:
    """
    Return a PipeWire monitor source name suitable for pw-record --target.

    Resolution order:
      1. AUDIO_INPUT_DEVICE env var — used directly as the target name.
      2. Parse 'pactl list sources short': prefer RUNNING monitor lines,
         fall back to any monitor line. Among candidates, prefer 'Speaker'
         over HDMI/Headphones. Fail loud if nothing found.
    """
    override = os.environ.get('AUDIO_INPUT_DEVICE')
    if override is not None:
        return override

    try:
        out = subprocess.check_output(
            ['pactl', 'list', 'sources', 'short'],
            text=True, stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        print(f"[audio] ERROR: pactl failed: {exc}", file=sys.stderr)
        sys.exit(1)

    running, fallback = [], []
    for line in out.splitlines():
        cols = line.split()
        if len(cols) < 2:
            continue
        name = cols[1]
        if 'monitor' not in name.lower():
            continue
        (running if cols[-1] == 'RUNNING' else fallback).append(name)

    candidates = running or fallback
    if not candidates:
        print(
            "[audio] ERROR: No monitor source found via pactl.\n"
            "  pactl output:\n" + "\n".join(f"    {l}" for l in out.splitlines()),
            file=sys.stderr,
        )
        sys.exit(1)

    def _score(name: str) -> int:
        n = name.lower()
        if 'speaker' in n:
            return 0
        if 'hdmi' in n or 'headphone' in n:
            return 2
        return 1

    candidates.sort(key=_score)
    return candidates[0]


def start_pw_record(monitor_name: str) -> subprocess.Popen:
    """
    Spawn pw-record capturing monitor_name as raw float32 mono at 44100 Hz.
    Starts a daemon reader thread that keeps _latest_chunk current.
    Returns the Popen object so the caller can terminate it on exit.
    """
    if not shutil.which('pw-record'):
        print(
            "[audio] ERROR: pw-record not found. "
            "Install with: sudo apt install pipewire-bin",
            file=sys.stderr,
        )
        sys.exit(1)

    proc = subprocess.Popen(
        [
            'pw-record',
            f'--target={monitor_name}',
            '--format=f32',
            '--rate=44100',
            '--channels=1',
            '-',
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=0,
    )
    print(f"[audio] Capturing monitor source: {monitor_name}")
    print(f"[audio] pw-record PID: {proc.pid}")

    def _reader():
        global _latest_chunk
        read_bytes = CHUNK * 4  # float32 = 4 bytes per sample
        while True:
            raw = proc.stdout.read(read_bytes)
            if not raw or len(raw) < read_bytes:
                break
            chunk = np.frombuffer(raw, dtype=np.float32)
            with _audio_lock:
                _latest_chunk = chunk.copy()

    threading.Thread(target=_reader, daemon=True).start()
    return proc


def _smooth_bands(
    chunk: np.ndarray,
    rate: int,
    bass_s: float,
    mid_s: float,
    high_s: float,
) -> tuple[float, float, float, float, float, float]:
    """FFT the mono float32 chunk; return (smoothed×3, raw×3) band energies."""
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
        bass_raw, mid_raw, high_raw,
    )


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    debug = os.environ.get('DEBUG_AUDIO', '') == '1'

    monitor_name = find_monitor_source()   # exits 1 if no monitor found
    proc         = start_pw_record(monitor_name)
    rate         = 44100                   # matches --rate flag passed to pw-record

    pygame.init()
    pygame.display.set_caption("Plasma Warp")
    window = pygame.display.set_mode((WIDTH, HEIGHT), pygame.DOUBLEBUF)
    clock  = pygame.time.Clock()
    font   = pygame.font.SysFont(None, 20)

    t      = 0.0
    bass_s = mid_s = high_s = 0.0
    frame  = 0
    running = True

    try:
        while running:
            dt = clock.tick(FPS) / 1000.0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False

            # Grab the latest chunk deposited by the callback thread.
            # If the callback hasn't fired yet, _latest_chunk is all zeros — fine.
            with _audio_lock:
                mono = _latest_chunk.copy()

            rms    = float(np.sqrt(np.mean(mono ** 2)))
            bass_s, mid_s, high_s, bass_r, mid_r, high_r = _smooth_bands(
                mono, rate, bass_s, mid_s, high_s,
            )
            frame += 1
            if debug and frame % 10 == 0:
                print(
                    f"[audio] raw: B={bass_r:.4f} M={mid_r:.4f} H={high_r:.4f}"
                    f"  smooth: B={bass_s:.4f} M={mid_s:.4f} H={high_s:.4f}"
                )

            # Compress dynamic range with sqrt before scaling to displacement.
            bass_v = math.sqrt(max(bass_s, 0.0)) * BASS_SCALE
            mid_v  = math.sqrt(max(mid_s,  0.0)) * MID_SCALE
            high_v = math.sqrt(max(high_s, 0.0)) * HIGH_SCALE

            plasma = generate_plasma(t, bass_v, mid_v, high_v)
            surf   = array_to_surface(plasma)
            scaled = pygame.transform.scale(surf, (WIDTH, HEIGHT))
            window.blit(scaled, (0, 0))
            draw_overlay(window, rms, bass_s, mid_s, high_s, font)
            pygame.display.flip()

            t += dt
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
        pygame.quit()


if __name__ == "__main__":
    main()
