"""
15_glitch_scroll.py — Glitch-aesthetic music video renderer.

White field + dark vertical streaks + sparse particles, bass-driven
horizontal scroll, onset-triggered inversion punches, heavy-bass
glitch tear blocks.  Headless; raw RGB24 frames piped to ffmpeg for
H.265/AAC output.  Same offline architecture as 14_music_video_render.py.

Input:  AUDIO_FILE env var, or ./assets/audio/audio_file.wav
Output: ./renders/<basename>_glitch_4k60.mp4

Usage:
    python3 15_glitch_scroll.py
    AUDIO_FILE=my_track.wav  python3 15_glitch_scroll.py
    SAVE_FIRST_FRAME=1       python3 15_glitch_scroll.py
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import sys
import time
from collections import deque

import numpy as np
from scipy.fftpack import fft
from scipy.io import wavfile

try:
    from PIL import Image
except ImportError:
    sys.exit("ERROR: Pillow not installed.  Run: pip install Pillow")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CHUNK = 1024
_BASS_ALPHA = 0.3          # exponential smoothing for bass band

WIDTH,  HEIGHT = 3840, 2160
FPS            = 60

VERTICAL_STREAK_COUNT = 100
STREAK_WIDTH_RANGE    = (2, 8)
STREAK_GRAY_RANGE     = (20, 80)   # very dark bars on white

PARTICLE_COUNT       = 300
PARTICLE_GRAY_RANGE  = (40, 120)
LARGE_DOT_COUNT      = 30
LARGE_DOT_COLOR      = (60, 110, 200)    # blue from aurora screenshot
PARTICLE_DRIFT_SPEED = 40.0              # pixels/sec downward

BASE_SCROLL_SPEED  = 10.0     # pixels/sec constant component
BASS_SCROLL_MULT   = 300.0    # additional pixels/sec per unit of bass energy
                               # tune upward if bass effect is too subtle

INVERSION_DURATION    = 0.16  # seconds total: 80ms hold + 80ms fade
ONSET_FLUX_MULTIPLIER = 1.5   # flux > rolling_avg * this → onset
ONSET_COOLDOWN        = 0.1   # seconds minimum between onsets

GLITCH_BASS_THRESHOLD = 0.3   # raw bass band energy; tune to taste
GLITCH_BLOCK_COUNT    = (1, 3)
GLITCH_HEIGHT_RANGE   = (5, 50)   # pixels per tear block
GLITCH_SHIFT_RANGE    = (-120, 120)   # horizontal pixel shift of tear

# ---------------------------------------------------------------------------
# Pre-generated streak and particle layouts — fixed relative positions,
# seeded so they're deterministic and consistent across runs.
# ---------------------------------------------------------------------------
_rng_s = np.random.default_rng(42)
_rng_p = np.random.default_rng(123)

_streak_x       = _rng_s.integers(0, WIDTH,  VERTICAL_STREAK_COUNT)
_streak_w       = _rng_s.integers(STREAK_WIDTH_RANGE[0],
                                   STREAK_WIDTH_RANGE[1] + 1,
                                   VERTICAL_STREAK_COUNT)
_streak_gray    = _rng_s.integers(STREAK_GRAY_RANGE[0],
                                   STREAK_GRAY_RANGE[1] + 1,
                                   VERTICAL_STREAK_COUNT)
_streak_h_frac  = _rng_s.uniform(0.3, 1.0, VERTICAL_STREAK_COUNT)
_streak_y0_frac = _rng_s.uniform(0.0, 0.3, VERTICAL_STREAK_COUNT)

_part_x      = _rng_p.integers(0, WIDTH,  PARTICLE_COUNT)
_part_y_base = _rng_p.integers(0, HEIGHT, PARTICLE_COUNT)
_part_gray   = _rng_p.integers(PARTICLE_GRAY_RANGE[0],
                                PARTICLE_GRAY_RANGE[1] + 1,
                                PARTICLE_COUNT)
_part_r      = _rng_p.integers(1, 4, PARTICLE_COUNT)   # 1–3 px box radius

_dot_x = _rng_p.integers(0, WIDTH,  LARGE_DOT_COUNT)
_dot_y = _rng_p.integers(0, HEIGHT, LARGE_DOT_COUNT)
_dot_r = _rng_p.integers(4, 10,     LARGE_DOT_COUNT)   # 4–9 px box radius

# Pre-render the static base canvas (white + streaks) once at import time.
# Copying it each frame is cheaper than re-drawing 100 streak slices.
_base_canvas = np.full((HEIGHT, WIDTH, 3), 255, dtype=np.uint8)
for _i in range(VERTICAL_STREAK_COUNT):
    _y0 = int(_streak_y0_frac[_i] * HEIGHT)
    _y1 = min(_y0 + int(_streak_h_frac[_i] * HEIGHT), HEIGHT)
    _x0 = int(_streak_x[_i])
    _x1 = min(_x0 + int(_streak_w[_i]), WIDTH)
    _base_canvas[_y0:_y1, _x0:_x1] = int(_streak_gray[_i])


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def load_audio(path: str) -> tuple[np.ndarray, int]:
    """Read wav → mono float32 normalised to ±0.95."""
    rate, data = wavfile.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    elif data.dtype != np.float32:
        data = data.astype(np.float32)
    peak = float(np.abs(data).max())
    if peak > 0:
        data = data / peak * 0.95
    return data, rate


def _bass_band(spectrum: np.ndarray, rate: int, n: int) -> float:
    """Mean amplitude of 0–200 Hz bins in a one-sided spectrum."""
    freqs = np.arange(len(spectrum)) * (rate / n)
    mask  = freqs < 200
    return float(spectrum[mask].mean()) if mask.any() else 0.0


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

def preflight(wav_path: str) -> None:
    if not os.path.exists(wav_path):
        sys.exit(f"ERROR: input file not found: {wav_path}")
    if not shutil.which('ffmpeg'):
        sys.exit("ERROR: ffmpeg not found.  Install: sudo apt install ffmpeg")
    probe = subprocess.run(
        ['ffmpeg', '-hide_banner', '-encoders'],
        capture_output=True, text=True,
    )
    if 'libx265' not in probe.stdout:
        sys.exit(
            "ERROR: ffmpeg missing libx265 support.\n"
            "  Ubuntu: sudo apt install ffmpeg  (universe repo includes x265)"
        )


# ---------------------------------------------------------------------------
# Per-frame rendering
# ---------------------------------------------------------------------------

def _draw_frame(
    frame_num: int,
    scroll_x: int,
    inv_alpha: float,
    bass_raw: float,
    rng_glitch: np.random.Generator,
) -> np.ndarray:
    """Return (HEIGHT, WIDTH, 3) uint8 RGB array for this frame."""

    # Start from the pre-rendered white+streaks canvas.
    arr = _base_canvas.copy()

    # --- small particles: drift downward at constant rate ---
    drift_px = int(frame_num * PARTICLE_DRIFT_SPEED / FPS) % HEIGHT
    for i in range(PARTICLE_COUNT):
        y = (_part_y_base[i] + drift_px) % HEIGHT
        r = int(_part_r[i])
        x = int(_part_x[i])
        g = int(_part_gray[i])
        arr[max(y - r, 0):min(y + r + 1, HEIGHT),
            max(x - r, 0):min(x + r + 1, WIDTH)] = g

    # --- large blue dots: fixed positions, no drift ---
    for i in range(LARGE_DOT_COUNT):
        r = int(_dot_r[i])
        y = int(_dot_y[i])
        x = int(_dot_x[i])
        arr[max(y - r, 0):min(y + r + 1, HEIGHT),
            max(x - r, 0):min(x + r + 1, WIDTH)] = LARGE_DOT_COLOR

    # --- horizontal scroll: roll entire canvas ---
    if scroll_x != 0:
        arr = np.roll(arr, scroll_x, axis=1)

    # --- glitch tear blocks on heavy bass ---
    if bass_raw >= GLITCH_BASS_THRESHOLD:
        n_blocks = int(rng_glitch.integers(
            GLITCH_BLOCK_COUNT[0], GLITCH_BLOCK_COUNT[1] + 1))
        for _ in range(n_blocks):
            y0    = int(rng_glitch.integers(0, HEIGHT))
            h     = int(rng_glitch.integers(
                GLITCH_HEIGHT_RANGE[0], GLITCH_HEIGHT_RANGE[1] + 1))
            y1    = min(y0 + h, HEIGHT)
            shift = int(rng_glitch.integers(
                GLITCH_SHIFT_RANGE[0], GLITCH_SHIFT_RANGE[1] + 1))
            slab  = arr[y0:y1, :, :].copy()
            arr[y0:y1, :, :] = np.roll(255 - slab, shift, axis=1)

    # --- inversion punch ---
    if inv_alpha >= 1.0:
        arr = 255 - arr
    elif inv_alpha > 0.0:
        arr = (arr.astype(np.float32) * (1.0 - inv_alpha)
               + (255 - arr).astype(np.float32) * inv_alpha
               ).astype(np.uint8)

    return arr


# ---------------------------------------------------------------------------
# Main render loop
# ---------------------------------------------------------------------------

def main() -> None:
    wav_path = os.environ.get('AUDIO_FILE', './assets/audio/audio_file.wav')
    preflight(wav_path)

    basename    = os.path.splitext(os.path.basename(wav_path))[0]
    os.makedirs('renders', exist_ok=True)
    output_path = f'renders/{basename}_glitch_4k60.mp4'

    print(f"[render] Input:  {wav_path}")
    print(f"[render] Output: {output_path}")

    audio, sample_rate = load_audio(wav_path)
    duration_sec  = len(audio) / sample_rate
    total_frames  = int(duration_sec * FPS)
    d_min, d_s    = divmod(int(duration_sec), 60)
    print(
        f"[render] Audio: {sample_rate} Hz, {d_min}:{d_s:02d}"
        f" → {total_frames} frames @ {FPS} fps"
    )

    ffmpeg_proc = subprocess.Popen(
        [
            'ffmpeg', '-y',
            '-f', 'rawvideo', '-pix_fmt', 'rgb24',
            '-s', f'{WIDTH}x{HEIGHT}', '-r', str(FPS),
            '-i', 'pipe:0',
            '-i', wav_path,
            '-c:v', 'libx265', '-preset', 'medium', '-crf', '23',
            '-c:a', 'aac', '-b:a', '192k',
            '-movflags', '+faststart',
            '-shortest',
            output_path,
        ],
        stdin=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )

    save_first_frame = os.environ.get('SAVE_FIRST_FRAME', '') == '1'

    dt             = 1.0 / FPS
    scroll_x_f     = 0.0          # float pixel accumulator
    bass_s         = 0.0          # smoothed bass energy
    inv_timer      = 0.0          # seconds of inversion remaining
    last_onset_frm = -9999
    prev_spectrum  = None
    flux_history   = deque(maxlen=int(0.5 * FPS))   # ~0.5s rolling window
    rng_glitch     = np.random.default_rng(7)        # reproducible glitch geometry
    render_start   = time.time()
    window_start   = render_start
    window_frames  = 0

    try:
        for frame_num in range(total_frames):
            # --- audio slice centred on playback position ---
            sample_idx = int(frame_num * sample_rate / FPS)
            chunk      = np.zeros(CHUNK, dtype=np.float32)
            src_end    = min(sample_idx + CHUNK, len(audio))
            src_len    = src_end - sample_idx
            if src_len > 0:
                chunk[:src_len] = audio[sample_idx:src_end]

            rms      = float(np.sqrt(np.mean(chunk ** 2)))
            N        = len(chunk)
            spectrum = np.abs(fft(chunk)[:N // 2]) * (2.0 / N)

            bass_raw = _bass_band(spectrum, sample_rate, N)
            bass_s   = bass_s + _BASS_ALPHA * (bass_raw - bass_s)
            bass_s   = max(bass_s, 1e-5)

            # --- onset detection: spectral flux vs rolling average ---
            if prev_spectrum is None:
                prev_spectrum = spectrum.copy()
            flux = float(np.sum(np.maximum(spectrum - prev_spectrum, 0.0)))
            flux_history.append(flux)
            flux_avg    = float(np.mean(flux_history)) if flux_history else 0.0
            cooldown_ok = (frame_num - last_onset_frm) > int(ONSET_COOLDOWN * FPS)
            onset       = (cooldown_ok
                           and flux_avg > 0.0
                           and flux > flux_avg * ONSET_FLUX_MULTIPLIER)
            if onset:
                last_onset_frm = frame_num
                inv_timer      = INVERSION_DURATION
            prev_spectrum = spectrum

            # --- inversion alpha: hold for first half, fade over second half ---
            inv_alpha = 0.0
            if inv_timer > 0.0:
                half = INVERSION_DURATION / 2.0
                inv_alpha = 1.0 if inv_timer > half else inv_timer / half
                inv_timer = max(inv_timer - dt, 0.0)

            # --- scroll accumulation ---
            scroll_x_f += (BASE_SCROLL_SPEED + bass_s * BASS_SCROLL_MULT) * dt
            scroll_x    = int(scroll_x_f) % WIDTH

            # --- draw ---
            arr = _draw_frame(frame_num, scroll_x, inv_alpha, bass_raw, rng_glitch)

            if save_first_frame and frame_num == 0:
                Image.fromarray(arr, mode='RGB').save('renders/first_frame_glitch.png')
                print("[render] Saved → renders/first_frame_glitch.png", flush=True)

            ffmpeg_proc.stdin.write(np.ascontiguousarray(arr).tobytes())

            # --- progress + diagnostics every 60 frames (= 1 s of video) ---
            window_frames += 1
            if window_frames == 60 or frame_num == total_frames - 1:
                now        = time.time()
                fps_render = window_frames / max(now - window_start, 1e-6)
                remaining  = total_frames - frame_num - 1
                eta_sec    = int(remaining / fps_render) if fps_render > 0 else 0
                music_pos  = (frame_num + 1) / FPS
                m_min, m_s = divmod(int(music_pos), 60)
                e_min, e_s = divmod(eta_sec, 60)
                e_hr, e_m  = divmod(e_min, 60)
                eta_str    = f"{e_hr}h {e_m:02d}m" if e_hr else f"{e_m}:{e_s:02d}"
                pct        = (frame_num + 1) / total_frames * 100
                print(
                    f"[render] frame {frame_num+1}/{total_frames}"
                    f"  ({pct:.1f}% — {m_min}:{m_s:02d}"
                    f" — ETA {eta_str})",
                    flush=True,
                )
                print(
                    f"[debug]  frame={frame_num}"
                    f"  rms={rms:.4f}"
                    f"  bass_raw={bass_raw:.5f}  bass_s={bass_s:.5f}"
                    f"  flux={flux:.3f}  flux_avg={flux_avg:.3f}"
                    f"  onset={onset}  inv={inv_alpha:.2f}",
                    flush=True,
                )
                window_start  = now
                window_frames = 0

        ffmpeg_proc.stdin.close()
        ret = ffmpeg_proc.wait()
        if ret != 0:
            sys.exit(f"[render] ERROR: ffmpeg exited with code {ret}")

    except KeyboardInterrupt:
        print("\n[render] Interrupted — terminating ffmpeg...", file=sys.stderr)
        try:
            ffmpeg_proc.stdin.close()
        except OSError:
            pass
        ffmpeg_proc.terminate()
        try:
            ffmpeg_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            ffmpeg_proc.kill()
        sys.exit(1)

    size_mb  = os.path.getsize(output_path) / (1024 * 1024)
    elapsed  = time.time() - render_start
    e_min, e_s = divmod(int(elapsed), 60)
    print(
        f"[render] Done in {e_min}:{e_s:02d}."
        f"  Output: {output_path} ({size_mb:.1f} MB)"
    )


if __name__ == "__main__":
    main()
