"""
14_music_video_render.py  —  Offline 4K@60fps music video renderer.

Reads a wav file and produces a plasma music video matching the visuals
of 13_video_warp.py. Headless: no window, no live audio capture. Raw
RGB24 frames are piped to ffmpeg for H.265/AAC muxing.

Input:  AUDIO_FILE env var, or ./assets/audio/audio_file.wav
Output: ./renders/<basename>_4k60.mp4  (creates renders/ if absent)

Usage:
    python3 14_music_video_render.py
    AUDIO_FILE=my_track.wav python3 14_music_video_render.py
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import sys
import time

# Must be set before pygame is imported to suppress window/audio init.
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')

import numpy as np
import pygame
from scipy.fftpack import fft
from scipy.io import wavfile

# ---------------------------------------------------------------------------
# CONFIG — copied verbatim from 13_video_warp.py
# ---------------------------------------------------------------------------
CHUNK      = 1024   # audio samples per FFT window, matches 13
_ALPHA     = 0.3    # exponential smoothing for FFT bands (0=frozen, 1=raw)

BASS_SCALE = 50.0
MID_SCALE  = 25.0
HIGH_SCALE = 12.0

WIDTH,  HEIGHT  = 3840, 2160   # 4K UHD output resolution
PLAS_W, PLAS_H  = 640,  360    # internal plasma resolution (scaled up 6×)
FPS             = 60            # output frame rate

TAU = math.tau  # 2π

# ---------------------------------------------------------------------------
# Coordinate grids — same shape and range as 13; computed once.
# ---------------------------------------------------------------------------
_px = np.linspace(0.0, TAU, PLAS_W, dtype=np.float32).reshape(1, PLAS_W)
_py = np.linspace(0.0, TAU, PLAS_H, dtype=np.float32).reshape(PLAS_H, 1)


# ---------------------------------------------------------------------------
# Plasma generator — verbatim copy from 13_video_warp.py
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


# ---------------------------------------------------------------------------
# FFT band smoothing — verbatim copy from 13_video_warp.py
# ---------------------------------------------------------------------------

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
# Audio loading
# ---------------------------------------------------------------------------

def load_audio(path: str) -> tuple[np.ndarray, int]:
    """Read wav → mono float32 in [-1, 1] and its sample rate."""
    rate, data = wavfile.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    elif data.dtype != np.float32:
        data = data.astype(np.float32)
    return data, rate


# ---------------------------------------------------------------------------
# Preflight checks
# ---------------------------------------------------------------------------

def preflight(wav_path: str) -> None:
    if not os.path.exists(wav_path):
        sys.exit(f"ERROR: input file not found: {wav_path}")

    if not shutil.which('ffmpeg'):
        sys.exit("ERROR: ffmpeg not found in PATH. Install with: sudo apt install ffmpeg")

    probe = subprocess.run(
        ['ffmpeg', '-hide_banner', '-encoders'],
        capture_output=True, text=True,
    )
    if 'libx265' not in probe.stdout:
        sys.exit(
            "ERROR: ffmpeg does not have libx265 support.\n"
            "  Ubuntu: sudo apt install ffmpeg  (universe repo includes x265)\n"
            "  Or build ffmpeg with --enable-libx265"
        )


# ---------------------------------------------------------------------------
# Main render loop
# ---------------------------------------------------------------------------

def main() -> None:
    wav_path = os.environ.get('AUDIO_FILE', './assets/audio/audio_file.wav')
    preflight(wav_path)

    basename    = os.path.splitext(os.path.basename(wav_path))[0]
    os.makedirs('renders', exist_ok=True)
    output_path = f'renders/{basename}_4k60.mp4'

    print(f"[render] Input:  {wav_path}")
    print(f"[render] Output: {output_path}")

    audio, sample_rate = load_audio(wav_path)
    duration_sec = len(audio) / sample_rate
    total_frames = int(duration_sec * FPS)
    d_min, d_s   = divmod(int(duration_sec), 60)
    print(
        f"[render] Audio: {sample_rate} Hz, {d_min}:{d_s:02d}"
        f" → {total_frames} frames at {FPS} fps"
    )

    # Spawn ffmpeg: reads raw RGB24 frames from stdin, audio directly from file.
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
        stderr=subprocess.DEVNULL,  # suppress per-frame encoding noise
    )

    pygame.init()

    # depth=24 (no alpha) gives a packed RGB surface with consistent channel
    # ordering when read back via surfarray.  Created once; reused every frame.
    canvas4k = pygame.Surface((WIDTH, HEIGHT), 0, 24)

    save_first_frame = os.environ.get('SAVE_FIRST_FRAME', '') == '1'

    bass_s = mid_s = high_s = 0.0
    render_start  = time.time()
    window_start  = render_start
    window_frames = 0

    try:
        for frame_num in range(total_frames):
            t = frame_num / FPS

            # Slice CHUNK samples centred on the playback position for this frame.
            sample_idx = int(frame_num * sample_rate / FPS)
            chunk      = np.zeros(CHUNK, dtype=np.float32)
            src_end    = min(sample_idx + CHUNK, len(audio))
            src_len    = src_end - sample_idx
            if src_len > 0:
                chunk[:src_len] = audio[sample_idx:src_end]

            bass_s, mid_s, high_s, _, _, _ = _smooth_bands(
                chunk, sample_rate, bass_s, mid_s, high_s,
            )

            bass_v = math.sqrt(max(bass_s, 0.0)) * BASS_SCALE
            mid_v  = math.sqrt(max(mid_s,  0.0)) * MID_SCALE
            high_v = math.sqrt(max(high_s, 0.0)) * HIGH_SCALE

            # Generate plasma at internal resolution, smoothscale into canvas4k.
            plasma = generate_plasma(t, bass_v, mid_v, high_v)
            small  = pygame.surfarray.make_surface(
                np.ascontiguousarray(plasma.swapaxes(0, 1))  # (W,H,3)
            )
            canvas4k.blit(pygame.transform.smoothscale(small, (WIDTH, HEIGHT)), (0, 0))

            # Optionally save first frame as PNG for visual sanity-check.
            if save_first_frame and frame_num == 0:
                pygame.image.save(canvas4k, 'renders/first_frame.png')
                print("[render] Saved first frame → renders/first_frame.png", flush=True)

            # array3d returns (W, H, 3) in SDL2's native byte order (BGR on Linux).
            # Transpose to row-major (H, W, 3), swap BGR→RGB, force C-contiguous
            # so ffmpeg's rgb24 decoder receives the correct channel order.
            frame_arr = pygame.surfarray.array3d(canvas4k)    # (W, H, 3) BGR
            frame_arr = frame_arr.transpose(1, 0, 2)           # (H, W, 3)
            frame_arr = frame_arr[:, :, [2, 1, 0]]             # BGR → RGB
            ffmpeg_proc.stdin.write(np.ascontiguousarray(frame_arr).tobytes())

            # Progress every 60 rendered frames (= 1 s of video).
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
                    f"  ({pct:.1f}% — {m_min}:{m_s:02d} of music"
                    f" — ETA {eta_str})",
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
    finally:
        pygame.quit()

    size_mb  = os.path.getsize(output_path) / (1024 * 1024)
    elapsed  = time.time() - render_start
    e_min, e_s = divmod(int(elapsed), 60)
    print(
        f"[render] Done in {e_min}:{e_s:02d}."
        f" Output: {output_path} ({size_mb:.1f} MB)"
    )


if __name__ == "__main__":
    main()
