#!/usr/bin/env python3
"""
render_dialup_dawn_frames.py

Option B: OFFLINE FRAME RENDERER

- Reads LONG_MASTER.wav
- Does a first pass to compute FFT energy bands (bass/mid/treble/overall)
- Normalizes them
- Does a second pass, generating one visual frame per analysis step
- Saves frames as JPEGs in frames_dialup_dawn/

Later, stitch into a video with FFmpeg:
  ffmpeg -framerate 20 -i frames_dialup_dawn/%06d.jpg -i LONG_MASTER.wav \
         -c:v libx264 -preset slow -crf 18 \
         -c:a aac -b:a 320k -shortest dialup_dawn_visualizer.mp4
"""

import math
import sys
import wave
import time
from pathlib import Path

import numpy as np
import pygame
from PIL import Image
import colorsys

# ------------- CONFIG --------------------------------------------------------

AUDIO_PATH = Path("LONG_MASTER.wav")   # must exist in same folder as this script

WIDTH, HEIGHT = 1280, 720              # 720p to keep file sizes sane
FPS = 20                               # render FPS (tweakable)
FRAMES_DIR = Path("frames_dialup_dawn")

NUM_RAYS = 64
NUM_PARTICLES = 80


# ------------- UTILITIES -----------------------------------------------------


def hsv_to_rgb_int(h, s, v):
    """h,s,v in [0,1] -> (r,g,b) in [0,255] ints."""
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return int(r * 255), int(g * 255), int(b * 255)


def analyze_audio(path: Path, fps: int):
    """
    First pass:
      - Read audio in fixed chunks corresponding to 1/FPS seconds
      - Compute FFT magnitudes
      - Extract bass/mid/treble/energy per chunk
      - Track maxima and normalize
    Returns:
      frames_info: list of dicts with time, bass_n, mid_n, treble_n, energy_n
      duration: float seconds
    """
    if not path.exists():
        print(f"[ERROR] Audio file not found: {path}")
        sys.exit(1)

    wf = wave.open(str(path), "rb")
    n_channels = wf.getnchannels()
    sampwidth = wf.getsampwidth()
    sample_rate = wf.getframerate()
    total_frames = wf.getnframes()
    duration = total_frames / float(sample_rate)

    print(f"[AUDIO] {path}")
    print(f"        channels={n_channels}, sampwidth={sampwidth} bytes, rate={sample_rate} Hz")
    print(f"        duration={duration/60:.2f} min")

    # we read this many PCM samples per visual frame
    samples_per_frame = int(sample_rate / fps)

    # sample width handling
    dtype_map = {
        1: np.int8,
        2: np.int16,
        4: np.int32,
    }
    if sampwidth not in dtype_map:
        print(f"[ERROR] Unsupported sample width: {sampwidth} bytes")
        wf.close()
        sys.exit(1)
    dtype = dtype_map[sampwidth]

    frames_info = []
    max_bass = max_mid = max_treble = max_energy = 1e-6

    frame_idx = 0
    start_time = time.time()

    while True:
        raw = wf.readframes(samples_per_frame)
        if not raw:
            break

        # convert bytes -> numpy array
        arr = np.frombuffer(raw, dtype=dtype).astype(np.float32)
        if arr.size == 0:
            continue

        # stereo -> mono
        if n_channels > 1:
            arr = arr.reshape(-1, n_channels).mean(axis=1)

        # apply window to reduce spectral leakage
        window = np.hanning(arr.size)
        windowed = arr * window

        spectrum = np.fft.rfft(windowed)
        mags = np.abs(spectrum)
        freqs = np.fft.rfftfreq(windowed.size, d=1.0 / sample_rate)

        # define bands (in Hz)
        bass_mask = freqs < 200
        mid_mask = (freqs >= 200) & (freqs < 2000)
        treble_mask = freqs >= 2000

        if np.any(bass_mask):
            bass = mags[bass_mask].mean()
        else:
            bass = 0.0
        if np.any(mid_mask):
            mid = mags[mid_mask].mean()
        else:
            mid = 0.0
        if np.any(treble_mask):
            treble = mags[treble_mask].mean()
        else:
            treble = 0.0

        energy = mags.mean()

        frames_info.append(
            {
                "time": frame_idx / float(fps),
                "bass": float(bass),
                "mid": float(mid),
                "treble": float(treble),
                "energy": float(energy),
            }
        )

        max_bass = max(max_bass, bass)
        max_mid = max(max_mid, mid)
        max_treble = max(max_treble, treble)
        max_energy = max(max_energy, energy)

        frame_idx += 1

        if frame_idx % 1000 == 0:
            elapsed = time.time() - start_time
            print(f"[AUDIO] analyzed {frame_idx} frames (~{frame_idx/fps/60:.1f} min), {elapsed/60:.1f} min elapsed")

    wf.close()

    # normalize
    for f in frames_info:
        f["bass_n"] = f["bass"] / max_bass if max_bass > 0 else 0.0
        f["mid_n"] = f["mid"] / max_mid if max_mid > 0 else 0.0
        f["treble_n"] = f["treble"] / max_treble if max_treble > 0 else 0.0
        f["energy_n"] = f["energy"] / max_energy if max_energy > 0 else 0.0

    print(f"[AUDIO] done. total visual frames: {len(frames_info)}")
    print(f"[AUDIO] maxima: bass={max_bass:.2f}, mid={max_mid:.2f}, treble={max_treble:.2f}, energy={max_energy:.2f}")
    return frames_info, duration


class Particle:
    def __init__(self, width, height, idx, total):
        # orbit radius interpolated
        self.cx = width / 2
        self.cy = height / 2
        self.orbit = np.interp(idx, [0, total - 1], [60, min(width, height) * 0.45])
        self.base_angle = np.random.uniform(0, math.tau)
        self.speed = np.random.uniform(0.05, 0.4)
        self.size = np.random.uniform(2.0, 5.0)
        self.twinkle_phase = np.random.uniform(0, 1000)

    def update(self, t, bass, mid, treble):
        # slight orbit wobble from mid + treble
        angle = self.base_angle + t * self.speed + bass * 1.5
        wobble = treble * 0.2 * math.sin(t * 2.0 + self.base_angle * 3.0)
        r = self.orbit * (1.0 + 0.25 * mid + wobble)

        x = self.cx + math.cos(angle) * r
        y = self.cy + math.sin(angle) * r

        twinkle = 0.5 + 0.5 * math.sin(t * 4.0 + self.twinkle_phase)
        twinkle *= (0.3 + treble * 0.7)
        return x, y, self.size, twinkle


def render_frames(frames_info, fps, width, height, out_dir: Path):
    """
    Second pass:
      - For each frame_info, render one visual frame
      - Save as JPEG: frames_dialup_dawn/000000.jpg, 000001.jpg, ...
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    pygame.init()
    surface = pygame.Surface((width, height))

    # Pre-generate particles
    particles = [Particle(width, height, i, NUM_PARTICLES) for i in range(NUM_PARTICLES)]

    total = len(frames_info)
    start = time.time()
    last_print = start

    for idx, f in enumerate(frames_info):
        t = f["time"]
        bass = f["bass_n"]
        mid = f["mid_n"]
        treble = f["treble_n"]
        energy = f["energy_n"]

        # --- Background: evolving hue field ---
        hue = (0.04 * t + bass * 0.12) % 1.0
        sat = 0.6 + 0.3 * mid
        val = 0.18 + 0.6 * energy
        bg = hsv_to_rgb_int(hue, min(sat, 1.0), min(val, 1.0))
        surface.fill(bg)

        cx, cy = width / 2, height / 2

        # --- Radial rays ---
        base_len = min(width, height) * (0.18 + 0.24 * energy + 0.18 * bass)
        for i in range(NUM_RAYS):
            ang = (i / NUM_RAYS) * math.tau
            ang += 0.25 * math.sin(t * 0.7 + i * 0.3) * treble
            length = base_len * (0.8 + 0.2 * math.sin(t * 1.3 + i))

            ray_hue = (hue + 0.12 * math.sin(t * 0.5 + i * 0.15)) % 1.0
            ray_sat = 0.7 + 0.3 * treble
            ray_val = 0.65 + 0.35 * energy
            ray_color = hsv_to_rgb_int(ray_hue, min(ray_sat, 1.0), min(ray_val, 1.0))

            x2 = cx + math.cos(ang) * length
            y2 = cy + math.sin(ang) * length
            width_px = max(1, int(1 + 4 * bass + 2 * treble))

            pygame.draw.line(surface, ray_color, (cx, cy), (x2, y2), width_px)

        # --- Orbiting particles ---
        for p in particles:
            x, y, size, twinkle = p.update(t, bass, mid, treble)
            ph = (hue + 0.2 + treble * 0.15) % 1.0
            ps = 0.5 + 0.5 * mid
            pv = 0.5 + 0.5 * twinkle
            color = hsv_to_rgb_int(ph, min(ps, 1.0), min(pv, 1.0))
            pygame.draw.circle(surface, color, (int(x), int(y)), int(size + 2 * energy))

        # --- Central "sun" ---
        sun_radius = 50 + 120 * bass + 80 * energy
        sun_hue = (hue + 0.03 * math.sin(t * 0.8)) % 1.0
        sun_sat = 0.5 + 0.5 * energy
        sun_val = 0.85 + 0.15 * bass
        sun_color = hsv_to_rgb_int(sun_hue, min(sun_sat, 1.0), min(sun_val, 1.0))
        pygame.draw.circle(surface, sun_color, (int(cx), int(cy)), int(sun_radius))

        # thin white halo
        halo_radius = int(sun_radius * (1.12 + 0.06 * treble))
        pygame.draw.circle(surface, (255, 255, 255), (int(cx), int(cy)), halo_radius, max(1, int(2 + 5 * treble)))

        # --- Save frame as JPEG ---
        frame_path = out_dir / f"{idx:06d}.jpg"
        # Convert Surface -> PIL Image
        raw_str = pygame.image.tostring(surface, "RGB")
        img = Image.frombytes("RGB", (width, height), raw_str)
        img.save(frame_path, "JPEG", quality=90, optimize=True)

        # progress logging
        now = time.time()
        if now - last_print > 10:  # every ~10s
            done_pct = 100.0 * (idx + 1) / total
            elapsed = now - start
            est_total = elapsed / (done_pct / 100.0) if done_pct > 0 else 0
            eta = est_total - elapsed
            print(
                f"[RENDER] frame {idx+1}/{total} "
                f"({done_pct:5.1f}%), elapsed {elapsed/60:.1f} min, ETA {eta/60:.1f} min"
            )
            last_print = now

    pygame.quit()
    print("[RENDER] done.")


def main():
    print("[STEP 1] Analyzing audio...")
    frames_info, duration = analyze_audio(AUDIO_PATH, FPS)
    print(f"[INFO] Audio duration: {duration/60:.2f} min, frames at {FPS} fps: {len(frames_info)}")

    print("[STEP 2] Rendering frames...")
    render_frames(frames_info, FPS, WIDTH, HEIGHT, FRAMES_DIR)

    print("\n[COMPLETE]")
    print(f"Frames saved to: {FRAMES_DIR}/%06d.jpg")
    print("\nNext, run this FFmpeg command (from the same folder):\n")
    print(
        f"  ffmpeg -framerate {FPS} -i {FRAMES_DIR}/%06d.jpg -i LONG_MASTER.wav "
        "-c:v libx264 -preset slow -crf 18 "
        "-c:a aac -b:a 320k -shortest dialup_dawn_visualizer.mp4"
    )


if __name__ == '__main__':
    main()
