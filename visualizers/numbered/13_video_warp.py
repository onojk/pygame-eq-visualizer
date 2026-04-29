"""
13_video_warp.py  —  Live procedural plasma, audio-warped.

No video file required. Generates a flowing plasma background each frame
and distorts it in real time using live audio FFT bands.

Audio source: PipeWire monitor captured via pw-record subprocess.
  Discovers the monitor source automatically using pactl; starts pw-record
  as a child process and reads raw float32 PCM from its stdout on a daemon
  thread. No PortAudio / sounddevice in the hot path — zero playback interference.
  Override target with AUDIO_INPUT_DEVICE env var (passed directly to pw-record --target).

Recovery: a background watchdog restarts pw-record automatically if the
  PipeWire monitor source stalls (chunks stop arriving for 2 s). Up to 50
  restarts are attempted before giving up. Set DEBUG_FREEZE=1 for verbose
  heartbeat + diagnostic dump logging to ./runlogs/.
"""

from __future__ import annotations

import fcntl
import math
import os
import shutil
import subprocess
import sys
import threading
import time

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

MAX_RESTARTS = 50  # abort after this many consecutive pw-record restarts

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
# _latest_raw is an immutable bytes object (CHUNK * 4 bytes of float32 PCM).
# Replacing the reference under the lock is O(1); the render loop converts
# to numpy outside the lock so the lock is held for the minimum possible time.
# ---------------------------------------------------------------------------
_audio_lock   = threading.Lock()
_latest_raw   = bytes(CHUNK * 4)  # zeroed silence until first chunk arrives
_chunks_total = 0                  # incremented by reader thread on each chunk
_latest_rms   = 0.0               # updated by main loop each frame

# Set by the watchdog when MAX_RESTARTS is exceeded; main loop checks this.
_quit_event = threading.Event()


# ---------------------------------------------------------------------------
# Debug / watchdog instrumentation
# ---------------------------------------------------------------------------

def _ts() -> str:
    t = time.time()
    return time.strftime("%H:%M:%S", time.localtime(t)) + f".{int(t * 1000) % 1000:03d}"


def _log(logfile, msg: str) -> None:
    """Print with timestamp; also write to logfile when it is not None."""
    line = f"[{_ts()}] {msg}"
    print(line, flush=True)
    if logfile is not None:
        try:
            logfile.write(line + "\n")
            logfile.flush()
        except (ValueError, OSError):
            pass  # logfile closed during shutdown; discard silently


def _dump_diagnostics(logfile, proc: subprocess.Popen) -> None:
    pid = proc.pid
    cmds: list[tuple[list[str], int | None]] = [
        (['pactl', 'list', 'sink-inputs', 'short'], None),
        (['pactl', 'list', 'sources', 'short'],      30),
        (['pactl', 'list', 'sinks', 'short'],         10),
        (['ps', '-o', 'pid,stat,cmd', '-p', str(pid)], None),
    ]
    for cmd, head in cmds:
        try:
            out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, timeout=5)
            if head is not None:
                out = '\n'.join(out.splitlines()[:head])
        except Exception as exc:
            out = f"ERROR: {exc}"
        _log(logfile, f"--- {' '.join(cmd)} ---\n{out}")

    proc_status = f'/proc/{pid}/status'
    try:
        with open(proc_status) as fh:
            out = ''.join(fh.readline() for _ in range(10))
    except Exception as exc:
        out = f"ERROR: {exc}"
    _log(logfile, f"--- {proc_status} (head 10) ---\n{out}")


def _watchdog_loop(
    logfile,
    proc_box: list,
    reader_box: list,
    monitor_name: str,
) -> None:
    """
    Runs forever as a daemon thread. Two jobs:

    1. Stall detector (always active): if chunks_total stops growing for 2 s
       while audio was previously flowing, reset the PipeWire monitor source
       via pactl suspend/resume and spawn a fresh pw-record.
       Gives up and sets _quit_event after MAX_RESTARTS attempts.

    2. Frozen-RMS watchdog (always active, dumps to logfile/stdout): if RMS
       value is unchanged for 3 consecutive seconds after we first had audio,
       dump a diagnostic snapshot. Fires at most once per stall episode.
    """
    last_chunks        = 0
    stall_ticks        = 0   # consecutive 1 s ticks with no new chunks
    had_audio          = False
    restart_count      = 0

    prev_rms_wd        = None
    wd_unchanged_ticks = 0
    had_nonzero_rms    = False
    freeze_dumped      = False

    while True:
        time.sleep(1.0)

        with _audio_lock:
            rms    = _latest_rms
            chunks = _chunks_total

        proc          = proc_box[0]
        reader_thread = reader_box[0]
        reader_alive  = reader_thread.is_alive()
        pw_alive      = proc.poll() is None

        # Heartbeat line — written to file only (avoids stdout noise in normal use).
        if logfile is not None:
            _log(logfile,
                 f"heartbeat: rms={rms:.4f} chunks_total={chunks}"
                 f" reader_alive={reader_alive} pw_alive={pw_alive}"
                 f" restarts={restart_count}")

        # ---- stall detector ------------------------------------------------
        if chunks > last_chunks:
            had_audio   = True
            stall_ticks = 0
        elif had_audio:
            stall_ticks += 1
        last_chunks = chunks

        if stall_ticks >= 2:
            if restart_count >= MAX_RESTARTS:
                _log(logfile,
                     f"STALL: max restarts ({MAX_RESTARTS}) reached, giving up.")
                _quit_event.set()
                return

            restart_count += 1
            _log(logfile,
                 f"STALL detected, recovering... (restart count: {restart_count})")

            old_proc = proc_box[0]
            old_proc.terminate()  # non-blocking; runs in parallel with pactl below

            _log(logfile, "  suspending monitor source")
            suspend = None
            try:
                suspend = subprocess.run(
                    ['pactl', 'suspend-source', monitor_name, '1'],
                    check=False,
                )
            except subprocess.TimeoutExpired:
                _log(logfile, "  suspend timed out; skipping resume")

            # old_proc is dead by now (pactl took ~1.8 s); cap wait defensively
            try:
                old_proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                old_proc.kill()
                old_proc.wait()

            if suspend is None or suspend.returncode != 0:
                _log(logfile, "  suspend failed (skipping resume)")
            else:
                time.sleep(0.1)
                _log(logfile, "  resuming monitor source")
                try:
                    subprocess.run(
                        ['pactl', 'suspend-source', monitor_name, '0'],
                        check=False,
                    )
                except subprocess.TimeoutExpired:
                    _log(logfile, "  resume timed out; continuing anyway")

            _log(logfile, "  spawning fresh pw-record")
            new_proc, new_reader = start_pw_record(monitor_name)
            proc_box[0]          = new_proc
            reader_box[0]        = new_reader
            _log(logfile, f"  new PID: {new_proc.pid}")

            stall_ticks        = 0
            prev_rms_wd        = None
            wd_unchanged_ticks = 0
            freeze_dumped      = False

        # ---- frozen-RMS watchdog -------------------------------------------
        if rms != prev_rms_wd:
            if rms > 0.0:
                had_nonzero_rms = True
            prev_rms_wd        = rms
            wd_unchanged_ticks = 0
            freeze_dumped      = False
        elif had_nonzero_rms:
            wd_unchanged_ticks += 1

        if wd_unchanged_ticks >= 3 and not freeze_dumped:
            freeze_dumped = True
            _log(logfile, "WATCHDOG: RMS frozen for 3s. Dumping diagnostic state...")
            _dump_diagnostics(logfile, proc_box[0])


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


def start_pw_record(monitor_name: str) -> tuple[subprocess.Popen, threading.Thread]:
    """
    Spawn pw-record capturing monitor_name as raw float32 mono at 44100 Hz.
    Starts a daemon reader thread that keeps _latest_raw current.
    Returns (proc, reader_thread). Safe to call multiple times for restarts.
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
            '--raw',
            f'--target={monitor_name}',
            '--format=f32',
            '--rate=48000',
            '--channels=1',
            '--media-category=Capture',
            '--media-role=Production',
            '-',
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )
    print(f"[audio] Capturing monitor source: {monitor_name}")
    print(f"[audio] pw-record PID: {proc.pid}")

    def _reader():
        global _latest_raw, _chunks_total
        frame_bytes = CHUNK * 4
        fd = proc.stdout.fileno()
        # Non-blocking so os.read never stalls; BlockingIOError means pipe is empty.
        fcntl.fcntl(fd, fcntl.F_SETFL,
                    fcntl.fcntl(fd, fcntl.F_GETFL) | os.O_NONBLOCK)
        buf = bytearray()
        while True:
            try:
                data = os.read(fd, 65536)
                if not data:
                    break                          # pw-record exited
                buf.extend(data)
                while len(buf) >= frame_bytes:
                    with _audio_lock:
                        _latest_raw    = bytes(buf[:frame_bytes])
                        _chunks_total += 1
                    del buf[:frame_bytes]
            except BlockingIOError:
                time.sleep(0.001)                 # pipe empty; yield briefly

    def _stderr_logger():
        for line in proc.stderr:
            print(f"[pw-record] {line.decode(errors='replace').rstrip()}", flush=True)

    reader_thread = threading.Thread(target=_reader, daemon=True)
    reader_thread.start()
    threading.Thread(target=_stderr_logger, daemon=True).start()
    return proc, reader_thread


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
    global _latest_rms

    debug        = os.environ.get('DEBUG_AUDIO',  '') == '1'
    debug_freeze = os.environ.get('DEBUG_FREEZE', '') == '1'

    monitor_name        = find_monitor_source()        # exits 1 if no monitor found
    proc, reader_thread = start_pw_record(monitor_name)
    rate                = 48000                        # matches --rate flag passed to pw-record

    # Mutable boxes so the watchdog thread can hot-swap proc/reader on restart.
    proc_box   = [proc]
    reader_box = [reader_thread]

    logfile = None
    if debug_freeze:
        os.makedirs('runlogs', exist_ok=True)
        logname = f"runlogs/13_freeze_debug_{time.strftime('%Y%m%d_%H%M%S')}.log"
        logfile = open(logname, 'w')
        _log(logfile, f"START: monitor={monitor_name} pw_pid={proc.pid}")
        print(f"[freeze] Logging to {logname}", flush=True)

    # Watchdog always runs: stall detection + auto-restart are not debug-only.
    threading.Thread(
        target=_watchdog_loop,
        args=(logfile, proc_box, reader_box, monitor_name),
        daemon=True,
    ).start()

    pygame.init()
    pygame.display.set_caption("Plasma Warp")
    window = pygame.display.set_mode((WIDTH, HEIGHT), pygame.DOUBLEBUF)
    clock  = pygame.time.Clock()
    font   = pygame.font.SysFont(None, 20)

    t            = 0.0
    bass_s       = mid_s = high_s = 0.0
    frame        = 0
    running      = True
    show_overlay = True

    try:
        while running:
            if _quit_event.is_set():
                running = False
                continue

            dt = clock.tick(FPS) / 1000.0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_h:
                        show_overlay = not show_overlay

            # Grab the latest raw bytes from the reader thread (O(1) under lock),
            # then convert to float32 numpy outside the lock.
            with _audio_lock:
                raw = _latest_raw
            mono = np.frombuffer(raw, dtype=np.float32).copy()

            rms = float(np.sqrt(np.mean(mono ** 2)))
            with _audio_lock:
                _latest_rms = rms

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
            if show_overlay:
                draw_overlay(window, rms, bass_s, mid_s, high_s, font)
            pygame.display.flip()

            t += dt
    finally:
        if logfile is not None:
            with _audio_lock:
                final_chunks = _chunks_total
                final_rms    = _latest_rms
            _log(logfile, f"EXIT: chunks_total={final_chunks} final_rms={final_rms:.4f}")
            logfile.close()
        proc_box[0].terminate()
        try:
            proc_box[0].wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc_box[0].kill()
        pygame.quit()


if __name__ == "__main__":
    main()
