#!/usr/bin/env python3
import os, time, math, random, wave, contextlib
import numpy as np
import pygame
try:
    import pyaudio
except Exception:
    pyaudio = None

# activate fireworks override
import fireworks_override  # noqa: F401
from aurora_engine import hsv255, FireworksSystem

# ===== Config =====
CHUNK, RATE, FPS = 512, 22050, 60
BG_PATH = "wmfkwonokewnfokewnokofno_bg.png"
FG_PATH = "wmfkwonokewnfokewnokofno_fg.png"
AUDIO_PATH = "soundtrack.wav"

offline = os.getenv("AURORA_OFFLINE") == "1"
if offline:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
else:
    os.environ.setdefault("SDL_AUDIODRIVER", "pulse")

# ===== Init pygame/display =====
pygame.init()
if offline:
    W, H = 1920, 1080
    pygame.display.set_mode((W, H))  # create a display surface so convert_alpha works
else:
    screen_fs = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    info = pygame.display.Info()
    W, H = info.current_w, info.current_h
    # recreate with explicit size for consistent scaling operations
    pygame.display.set_mode((W, H))

screen = pygame.display.get_surface()
clock = pygame.time.Clock()

# Load images (require a display surface for convert_alpha)
bg_img = pygame.image.load(BG_PATH).convert_alpha()
fg_img = pygame.image.load(FG_PATH).convert_alpha()
bg_img = pygame.transform.smoothscale(bg_img, (W, H))
fg_img = pygame.transform.smoothscale(fg_img, (W, H))

fw = FireworksSystem(W, H)
fw.set_foreground(fg_img)

# ===== Audio helpers =====
def analyze_features(xi):
    x = xi.astype(np.float32) / 32768.0
    if x.size == 0:
        return {"energy":0.0,"bass":0.0,"treble":0.0,"beat_gate":0.0}
    sp = np.abs(np.fft.rfft(x * np.hanning(x.size)))
    m = sp.max()
    if m <= 1e-9:
        return {"energy":0.0,"bass":0.0,"treble":0.0,"beat_gate":0.0}
    sp /= m
    n = sp.size
    bass = float(np.mean(sp[:max(2, n//24)]))
    treble = float(np.mean(sp[max(2, n//6):]))
    energy = float(np.mean(sp))
    return {"energy":energy,"bass":bass,"treble":treble,"beat_gate":1.0 if (bass>0.25 and energy>0.2) else 0.0}

def wav_duration(path):
    try:
        with contextlib.closing(wave.open(path, 'rb')) as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            return frames / float(rate)
    except Exception:
        return 0.0

# ===== Offline vs Live =====
if offline:
    # prepare output dir
    os.makedirs("frames_out", exist_ok=True)
    # compute exact frame count from soundtrack
    dur = wav_duration(AUDIO_PATH)
    if dur <= 0.0:
        print("[error] soundtrack.wav not readable or duration unknown.")
        pygame.quit(); raise SystemExit(1)
    total_frames = int(math.ceil(dur * FPS))
    print(f"[info] Offline: duration={dur:.2f}s, frames={total_frames} @ {FPS} fps, size={W}x{H}")

    # simple “time-sliced” audio proxy driving features (no live audio needed)
    # we synthesize Xi as silence (features still vary from visuals),
    # or you could map an LFO if you want motion; here, keep it deterministic:
    xi_silence = np.zeros(CHUNK, dtype=np.int16)

    frame_idx = 0
    while frame_idx < total_frames:
        dt = 1.0 / FPS
        feat = analyze_features(xi_silence)
        fw.notify_audio(feat, dt)

        power = 0.9 + 0.6  # stable mid-power for offline
        if random.random() < (0.35) * dt:
            fw.launch(speed_scale=power)
        fw.step(dt, 1.0)

        # draw
        screen.blit(bg_img, (0,0))
        fw.draw(screen)
        screen.blit(fg_img, (0,0))
        fw.draw_fg_flash(screen)

        # save frame
        pygame.image.save(screen, f"frames_out/frame_{frame_idx:06d}.png")
        if (frame_idx % 300) == 0:
            print(f"[info] wrote frame {frame_idx}/{total_frames}")
        frame_idx += 1

    pygame.quit()
else:
    # Live mode (unchanged)
    stream = None
    if pyaudio is not None:
        try:
            p = pyaudio.PyAudio()
            def find_input_index(substr):
                s = substr.lower()
                for i in range(p.get_device_count()):
                    d = p.get_device_info_by_index(i)
                    if d.get("maxInputChannels",0) >= 1 and s in d.get("name","").lower():
                        return i
                return p.get_default_input_device_info()["index"]
            idx = find_input_index(os.getenv("PREFERRED_DEVICE_SUBSTR","pulse"))
            stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True,
                            input_device_index=idx, frames_per_buffer=CHUNK)
        except Exception:
            stream = None

    running = True
    while running:
        dt = clock.tick(FPS)/1000.0
        for e in pygame.event.get():
            if e.type == pygame.QUIT: running = False
            elif e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE: running = False

        # audio
        if stream is not None:
            try:
                xi = np.frombuffer(stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16)
            except Exception:
                xi = np.zeros(CHUNK, dtype=np.int16)
        else:
            xi = np.zeros(CHUNK, dtype=np.int16)

        feat = analyze_features(xi)
        fw.notify_audio(feat, dt)

        power = 0.9 + 1.1*feat["energy"]
        if random.random() < (0.25 + 0.60*feat["bass"]) * dt:
            fw.launch(speed_scale=power)
        fw.step(dt, 1.0)

        screen.blit(bg_img,(0,0))
        fw.draw(screen)
        screen.blit(fg_img,(0,0))
        fw.draw_fg_flash(screen)
        pygame.display.flip()

    if stream is not None:
        try: stream.stop_stream(); stream.close(); p.terminate()
        except Exception: pass
    pygame.quit()
