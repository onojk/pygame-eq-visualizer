# 🎵 Pygame EQ Visualizer

A real-time audio visualization toolkit built with Python + Pygame.

A collection of audio-reactive visualizers that turn any song or full album into dynamic
bars, waves, particles, kaleidoscopes, lightning, and themed animations — including a
dedicated visualizer for *"The Peace We Build Each Day."*

## Features

- Real-time frequency spectrum visualization (FFT)
- Multiple visual modes: equalizer bars, radial waves, particle bursts, kaleidoscopes, lightning, beat-reactive, image/video warping
- Single-song and full-album playback
- Themed visualizers (Christmas, Peace Series, vintage crooner)
- Video export via FFmpeg
- Supports MP3, WAV, FLAC, and more
- Live mic / line-in input via PyAudio
- A central launcher (`main.py`) that auto-discovers and smoke-tests every visualizer

## Quick start

```bash
# 1. Install system audio dependency (Linux)
sudo apt install portaudio19-dev python3-pyaudio ffmpeg pavucontrol

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Launch the interactive picker
python3 main.py

# Or smoke-test every visualizer (each runs ~8s, then auto-exits)
python3 main.py --test-all

# Or run a specific one
python3 main.py --run visualizers/numbered/12_lightning.py
```

The launcher writes per-script logs to `runlogs/<script>.log` so you can debug what didn't open.

## Project layout

```
pygame-eq-visualizer/
├── main.py                          ← interactive launcher / smoke-tester
├── requirements.txt
├── visualizers/
│   ├── numbered/        ← 15 numbered iterations, each adds a technique (see INDEX.md)
│   ├── themed/          ← peace, christmas, crooner, full-album
│   ├── entropic/        ← pulsefield, worm-swarm, text-HUD
│   ├── warpfield/       ← warpfield engine + 4K offline renderer
│   ├── aurora/          ← aurora-engine + sunset-garden (full + minimal)
│   ├── reverse_analysis/ ← librosa/cv2 audio-video correlation tools
│   └── experimental/    ← 3D vis, kaleidoscope, coalescing grid, etc.
├── tools/               ← clustering, video render, MIDI, install helpers
├── tests/               ← moviepy & OpenGL smoke tests
├── native/              ← C source for entropic_worms (build with make)
├── composer/            ← MIDI composer experiment
├── assets/
│   ├── audio/           ← sample audio_file.mp3
│   └── images/          ← textures used by image-warp visualizers
└── archive/             ← old/duplicate/junk files preserved for reference
```

Each major directory has an `INDEX.md` or `README.md` describing what's inside.

## Audio input on Linux (pavucontrol)

For systems with multiple input devices (built-in mic, USB soundcard, HDMI, etc.):

```bash
sudo apt install pavucontrol
pavucontrol
```

While a visualizer is running, open pavucontrol's **Recording** tab and pick the right
source from the dropdown next to the visualizer process. To visualize a specific audio
file rather than mic input, edit the input source inside the script.

## Customization

Most scripts have a configuration block near the top — bar count, color palette, sensitivity
curves, frame rate, etc. Common knobs:

```python
BAR_COLOR = (255, 0, 0)        # bar color (RGB)
BAR_WIDTH = 10                 # pixel width per bar
MAX_BAR_HEIGHT = 300
FRAME_RATE = 60
LOW_FREQ_BOOST = 1.8           # dial low/mid/high response
MID_FREQ_BOOST = 2.5
HIGH_FREQ_BOOST = 4.0
```

## Audio file setup

Drop your audio in `assets/audio/` or update the path inside the visualizer. Many scripts
honor an `AUDIO_FILE` environment variable as a fallback:

```bash
AUDIO_FILE=~/Music/song.mp3 python3 main.py --run visualizers/numbered/04_particle_wave.py
```

## Native code

The `native/entropic_worms.c` file is a C accelerator for one of the entropic visualizers.
Build it with:

```bash
cd native
make            # produces ./entropic_worms
```

(See `native/Makefile`.)

## Contributing

PRs welcome. Especially:

- new visual modes
- improved FFT responsiveness
- screen / glow effects
- theme presets
- replacing `pyaudio` with `sounddevice` (more portable)

Run `python3 main.py --test-all` before submitting to make sure nothing regressed.

## See also

[Abstrakt](https://github.com/onojk/abstrakt) — feeds these visualizers through an FFmpeg kaleidoscope post-stack to produce symmetric mandala-style music videos from an audio file.

## License

MIT — see [LICENSE](LICENSE).

## Closing note

> *"The Peace We Build Each Day"* is visualized with intention — soft movement, harmonic
> colors, and a sense of unity through sound. Enjoy the music. Enjoy the light.
> — onojk
