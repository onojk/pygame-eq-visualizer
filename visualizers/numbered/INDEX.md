# visualizers/numbered/

The numbered series — successive iterations of a music visualizer, each adding a new
visual mode or technique. Renamed from `pygamemusicvisualizerNoX.py` to descriptive names.

| File                              | Original name                  | What it does                                                            |
|-----------------------------------|--------------------------------|-------------------------------------------------------------------------|
| 00_safe_opener.py                 | pygamemusicvisualizer.py (top) | Lightweight "safe opener" — animated pattern + audio playback, no FFT   |
| 01_full_bars_with_audio.py        | (basic, from backups dir)      | Full FFT bars + colorsys + sensitivity boosts                           |
| 02_kaleidoscope_spokes.py         | pygamemusicvisualizerNo2.py    | Notched bars in a kaleidoscope, rotating spokes (120), beat circles     |
| 03_bars_color_modes.py            | pygamemusicvisualizerNo1.py    | Bars with 8 selectable color modes (rainbow, mono, gradient, fire, etc) |
| 04_particle_wave.py               | pygamemusicvisualizerNo4.py    | Particle + Flare classes, particle wave driven by low frequencies       |
| 06_rotating_spokes_circles.py     | pygamemusicvisualizerNo6.py    | Rotating spokes (40), expanding circles, particles, layered rendering   |
| 06b_rotating_spokes_dense.py      | pygamemusicvisualizerNo6b.py   | Same family as 06 but 240 spokes — denser/more chaotic look             |
| 07_kaleidoscope_lines.py          | pygamemusicvisualizerNo7.py    | Simple lines, broken-black-wave overlay, kaleidoscope                   |
| 08_speaker_grid_ghost.py          | pygamemusicvisualizerNo8.py    | Grid of speaker icons reacting per band, with ghost trails              |
| 09_beat_reactive.py               | pygamemusicvisualizerNo9.py    | Beat detection, large-frequency-shift detection, opacity-controlled     |
| 10_image_warp.py                  | pygamemusicvisualizerNo10.py   | Loads an image, warps/transforms it on each audio frame                 |
| 11_freq_spectrum.py               | pygamemusicvisualizerNo11.py   | Plain frequency spectrum (20 Hz – 20 kHz human range)                   |
| 12_lightning.py                   | pygamemusicvisualizerNo12.py   | Recursive branching lightning paths driven by audio                     |
| 13_video_warp.py                  | pygamemusicvisualizerNo13.py   | Like 10 but warps a video frame instead of a still image                |
| 14_white_line_kaleidoscope.py     | pygamemusicvisualizerNo14.py   | Extracts white lines from an image, smoothes them, kaleidoscopes them   |

## What's NOT here

The file numbers 3, 5 — and 9–14 at the **top level** of the original repo — were all
byte-identical copies of the basic visualizer (someone seems to have done a stray bulk-cp).
Those duplicates are preserved in `archive/duplicate_misnamed_top_level/` for reference.
The unique versions came from `backups_minimal_20251016_235901/` and replace them here.
