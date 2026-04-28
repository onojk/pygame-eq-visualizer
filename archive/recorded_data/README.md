# archive/recorded_data/

Captured outputs from past visualizer runs. Kept because they're large but might still be
useful as test inputs (especially for the reverse-analysis tools).

| File                | Size  | What it is                                                  |
|---------------------|-------|-------------------------------------------------------------|
| `spectrum_data.txt` | 25 MB | CSV-style spectrum samples from a recording session         |
| `alignment.json`    | 17 MB | Per-frame audio features (frame_idx, time, RMS, chroma)     |
| `spectrum_output.mid`| 3 KB | MIDI rendered from spectrum data via `tools/spectrumtomidi.py` |

`.gitignore` now excludes regenerating these at the project root, so future runs that
create them won't accidentally get committed again.
