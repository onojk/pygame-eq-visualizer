# visualizers/reverse_analysis/

"Reverse" tooling — instead of audio → visuals, these analyze finished media and try to
correlate audio features with visual ones.

| File                                | What it does                                                            |
|-------------------------------------|-------------------------------------------------------------------------|
| `audio_analyzer_librosa.py`         | Extracts tempo / RMS / chroma / spectrogram from an audio file (librosa)|
| `frame_analyzer_cv2.py`             | Per-frame brightness + dominant-color analysis using OpenCV             |
| `audio_video_correlator.py`         | Plots the correlation between audio RMS energy and frame brightness     |
| `reverse_impulse.py`                | Impulse-response / reverse-pulse experiment                             |

## Important

These three were rescued from `backups_minimal_20251016_235901/` — the top-level
`reversevisualszNo*.py` files were misnamed copies of the basic music visualizer and
contained none of this code. The originals are now in
`archive/duplicate_misnamed_top_level/` for reference only.
