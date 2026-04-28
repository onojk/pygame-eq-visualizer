# tools/

Helper scripts used to analyze, render, install, or reorganize content — not meant to be
launched by `main.py` as visualizers.

| File                       | What it does                                                            |
|----------------------------|-------------------------------------------------------------------------|
| `install_requirements.py`  | Checks which Python deps are missing and installs them                  |
| `cluster_by_code.py`       | Clusters .py files by code similarity                                   |
| `cluster_visualizers.py`   | Clusters visualizer scripts (output: code_clusters.{csv,json})          |
| `import_pyaudio_tool.py`   | Standalone PyAudio import diagnostic                                    |
| `spectrumtomidi.py`        | Converts FFT spectrum recordings to MIDI                                |
| `visualizer_to_video.py`   | Renders a visualizer's output to MP4 via FFmpeg                         |
| `analyze_video.sh`         | Bash helper for video frame analysis                                    |
| `render_all_wavs.sh`       | Bash helper to render every WAV in a folder                             |
| `code_clusters.csv/.json`  | Output of cluster_visualizers.py — the discovered groupings             |
| `marks_review.csv/.json`   | Output of marking/review pass over scripts                              |
| `clustering_all_py.txt`    | Was `.all_py` at repo root — input list for clustering                  |
| `clustering_marked.txt`    | Was `.marked` at repo root — list of files marked-up by the cluster pass|
| `campfire_ember_loop.fc`   | Fragment-shader loop file (used by some experimental visualizer)        |
