k🎵 Pygame Equalizer Visualizer

A real-time audio visualization toolkit built with Python + Pygame

A collection of audio-reactive visualizers that turn any song or full album into dynamic bars, waves, particles, and themed animations — including a dedicated visualizer for SONG 11 — “The Peace We Build Each Day.”

✨ Features

Real-time frequency spectrum visualization (FFT-powered)

Multiple visual modes: equalizer bars, radial waves, particle bursts, ambient pulses

Single-song and full-album playback

Themed visualizers, including Christmas and Peace Series

Video export using FFmpeg

Supports MP3, WAV, FLAC, and more

Lightweight, customizable, and beginner-friendly

📁 Visualizers Included
File	Description
pygame_peace_visualizer.py	Dedicated visualizer for SONG 11 — “The Peace We Build Each Day”
pygame_visualizer_full_album.py	Visualizes an entire album with smooth transitions
pygame_visualizer_full_album_clean.py	Minimalist, performance-optimized album visualizer
pygame_christmas_visualizer.py	Festive holiday-themed audio visualizer
pygame_crooner_xmas.py	Vintage crooner-style Christmas visualizer
pygame_visualizer_to_video.py	Captures any visualizer output to MP4 (requires FFmpeg)
📦 Requirements

Python 3.8+

pygame

pyaudio

numpy

FFmpeg (optional, for video export)

🛠 Installation
# Clone the repository
git clone https://github.com/onojk/pygame-eq-visualizer.git
cd pygame-eq-visualizer

# Install dependencies
pip install pygame pyaudio numpy

Install FFmpeg (Optional but Recommended)

Ubuntu / Debian

sudo apt install ffmpeg


macOS

brew install ffmpeg


Windows
Download from: https://ffmpeg.org/download.html

▶️ Usage
Run the Peace Visualizer (SONG 11)
python pygame_peace_visualizer.py

Run the Full Album Visualizer
python pygame_visualizer_full_album.py

Export a Visualization to Video
python pygame_visualizer_to_video.py

🎵 Audio File Setup

Place your audio files in the project folder, or update the file paths inside the visualizer scripts.

🎨 Customization

You can modify any visualizer to change:

Color palettes

Bar count, thickness, and smoothing

Particle density & motion

Sensitivity curves

Background animations

Radial geometry for circular visualizers

All scripts are written for easy editing.

🤝 Contributing

Contributions are welcome!
Ideas include:

New visual modes

Improved FFT responsiveness

Screen effects / glow layers

Theme-based visualizer presets

Submit a pull request anytime.

📄 License

MIT License — free to use, modify, and share.

🕊️ Closing Note

“The Peace We Build Each Day” is visualized with intention — soft movement, harmonic colors, and a sense of unity through sound.

Enjoy the music. Enjoy the light.
— onojk
