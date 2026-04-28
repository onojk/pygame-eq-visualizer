# Pygame EQ Visualizer: Real-Time Audio Visualization with Pygame

## Overview

**Pygame EQ Visualizer** is a real-time audio visualizer built using Python and Pygame. It captures audio input and generates a dynamic visual representation of sound frequencies, similar to an equalizer display. This project showcases Pygame's versatility in a fun, interactive way, extending its use beyond game development.

## Features

- **Real-Time Audio Processing**: Visualizes frequencies from live audio input or audio files.
- **Customizable Display**: Modify colors, bar sizes, animation speed, and more.
- **Simple and Lightweight**: Ideal for beginners learning audio processing and Pygame.

## Getting Started

### Prerequisites

To run this project, you will need:

- **Python 3.x**
- **Pygame** installed:
  ```bash
  pip3 install pygame
  ```
- **PyAudio** for real-time audio input:
  ```bash
  pip3 install pyaudio
  ```
- **PortAudio** system dependency for PyAudio (Linux):
  ```bash
  sudo apt install portaudio19-dev python3-pyaudio
  ```

### Installation

1. **Clone this repository**:

   ```bash
   git clone https://github.com/onojk/pygame-eq-visualizer.git
   ```

2. **Navigate to the project directory**:

   ```bash
   cd pygame-eq-visualizer
   ```

3. **Install the necessary system dependencies (Linux)**:

   ```bash
   sudo apt update
   sudo apt install portaudio19-dev python3-pyaudio
   ```

   For Windows and macOS, follow the specific PyAudio installation instructions from [PyAudio's official page](https://pypi.org/project/PyAudio/).

4. **Install the Python dependencies**:

   ```bash
   pip3 install -r requirements.txt
   ```

5. **Use the provided install\_requirements.py script** to automatically check and install dependencies:

   ```bash
   python3 install_requirements.py
   ```

### Running the Visualizer

1. Ensure your microphone or audio input device is connected.
2. Run the visualizer:
   ```bash
   python3 pygameeq.py
   ```

### Managing Audio Input with pavucontrol (Linux Only)

In systems with multiple audio input devices (e.g., built-in audio, USB soundcards, or HDMI outputs), managing the correct input source for the visualizer can be tricky. **pavucontrol** (PulseAudio Volume Control) provides an intuitive interface to switch between audio sources and adjust audio stream levels.

#### Installation

To install **pavucontrol** on Linux:

```bash
sudo apt install pavucontrol
```

#### Usage

1. Open pavucontrol by typing:
   ```bash
   pavucontrol
   ```
2. Navigate to the **Recording** tab while the visualizer is running.
3. Select the correct audio input device (e.g., microphone, system audio) from the dropdown next to the visualizer process.
4. Adjust the input volume if needed.

By using pavucontrol, you can easily direct audio from different sources into the visualizer, ensuring the right input is used for your audio visualization. To visualize a specific audio file, modify the input source within the script.

## Customization

You can tweak the following settings within the code:

- **Bar Colors**:

  ```python
  # Example: Changing Bar Colors
  BAR_COLOR = (255, 0, 0)  # Red
  ```

- **Bar Size**:

  ```python
  # Example: Adjusting Bar Size
  BAR_WIDTH = 10  # Width of each bar
  MAX_BAR_HEIGHT = 300  # Maximum height of bars
  ```

- **Animation Speed**:

  ```python
  # Example: Modifying Animation Speed
  FRAME_RATE = 60  # Frames per second
  ```

Customize these settings in the code to achieve a personalized look and behavior for your visualizer.

## Target Audience

- Python developers learning real-time audio processing.
- Developers exploring Pygame beyond game development.
- Audio and visualization enthusiasts interested in creating personalized tools.

## Comparison

Compared to other audio visualizers, this project is:

- **Lightweight and Easy to Use**: Simple enough for beginners without requiring advanced audio processing libraries like OpenGL or external DSP tools.
- **Pygame-Centric**: Built entirely using Pygame, offering a great learning experience for those familiar with the library.

## Contributing

Contributions are welcome! Here’s how you can help:

1. Fork the repository.
2. Create a new branch for your feature or bugfix:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add feature-name"
   ```
4. Push to the branch:
   ```bash
   git push origin feature-name
   ```
5. Submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contact

If you have any questions or run into issues, feel free to contact the project owner:

- Email: **[onojk123@gmail.com](mailto\:onojk123@gmail.com)**
- GitHub: [https://github.com/onojk/pygame-eq-visualizer](https://github.com/onojk/pygame-eq-visualizer)

