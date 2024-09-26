# Pygame EQ Visualizer

## Overview
**Pygame EQ Visualizer** is a real-time audio visualizer built using Python and Pygame. It captures audio input and generates a dynamic visual representation of sound frequencies, similar to an equalizer display. This project demonstrates the versatility of Pygame, offering a fun and interactive way to visualize audio data beyond game development.

## Features
- **Real-Time Audio Processing**: Visualizes frequencies from live audio input or audio files.
- **Customizable Display**: Modify colors, bar sizes, animation speed, and more.
- **Simple and Lightweight**: Ideal for beginners learning audio processing and Pygame.

## Getting Started

### Prerequisites

To run this project, you will need:

- **Python 3.x**
- **Pygame** installed (`pip install pygame`)
- **PyAudio** for real-time audio input (`pip install pyaudio`)
- **PortAudio** system dependency for PyAudio:
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

3. **Install the necessary system dependencies (required for PyAudio)**:
    ```bash
    sudo apt update
    sudo apt install portaudio19-dev python3-pyaudio
    ```

4. **Install the Python dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

5. **You can also use the provided install_requirements.py script** to automatically check and install dependencies:
    ```bash
    python install_requirements.py
    ```

### Running the Visualizer

1. Make sure your microphone or audio input device is connected.
2. Run the visualizer:
    ```bash
    python pygameeq.py
    ```

By default, the visualizer will capture the system's audio input and display the frequency spectrum in real-time. To visualize a specific audio file, you can modify the input source within the script.

## Customization

You can tweak the following settings within the code:

- **Bar Colors**: Customize the color of the equalizer bars for a personalized look.
- **Bar Size**: Modify the height and width of the bars to suit your preference.
- **Animation Speed**: Adjust the speed at which the bars react to audio input.

## Target Audience

- Python developers learning real-time audio processing.
- Developers interested in exploring Pygame beyond game development.
- Anyone looking to create an audio visualizer for personal or educational use.

## Comparison

Compared to other audio visualizers, this project is:

- **Lightweight and Easy to Use**: Simple enough for beginners without requiring advanced audio processing libraries like OpenGL or external DSP tools.
- **Pygame-Centric**: Built entirely using Pygame, offering a great learning experience for those familiar with the library.

## Contributing

Contributions are welcome! If you have any ideas for improving the project, feel free to submit a pull request or open an issue.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contact

If you have any questions or run into issues, feel free to contact the project owner at **onojk123@gmail.com**.

