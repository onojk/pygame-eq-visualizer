import mido
import numpy as np

# Configuration
input_file = "/home/onojk123/pygame-eq-visualizer/spectrum_data.txt"
output_midi_file = "/home/onojk123/pygame-eq-visualizer/spectrum_output.mid"

# Parameters for optimization
notes_per_slice = 8  # Reduce number of slices
frame_skip = 5       # Skip every 5 frames
max_frames = 1000    # Limit the number of frames
midi_tempo = 500000  # Default 120 BPM

def spectrum_to_midi(input_file, output_file):
    """Convert spectrum data into a compact MIDI file."""
    # Read the spectrum data from file
    with open(input_file, "r") as f:
        lines = [line.strip() for line in f if line and line[0].isdigit()]

    # Parse spectrum data
    data = []
    for line in lines:
        try:
            spectrum = list(map(float, line.split(",")))
            data.append(spectrum)
        except ValueError:
            continue

    # Limit to max frames
    data = data[:max_frames]

    # Normalize spectrum to MIDI pitch range
    pitches = np.linspace(36, 84, notes_per_slice)  # MIDI notes C2 to C6
    velocities = lambda mag: int(np.clip(mag * 127, 0, 127))  # Scale to MIDI velocity

    # Create MIDI file
    midi_file = mido.MidiFile()
    track = mido.MidiTrack()
    midi_file.tracks.append(track)

    # Add tempo
    track.append(mido.MetaMessage('set_tempo', tempo=midi_tempo))

    # Generate MIDI events
    for frame_index, spectrum in enumerate(data):
        if frame_index % frame_skip != 0:
            continue  # Skip frames for density reduction

        # Merge slices to reduce note density
        slice_step = len(spectrum) // notes_per_slice
        for i in range(notes_per_slice):
            avg_magnitude = np.mean(spectrum[i * slice_step: (i + 1) * slice_step])
            pitch = int(pitches[i])
            velocity = velocities(avg_magnitude)
            if velocity > 0:  # Only add notes with non-zero velocity
                track.append(mido.Message('note_on', note=pitch, velocity=velocity, time=0))
        # Add a time delay for the next spectrum
        track.append(mido.Message('note_off', note=pitch, velocity=0, time=960))  # Longer delay for compactness

    # Save MIDI file
    midi_file.save(output_file)
    print(f"MIDI file saved to {output_file}")

# Run conversion
spectrum_to_midi(input_file, output_midi_file)

