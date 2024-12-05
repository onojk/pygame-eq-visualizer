from mido import Message, MidiFile, MidiTrack, MetaMessage
import random
import json

# Constants for the MIDI file
TICKS_PER_BEAT = 480  # Standard resolution for timing
BASE_TEMPO = 500000  # Microseconds per beat (120 BPM)

# Chord progressions (MIDI note numbers)
chord_progression = [
    [60, 64, 67, 71],  # Cmaj7
    [62, 65, 69, 72],  # Dm7
    [57, 60, 64, 67],  # Am7
    [65, 69, 72, 76],  # Fmaj7
    [67, 71, 74, 77],  # G7
    [59, 62, 65, 69],  # Bdim7
]

# Melody notes in the C major scale
scale_notes = [60, 62, 64, 65, 67, 69, 71]  # C, D, E, F, G, A, B

# Harmony intervals (in semitones)
harmony_intervals = [3, 4, 5, 7, 9, 12]  # Third, fourth, fifth, seventh, ninth, octave

# Tempo variations for creativity
tempo_variations = [500000, 400000, 600000, 450000]  # Faster and slower tempos

# Load alignment data (generated from audio analysis)
alignment_file = "alignment.json"
with open(alignment_file, "r") as f:
    alignment_data = json.load(f)

# Create a new MIDI file and four tracks (for chords, melody, harmony, and bass)
midi = MidiFile(ticks_per_beat=TICKS_PER_BEAT)

# Chord track
chord_track = MidiTrack()
chord_track.append(Message('program_change', program=80, time=0))  # Synth pad
midi.tracks.append(chord_track)

# Melody track
melody_track = MidiTrack()
melody_track.append(Message('program_change', program=81, time=0))  # Synth lead
midi.tracks.append(melody_track)

# Harmony track
harmony_track = MidiTrack()
harmony_track.append(Message('program_change', program=82, time=0))  # Electric piano
midi.tracks.append(harmony_track)

# Bassline track
bass_track = MidiTrack()
bass_track.append(Message('program_change', program=38, time=0))  # Synth bass
midi.tracks.append(bass_track)

# Generate the song
for frame_data in alignment_data:
    frame_idx = frame_data["frame_idx"]
    time_in_seconds = frame_data["time"]
    rms = frame_data["rms"]
    chroma = frame_data["chroma"]
    spectrogram = frame_data["spectrogram"]

    # Determine tempo based on RMS energy
    rms_based_tempo = BASE_TEMPO - int(rms * 200000)
    chord_track.append(MetaMessage('set_tempo', tempo=max(rms_based_tempo, 300000), time=0))

    # Select a chord based on the frame index
    chord = chord_progression[frame_idx % len(chord_progression)]

    # Add the chord (whole notes with variation)
    for note in chord:
        chord_track.append(Message('note_on', note=note, velocity=64, time=0))
    chord_track.append(Message('note_off', note=chord[0], velocity=64, time=TICKS_PER_BEAT))  # Whole note duration

    # Add melody based on chroma
    for pitch_class, intensity in enumerate(chroma):
        if intensity > 0.5:  # Use prominent chroma classes for melody
            melody_track.append(Message('note_on', note=scale_notes[pitch_class % len(scale_notes)], velocity=80, time=0))
            melody_track.append(Message('note_off', note=scale_notes[pitch_class % len(scale_notes)], velocity=80, time=TICKS_PER_BEAT // 4))

    # Add harmony based on chroma relationships
    for pitch_class, intensity in enumerate(chroma):
        if intensity > 0.2:  # Use slightly less prominent chroma for harmony
            harmony_note = scale_notes[pitch_class % len(scale_notes)] + random.choice(harmony_intervals)
            if 48 <= harmony_note <= 84:  # Ensure harmony stays within range
                harmony_track.append(Message('note_on', note=harmony_note, velocity=70, time=0))
                harmony_track.append(Message('note_off', note=harmony_note, velocity=70, time=TICKS_PER_BEAT // 8))

    # Add bassline based on spectrogram intensity
    bass_note = chord[0] - 12  # Bassline one octave below root note
    bass_velocity = min(100, max(30, int(spectrogram[0][frame_idx % len(spectrogram[0])] * 100)))
    bass_track.append(Message('note_on', note=bass_note, velocity=bass_velocity, time=0))
    bass_track.append(Message('note_off', note=bass_note, velocity=bass_velocity, time=TICKS_PER_BEAT // 2))

# Save the MIDI file
output_file = "aligned_audio_to_midi_song.mid"
midi.save(output_file)
print(f"Aligned audio-to-MIDI song written to {output_file}. Play it with a MIDI player!")

