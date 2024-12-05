import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import json

def analyze_audio(audio_path):
    """
    Analyzes audio to extract tempo, RMS energy, chroma, and spectrogram.
    """
    print(f"Analyzing audio: {audio_path}")
    
    # Attempt to load audio
    try:
        y, sr = librosa.load(audio_path)
        print("Audio loaded successfully.")
    except Exception as e:
        print(f"Error loading audio: {e}")
        return None, None, None, None, None
    
    # Extract tempo and beat frames
    try:
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        print(f"Tempo detected: {float(tempo):.2f} BPM")
    except Exception as e:
        print(f"Error detecting tempo and beats: {e}")
        return None, None, None, None, None
    
    # Compute RMS energy
    try:
        print("Calculating RMS energy...")
        rms = librosa.feature.rms(y=y)
    except Exception as e:
        print(f"Error calculating RMS energy: {e}")
        return None, None, None, None, None
    
    # Compute chromagram (pitch classes)
    try:
        print("Calculating chromagram...")
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    except Exception as e:
        print(f"Error calculating chromagram: {e}")
        return None, None, None, None, None
    
    # Compute spectrogram
    try:
        print("Calculating spectrogram...")
        spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    except Exception as e:
        print(f"Error calculating spectrogram: {e}")
        return None, None, None, None, None
    
    print("Audio analysis complete.")
    return beats, rms, chroma, spectrogram, sr

def visualize_audio_features(rms, chroma, spectrogram, sr, output_path="audio_visualization.png"):
    """
    Visualizes RMS energy, chromagram, and spectrogram, and saves the plot as a file.
    """
    print("Starting visualization...")
    plt.figure(figsize=(12, 8))
    
    # Plot RMS energy
    plt.subplot(3, 1, 1)
    plt.plot(rms[0])
    plt.title("RMS Energy (Volume)")
    plt.xlabel("Frames")
    plt.ylabel("Energy")
    
    # Plot chromagram
    plt.subplot(3, 1, 2)
    librosa.display.specshow(chroma, x_axis="time", y_axis="chroma", sr=sr)
    plt.colorbar()
    plt.title("Chromagram (Pitch Classes)")
    
    # Plot spectrogram
    plt.subplot(3, 1, 3)
    librosa.display.specshow(librosa.power_to_db(spectrogram, ref=np.max), x_axis="time", y_axis="mel", sr=sr)
    plt.colorbar(format="%+2.0f dB")
    plt.title("Mel Spectrogram (Frequency Content)")
    
    plt.tight_layout()
    
    # Save the plot as a file
    plt.savefig(output_path)
    print(f"Visualization saved as: {output_path}")
    
    plt.show()
    print("Visualization complete.")

def align_audio_with_frames(beats, rms, chroma, spectrogram, sr, video_fps, num_frames):
    """
    Aligns audio features with video frames.
    
    Parameters:
    - beats: Beat timings in frames
    - rms: RMS energy
    - chroma: Chromagram
    - spectrogram: Spectrogram
    - sr: Sampling rate of the audio
    - video_fps: Frames per second of the video
    - num_frames: Total number of frames in the video
    
    Returns:
    - alignment: Dictionary with frame-by-frame alignment of features.
    """
    print("Aligning audio features with video frames...")
    
    # Time per frame in seconds
    frame_duration = 1 / video_fps
    
    # Create alignment for each frame
    alignment = []
    for frame_idx in range(num_frames):
        frame_time = frame_idx * frame_duration  # Time in seconds
        audio_idx = int(frame_time * sr)  # Corresponding audio index
        
        # Align features
        frame_data = {
            "frame_idx": frame_idx,
            "time": frame_time,
            "rms": float(rms[0][audio_idx]) if audio_idx < len(rms[0]) else 0,
            "chroma": chroma[:, audio_idx].tolist() if audio_idx < chroma.shape[1] else [0] * chroma.shape[0],
            "spectrogram": spectrogram[:, audio_idx].tolist() if audio_idx < spectrogram.shape[1] else [0] * spectrogram.shape[0],
        }
        alignment.append(frame_data)
    
    print("Alignment complete.")
    return alignment

if __name__ == "__main__":
    # Path to the extracted MP3 file
    audio_path = "/home/onojk123/pygame-eq-visualizer/audio_file.mp3"

    # Video details
    video_fps = 30  # Frames per second of your video
    num_frames = 7500  # Total number of frames (example, adjust as needed)

    try:
        # Step 1: Analyze the audio
        beats, rms, chroma, spectrogram, sr = analyze_audio(audio_path)
        if beats is None:
            print("Audio analysis failed. Exiting.")
        else:
            # Step 2: Visualize the extracted features
            visualize_audio_features(rms, chroma, spectrogram, sr, output_path="audio_visualization.png")
            
            # Step 3: Align audio with video frames
            alignment = align_audio_with_frames(beats, rms, chroma, spectrogram, sr, video_fps, num_frames)
            
            # Save the alignment data
            with open("alignment.json", "w") as f:
                json.dump(alignment, f, indent=4)
            print("Alignment data saved as 'alignment.json'.")
            
    except Exception as e:
        print(f"An error occurred: {e}")

