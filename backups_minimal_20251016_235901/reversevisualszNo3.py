def visualize_data(synced_data):
    brightness = [analyze_frame(d['frame'])['brightness'] for d in synced_data]
    rms_energy = [d['audio']['rms'] for d in synced_data]

    plt.figure(figsize=(10, 5))
    plt.plot(brightness, label="Frame Brightness")
    plt.plot(rms_energy, label="Audio RMS Energy", alpha=0.7)
    plt.legend()
    plt.xlabel("Frame Index")
    plt.ylabel("Metric Value")
    plt.title("Audio-Visual Correlation")
    plt.show()

# Example usage
visualize_data(synced_data)

