import cv2
import numpy as np

def analyze_frame(frame_path):
    frame = cv2.imread(frame_path)
    if frame is None:
        return None
    # Convert to grayscale and calculate brightness
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    # Extract dominant colors
    colors = cv2.mean(frame)[:3]  # RGB values
    return {"brightness": brightness, "colors": colors}

# Example usage
frame_data = analyze_frame("frames/frame_0001.jpg")
print(frame_data)

