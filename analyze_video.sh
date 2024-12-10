#!/bin/bash

# Ensure the script is run with a video file as an argument
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <video_file>"
    exit 1
fi

VIDEO_FILE="$1"

# Check if the file exists
if [ ! -f "$VIDEO_FILE" ]; then
    echo "Error: File '$VIDEO_FILE' not found!"
    exit 1
fi

# Use ffmpeg to extract metadata
echo "Analyzing video file: $VIDEO_FILE"
echo "-------------------------------------"
ffmpeg -i "$VIDEO_FILE" 2>&1 | grep -E "(Duration:|Stream|bitrate:|Audio:|Video:)"

echo "-------------------------------------"
echo "Analysis complete!"

