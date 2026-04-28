#!/usr/bin/env bash
set -Eeuo pipefail

# Inputs (edit these paths if needed)
PATTERN="./videos/Kaleidoscope VIdeo 3843.mp4"
MASK_GLOB="4kfirealone*.png"
BG="emberlogs.png"
MUSIC_DIR="./music"
OUTDIR="./renders"

mkdir -p "$OUTDIR"

shopt -s nullglob
wavs=( "$MUSIC_DIR"/*.wav )

if (( ${#wavs[@]} == 0 )); then
  echo "No WAVs found in $MUSIC_DIR" >&2
  exit 1
fi

for wav in "${wavs[@]}"; do
  base="$(basename "$wav")"
  name="${base%.*}"
  out="$OUTDIR/${name}_pattern_fire_4k.mp4"

  echo ">>> Rendering: $base -> $(basename "$out")"

  ffmpeg -y \
    -stream_loop -1 -i "$PATTERN" \
    -stream_loop -1 -framerate 12 -pattern_type glob -i "$MASK_GLOB" \
    -i "$BG" \
    -i "$wav" \
    -filter_complex_script campfire_ember_loop.fc \
    -map "[VOUT]" -map 3:a \
    -c:v libx264 -preset veryfast -crf 18 -pix_fmt yuv420p \
    -c:a aac -b:a 320k -shortest -movflags +faststart \
    "$out"
done
