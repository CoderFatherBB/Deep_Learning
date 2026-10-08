#!/usr/bin/env bash
set -euo pipefail

# Dependencies and weights are provisioned separately; see README.md.
asset_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
wav2lip_dir="${WAV2LIP_DIR:-/workspace/Wav2Lip}"
python_bin="${LIPSYNC_PYTHON:-/workspace/lipsync-venv/bin/python}"
voice_model="${PIPER_MODEL:-/workspace/lipsync-assets/en_US-ryan-medium.onnx}"

for required_file in "$asset_dir/avatar.png" "$asset_dir/intro-tts.txt" "$voice_model" "$voice_model.json" "$wav2lip_dir/checkpoints/wav2lip.pth" "$wav2lip_dir/checkpoints/mobilenet.pth"; do
  if [[ ! -f "$required_file" ]]; then
    printf 'Missing required file: %s\n' "$required_file" >&2
    exit 1
  fi
done

"$python_bin" -m piper -m "$voice_model" -i "$asset_dir/intro-tts.txt" -f "$asset_dir/intro.wav" --length-scale 1.05 --sentence-silence 0.25
cd -- "$wav2lip_dir"
mkdir -p temp results
NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-$asset_dir/.numba-cache}" MPLCONFIGDIR="${MPLCONFIGDIR:-$asset_dir/.mpl-cache}" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "$python_bin" inference.py \
  --checkpoint_path checkpoints/wav2lip.pth \
  --face "$asset_dir/avatar.png" \
  --audio "$asset_dir/intro.wav" \
  --outfile "$asset_dir/intro-lipsync.mp4" \
  --wav2lip_batch_size 8 --fps 25

ffmpeg -hide_banner -loglevel error -y -i "$asset_dir/intro-lipsync.mp4" \
  -vf 'scale=1280:720,tpad=stop_mode=clone:stop_duration=0.12' \
  -c:v libx264 -crf 20 -preset medium -pix_fmt yuv420p -c:a aac -b:a 96k -ar 48000 \
  -movflags +faststart -shortest "$asset_dir/intro.mp4"
ffmpeg -hide_banner -loglevel error -y -i "$asset_dir/intro.mp4" \
  -c:v libvpx-vp9 -crf 32 -b:v 0 -row-mt 1 -c:a libopus -b:a 80k "$asset_dir/intro.webm"
