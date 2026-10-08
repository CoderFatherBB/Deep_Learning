# Bhavin's portfolio introduction

Generated from Bhavin's supplied avatar and a synthetic male voice using this project's Wav2Lip workflow, with Piper replacing the notebook's Coqui TTS voice generator. The original `input.png`, `input.wav`, and `output.mp4` examples remain intact.

`intro-script.txt` contains the displayed narration; `intro-tts.txt` changes the spelling of Bhavin's name and “AI” to guide pronunciation. The voice is **Piper en_US-ryan-medium**, a synthetic US English male voice, not a clone of Bhavin's voice. `avatar.png` is the supplied full-body character in a black blazer and burgundy shirt.

The output animates the mouth of a still image. It does not generate gestures or natural head movement. The full-body avatar makes the mouth relatively small; an enlarged crop can make speech easier to see.

## Reproduce

Requires FFmpeg, Python 3.12, and the external [Wav2Lip implementation](https://github.com/justinjohn0306/Wav2Lip) at commit `ebe4687d9ecc5a1e255793d149e2738f9f927d09`. The successful local render used a CPU; a GPU is optional.

Install into an isolated environment using `uv`:

```bash
export UV_CACHE_DIR=/workspace/.uv-cache
uv venv --python 3.12 /workspace/lipsync-venv
uv pip install --python /workspace/lipsync-venv/bin/python torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python /workspace/lipsync-venv/bin/python piper-tts==1.8.0 librosa==0.10.2.post1 batch-face==1.5.4 opencv-python==4.11.0.86 numpy==1.26.4 tqdm scipy
git clone https://github.com/justinjohn0306/Wav2Lip.git /workspace/Wav2Lip
git -C /workspace/Wav2Lip checkout ebe4687d9ecc5a1e255793d149e2738f9f927d09
```

Download `wav2lip.pth` and the RetinaFace `mobilenet.pth` checkpoint following the upstream repository's instructions, and place both in `/workspace/Wav2Lip/checkpoints/`. Download the [Piper Ryan medium voice](https://huggingface.co/rhasspy/piper-voices/tree/main/en/en_US/ryan/medium): both `en_US-ryan-medium.onnx` and its `.onnx.json` configuration are required. Models are deliberately excluded from this project folder.

Run:

```bash
bash render.sh
```

Set `WAV2LIP_DIR`, `LIPSYNC_PYTHON`, and `PIPER_MODEL` to use other installation paths. The script recreates `intro.wav`, raw `intro-lipsync.mp4`, and browser-ready `intro.mp4` / `intro.webm` beside the avatar. The web MP4 is 1280×720, H.264/AAC with faststart; WebM uses VP9/Opus. These clips are not a seamless speech loop. The original successful render produced 347 video frames at 25 fps and approximately 14 seconds of narration. No upstream source compatibility patches were needed.

## Checks and usage

The original render was checked with FFprobe and visual contact sheets: the video and audio streams were present, the avatar remained stable, and mouth movement was visible. Human review of name pronunciation and audiovisual synchronization remains useful before publishing.

Wav2Lip's pretrained models are restricted by its upstream license to research, academic, or personal use; review that license before commercial use. The Piper Ryan voice model card lists its dataset license as CC BY-NC-SA 4.0; see `VOICE_MODEL_CARD.txt`. The avatar and narration are personal portfolio assets supplied or requested by Bhavin.
