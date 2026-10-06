# IMDER - Bot & AI Agent Usage Guide

This document provides comprehensive instructions for AI agents, bots, and automated systems on how to effectively use IMDER for image and video processing tasks. AI agents typically operate in headless environments without continuous terminal access, so this guide focuses on one-liner commands and batch processing patterns.

## Table of Contents

1. [Purpose](#purpose)
2. [Quick Start for AI Agents](#quick-start-for-ai-agents)
3. [Installation](#installation)
4. [FFmpeg Setup](#ffmpeg-setup)
5. [One-Liner Command Patterns](#one-liner-command-patterns)
6. [Command Reference](#command-reference)
7. [CLI vs GUI Feature Comparison](#cli-vs-gui-feature-comparison)
8. [Video Processing Rationale](#video-processing-rationale)
9. [Limitations](#limitations)
10. [Troubleshooting](#troubleshooting)
11. [Example Workflows](#example-workflows)

---

## Purpose

IMDER is an image blender tool that creates smooth animations by blending pixels between two images or videos. For AI agents operating in automated pipelines, IMDER offers:

- **Fast processing**: Seconds vs. minutes/hours for similar tools
- **CLI-first design**: All core features accessible via command line
- **No GUI required**: Runs entirely in headless terminals
- **Video support**: Frame-by-frame processing for video transformations
- **Audio generation**: Synthesize or extract audio for video output

---

## Quick Start for AI Agents

AI agents typically cannot maintain interactive terminal sessions. Use the following pattern:

```bash
# Option A: the native binary from the releases (no python deps, single small exe)
# Download IMDER_windows_v1.3.0.zip or imder_linux_v1.3.0_cpp.zip, unzip, then:
./imder /path/to/base.png /path/to/target.png shuffle 512

# Option B: from the python source
git clone https://github.com/HAKORADev/IMDER.git && cd IMDER/src

# Install dependencies (one-liner)
pip install opencv-python numpy PyQt5 pillow pyfiglet

# Process files immediately (one-liner, positional syntax)
python imder.py /path/to/base.png /path/to/target.png shuffle 512

# Chain multiple operations
./imder base1.png target1.png merge && ./imder base2.png target2.png merge
```

---

## Installation

### Python Dependencies

Install all required packages in a single command:

```bash
pip install opencv-python numpy PyQt5 pillow pyfiglet
```

**Package explanations:**

| Package | Purpose |
|---------|---------|
| `opencv-python` | Image processing, video frame extraction |
| `numpy` | Numerical operations for pixel manipulation |
| `PyQt5` | GUI framework (required even for CLI mode) |
| `Pillow` | Image format support, GIF creation |
| `pyfiglet` | ASCII banner display (optional, enhances CLI) |

### Verify Installation

```bash
python -c "import cv2; import numpy; import PyQt5; from PIL import Image; print('All dependencies OK')"
```

---

## FFmpeg Setup

**⚠️ CRITICAL: FFmpeg is REQUIRED for video processing with audio.**

FFmpeg handles video encoding, decoding, and audio extraction/merging. Without FFmpeg in your system PATH, video processing with audio features will fail.

### Install FFmpeg

**Windows (winget):**
```powershell
winget install FFmpeg
```

**Windows (manual):**
```powershell
# Download from https://www.gyan.dev/ffmpeg/builds/
# Extract to C:\ffmpeg
# Add C:\ffmpeg\bin to system PATH
setx PATH "%PATH%;C:\ffmpeg\bin" /M
```

**macOS (Homebrew):**
```bash
brew install ffmpeg
```

**Linux (apt):**
```bash
sudo apt update && sudo apt install ffmpeg
```

### Verify FFmpeg Installation

```bash
ffmpeg -version
```

### Automated FFmpeg Download (Linux/macOS)

```bash
# Download and install FFmpeg if not present
if ! command -v ffmpeg &> /dev/null; then
    cd /tmp
    wget https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.tar.xz
    tar -xf ffmpeg-release-essentials.tar.xz
    sudo cp ffmpeg-*/*/bin/ffmpeg /usr/local/bin/
    sudo cp ffmpeg-*/*/bin/ffprobe /usr/local/bin/
    rm -rf ffmpeg-*
fi
```

---

## One-Liner Command Patterns

AI agents can chain commands using `&&` or `;` in shell environments.

### Basic One-Liner Pattern (Native Binary — v1.3.0, matches the Python imder.py CLI)

```bash
./imder <base_path> <target_path> [algorithm] [resolution] [sound_option] [quality]
```

Image pairs export Frame (PNG) + GIF + Animation (MP4); runs with a video export MP4 + GIF. Everything lands inside `results/`.

### Basic One-Liner Pattern (Python Source)

```bash
python imder.py <base_path> <target_path> [algorithm] [resolution] [sound_option] [quality]
```

### Command Chaining Examples

**Multiple image processing operations (native binary):**

```bash
./imder image1.png image2.png shuffle && ./imder image3.png image4.png merge && ./imder image5.png image6.png fusion
```

**Video processing with audio (native binary):**

```bash
./imder video1.mp4 video2.mp4 merge 512 target-sound 8 && ./imder photo.png video.mp4 shuffle 512 sound
```

**Batch process with output location:**

```bash
cd /workspace && ./imder /data/base.png /data/target.png merge
```

### Interactive Mode (Not Recommended for Bots)

Interactive mode (`./imder cli`, `cli.bat`, `cli.sh`, or `python imder.py` with no arguments) requires continuous terminal input and is not suitable for AI agents. Use direct one-liner commands instead.

---

## Command Reference

### Syntax (Native Binary — v1.3.0)

```bash
./imder <base_path> <target_path> [algorithm] [resolution] [sound_option] [quality]
```

### Parameters (Native Binary)

| Parameter | Description | Default |
|-----------|-------------|---------|
| `base_path` | Path to base image or video | Required |
| `target_path` | Path to target image or video | Required |
| `algorithm` | `shuffle`, `merge`, `missform`, `fusion` (fusion for image pairs only) | `merge` |
| `resolution` | Processing resolution in pixels | `512` |
| `sound_option` | `mute`, `sound`, `target-sound` | `mute` |
| `quality` | Sound quality 1-10, only with `target-sound` | `3` (30%) |

Image pairs export `image_<timestamp>.png`, `animation_<timestamp>.gif`, and `video_<timestamp>.mp4`; runs with a video export `video_<timestamp>.mp4` and `animation_<timestamp>.gif` — all inside `results/`.

### Syntax (Python Source)

```bash
python imder.py <base_path> <target_path> [algorithm] [resolution] [sound_option] [quality]
```

### Algorithm Options

**For Images:**
- `shuffle` - Random pixel swapping with brightness balance
- `merge` - Grayscale sorting for smooth transitions
- `fusion` - Artistic pixel sorting animations

**For Videos:**
- `shuffle` - Random pixel swapping between frames
- `merge` - Grayscale sorting for frame transitions

### Resolution Options

```
128, 256, 512, 768, 1024, 2048
```

Higher resolutions produce better quality but take longer to process.

### Sound Options

| Option | Description |
|--------|-------------|
| `mute` | No audio (default) |
| `sound` | Synthesize audio from pixel colors |
| `target-sound` | Extract audio from target video |

### Quality Parameter

Used only with `target-sound` option:

| Value | Quality |
|-------|---------|
| 1 | 10% (lowest) |
| 3 | 30% (default) |
| 5 | 50% |
| 7 | 70% |
| 10 | 100% (original) |

### Examples (Native Binary)

**Image to Image (no audio):**

```bash
./imder flower.png obama.png shuffle 512
./imder photo1.jpg photo2.jpg merge 1024
./imder image1.webp image2.webp fusion 256
```

**Video to Video (with target audio):**

```bash
./imder video1.mp4 video2.mp4 merge 512 target-sound 10
./imder clip1.mov clip2.avi merge 512 target-sound 7
./imder video1.mkv video2.mkv shuffle 512 target-sound 5
```

**Image to Video (with pixel sound):**

```bash
./imder photo.png video.mp4 merge 512 sound
./imder image.jpg clip.mov shuffle 512 sound
```

**Video to Image inputs (with generated sound):**

```bash
./imder video.mp4 image.png merge 512 sound
```

### Examples (Python Source)

```bash
python imder.py flower.png obama.png shuffle 512
python imder.py video1.mp4 video2.mp4 merge 256 target-sound 10
```

---

## CLI vs GUI Feature Comparison

IMDER has features exclusive to each mode. Understanding these differences helps AI agents choose the right approach.

### CLI-Only Features

These features are only available via command line:

| Feature | Description |
|---------|-------------|
| **Headless Operation** | No GUI required, fully automated |
| **Batch Processing** | Chain multiple commands with `&&` |
| **One-Liner Execution** | Single command processing |
| **ASCII Banner Interactive Mode** | `imder cli` / `cli.bat` / `cli.sh` guided prompts |

### GUI-Only Features

These features require the graphical interface:

| Feature | Description |
|---------|-------------|
| **Shape Analysis** | Auto-detect and select regions (k-means) |
| **Pen Tool** | Manual mask drawing with include (+) / exclude (-) shapes and Clear Shapes |
| **Smart Analyze** | Refines each drawn shape toward the real object underneath |
| **Pattern Algorithm** | Texture transfer based on color quantization |
| **Disguise Algorithm** | Shape-aware transformations |
| **Navigate Algorithm** | Gradient-guided pixel movement |
| **Swap Algorithm** | Bidirectional pixel exchange |
| **Blend Algorithm** | Physics-inspired animated transitions |
| **Reborn Algorithm** | Shape-pair pixel transplant between drawn regions on base and target |
| **Drawer Algorithm** | Canvas-based sketch to image transformation |
| **Real-time Streamed Preview** | Watch the run live on a frame timeline, then replay or reverse it from the cache |
| **Interactive Shape Selection** | Click to select/deselect regions |

### Shared Features

Available in both CLI and GUI:

- `shuffle`, `merge`, `missform` algorithms (`fusion` also in both, images only)
- **Video processing** (video-to-video, video-to-image, image-to-video)
- **Target audio extraction** and pixel-generated sound
- Resolution selection (any size 1-16384 in the CLI, presets + custom in the GUI)
- Frame export (PNG), Animation export (MP4, GIF)
- Progress tracking (CLI: live text bar, GUI: progress bar with stage info + frame timeline)

---

## Video Processing Rationale

**Why video runs are streamed:**

1. **Time Efficiency**: A 10-second video at 30fps has 300 frames. Animating each frame in the GUI would take 300×10 seconds = 50 minutes minimum. The processing pipeline renders all frames as fast as the machine allows, and the GUI streams them to a frame timeline you can scrub instead of blocking.

2. **Memory Law**: Both the CLI and the GUI stream videos frame by frame — the base frame, the target frame and the processed frame are the only pixels in memory, so long videos no longer load entirely into RAM.

3. **Automation Friendly**: The CLI allows batch processing of multiple videos without user interaction.

4. **Frame Timing**: Video-video runs keep the base video's fps (like the Python library), single-video runs keep the video's own fps, and extra frames beyond the shorter input are ignored — the audio tracks (pixel-sound or target audio) are synthesized at exactly 1/fps per frame so they always match the video duration.

**Video Processing Capabilities:**

- Frame extraction and processing
- Audio extraction from target video
- Audio synthesis from pixel colors
- Output as MP4 with merged audio
- Output as GIF animation
- Automatic frame count matching
- FPS preservation from source video

---

## Limitations

### CLI Mode Limitations

1. **No Shape Analysis**: Cannot use Pattern, Disguise, Navigate, Swap, Blend or Reborn. These require visual shape selection.

2. **No Manual Mask Drawing**: Pen tool and shape selection not available.

3. **Limited Algorithms**: Only shuffle, merge, missform and fusion available.

4. **No Interactive Adjustments**: Cannot tweak parameters during processing.

### GUI Mode Limitations

1. **No Headless Operation**: Requires display and user interaction.

2. **No Batch Processing**: Must process files one at a time manually.

3. **No Command Chaining**: Cannot chain multiple operations.

4. **Image-only Transforms**: Rotate/Flip and the shape tools are disabled for video inputs (video frames are processed as-is).

### FFmpeg Dependencies

1. **Video + Audio Requires FFmpeg**: Without FFmpeg in PATH, video processing with sound fails.

2. **Audio Extraction Requires FFmpeg**: Target sound extraction needs FFmpeg.

3. **Video Encoding Requires FFmpeg**: MP4 output with audio needs FFmpeg.

---

## Troubleshooting

### Issue: "Error: File not found"

**Cause**: Incorrect file path

**Solution**: Use absolute paths and verify file exists:

```bash
python imder.py /absolute/path/to/base.png /absolute/path/to/target.png merge 512
```

### Issue: "Error: Invalid algorithm"

**Cause**: Using unsupported algorithm for file type

**Solution**: For videos, use only `shuffle` or `merge`:

```bash
# Wrong
python imder.py video.mp4 video2.mp4 fusion 512

# Correct
python imder.py video.mp4 video2.mp4 merge 512
```

### Issue: "Error: Target Sound option requires target to be a video file"

**Cause**: Using target-sound with image target

**Solution**: Either change target to video or use `sound` instead:

```bash
# Use video as target
python imder.py image.png video.mp4 merge 512 target-sound 7

# Or use generated sound
python imder.py image.png image2.png merge 512 sound
```

### Issue: "Error: ffmpeg is not installed or not found in PATH"

**Cause**: FFmpeg not installed or not in system PATH

**Solution**: Install FFmpeg and add to PATH (see FFmpeg Setup section)

### Issue: "Error: Missing required arguments"

**Cause**: Not enough arguments provided

**Solution**: Provide base and target (result folder is fixed to `results/`):

```bash
# Wrong
./imder image.png

# Correct
./imder image.png target.png merge
```

### Issue: ImportError or ModuleNotFoundError

**Cause**: Python packages not installed

**Solution**: Install dependencies:

```bash
pip install opencv-python numpy PyQt5 pillow pyfiglet
```

### Issue: No audio in output video

**Cause**: Either FFmpeg not found or mute option used

**Solution**:
1. Verify FFmpeg is in PATH: `ffmpeg -version`
2. Use sound or target-sound option:

```bash
python imder.py video1.mp4 video2.mp4 merge 512 sound
python imder.py base.png target.mp4 merge 512 target-sound 7
```

### Justification: IMDER Itself Has No Known Issues

IMDER is a mature, well-tested tool. When issues occur, they are almost always due to:

1. **Missing Python libraries**: Solved by `pip install` command
2. **FFmpeg not in PATH**: Solved by FFmpeg installation
3. **Invalid file paths**: Solved by using absolute paths
4. **Wrong algorithm for file type**: Solved by using shuffle/merge for videos

IMDER handles all internal error cases gracefully with clear error messages. The tool does not crash, hang, or produce corrupt output when used correctly.

---

## Example Workflows

### Workflow 1: Image Transformation Pipeline

```bash
# Setup — grab the native binary from the releases (no python deps)
cd /workspace
unzip imder_linux_v1.3.0_cpp.zip

# Process multiple images
./imder/imder ../data/image1.png ../data/image2.png shuffle 512 && \
./imder/imder ../data/image3.png ../data/image4.png merge 1024 && \
./imder/imder ../data/image5.png ../data/image6.png fusion 256

# Move results (they land inside ./results)
ls results/
```

### Workflow 2: Video Transformation with Audio

```bash
# Install FFmpeg if needed
command -v ffmpeg || (wget -q https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.tar.xz -O /tmp/ffmpeg.tar.xz && \
    tar -xf /tmp/ffmpeg.tar.xz -C /tmp && \
    sudo cp /tmp/ffmpeg-*/bin/ffmpeg /usr/local/bin/ && \
    sudo cp /tmp/ffmpeg-*/bin/ffprobe /usr/local/bin/ && \
    rm -rf /tmp/ffmpeg*)

# Process videos with target audio (native binary)
./imder video1.mp4 video2.mp4 merge 512 target-sound 10 && \
./imder photo.png video.mp4 shuffle 512 target-sound 8 && \
./imder intro.mp4 main.mp4 merge 512 target-sound 7
```

### Workflow 3: Batch Video Processing with Generated Audio

```bash
# Create the results directory (fixed output location)
mkdir -p results

# Process all pairs (native binary)
./imder video_A1.mp4 video_A2.mp4 merge 512 sound && \
./imder video_B1.mp4 video_B2.mp4 merge 512 sound && \
./imder video_C1.mp4 video_C2.mp4 merge 512 sound

# Outputs land inside ./results as video_<timestamp>.mp4 and animation_<timestamp>.gif
ls results/
```

### Workflow 4: Single Command with All Parameters

```bash
./imder /path/to/base.png /path/to/target.mp4 merge 1024 target-sound 10
```

This processes base.png against target.mp4 using the merge algorithm at 1024x1024 resolution, extracting audio from the target video at 100% quality into results/ as video_<timestamp>.mp4.

---

## Summary for AI Agents

1. **Prefer the native binary**: one small exe, no python deps, same positional syntax as the Python imder.py CLI
2. **Always use one-liner commands**: `./imder base target merge && ./imder base target shuffle`
3. **Fixed output location**: everything lands inside `results/` — image pairs produce png + gif + mp4, runs with a video produce mp4 + gif
4. **Install FFmpeg**: Required for video + audio features
5. **Use absolute paths**: Avoid relative path issues
6. **For videos**: Use only `shuffle`, `merge` or `missform` algorithms; fusion is image-pairs only
7. **For audio**: Use `sound` (synthesized) or `target-sound` (extracted), with quality 1-10 for target-sound
8. **Output names**: `image_<timestamp>.png`, `animation_<timestamp>.gif`, `video_<timestamp>.mp4` inside `results/`
9. **No shape analysis in CLI**: Use the GUI for mask-based algorithms and Reborn
10. **Video runs stream**: no full-video memory loads; video-video runs pick the fps of the shorter-duration side like the Python CLI

---

**For questions or issues, visit: https://github.com/HAKORADev/IMDER**
