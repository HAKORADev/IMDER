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
# Clone the repository
git clone https://github.com/HAKORADev/IMDER.git && cd IMDER/src

# Install dependencies (one-liner)
pip install opencv-python numpy PyQt5 pillow pyfiglet

# Process files immediately (one-liner)
python imder.py /path/to/base.png /path/to/target.png shuffle 512

# Chain multiple operations
python imder.py base1.png target1.png merge 256 && python imder.py base2.png target2.png merge 256 && python imder.py base3.png target3.png merge 256
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

### Basic One-Liner Pattern

```bash
python imder.py <base_path> <target_path> [algorithm] [resolution] [sound_option] [quality]
```

### Command Chaining Examples

**Multiple image processing operations:**

```bash
python imder.py image1.png image2.png shuffle 512 && python imder.py image3.png image4.png merge 512 && python imder.py image5.png image6.png fusion 256
```

**Video processing with audio:**

```bash
python imder.py video1.mp4 video2.mp4 merge 256 target-sound 10 && python imder.py photo.png video.mp4 shuffle 512 sound
```

**Batch process with output location:**

```bash
cd /workspace && mkdir -p results && cd IMDER/src && python imder.py /data/base.png /data/target.png merge 512 mute && mv results/* /data/results/
```

### Interactive Mode (Not Recommended for Bots)

Interactive mode (`python imder.py cli`) requires continuous terminal input and is not suitable for AI agents. Use direct one-liner commands instead.

---

## Command Reference

### Syntax

```bash
python imder.py <base_path> <target_path> [algorithm] [resolution] [sound_option] [quality]
```

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `base_path` | Path to base image or video | Required |
| `target_path` | Path to target image or video | Required |
| `algorithm` | Processing algorithm | `merge` |
| `resolution` | Processing resolution (pixels) | `512` |
| `sound_option` | Audio option: `mute`, `sound`, `target-sound` | `mute` |
| `quality` | Quality 1-10 (for target-sound only) | `3` |

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

### Examples

**Image to Image (no audio):**

```bash
python imder.py flower.png obama.png shuffle 512
python imder.py photo1.jpg photo2.jpg merge 1024
python imder.py image1.webp image2.webp fusion 256
```

**Video to Video (with target audio):**

```bash
python imder.py video1.mp4 video2.mp4 merge 256 target-sound 10
python imder.py clip1.mov clip2.avi merge 512 target-sound 7
python imder.py video1.mkv video2.mkv shuffle 256 target-sound 5
```

**Image to Video (with pixel sound):**

```bash
python imder.py photo.png video.mp4 merge 512 sound
python imder.py image.jpg clip.mov shuffle 256 sound
```

**Video to Image (with generated sound):**

```bash
python imder.py video.mp4 image.png merge 512 sound
```

---

## CLI vs GUI Feature Comparison

IMDER has features exclusive to each mode. Understanding these differences helps AI agents choose the right approach.

### CLI-Only Features

These features are only available via command line:

| Feature | Description |
|---------|-------------|
| **Video Processing** | Frame-by-frame video transformations |
| **Target Audio Extraction** | Extract audio from target video |
| **Batch Processing** | Chain multiple commands with `&&` |
| **Headless Operation** | No GUI required, fully automated |
| **One-Liner Execution** | Single command processing |

### GUI-Only Features

These features require the graphical interface:

| Feature | Description |
|---------|-------------|
| **Shape Analysis** | Auto-detect and select regions |
| **Pen Tool** | Manual mask drawing |
| **Pattern Algorithm** | Texture transfer based on color quantization |
| **Disguise Algorithm** | Shape-aware transformations |
| **Navigate Algorithm** | Gradient-guided pixel movement |
| **Swap Algorithm** | Bidirectional pixel exchange |
| **Blend Algorithm** | Physics-inspired animated transitions |
| **Real-time Preview** | Watch animation as it processes |
| **Interactive Shape Selection** | Click to select/deselect regions |

### Shared Features

Available in both CLI and GUI:

- `shuffle` algorithm
- `merge` algorithm
- `fusion` algorithm (CLI only, GUI supports it)
- Resolution selection (128-2048)
- Mute/Sound audio options
- Frame export (PNG)
- Animation export (MP4, GIF)
- Progress tracking (CLI: text output, GUI: progress bar)

---

## Video Processing Rationale

**Why video processing is CLI-only:**

1. **Time Efficiency**: A 10-second video at 30fps has 300 frames. Animating each frame in the GUI would take 300×10 seconds = 50 minutes minimum. CLI processes all frames in seconds.

2. **No Visual Benefit**: Watching 300 frames animate sequentially provides no value—the final result is what matters.

3. **Automation Friendly**: CLI allows batch processing of multiple videos without user interaction.

4. **Resource Efficiency**: GUI mode allocates resources to real-time rendering. CLI mode allocates all resources to batch processing.

5. **Pipeline Integration**: CLI can be integrated into larger automated pipelines (data processing, content generation, etc.).

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

1. **No Shape Analysis**: Cannot use Pattern, Disguise, Navigate, Swap, or Blend algorithms. These require visual shape selection.

2. **No Real-time Preview**: Cannot watch animation as it processes. Only see final output.

3. **No Manual Mask Drawing**: Pen tool and shape selection not available.

4. **Limited Algorithms**: Only shuffle, merge, and fusion available.

5. **No Interactive Adjustments**: Cannot tweak parameters during processing.

### GUI Mode Limitations

1. **No Video Processing**: Cannot process video files.

2. **No Target Audio**: Cannot extract audio from videos.

3. **No Headless Operation**: Requires display and user interaction.

4. **No Batch Processing**: Must process files one at a time manually.

5. **No Command Chaining**: Cannot chain multiple operations.

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

**Solution**: Provide at minimum base_path and target_path:

```bash
# Wrong
python imder.py image.png

# Correct
python imder.py image.png target.png
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
# Setup
cd /workspace
git clone https://github.com/HAKORADev/IMDER.git
cd IMDER/src
pip install opencv-python numpy PyQt5 pillow pyfiglet

# Process multiple images
python imder.py ../data/image1.png ../data/image2.png shuffle 512 && \
python imder.py ../data/image3.png ../data/image4.png merge 1024 && \
python imder.py ../data/image5.png ../data/image6.png fusion 256 && \

# Move results
mv results/*.png ../data/output/ 2>/dev/null || true
mv results/*.mp4 ../data/output/ 2>/dev/null || true
mv results/*.gif ../data/output/ 2>/dev/null || true
```

### Workflow 2: Video Transformation with Audio

```bash
# Install FFmpeg if needed
command -v ffmpeg || (wget -q https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.tar.xz -O /tmp/ffmpeg.tar.xz && \
    tar -xf /tmp/ffmpeg.tar.xz -C /tmp && \
    sudo cp /tmp/ffmpeg-*/bin/ffmpeg /usr/local/bin/ && \
    sudo cp /tmp/ffmpeg-*/bin/ffprobe /usr/local/bin/ && \
    rm -rf /tmp/ffmpeg*)

# Process videos with target audio
python imder.py video1.mp4 video2.mp4 merge 256 target-sound 10 && \
python imder.py photo.png video.mp4 shuffle 512 target-sound 8 && \
python imder.py intro.mp4 main.mp4 merge 512 target-sound 7
```

### Workflow 3: Batch Video Processing with Generated Audio

```bash
cd IMDER/src

# Create output directory
mkdir -p ../processed

# Process all pairs (assuming naming convention)
python imder.py video_A1.mp4 video_A2.mp4 merge 256 sound && \
python imder.py video_B1.mp4 video_B2.mp4 merge 256 sound && \
python imder.py video_C1.mp4 video_C2.mp4 merge 256 sound && \

# Move results
mv results/*.mp4 ../processed/
mv results/*.gif ../processed/
```

### Workflow 4: Single Command with All Parameters

```bash
python imder.py /path/to/base.png /path/to/target.png shuffle 1024 target-sound 10
```

This processes base.png with target.png using shuffle algorithm at 1024x1024 resolution, extracting audio from target.mp4 at 100% quality.

---

## Summary for AI Agents

1. **Always use one-liner commands**: `python imder.py <args> && python imder.py <args>`
2. **Install dependencies first**: `pip install opencv-python numpy PyQt5 pillow pyfiglet`
3. **Install FFmpeg**: Required for video + audio features
4. **Use absolute paths**: Avoid relative path issues
5. **For videos**: Use only `shuffle` or `merge` algorithms
6. **For audio**: Use `sound` (synthesized) or `target-sound` (extracted)
7. **Quality parameter**: Only for `target-sound`, values 1-10
8. **Output location**: Results saved to `IMDER/src/results/` directory
9. **No shape analysis in CLI**: Use GUI for advanced algorithms
10. **Video processing is CLI-only**: For efficiency and automation

---

**For questions or issues, visit: https://github.com/HAKORADev/IMDER**
