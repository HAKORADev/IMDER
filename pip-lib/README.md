# IMDER

[![PyPI version](https://badge.fury.io/py/imder.svg)](https://pypi.org/project/imder/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Image & Video Pixel Blender - Plugin Edition**

A streamlined, CLI-first Python package for pixel-sorting image and video transformations. This is the lightweight, automation-friendly version of the [full IMDER tool](https://github.com/HAKORADev/IMDER), designed for integration into pipelines, scripts, and other applications.

## What it does

IMDER rearranges pixels between a base and target media using spatial sorting algorithms, creating glitch-art style transitions and data-bending effects without neural networks or heavy dependencies.

Supported formats:
- **Images**: PNG, JPG, WebP
- **Video**: MP4, AVI, MOV, MKV, FLV, WMV

## Installation

```bash
pip install imder
```

Requirements: Python 3.8+, OpenCV, NumPy, Pillow. FFmpeg optional (required for audio features).

## Usage

### As a Python Library

```python
import imder

# Process two images, output PNG and GIF
imder.process(
    base="path/to/image1.jpg",
    target="path/to/image2.jpg",
    result="path/to/output/",
    results=["png", "gif"],      # Options: png, gif, mp4
    algo="shuffle",              # shuffle, merge, missform, fusion
    res=512,                     # 128, 256, 512, 1024, 2048
    sound="mute"                 # mute, gen (generated), target (from video)
)

# CLI mode
imder.launch_interactive()
```

### Command Line

```bash
# Interactive mode
imder

# Direct processing
imder base.jpg target.jpg ./output --results gif mp4 --algo missform --res 1024

# With target audio extraction (quality 1-10)
imder base.jpg target.mp4 ./out --results mp4 --sound target --sq 5
```

### Algorithm Selection

| Algorithm | Images | Videos | Description |
|-----------|--------|--------|-------------|
| `shuffle` | ✅ | ✅ | Random pixel swapping by luminosity bins |
| `merge` | ✅ | ✅ | Grayscale-sorted pixel replacement |
| `missform` | ✅ | ✅ | Binary threshold morphing |
| `fusion` | ✅ | ❌ | Animated pixel sorting with interpolation |

**Sound options:**
- `mute`: Silent output
- `gen`: Generate synthetic audio from pixel color values
- `target`: Extract audio from target video (requires FFmpeg, `--sq` for quality 1-10)

## Limitations vs Full Version

This PyPI distribution is a **minimal, headless build** optimized for automation:

| Feature | PyPI Package | [Full Version](https://github.com/HAKORADev/IMDER) |
|---------|--------------|---------------------------------------------------|
| Interface | CLI / Python API | GUI (PyQt5) + CLI |
| Shape Analysis | ❌ | ✅ (Interactive segmentation) |
| Pen Tool | ❌ | ✅ (Manual mask drawing) |
| Preview | ❌ | ✅ (Real-time animation preview) |
| Algorithms | 4 core modes | 9+ modes (Pattern, Disguise, Navigate, etc.) |
| Dependencies | Lightweight | Full Qt stack |

Use this package when you need to **batch process** files, integrate into **web services**, or call from **other Python applications**. Use the [GitHub version](https://github.com/HAKORADev/IMDER) for interactive artistic workflows.

## GitHub Repository

For the full-featured GUI application, source code, and detailed algorithm documentation:  
👉 **https://github.com/HAKORADev/IMDER**

## License

MIT License
