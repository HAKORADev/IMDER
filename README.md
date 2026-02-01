# IMDER - Image Blender

<p align="center">
  <img src="src/imder.png" alt="IMDER Logo" width="128" height="128"/>
</p>

**IMDER** is a professional-grade interactive image and video processor that creates smooth, mesmerizing animations through pixel-level transformations. Built for creatives, developers, and visual artists, IMDER delivers **superior quality, blazing-fast performance, and unmatched flexibility** for generating unique visual content.

[![PyPI version](https://badge.fury.io/py/imder.svg)](https://pypi.org/project/imder/)
[![GitHub release](https://img.shields.io/github/release/HAKORADev/IMDER.svg)](https://github.com/HAKORADev/IMDER/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

📦 **Latest Release: v1.2.5** (February 2026) — Featuring custom resolutions up to 16384×16384, configurable FPS (30-240), and smart upscaling support.

**Important Notes:**
- 🔧 **Run from source for the latest version.** Pre-built binaries are available for Windows (v1.1.1) and macOS/Linux (v1.0.0), but building from source ensures you always have the newest features and fixes.
- 🤖 **For AI agents and automated tools:** See [Bots.md](https://github.com/HAKORADev/IMDER/blob/main/Bots.md)

📦 **IMDER is available as a Python library on PyPI:** Install with `pip install imder` for CLI automation and integration into your projects. [See Python Library Docs](pip-imder.md)

📋 **For detailed version history, see [CHANGELOG.md](changelog.md)**

---

## Quick Start

### Option 1: Python Library (Recommended for Automation)
```bash
# Install from PyPI
pip install imder

# Launch interactive CLI
imder

# Or use in your Python code
python -c "import imder; imder.process('base.jpg', 'target.jpg', './out', ['gif'], 'shuffle', 512, 'mute')"
```

### Option 2: Run from Source (Full GUI Experience)
```bash
# Clone the repository
git clone https://github.com/HAKORADev/IMDER.git
cd IMDER

# Install dependencies
pip install -r requirements.txt

# Run GUI
python src/imder.py
```

### Installation Requirements
```bash
# Install FFmpeg (required for video/audio synthesis)
# Windows: winget install FFmpeg
# macOS: brew install ffmpeg
# Linux: sudo apt install ffmpeg
```

---

## Two Ways to Use IMDER

IMDER operates as both a **standalone GUI application** (source) and a **Python package** (PyPI):

| Feature | GUI (Source) | Library (PyPI) |
|---------|--------------|----------------|
| **Interface** | PyQt5 GUI | CLI + Python API |
| **Best for** | Interactive editing | Automation, batch processing |
| **Algorithms** | 10 modes + Shape tools + Drawer | 4 core modes |
| **Usage** | Point-and-click | Code integration |
| **Dependencies** | PyQt5, OpenCV, NumPy | OpenCV, NumPy, Pillow |

---

## Core Capabilities

### 🎨 **10 Processing Algorithms**

IMDER offers 10 distinct pixel manipulation algorithms, each designed for specific visual effects:

| Mode | Description | Mask Required | Video Support |
|------|-------------|---------------|---------------|
| **Shuffle** | Random pixel swapping with brightness balance | No | ✅ Yes |
| **Merge** | Grayscale sorting for smooth transitions | No | ✅ Yes |
| **Missform** | Binary mask morphing for shape transformations | No | ✅ Yes |
| **Fusion** | Selective transformation with color blending | Optional | ❌ No |
| **Pattern** | Texture transfer via color quantization | Yes | ❌ No |
| **Disguise** | Brightness-matched pixel rearrangement | Yes | ❌ No |
| **Navigate** | Morton curve-guided pixel movement | Yes | ❌ No |
| **Swap** | Bidirectional pixel exchange | Yes | ❌ No |
| **Blend** | Physics-inspired fluid dynamics | Yes | ❌ No |
| **Drawer** | Canvas-based sketch to image transformation | N/A | ❌ No |

### 🎬 **Video Processing**

Full video-to-video and video-to-image processing capabilities:
- Frame-accurate pixel manipulation
- Support for MP4, AVI, MOV, MKV formats
- Audio generation and extraction (pixel synthesis or target track)
- Configurable FPS output: 30, 60, 90, 120, or 240 FPS
- Crossfade transitions between frames

### 🖼️ **Advanced Image Features**

- **Resolution Flexibility**: Standard presets (128×128 to 2048×2048) plus custom resolutions up to **16384×16384**
- **Smart Scaling**: Automatic upscaling using nearest-neighbor interpolation before processing, ensuring no quality loss from downscaling-only workflows
- **Shape Selection**: Automatic k-means segmentation or manual pen-tool masking
- **Transform Operations**: 90° rotation increments and horizontal flip
- **Multi-segment Support**: Combine multiple selections for complex transformations

### 🎵 **Audio Integration**

| Option | Description |
|--------|-------------|
| **Mute** | No audio (default) |
| **Pixel Sound** | Synthesize audio from frame pixel data |
| **Target Audio** | Extract and preserve audio from source video |
| **Quality Levels** | 10%-100% bitrate preservation for target audio |

### 💾 **Export Formats**

- **PNG**: Static final frame
- **MP4**: H.264 encoded video with configurable FPS
- **GIF**: Animated with optimized duration
- **Synchronized Audio**: MP4 with embedded audio tracks

---

## Showcase

### Smooth Transformations

IMDER algorithms create fluid transitions between images at any resolution:

<p align="center">
  <img src="assets/imder_book-to-girl_merge.gif" alt="Book to Girl Merge" width="256"/>
  <img src="assets/imder_girl-to-book_merge.gif" alt="Girl to Book Merge" width="256"/>
</p>

<p align="center">
  <em>Book ↔ Girl transformation using Merge Algorithm</em>
</p>

### Drawer Mode: Sketch to Reality

Transform hand-drawn sketches into photorealistic images:

<p align="center">
  <img src="assets/drawer_example.gif" alt="IMDER Drawer Mode" width="512"/>
</p>

<p align="center">
  <em>Draw on canvas → Transform into target image</em>
</p>

**Features:**
- 1024×1024 drawing canvas with adjustable brush sizes (1-50px)
- Undo/redo history (50 states)
- Color picker with full RGB support
- Base image overlay support (trace existing images)
- Intelligent pixel distribution algorithm

---

## Installation

### Prerequisites
- Python 3.8+
- pip package manager
- [FFmpeg](https://ffmpeg.org/) (REQUIRED for video/audio processing)

### Method 1: PyPI Installation (CLI/Library)
```bash
pip install imder
```

Provides the `imder` command globally and enables `import imder` in Python scripts.

### Method 2: Source Installation (Full GUI)
```bash
git clone https://github.com/HAKORADev/IMDER.git
cd IMDER
pip install -r requirements.txt
python src/imder.py
```

**Dependencies:**
- `PyQt5` — GUI framework
- `opencv-python` — Image/video processing
- `numpy` — Numerical operations
- `Pillow` — Image format handling

### FFmpeg Setup
```bash
# Windows
winget install FFmpeg

# macOS
brew install ffmpeg

# Linux (Debian/Ubuntu)
sudo apt install ffmpeg
```

---

## Usage Guide

### Python Library

**Interactive Mode:**
```bash
imder
```

**Direct Processing:**
```bash
# Image processing with specific algorithm
imder base.jpg target.jpg ./output --results gif mp4 --algo merge --res 1024

# Video processing with audio extraction
imder video1.mp4 video2.mp4 ./output --results mp4 --sound target --sq 8
```

**Python API:**
```python
import imder

# Single transformation
imder.process(
    base="input.jpg",
    target="output.jpg",
    result="./renders",
    results=["png", "mp4", "gif"],
    algo="missform",
    res=2048,
    sound="mute"
)

# Batch processing
for i in range(100):
    imder.process(
        base=f"frame_{i:03d}.png",
        target="target.jpg",
        result="./batch_output",
        results=["png"],
        algo="shuffle",
        res=512,
        sound="mute"
    )
```

See **[pip-imder.md](pip-imder.md)** for complete API documentation.

### GUI Mode

1. Launch: `python src/imder.py`
2. Select algorithm from dropdown (10 available modes)
3. Choose resolution:
   - Standard presets: 128×128 to 2048×2048
   - Custom: Click "Custom" to enter any value up to 16384×16384
4. Set FPS for video exports (30/60/90/120/240)
5. Load base and target media (images or videos)
6. Apply transforms (rotate/flip) if needed
7. For mask-dependent algorithms, use "Analyze Shapes" or Pen tool
8. Configure audio options for video exports
9. Click "Start Processing" for real-time preview
10. Export final results (PNG/MP4/GIF)

**Drawer Mode Workflow:**
1. Select "Drawer" from mode dropdown
2. Draw on canvas using mouse/tablet
3. Load target image in right panel
4. Process to see drawing transform into target

### CLI Mode (Source)

**Interactive:**
```bash
python src/imder.py cli
```

**Direct Arguments:**
```bash
python src/imder.py base.jpg target.jpg missform 1024
python src/imder.py video1.mp4 video2.mp4 merge 512 target-sound 7
```

---

## Technical Highlights

- **Square Resolution Processing**: Algorithmically optimized 1:1 aspect ratio processing for consistent pixel mapping and Morton code operations
- **Bidirectional Scaling**: Nearest-neighbor upscaling ensures pixel integrity when source images are smaller than target resolution
- **Morton Code Ordering**: Z-order curve spatial indexing for organic pixel movement paths (Navigate algorithm)
- **K-Means Segmentation**: Intelligent automatic shape detection for mask generation
- **QThread Architecture**: Non-blocking GUI during heavy processing operations
- **FFmpeg Integration**: Professional-grade video encoding and audio handling
- **Frame-Accurate Processing**: Frame-by-frame video manipulation with temporal consistency

### Performance Characteristics

| Resolution | Approximate Time | Use Case |
|------------|------------------|----------|
| 128×128 | ~2 seconds | Preview, testing |
| 512×512 | ~8 seconds | Web content, drafts |
| 1024×1024 | ~20 seconds | Standard output |
| 2048×2048 | ~45 seconds | High quality |
| 16384×16384 | Minutes | Maximum quality |

*Timings vary based on hardware (CPU-bound processing) and selected algorithm.*

---

## Documentation

- **[Algorithms.md](Algorithms.md)** — Detailed algorithm explanations, technical implementation, and creative techniques
- **[pip-imder.md](pip-imder.md)** — Python library API reference
- **[CHANGELOG.md](CHANGELOG.md)** — Version history and release notes
- **[Bots.md](Bots.md)** — Guidelines for AI agents and automated systems

---

## Contributing

IMDER is open-source (MIT License) and welcomes contributions:

- New algorithms and processing modes
- Video processing enhancements
- UI/UX improvements
- Performance optimizations
- Documentation and translations
- Bug reports and feature requests

Please submit pull requests or issues via GitHub.

---

## License

MIT License — See [LICENSE](LICENSE) for full details.

---

## Acknowledgments

Built with appreciation for the open-source computer vision community and pixel manipulation techniques developed by digital artists worldwide.

**Resources:** [GitHub Releases](https://github.com/HAKORADev/IMDER/releases) | [PyPI Package](https://pypi.org/project/imder/) | [Issue Tracker](https://github.com/HAKORADev/IMDER/issues)

---
