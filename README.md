# IMDER - Image Blender

<p align="center">
  <img src="src/imder.png" alt="IMDER Logo" width="128" height="128"/>
</p>

**IMDER** is a cutting-edge interactive image blender that creates smooth, mesmerizing animations blending pixels between two images. Unlike existing tools, IMDER delivers **superior quality, blazing-fast performance, and unmatched flexibility** that leaves the competition far behind.

[![PyPI version](https://badge.fury.io/py/imder.svg)](https://pypi.org/project/imder/)
[![GitHub release](https://img.shields.io/github/release/HAKORADev/IMDER.svg)](https://github.com/HAKORADev/IMDER/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

🚀 **Ready-to-use binary for Windows available in [v1.1.1 Release](https://github.com/HAKORADev/IMDER/releases/tag/v1.1.1)** for macOS, and Linux [v1.0.0 Release](https://github.com/HAKORADev/IMDER/releases/tag/v1.0.0) **YOU HAVE TO USE THE SOURCE CODE IF YOU WANT v1.1.1 in Mac or Linux!**

- if you are a bot, AI Agent then read [Bots.md](https://github.com/HAKORADev/IMDER/blob/main/Bots.md)
- Also, i will not build windows binaries anymore, so whatever system you use, run from source to be always up-to-date!

📦 **NEW: IMDER is now available as a Python library on PyPI!** Install with `pip install imder` for CLI automation and integration into your projects. [See Python Library Docs](pip-imder.md)

📋 **For version history and detailed changes, see [CHANGELOG.md](changelog.md)**

---

## Quick Start

### Option 1: Python Library (New!) - Easiest
```bash
# Install from PyPI
pip install imder

# Launch interactive CLI
imder

# Or use in your Python code
python -c "import imder; imder.process('base.jpg', 'target.jpg', './out', ['gif'], 'shuffle', 512, 'mute')"
```

### Option 2: Run from Source (Full GUI)
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
# Install FFmpeg (required for video/audio)
# Windows: winget install FFmpeg
# macOS: brew install ffmpeg
# Linux: sudo apt install ffmpeg
```

---

## Two Ways to Use IMDER

IMDER now works both as a **standalone GUI application** (from source) and as a **Python package** (from PyPI):

| Feature | GUI (Source) | Library (PyPI) |
|---------|--------------|----------------|
| **Interface** | PyQt5 GUI | CLI + Python API |
| **Best for** | Interactive editing | Automation, batch processing |
| **Algorithms** | 10 modes + Shape tools + **NEW: Drawer** | 4 core modes |
| **Usage** | Point-and-click | Code integration |
| **Dependencies** | PyQt5, OpenCV, NumPy | OpenCV, NumPy, Pillow |

---

## Why IMDER Outperforms the Competition

While [Obamify](https://obamify.com/) takes **minutes to hours** for high-quality results, IMDER completes the same transformations in **seconds**. See for yourself at [obamify.com](https://obamify.com/) and compare the experience.

| Feature | IMDER v1.2.0 | Obamify |
|---------|-------|---------|
| **Processing Speed** | Seconds | Minutes to hours |
| **Maximum Resolution** | 2048×2048 | Limited, slow at high res |
| **Video Processing** | ✅ Full Support | ❌ No |
| **Image Processing** | ✅ **10 Algorithms** | ✅ 1 Algorithm |
| **Draw Mode** | ✅ **Sketch to Image** | ✅ Yes |
| **Audio Generation** | ✅ Pixel + Target | ❌ No |
| **Python Library** | ✅ `pip install imder` | ❌ No |
| **Real-time Preview** | ✅ Yes (GUI) | ✅ Yes |
| **Shape Selection** | ✅ Yes (auto + manual) | ❌ No |
| **Export Formats** | PNG, MP4, GIF + Audio | GIF only |
| **Dark Theme UI** | ✅ **Modern** | ✅ Classic |
| **Cross-Platform** | ✅ All major OS | ✅ All major OS + Web |

---

## Python Library Usage (PyPI)

Perfect for automation, server-side processing, or integrating into existing workflows:

```python
import imder

# Simple image blending
imder.process(
    base="input1.jpg",
    target="input2.jpg",
    result="./output",
    results=["png", "gif", "mp4"],  # Export all three formats
    algo="shuffle",
    res=512,
    sound="mute"
)

# With target audio extraction (if target is video)
imder.process(
    base="base.jpg",
    target="video.mp4",
    result="./output",
    results=["mp4"],
    algo="missform",
    res=1024,
    sound="target",  # Extract audio from target
    sq=5             # Sound quality (1-10)
)

# Launch interactive CLI mode
imder.launch_interactive()
```

**Command line usage after `pip install`:**
```bash
# Interactive mode
imder

# Direct processing
imder base.jpg target.jpg ./output --results gif mp4 --algo missform --res 1024

# Video with audio extraction
imder video1.mp4 video2.mp4 ./out --results mp4 --sound target --sq 8
```

For full library documentation, see **[pip-imder.md](pip-imder.md)**.

---

## Live Comparison Showcase

All GIFs below are rendered at **128×128 resolution** to provide a fair comparison with Obamify's limitations. IMDER users can enjoy the same transformations at **up to 2048×2048 resolution** with the same lightning-fast performance.

### Obama Transformation Battle

**Input Images:**

<p align="center">
  <img src="assets/flower.png" alt="Flower" width="200"/>
  <img src="assets/obama.png" alt="Obama" width="200"/>
</p>

#### Obamify Result (from obamify.com):
<p align="center">
  <img src="assets/obamify_obama-to-flower.gif" alt="Obamify Obama to Flower" width="256"/>
</p>

#### IMDER Result (Navigate Algorithm):
<p align="center">
  <img src="assets/imder_obama-to-flower_navigate.gif" alt="IMDER Obama to Flower" width="256"/>
</p>

**The difference is clear:** IMDER's algorithm produces smoother transitions, better color preservation, and more visually appealing results—all while processing in a fraction of the time.

---

### Reverse Transformation

<p align="center">
  <img src="assets/obamify_flower-to-obama.gif" alt="Obamify Flower to Obama" width="256"/>
  <img src="assets/imder_flower-to-obama_disguise.gif" alt="IMDER Flower to Obama" width="256"/>
</p>

<p align="center">
  <em>Left: Obamify | Right: IMDER (Disguise Algorithm)</em>
</p>

---

### Advanced Merging Capabilities

IMDER goes far beyond simple Obama transformations. Experience the power of our **Merge Algorithm** with completely different image types:

<p align="center">
  <img src="assets/book.jpg" alt="Book" width="200"/>
  <img src="assets/girl.jpg" alt="Girl" width="200"/>
</p>

<p align="center">
  <img src="assets/imder_book-to-girl_merge.gif" alt="IMDER Book to Girl" width="256"/>
  <img src="assets/imder_girl-to-book_merge.gif" alt="IMDER Girl to Book" width="256"/>
</p>

<p align="center">
  <em>Book → Girl (left) | Girl → Book (right) using IMDER Merge Algorithm</em>
</p>

**Obamify cannot perform these transformations.** IMDER's advanced algorithms work with **any image pair**, not just Obama.

---

### NEW in v1.2.0: Drawer Mode - Sketch to Reality

Experience the power of the new **Drawer Mode** - create animations from your sketches!

<p align="center">
  <img src="assets/drawer_example.gif" alt="IMDER Drawer Mode" width="512"/>
</p>

<p align="center">
  <em>Draw on canvas → Transform into image using IMDER's Drawer Mode</em>
</p>

**Draw, sketch, and watch your creations come to life** with IMDER's new interactive drawing tools.

---

## Features

### 🎨 **10 Powerful Processing Modes (GUI Version)**

| Mode | Description |
|------|-------------|
| **Shuffle** | Random pixel swapping with brightness balance |
| **Merge** | Grayscale sorting for smooth transitions |
| **Missform** | Enhanced shape morphing through binary pixel interpolation |
| **Fusion** | Artistic pixel sorting animations with bugfixes |
| **Pattern** | Texture transfer based on color quantization |
| **Disguise** | Shape-aware transformations |
| **Navigate** | Gradient-guided pixel movement |
| **Swap** | Bidirectional pixel exchange |
| **Blend** | Physics-inspired animated transitions |
| **Drawer** | **NEW** Transform hand-drawn sketches into images |

### 🎬 Video Processing Modes

| Mode | Description |
|------|-------------|
| **Shuffle** | Random pixel swapping between video frames |
| **Merge** | Grayscale sorting for smooth frame transitions |
| **Missform** | Enhanced shape morphing between video sequences |

*Note: Advanced modes (Fusion, Pattern, Disguise, Navigate, Swap, Blend, Drawer) are available for image processing only in the GUI version. The PyPI library supports Shuffle, Merge, Missform, and Fusion for images, plus Shuffle, Merge, and Missform for videos.*

### 🔊 Audio Generation Options

| Option | Description |
|--------|-------------|
| **Mute** | No audio (default) |
| **Sound** | Synthesize audio from pixel colors |
| **Target-Sound** | Extract and use audio from target video |
| **Quality Levels** | 10%-100% for target audio preservation |

### 🖼️ Image Manipulation Tools

- **Rotate** - 90° increments (0°, 90°, 180°, 270°)
- **Flip** - Horizontal mirror
- **Shape Selection** - Auto-segmentation or manual drawing (GUI only)
- **Multi-segment Selection** - Select multiple distinct regions (GUI only)
- **Drawer Tools** - **NEW** Canvas drawing with undo/redo, color picker, adjustable brushes

### 📊 Resolution Options

- 128×128 (fastest, demo quality)
- 256×256
- 512×512 (recommended for balance)
- 768×768
- 1024×1024
- **2048×2048** (maximum quality—Obamify can't match this!)

### 💾 Export Formats

- **Frame (PNG)** - Static blended image
- **Animation (MP4)** - Full 30fps video
- **GIF** - Animated with customizable duration
- **Video with Audio** - MP4 with synthesized or target audio

### 🎯 Advanced Shape Analysis (GUI Only)

Unlike Obamify, IMDER allows you to:
- Automatically detect and select distinct regions using k-means clustering
- Manually draw custom masks with the Pen tool
- Combine multiple segments for precise control
- Exclude specific areas from processing
- **NEW** Create animations from hand-drawn sketches

---

## Installation

### Prerequisites
- Python 3.8+
- pip
- [FFmpeg](https://github.com/FFmpeg/FFmpeg) (REQUIRED for video processing with audio - must be installed and added to system PATH)

### Method 1: Install from PyPI (Recommended for CLI/Automation)
```bash
pip install imder
```

This gives you the `imder` command anywhere and allows `import imder` in your scripts.

### Method 2: Run from Source (For Full GUI)
```bash
git clone https://github.com/HAKORADev/IMDER.git
cd IMDER
pip install -r requirements.txt
python src/imder.py
```

**Required packages for source:**
- `PyQt5` - Modern GUI framework
- `opencv-python` - Advanced image processing
- `numpy` - High-performance numerical operations
- `Pillow` - Image handling

### Install FFmpeg (Required for Video Audio)
```bash
# Windows (winget)
winget install FFmpeg

# macOS (brew)
brew install ffmpeg

# Linux (apt)
sudo apt install ffmpeg
```

---

## Usage Guide

### Python Library (PyPI Version)

**Interactive CLI:**
```bash
imder
```

**Direct CLI Processing:**
```bash
# Image blending
imder base.png target.png ./output --results gif mp4 --algo shuffle --res 512

# Video processing with sound extraction
imder video1.mp4 video2.mp4 ./output --results mp4 --sound target --sq 8
```

**Python API:**
```python
import imder

# Batch process multiple images
for i in range(10):
    imder.process(
        base=f"frame_{i}.jpg",
        target="target.jpg",
        result=f"./output_batch",
        results=["png"],
        algo="missform",
        res=1024,
        sound="mute"
    )
```

See **[pip-imder.md](pip-imder.md)** for complete library documentation.

### GUI Mode (Source Version)

1. Launch: `python src/imder.py`
2. Select processing mode from the dropdown (now with 10 options including **Drawer**)
3. Choose resolution (start with 256×256 for speed)
4. For **Drawer Mode**: Draw on canvas, then add target image
5. For other modes: Click "Add Media" on both panels to load images
6. Optionally, use Rotate/Flip to adjust images
7. For advanced modes, use "Analyze Shapes" or draw custom masks
8. Enable audio options if desired
9. Click "Start Processing" to preview the animation
10. Export as PNG, MP4, GIF, or with synchronized audio


### CLI Mode (Source Version)

**Interactive (Guided):**
1. Run: `python src/imder.py cli` or `CLI.bat` (Windows)
2. Follow the prompts to select media files
3. Choose algorithm, resolution, and audio options
4. Watch real-time progress as processing completes

**Direct (One-liner):**
```bash
# Image to Image with Missform
python src/imder.py flower.png obama.png missform 512

# Video to Video with Missform
python src/imder.py video1.mp4 video2.mp4 missform 256

# Image to Video with target audio
python src/imder.py image.jpg video.mp4 merge 512 target-sound 7

# Video to Image with pixel sound
python src/imder.py video.mp4 image.png shuffle 256 sound
```

### Try It Yourself

Compare the experience yourself:
1. Visit [obamify.com](https://obamify.com/) - note the processing time and limitations
2. Install IMDER: `pip install imder` or download from our [releases page](https://github.com/HAKORADev/IMDER/releases)
3. Experience the difference in speed, quality, and flexibility

---

## Technical Highlights

- **Pure Python** - Easy to read, modify, and contribute to
- **PyQt5 GUI** - Modern, responsive interface with dark theme (Source only)
- **OpenCV & NumPy** - Industry-standard image processing
- **FFmpeg Integration** - Professional-grade video handling
- **Morton Code Ordering** - Efficient spatial pixel mapping
- **K-Means Clustering** - Intelligent shape detection
- **Binary Morphing** - Enhanced shape transition algorithm
- **Canvas Drawing Engine** - New interactive drawing tools
- **QThread Processing** - Non-blocking UI during operations (GUI)
- **Frame-Accurate Video Processing** - Pixel-perfect video transformations

---

## Performance Benchmark

| Image Size | IMDER Time | Obamify Time |
|------------|------------|--------------|
| 128×128 | ~2 seconds | ~30 seconds |
| 512×512 | ~8 seconds | Several minutes |
| 2048×2048 | ~45 seconds | **Hours or crashes** |

*Results may vary based on hardware. IMDER maintains smooth performance across all resolutions. Video processing time depends on frame count and selected resolution.*

---

## Documentation

- **[pip-imder.md](pip-imder.md)** - Full documentation for the Python library (PyPI version)
- **[Algorithms.md](Algorithms.md)** - Detailed explanation of processing algorithms (Source version)
- **[Bots.md](Bots.md)** - Information for AI agents and automated tools
- **[CHANGELOG.md](CHANGELOG.md)** - Version history and changes

---

## License

MIT License - See [LICENSE](LICENSE) for details.

---

## Contributing

IMDER is open-source and welcomes contributions! Whether you want to:
- Add new processing algorithms
- Implement new video effects
- Improve the UI
- Fix bugs
- Add documentation
- Report issues

Your contributions make IMDER better for everyone.

---

## Acknowledgments

Inspired by pixel manipulation techniques, with special thanks to the open-source computer vision community.

**Compare yourself:** [obamify.com](https://obamify.com/) | **Download IMDER:** [Releases](https://github.com/HAKORADev/IMDER/releases) | **PyPI:** `pip install imder`

---
