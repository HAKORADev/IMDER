# IMDER - Image Blender

<p align="center">
  <img src="src/imder.png" alt="IMDER Logo" width="128" height="128"/>
</p>

**IMDER** is a cutting-edge interactive image blender that creates smooth, mesmerizing animations blending pixels between two images. Unlike existing tools, IMDER delivers **superior quality, blazing-fast performance, and unmatched flexibility** that leaves the competition far behind.

🚀 **Ready-to-use binary for Windows available in [v1.1.1 Release](https://github.com/HAKORADev/IMDER/releases/tag/v1.1.1)** for macOS, and Linux [v1.0.0 Release](https://github.com/HAKORADev/IMDER/releases/tag/v1.0.0) **YOU HAVE TO USE THE SOURCE CODE IF YOU WANT v1.1.1 in Mac or Linux!**

if you are a bot, AI Agent then read [Bots.md](https://github.com/HAKORADev/IMDER/blob/main/Bots.md)
---

## What's New in v1.1.1

### 🆕 Missform Algorithm
Introducing a powerful new morphing algorithm that creates stunning shape transitions:

- **Binary Morphing**: Creates smooth transitions between shapes using pixel position interpolation
- **Shape-Aware Processing**: Uses binary masks to identify and transform distinct shapes
- **Video Support**: Works seamlessly with video processing alongside image processing
- **Enhanced Visual Effects**: Produces unique morphing animations unavailable in other tools

### 🐛 Fusion Algorithm Bugfix
Improved stability and visual quality for the Fusion algorithm:

- **Mask Handling**: Fixed issues with shape masking in Fusion mode
- **Color Blending**: Enhanced color transition smoothness
- **Consistent Output**: More reliable results across different image types
- **Performance Optimization**: Faster processing with better memory management

### 🎬 Enhanced Video Processing
Expanded video support to include the new Missform algorithm for stunning video-to-video transformations:

- **Missform Video Support**: Apply shape morphing between video sequences
- **Extended Algorithm Choices**: Now 3 algorithms available for video processing (Shuffle, Merge, Missform)
- **Better Frame Handling**: Improved synchronization for mixed media types

### 📦 Download the Latest
Get the newest features and improvements:

- **Windows**: [v1.1.1 Release](https://github.com/HAKORADev/IMDER/releases/tag/v1.1.1) (ready-to-use binary)
- **Source Code**: Updated with all new algorithms and bug fixes
- **Cross-Platform**: Available for all major operating systems

---

## What's New in v1.1.0

### 🎬 Video Processing Support (Major Update!)
Transform videos like never before with our all-new video processing engine:

- **Video-to-Video**: Process two videos frame-by-frame for seamless pixel transitions
- **Video-to-Image**: Transform each frame of a video using a static target image
- **Image-to-Video**: Blend a static image into every frame of a video
- **Supported Formats**: MP4, AVI, MOV, MKV, FLV, WMV
- **Frame-Accurate Processing**: Each frame is individually processed for perfect synchronization
- **Auto-Duration Matching**: Automatically matches video lengths for smooth output

### 🔊 Advanced Audio Generation
Create complete multimedia experiences with synchronized soundtracks:

- **Pixel Sound Synthesis**: Generate unique audio based on each frame's pixel colors
- **Target Audio Extraction**: Extract and use audio from target videos
- **Quality Control**: 10 quality levels (10%-100%) for target audio preservation
- **Seamless Integration**: Audio is merged with video output automatically
- **Multiple Options**: Mute, Sound (generated), or Target-Sound (extracted)

### 💻 Enhanced CLI Experience
Power users now have complete control from the command line:

- **Windows Users**: Simply double-click `CLI.bat` for instant CLI access
- **Direct Processing**: `python imder.py <base> <target> [options]`
- **Interactive Mode**: `python imder.py cli` with guided step-by-step prompts
- **Smart Detection**: Automatically detects video vs image inputs
- **Progress Tracking**: Real-time progress bars and detailed processing status
- **Batch-Friendly**: Perfect for scripting and automation

### 📦 Multi-Format Export
Export your creations in any format you need:

- **Frame (PNG)**: Static blended image output
- **Animation (MP4)**: Full 30fps video with optional audio
- **GIF**: Animated with customizable duration
- **Video with Audio**: Merged soundtracks from pixel synthesis or target extraction

---

## Why IMDER Outperforms the Competition

### Speed Comparison: IMDER vs Obamify

While [Obamify](https://obamify.com/) takes **minutes to hours** for high-quality results, IMDER completes the same transformations in **seconds**. See for yourself at [obamify.com](https://obamify.com/) and compare the experience.

| Feature | IMDER | Obamify |
|---------|-------|---------|
| **Processing Speed** | Seconds | Minutes to hours |
| **Maximum Resolution** | 2048×2048 | Limited, slow at high res |
| **Video Processing** | ✅ Full Support | ❌ No |
| **Image Processing** | ✅ **9 Algorithms** | ✅ 1 Algorithm |
| **Audio Generation** | ✅ Pixel + Target | ❌ No |
| **Real-time Preview** | ✅ Yes | ✅ Yes |
| **Shape Selection** | ✅ Yes (auto + manual) | ❌ No |
| **Export Formats** | PNG, MP4, GIF + Audio | GIF only |
| **Dark Theme UI** | ✅ Modern | ❌ Outdated |
| **Cross-Platform** | ✅ All major OS | ✅ All major OS + Web |

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

## Features

### 🎨 **9 Powerful Processing Modes (Images)**

| Mode | Description |
|------|-------------|
| **Shuffle** | Random pixel swapping with brightness balance |
| **Merge** | Grayscale sorting for smooth transitions |
| **Missform** | **NEW** Shape morphing through binary pixel interpolation |
| **Fusion** | Artistic pixel sorting animations (now with bugfixes) |
| **Pattern** | Texture transfer based on color quantization |
| **Disguise** | Shape-aware transformations |
| **Navigate** | Gradient-guided pixel movement |
| **Swap** | Bidirectional pixel exchange |
| **Blend** | Physics-inspired animated transitions |

# More here: (Algorithms)[https://github.com/HAKORADev/IMDER/blob/main/Algorithms.md]

### 🎬 Video Processing Modes

| Mode | Description |
|------|-------------|
| **Shuffle** | Random pixel swapping between video frames |
| **Merge** | Grayscale sorting for smooth frame transitions |
| **Missform** | **NEW** Shape morphing between video sequences |

*Note: Advanced modes (Fusion, Pattern, Disguise, Navigate, Swap, Blend) are available for image processing only.*

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
- **Shape Selection** - Auto-segmentation or manual drawing
- **Multi-segment Selection** - Select multiple distinct regions

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

### 🎯 Advanced Shape Analysis

Unlike Obamify, IMDER allows you to:
- Automatically detect and select distinct regions using k-means clustering
- Manually draw custom masks with the Pen tool
- Combine multiple segments for precise control
- Exclude specific areas from processing

---

## Installation

### Prerequisites
- Python 3.8+
- pip
- [FFmpeg](https://github.com/FFmpeg/FFmpeg) (REQUIRED for video processing with audio - must be installed and added to system PATH)

### Install Dependencies

```bash
pip install -r requirements.txt
```

**Required packages:**
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

### Run from Source

```bash
cd IMDER
python src/imder.py
```

### Command Line Usage

**Windows CLI (easiest):**
```batch
CLI.bat
```

**Interactive Mode:**
```bash
python src/imder.py cli
```

**Direct Processing:**
```bash
python src/imder.py <base_image> <target_image> [algorithm] [resolution]
python src/imder.py <base_video> <target_video> merge 512
python src/imder.py image.jpg video.mp4 shuffle 256 --sound target-sound 8
```

**Sound Options:**
```bash
python imder.py base.png target.png shuffle 512 mute          # No audio
python imder.py base.png target.png shuffle 512 sound         # Generated audio
python imder.py base.mp4 target.mp4 merge 512 target-sound 8  # Target audio (quality 8)
```

---

## Usage Guide

### GUI Mode

1. Launch: `python src/imder.py`
2. Select processing mode from the dropdown (now with 9 options including **Missform**)
3. Choose resolution (start with 256×256 for speed)
4. Click "Add Media" on both panels to load images
5. Optionally, use Rotate/Flip to adjust images
6. For advanced modes, use "Analyze Shapes" or draw custom masks
7. Enable audio options if desired
8. Click "Start Processing" to preview the animation
9. Export as PNG, MP4, GIF, or with synchronized audio

### CLI Mode

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
2. Download IMDER from our [releases page](https://github.com/HAKORADev/IMDER/releases/tag/v1.1.1)
3. Experience the difference in speed, quality, and flexibility

---

## Technical Highlights

- **Pure Python** - Easy to read, modify, and contribute to
- **PyQt5 GUI** - Modern, responsive interface with dark theme
- **OpenCV & NumPy** - Industry-standard image processing
- **FFmpeg Integration** - Professional-grade video handling
- **Morton Code Ordering** - Efficient spatial pixel mapping
- **K-Means Clustering** - Intelligent shape detection
- **Binary Morphing** - New shape transition algorithm
- **QThread Processing** - Non-blocking UI during operations
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

**Compare yourself:** [obamify.com](https://obamify.com/) | **Download IMDER:** [v1.1.1 Release](https://github.com/HAKORADev/IMDER/releases/tag/v1.1.1)

---

## Changelog

### v1.1.1 (Current)
- Added **Missform algorithm** for shape morphing
- Fixed bugs in **Fusion algorithm** for better stability
- Extended video processing to support Missform
- Improved overall performance and memory management

### v1.1.0
- Added video processing support
- Implemented advanced audio generation
- Enhanced CLI experience
- Multi-format export capabilities

### v1.0.0
- Initial release with 8 image processing algorithms
- Real-time preview and shape selection
- Cross-platform GUI interface
