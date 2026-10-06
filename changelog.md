# Changelog

All notable changes to IMDER - Image Blender will be documented in this file.

## [v1.3.0] - 2026-10-06

### Added
- **Reborn Algorithm**: Brand new shape-pair algorithm placed right before Drawer — draw a shape on the base and a shape on the target, and only the pixels inside the drawn base shape move and morph to look like the ones inside the matched target shape. Shapes are chained by draw order (first base shape with first target shape) and extras on either side are ignored
- **Smart Analyze**: The Analyze button gets a drop-down once pen shapes exist — "As-is" keeps the current fill logic and "Smart" refines every drawn shape toward the real object underneath (a rough circle drawn on a car snaps to cover the car more accurately)
- **Multi-Analyze Support**: Include (+) and exclude (-) shapes persist between passes — draw more shapes on top of an existing analysis, exclude shapes eat into include shapes, and every new analyze pass overwrites the previous mask accurately; a Clear Shapes entry lives in the Pen drop-down instead of an auto-reset
- **Video in the GUI**: The GUI now loads videos (MP4/AVI/MOV/MKV/FLV/WMV) and processes them with the same frame-accurate pipeline as the CLI — mode options gray out based on the loaded media (image-image, video-video, image-video, video-image), export choices adapt, and the sound picker unlocks target audio for video targets
- **Streamed Preview Cache**: Preview frames stream to a disk cache with a small memory footprint — a frame-number timeline tracks the run live and can be scrubbed, and Replay plays the cached run back like a normal video (forward or reverse); the cache is cleared on every IMDER open and close
- **Sound Options in the GUI**: Mute / Sound / Target Sound selection with a 1-10 quality picker for target audio, matching the CLI
- **CLI Restored 1:1 with the Real Python CLI (imder.py)**: positional one-liners — `imder <base> <target> [algorithm] [resolution] [sound_option] [quality]` with sound named mute/sound/target-sound and quality 1-10 — plus the full interactive mode (`imder cli`) with the colored block-letter banner, the algorithm/resolution/sound menus, drag & drop prompts, and the "What's Next?" loop; images export Frame (PNG) + GIF + Animation (MP4) into `results/`, videos stream frame by frame with the explore/trim/compile stdout flow of the Python version
- **Live CLI Progress**: The CLI speaks exactly like the Python version — frame counters while exploring videos (`exploring Base frames:`), per-frame percentages (`Frame 12/50 24.0%`), compile progress, the GIF pass, and the `|████----| 42.0%` bar during image exports; diagnostics for the agents stay on stderr
- **App Icon Baked In**: The taskbar/window icon is compiled into the executable through a Qt resource — no imder.png file sits next to the binary anymore
- **Helper Files in the Packages**: The Windows zip ships cli.bat and the Linux zip ships cli.sh plus install.sh (desktop entry and shell alias installer)

### Fixed
- **JPG Files Fail to Read**: Image loading now goes through a layered reader — OpenCV first, an stb_image fallback when the static build's own jpeg codec refuses, and an ffmpeg decode as the last resort — so .jpg/.jpeg inputs read reliably on every platform
- **Drawer Connecting Lines**: Starting a new stroke near the previous one no longer draws a spurious line from the old stroke's end — the canvas only connects from a real pen-down point
- **Dropdown White Edges and Lag**: Popups render on a dark application palette with square frames and no drop-shadow hint — no more white borders at the top and bottom of dropdowns and a snappier feel
- **Audio/Frame Timing**: Video-video runs pick the fps of the side with the shorter total duration (count × fps) exactly like the Python CLI, single-video runs keep the video's own fps, and image-image animations run 300 frames (`fps × 10 seconds` at the default 30fps) like imder.py; pixel-sound synthesizes exactly 1/fps of audio per frame so audio and video durations always line up, and extra frames beyond the shorter input are ignored the same way the Python version does it

### Changed
- **CLI Syntax**: The one-liner is the real `imder.py` positional form — `<base> <target> [algorithm] [resolution] [sound_option] [quality]` with sound named mute/sound/target-sound — replacing the interim flag-based interface
- **Output Naming**: All exports carry the Python names — `image_<timestamp>.png`, `animation_<timestamp>.gif`, `video_<timestamp>.mp4` (plus the `_silent` intermediate while muxing audio) inside `results/`

## [v1.2.5] - 2026-02-01

### Added
- **Custom Resolution Support**: Configure any resolution up to 16384×16384 via the new Custom option in the resolution dropdown
- **Smart Upscaling**: Images now automatically upscale to match targeted resolution, eliminating the previous downscale-only limitation
- **FPS Configuration**: New FPS dropdown (30/60/90/120/240) for video export control, replacing the fixed 30fps limit

### Fixed
- **Drawer Mode Improvements**: Fixed undo/redo history tracking for more reliable drawing operations
- **GUI Refinements**: Updated button labels ("Add Media" → "Add", "Replace Media" → "Replace", etc.) for cleaner interface
- **GUI Layout Fixes**: Improved spacing and alignment in control panels

## [v1.2.0] - 2026-01-31

### Added
- **Drawer Algorithm**: Brand new interactive drawing mode allowing users to create animations from hand-drawn sketches on canvas
- **Advanced Drawing Tools**: Complete drawing toolkit with undo/redo functionality, color picker, and adjustable brush sizes
- **Canvas Drawing Engine**: Real-time drawing canvas that can be transformed into animations with target images
- **Interactive Drawing Interface**: Draw directly in the application and see immediate transformations

*Note: All new features have been thoroughly tested and are production-ready with no major bugs reported.*

## [v1.1.2] - 2026-01-24

### Fixed
- **Missform Algorithm Performance**: Fixed slow processing for image-image transformations to match the speed of other algorithms
- **Video Processing Bug**: Fixed CLI video processing for Missform algorithm that was generating 300 animation frames instead of processing directly, resulting in 300x performance improvement

## [v1.1.1] - 2026-01-24

### Added
- **Missform Algorithm**: New shape morphing algorithm using binary pixel interpolation for stunning shape transitions
- **Extended Video Support**: Missform algorithm now available for video-to-video and video-to-image processing
- **Enhanced Audio Options**: Improved audio generation and extraction capabilities for video processing

### Fixed
- **Fusion Algorithm Bugs**: Fixed issues with shape masking and color blending in Fusion mode
- **Performance Improvements**: Optimized memory management and processing speed across all algorithms
- **Consistency Fixes**: Improved reliability of results across different image types

## [v1.1.0] - 2026-01-23

### Major Features
- **Video Processing Engine**: Complete video support including video-to-video, video-to-image, and image-to-video transformations
- **Advanced Audio Generation**: Pixel sound synthesis and target audio extraction with 10 quality levels (10%-100%)
- **Enhanced CLI Interface**: Comprehensive command-line support with interactive and direct processing modes
- **Multi-Format Export**: Support for PNG, MP4, GIF, and video with synchronized audio

### Technical Improvements
- **Frame-Accurate Processing**: Individual frame processing for perfect video synchronization
- **Auto-Duration Matching**: Automatic video length matching for smooth output
- **Real-time Progress Tracking**: Detailed progress bars and processing status updates
- **Smart Media Detection**: Automatic detection of video vs image inputs

## [v1.0.0] - 2026-01-22

### Initial Release
- **8 Image Processing Algorithms**: Shuffle, Merge, Fusion, Pattern, Disguise, Navigate, Swap, and Blend
- **Real-time Preview**: Live animation preview during processing
- **Shape Selection**: Auto-segmentation using k-means clustering and manual mask drawing
- **Cross-Platform GUI**: Modern PyQt5 interface with dark theme
- **Resolution Options**: Six resolution levels from 128×128 to 2048×2048
- **Image Manipulation Tools**: Rotate, flip, and multi-segment selection
- **Export Formats**: PNG static images and animated GIFs

---
