# Changelog

All notable changes to IMDER - Image Blender will be documented in this file.

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
