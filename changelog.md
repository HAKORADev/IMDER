# Changelog

All notable changes to IMDER - Image Blender will be documented in this file.

## [v1.1.2] - 202X-01-24

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
