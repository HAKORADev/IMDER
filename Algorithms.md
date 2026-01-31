# IMDER Algorithms Reference

## Table of Contents
- [Introduction](#introduction)
- [Image Processing Algorithms](#image-processing-algorithms)
  - [Shuffle](#shuffle)
  - [Merge](#merge)
  - [Missform](#missform)
  - [Fusion](#fusion)
  - [Pattern](#pattern)
  - [Disguise](#disguise)
  - [Navigate](#navigate)
  - [Swap](#swap)
  - [Blend](#blend)
  - [Drawer](#drawer)
- [Video Processing](#video-processing)
  - [Why Video Has Limited Algorithms](#why-video-has-limited-algorithms)
  - [Video Processing Design](#video-processing-design)
- [Algorithm Comparison Cheat Sheet](#algorithm-comparison-cheat-sheet)
- [Advanced Techniques](#advanced-techniques)
- [Performance Optimization](#performance-optimization)

## Introduction

IMDER features **10 distinct image processing algorithms** (increased from 9), each designed for specific visual effects and use cases. Unlike tools that offer only one transformation method, IMDER gives you a toolkit of approaches, each with unique characteristics. This document explains how each algorithm works, when to use it, and what makes it special.

**Key Concepts:**
- **Pixel Assignment**: How pixels from the base image map to positions in the target image
- **Temporal Interpolation**: How pixels move over the 300-frame animation
- **Mask Dependence**: Whether the algorithm requires shape selection
- **Visual Style**: The aesthetic result of each transformation

## Image Processing Algorithms

### Shuffle
**What it does:** Randomly swaps pixels between images while respecting brightness levels
**How it works:**
1. Separates pixels into "bright" (>127 grayscale) and "dark" groups for both images
2. Randomly matches bright pixels with bright, dark with dark
3. Any remaining pixels get random assignments
4. Pixels follow straight-line paths during animation

**Why it's like that:**
Shuffle maintains overall brightness distribution while creating a chaotic, energetic transformation. The brightness separation prevents dark areas from ending up in bright zones (and vice versa), preserving some structural integrity.

**Best for:**
- Abstract, energetic transformations
- When you want to preserve overall brightness balance
- Quick, random-looking effects

**Example:**
```python
# Base: Portrait photo
# Target: Cityscape
# Result: Portrait pixels scatter into cityscape positions
```

**Visual Characteristics:**
- Chaotic but controlled
- Maintains overall brightness
- Medium visual coherence
- Good for artistic abstraction
- can be used for video processing to give CRT-like noise effect in low resolutions

---

### Merge
**What it does:** Sorts pixels by brightness and matches them across images
**How it works:**
1. Converts both images to grayscale
2. Sorts all pixels by brightness (darkest to brightest)
3. Maps the darkest base pixel to darkest target pixel, etc.
4. Creates smooth brightness-based transitions

**Why it's like that:**
Merge creates the smoothest possible transition by matching brightness levels. This produces gradual, elegant transformations that feel natural and organic.

**Best for:**
- Smooth, elegant transitions
- When subject matter differs greatly
- Professional-looking transformations
- Video processing (along with Shuffle and Missform)

**Example:**
```python
# Base: Book cover
# Target: Human face
# Result: Dark book areas flow to dark face areas, creating organic morph
```

**Visual Characteristics:**
- Smooth, gradual transitions
- High visual coherence
- Organic feeling
- Excellent for dramatic before/after reveals

---

### Missform
**What it does:** Morphs between binary shape masks of images
**How it works:**
1. Creates binary masks (black/white) from both images using thresholding
2. Extracts pixel positions from white areas of each mask
3. Pairs corresponding pixels between masks
4. Interpolates positions over time, moving base pixels along the path

**Why it's like that:**
Missform focuses on shape transformation rather than color blending. By working with binary masks, it creates clear shape morphing effects that emphasize form over texture.

**Best for:**
- Shape-based transformations
- Logo morphing
- When forms are more important than textures
- Video processing (unique among advanced algorithms)

**Pro Tips:**
- use bright base to see better results
- brighter base = more pixels to use = more details in the result = more computing
- in video processing, use it with bright base image and video target, you will see unique results!

**Example:**
```python
# Base: Square logo
# Target: Circle logo
# Result: Square smoothly morphs into circle shape
```

**Visual Characteristics:**
- Clear shape transformation
- Less focus on color/texture
- Geometric precision
- Excellent for logo animations

---

### Fusion
**What it does:** Blends colors while optionally preserving unmasked areas
**How it works:**
1. With mask: Only transforms pixels within selected shapes
2. Without mask: Transforms entire image
3. Colors interpolate from base to target values
4. Optionally shuffles base colors within mask for artistic effect

**Why it's like that:**
Fusion is designed for selective transformation. The mask allows you to transform only specific areas while leaving the rest of the base image intact. Without a mask, it provides simple color blending.

**Best for:**
- Selective transformations
- When you want to preserve background
- Artistic color effects
- Partial image morphing

**Example:**
```python
# Base: Person in landscape
# Target: Abstract painting
# Mask: Select only the person
# Result: Person transforms to painting style, landscape remains
```

**Visual Characteristics:**
- Selective transformation
- Background preservation
- Color interpolation focus
- Artistic, controlled effects

---

### Pattern
**What it does:** Transfers texture patterns based on color quantization
**How it works:**
1. Quantizes target image colors to limited palette
2. Sorts base pixels by brightness
3. Maps sorted base pixels to sorted target quantized colors
4. Creates pattern-like texture transfer

**Why it's like that:**
Pattern focuses on texture and pattern transfer rather than exact color matching. The quantization creates distinct color bands that produce a stylized, patterned look.

**Best for:**
- Texture transfer
- Stylized effects
- Creating patterned looks
- When you want artistic rather than realistic results

**Example:**
```python
# Base: Smooth gradient
# Target: Textured fabric
# Result: Gradient takes on fabric-like pattern
```

**Visual Characteristics:**
- Pattern/texture emphasis
- Color banding effects
- Stylized appearance
- Good for artistic filters

**Pro Tip:**
- use pen to include a specific pixels from target image
- use it in high-res images so you can get more smoothy result!

---

### Disguise
**What it does:** Reorders pixels within selected shapes based on brightness
**How it does:**
1. Only works within selected mask areas
2. Sorts both base and target pixels within mask by brightness
3. Maps brightest base pixel to brightest target pixel, etc.
4. Preserves pixel colors but repositions them

**Why it's like that:**
Disguise creates the illusion that shapes are transforming while actually just rearranging existing pixels. This maintains color palette integrity while changing arrangement.

**Best for:**
- Shape transformation with color preservation
- When you want to keep original colors
- Subtle, clever transformations
- Puzzle-like effects

**Example:**
```python
# Base: Red apple
# Target: Green pear (shape)
# Mask: Select pear shape
# Result: Apple pixels rearrange into pear shape
```

**Visual Characteristics:**
- Color palette preservation
- Shape-focused transformation
- Puzzle-like rearrangement
- Clever, subtle effects

**Pro Tip:**
- it remap the base pixels in the same area that selected in target shape
- using empty base image or single or two colors base image with detailed target shape would give the best result for this algorithm!

---

### Navigate
**What it does:** Moves pixels along Morton (Z-order) curve paths
**How it works:**
1. Converts pixel colors to Morton codes (3D space-filling curve)
2. Sorts pixels by Morton code in both images
3. Maps sorted pixels between images
4. Creates winding, organic movement paths

**Why it's like that:**
Navigate creates complex, winding movement paths that feel organic and natural. The Morton curve ensures spatially nearby pixels in 3D color space travel together.

**Best for:**
- Organic, flowing transformations
- When you want complex movement paths
- Artistic, abstract effects
- Natural-feeling animations

**Example:**
```python
# Base: Forest scene
# Target: Ocean waves
# Result: Forest pixels flow in winding paths to wave positions
```

**Visual Characteristics:**
- Complex movement paths
- Organic flow
- Spatially coherent color movement
- Artistic, abstract appeal

**Pro Tip:**
- it works like disguise, but uses best pixels from the whole image to the masked area from target in base
- also less detailed base with detailed target mask will give the best results, still can be used with whatever images you set

---

### Swap
**What it does:** Bidirectional pixel exchange between images
**How it works:**
1. Only transforms pixels within mask
2. Finds best color matches between base and target within mask
3. Creates two-way assignments (pixel A goes to B, B goes to A)
4. Results in pixel exchange rather than one-way movement

**Why it's like that:**
Swap creates the illusion that pixels are trading places. This bidirectional mapping creates a balanced, symmetrical transformation that feels like a true exchange.

**Best for:**
- Balanced transformations
- When both images should contribute equally
- Symmetrical effects
- Fair trade visualizations

**Example:**
```python
# Base: Day sky
# Target: Night sky
# Mask: Select sky area
# Result: Day and night pixels swap places
```

**Visual Characteristics:**
- Bidirectional movement
- Balanced transformation
- Exchange illusion
- Symmetrical appeal

---

### Blend
**What it does:** Physics-inspired movement with gradient guidance
**How it works:**
1. Uses Sobel edge detection to find target image gradients
2. Pixels are attracted to target positions but also follow gradients
3. Implements "home force" (pull toward original position)
4. Creates swirling, fluid-like motion

**Why it's like that:**
Blend simulates physical movement with forces and attractions. The gradient following creates swirling patterns that follow image contours, producing fluid, dynamic animations.

**Best for:**
- Fluid, dynamic animations
- When you want motion to follow image features
- Artistic, painterly effects
- Complex, multi-stage movement

**Example:**
```python
# Base: Paint splatter
# Target: Human silhouette
# Result: Paint flows along silhouette edges into final form
```

**Visual Characteristics:**
- Fluid, swirling motion
- Edge-following movement
- Dynamic, energetic animation
- Artistic, painterly style

---

### Drawer
**What it does:** Transforms hand-drawn sketches into detailed target images while preserving artistic intent
**How it works:**
1. **Canvas System**: Operates on a 1024×1024 canvas with real-time drawing tools
2. **Pixel Classification**: Automatically distinguishes drawn pixels (non-white) from canvas background (white, RGB ≥ 245)
3. **Intelligent Distribution**: Assigns drawn pixels to target image positions based on brightness matching
4. **Density Optimization**: Evenly distributes drawn pixels across the target to prevent clustering
5. **Color-Aware Processing**: Considers both RGB values and brightness levels for optimal matching

**Technical Implementation:**
- **Background Detection**: Any pixel with RGB values all above 245 is treated as "white background" and ignored
- **Brightness Sorting**: Drawn pixels are sorted by average brightness (R+G+B)/3
- **Target Analysis**: All target pixels are sorted by brightness
- **Adaptive Matching**: For N drawn pixels and M target pixels:
  - If N ≤ M: Each drawn pixel gets matched to a unique target position
  - If N > M: Drawn pixels are distributed to avoid repetition, ensuring all drawn pixels contribute

**Why it's like that:**
Drawer bridges the gap between freehand drawing and digital transformation. It respects the artist's stroke density, color choices, and composition while intelligently mapping them to complex target images. The white-background preservation allows for clean sketches that transform elegantly.

**Best for:**
- Converting sketches to photorealistic images
- Artistic style transformations
- Creative brainstorming and visualization
- Educational demonstrations of drawing-to-reality concepts
- Creating unique animations from simple drawings

**Example:**
```python
# Base: Hand-drawn smiley face
# Target: Detailed portrait photograph
# Result: Simple drawing transforms into detailed portrait while maintaining facial structure
```

**Visual Characteristics:**
- Maintains drawing composition and density
- Elegant transformation from simple to complex
- Respects color choices from the drawing
- Creates surprising yet coherent results
- Excellent for artistic explorations

**Technical Details:**
- **Canvas Size**: Fixed at 1024×1024 pixels for optimal performance and clarity
- **Pen Tools**: Adjustable size (1-50 pixels), custom color selection, undo/redo support
- **Background Handling**: Pure white (RGB ≥ 245) is treated as transparent/ignored
- **Pixel Matching**: Uses advanced distribution algorithms to prevent pixel clustering

**Creative Tips & Tricks:**

1. **Color Palette Strategy**:
   - Use varied colors in your drawing even for monochrome sketches - different RGB values create more matching opportunities
   - For "invisible white" drawing, use colors like (254,254,254) instead of (255,255,255) - visually white but algorithmically active

2. **Density Matters**:
   - More drawn pixels = richer detail in final result
   - Sparsely drawn areas will create elegant, minimalist transformations
   - Dense drawings with many strokes yield highly detailed results

3. **Hybrid Workflows**:
   - **Draw → Transform**: Create in Drawer mode, then switch to any other mode with your drawing as base
   - **Import → Modify**: Load an image as base, switch to Drawer to add annotations or modifications
   - **Reverse Magic**: Draw something simple, add complex target, then use "Reverse Swap" to make the drawing become the target

4. **Advanced Techniques**:
   - **Layered Drawing**: Start with light colors, add darker details for hierarchical transformation
   - **Negative Space**: Leave areas undrawn (white) to let the target image "fill in the gaps"
   - **Color Coding**: Use specific colors to target different areas (e.g., blue for sky, green for grass)

5. **Algorithm Intelligence**:
   - The system automatically prevents pixel clustering - even if you draw a single dot, it will distribute intelligently
   - Color variations are leveraged to match complex target patterns
   - Brightness-based matching ensures logical transformations (dark strokes go to dark areas, etc.)

6. **Canvas Optimization**:
   - Use the full 1024×1024 space for maximum detail
   - Thicker pens (size 20-30) for broad strokes, thinner pens (1-5) for fine details
   - The undo/redo system allows for experimental drawing without risk

**Pro Tips:**
1. **For best results**: Draw with intentional strokes rather than random scribbles
2. **Color experimentation**: Even slight color variations (e.g., different shades of gray) create more interesting transformations
3. **Save iterations**: Export your drawing at different stages to use as bases for different algorithms
4. **Combine with other algorithms**: A drawing created in Drawer mode can be used as base for Pattern, Disguise, or Navigate algorithms for unique layered effects
5. **Educational use**: Perfect for demonstrating how simple shapes can transform into complex images
6. **Multi Use**: One of the best things can be done with this mode is your drawings became base image when navigating to other modes

**Technical Note**: Drawer mode includes a complete drawing engine with layer support (drawing over base images), adjustable tools, and real-time preview. The algorithm is optimized to handle everything from minimalist sketches to detailed drawings while ensuring visually pleasing transformations regardless of drawing complexity.

---

## Video Processing

### Why Video Has Limited Algorithms

IMDER currently supports only **Shuffle**, **Merge**, and **Missform** algorithms for video processing. Here's why:

**1. Mask/Practicality Limitation:**
Advanced algorithms like Pattern, Disguise, Navigate, Swap, and Blend require shape masks. For video:
- **Frame-by-frame masking would be impractical** - Users would need to mask hundreds or thousands of frames
- **Auto-masking inconsistencies** - Automatic shape detection would vary between frames, causing jitter
- **Performance concerns** - Per-frame shape analysis would drastically slow processing

**2. Fusion Algorithm Limitation:**
Fusion isn't included for video because:
- **Transformation goal mismatch** - Fusion converts base pixels to target pixels over time
- For video, each frame already represents the target at that moment
- The animation would just show pixels arriving at their already-visible destinations
- **Useless in practice** - No meaningful visual effect for video sequences

**3. Drawer Algorithm Limitation:**
Drawer isn't included for video because:
- **Interactive requirement** - Drawing is inherently an interactive, real-time process
- **Temporal inconsistency** - Maintaining consistent drawings across hundreds of frames is impractical
- **Performance impact** - Real-time drawing processing for each video frame would be computationally prohibitive

**4. Performance Considerations:**
- Video already processes hundreds to thousands of frames
- Complex per-frame algorithms would multiply processing time exponentially
- **Shuffle, Merge, and Missform** provide a good balance of visual variety and performance

### Video Processing Design

**Why Video is CLI-Only in IMDER:**

The GUI's primary purpose is **real-time animation visualization** - watching the 300-frame transformation sequence. For video:

**Mathematical Reality:**
- A 10-second video at 30 FPS = 300 frames
- If each frame animated with 300 transformation frames → 300 × 300 = 90,000 frames
- At 30 FPS display → 90,000 / 30 = 3,000 seconds = **50 minutes of animation**
- This serves no practical purpose since video frames represent the temporal sequence already

**Practical Design Choice:**
- Video processing skips the intermediate animation
- Calculates each frame's final result directly
- CLI provides progress feedback without unnecessary visualization
- Results are exported as video files for playback

**Not a Technical Limitation:**
The GUI *could* animate video transformations frame-by-frame, but it would:
- Waste enormous processing time
- Create impractically long animations
- Provide no additional value over the final video

## Algorithm Comparison Cheat Sheet

| Algorithm | Mask Required | Best For | Motion Style | Color Handling | Speed | Video Support |
|-----------|---------------|----------|--------------|----------------|-------|---------------|
| **Shuffle** | No | Abstract, random | Straight lines | Brightness groups | Fast | ✅ Yes |
| **Merge** | No | Smooth transitions | Direct paths | Brightness sort | Fast | ✅ Yes |
| **Missform** | No | Shape morphing | Shape paths | Binary mask focus | Medium | ✅ Yes |
| **Fusion** | Optional | Selective transform | Direct paths | Color blending | Medium | ❌ No* |
| **Pattern** | Yes | Texture transfer | Direct paths | Quantized colors | Fast | ❌ No |
| **Disguise** | Yes | Color-preserving | Direct paths | Original colors | Fast | ❌ No |
| **Navigate** | Yes | Organic flow | Curved paths | Spatial sorting | Slow | ❌ No |
| **Swap** | Yes | Balanced exchange | Bidirectional | Best match colors | Slow | ❌ No |
| **Blend** | Yes | Fluid dynamics | Swirling paths | Gradient guided | Medium | ❌ No |
| **Drawer** | N/A (Canvas) | Sketch to reality | Direct paths | Drawing colors | Medium | ❌ No |

*Fusion without mask works technically but not included due to transformation logic mismatch

## Advanced Techniques

### Algorithm Combinations
1. **Multi-stage transformations**: Export frame from one algorithm, use as base for another
2. **Drawer Hybrid Workflow**: Create base in Drawer mode, then process with Pattern/Disguise/Navigate for specialized effects
3. **Resolution progression**: Start low-res for preview, final export at high-res
4. **Selective masking**: Use different masks for different algorithm runs on same image pair

### Drawer-Specific Techniques
1. **Color Layering**: Start with light background colors, add progressively darker details
2. **Selective Detail**: Draw detailed areas where you want transformation focus, leave other areas sparse
3. **Import & Annotate**: Load any image as base, switch to Drawer to add annotations or modifications
4. **Reverse Transformation**: Draw simple shapes, add complex target, use "Reverse Swap" in other modes
5. **Invisible White**: Use (254,254,254) instead of pure white for "hidden" strokes that still affect transformation

### Performance Optimization
1. **Preview at 128×128** - Test algorithms quickly before final render
2. **Use appropriate resolution** - 512×512 is often optimal for quality/speed balance
3. **Close other applications** - IMDER uses significant CPU resources (use one core 100%)
4. **Batch processing** - Use CLI for multiple transformations overnight
5. **Drawer Efficiency**: For complex drawings, use thicker pens for broad areas, save fine details for final passes

### Creative Applications
1. **Logo animations** - Missform for shape, Merge for color transitions
2. **Art style transfer** - Pattern algorithm for texture, Blend for painterly effects
3. **Educational visualizations** - Swap for showing exchanges, Navigate for flow demonstrations
4. **Abstract art generation** - Shuffle for randomness, Fusion for controlled chaos
5. **Sketch-to-Reality** - Drawer for creative exploration, converting ideas to images
6. **Hybrid Art**: Combine hand-drawn elements (Drawer) with photographic elements (other modes)

## Troubleshooting Guide

### Common Issues and Solutions

**Algorithm produces poor results:**
- Try different algorithm - each works better with certain image types
- Adjust image contrast/preprocessing before importing
- Try different mask selections for mask-dependent algorithms
- For Drawer: Ensure you're not using pure white (255,255,255) for active drawing

**Processing is too slow:**
- Reduce resolution (512×512 is often sufficient)
- Close other resource-intensive applications
- Use simpler algorithms (Shuffle/Merge are fastest)
- For Drawer: Reduce canvas drawing complexity

**Mask not working as expected:**
- Use Pen tool for precise manual masks
- Combine multiple segments for complex shapes
- For Disguise/Swap, ensure mask covers areas with sufficient pixel variety

**Drawer-specific issues:**
- **Too few pixels drawn**: The algorithm intelligently distributes pixels, but very sparse drawings may create minimalist results
- **Pure white strokes**: Use near-white (254,254,254) instead of pure white (255,255,255)
- **Brush responsiveness**: Adjust pen size for different detail levels
- **Canvas refresh issues**: Use the "Reset Canvas" option if drawing display becomes inconsistent

**Video processing issues:**
- Ensure FFmpeg is installed and in PATH
- Check video codec compatibility (MP4/H.264 recommended)
- Reduce resolution for faster video processing

---

## Conclusion

IMDER's **10 algorithms** provide a comprehensive toolkit for image transformation, each with unique characteristics and best-use scenarios. Understanding these algorithms allows you to:
- Choose the right tool for your creative vision
- Combine techniques for complex effects
- Optimize processing for your hardware
- Create professional-quality transformations

The addition of **Drawer mode** opens entirely new creative possibilities, allowing hand-drawn sketches to transform into detailed images. This bridges the gap between traditional drawing and digital transformation, making IMDER not just a tool for existing images, but also for creating new ones.

Remember that **Shuffle**, **Merge**, and **Missform** are your go-to choices for video, while the full palette is available for still image transformations. Each algorithm opens different creative possibilities - experiment to discover which ones best suit your projects.

**Pro Tips:**
- Keep an "algorithm test" folder where you run the same image pair through all 10 algorithms to build your intuition about each one's visual signature.
- Use same media as base and target at the same time to apply effects and to see some magical stuff!
- **Drawer Exploration**: Try drawing the same simple shape and testing it with wildly different target images to understand the transformation patterns.
- **Hybrid Creativity**: Don't limit yourself to one mode - the most interesting results often come from combining multiple algorithms in sequence.
