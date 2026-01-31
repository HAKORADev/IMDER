# IMDER PyPI Library Documentation

**The lightweight, automation-friendly pixel blender for Python.**

🔗 **PyPI Package:** [imder](https://pypi.org/project/imder/)  
📦 **Install:** `pip install imder`  
🐍 **Python:** 3.8+ required  
📖 **Source Code:** [HAKORADev/IMDER](https://github.com/HAKORADev/IMDER) (GUI/CLI version)  
🔄 **Version:** 1.0.0+ (API stable)

---

## Why This Library Exists

While the [main IMDER repository](https://github.com/HAKORADev/IMDER) provides a full-featured PyQt5 GUI application, this PyPI distribution strips away the interface to give you **pure processing power** that integrates anywhere. No window management, no event loops, no GUI dependencies—just the core pixel-sorting algorithms accessible via Python API or CLI.

**Use this when:**
- Building automated pipelines or batch processors
- Integrating image blending into web apps or Discord bots
- Creating your own UI wrapper (React, Tkinter, web interface)
- Running on headless servers without display capabilities
- Using as an AI agent skill or automation tool
- You simply want to `pip install` and run commands

---

## What IMDER Does

IMDER is a **pixel-morphing engine** that creates smooth transitions between two images or videos by intelligently reassigning pixels based on their brightness, color, and position. Unlike simple crossfades, IMDER algorithms preserve structural relationships between pixels, creating organic, physics-inspired transformations.

**Key Features:**
- 🎨 **4 Core Algorithms:** Shuffle, Merge, Missform, Fusion
- 📁 **Multi-Format Export:** PNG, GIF, MP4 in one operation
- 🎵 **Audio Integration:** Mute, generated audio, or target video audio
- 🎬 **Video Support:** Blend videos, videos+images, or images+videos
- ⚡ **Headless Operation:** No GUI dependencies, pure computation
- 🔧 **Automation Ready:** Simple API for scripting and integration

---

## Installation & Dependencies

### Basic Installation
```bash
pip install imder
```

### System Dependencies

**FFmpeg (Optional but Recommended)**
- **Why:** Required only for video processing with audio extraction (`sound='target'`)
- **Without it:** Video processing works fine, but audio features are auto-disabled
- **Install:**
  ```bash
  # Windows
  winget install FFmpeg
  
  # macOS
  brew install ffmpeg
  
  # Linux (Ubuntu/Debian)
  sudo apt install ffmpeg
  
  # Linux (Fedora/RHEL)
  sudo dnf install ffmpeg
  ```

**Python Dependencies (Auto-installed):**
- `opencv-python` (>=4.5.0) - Core image/video processing
- `numpy` (>=1.21.0) - High-performance array operations
- `Pillow` (>=9.0.0) - GIF export and image I/O
- `pyfiglet` (>=0.8.0) - CLI banner display

**No GUI Dependencies:** Unlike the source version, this doesn't require PyQt5, making it ideal for server environments.

---

## Core API Reference

### Primary Function

#### `imder.process(base, target, result, results, algo, res, sound, sq=None, sq_hz=None)`

The main processing function that blends two media files. Returns a list of output file paths.

**Parameters:**

| Parameter | Type | Default | Description | Valid Values |
|-----------|------|---------|-------------|--------------|
| `base` | str | **Required** | Path to base image/video | Any valid image (jpg, png, etc.) or video (mp4, avi, etc.) path |
| `target` | str | **Required** | Path to target image/video | Same as base |
| `result` | str | **Required** | Output directory path | Any writable directory (created if not exists) |
| `results` | list | **Required** | List of formats to export | `['png']`, `['gif']`, `['mp4']`, or combinations like `['gif', 'mp4']` |
| `algo` | str | "merge" | Processing algorithm | `'shuffle'`, `'merge'`, `'missform'`, `'fusion'` |
| `res` | int | 512 | Processing resolution (square) | Any integer 1-16384, typical: 128, 256, 512, 1024, 2048 |
| `sound` | str | "mute" | Audio option | `'mute'` (no audio), `'gen'` (generate), `'target'` (extract from target) |
| `sq` | int | None | Sound quality (1-10) | Only when `sound='target'`; 1=lowest, 10=highest |
| `sq_hz` | int | None | Sample rate in Hz | Only when `sound='target'`; 8000-192000; alternative to `sq` |

**Returns:** `List[str]` - Absolute paths to all generated files, in order of `results` list.

**Algorithm Availability:**
- **Images (all algorithms):** `shuffle`, `merge`, `missform`, `fusion`
- **Videos (limited):** `shuffle`, `merge`, `missform` (fusion not supported for video)

**Sound Options:**
- `'mute'` - No audio in output videos
- `'gen'` - Generate synthetic audio from pixel values (unique per frame)
- `'target'` - Extract audio from target video (requires FFmpeg, target must be video)

**Audio Quality Parameters (mutually exclusive):**
- `sq` (1-10): Maps to bitrates (10 = 32k, 20 = 64k, ..., 100 = copy original)
- `sq_hz` (8000-192000): Direct sample rate control

### Interactive Function

#### `imder.launch_interactive()`

Launches the interactive CLI wizard (same as running `imder` in terminal). Guides through all options step-by-step with validation.

---

## Algorithm Details

### 🎲 **Shuffle**
Randomly reassigns pixels between images based on brightness groups. Dark pixels from base map to dark pixels in target, light to light. Creates a chaotic but structured transformation.

**Best for:** Abstract art, texture blending, chaotic transformations.

### 🔄 **Merge**
Sorts pixels by brightness and reassigns in order. Creates smooth, gradient-like transitions where brightness flows naturally from source to target.

**Best for:** Natural transitions, gradient effects, mood shifts.

### 🌀 **Missform**
Uses binary thresholding to identify "foreground" pixels, then morphs their positions. Creates particle-like effects where elements appear to dissolve and reform.

**Best for:** Particle effects, object morphing, sci-fi transformations.

### ✨ **Fusion** (Images Only)
Combines color interpolation with positional reassignment. Pixels move while also changing color smoothly. Creates the most complex and organic transformations.

**Best for:** High-quality image blends, artistic effects, smooth color transitions.

---

## Usage Examples

### Basic Image Blending (PNG Export)
```python
import imder

# Simple single image export
files = imder.process(
    base="portrait.jpg",
    target="landscape.jpg",
    result="./output",
    results=["png"],
    algo="shuffle",
    res=512,
    sound="mute"
)
print(f"Saved to: {files[0]}")
```

### Animation Export (GIF + MP4)
```python
# Export both GIF and video with generated audio
files = imder.process(
    base="input1.png",
    target="input2.png",
    result="./animations",
    results=["gif", "mp4"],  # Creates both formats
    algo="missform",
    res=1024,
    sound="gen"  # Add generated audio to MP4
)

for f in files:
    print(f"Created: {f}")
```

### Video Processing with Target Audio
```python
# Blend two videos, keeping target audio at custom quality
files = imder.process(
    base="clip1.mp4",
    target="clip2.mp4",
    result="./blended_videos",
    results=["mp4"],
    algo="merge",
    res=512,
    sound="target",
    sq=5  # 50% audio quality (bitrate mapping)
)

# Alternative: specify exact sample rate
files = imder.process(
    base="clip1.mp4",
    target="clip2.mp4",
    result="./blended_videos",
    results=["mp4"],
    algo="merge",
    res=512,
    sound="target",
    sq_hz=44100  # CD quality sample rate
)
```

### Image + Video Blending
```python
# Use image as base, video as target
files = imder.process(
    base="background.png",
    target="animation.mp4",
    result="./mixed_media",
    results=["mp4"],
    algo="shuffle",
    res=256,
    sound="target"  # Use audio from video
)
```

### Batch Processing Multiple Files
```python
import os
import imder

base_image = "pattern.jpg"
targets = ["cat.jpg", "dog.jpg", "bird.jpg"]

for i, target in enumerate(targets, 1):
    print(f"Processing {i}/{len(targets)}: {target}")
    
    files = imder.process(
        base=base_image,
        target=target,
        result=f"./batch_results/set_{i}",
        results=["png", "gif"],
        algo="shuffle",
        res=256,
        sound="mute"
    )
    
    print(f"  Created: {', '.join(os.path.basename(f) for f in files)}")
```

---

## Command Line Usage

After installation, the `imder` command is available globally.

### Interactive Mode (Wizard)
```bash
imder
```
Launches a step-by-step wizard prompting for all parameters with validation.

### Direct Command Execution
```bash
# Basic syntax
imder <base> <target> <result_folder> --results <formats> --algo <name> --res <int> --sound <type>

# Examples

# Image to image, multiple formats
imder photo.jpg art.jpg ./output --results png gif --algo shuffle --res 512

# Video to video with audio extraction
imder video1.mp4 video2.mp4 ./out --results mp4 --algo missform --res 256 --sound target --sq 8

# Image to video with high-quality audio
imder image.png target.mp4 ./mixed --results mp4 --algo merge --res 1024 --sound target --sq_hz 48000

# Quick single format
imder a.jpg b.jpg ./quick --results gif --algo fusion --res 128
```

### Command Line Options
| Option | Description | Required |
|--------|-------------|----------|
| `base` | Base media file path | Yes |
| `target` | Target media file path | Yes |
| `result` | Output directory | Yes |
| `--results` | Output formats (space-separated) | Yes |
| `--algo` | Algorithm name | No (default: merge) |
| `--res` | Resolution (integer) | No (default: 512) |
| `--sound` | Audio option | No (default: mute) |
| `--sq` | Sound quality 1-10 | No |
| `--sq_hz` | Sample rate 8000-192000 | No |

---

## Advanced Integration Examples

### 1. Build Your Own GUI Wrapper

**Tkinter Example:**
```python
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import imder
import threading
import os

class ImderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("IMDER Custom GUI")
        self.root.geometry("500x400")
        
        # Variables
        self.base_path = tk.StringVar()
        self.target_path = tk.StringVar()
        self.output_dir = tk.StringVar(value="./output")
        self.algorithm = tk.StringVar(value="shuffle")
        self.resolution = tk.IntVar(value=512)
        self.formats = tk.StringVar(value="gif")
        self.sound_option = tk.StringVar(value="mute")
        
        # UI Setup
        self.create_widgets()
    
    def create_widgets(self):
        # File selection
        tk.Label(self.root, text="Base Image/Video:").grid(row=0, column=0, sticky="w", padx=10, pady=5)
        tk.Entry(self.root, textvariable=self.base_path, width=40).grid(row=0, column=1, padx=5)
        tk.Button(self.root, text="Browse", command=self.browse_base).grid(row=0, column=2)
        
        tk.Label(self.root, text="Target Image/Video:").grid(row=1, column=0, sticky="w", padx=10, pady=5)
        tk.Entry(self.root, textvariable=self.target_path, width=40).grid(row=1, column=1, padx=5)
        tk.Button(self.root, text="Browse", command=self.browse_target).grid(row=1, column=2)
        
        # Algorithm selection
        tk.Label(self.root, text="Algorithm:").grid(row=2, column=0, sticky="w", padx=10, pady=5)
        algo_combo = ttk.Combobox(self.root, textvariable=self.algorithm, 
                                 values=["shuffle", "merge", "missform", "fusion"])
        algo_combo.grid(row=2, column=1, sticky="w", padx=5)
        
        # Resolution slider
        tk.Label(self.root, text="Resolution:").grid(row=3, column=0, sticky="w", padx=10, pady=5)
        tk.Scale(self.root, from_=128, to=2048, orient=tk.HORIZONTAL, 
                variable=self.resolution, length=200).grid(row=3, column=1, sticky="w", padx=5)
        
        # Format selection
        tk.Label(self.root, text="Output Formats:").grid(row=4, column=0, sticky="w", padx=10, pady=5)
        format_frame = tk.Frame(self.root)
        format_frame.grid(row=4, column=1, sticky="w", padx=5)
        
        for fmt in ["png", "gif", "mp4"]:
            tk.Checkbutton(format_frame, text=fmt, 
                          variable=tk.BooleanVar(value=fmt in self.formats.get()),
                          command=lambda f=fmt: self.toggle_format(f)).pack(side=tk.LEFT, padx=2)
        
        # Process button
        tk.Button(self.root, text="Process", command=self.process, 
                 bg="green", fg="white", padx=20).grid(row=5, column=1, pady=20)
        
        # Status label
        self.status = tk.Label(self.root, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status.grid(row=6, column=0, columnspan=3, sticky="ew", padx=10, pady=5)
    
    def browse_base(self):
        filename = filedialog.askopenfilename(
            title="Select Base File",
            filetypes=[("All files", "*.*"), ("Images", "*.jpg *.png *.jpeg"), 
                      ("Videos", "*.mp4 *.avi *.mov")]
        )
        if filename:
            self.base_path.set(filename)
    
    def browse_target(self):
        filename = filedialog.askopenfilename(
            title="Select Target File",
            filetypes=[("All files", "*.*"), ("Images", "*.jpg *.png *.jpeg"), 
                      ("Videos", "*.mp4 *.avi *.mov")]
        )
        if filename:
            self.target_path.set(filename)
    
    def toggle_format(self, fmt):
        # Toggle format in formats string
        current = self.formats.get().split()
        if fmt in current:
            current.remove(fmt)
        else:
            current.append(fmt)
        self.formats.set(" ".join(current))
    
    def process(self):
        if not self.base_path.get() or not self.target_path.get():
            messagebox.showerror("Error", "Please select both base and target files")
            return
        
        formats = self.formats.get().split()
        if not formats:
            messagebox.showerror("Error", "Select at least one output format")
            return
        
        self.status.config(text="Processing...")
        
        def run_processing():
            try:
                files = imder.process(
                    base=self.base_path.get(),
                    target=self.target_path.get(),
                    result=self.output_dir.get(),
                    results=formats,
                    algo=self.algorithm.get(),
                    res=self.resolution.get(),
                    sound=self.sound_option.get()
                )
                
                # Update UI in main thread
                self.root.after(0, lambda: self.status.config(
                    text=f"Created {len(files)} file(s): " + ", ".join(os.path.basename(f) for f in files)
                ))
                
                messagebox.showinfo("Success", f"Processing complete!\nCreated {len(files)} file(s).")
                
            except Exception as e:
                self.root.after(0, lambda: self.status.config(text=f"Error: {str(e)}"))
                messagebox.showerror("Error", str(e))
        
        # Run in background thread
        threading.Thread(target=run_processing, daemon=True).start()

if __name__ == "__main__":
    root = tk.Tk()
    app = ImderApp(root)
    root.mainloop()
```

**Web Interface (Flask API):**
```python
from flask import Flask, request, jsonify, send_file
import imder
import os
import uuid

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB limit

@app.route('/api/blend', methods=['POST'])
def blend_api():
    """REST API endpoint for blending images/videos"""
    try:
        # Get parameters
        data = request.json
        base_path = data.get('base_path')
        target_path = data.get('target_path')
        
        if not base_path or not target_path:
            return jsonify({"error": "Missing base_path or target_path"}), 400
        
        # Generate unique output directory
        output_dir = os.path.join(app.config['UPLOAD_FOLDER'], str(uuid.uuid4()))
        os.makedirs(output_dir, exist_ok=True)
        
        # Process
        files = imder.process(
            base=base_path,
            target=target_path,
            result=output_dir,
            results=data.get('formats', ['png']),
            algo=data.get('algo', 'shuffle'),
            res=data.get('res', 512),
            sound=data.get('sound', 'mute'),
            sq=data.get('sq'),
            sq_hz=data.get('sq_hz')
        )
        
        # Return file URLs
        file_urls = [f"/api/download/{os.path.basename(output_dir)}/{os.path.basename(f)}" 
                    for f in files]
        
        return jsonify({
            "success": True,
            "files": files,
            "urls": file_urls,
            "output_dir": os.path.basename(output_dir)
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/download/<dir_name>/<file_name>')
def download_file(dir_name, file_name):
    """Serve generated files"""
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], dir_name, file_name)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    return "File not found", 404

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True, port=5000)
```

### 2. AI Agent Skill / Automation Tool

**AI Pipeline Integration:**
```python
import imder
import time
import os

class ImderSkill:
    def __init__(self, workspace_dir="./ai_workspace"):
        self.workspace_dir = workspace_dir
        os.makedirs(workspace_dir, exist_ok=True)
    
    def blend_images(self, image_a: str, image_b: str, 
                     style: str = "shuffle", 
                     output_formats: list = ["gif"]) -> dict:
        """
        Blend two images using IMDER algorithms.
        Returns dictionary with paths and metadata.
        """
        timestamp = int(time.time())
        output_dir = f"{self.workspace_dir}/{timestamp}"
        
        try:
            files = imder.process(
                base=image_a,
                target=image_b,
                result=output_dir,
                results=output_formats,
                algo=style,
                res=512,
                sound="mute"
            )
            
            return {
                "success": True,
                "files": files,
                "output_dir": output_dir,
                "formats": output_formats,
                "algorithm": style,
                "timestamp": timestamp
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "timestamp": timestamp
            }
    
    def batch_blend(self, base_image: str, target_images: list, 
                   algorithm: str = "merge") -> list:
        """
        Blend one base image with multiple targets.
        """
        results = []
        for i, target in enumerate(target_images):
            result = self.blend_images(base_image, target, algorithm)
            results.append({
                "target": target,
                "result": result
            })
            print(f"Processed {i+1}/{len(target_images)}")
        
        return results
```

**Discord Bot Integration:**
```python
import discord
from discord.ext import commands
import imder
import aiohttp
import asyncio
import os

bot = commands.Bot(command_prefix="!")

@bot.event
async def on_ready():
    print(f'{bot.user} has connected to Discord!')

@bot.command(name="blend")
async def blend_command(ctx, image1_url: str, image2_url: str):
    """Blend two images from URLs"""
    await ctx.send("🔄 Downloading and processing images...")
    
    # Download images
    async with aiohttp.ClientSession() as session:
        async with session.get(image1_url) as resp:
            if resp.status == 200:
                with open("temp1.jpg", "wb") as f:
                    f.write(await resp.read())
        
        async with session.get(image2_url) as resp:
            if resp.status == 200:
                with open("temp2.jpg", "wb") as f:
                    f.write(await resp.read())
    
    # Process in thread pool to avoid blocking
    def process_images():
        return imder.process(
            base="temp1.jpg",
            target="temp2.jpg",
            result="./discord_output",
            results=["gif"],
            algo="missform",
            res=256,
            sound="mute"
        )
    
    loop = asyncio.get_event_loop()
    files = await loop.run_in_executor(None, process_images)
    
    # Send result
    if files and os.path.exists(files[0]):
        await ctx.send("✅ Processing complete!", file=discord.File(files[0]))
        
        # Cleanup
        os.remove("temp1.jpg")
        os.remove("temp2.jpg")
        os.remove(files[0])
    else:
        await ctx.send("❌ Processing failed. Please check the URLs.")

# Run bot
bot.run('YOUR_BOT_TOKEN')
```

### 3. Plugin Architecture for Existing Applications

```python
# Standardized plugin interface
from typing import Dict, Any, List
import imder

class IMDERPlugin:
    """IMDER plugin for media processing pipelines"""
    
    name = "imder_blender"
    version = "1.0.0"
    description = "Pixel blending and morphing plugin"
    supports = ["image", "video"]
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {
            "default_algorithm": "shuffle",
            "default_resolution": 512,
            "default_sound": "mute",
            "output_dir": "./imder_output"
        }
    
    def validate_inputs(self, base_path: str, target_path: str) -> bool:
        """Validate input files"""
        import os
        return os.path.exists(base_path) and os.path.exists(target_path)
    
    def process(self, base_path: str, target_path: str, 
               options: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process two media files with IMDER.
        
        Args:
            base_path: Path to base image/video
            target_path: Path to target image/video
            options: Processing options (overrides defaults)
            
        Returns:
            Dictionary with results and metadata
        """
        options = options or {}
        
        # Merge config with options
        params = {
            "algo": options.get("algorithm", self.config["default_algorithm"]),
            "res": options.get("resolution", self.config["default_resolution"]),
            "sound": options.get("sound", self.config["default_sound"]),
            "results": options.get("formats", ["png"]),
            "sq": options.get("sq"),
            "sq_hz": options.get("sq_hz")
        }
        
        try:
            # Generate unique output directory
            import uuid
            output_dir = f"{self.config['output_dir']}/{uuid.uuid4().hex[:8]}"
            
            # Process
            files = imder.process(
                base=base_path,
                target=target_path,
                result=output_dir,
                **params
            )
            
            return {
                "success": True,
                "files": files,
                "output_dir": output_dir,
                "parameters": params,
                "input_files": {
                    "base": base_path,
                    "target": target_path
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "parameters": params
            }
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Return plugin capabilities"""
        return {
            "algorithms": ["shuffle", "merge", "missform", "fusion"],
            "formats": ["png", "gif", "mp4"],
            "audio_options": ["mute", "gen", "target"],
            "max_resolution": 16384,
            "supports_video": True
        }

# Usage in main application
class MediaPipeline:
    def __init__(self):
        self.plugins = {}
    
    def register_plugin(self, name: str, plugin):
        self.plugins[name] = plugin
    
    def process_with_plugin(self, plugin_name: str, *args, **kwargs):
        if plugin_name in self.plugins:
            return self.plugins[plugin_name].process(*args, **kwargs)
        raise ValueError(f"Plugin '{plugin_name}' not found")

# Initialize
pipeline = MediaPipeline()
pipeline.register_plugin("imder", IMDERPlugin())

# Use it
result = pipeline.process_with_plugin(
    "imder",
    base_path="input1.jpg",
    target_path="input2.jpg",
    options={
        "algorithm": "fusion",
        "resolution": 1024,
        "formats": ["gif", "mp4"]
    }
)
```

---

## Performance Characteristics & Optimization

### Resource Usage

**CPU Usage:**
- Single-threaded, CPU-bound processing
- Optimized NumPy operations for vectorized computation
- No GPU acceleration (runs on any CPU)

**Memory Requirements:**
- RAM ≈ `(resolution² × 3 × 4) × 2` bytes during peak processing
- Examples:
  - 128×128: ~0.1 GB
  - 512×512: ~1.5 GB
  - 1024×1024: ~6 GB
  - 2048×2048: ~12 GB minimum

**Storage Requirements:**
- Temporary files created during video processing
- Output file sizes:
  - PNG: 1-10 MB (depending on resolution)
  - GIF: 5-50 MB (302 frames)
  - MP4: 10-100 MB (30fps video)

### Processing Time Estimates

| Resolution | Frames | Estimated Time | Notes |
|------------|--------|----------------|-------|
| 128×128 | 302 | 1-3 seconds | Instant preview |
| 256×256 | 302 | 3-8 seconds | Quick processing |
| 512×512 | 302 | 10-30 seconds | Standard quality |
| 1024×1024 | 302 | 45-120 seconds | High quality |
| 2048×2048 | 302 | 3-8 minutes | Maximum practical |

**Factors affecting speed:**
- CPU speed (single core performance)
- Input file sizes and formats
- Number of output formats requested
- Audio processing (adds ~10-20% time)

### Progress Monitoring

Since the library runs headless, implement progress tracking:

```python
import imder
import threading
import time

def process_with_progress(base, target, **kwargs):
    """Process with progress indication"""
    
    result = {"files": None, "error": None, "progress": 0}
    
    def run():
        try:
            result["files"] = imder.process(base, target, **kwargs)
            result["progress"] = 100
        except Exception as e:
            result["error"] = str(e)
    
    # Start processing thread
    thread = threading.Thread(target=run)
    thread.start()
    
    # Progress monitoring (simplified - actual progress would need hooks)
    print("Starting processing...")
    while thread.is_alive():
        # In real implementation, you might:
        # 1. Monitor output directory for temp files
        # 2. Estimate based on typical timing
        # 3. Use a callback system if implemented
        time.sleep(0.5)
        print(".", end="", flush=True)
    
    print("\nDone!")
    thread.join()
    
    if result["error"]:
        print(f"Error: {result['error']}")
    else:
        print(f"Created {len(result['files'])} file(s)")
    
    return result

# Usage
process_with_progress(
    base="a.jpg",
    target="b.jpg",
    result="./output",
    results=["gif"],
    algo="shuffle",
    res=512,
    sound="mute"
)
```

---

## Troubleshooting & Common Issues

### Installation Problems

**"No module named 'imder'"**
```bash
# Solution 1: Reinstall in current environment
pip uninstall imder
pip install imder

# Solution 2: Check Python environment
python -c "import sys; print(sys.executable)"
pip --version  # Ensure pip matches python

# Solution 3: Install with verbose output
pip install imder -v
```

**"Failed building wheel for opencv-python"**
```bash
# Install pre-built binaries
pip install --upgrade pip
pip install opencv-python-headless  # Lighter version
# Or use conda:
conda install -c conda-forge opencv
```

### Runtime Errors

**"FFmpeg not found" when using audio**
```python
# Option 1: Install FFmpeg (recommended)
# See installation section

# Option 2: Disable audio
imder.process(..., sound="mute")

# Option 3: Check FFmpeg path
import subprocess
try:
    subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    print("FFmpeg found")
except:
    print("FFmpeg not in PATH")
```

**Memory Error / "Resolution too large"**
```python
# Reduce resolution
imder.process(..., res=512)  # Instead of 2048

# Check available memory
import psutil
memory_gb = psutil.virtual_memory().available / (1024**3)
print(f"Available RAM: {memory_gb:.1f} GB")
# 512px ≈ 1.5GB, 1024px ≈ 6GB, 2048px ≈ 12GB needed
```

**Processing takes forever**
- This is normal for high resolutions
- Check CPU usage (should be near 100% on one core)
- Lower resolution for testing (128 or 256)
- Video processing is slower than images

**Output files not created**
```python
import os

# Check file existence
if not os.path.exists("input.jpg"):
    print("Input file missing")

# Check write permissions
output_dir = "./output"
try:
    os.makedirs(output_dir, exist_ok=True)
    test_file = os.path.join(output_dir, "test.txt")
    with open(test_file, "w") as f:
        f.write("test")
    os.remove(test_file)
    print("Directory writable")
except Exception as e:
    print(f"Write permission error: {e}")

# Wrap in try-except for error details
try:
    files = imder.process(...)
except Exception as e:
    print(f"Processing error: {e}")
    import traceback
    traceback.print_exc()
```

### Quality Issues

**Low-quality GIF output**
```python
# GIF has color palette limitations (256 colors max)
# For better quality, use MP4 instead
imder.process(..., results=["mp4"])

# Or export PNG sequence and convert with better tools
imder.process(..., results=["png"])
# Then use: convert *.png output.gif
```

**Audio quality problems**
```python
# Use sq_hz for precise control instead of sq
imder.process(..., sound="target", sq_hz=44100)  # CD quality

# Or use highest quality setting
imder.process(..., sound="target", sq=10)  # Best bitrate
```

**Video frame mismatch**
```python
# Videos with different lengths are automatically truncated
# to the shorter video's length

# To preserve full length, extend shorter video first
# or pre-process videos to same length
```

---

## Advanced Configuration & Customization

### Environment Variables

Set these before importing/imder for custom behavior:

```python
import os

# Control temporary directory (useful for limited storage)
os.environ["IMDER_TEMP_DIR"] = "/tmp/imder_temp"

# Disable FFmpeg check (if you know it's installed)
os.environ["IMDER_NO_FFMPEG_CHECK"] = "1"

# Set default resolution
os.environ["IMDER_DEFAULT_RES"] = "1024"

# Now import
import imder
```

### Extending the Library

Create custom algorithm wrappers:

```python
import imder
import numpy as np

class CustomBlender:
    def __init__(self):
        self.base = None
        self.target = None
    
    def load_images(self, base_path, target_path):
        import cv2
        self.base = cv2.imread(base_path)
        self.target = cv2.imread(target_path)
        return self
    
    def custom_algorithm(self, resolution=512):
        """Custom blending logic using IMDER as backend"""
        # Pre-process images
        import cv2
        
        # Resize to target resolution
        base_resized = cv2.resize(self.base, (resolution, resolution))
        target_resized = cv2.resize(self.target, (resolution, resolution))
        
        # Apply custom transformation
        # (Example: blend based on edge detection)
        base_gray = cv2.cvtColor(base_resized, cv2.COLOR_BGR2GRAY)
        target_gray = cv2.cvtColor(target_resized, cv2.COLOR_BGR2GRAY)
        
        edges_base = cv2.Canny(base_gray, 100, 200)
        edges_target = cv2.Canny(target_gray, 100, 200)
        
        # Create mask from edges
        mask = (edges_base > 0) | (edges_target > 0)
        
        # Use IMDER for the actual blending
        result_files = imder.process(
            base="temp_base.jpg",
            target="temp_target.jpg",
            result="./custom_output",
            results=["png"],
            algo="fusion",
            res=resolution,
            sound="mute"
        )
        
        # Post-process with custom mask
        result = cv2.imread(result_files[0])
        
        # Apply edge-based blending
        for c in range(3):
            result[:, :, c] = np.where(
                mask,
                base_resized[:, :, c] * 0.7 + result[:, :, c] * 0.3,
                result[:, :, c]
            )
        
        return result

# Usage
blender = CustomBlender()
blender.load_images("a.jpg", "b.jpg")
result = blender.custom_algorithm(512)
cv2.imwrite("custom_blend.jpg", result)
```

### Batch Processing Script

```python
#!/usr/bin/env python3
"""
Advanced batch processor using IMDER
Features: Progress tracking, error recovery, config presets
"""

import imder
import os
import json
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

class BatchProcessor:
    def __init__(self, config_file="batch_config.json"):
        self.config = self.load_config(config_file)
        self.results = []
        self.errors = []
        
    def load_config(self, config_file):
        """Load batch processing configuration"""
        default_config = {
            "input_dir": "./input",
            "output_dir": "./batch_output",
            "base_image": None,  # If None, process all pairs
            "algorithms": ["shuffle", "merge"],
            "resolutions": [256, 512],
            "formats": ["gif"],
            "sound": "mute",
            "max_workers": 2,  # Parallel processing
            "retry_failed": True,
            "log_file": "batch_log.json"
        }
        
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config
    
    def discover_files(self):
        """Find all image/video files in input directory"""
        media_extensions = {
            '.jpg', '.jpeg', '.png', '.bmp', '.tiff',
            '.mp4', '.avi', '.mov', '.mkv', '.webm'
        }
        
        files = []
        for root, _, filenames in os.walk(self.config["input_dir"]):
            for filename in filenames:
                ext = os.path.splitext(filename)[1].lower()
                if ext in media_extensions:
                    files.append(os.path.join(root, filename))
        
        return files
    
    def generate_jobs(self, files):
        """Generate processing jobs from file list"""
        jobs = []
        
        if self.config["base_image"]:
            # One base, many targets
            base = self.config["base_image"]
            if not os.path.exists(base):
                raise FileNotFoundError(f"Base image not found: {base}")
            
            for target in files:
                if target != base:
                    for algo in self.config["algorithms"]:
                        for res in self.config["resolutions"]:
                            jobs.append({
                                "base": base,
                                "target": target,
                                "algo": algo,
                                "res": res,
                                "formats": self.config["formats"],
                                "sound": self.config["sound"],
                                "output_subdir": f"{os.path.basename(base)}_{os.path.basename(target)}"
                            })
        else:
            # Process all pairs
            for i in range(len(files)):
                for j in range(i+1, len(files)):
                    for algo in self.config["algorithms"]:
                        for res in self.config["resolutions"]:
                            jobs.append({
                                "base": files[i],
                                "target": files[j],
                                "algo": algo,
                                "res": res,
                                "formats": self.config["formats"],
                                "sound": self.config["sound"],
                                "output_subdir": f"pair_{i}_{j}"
                            })
        
        return jobs
    
    def process_job(self, job):
        """Process a single job with retry logic"""
        max_retries = 3 if self.config["retry_failed"] else 1
        
        for attempt in range(max_retries):
            try:
                output_dir = os.path.join(
                    self.config["output_dir"],
                    job["output_subdir"],
                    f"{job['algo']}_{job['res']}"
                )
                
                print(f"Processing: {os.path.basename(job['base'])} + "
                      f"{os.path.basename(job['target'])} "
                      f"[{job['algo']}@{job['res']}px] "
                      f"(attempt {attempt+1}/{max_retries})")
                
                start_time = time.time()
                
                files = imder.process(
                    base=job["base"],
                    target=job["target"],
                    result=output_dir,
                    results=job["formats"],
                    algo=job["algo"],
                    res=job["res"],
                    sound=job["sound"]
                )
                
                elapsed = time.time() - start_time
                
                result = {
                    "success": True,
                    "job": job,
                    "files": files,
                    "output_dir": output_dir,
                    "elapsed_seconds": elapsed,
                    "timestamp": datetime.now().isoformat()
                }
                
                print(f"  ✓ Success ({elapsed:.1f}s)")
                return result
                
            except Exception as e:
                print(f"  ✗ Error: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(1)  # Wait before retry
                else:
                    return {
                        "success": False,
                        "job": job,
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    }
    
    def run(self):
        """Execute batch processing"""
        print("=" * 60)
        print("IMDER Batch Processor")
        print("=" * 60)
        
        # Prepare
        files = self.discover_files()
        print(f"Found {len(files)} media files")
        
        jobs = self.generate_jobs(files)
        print(f"Generated {len(jobs)} processing jobs")
        
        os.makedirs(self.config["output_dir"], exist_ok=True)
        
        # Process in parallel
        print(f"Processing with {self.config['max_workers']} workers...")
        start_total = time.time()
        
        with ThreadPoolExecutor(max_workers=self.config["max_workers"]) as executor:
            future_to_job = {
                executor.submit(self.process_job, job): job 
                for job in jobs
            }
            
            for future in as_completed(future_to_job):
                result = future.result()
                if result["success"]:
                    self.results.append(result)
                else:
                    self.errors.append(result)
        
        # Summary
        elapsed_total = time.time() - start_total
        print("\n" + "=" * 60)
        print("BATCH PROCESSING COMPLETE")
        print("=" * 60)
        print(f"Total time: {elapsed_total:.1f}s")
        print(f"Jobs: {len(jobs)}")
        print(f"Successful: {len(self.results)}")
        print(f"Failed: {len(self.errors)}")
        
        # Save log
        log_data = {
            "config": self.config,
            "summary": {
                "total_jobs": len(jobs),
                "successful": len(self.results),
                "failed": len(self.errors),
                "total_time": elapsed_total
            },
            "results": self.results,
            "errors": self.errors
        }
        
        with open(self.config["log_file"], 'w') as f:
            json.dump(log_data, f, indent=2, default=str)
        
        print(f"\nLog saved to: {self.config['log_file']}")

if __name__ == "__main__":
    processor = BatchProcessor("my_batch_config.json")
    processor.run()
```

**Sample batch_config.json:**
```json
{
  "input_dir": "./my_images",
  "output_dir": "./batch_results",
  "base_image": "./my_images/base_pattern.jpg",
  "algorithms": ["shuffle", "merge"],
  "resolutions": [256, 512],
  "formats": ["gif", "mp4"],
  "sound": "gen",
  "max_workers": 4,
  "retry_failed": true
}
```

---

## Version Compatibility & Migration

### Current Version: 1.x.x
- **API Stability:** All public functions maintain backward compatibility
- **Output Consistency:** Identical results to source IMDER v1.1.x with same parameters
- **Dependencies:** Locked to compatible versions of OpenCV, NumPy, Pillow

### Upgrading
```bash
# Always safe to upgrade within 1.x.x
pip install --upgrade imder

# Check version
python -c "import imder; print(imder.__version__)"  # If available
```

### Migration Notes

**From GUI version to PyPI library:**
- Remove all PyQt5 imports and GUI code
- Replace interactive widgets with API calls
- Handle progress indication differently (no live preview)
- Use file system monitoring for completion detection

**From older PyPI versions:**
- New parameters `sq` and `sq_hz` are optional
- Existing code continues to work without changes
- Enhanced error messages and validation

---

## Support & Community

- **GitHub Issues:** [HAKORADev/IMDER/issues](https://github.com/HAKORADev/IMDER/issues)
- **PyPI Page:** [pypi.org/project/imder](https://pypi.org/project/imder/)
- **Documentation:** This README is the primary documentation

**Reporting Issues:**
When reporting issues, include:
1. Python version: `python --version`
2. IMDER version: `pip show imder`
3. Complete error traceback
4. Example code that reproduces the issue
5. Input file types and sizes

**Feature Requests:**
Submit via GitHub issues. Popular requests may be implemented in future versions.

---

## License & Attribution

IMDER PyPI Library is released under the MIT License.

**Attribution:**
If you use IMDER in your project, consider:
- Mentioning IMDER in your documentation
- Linking to the GitHub repository
- Sharing your creations with the community

**Commercial Use:**
Allowed without restrictions under MIT License.

---

**Ready to blend?** Start with:
```bash
pip install imder
python -c "import imder; print('IMDER loaded successfully!')"
```

Then try the interactive mode:
```bash
imder
```

Or dive right into the API:
```python
import imder
# Your creative code here...
```
