# IMDER PyPI Library Documentation

**The lightweight, automation-friendly pixel blender for Python.**

🔗 **PyPI Package:** [imder](https://pypi.org/project/imder/)  
📦 **Install:** `pip install imder`  
🐍 **Python:** 3.8+ required  
💻 **Source Code:** [HAKORADev/IMDER](https://github.com/HAKORADev/IMDER) (GUI/CLI version)

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

## Installation & Dependencies

```bash
pip install imder
```

### System Dependencies

**FFmpeg (Optional but Recommended)**
- **Why:** Required only for video processing with audio extraction (`sound='target'`)
- **Without it:** Video processing works fine, but audio features are auto-disabled
- **Install:**
  - Windows: `winget install FFmpeg`
  - macOS: `brew install ffmpeg`
  - Linux: `sudo apt install ffmpeg`

**Python Dependencies (Auto-installed):**
- `opencv-python` (>=4.5.0) - Core image/video processing
- `numpy` (>=1.21.0) - High-performance array operations
- `Pillow` (>=9.0.0) - GIF export and image I/O
- `pyfiglet` (>=0.8.0) - CLI banner display

**No GUI Dependencies:** Unlike the source version, this doesn't require PyQt5, making it ideal for server environments.

---

## Core API Reference

### `imder.process(base, target, result, results, algo, res, sound, sq=None)`

The main processing function. Returns a list of output file paths.

**Parameters:**

| Parameter | Type | Description | Options |
|-----------|------|-------------|---------|
| `base` | str | Path to base image/video | Any valid media path |
| `target` | str | Path to target image/video | Any valid media path |
| `result` | str | Output directory path | Will create if not exists |
| `results` | list | List of formats to export | `['png']`, `['gif']`, `['mp4']`, or combinations |
| `algo` | str | Processing algorithm | `'shuffle'`, `'merge'`, `'missform'`, `'fusion'` |
| `res` | int | Processing resolution | `128`, `256`, `512`, `1024`, `2048` |
| `sound` | str | Audio option | `'mute'`, `'gen'`, `'target'` |
| `sq` | int (optional) | Sound quality (1-10) | Only used when `sound='target'` |

**Returns:** `List[str]` - Absolute paths to generated files

**Algorithm Availability:**
- **Images:** `shuffle`, `merge`, `missform`, `fusion`
- **Videos:** `shuffle`, `merge`, `missform` (fusion not supported for video)

**Sound Options:**
- `mute` - No audio
- `gen` - Generate synthetic audio from pixel values
- `target` - Extract audio from target video (requires FFmpeg, target must be video)

### `imder.launch_interactive()`

Launches the interactive CLI wizard (same as running `imder` in terminal).

---

## Usage Examples

### Basic Image Blending

```python
import imder

# Simple PNG export
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
# Export both GIF and video
files = imder.process(
    base="input1.png",
    target="input2.png",
    result="./animations",
    results=["gif", "mp4"],
    algo="missform",
    res=1024,
    sound="gen"  # Add generated audio to MP4
)

for f in files:
    print(f"Created: {f}")
```

### Video Processing with Audio

```python
# Blend two videos, keeping target audio at 50% quality
imder.process(
    base="clip1.mp4",
    target="clip2.mp4",
    result="./blended_videos",
    results=["mp4"],
    algo="merge",
    res=512,
    sound="target",
    sq=5  # 50% audio quality
)
```

### Batch Processing

```python
import os

targets = ["cat.jpg", "dog.jpg", "bird.jpg"]
base_image = "pattern.jpg"

for i, target in enumerate(targets):
    imder.process(
        base=base_image,
        target=target,
        result=f"./batch_results/set_{i}",
        results=["png", "gif"],
        algo="shuffle",
        res=256,
        sound="mute"
    )
    print(f"Processed {i+1}/{len(targets)}")
```

---

## Command Line Usage

After `pip install`, the `imder` command is available globally.

### Interactive Mode
```bash
imder
```
Full interactive wizard with prompts for all options.

### Direct Execution
```bash
# Basic syntax
imder <base> <target> <result_folder> --results <formats> --algo <name> --res <int> --sound <type>

# Examples
imder photo.jpg art.jpg ./output --results png gif --algo shuffle --res 512

imder video1.mp4 video2.mp4 ./out --results mp4 --algo missform --res 256 --sound target --sq 8

imder image.png target.mp4 ./mixed --results mp4 --algo merge --res 1024 --sound target --sq 10
```

---

## Real-World Integration Examples

### 1. Build Your Own GUI

Since this library has no GUI, you can wrap it in any framework:

**Tkinter Example:**
```python
import tkinter as tk
from tkinter import ttk, filedialog
import imder
import threading

class ImderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("My IMDER GUI")
        
        # Setup UI
        tk.Button(root, text="Select Base", command=self.load_base).pack()
        tk.Button(root, text="Select Target", command=self.load_target).pack()
        tk.Button(root, text="Process", command=self.run).pack()
        
        self.status = tk.Label(root, text="Ready")
        self.status.pack()
        
        self.base_path = None
        self.target_path = None
    
    def load_base(self):
        self.base_path = filedialog.askopenfilename()
    
    def load_target(self):
        self.target_path = filedialog.askopenfilename()
    
    def run(self):
        if not self.base_path or not self.target_path:
            return
        
        self.status.config(text="Processing...")
        
        def process():
            try:
                files = imder.process(
                    self.base_path,
                    self.target_path,
                    "./my_output",
                    ["gif"],
                    "shuffle",
                    512,
                    "mute"
                )
                # Display result
                self.status.config(text=f"Done! Saved to {files[0]}")
            except Exception as e:
                self.status.config(text=f"Error: {e}")
        
        threading.Thread(target=process).start()

root = tk.Tk()
app = ImderApp(root)
root.mainloop()
```

**Web Interface (Flask):**
```python
from flask import Flask, request, jsonify
import imder
import os

app = Flask(__name__)

@app.route('/blend', methods=['POST'])
def blend():
    data = request.json
    try:
        files = imder.process(
            data['base_path'],
            data['target_path'],
            "./web_output",
            data.get('formats', ['png']),
            data.get('algo', 'shuffle'),
            data.get('res', 512),
            data.get('sound', 'mute')
        )
        return jsonify({"success": True, "files": files})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
```

### 2. AI Agent Skill / Automation Tool

**Use case:** AI image generation pipeline where the agent blends intermediate results.

```python
# In your agent's skill set
class ImderSkill:
    def blend_images(self, image_a: str, image_b: str, style: str = "shuffle") -> str:
        """
        Blend two images using IMDER algorithms.
        Returns path to generated GIF.
        """
        output_dir = f"./agent_workspace/{int(time.time())}"
        result = imder.process(
            base=image_a,
            target=image_b,
            result=output_dir,
            results=["gif"],
            algo=style,
            res=512,
            sound="mute"
        )
        return result[0] if result else None
```

**Discord Bot Integration:**
```python
@bot.command()
async def blend(ctx, base_url: str, target_url: str):
    """Blend two images and send the result"""
    # Download images from URLs...
    
    await ctx.send("Processing... this may take a moment.")
    
    files = imder.process(
        base="downloaded_base.jpg",
        target="downloaded_target.jpg",
        result="./discord_output",
        results=["gif"],
        algo="missform",
        res=256,
        sound="mute"
    )
    
    await ctx.send(file=discord.File(files[0]))
```

### 3. Plugin Architecture

```python
# In your main application
class ImageProcessorPlugin:
    def __init__(self):
        self.name = "imder"
        self.supports_video = True
    
    def process(self, input_data, config):
        """Standardized plugin interface"""
        return imder.process(
            base=input_data['base'],
            target=input_data['target'],
            result=config['output_dir'],
            results=config['formats'],
            algo=config.get('algorithm', 'shuffle'),
            res=config.get('resolution', 512),
            sound=config.get('sound', 'mute')
        )

# Register plugin
plugins = {'imder': ImageProcessorPlugin()}
```

---

## Limitations vs Source Code

This library is intentionally minimal. Here's what you trade for portability:

| Feature | PyPI Library | Source GUI |
|---------|--------------|------------|
| **Live Preview** | ❌ Not available (see workaround below) | ✅ Real-time animation preview |
| **Shape Tools** | ❌ No manual masking or auto-segmentation | ✅ Pen tool + K-means clustering |
| **Algorithm Count** | 4 core modes | 9+ advanced modes |
| **Video Modes** | Shuffle, Merge, Missform | Same limited set |
| **Interactive Widgets** | CLI only | Full Qt5 GUI |

### Live Preview Workaround

Since the library runs headless, implement your own preview:

```python
import imder
from PIL import Image
import tkinter as tk

# Process first (no preview during)
files = imder.process("a.jpg", "b.jpg", "./out", ["gif"], "shuffle", 256, "mute")

# Then display result (popup or embed)
gif = Image.open(files[0])
# Show in your UI framework of choice
```

**For video workflows:** Process a low-res version first (res=128) to verify, then run final at full resolution.

---

## Performance Characteristics

### Reliability
- **Zero crashes:** The library is pure computation—no window managers, no threading conflicts, no GUI event loop crashes
- **Memory safe:** Automatically handles large images via resizing; never crashes from OOM
- **Error transparency:** If something fails (missing file, invalid format), you get a clear exception immediately—not a frozen window

### Resource Usage
- **CPU Bound:** Processing uses single-core CPU extensively (good for containerized environments)
- **RAM Usage:** Roughly `(resolution² × 3 × 4) × 2` bytes during processing
  - 512×512 ≈ 1.5GB RAM
  - 1024×1024 ≈ 6GB RAM
  - 2048×2048 ≈ 12GB RAM minimum
- **No GPU Required:** Runs on any CPU, from Raspberry Pi to server farms
- **Time Estimates:**
  - 128×128: 1-3 seconds
  - 512×512: 5-15 seconds
  - 1024×1024: 30-90 seconds
  - 2048×2048: 2-5 minutes (depending on CPU)

### Progress Indication
By default, the library runs silently. For progress tracking, wrap the call:

```python
import threading
import time

def process_with_status():
    result = [None]
    
    def run():
        result[0] = imder.process("base.jpg", "target.jpg", "./out", 
                                  ["gif"], "shuffle", 512, "mute")
    
    t = threading.Thread(target=run)
    t.start()
    
    # Show spinner while processing
    while t.is_alive():
        print("Processing...", end="\r")
        time.sleep(0.5)
    
    print(f"\nDone: {result[0]}")

process_with_status()
```

---

## Troubleshooting

**"No module named 'imder'"**
- Ensure you're in the same Python environment where you ran `pip install imder`
- Try `python -m pip install imder`

**"FFmpeg not found" when using audio**
- Install FFmpeg and ensure it's in your system PATH
- Or use `sound='mute'` to skip audio

**"Resolution too large" / Memory Error**
- Reduce `res` parameter (512 works on most systems, 2048 requires 16GB+ RAM)

**Processing takes forever**
- This is normal for high resolutions on slower CPUs. The library is working—check your CPU usage in Task Manager/Activity Monitor.

**Output files not created**
- Check that the `result` directory path is writable
- Ensure input files exist and are valid images/videos
- The library raises exceptions on failure—wrap in try/except to see errors

---

## Version Compatibility

This PyPI package follows semantic versioning:
- v1.x.x matches the core algorithms from source v1.1.x
- Output files are identical between PyPI and source versions when using the same algorithm
- API is stable—future updates will maintain backward compatibility

---

**Ready to integrate?** `pip install imder` and start blending.
