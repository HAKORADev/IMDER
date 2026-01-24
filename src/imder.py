import sys
import os
import time
import random
import tempfile
import hashlib
import wave
import struct
import subprocess
import cv2
import numpy as np
from PIL import Image, ImageFilter
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QFileDialog, 
                             QMessageBox, QProgressBar, QFrame, QSizePolicy, 
                             QDesktopWidget, QComboBox, QMenu, QAction)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize, QPoint
from PyQt5.QtGui import QImage, QPixmap, QIcon, QFont, QColor, QPalette, QPainter, QPen

def get_icon_path():
    icon_name = 'imder.png'
    
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        icon_path = os.path.join(sys._MEIPASS, icon_name)
    else:
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            base_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
        icon_path = os.path.join(base_dir, icon_name)
    
    return icon_path

def load_app_icon():
    icon_path = get_icon_path()
    if os.path.exists(icon_path):
        return QIcon(icon_path)
    return None

try:
    import pyfiglet
    HAS_PYFIGLET = True
except ImportError:
    HAS_PYFIGLET = False

THEME = {
    'background': '#0A0A0A',
    'surface': '#1a1a1a',
    'surface_hover': '#2a2a2a',
    'surface_active': '#3a3a3a',
    'text': '#E5E5E5',
    'text_secondary': '#A0A0A0',
    'accent': '#4CAF50',
    'accent_hover': '#45a049',
    'accent_pressed': '#3d8b40',
    'accent_disabled': '#2d5a30',
    'border': '#404040',
    'border_light': '#E5E5E5',
    'border_disabled': '#555555',
    'error': '#f44336',
    'warning': '#ff9800',
    'success': '#4CAF50',
    'panel_background': '#121212',
    'panel_border': '#E5E5E5',
}

def get_main_button_style():
    return """
        QPushButton {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1, 
                stop:0 #121212, stop:0.3 #121212, stop:0.7 #1a1a1a, stop:1 #121212);
            border: 2px solid #E5E5E5;
            border-radius: 8px;
            font-size: 14px;
            font-weight: bold;
            color: white;
            padding: 8px 16px;
        }
        QPushButton:hover {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1, 
                stop:0 #121212, stop:0.3 #161616, stop:0.7 #1e1e1e, stop:1 #121212);
            border: 2px solid #E5E5E5;
        }
        QPushButton:pressed {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1, 
                stop:0 #0e0e0e, stop:0.3 #121212, stop:0.7 #161616, stop:1 #0e0e0e);
            border: 2px solid #E5E5E5;
        }
        QPushButton:disabled {
            background-color: #2a2a2a;
            border: 2px solid #555555;
            color: #666666;
        }
    """

def get_secondary_button_style():
    return """
        QPushButton {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1, 
                stop:0 #121212, stop:0.3 #121212, stop:0.7 #1a1a1a, stop:1 #121212);
            border: 2px solid #E5E5E5;
            border-radius: 8px;
            font-size: 14px;
            font-weight: bold;
            color: white;
            padding: 8px 16px;
        }
        QPushButton:hover {
            background: qlineargradient(x1:0, y1:0, x2:1, y2=1, 
                stop:0 #121212, stop:0.3 #161616, stop:0.7 #1e1e1e, stop:1 #121212);
            border: 2px solid #E5E5E5;
        }
        QPushButton:pressed {
            background: qlineargradient(x1:0, y1=0, x2:1, y2=1, 
                stop:0 #0e0e0e, stop:0.3 #121212, stop:0.7 #161616, stop:1 #0e0e0e);
            border: 2px solid #E5E5E5;
        }
        QPushButton:disabled {
            background-color: #2a2a2a;
            border: 2px solid #555555;
            color: #666666;
        }
    """

def get_surface_button_style():
    return """
        QPushButton {
            background-color: #2a2a2a;
            color: white;
            border: 1px solid #3a3a3a;
            border-radius: 5px;
            font-size: 12px;
            padding: 6px 12px;
        }
        QPushButton:hover {
            background-color: #3a3a3a;
            border: 1px solid #E5E5E5;
        }
        QPushButton:pressed {
            background-color: #4a4a4a;
            border: 1px solid #E5E5E5;
        }
        QPushButton:disabled {
            background-color: #2a2a2a;
            border: 1px solid #404040;
            color: #666666;
        }
    """

def get_panel_style():
    return f"""
        QFrame {{
            background-color: {THEME['panel_background']};
            border: 2px solid {THEME['panel_border']};
            border-radius: 8px;
        }}
    """

def get_preview_style():
    return f"""
        QLabel {{
            background-color: {THEME['surface']};
            border: 1px solid {THEME['border']};
            border-radius: 4px;
        }}
    """

def get_combo_box_style():
    return f"""
        QComboBox {{
            background-color: {THEME['surface']};
            color: {THEME['text']};
            border: 2px solid {THEME['border_light']};
            border-radius: 6px;
            padding: 6px 12px;
            min-width: 120px;
            font-size: 13px;
            selection-background-color: {THEME['surface_hover']};
            selection-color: {THEME['text']};
        }}
        QComboBox::drop-down {{
            border: none;
            subcontrol-origin: padding;
            subcontrol-position: right center;
            width: 24px;
        }}
        QComboBox::down-arrow {{
            image: none();
            width: 0px;
            height: 0px;
        }}
        QComboBox:hover {{
            border: 2px solid #E5E5E5;
        }}
        QComboBox:disabled {{
            background-color: #2a2a2a;
            border: 2px solid #555555;
            color: #666666;
        }}
        QComboBox QAbstractItemView {{
            background-color: {THEME['surface']};
            color: {THEME['text']};
            border: 1px solid {THEME['border_light']};
            border-radius: 4px;
            selection-background-color: {THEME['surface_hover']};
            selection-color: {THEME['text']};
        }}
    """

def get_progress_bar_style():
    return f"""
        QProgressBar {{
            border: 1px solid {THEME['border']};
            background-color: {THEME['surface']};
            height: 8px;
            border-radius: 4px;
            text-align: center;
            color: {THEME['text_secondary']};
        }}
        QProgressBar::chunk {{
            background-color: {THEME['text_secondary']};
            border-radius: 3px;
        }}
    """

def get_title_label_style():
    return f"""
        color: {THEME['text']};
        font-weight: bold;
        font-size: 16px;
    """

def get_subtitle_label_style():
    return f"""
        color: {THEME['text_secondary']};
        font-size: 12px;
    """

def get_status_bar_style():
    return f"""
        color: {THEME['text_secondary']};
        padding: 6px 12px;
        font-size: 12px;
    """

def get_highlight_style():
    return f"""
        color: {THEME['accent']};
        font-weight: bold;
    """

def get_window_style():
    return f"""
        background-color: {THEME['background']};
        color: {THEME['text']};
    """

class Missform:
    def __init__(self, base_image, target_image, threshold=127):
        self.base_image = base_image.astype(np.float32)
        self.target_image = target_image.astype(np.float32)
        self.threshold = threshold
        
    def _create_binary_mask(self, image):
        gray = np.mean(image, axis=2)
        binary = (gray > self.threshold).astype(np.uint8) * 255
        return binary
    
    def interpolate_position(self, start_pos, end_pos, progress):
        x = start_pos[0] + (end_pos[0] - start_pos[0]) * progress
        y = start_pos[1] + (end_pos[1] - start_pos[1]) * progress
        return (int(x), int(y))
    
    def generate_morph_sequence(self, duration=10, fps=30):
        base_binary = self._create_binary_mask(self.base_image)
        target_binary = self._create_binary_mask(self.target_image)
        
        base_positions = np.column_stack(np.where(base_binary == 255))
        target_positions = np.column_stack(np.where(target_binary == 255))
        
        min_positions = min(len(base_positions), len(target_positions))
        if min_positions == 0:
            raise ValueError("No valid pixels found for morphing")
            
        base_positions = base_positions[:min_positions]
        target_positions = target_positions[:min_positions]
        
        total_frames = int(duration * fps)
        if total_frames <= 0:
            total_frames = 1
            
        frames = []
        height, width, _ = self.base_image.shape
        
        for frame_idx in range(total_frames):
            progress = frame_idx / max(1, total_frames - 1)
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            for i in range(min_positions):
                current_pos = self.interpolate_position(
                    base_positions[i], 
                    target_positions[i], 
                    progress
                )
                
                x, y = current_pos
                if 0 <= x < height and 0 <= y < width:
                    pixel_value = self.base_image[base_positions[i][0], base_positions[i][1]].astype(np.uint8)
                    frame[x, y] = pixel_value
            
            frames.append(frame)
        
        return frames
    
    def generate_single_morph(self, progress):
        base_binary = self._create_binary_mask(self.base_image)
        target_binary = self._create_binary_mask(self.target_image)
        
        base_positions = np.column_stack(np.where(base_binary == 255))
        target_positions = np.column_stack(np.where(target_binary == 255))
        
        min_positions = min(len(base_positions), len(target_positions))
        if min_positions == 0:
            raise ValueError("No valid pixels found for morphing")
            
        base_positions = base_positions[:min_positions]
        target_positions = target_positions[:min_positions]
        
        height, width, _ = self.base_image.shape
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        for i in range(min_positions):
            current_pos = self.interpolate_position(
                base_positions[i], 
                target_positions[i], 
                progress
            )
            
            x, y = current_pos
            if 0 <= x < height and 0 <= y < width:
                pixel_value = self.base_image[base_positions[i][0], base_positions[i][1]].astype(np.uint8)
                frame[x, y] = pixel_value
        
        return frame

def apply_opencv_transforms(img, rotate_steps, is_flipped):
    if img is None:
        return None
    
    if is_flipped:
        img = cv2.flip(img, 1)
    
    steps = rotate_steps % 4
    if steps == 1:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif steps == 2:
        img = cv2.rotate(img, cv2.ROTATE_180)
    elif steps == 3:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        
    return img

def quantize_image(img, levels=4):
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
        
    factor = 255 / (levels - 1)
    quantized = np.round(gray / factor) * factor
    return quantized.astype(np.uint8)

def get_morton_codes(pixels):
    x = pixels[:, 0].astype(np.uint32)
    y = pixels[:, 1].astype(np.uint32)
    z = pixels[:, 2].astype(np.uint32)
    
    def spread(n):
        n = (n | (n << 16)) & 0x030000FF
        n = (n | (n <<  8)) & 0x0300F00F
        n = (n | (n <<  4)) & 0x030C30C3
        n = (n | (n <<  2)) & 0x09249249
        return n
    
    return spread(x) | (spread(y) << 1) | (spread(z) << 2)

def assign_pixels(source_pixels, target_pixels, mode='shuffle', mask=None):
    source_flat = source_pixels.reshape(-1, 3)
    target_flat = target_pixels.reshape(-1, 3)
    num_total = len(target_flat)
    
    if mode == 'disguise':
        assignments = np.arange(num_total, dtype=np.int32)
    else:
        assignments = np.full(num_total, -1, dtype=np.int32)
    
    if (mode == 'swap' or mode == 'blend') and mask is not None:
        mask_flat = mask.flatten()
        valid_indices = np.where(mask_flat > 0)[0]
        
        if len(valid_indices) == 0:
            return assignments

        target_valid = target_flat[valid_indices]
        
        all_indices = np.arange(num_total)
        candidate_indices = all_indices[mask_flat == 0]
        candidate_pixels = source_flat[candidate_indices]
        
        if len(candidate_pixels) == 0:
            return assignments
            
        s_codes = get_morton_codes(candidate_pixels)
        s_sort_idx = np.argsort(s_codes)
        s_codes_sorted = s_codes[s_sort_idx]
        
        t_codes = get_morton_codes(target_valid)
        t_sort_local_idx = np.argsort(t_codes)
        t_codes_sorted = t_codes[t_sort_local_idx]
        
        matches_idx = np.searchsorted(s_codes_sorted, t_codes_sorted)
        matches_idx = np.clip(matches_idx, 0, len(s_codes_sorted) - 1)
        
        for i in range(1, len(matches_idx)):
            if matches_idx[i] <= matches_idx[i-1]:
                matches_idx[i] = matches_idx[i-1] + 1
        
        matches_idx = np.clip(matches_idx, 0, len(s_codes_sorted) - 1)
        
        s_chosen_local = s_sort_idx[matches_idx]
        s_chosen_global = candidate_indices[s_chosen_local]
        t_chosen_global = valid_indices[t_sort_local_idx]
        
        assignments[s_chosen_global] = t_chosen_global
        assignments[t_chosen_global] = s_chosen_global
        
        return assignments

    if mode == 'navigate' and mask is not None:
        mask_flat = mask.flatten()
        valid_indices = np.where(mask_flat > 0)[0]
        
        if len(valid_indices) == 0:
            return assignments

        target_valid = target_flat[valid_indices]
        
        s_codes = get_morton_codes(source_flat)
        s_sort_idx = np.argsort(s_codes)
        s_codes_sorted = s_codes[s_sort_idx]
        
        t_codes = get_morton_codes(target_valid)
        t_sort_local_idx = np.argsort(t_codes)
        t_codes_sorted = t_codes[t_sort_local_idx]
        
        matches_idx = np.searchsorted(s_codes_sorted, t_codes_sorted)
        matches_idx = np.clip(matches_idx, 0, len(s_codes_sorted) - 1)
        
        for i in range(1, len(matches_idx)):
            if matches_idx[i] <= matches_idx[i-1]:
                matches_idx[i] = matches_idx[i-1] + 1
        
        matches_idx = np.clip(matches_idx, 0, len(s_codes_sorted) - 1)
        
        s_chosen_indices = s_sort_idx[matches_idx]
        t_chosen_indices = valid_indices[t_sort_local_idx]
        
        assignments[s_chosen_indices] = t_chosen_indices
        
        return assignments

    if mode == 'pattern' and mask is not None:
        mask_flat = mask.flatten()
        valid_indices = np.where(mask_flat > 0)[0]
        
        if len(valid_indices) == 0:
            return assignments
            
        target_valid = target_flat[valid_indices]
        
        s_gray = np.mean(source_flat, axis=1)
        s_sort_idx = np.argsort(s_gray)
        
        t_valid_rgb = target_valid.reshape(1, -1, 3).astype(np.uint8)
        t_quant = quantize_image(t_valid_rgb, levels=4).flatten()
        t_quant_noisy = t_quant.astype(np.float32) + np.random.random(len(t_quant)) * 0.5
        t_sort_local_idx = np.argsort(t_quant_noisy)
        
        source_indices_to_use = np.linspace(0, len(source_flat)-1, len(valid_indices)).astype(int)
        s_chosen_sorted_idx = s_sort_idx[source_indices_to_use]
        
        assignments[s_chosen_sorted_idx] = valid_indices[t_sort_local_idx]
        
        return assignments

    elif mode == 'disguise' and mask is not None:
        mask_flat = mask.flatten()
        valid_indices = np.where(mask_flat > 0)[0]
        
        if len(valid_indices) == 0:
            return assignments

        s_local = source_flat[valid_indices]
        t_local = target_flat[valid_indices]
        
        s_gray = np.mean(s_local, axis=1)
        t_gray = np.mean(t_local, axis=1)
        
        s_sort = np.argsort(s_gray)
        t_sort = np.argsort(t_gray)
        
        assignments[valid_indices[s_sort]] = valid_indices[t_sort]
        
        return assignments

    elif mode == 'fusion' and mask is not None:
        mask_flat = mask.flatten()
        valid_indices = np.where(mask_flat > 0)[0]
        
        if len(valid_indices) == 0:
            return assignments
        
        source_masked = source_flat[valid_indices]
        target_masked = target_flat[valid_indices]
        
        source_gray = np.mean(source_masked, axis=1)
        target_gray = np.mean(target_masked, axis=1)
        
        s_sort = np.argsort(source_gray)
        t_sort = np.argsort(target_gray)
        
        assignments[valid_indices[s_sort]] = valid_indices[t_sort]
        
        return assignments

    num_pixels = len(source_flat)
    
    if mode == 'shuffle':
        threshold = 127
        source_gray = np.mean(source_flat, axis=1)
        target_gray = np.mean(target_flat, axis=1)
        
        s_binary = source_gray > threshold
        t_binary = target_gray > threshold
        
        s_indices = np.arange(num_pixels)
        t_indices = np.arange(num_pixels)
        
        s_black = s_indices[~s_binary]
        s_white = s_indices[s_binary]
        t_black = t_indices[~t_binary]
        t_white = t_indices[t_binary]
        
        np.random.shuffle(s_black)
        np.random.shuffle(s_white)
        np.random.shuffle(t_black)
        np.random.shuffle(t_white)
        
        min_black = min(len(s_black), len(t_black))
        assignments[s_black[:min_black]] = t_black[:min_black]
        
        min_white = min(len(s_white), len(t_white))
        assignments[s_white[:min_white]] = t_white[:min_white]
        
        s_remain = np.concatenate([s_black[min_black:], s_white[min_white:]])
        t_remain = np.concatenate([t_black[min_black:], t_white[min_white:]])
        
        if len(s_remain) > 0 and len(t_remain) > 0:
            np.random.shuffle(s_remain)
            np.random.shuffle(t_remain)
            assignments[s_remain] = t_remain
            
    else:
        s_gray = np.mean(source_flat, axis=1)
        t_gray = np.mean(target_flat, axis=1)
        
        s_sort_idx = np.argsort(s_gray)
        t_sort_idx = np.argsort(t_gray)
        
        assignments[s_sort_idx] = t_sort_idx
        
    return assignments

def extract_video_frames_with_progress(video_path, description):
    cap = cv2.VideoCapture(video_path)
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"exploring {description} frames:")
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        if (i + 1) % 10 == 0 or i == 0 or i == total_frames - 1:
            print(f"  {i + 1}", end=' ', flush=True)
    print()
    print(f"{description} Frames = {len(frames)}")
    
    cap.release()
    return frames, fps, len(frames)

def is_video_file(path):
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    return any(path.lower().endswith(ext) for ext in video_extensions)

def process_frame_pair(base_frame, target_frame, algo_mode, resolution):
    h_b, w_b = base_frame.shape[:2]
    h_t, w_t = target_frame.shape[:2]
    
    limit_res = min(h_b, w_b, h_t, w_t)
    process_res = min(resolution, limit_res)
    
    base_frame = cv2.resize(base_frame, (process_res, process_res))
    target_frame = cv2.resize(target_frame, (process_res, process_res))
    
    base_frame = cv2.cvtColor(base_frame, cv2.COLOR_BGR2RGB)
    target_frame = cv2.cvtColor(target_frame, cv2.COLOR_BGR2RGB)
    
    if algo_mode == 'missform':
        missform = Missform(base_frame, target_frame, threshold=127)
        return missform.generate_single_morph(1.0)
    
    assignments = assign_pixels(base_frame, target_frame, algo_mode, None)
    
    source_flat = base_frame.reshape(-1, 3).astype(np.float32)
    target_flat = target_frame.reshape(-1, 3).astype(np.float32)
    
    if algo_mode == 'disguise':
        valid_pixel_indices = np.arange(len(source_flat))
    elif algo_mode in ['pattern', 'navigate', 'swap', 'blend', 'fusion']:
        valid_pixel_indices = np.where(assignments != -1)[0]
    else:
        valid_pixel_indices = np.arange(len(source_flat))
    
    width, height = process_res, process_res
    start_x_all, start_y_all = np.meshgrid(np.arange(width), np.arange(height))
    start_x_all = start_x_all.flatten()
    start_y_all = start_y_all.flatten()
    
    start_x = start_x_all[valid_pixel_indices]
    start_y = start_y_all[valid_pixel_indices]
    source_colors = source_flat[valid_pixel_indices]
    
    dest_indices = assignments[valid_pixel_indices]
    end_y = dest_indices // width
    end_x = dest_indices % width
    
    t = 1.0
    curr_x = (start_x + (end_x - start_x) * t).astype(int)
    curr_y = (start_y + (end_y - start_y) * t).astype(int)
    
    curr_x = np.clip(curr_x, 0, width - 1)
    curr_y = np.clip(curr_y, 0, height - 1)
    
    if algo_mode == 'disguise':
        frame = np.zeros((height, width, 3), dtype=np.uint8)
    elif algo_mode in ['navigate', 'swap', 'blend']:
        frame = base_frame.copy()
    elif algo_mode == 'fusion':
        frame = np.zeros((height, width, 3), dtype=np.uint8)
    else:
        frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    current_colors = source_colors
    current_colors = np.clip(current_colors, 0, 255).astype(np.uint8)
    frame[curr_y, curr_x] = current_colors
    
    return frame

def generate_sound_from_frame(frame, frame_duration, sample_rate=44100):
    frame_bytes = frame.tobytes()
    frame_hash = hashlib.sha256(frame_bytes).hexdigest()
    
    hash_int = int(frame_hash[:8], 16)
    np.random.seed(hash_int)
    
    num_samples = int(sample_rate * frame_duration)
    t = np.linspace(0, frame_duration, num_samples, False)
    
    frequencies = []
    amplitudes = []
    
    for i in range(3):
        freq_seed = int(frame_hash[i*4:(i+1)*4], 16)
        freq = 50 + (freq_seed % 4000)
        amp_seed = int(frame_hash[(i+3)*4:(i+4)*4], 16)
        amp = 0.1 + (amp_seed % 9000) / 10000.0
        frequencies.append(freq)
        amplitudes.append(amp)
    
    sound = np.zeros(num_samples)
    for freq, amp in zip(frequencies, amplitudes):
        sound += amp * np.sin(2 * np.pi * freq * t)
    
    sound = sound / np.max(np.abs(sound)) if np.max(np.abs(sound)) > 0 else sound
    sound = (sound * 32767).astype(np.int16)
    
    return sound, frame_hash

def check_ffmpeg_available():
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def extract_audio_from_video(video_path, output_audio_path, duration=None, quality=30):
    if not check_ffmpeg_available():
        print("Error: ffmpeg is not installed or not found in PATH. Cannot extract audio.")
        return False
    
    video_path = os.path.abspath(video_path)
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return False
    
    cmd = ['ffmpeg', '-i', video_path]
    
    if duration:
        cmd.extend(['-t', str(duration)])
    
    quality_map = {
        10: '32k', 20: '64k', 30: '96k', 40: '128k', 50: '160k',
        60: '192k', 70: '224k', 80: '256k', 90: '320k', 100: 'copy'
    }
    
    bitrate = quality_map.get(quality, '96k')
    
    if bitrate == 'copy':
        cmd.extend(['-c:a', 'copy'])
    else:
        cmd.extend(['-b:a', bitrate])
    
    cmd.extend(['-y', output_audio_path])
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error extracting audio: {e}")
        print(f"ffmpeg stderr: {e.stderr}")
        return False

def add_audio_to_video(video_path, frames, fps, output_path, sound_option='mute', target_audio_path=None, audio_quality=30):
    if sound_option == 'mute':
        return video_path
    
    if not check_ffmpeg_available():
        print("Error: ffmpeg is not installed or not found in PATH. Cannot add audio.")
        return video_path
    
    video_path = os.path.abspath(video_path)
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return video_path
    
    temp_dir = tempfile.mkdtemp()
    output_path = os.path.abspath(output_path)
    
    if sound_option == 'target-sound' and target_audio_path:
        audio_path = os.path.join(temp_dir, "target_audio.mp3")
        if os.path.exists(target_audio_path):
            duration = len(frames) / fps if frames else None
            if extract_audio_from_video(target_audio_path, audio_path, duration, audio_quality):
                print(f"Using target video audio with {audio_quality}% quality")
            else:
                return video_path
        else:
            print(f"Error: Target audio file not found: {target_audio_path}")
            return video_path
    elif sound_option == 'sound':
        frame_duration = 1.0 / fps
        sample_rate = 44100
        
        print(f"\nGenerating sound for each frame:")
        audio_chunks = []
        total_frames = len(frames)
        
        for i, frame in enumerate(frames):
            sound_data, frame_hash = generate_sound_from_frame(frame, frame_duration, sample_rate)
            audio_chunks.append(sound_data)
            
            percent = ((i + 1) / total_frames) * 100
            print(f"Frame {i + 1}/{total_frames} pixels analyzed for sound {percent:.1f}%")
        
        print(f"\nCompiling audio chunks:")
        full_audio = np.concatenate(audio_chunks)
        
        audio_path = os.path.join(temp_dir, "audio.wav")
        audio_path = os.path.abspath(audio_path)
        with wave.open(audio_path, 'w') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(full_audio.tobytes())
    else:
        return video_path
    
    print(f"\nMerging audio with video...")
    
    cmd = [
        'ffmpeg', '-i', video_path, '-i', audio_path,
        '-c:v', 'copy', '-c:a', 'aac', '-map', '0:v:0', '-map', '1:a:0',
        '-shortest', '-y', output_path
    ]
    
    try:
        print(f"Merging audio with video using ffmpeg...")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"Successfully added sound to video: {os.path.basename(output_path)}")
    except subprocess.CalledProcessError as e:
        print(f"Error adding audio: {e}")
        print(f"ffmpeg stderr: {e.stderr}")
        return video_path
    except Exception as e:
        print(f"Unexpected error: {e}")
        return video_path
    
    import shutil
    shutil.rmtree(temp_dir)
    
    return output_path

class ProcessingThread(QThread):
    progress_signal = pyqtSignal(int)
    frame_signal = pyqtSignal(object)
    finished_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)

    def __init__(self, base_path, target_path, mode='preview', algo_mode='shuffle', 
                 base_transforms=None, target_transforms=None, mask=None, resolution=128,
                 sound_option='mute', audio_quality=30):
        super().__init__()
        self.base_path = base_path
        self.target_path = target_path
        self.mode = mode
        self.algo_mode = algo_mode
        self.base_transforms = base_transforms or {'rotate': 0, 'flip': False}
        self.target_transforms = target_transforms or {'rotate': 0, 'flip': False}
        self.mask = mask
        self.resolution = resolution
        self.sound_option = sound_option
        self.audio_quality = audio_quality
        self.running = True
        self.output_dir = "results"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def run(self):
        try:
            if self.algo_mode in ['pattern', 'disguise', 'navigate', 'swap', 'blend'] and self.mask is None:
                raise ValueError(f"{self.algo_mode.title()} mode requires analyzing shapes first.")
            
            self.process_image()
                
        except Exception as e:
            self.error_signal.emit(str(e))

    def process_image(self):
        base_img = cv2.imread(self.base_path)
        target_img = cv2.imread(self.target_path)
        
        if base_img is None or target_img is None:
            raise ValueError("Could not load images")

        h_b, w_b = base_img.shape[:2]
        h_t, w_t = target_img.shape[:2]
        
        limit_res = min(h_b, w_b, h_t, w_t)
        process_res = min(self.resolution, limit_res)
        
        base_img = apply_opencv_transforms(base_img, self.base_transforms['rotate'], self.base_transforms['flip'])
        target_img = apply_opencv_transforms(target_img, self.target_transforms['rotate'], self.target_transforms['flip'])
        
        base_img = cv2.resize(base_img, (process_res, process_res))
        target_img = cv2.resize(target_img, (process_res, process_res))
        
        base_img = cv2.cvtColor(base_img, cv2.COLOR_BGR2RGB)
        target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
        
        if self.algo_mode == 'missform':
            missform = Missform(base_img, target_img, threshold=127)
            frames = 302
            width, height = process_res, process_res
            
            if self.mode == 'export_video':
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                if self.sound_option != 'mute':
                    silent_path = os.path.join(self.output_dir, f"video_{timestamp}_silent.mp4")
                    silent_path = os.path.abspath(silent_path)
                    out_path = os.path.join(self.output_dir, f"video_{timestamp}.mp4")
                    out_path = os.path.abspath(out_path)
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(silent_path, fourcc, 30.0, (width, height))
                    self.video_frames = []
                else:
                    out_path = os.path.join(self.output_dir, f"video_{timestamp}.mp4")
                    out_path = os.path.abspath(out_path)
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(out_path, fourcc, 30.0, (width, height))
            elif self.mode == 'export_gif':
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                out_path = os.path.join(self.output_dir, f"animation_{timestamp}.gif")
                gif_frames = []
            
            for f in range(frames):
                if not self.running:
                    break
                    
                progress = f / (frames - 1)
                self.progress_signal.emit(int(progress * 100))
                
                t = progress * progress * (3 - 2 * progress)
                frame = missform.generate_single_morph(t)
                
                if self.mode == 'preview':
                    h, w, ch = frame.shape
                    bytes_per_line = ch * w
                    qimg = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    self.frame_signal.emit(qimg)
                    time.sleep(1/60)
                elif self.mode == 'export_video':
                    if self.sound_option != 'mute':
                        self.video_frames.append(frame.copy())
                    out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                elif self.mode == 'export_gif':
                    gif_frames.append(Image.fromarray(frame))
            
            if self.mode == 'export_video':
                out.release()
                if self.sound_option != 'mute':
                    final_path = add_audio_to_video(silent_path, self.video_frames, 30.0, out_path, 
                                                   self.sound_option, self.target_path if self.sound_option == 'target-sound' else None,
                                                   self.audio_quality)
                    if os.path.exists(silent_path):
                        os.remove(silent_path)
                    self.finished_signal.emit(f"Saved to {final_path}")
                else:
                    self.finished_signal.emit(f"Saved to {out_path}")
            elif self.mode == 'export_image':
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                out_path = os.path.join(self.output_dir, f"image_{timestamp}.png")
                final_frame = missform.generate_single_morph(1.0)
                cv2.imwrite(out_path, cv2.cvtColor(final_frame, cv2.COLOR_RGB2BGR))
                self.finished_signal.emit(f"Saved to {out_path}")
            elif self.mode == 'export_gif' and gif_frames:
                gif_frames[0].save(out_path, save_all=True, append_images=gif_frames[1:], optimize=False, duration=33, loop=0)
                self.finished_signal.emit(f"Saved to {out_path}")
            else:
                self.finished_signal.emit("Preview finished")
            
            return
        
        proc_mask = None
        if self.algo_mode in ['pattern', 'disguise', 'navigate', 'swap', 'blend', 'fusion'] and self.mask is not None:
            proc_mask = cv2.resize(self.mask.astype(np.uint8), (process_res, process_res), interpolation=cv2.INTER_NEAREST)
        
        assignments = assign_pixels(base_img, target_img, self.algo_mode, proc_mask)
        
        frames = 302 
        width, height = process_res, process_res
        
        source_flat = base_img.reshape(-1, 3).astype(np.float32)
        target_flat = target_img.reshape(-1, 3).astype(np.float32)
        
        if self.algo_mode == 'disguise':
            valid_pixel_indices = np.arange(len(source_flat))
        elif self.algo_mode in ['pattern', 'navigate', 'swap', 'blend', 'fusion']:
            valid_pixel_indices = np.where(assignments != -1)[0]
        else:
            valid_pixel_indices = np.arange(len(source_flat))
        
        start_x_all, start_y_all = np.meshgrid(np.arange(width), np.arange(height))
        start_x_all = start_x_all.flatten()
        start_y_all = start_y_all.flatten()
        
        start_x = start_x_all[valid_pixel_indices]
        start_y = start_y_all[valid_pixel_indices]
        source_colors = source_flat[valid_pixel_indices]
        
        dest_indices = assignments[valid_pixel_indices]
        end_y = dest_indices // width
        end_x = dest_indices % width
        
        target_colors_aligned = target_flat[dest_indices]

        shuffled_source_colors = None
        if self.algo_mode == 'fusion' and self.mask is not None:
            np.random.seed(42)
            shuffle_indices = np.random.permutation(len(source_colors))
            shuffled_source_colors = source_colors[shuffle_indices]

        if self.algo_mode == 'blend':
            target_gray = cv2.cvtColor(target_img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
            grad_x = cv2.Sobel(target_gray, cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(target_gray, cv2.CV_32F, 0, 1, ksize=3)
            
            source_luma = (0.299 * source_colors[:,0] + 0.587 * source_colors[:,1] + 0.114 * source_colors[:,2]) / 255.0
            
            curr_pos_x = start_x.astype(np.float32)
            curr_pos_y = start_y.astype(np.float32)

        if self.mode == 'export_video':
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            if self.sound_option != 'mute':
                silent_path = os.path.join(self.output_dir, f"video_{timestamp}_silent.mp4")
                silent_path = os.path.abspath(silent_path)
                out_path = os.path.join(self.output_dir, f"video_{timestamp}.mp4")
                out_path = os.path.abspath(out_path)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(silent_path, fourcc, 30.0, (width, height))
                self.video_frames = []
            else:
                out_path = os.path.join(self.output_dir, f"video_{timestamp}.mp4")
                out_path = os.path.abspath(out_path)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(out_path, fourcc, 30.0, (width, height))
        elif self.mode == 'export_gif':
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            out_path = os.path.join(self.output_dir, f"animation_{timestamp}.gif")
            gif_frames = []

        for f in range(frames):
            if not self.running:
                break
                
            progress = f / (frames - 1)
            self.progress_signal.emit(int(progress * 100))
            
            if self.algo_mode == 'blend':
                cx_int = np.clip(curr_pos_x, 0, width - 1).astype(int)
                cy_int = np.clip(curr_pos_y, 0, height - 1).astype(int)
                
                target_luma_at_pos = target_gray[cy_int, cx_int]
                diff = source_luma - target_luma_at_pos
                
                gx_sample = grad_x[cy_int, cx_int]
                gy_sample = grad_y[cy_int, cx_int]
                
                k_dest = 0.05 + (progress * 0.5)
                k_home = 0.05 * (1.0 - progress)
                k_grad = 6.0 * (1.0 - progress * 0.2)
                
                force_grad_x = gx_sample * diff * k_grad 
                force_grad_y = gy_sample * diff * k_grad
                
                force_home_x = (start_x - curr_pos_x) * k_home
                force_home_y = (start_y - curr_pos_y) * k_home
                
                force_dest_x = (end_x - curr_pos_x) * k_dest
                force_dest_y = (end_y - curr_pos_y) * k_dest
                
                curr_pos_x += force_grad_x + force_home_x + force_dest_x
                curr_pos_y += force_grad_y + force_home_y + force_dest_y
                
                curr_x = np.clip(curr_pos_x, 0, width - 1).astype(int)
                curr_y = np.clip(curr_pos_y, 0, height - 1).astype(int)
                
            else:
                t = progress * progress * (3 - 2 * progress)
                curr_x = (start_x + (end_x - start_x) * t).astype(int)
                curr_y = (start_y + (end_y - start_y) * t).astype(int)
                
                curr_x = np.clip(curr_x, 0, width - 1)
                curr_y = np.clip(curr_y, 0, height - 1)
            
            if self.algo_mode == 'fusion' and self.mask is not None and shuffled_source_colors is not None:
                current_colors = shuffled_source_colors * (1 - t) + target_colors_aligned * t
            elif self.algo_mode == 'fusion' and self.mask is None:
                current_colors = source_colors * (1 - t) + target_colors_aligned * t
            else:
                current_colors = source_colors
            
            current_colors = np.clip(current_colors, 0, 255).astype(np.uint8)
            
            if self.algo_mode == 'disguise':
                 frame = np.zeros((height, width, 3), dtype=np.uint8)
            elif self.algo_mode in ['navigate', 'swap', 'blend']:
                 frame = base_img.copy()
            elif self.algo_mode == 'fusion':
                if self.mask is None:
                    frame = np.zeros((height, width, 3), dtype=np.uint8)
                else:
                    frame = np.zeros((height, width, 3), dtype=np.uint8)
                    unmasked_mask = proc_mask == 0
                    if np.any(unmasked_mask):
                        frame[unmasked_mask] = base_img[unmasked_mask]
            else:
                 frame = np.zeros((height, width, 3), dtype=np.uint8)
                 
            frame[curr_y, curr_x] = current_colors
            
            if self.mode == 'preview':
                h, w, ch = frame.shape
                bytes_per_line = ch * w
                qimg = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.frame_signal.emit(qimg)
                time.sleep(1/60)
            elif self.mode == 'export_video':
                if self.sound_option != 'mute':
                    self.video_frames.append(frame.copy())
                out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            elif self.mode == 'export_gif':
                gif_frames.append(Image.fromarray(frame))
                
        if self.mode == 'export_video':
            out.release()
            if self.sound_option != 'mute':
                final_path = add_audio_to_video(silent_path, self.video_frames, 30.0, out_path, 
                                               self.sound_option, self.target_path if self.sound_option == 'target-sound' else None,
                                               self.audio_quality)
                if os.path.exists(silent_path):
                    os.remove(silent_path)
                self.finished_signal.emit(f"Saved to {final_path}")
            else:
                self.finished_signal.emit(f"Saved to {out_path}")
        elif self.mode == 'export_image':
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            out_path = os.path.join(self.output_dir, f"image_{timestamp}.png")
            
            if self.algo_mode == 'fusion' and self.mask is not None:
                final_frame = np.zeros((height, width, 3), dtype=np.uint8)
                unmasked_mask = proc_mask == 0
                if np.any(unmasked_mask):
                    final_frame[unmasked_mask] = base_img[unmasked_mask]
            else:
                final_frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            final_frame[end_y, end_x] = current_colors 
            
            cv2.imwrite(out_path, cv2.cvtColor(final_frame, cv2.COLOR_RGB2BGR))
            self.finished_signal.emit(f"Saved to {out_path}")
        elif self.mode == 'export_gif' and gif_frames:
            gif_frames[0].save(out_path, save_all=True, append_images=gif_frames[1:], optimize=False, duration=33, loop=0)
            self.finished_signal.emit(f"Saved to {out_path}")
        else:
            self.finished_signal.emit("Preview finished")
    
    def stop(self):
        self.running = False

class ScalableImageLabel(QLabel):
    clicked = pyqtSignal(int, int)
    drawn = pyqtSignal(int, int)
    shape_completed = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.setAlignment(Qt.AlignCenter)
        self._pixmap = None
        self.drawing = False

    def setPixmap(self, pixmap):
        self._pixmap = pixmap
        self.update_display()

    def resizeEvent(self, event):
        self.update_display()
        super().resizeEvent(event)
        
    def _get_coords(self, event):
        if self._pixmap and not self._pixmap.isNull():
            lbl_size = self.size()
            scaled_pixmap = self._pixmap.scaled(lbl_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            
            dx = (lbl_size.width() - scaled_pixmap.width()) / 2
            dy = (lbl_size.height() - scaled_pixmap.height()) / 2
            
            click_x = event.x() - dx
            click_y = event.y() - dy
            
            if 0 <= click_x < scaled_pixmap.width() and 0 <= click_y < scaled_pixmap.height():
                orig_w = self._pixmap.width()
                orig_h = self._pixmap.height()
                
                scale_x = orig_w / scaled_pixmap.width()
                scale_y = orig_h / scaled_pixmap.height()
                
                return int(click_x * scale_x), int(click_y * scale_y)
        return None

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            coords = self._get_coords(event)
            if coords:
                self.clicked.emit(*coords)
                self.drawn.emit(*coords)
        super().mousePressEvent(event)
        
    def mouseMoveEvent(self, event):
        if self.drawing:
            coords = self._get_coords(event)
            if coords:
                self.drawn.emit(*coords)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False
            self.shape_completed.emit()
        super().mouseReleaseEvent(event)

    def update_display(self):
        if self._pixmap and not self._pixmap.isNull():
            scaled = self._pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            super().setPixmap(scaled)
        else:
            super().setPixmap(QPixmap())

class MediaPanel(QFrame):
    media_loaded = pyqtSignal(str)
    media_cleared = pyqtSignal()

    def __init__(self, title, is_target=False):
        super().__init__()
        self.file_path = None
        self.is_target = is_target
        self.rotate_steps = 0
        self.is_flipped = False
        self.is_analyzing = False
        self.segments = None 
        self.selected_segments = set()
        self.pen_mode = None
        self.shapes = []
        self.current_shape = []
        self.manual_mask = None
        self.setup_ui(title)
        
    def setup_ui(self, title):
        self.setStyleSheet(get_panel_style())
        self.setObjectName("mediaPanel")
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)
        
        title_lbl = QLabel(title)
        title_lbl.setStyleSheet(get_title_label_style())
        title_lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_lbl)
        
        tools_layout = QHBoxLayout()
        
        self.analyze_btn = QPushButton("Analyze Shapes")
        self.analyze_btn.setStyleSheet(get_surface_button_style() + "border: 1px solid #FFEB3B; color: #FFEB3B;")
        self.analyze_btn.setCursor(Qt.PointingHandCursor)
        self.analyze_btn.clicked.connect(self.analyze_shapes)
        self.analyze_btn.setVisible(False) 
        
        self.pen_btn = QPushButton("Pen")
        self.pen_btn.setStyleSheet(get_surface_button_style())
        self.pen_btn.setCursor(Qt.PointingHandCursor)
        self.pen_menu = QMenu()
        self.pen_menu.setStyleSheet(f"""
            QMenu {{
                background-color: {THEME['surface']};
                color: {THEME['text']};
                border: 1px solid {THEME['border_light']};
                border-radius: 4px;
                padding: 4px;
            }}
            QMenu::item {{
                padding: 6px 16px;
                border-radius: 2px;
            }}
            QMenu::item:selected {{
                background-color: {THEME['surface_hover']};
            }}
        """)
        self.pen_menu.addAction("+ (Include)", lambda: self.set_pen_mode('plus'))
        self.pen_menu.addAction("- (Exclude)", lambda: self.set_pen_mode('minus'))
        self.pen_btn.setMenu(self.pen_menu)
        self.pen_btn.setVisible(False)
        
        tools_layout.addWidget(self.analyze_btn)
        tools_layout.addWidget(self.pen_btn)
        
        self.preview = ScalableImageLabel()
        self.preview.clicked.connect(self.on_preview_clicked)
        self.preview.drawn.connect(self.on_preview_drawn)
        self.preview.shape_completed.connect(self.on_shape_completed)
        self.preview.setStyleSheet(f"""
            background-color: {THEME['surface']};
            border: 1px solid {THEME['border']};
            border-radius: 4px;
        """)
        self.preview.setMinimumSize(200, 200)
        
        layout.addLayout(tools_layout)
        layout.addWidget(self.preview, stretch=1)
        
        self.info_lbl = QLabel("No media loaded")
        self.info_lbl.setStyleSheet(get_subtitle_label_style())
        self.info_lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.info_lbl)
        
        btn_layout = QHBoxLayout()
        
        self.add_btn = QPushButton("Add Media")
        self.add_btn.setStyleSheet(get_main_button_style())
        self.add_btn.setCursor(Qt.PointingHandCursor)
        self.add_btn.clicked.connect(self.load_media)
        
        self.remove_btn = QPushButton("Remove")
        self.remove_btn.setStyleSheet(get_surface_button_style())
        self.remove_btn.setCursor(Qt.PointingHandCursor)
        self.remove_btn.clicked.connect(self.clear_media)
        self.remove_btn.setEnabled(False)
        
        btn_layout.addWidget(self.add_btn)
        btn_layout.addWidget(self.remove_btn)
        layout.addLayout(btn_layout)
        
        manip_layout = QHBoxLayout()
        
        self.rotate_btn = QPushButton("Rotate")
        self.rotate_btn.setStyleSheet(get_surface_button_style())
        self.rotate_btn.setCursor(Qt.PointingHandCursor)
        self.rotate_btn.clicked.connect(self.rotate_media)
        self.rotate_btn.setEnabled(False)
        
        self.flip_btn = QPushButton("Flip")
        self.flip_btn.setStyleSheet(get_surface_button_style())
        self.flip_btn.setCursor(Qt.PointingHandCursor)
        self.flip_btn.clicked.connect(self.flip_media)
        self.flip_btn.setEnabled(False)
        
        manip_layout.addWidget(self.rotate_btn)
        manip_layout.addWidget(self.flip_btn)
        layout.addLayout(manip_layout)

    def set_pen_mode(self, mode):
        self.pen_mode = mode
        self.shapes = []
        self.current_shape = []
        self.manual_mask = None
        self.is_analyzing = False
        self.segments = None
        self.selected_segments = set()
        self.update_preview()
        
    def on_preview_drawn(self, x, y):
        if self.pen_mode and not self.is_analyzing:
            self.current_shape.append((x, y))
            self.update_preview()
    
    def on_shape_completed(self):
        if self.pen_mode and not self.is_analyzing and len(self.current_shape) > 0:
            self.shapes.append(self.current_shape.copy())
            self.current_shape = []
            self.info_lbl.setText(f"Shape {len(self.shapes)} completed. Draw another or click Analyze.")
    
    def analyze_shapes(self):
        if not self.file_path: return
        
        try:
            img = cv2.imread(self.file_path)
            img = apply_opencv_transforms(img, self.rotate_steps, self.is_flipped)
            h, w = img.shape[:2]
            
            if self.pen_mode and (self.shapes or self.current_shape):
                if len(self.current_shape) > 0:
                    self.shapes.append(self.current_shape.copy())
                    self.current_shape = []
                
                mask = np.zeros((h, w), dtype=np.uint8) if self.pen_mode == 'plus' else np.ones((h, w), dtype=np.uint8) * 255
                color = 255 if self.pen_mode == 'plus' else 0
                
                for shape in self.shapes:
                    if len(shape) >= 3:
                        pts = np.array(shape, np.int32)
                        pts = pts.reshape((-1, 1, 2))
                        cv2.fillPoly(mask, [pts], color)
                
                self.manual_mask = (mask > 0)
                self.is_analyzing = True
                self.update_preview()
                shape_count = len(self.shapes)
                if shape_count <= 1:
                    self.info_lbl.setText("Shape analyzed.")
                else:
                    self.info_lbl.setText(f"{shape_count} shapes analyzed (combined mask, visually separate).")
                return

            self.proc_w, self.proc_h = 256, 256
            small = cv2.resize(img, (self.proc_w, self.proc_h))
            
            pil_img = Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
            pil_img = pil_img.filter(ImageFilter.SMOOTH_MORE)
            
            rgb = np.array(pil_img)
            pixel_values = rgb.reshape((-1, 3))
            pixel_values = np.float32(pixel_values)
            
            k = 6 
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            _, labels, _ = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            
            self.segments = labels.reshape((self.proc_h, self.proc_w))
            self.selected_segments = set() 
            self.is_analyzing = True
            
            self.update_preview()
            self.info_lbl.setText("Click shape to select (Green), others (Red)")
            
        except Exception as e:
            QMessageBox.warning(self, "Analysis Error", str(e))
    
    def stop_analysis(self):
        self.is_analyzing = False
        self.segments = None
        self.selected_segments = set()
        self.pen_mode = None
        self.shapes = []
        self.current_shape = []
        self.manual_mask = None
        self.update_preview()

    def on_preview_clicked(self, x, y):
        if not self.is_analyzing: return
        
        if self.manual_mask is not None:
            return
            
        if self.segments is not None and self.preview._pixmap:
            orig_w = self.preview._pixmap.width()
            orig_h = self.preview._pixmap.height()
            
            sx = int(x * (self.proc_w / orig_w))
            sy = int(y * (self.proc_h / orig_h))
            
            sx = max(0, min(sx, self.proc_w - 1))
            sy = max(0, min(sy, self.proc_h - 1))
            
            clicked_label = self.segments[sy, sx]
            
            if clicked_label in self.selected_segments:
                self.selected_segments.remove(clicked_label)
            else:
                self.selected_segments.add(clicked_label)
                
            self.update_preview()

    def get_mask(self):
        if self.manual_mask is not None:
            return self.manual_mask
            
        if self.segments is None or not self.selected_segments:
            return None
        
        mask = np.isin(self.segments, list(self.selected_segments))
        return mask

    def load_media(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Open Media", "", "Images (*.png *.jpg *.jpeg *.webp)")
        if fname:
            self.set_media_data(fname, 0, False)
            self.media_loaded.emit(fname)

    def set_media_data(self, path, rotate_steps, is_flipped):
        self.file_path = path
        self.rotate_steps = rotate_steps
        self.is_flipped = is_flipped
        self.is_analyzing = False 
        self.segments = None
        self.pen_mode = None
        self.shapes = []
        self.current_shape = []
        
        self.manual_mask = None
        
        if path:
            size = os.path.getsize(path) / (1024*1024)
            name = os.path.basename(path)
            
            img = cv2.imread(path)
            if img is not None:
                h, w = img.shape[:2]
                self.info_lbl.setText(f"{name} ({size:.1f} MB) {w}x{h}")
            else:
                self.info_lbl.setText(f"{name} ({size:.1f} MB)")
                
            self.add_btn.setText("Replace Media")
            self.remove_btn.setEnabled(True)
            self.rotate_btn.setEnabled(True)
            self.flip_btn.setEnabled(True)
            self.update_preview()
        else:
            self.clear_media()

    def update_preview(self):
        if not self.file_path:
            return
        
        try:
            img = cv2.imread(self.file_path)
            if img is not None:
                img = apply_opencv_transforms(img, self.rotate_steps, self.is_flipped)
                
                if self.is_analyzing:
                    if self.manual_mask is not None:
                        viz = img.copy()
                        overlay = np.zeros_like(img)
                        overlay[self.manual_mask] = [0, 255, 0]
                        overlay[~self.manual_mask] = [0, 0, 255]
                        
                        cv2.addWeighted(viz, 0.7, overlay, 0.3, 0, viz)
                        img = viz
                        
                    elif self.segments is not None:
                        overlay_img = cv2.resize(img, (256, 256))
                        viz = overlay_img.copy()
                        
                        if not self.selected_segments:
                            for label_id in np.unique(self.segments):
                                mask = (self.segments == label_id).astype(np.uint8)
                                cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                cv2.drawContours(viz, cnts, -1, (0, 255, 255), 2) 
                        else:
                            for label_id in np.unique(self.segments):
                                mask = (self.segments == label_id).astype(np.uint8)
                                cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                
                                if label_id in self.selected_segments:
                                    cv2.drawContours(viz, cnts, -1, (0, 255, 0), 2) 
                                else:
                                    cv2.drawContours(viz, cnts, -1, (0, 0, 255), 1) 
                        img = viz
                        
                elif self.pen_mode and (self.shapes or self.current_shape):
                    viz = img.copy()
                    color = (0, 255, 0) if self.pen_mode == 'plus' else (0, 0, 255)
                    
                    for shape in self.shapes:
                        if len(shape) >= 2:
                            pts = np.array(shape, np.int32)
                            pts = pts.reshape((-1, 1, 2))
                            cv2.polylines(viz, [pts], False, color, 2)
                    
                    if len(self.current_shape) >= 2:
                        pts = np.array(self.current_shape, np.int32)
                        pts = pts.reshape((-1, 1, 2))
                        cv2.polylines(viz, [pts], False, color, 2)
                    
                    img = viz

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w, ch = img.shape
                qimg = QImage(img.data, w, h, w*ch, QImage.Format_RGB888)
                self.preview.setPixmap(QPixmap.fromImage(qimg))
            else:
                self.info_lbl.setText("Error loading image")
        except Exception as e:
            self.info_lbl.setText(f"Error: {str(e)}")

    def clear_media(self):
        self.file_path = None
        self.rotate_steps = 0
        self.is_flipped = False
        self.is_analyzing = False
        self.segments = None
        self.pen_mode = None
        self.shapes = []
        self.current_shape = []
        
        self.manual_mask = None
        self.preview.setPixmap(QPixmap())
        self.info_lbl.setText("No media loaded")
        self.add_btn.setText("Add Media")
        self.remove_btn.setEnabled(False)
        self.rotate_btn.setEnabled(False)
        self.flip_btn.setEnabled(False)
        self.media_cleared.emit()

    def rotate_media(self):
        if self.file_path:
            self.rotate_steps = (self.rotate_steps + 1) % 4
            self.update_preview()

    def flip_media(self):
        if self.file_path:
            self.is_flipped = not self.is_flipped
            self.update_preview()

class ImderGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Imder - Image Blender")
        self.resize(1100, 750)
        
        self.setStyleSheet(get_window_style())
        
        app_icon = load_app_icon()
        if app_icon:
            self.setWindowIcon(app_icon)
            QApplication.setWindowIcon(app_icon)
        
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(16)
        
        header_layout = QHBoxLayout()
        header_layout.setSpacing(12)
        
        header_lbl = QLabel("Mode:")
        header_lbl.setStyleSheet(get_subtitle_label_style())
        
        self.mode_combo = QComboBox()
        self.mode_combo.addItems([
            "Shuffle", 
            "Merge", 
            "Missform",
            "Fusion", 
            "Pattern", 
            "Disguise", 
            "Navigate", 
            "Swap", 
            "Blend"
        ])
        self.mode_combo.setToolTip("Select processing mode. All modes work with Images.")
        self.mode_combo.setStyleSheet(get_combo_box_style())
        self.mode_combo.currentTextChanged.connect(self.on_mode_changed)
        
        header_layout.addWidget(header_lbl)
        header_layout.addWidget(self.mode_combo)
        
        res_lbl = QLabel("Resolution:")
        res_lbl.setStyleSheet(get_subtitle_label_style())
        
        self.res_combo = QComboBox()
        self.res_combo.addItems(["128x128", "256x256", "512x512", "768x768", "1024x1024", "2048x2048"])
        self.res_combo.setStyleSheet(get_combo_box_style())
        self.res_combo.setMinimumWidth(120)
        
        header_layout.addWidget(res_lbl)
        header_layout.addWidget(self.res_combo)
        
        header_layout.addStretch()
        main_layout.addLayout(header_layout)
        
        panels_layout = QHBoxLayout()
        panels_layout.setSpacing(16)
        
        self.base_panel = MediaPanel("Base Image")
        
        self.preview_panel = QFrame()
        self.preview_panel.setStyleSheet(get_panel_style())
        p_layout = QVBoxLayout(self.preview_panel)
        p_layout.setContentsMargins(12, 12, 12, 12)
        p_layout.setSpacing(10)
        
        self.reverse_btn = QPushButton("Reverse Swap")
        self.reverse_btn.setStyleSheet(get_surface_button_style())
        self.reverse_btn.setCursor(Qt.PointingHandCursor)
        self.reverse_btn.clicked.connect(self.swap_media)
        self.reverse_btn.setEnabled(False)
        p_layout.addWidget(self.reverse_btn)
        
        p_lbl = QLabel("Animation Preview")
        p_lbl.setStyleSheet(get_title_label_style())
        p_lbl.setAlignment(Qt.AlignCenter)
        p_layout.addWidget(p_lbl)
        
        self.preview_display = ScalableImageLabel()
        self.preview_display.setStyleSheet(f"""
            background-color: {THEME['surface']};
            border: 1px solid {THEME['border']};
            border-radius: 4px;
        """)
        self.preview_display.setMinimumSize(200, 200)
        p_layout.addWidget(self.preview_display, stretch=1)
        
        self.target_panel = MediaPanel("Target Image", is_target=True)
        self.target_panel.setEnabled(False)
        
        panels_layout.addWidget(self.base_panel, stretch=1)
        panels_layout.addWidget(self.preview_panel, stretch=1)
        panels_layout.addWidget(self.target_panel, stretch=1)
        
        main_layout.addLayout(panels_layout)
        
        controls = QHBoxLayout()
        controls.setSpacing(12)
        
        self.start_btn = QPushButton("Start Processing")
        self.start_btn.setStyleSheet(get_main_button_style())
        self.start_btn.setCursor(Qt.PointingHandCursor)
        
        self.replay_btn = QPushButton("Replay")
        self.replay_btn.setStyleSheet(get_secondary_button_style())
        self.replay_btn.setCursor(Qt.PointingHandCursor)
        self.replay_menu = QMenu()
        self.replay_menu.setStyleSheet(f"""
            QMenu {{
                background-color: {THEME['surface']};
                color: {THEME['text']};
                border: 1px solid {THEME['border_light']};
                border-radius: 4px;
                padding: 4px;
            }}
            QMenu::item {{
                padding: 6px 16px;
                border-radius: 2px;
            }}
            QMenu::item:selected {{
                background-color: {THEME['surface_hover']};
            }}
        """)
        self.replay_menu.addAction("Show Animate", lambda: self.run_replay(False))
        self.replay_menu.addAction("Reverse Animate", lambda: self.run_replay(True))
        self.replay_btn.setMenu(self.replay_menu)
        
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setStyleSheet(get_surface_button_style())
        self.stop_btn.setCursor(Qt.PointingHandCursor)
        
        self.export_btn = QPushButton("Export")
        self.export_btn.setStyleSheet(get_secondary_button_style())
        self.export_btn.setCursor(Qt.PointingHandCursor)
        self.export_menu = QMenu()
        self.export_menu.setStyleSheet(f"""
            QMenu {{
                background-color: {THEME['surface']};
                color: {THEME['text']};
                border: 1px solid {THEME['border_light']};
                border-radius: 4px;
                padding: 4px;
            }}
            QMenu::item {{
                padding: 6px 16px;
                border-radius: 2px;
            }}
            QMenu::item:selected {{
                background-color: {THEME['surface_hover']};
            }}
        """)
        self.export_btn.setMenu(self.export_menu)
        
        controls.addWidget(self.start_btn)
        controls.addWidget(self.replay_btn)
        controls.addWidget(self.stop_btn)
        controls.addWidget(self.export_btn)
        
        self.stop_btn.setEnabled(False)
        self.replay_btn.setEnabled(False)
        self.export_btn.setEnabled(False)
        self.start_btn.setEnabled(False)
        
        main_layout.addLayout(controls)
        
        self.status_bar = QLabel("Ready")
        self.status_bar.setStyleSheet(get_status_bar_style())
        main_layout.addWidget(self.status_bar)
        
        self.progress = QProgressBar()
        self.progress.setStyleSheet(get_progress_bar_style())
        main_layout.addWidget(self.progress)
        
        self.base_panel.media_loaded.connect(self.on_base_loaded)
        self.base_panel.media_cleared.connect(self.on_base_cleared)
        self.target_panel.media_loaded.connect(self.on_target_loaded)
        self.target_panel.media_cleared.connect(self.check_ready)
        
        self.start_btn.clicked.connect(lambda: self.start_process('preview'))
        self.stop_btn.clicked.connect(self.stop_process)
        
        self.worker = None
        self.cached_frames = []

    def on_mode_changed(self, text):
        needs_analysis = "Pattern" in text or "Disguise" in text or "Navigate" in text or "Swap" in text or "Blend" in text or "Fusion" in text
        self.target_panel.analyze_btn.setVisible(needs_analysis)
        self.target_panel.pen_btn.setVisible(needs_analysis)
        if not needs_analysis:
            self.target_panel.stop_analysis()

    def swap_media(self):
        b_path = self.base_panel.file_path
        b_rot = self.base_panel.rotate_steps
        b_flip = self.base_panel.is_flipped
        
        t_path = self.target_panel.file_path
        t_rot = self.target_panel.rotate_steps
        t_flip = self.target_panel.is_flipped
        
        self.base_panel.set_media_data(t_path, t_rot, t_flip)
        self.target_panel.set_media_data(b_path, b_rot, b_flip)
        
        self.check_ready()

    def on_base_loaded(self, path):
        self.target_panel.setEnabled(True)
        self.check_ready()

    def on_base_cleared(self):
        self.target_panel.setEnabled(False)
        self.target_panel.clear_media()
        self.check_ready()

    def on_target_loaded(self, path):
        self.check_ready()

    def check_ready(self):
        ready = bool(self.base_panel.file_path and self.target_panel.file_path)
        self.start_btn.setEnabled(ready)
        self.export_btn.setEnabled(ready)
        self.reverse_btn.setEnabled(ready)
        
        if ready:
            self.export_menu.clear()
            self.export_menu.addAction("Frame", lambda: self.start_process('export_image'))
            self.export_menu.addAction("Animation", lambda: self.start_process('export_video'))
            self.export_menu.addAction("GIF", lambda: self.start_process('export_gif'))

    def validate(self):
        if not self.base_panel.file_path or not self.target_panel.file_path:
            return False
            
        mode_text = self.mode_combo.currentText()
        mode = mode_text.lower()
        
        if mode in ['pattern', 'disguise', 'navigate', 'swap', 'blend']:
            has_manual = self.target_panel.manual_mask is not None
            has_auto = self.target_panel.is_analyzing and self.target_panel.selected_segments
            
            if not has_manual and not has_auto:
                 QMessageBox.warning(self, "Selection Required", f"For {mode.title()} mode, please select a shape on Target image (via Analyze or Pen tool).")
                 return False
                
        return True

    def start_process(self, mode):
        if not self.validate():
            return
            
        self.status_bar.setText("Processing...")
        self.progress.setValue(0)
        self.set_processing_state(True)
        self.cached_frames = []
        
        b_trans = {'rotate': self.base_panel.rotate_steps, 'flip': self.base_panel.is_flipped}
        t_trans = {'rotate': self.target_panel.rotate_steps, 'flip': self.target_panel.is_flipped}
        
        algo_text = self.mode_combo.currentText()
        algo = algo_text.lower()
        
        if self.target_panel.manual_mask is not None:
            mask = self.target_panel.manual_mask
        elif algo == 'fusion':
            segments_mask = self.target_panel.get_mask()
            mask = segments_mask if segments_mask is not None else None
        else:
            mask = self.target_panel.get_mask() if algo in ['pattern', 'disguise', 'navigate', 'swap', 'blend'] else None
        
        res_text = self.res_combo.currentText()
        resolution = int(res_text.split('x')[0])
        
        sound_option = 'mute'
        audio_quality = 30
        
        self.worker = ProcessingThread(
            self.base_panel.file_path, 
            self.target_panel.file_path,
            mode,
            algo,
            b_trans,
            t_trans,
            mask,
            resolution,
            sound_option,
            audio_quality
        )
        self.worker.frame_signal.connect(self.update_preview)
        self.worker.progress_signal.connect(self.progress.setValue)
        self.worker.finished_signal.connect(self.process_finished)
        self.worker.error_signal.connect(self.process_error)
        self.worker.start()

    def stop_process(self):
        if self.worker:
            self.worker.stop()
            self.status_bar.setText("Stopping...")

    def update_preview(self, qimg):
        self.preview_display.setPixmap(QPixmap.fromImage(qimg))
        self.cached_frames.append(qimg.copy())

    def process_finished(self, msg):
        self.status_bar.setText(msg)
        self.set_processing_state(False)
        self.progress.setValue(100)
        self.replay_btn.setEnabled(True)

    def process_error(self, err):
        self.status_bar.setText(f"Error: {err}")
        QMessageBox.critical(self, "Error", err)
        self.set_processing_state(False)

    def set_processing_state(self, processing):
        self.start_btn.setEnabled(not processing)
        self.export_btn.setEnabled(not processing)
        self.base_panel.setEnabled(not processing)
        self.target_panel.setEnabled(not processing)
        self.stop_btn.setEnabled(processing)
        self.reverse_btn.setEnabled(not processing)
        self.mode_combo.setEnabled(not processing)
        self.res_combo.setEnabled(not processing)
        self.replay_btn.setEnabled(False)

    def run_replay(self, reverse):
        if not self.cached_frames: return
        
        frames = self.cached_frames[::-1] if reverse else self.cached_frames
        self.replay_index = 0
        self.replay_list = frames
        self.replay_timer = QTimer()
        self.replay_timer.timeout.connect(self.play_next_frame)
        self.replay_timer.start(33)

    def play_next_frame(self):
        if self.replay_index < len(self.replay_list):
            self.preview_display.setPixmap(QPixmap.fromImage(self.replay_list[self.replay_index]))
            self.replay_index += 1
        else:
            self.replay_timer.stop()

def print_banner():
    if HAS_PYFIGLET:
        banner = pyfiglet.figlet_format("IMDER", font="big")
        print("\033[97m" + banner + "\033[0m")
    else:
        print("""
██████  ███    ███ ██████  ███████ ██████  
  ██    ████  ████ ██   ██ ██      ██   ██
  ██    ██ ████ ██ ██   ██ █████   ██████  
  ██    ██  ██  ██ ██   ██ ██      ██   ██ 
██████  ██      ██ ██████  ███████ ██   ██
        """)
    print("=" * 60)
    print("Interactive CLI Mode - Image Blender Tool")
    print("=" * 60)

def get_input(prompt, validate_fn=None):
    while True:
        try:
            value = input(prompt).strip()
            if validate_fn:
                if validate_fn(value):
                    return value
                else:
                    print("Invalid input. Please try again.")
            else:
                return value
        except KeyboardInterrupt:
            print("\nOperation cancelled.")
            sys.exit(0)

def validate_file_exists(path):
    if os.path.exists(path):
        return True
    print(f"Error: File not found: {path}")
    return False

def validate_media_file(path):
    if not validate_file_exists(path):
        return False
    valid_extensions = ['.png', '.jpg', '.jpeg', '.webp', '.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    ext = os.path.splitext(path)[1].lower()
    if ext not in valid_extensions:
        print(f"Error: Invalid file format. Supported formats: {', '.join(valid_extensions)}")
        return False
    return True

def select_algorithm():
    print("\nSelect Algorithm:")
    print("1. Shuffle   - Randomly swap pixels between images")
    print("2. Merge     - Blend images with grayscale sorting")
    print("3. Missform  - Morph between binary pixel positions")
    print("4. Fusion    - Create animation with pixel sorting")
    
    while True:
        choice = input("\nEnter your choice (1-4): ").strip()
        if choice in ['1', '2', '3', '4']:
            algorithms = ['shuffle', 'merge', 'missform', 'fusion']
            return algorithms[int(choice) - 1]
        print("Invalid choice. Please enter 1, 2, 3, or 4.")

def select_resolution():
    print("\nSelect Resolution:")
    print("1. 128x128")
    print("2. 256x256")
    print("3. 512x512")
    print("4. 768x768")
    print("5. 1024x1024")
    print("6. 2048x2048")
    
    resolutions = [128, 256, 512, 768, 1024, 2048]
    while True:
        choice = input("\nEnter your choice (1-6): ").strip()
        if choice in ['1', '2', '3', '4', '5', '6']:
            return resolutions[int(choice) - 1]
        print("Invalid choice. Please enter 1, 2, 3, 4, 5, or 6.")

def select_sound_option(target_is_video=False):
    print("\nSelect Sound Option:")
    print("1. Mute (default)")
    print("2. Sound (generate audio from pixel colors)")
    
    if target_is_video:
        print("3. Target Sound (use audio from target video)")
    
    while True:
        choice = input("Enter your choice (1-2" + (", 3" if target_is_video else "") + ", default 1): ").strip()
        
        if choice == '':
            return 'mute', 30
        elif choice == '1':
            return 'mute', 30
        elif choice == '2':
            return 'sound', 30
        elif choice == '3' and target_is_video:
            return select_target_sound_quality()
        else:
            print(f"Invalid choice. Please enter 1, 2" + (", or 3" if target_is_video else "") + ".")

def select_target_sound_quality():
    print("\nSelect Target Sound Quality (1-10, where 10=100% original quality, 3=30% default):")
    print("1. 10% (lowest quality)")
    print("2. 20%")
    print("3. 30% (default)")
    print("4. 40%")
    print("5. 50%")
    print("6. 60%")
    print("7. 70%")
    print("8. 80%")
    print("9. 90%")
    print("10. 100% (original quality)")
    
    while True:
        choice = input("Enter your choice (1-10, default 3): ").strip()
        
        if choice == '':
            return 'target-sound', 30
        
        try:
            quality = int(choice)
            if 1 <= quality <= 10:
                quality_percent = quality * 10
                return 'target-sound', quality_percent
            else:
                print("Please enter a number between 1 and 10.")
        except ValueError:
            print("Please enter a valid number.")

def print_progress_bar(iteration, total, prefix='Progress:', length=50, fill='█'):
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} |{bar}| {percent}%')
    sys.stdout.flush()
    
    if iteration == total:
        print()

def cli_video_process(base_path, target_path, algo_mode, resolution, sound_option='mute', audio_quality=30):
    print(f"\nProcessing: {os.path.basename(base_path)} -> {os.path.basename(target_path)}")
    print(f"Algorithm: {algo_mode}")
    print(f"Resolution: {resolution}x{resolution}")
    print(f"Sound: {sound_option}" + (f" (quality: {audio_quality}%)" if sound_option == 'target-sound' else ""))
    print()
    
    output_dir = "results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    base_is_video = is_video_file(base_path)
    target_is_video = is_video_file(target_path)
    
    if base_is_video:
        print("Exploring Base frames:")
        base_frames, base_fps, base_count = extract_video_frames_with_progress(base_path, "Base")
    else:
        print("Base is image, 1 frame")
        base_img = cv2.imread(base_path)
        if base_img is None:
            print(f"Error: Could not load image {base_path}")
            sys.exit(1)
        base_frames = [base_img]
        base_fps = 30
        base_count = 1
    
    if target_is_video:
        print("\nExploring Target frames:")
        target_frames, target_fps, target_count = extract_video_frames_with_progress(target_path, "Target")
    else:
        print("Target is image, 1 frame")
        target_img = cv2.imread(target_path)
        if target_img is None:
            print(f"Error: Could not load image {target_path}")
            sys.exit(1)
        target_frames = [target_img]
        target_fps = 30
        target_count = 1
    
    if base_is_video and target_is_video:
        total_frames = min(base_count, target_count)
        print(f"\nShorter video has {total_frames} frames")
        
        if base_count > total_frames:
            print(f"\nExtracting first {total_frames} frames from Base:")
            base_frames = base_frames[:total_frames]
            for i in range(total_frames):
                if (i + 1) % max(1, total_frames // 20) == 0 or i == 0 or i == total_frames - 1:
                    percent = ((i + 1) / total_frames) * 100
                    print(f"Extracting frames from Base {i + 1}/{total_frames} {percent:.1f}%")
        
        if target_count > total_frames:
            print(f"\nExtracting first {total_frames} frames from Target:")
            target_frames = target_frames[:total_frames]
            for i in range(total_frames):
                if (i + 1) % max(1, total_frames // 20) == 0 or i == 0 or i == total_frames - 1:
                    percent = ((i + 1) / total_frames) * 100
                    print(f"Extracting frames from Target {i + 1}/{total_frames} {percent:.1f}%")
        
        if base_count * base_fps <= target_count * target_fps:
            fps = base_fps
            duration = total_frames / base_fps
        else:
            fps = target_fps
            duration = total_frames / target_fps
    
    elif base_is_video and not target_is_video:
        total_frames = base_count
        print(f"\nBase video has {total_frames} frames")
        print("Target is image, will be repeated for each frame")
        target_frames = [target_frames[0]] * total_frames
        fps = base_fps
        duration = total_frames / base_fps
    
    elif not base_is_video and target_is_video:
        total_frames = target_count
        print(f"\nTarget video has {total_frames} frames")
        print("Base is image, will be repeated for each frame")
        base_frames = [base_frames[0]] * total_frames
        fps = target_fps
        duration = total_frames / target_fps
    
    else:
        total_frames = 1
        fps = 30
        duration = 1.0
    
    print(f"\nVideo processing starts:")
    processed_frames = []
    
    for i in range(total_frames):
        percent = ((i + 1) / total_frames) * 100
        print(f"Frame {i + 1}/{total_frames} {percent:.1f}%")
        
        base_frame = base_frames[i]
        target_frame = target_frames[i]
        
        processed_frame = process_frame_pair(base_frame, target_frame, algo_mode, resolution)
        processed_frames.append(processed_frame)
        
        print(f"frame processing result .png saved in memory {i + 1}.png")
    
    print("\nframes processing finished")
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    if sound_option != 'mute':
        silent_path = os.path.join(output_dir, f"video_{timestamp}_silent.mp4")
        silent_path = os.path.abspath(silent_path)
        video_path = os.path.join(output_dir, f"video_{timestamp}.mp4")
        video_path = os.path.abspath(video_path)
        gif_path = os.path.join(output_dir, f"animation_{timestamp}.gif")
        
        print("\nCompiling silent video:")
        height, width = processed_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(silent_path, fourcc, fps, (width, height))
        
        for i, frame in enumerate(processed_frames):
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            percent = ((i + 1) / total_frames) * 100
            if (i + 1) % max(1, total_frames // 10) == 0 or i == 0 or i == total_frames - 1:
                print(f"Frame {i + 1}/{total_frames} = {percent:.1f}%")
        
        out.release()
        
        target_audio_path = target_path if sound_option == 'target-sound' and target_is_video else None
        final_video_path = add_audio_to_video(silent_path, processed_frames, fps, video_path, 
                                             sound_option, target_audio_path, audio_quality)
        if os.path.exists(silent_path):
            os.remove(silent_path)
        video_path = final_video_path
    else:
        video_path = os.path.join(output_dir, f"video_{timestamp}.mp4")
        video_path = os.path.abspath(video_path)
        gif_path = os.path.join(output_dir, f"animation_{timestamp}.gif")
        
        print("\nCompiling Frames to export:")
        height, width = processed_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
        
        for i, frame in enumerate(processed_frames):
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            percent = ((i + 1) / total_frames) * 100
            print(f"Frame {i + 1}/{total_frames} = {percent:.1f}%")
        
        out.release()
    
    print("\nexport result as gif:")
    gif_frames = []
    for i, frame in enumerate(processed_frames):
        gif_frames.append(Image.fromarray(frame))
        if (i + 1) % max(1, total_frames // 10) == 0 or i == 0 or i == total_frames - 1:
            percent = ((i + 1) / total_frames) * 100
            print(f"export result as gif {percent:.1f}%")
    
    if gif_frames:
        gif_frames[0].save(gif_path, save_all=True, append_images=gif_frames[1:], optimize=False, duration=int(1000/fps), loop=0)
    
    print(f"\nprocessing finished, results saved to results/")
    print(f"Video: {os.path.basename(video_path)}")
    print(f"GIF: {os.path.basename(gif_path)}")
    print(f"Duration: {duration:.1f}s, FPS: {fps:.1f}, Frames: {total_frames}")
    
    return processed_frames, fps, duration

def cli_image_process(base_path, target_path, algo_mode, resolution, sound_option='mute', audio_quality=30):
    output_dir = "results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    formats = [
        ('export_image', 'Frame (PNG)'),
        ('export_gif', 'GIF'),
        ('export_video', 'Animation (MP4)')
    ]
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    total_steps = 302
    
    for mode, desc in formats:
        print(f"Exporting {desc}...")
        
        worker = ProcessingThread(
            base_path, target_path, mode, algo_mode,
            {'rotate': 0, 'flip': False},
            {'rotate': 0, 'flip': False},
            None,
            resolution,
            sound_option,
            audio_quality
        )
        
        frames_processed = 0
        
        def on_progress(val):
            nonlocal frames_processed
            frames_processed = val
            print_progress_bar(val, 100, prefix=f'  {desc}:')
        
        def on_finish(msg):
            print(f"  {desc}: {msg}")
        
        def on_error(err):
            print(f"  Error: {err}")
        
        worker.progress_signal.connect(on_progress)
        worker.finished_signal.connect(on_finish)
        worker.error_signal.connect(on_error)
        worker.start()
        
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        
        while worker.isRunning():
            app.processEvents()
            time.sleep(0.01)
        
        print()

def cli_process_and_export(base_path, target_path, algo_mode, resolution, sound_option='mute', audio_quality=30):
    base_is_video = is_video_file(base_path)
    target_is_video = is_video_file(target_path)
    
    if sound_option == 'target-sound' and not target_is_video:
        print("Error: Target Sound option requires target to be a video file.")
        sys.exit(1)
    
    if base_is_video or target_is_video:
        if algo_mode == 'fusion':
            print("Error: Fusion algorithm cannot be used with video files.")
            sys.exit(1)
        if algo_mode not in ['merge', 'shuffle', 'missform']:
            print("Warning: Video processing only supports merge, shuffle, or missform algorithms. Using merge.")
            algo_mode = 'merge'
        cli_video_process(base_path, target_path, algo_mode, resolution, sound_option, audio_quality)
    else:
        cli_image_process(base_path, target_path, algo_mode, resolution, sound_option, audio_quality)

def interactive_cli_mode():
    while True:
        print_banner()
        
        print("\n--- Media Selection ---")
        base_path = get_input(
            "Enter base media path (or drag & drop file): ",
            validate_media_file
        )
        
        target_path = get_input(
            "Enter target media path (or drag & drop file): ",
            validate_media_file
        )
        
        base_is_video = is_video_file(base_path)
        target_is_video = is_video_file(target_path)
        
        if base_is_video or target_is_video:
            print("\nVideo mode detected. Select algorithm:")
            print("1. Merge (default)")
            print("2. Shuffle")
            print("3. Missform")
            choice = input("Enter your choice (1-3, default 1): ").strip()
            if choice == '2':
                algo_mode = 'shuffle'
            elif choice == '3':
                algo_mode = 'missform'
            else:
                algo_mode = 'merge'
            
            if base_is_video and target_is_video:
                print("Both files are videos. Will process frame-by-frame.")
            elif base_is_video:
                print("Base is video, target is image. Will process each frame with target image.")
            else:
                print("Base is image, target is video. Will process base image with each frame.")
        else:
            algo_mode = select_algorithm()
        
        resolution = select_resolution()
        
        sound_option, audio_quality = select_sound_option(target_is_video)
        
        print("\n--- Processing ---")
        cli_process_and_export(base_path, target_path, algo_mode, resolution, sound_option, audio_quality)
        
        print("\n--- What's Next? ---")
        print("1. Blend Again")
        print("2. Exit")
        
        while True:
            choice = input("\nEnter your choice (1-2): ").strip()
            if choice == '1':
                print("\n" + "=" * 60 + "\n")
                break
            elif choice == '2':
                print("\nThank you for using IMDER!")
                print("Results saved to: results/")
                sys.exit(0)
            print("Invalid choice. Please enter 1 or 2.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    is_cli = False
    arg_offset = 1
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'cli':
            if len(sys.argv) == 2:
                interactive_cli_mode()
                sys.exit(0)
            else:
                is_cli = True
                arg_offset = 2
        elif len(sys.argv) >= 3:
            is_cli = True
            arg_offset = 1
    
    if is_cli:
        if len(sys.argv) - arg_offset < 2:
            print("Error: Missing required arguments. Usage:")
            print("  python imder.py <base_path> <target_path> [algorithm] [resolution] [sound_option] [quality]")
            print("  python imder.py cli [interactive mode]")
            print("\nSound options: mute, sound, target-sound")
            print("Quality: 1-10 (only for target-sound, default 3)")
            sys.exit(1)
        
        base_path = sys.argv[arg_offset]
        target_path = sys.argv[arg_offset + 1]
        
        base_is_video = is_video_file(base_path)
        target_is_video = is_video_file(target_path)
        
        if base_is_video or target_is_video:
            algo_mode = 'merge'
            if len(sys.argv) - arg_offset >= 3:
                algo_mode = sys.argv[arg_offset + 2].lower()
                if algo_mode not in ['merge', 'shuffle', 'missform']:
                    print("Warning: Video processing only supports merge, shuffle, or missform. Using merge.")
                    algo_mode = 'merge'
        else:
            algo_mode = 'merge'
            if len(sys.argv) - arg_offset >= 3:
                algo_mode = sys.argv[arg_offset + 2].lower()
                valid_algos = ['shuffle', 'merge', 'missform', 'fusion']
                if algo_mode not in valid_algos:
                    print(f"Error: Invalid algorithm '{algo_mode}'. Valid options: {', '.join(valid_algos)}")
                    sys.exit(1)
        
        resolution = 512
        if len(sys.argv) - arg_offset >= 4:
            try:
                resolution = int(sys.argv[arg_offset + 3])
            except ValueError:
                print("Error: Resolution must be a number")
                sys.exit(1)
        
        sound_option = 'mute'
        audio_quality = 30
        if len(sys.argv) - arg_offset >= 5:
            sound_option = sys.argv[arg_offset + 4].lower()
            if sound_option == 'target-sound' and not target_is_video:
                print("Error: Target Sound option requires target to be a video file.")
                sys.exit(1)
            
            if sound_option not in ['mute', 'sound', 'target-sound']:
                print("Warning: Sound option must be 'mute', 'sound', or 'target-sound'. Using mute.")
                sound_option = 'mute'
            
            if len(sys.argv) - arg_offset >= 6 and sound_option == 'target-sound':
                try:
                    quality_input = int(sys.argv[arg_offset + 5])
                    if 1 <= quality_input <= 10:
                        audio_quality = quality_input * 10
                    else:
                        print("Warning: Quality must be between 1 and 10. Using default 3 (30%).")
                        audio_quality = 30
                except ValueError:
                    print("Warning: Quality must be a number. Using default 3 (30%).")
                    audio_quality = 30
            elif len(sys.argv) - arg_offset >= 6 and sound_option != 'target-sound':
                print("Warning: Quality parameter is only supported for 'target-sound' option. Ignoring.")
        
        print_banner()
        print(f"\nCLI Mode: Processing {os.path.basename(base_path)} -> {os.path.basename(target_path)}")
        print(f"Algorithm: {algo_mode}")
        print(f"Resolution: {resolution}x{resolution}")
        print(f"Sound: {sound_option}" + (f" (quality: {audio_quality}%)" if sound_option == 'target-sound' else ""))
        print()
        
        cli_process_and_export(base_path, target_path, algo_mode, resolution, sound_option, audio_quality)
        
        print(f"\nAll outputs saved to: results/")
        sys.exit(0)
    
    window = ImderGUI()
    window.show()
    sys.exit(app.exec_())
