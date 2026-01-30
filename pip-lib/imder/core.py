import cv2
import numpy as np
import os
import hashlib
import wave
import subprocess
import tempfile
import shutil

def check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except:
        return False

def generate_sound(frame, duration, sr=44100):
    frame_bytes = frame.tobytes()
    frame_hash = hashlib.sha256(frame_bytes).hexdigest()
    np.random.seed(int(frame_hash[:8], 16))
    num_samples = int(sr * duration)
    t = np.linspace(0, duration, num_samples, False)
    freqs, amps = [], []
    for i in range(3):
        f = 50 + (int(frame_hash[i*4:(i+1)*4], 16) % 4000)
        a = 0.1 + (int(frame_hash[(i+3)*4:(i+4)*4], 16) % 9000) / 10000.0
        freqs.append(f)
        amps.append(a)
    sound = np.zeros(num_samples)
    for f, a in zip(freqs, amps):
        sound += a * np.sin(2 * np.pi * f * t)
    sound = sound / np.max(np.abs(sound)) if np.max(np.abs(sound)) > 0 else sound
    return (sound * 32767).astype(np.int16), frame_hash

def extract_audio(video_path, output_path, duration=None, quality_param=30, is_hz=False):
    if not check_ffmpeg():
        return False
    cmd = ["ffmpeg", "-i", video_path]
    if duration:
        cmd.extend(["-t", str(duration)])
    
    if is_hz:
        cmd.extend(["-ar", str(quality_param)])
    else:
        qmap = {10: "32k", 20: "64k", 30: "96k", 40: "128k", 50: "160k", 60: "192k", 70: "224k", 80: "256k", 90: "320k", 100: "copy"}
        br = qmap.get(quality_param, "96k")
        if br == "copy":
            cmd.extend(["-c:a", "copy"])
        else:
            cmd.extend(["-b:a", br])
    
    cmd.extend(["-y", output_path])
    try:
        subprocess.run(cmd, capture_output=True, check=True)
        return True
    except:
        return False

def add_audio(video_path, frames, fps, output_path, sound_opt, target_audio=None, quality_param=30, is_hz=False):
    if sound_opt == "mute" or not check_ffmpeg():
        return video_path
    tmp = tempfile.mkdtemp()
    try:
        if sound_opt == "target" and target_audio and os.path.exists(target_audio):
            ap = os.path.join(tmp, "a.mp3")
            dur = len(frames) / fps if frames else None
            if not extract_audio(target_audio, ap, dur, quality_param, is_hz):
                return video_path
        elif sound_opt == "gen":
            chunks = [generate_sound(f, 1.0/fps)[0] for f in frames]
            full = np.concatenate(chunks)
            ap = os.path.join(tmp, "a.wav")
            with wave.open(ap, "w") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(44100)
                w.writeframes(full.tobytes())
        else:
            return video_path
        
        cmd = ["ffmpeg", "-i", video_path, "-i", ap, "-c:v", "copy", "-c:a", "aac", 
               "-map", "0:v:0", "-map", "1:a:0", "-shortest", "-y", output_path]
        subprocess.run(cmd, capture_output=True, check=True)
        return output_path
    except:
        return video_path
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

class Missform:
    def __init__(self, base, target, threshold=127):
        self.base = base.astype(np.float32)
        self.target = target.astype(np.float32)
        self.thresh = threshold
        self._precompute()
    
    def _precompute(self):
        base_bin = ((np.mean(self.base, axis=2) > self.thresh) * 255).astype(np.uint8)
        tgt_bin = ((np.mean(self.target, axis=2) > self.thresh) * 255).astype(np.uint8)
        b_pos = np.column_stack(np.where(base_bin == 255))
        t_pos = np.column_stack(np.where(tgt_bin == 255))
        self.min_pos = min(len(b_pos), len(t_pos))
        if self.min_pos == 0:
            raise ValueError("No valid pixels")
        self.b_pos = b_pos[:self.min_pos]
        self.t_pos = t_pos[:self.min_pos]
        self.b_col = self.base[self.b_pos[:, 0], self.b_pos[:, 1]]
        self.h, self.w, _ = self.base.shape
    
    def generate(self, progress):
        frame = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        t = progress * progress * (3 - 2 * progress)
        cx = (self.b_pos[:, 0] + (self.t_pos[:, 0] - self.b_pos[:, 0]) * t).astype(int)
        cy = (self.b_pos[:, 1] + (self.t_pos[:, 1] - self.b_pos[:, 1]) * t).astype(int)
        mask = (cx >= 0) & (cx < self.h) & (cy >= 0) & (cy < self.w)
        if np.any(mask):
            frame[cx[mask], cy[mask]] = self.b_col[mask].astype(np.uint8)
        return frame

def is_video(path):
    return any(path.lower().endswith(e) for e in [".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"])

def extract_video_frames(video_path, max_frames=None):
    """Extract frames from video, optionally limited to max_frames"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30
    
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    limit = min(total, max_frames) if max_frames else total
    
    for i in range(limit):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    return frames, fps, len(frames)

def _compute_shuffle_assignments(bf, tf):
    n = len(tf)
    sg = np.mean(bf, axis=1)
    tg = np.mean(tf, axis=1)
    sb = sg > 127
    tb = tg > 127
    ix = np.arange(n)
    sblk, swht = ix[~sb], ix[sb]
    tblk, twht = ix[~tb], ix[tb]
    np.random.shuffle(sblk)
    np.random.shuffle(swht)
    np.random.shuffle(tblk)
    np.random.shuffle(twht)
    asgn = np.arange(n)
    mb = min(len(sblk), len(tblk))
    if mb:
        asgn[sblk[:mb]] = tblk[:mb]
    mw = min(len(swht), len(twht))
    if mw:
        asgn[swht[:mw]] = twht[:mw]
    rem_s = np.concatenate([sblk[mb:], swht[mw:]])
    rem_t = np.concatenate([tblk[mb:], twht[mw:]])
    if len(rem_s) and len(rem_t):
        np.random.shuffle(rem_s)
        np.random.shuffle(rem_t)
        asgn[rem_s] = rem_t
    return asgn

def _compute_merge_assignments(bf, tf):
    sg = np.mean(bf, axis=1)
    tg = np.mean(tf, axis=1)
    s_sort_idx = np.argsort(sg)
    t_sort_idx = np.argsort(tg)
    asgn = np.arange(len(bf))
    asgn[s_sort_idx] = t_sort_idx
    return asgn

def _generate_animated_frame(b, t, algo, res, progress):
    """Generate single frame with proper interpolation"""
    hb, wb = b.shape[:2]
    ht, wt = t.shape[:2]
    limit = min(hb, wb, ht, wt)
    r = min(res, limit)
    
    b_resized = cv2.resize(b, (r, r))
    t_resized = cv2.resize(t, (r, r))
    b_rgb = cv2.cvtColor(b_resized, cv2.COLOR_BGR2RGB)
    t_rgb = cv2.cvtColor(t_resized, cv2.COLOR_BGR2RGB)
    
    bf = b_rgb.reshape(-1, 3).astype(np.float32)
    tf = t_rgb.reshape(-1, 3).astype(np.float32)
    
    if algo == "shuffle":
        assignments = _compute_shuffle_assignments(bf, tf)
    else:
        assignments = _compute_merge_assignments(bf, tf)
    
    sx, sy = np.meshgrid(np.arange(r), np.arange(r))
    start_x = sx.flatten()
    start_y = sy.flatten()
    end_x = assignments % r
    end_y = assignments // r
    
    t_smooth = progress * progress * (3 - 2 * progress)
    
    curr_x = (start_x + (end_x - start_x) * t_smooth).astype(int)
    curr_y = (start_y + (end_y - start_y) * t_smooth).astype(int)
    
    curr_x = np.clip(curr_x, 0, r - 1)
    curr_y = np.clip(curr_y, 0, r - 1)
    
    if algo == "fusion":
        target_colors_aligned = tf[assignments]
        current_colors = (bf * (1 - t_smooth) + target_colors_aligned * t_smooth)
        current_colors = np.clip(current_colors, 0, 255).astype(np.uint8)
    else:
        current_colors = bf.astype(np.uint8)
    
    frame = np.zeros((r, r, 3), dtype=np.uint8)
    frame[curr_y, curr_x] = current_colors
    
    return frame

def process_pair(b, t, algo, res):
    """Process single pair (for PNG export)"""
    if algo == "missform":
        hb, wb = b.shape[:2]
        ht, wt = t.shape[:2]
        limit = min(hb, wb, ht, wt)
        r = min(res, limit)
        b_resized = cv2.resize(b, (r, r))
        t_resized = cv2.resize(t, (r, r))
        b_rgb = cv2.cvtColor(b_resized, cv2.COLOR_BGR2RGB)
        t_rgb = cv2.cvtColor(t_resized, cv2.COLOR_BGR2RGB)
        return Missform(b_rgb, t_rgb).generate(1.0)
    else:
        return _generate_animated_frame(b, t, algo, res, 1.0)

def generate_sequence(b, t, algo, res, frames=302):
    """Generate animation sequence (for GIF/MP4 export)"""
    seq = []
    
    if algo == "missform":
        hb, wb = b.shape[:2]
        ht, wt = t.shape[:2]
        limit = min(hb, wb, ht, wt)
        r = min(res, limit)
        b_resized = cv2.resize(b, (r, r))
        t_resized = cv2.resize(t, (r, r))
        b_rgb = cv2.cvtColor(b_resized, cv2.COLOR_BGR2RGB)
        t_rgb = cv2.cvtColor(t_resized, cv2.COLOR_BGR2RGB)
        m = Missform(b_rgb, t_rgb)
        
        for i in range(frames):
            p = i / max(1, frames - 1)
            seq.append(m.generate(p))
    else:
        for i in range(frames):
            p = i / max(1, frames - 1)
            seq.append(_generate_animated_frame(b, t, algo, res, p))
    
    return seq
