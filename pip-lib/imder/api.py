import os
import cv2
import time
import numpy as np
from PIL import Image
from .core import (
    process_pair, is_video, generate_sequence, add_audio, extract_video_frames
)

def process(base, target, result, results, algo, res, sound, sq=None, sq_hz=None):
    b = os.path.abspath(base)
    t = os.path.abspath(target)
    r = os.path.abspath(result)
    
    if not os.path.exists(b):
        raise FileNotFoundError(f"Base file not found: {base}")
    if not os.path.exists(t):
        raise FileNotFoundError(f"Target file not found: {target}")
    if not os.path.exists(r):
        os.makedirs(r)
    
    valid_results = ['png', 'gif', 'mp4']
    if not results:
        raise ValueError("Results list cannot be empty")
    for fmt in results:
        if fmt.lower() not in valid_results:
            raise ValueError(f"Invalid format '{fmt}'. Valid: png, gif, mp4")
    
    valid_sounds = ['mute', 'gen', 'target']
    if sound not in valid_sounds:
        raise ValueError(f"Invalid sound option '{sound}'. Valid: mute, gen, target")
    
    if not isinstance(res, int) or res < 1 or res > 16384:
        raise ValueError("Resolution must be integer between 1 and 16384")
    
    if sq is not None and sq_hz is not None:
        raise ValueError("Cannot use both sq and sq_hz. Choose one.")
    
    if sq is not None:
        if not isinstance(sq, int) or sq < 1 or sq > 10:
            raise ValueError("SQ must be integer between 1 and 10")
        sq = sq * 10
    
    if sq_hz is not None:
        if sound != "target":
            raise ValueError("sq_hz only valid with sound='target'")
        if not isinstance(sq_hz, int) or sq_hz < 8000 or sq_hz > 192000:
            raise ValueError("sq_hz must be integer between 8000 and 192000")
    
    bv = is_video(b)
    tv = is_video(t)
    
    if bv or tv:
        if algo not in ["shuffle", "merge", "missform"]:
            raise ValueError(f"Video only supports: shuffle, merge, missform")
        if "png" in [x.lower() for x in results]:
            raise ValueError("PNG not supported for video input")
    else:
        if algo not in ["shuffle", "merge", "missform", "fusion"]:
            raise ValueError(f"Valid algorithms: shuffle, merge, missform, fusion")
    
    if sound == "target" and not tv:
        raise ValueError("Target sound requires video target")
    
    ts = time.strftime("%Y%m%d_%H%M%S")
    outs = []
    
    if bv or tv:
        # NEW: 1:1 video frame trimming when both inputs are videos
        if bv and tv:
            # Extract all frames from both videos
            b_frames, b_fps, _ = extract_video_frames(b)
            t_frames, t_fps, _ = extract_video_frames(t)
            
            # Use the SHORTER duration (1:1 frame matching)
            min_frames = min(len(b_frames), len(t_frames))
            b_frames = b_frames[:min_frames]
            t_frames = t_frames[:min_frames]
            
            # Use fps from the video that wasn't trimmed (or base if equal)
            fps = b_fps if len(b_frames) >= len(t_frames) else t_fps
            
            # Process pairs
            frames = []
            for bf, tf in zip(b_frames, t_frames):
                frames.append(process_pair(bf, tf, algo, res))
        else:
            # One is image, one is video (original logic)
            bc = cv2.VideoCapture(b) if bv else None
            tc = cv2.VideoCapture(t) if tv else None
            
            fcount = min(
                int(bc.get(cv2.CAP_PROP_FRAME_COUNT)) if bc else 1,
                int(tc.get(cv2.CAP_PROP_FRAME_COUNT)) if tc else 1
            )
            fps = 30
            if bc:
                fps = bc.get(cv2.CAP_PROP_FPS) or 30
            elif tc:
                fps = tc.get(cv2.CAP_PROP_FPS) or 30
            
            frames = []
            for _ in range(fcount):
                if bc:
                    ret, bf = bc.read()
                    if not ret:
                        break
                else:
                    bf = cv2.imread(b)
                    if bf is None:
                        raise ValueError(f"Failed to load base image: {b}")
                if tc:
                    ret, tf = tc.read()
                    if not ret:
                        break
                else:
                    tf = cv2.imread(t)
                    if tf is None:
                        raise ValueError(f"Failed to load target image: {t}")
                frames.append(process_pair(bf, tf, algo, res))
            
            if bc:
                bc.release()
            if tc:
                tc.release()
        
        if not frames:
            raise ValueError("No frames were processed. Check video files.")
        
        for fmt in results:
            fmt = fmt.lower().strip()
            if fmt == "gif":
                p = os.path.join(r, f"imder_{ts}.gif")
                imgs = [Image.fromarray(x) for x in frames]
                imgs[0].save(p, save_all=True, append_images=imgs[1:], duration=int(1000/fps), loop=0)
                outs.append(p)
            elif fmt == "mp4":
                p = os.path.join(r, f"imder_{ts}.mp4")
                tp = p.replace(".mp4", "_t.mp4")
                h, w = frames[0].shape[:2]
                out = cv2.VideoWriter(tp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                for f in frames:
                    out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
                out.release()
                if sound != "mute":
                    quality_param = sq_hz if sq_hz is not None else (sq if sq is not None else 30)
                    is_hz = sq_hz is not None
                    fp = add_audio(tp, frames, fps, p, sound, t if sound=="target" else None, quality_param, is_hz)
                    if os.path.exists(tp) and fp != tp:
                        os.remove(tp)
                    outs.append(fp)
                else:
                    os.rename(tp, p)
                    outs.append(p)
    else:
        # Image processing (unchanged)
        bimg = cv2.imread(b)
        if bimg is None:
            raise ValueError(f"Failed to load base image: {b}")
        timg = cv2.imread(t)
        if timg is None:
            raise ValueError(f"Failed to load target image: {t}")
        
        for fmt in results:
            fmt = fmt.lower().strip()
            if fmt == "png":
                p = os.path.join(r, f"imder_{ts}.png")
                f = process_pair(bimg, timg, algo, res)
                cv2.imwrite(p, cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
                outs.append(p)
            elif fmt in ["gif", "mp4"]:
                seq = generate_sequence(bimg, timg, algo, res)
                if fmt == "gif":
                    p = os.path.join(r, f"imder_{ts}.gif")
                    imgs = [Image.fromarray(x) for x in seq]
                    imgs[0].save(p, save_all=True, append_images=imgs[1:], duration=33, loop=0)
                    outs.append(p)
                else:
                    p = os.path.join(r, f"imder_{ts}.mp4")
                    tp = p.replace(".mp4", "_t.mp4")
                    h, w = seq[0].shape[:2]
                    out = cv2.VideoWriter(tp, cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
                    for f in seq:
                        out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
                    out.release()
                    if sound != "mute":
                        quality_param = sq_hz if sq_hz is not None else (sq if sq is not None else 30)
                        is_hz = sq_hz is not None
                        ta = t if sound=="target" and tv else None
                        fp = add_audio(tp, seq, 30, p, sound, ta, quality_param, is_hz)
                        if os.path.exists(tp) and fp != tp:
                            os.remove(tp)
                        outs.append(fp)
                    else:
                        os.rename(tp, p)
                        outs.append(p)
    return outs
