import os
import cv2
import time
import numpy as np
from PIL import Image
from .core import process_pair, is_video, generate_sequence, add_audio

def process(base, target, result, results, algo, res, sound, sq=None):
    b = os.path.abspath(base)
    t = os.path.abspath(target)
    r = os.path.abspath(result)
    
    if not os.path.exists(b): raise FileNotFoundError(base)
    if not os.path.exists(t): raise FileNotFoundError(t)
    if not os.path.exists(r): os.makedirs(r)
    
    bv = is_video(b)
    tv = is_video(t)
    
    if bv or tv:
        if algo not in ["shuffle", "merge", "missform"]:
            raise ValueError(f"Video supports: shuffle, merge, missform")
        if "png" in [x.lower() for x in results]:
            raise ValueError("PNG not supported for video input")
    else:
        if algo not in ["shuffle", "merge", "missform", "fusion"]:
            raise ValueError(f"Valid algorithms: shuffle, merge, missform, fusion")
    
    if sound == "target" and not tv:
        raise ValueError("Target sound requires video target")
    if sq and sound != "target":
        raise ValueError("SQ only valid with target sound")
    if sq:
        sq = int(sq) * 10
        if not 10 <= sq <= 100:
            raise ValueError("SQ must be 1-10")
    
    ts = time.strftime("%Y%m%d_%H%M%S")
    outs = []
    
    if bv or tv:
        bc = cv2.VideoCapture(b) if bv else None
        tc = cv2.VideoCapture(t) if tv else None
        
        fcount = min(
            int(bc.get(cv2.CAP_PROP_FRAME_COUNT)) if bc else 1,
            int(tc.get(cv2.CAP_PROP_FRAME_COUNT)) if tc else 1
        )
        fps = 30
        if bc: fps = bc.get(cv2.CAP_PROP_FPS) or 30
        elif tc: fps = tc.get(cv2.CAP_PROP_FPS) or 30
        
        frames = []
        for _ in range(fcount):
            if bc:
                ret, bf = bc.read()
                if not ret: break
            else:
                bf = cv2.imread(b)
            if tc:
                ret, tf = tc.read()
                if not ret: break
            else:
                tf = cv2.imread(t)
            frames.append(process_pair(bf, tf, algo, res))
        
        if bc: bc.release()
        if tc: tc.release()
        
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
                    fp = add_audio(tp, frames, fps, p, sound, t if sound=="target" else None, sq or 30)
                    if os.path.exists(tp) and fp != tp:
                        os.remove(tp)
                    outs.append(fp)
                else:
                    os.rename(tp, p)
                    outs.append(p)
    else:
        bimg = cv2.imread(b)
        timg = cv2.imread(t)
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
                        ta = t if sound=="target" and tv else None
                        fp = add_audio(tp, seq, 30, p, sound, ta, sq or 30)
                        if os.path.exists(tp) and fp != tp:
                            os.remove(tp)
                        outs.append(fp)
                    else:
                        os.rename(tp, p)
                        outs.append(p)
    return outs
