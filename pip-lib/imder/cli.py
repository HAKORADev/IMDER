import os
import sys
from .api import process
from .core import is_video

try:
    import pyfiglet
    HASF = True
except:
    HASF = False

def banner():
    if HASF:
        print(pyfiglet.figlet_format("IMDER", font="big"))
    else:
        print("IMDER")
    print("=" * 60)

def interactive():
    while True:
        banner()
        b = input("Base: ").strip()
        while not os.path.exists(b):
            print("Not found")
            b = input("Base: ").strip()
        t = input("Target: ").strip()
        while not os.path.exists(t):
            print("Not found")
            t = input("Target: ").strip()
        
        bv = is_video(b)
        tv = is_video(t)
        
        print("\nAlgorithm:")
        if bv or tv:
            opts = ["merge", "shuffle", "missform"]
        else:
            opts = ["shuffle", "merge", "missform", "fusion"]
        for i, o in enumerate(opts, 1):
            print(f"{i}. {o}")
        c = input("Select: ").strip()
        algo = opts[int(c)-1] if c.isdigit() and 0<int(c)<=len(opts) else opts[0]
        
        print("\nResolution (128/256/512/1024/2048):")
        res = int(input("Res: ").strip() or "512")
        
        print("\nSound (mute/gen" + ("/target" if tv else "") + "):")
        snd = input("Sound: ").strip() or "mute"
        
        sq = None
        if snd == "target":
            sq = int(input("SQ (1-10): ").strip() or "3")
        
        print("\nResults (space separated):")
        if bv or tv:
            print("Valid: gif mp4")
        else:
            print("Valid: png gif mp4")
        rlst = input("Formats: ").strip().split()
        
        out = input("Result folder: ").strip() or "results"
        
        print("\nProcessing...")
        try:
            files = process(b, t, out, rlst, algo, res, snd, sq)
            print("Done:")
            for f in files:
                print(f"  {f}")
        except Exception as e:
            print(f"Error: {e}")
        
        print("\n1. Again\n2. Exit")
        if input("Choice: ").strip() == "2":
            break

def launch_interactive():
    interactive()

def main():
    if len(sys.argv) == 1:
        interactive()
    else:
        import argparse
        p = argparse.ArgumentParser()
        p.add_argument("base")
        p.add_argument("target")
        p.add_argument("result")
        p.add_argument("--results", nargs="+", required=True)
        p.add_argument("--algo", default="merge")
        p.add_argument("--res", type=int, default=512)
        p.add_argument("--sound", default="mute")
        p.add_argument("--sq", type=int, default=None)
        a = p.parse_args()
        files = process(a.base, a.target, a.result, a.results, a.algo, a.res, a.sound, a.sq)
        for f in files:
            print(f)
