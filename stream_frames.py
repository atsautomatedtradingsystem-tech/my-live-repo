#!/usr/bin/env python3
# stream_frames.py — fetch /snapshot.png and pipe to ffmpeg
import os, sys, time, subprocess, io, traceback
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import requests

# Config from env
WIDTH = int(os.environ.get("STREAM_WIDTH", 640))
HEIGHT = int(os.environ.get("STREAM_HEIGHT", 360))
FPS = float(os.environ.get("STREAM_FPS", 5))
BITRATE = os.environ.get("STREAM_BITRATE", "400k")
YT_KEY = os.environ.get("YT_KEY")
YT_PRIMARY = os.environ.get("YT_PRIMARY", "rtmp://a.rtmp.youtube.com/live2")
YT_BACKUP = os.environ.get("YT_BACKUP", "rtmp://b.rtmp.youtube.com/live2?backup=1")
SERVER_URL = os.environ.get("SERVER_URL", "http://127.0.0.1:8050")
LIVE_PY_CMD = os.environ.get("LIVE_PY_CMD", "python3 live.py")

if not YT_KEY:
    print("ERROR: YT_KEY environment variable not set", file=sys.stderr)
    # don't exit — allow local testing
PRIMARY_RTMP = f"{YT_PRIMARY.rstrip('/')}/{YT_KEY}"
BACKUP_RTMP  = f"{YT_BACKUP.rstrip('/')}/{YT_KEY}" if "backup=1" not in YT_BACKUP else f"{YT_BACKUP}&{YT_KEY}"

# ffmpeg command
gop = max(2, int(FPS*2))
ffmpeg_cmd = [
    "ffmpeg", "-hide_banner", "-loglevel", "info",
    "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{WIDTH}x{HEIGHT}", "-r", str(int(FPS)), "-i", "-",
    "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
    "-c:v", "libx264", "-preset", "ultrafast", "-b:v", BITRATE, "-pix_fmt", "yuv420p", "-g", str(gop),
    "-c:a", "aac", "-b:a", "128k", "-ar", "44100",
    "-f", "flv", PRIMARY_RTMP,
    "-f", "flv", BACKUP_RTMP
]

def get_snapshot_via_http(timeout=2):
    url = SERVER_URL.rstrip('/') + "/snapshot.png"
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert("RGB")
    except Exception:
        return None

def create_fallback_image():
    img = Image.new("RGB", (WIDTH, HEIGHT), (20, 20, 20))
    d = ImageDraw.Draw(img)
    t = time.strftime("%Y-%m-%d %H:%M:%S")
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    d.text((8, 8), f"NO SNAPSHOT\n{t}", fill=(255,255,255), font=font)
    return img

def main():
    print("Starting ffmpeg:", " ".join(ffmpeg_cmd), file=sys.stderr)
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
    frame_interval = 1.0 / max(1.0, FPS)
    try:
        while True:
            start = time.time()
            img = get_snapshot_via_http(timeout=max(0.5, frame_interval*0.8))
            if img is None:
                img = create_fallback_image()
            if img.size != (WIDTH, HEIGHT):
                img = img.resize((WIDTH, HEIGHT))
            arr = np.array(img)
            try:
                proc.stdin.write(arr.tobytes())
            except BrokenPipeError:
                print("ffmpeg pipe broken — exiting", file=sys.stderr)
                break
            # sleep to maintain FPS
            elapsed = time.time() - start
            to_sleep = frame_interval - elapsed
            if to_sleep > 0:
                time.sleep(to_sleep)
    except KeyboardInterrupt:
        print("Interrupted, terminating ffmpeg", file=sys.stderr)
    except Exception:
        traceback.print_exc()
    finally:
        try:
            proc.stdin.close()
        except Exception:
            pass
        proc.terminate()
        proc.wait(timeout=5)

if __name__ == "__main__":
    main()
