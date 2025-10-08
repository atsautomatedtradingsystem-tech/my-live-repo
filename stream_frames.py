#!/usr/bin/env python3
"""
Простий фрейм-стрімер:
- намагається GET ${SERVER_URL:-http://127.0.0.1:8050}/snapshot.png
- якщо не вдається — малює fallback кадр з таймстампом
- шле rawvideo в ffmpeg, який паблішить на YT RTMP URL
"""
import os, time, subprocess, sys, io
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import requests

WIDTH = int(os.environ.get("STREAM_WIDTH", "640"))
HEIGHT = int(os.environ.get("STREAM_HEIGHT", "360"))
FPS = int(os.environ.get("STREAM_FPS", "10"))
BITRATE = os.environ.get("STREAM_BITRATE", "600k")
YT_KEY = os.environ.get("YT_KEY", "")   # наприклад v99b-xxxx-xxxx
YT_PRIMARY = os.environ.get("YT_PRIMARY", "rtmps://a.rtmp.youtube.com/live2")
YT_BACKUP  = os.environ.get("YT_BACKUP", "rtmps://b.rtmp.youtube.com/live2")
SERVER_URL = os.environ.get("SERVER_URL", "http://127.0.0.1:8050")
MAX_DURATION = int(os.environ.get("STREAM_MAX_SECONDS", "0"))  # 0 = без ліміту

if not YT_KEY:
    print("ERROR: YT_KEY not set (environment variable). Exiting.", file=sys.stderr)
    sys.exit(2)

rtmp_primary = f"{YT_PRIMARY.rstrip('/')}/{YT_KEY}"
rtmp_backup  = f"{YT_BACKUP.rstrip('/')}/{YT_KEY}"

FFMPEG_CMD = [
    "ffmpeg", "-hide_banner", "-loglevel", "info",
    "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", "-",
    "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
    "-c:v", "libx264", "-preset", "ultrafast", "-b:v", BITRATE, "-pix_fmt", "yuv420p", "-g", str(max(2,int(FPS*2))),
    "-c:a", "aac", "-b:a", "128k", "-ar", "44100",
    "-f", "flv", rtmp_primary,
    "-f", "flv", rtmp_backup + "?backup=1"
]

def get_snapshot_via_http(timeout=2.0):
    url = SERVER_URL.rstrip('/') + "/snapshot.png"
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code == 200:
            return Image.open(io.BytesIO(r.content)).convert("RGB").resize((WIDTH, HEIGHT))
    except Exception as e:
        # print("snapshot http error", e, file=sys.stderr)
        return None

def draw_fallback():
    img = Image.new("RGB", (WIDTH, HEIGHT), (20,20,20))
    d = ImageDraw.Draw(img)
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    try:
        fnt = ImageFont.load_default()
    except:
        fnt = None
    d.text((10,10), f"Fallback frame\n{ts}", fill=(200,200,200), font=fnt)
    return img

def main():
    start = time.time()
    p = subprocess.Popen(FFMPEG_CMD, stdin=subprocess.PIPE)
    frame_interval = 1.0 / FPS
    try:
        while True:
            if MAX_DURATION>0 and (time.time()-start) > MAX_DURATION:
                print("Max duration reached, exiting stream_frames", file=sys.stderr)
                break
            img = get_snapshot_via_http(timeout=max(0.5, frame_interval*0.8))
            if img is None:
                img = draw_fallback()
            img = img.resize((WIDTH, HEIGHT)).convert("RGB")
            arr = np.asarray(img)
            # rgb24 raw bytes
            p.stdin.write(arr.tobytes())
            # flush occasionally
            p.stdin.flush()
            time.sleep(frame_interval)
    except KeyboardInterrupt:
        print("Interrupted by user", file=sys.stderr)
    except BrokenPipeError:
        print("ffmpeg pipe broken", file=sys.stderr)
    finally:
        try:
            p.stdin.close()
        except:
            pass
        p.terminate()
        p.wait()

if __name__ == "__main__":
    main()
