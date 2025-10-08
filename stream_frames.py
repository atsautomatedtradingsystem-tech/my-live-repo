#!/usr/bin/env python3
import os, sys, time, subprocess, io
from PIL import Image, ImageDraw, ImageFont
import requests

WIDTH = int(os.environ.get("STREAM_WIDTH", 640))
HEIGHT = int(os.environ.get("STREAM_HEIGHT", 360))
FPS = int(os.environ.get("STREAM_FPS", 10))
BITRATE = os.environ.get("STREAM_BITRATE", "600k")
YT_KEY = os.environ.get("YT_KEY")
YT_PRIMARY = os.environ.get("YT_PRIMARY", "rtmps://a.rtmp.youtube.com/live2")
YT_BACKUP  = os.environ.get("YT_BACKUP", "rtmps://b.rtmp.youtube.com/live2")

SERVER_URL = os.environ.get("SERVER_URL", "http://127.0.0.1:8050")
FRAME_INTERVAL = 1.0 / FPS

def get_snapshot_via_http(timeout=2.0):
    try:
        url = SERVER_URL.rstrip('/') + "/snapshot.png"
        r = requests.get(url, timeout=timeout)
        if r.status_code == 200:
            return Image.open(io.BytesIO(r.content)).convert("RGB")
    except Exception:
        return None

def make_fallback_image(text=None):
    img = Image.new("RGB", (WIDTH, HEIGHT), (30,30,30))
    d = ImageDraw.Draw(img)
    t = text or f"No snapshot — {time.strftime('%Y-%m-%d %H:%M:%S')}"
    try:
        fnt = ImageFont.load_default()
    except:
        fnt = None
    d.text((10,10), t, fill=(220,220,220), font=fnt)
    return img

def start_ffmpeg_process():
    args = [
        "ffmpeg","-hide_banner","-loglevel","info",
        "-f","rawvideo","-pix_fmt","rgb24",
        "-s","%dx%d" % (WIDTH, HEIGHT),
        "-r", str(FPS), "-i","-",
        "-f","lavfi","-i","anullsrc=channel_layout=stereo:sample_rate=44100",
        "-c:v","libx264","-preset","ultrafast","-b:v", BITRATE,
        "-pix_fmt","yuv420p","-g", str(int(FPS*2)),
        "-c:a","aac","-b:a","128k","-ar","44100",
        "-f","flv",
        f"{YT_PRIMARY}/{YT_KEY}",
        "-f","flv",
        f"{YT_BACKUP}/{YT_KEY}?backup=1"
    ]
    return subprocess.Popen(args, stdin=subprocess.PIPE)

def main():
    if not YT_KEY:
        print("ERROR: YT_KEY not set", file=sys.stderr)
        sys.exit(2)
    print("Starting ffmpeg...")
    proc = start_ffmpeg_process()
    try:
        while True:
            shot = get_snapshot_via_http(timeout=max(0.8, FRAME_INTERVAL*0.8))
            if shot is None:
                shot = make_fallback_image()
            shot = shot.resize((WIDTH, HEIGHT)).convert("RGB")
            proc.stdin.write(shot.tobytes())
            proc.stdin.flush()
            time.sleep(FRAME_INTERVAL)
    except KeyboardInterrupt:
        print("Interrupted, terminating ffmpeg...")
    finally:
        try:
            proc.stdin.close()
            proc.wait(timeout=3)
        except Exception:
            proc.kill()

if __name__ == "__main__":
    main()
