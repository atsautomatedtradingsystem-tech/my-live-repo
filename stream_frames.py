#!/usr/bin/env python3
"""
stream_frames.py
Запитує SERVER_URL/snapshot.png і шле кадри в ffmpeg для RTMP (YouTube).
"""

import os, sys, time, subprocess, requests, io
from PIL import Image

SERVER_URL = os.environ.get("SERVER_URL", "http://127.0.0.1:8050")
SNAPSHOT_URL = SERVER_URL.rstrip("/") + "/snapshot.png"
WIDTH = int(os.environ.get("STREAM_WIDTH", 640))
HEIGHT = int(os.environ.get("STREAM_HEIGHT", 360))
FPS = int(os.environ.get("STREAM_FPS", 10))
BITRATE = os.environ.get("STREAM_BITRATE", "600k")
YT_PRIMARY = os.environ.get("YT_PRIMARY", "rtmps://a.rtmp.youtube.com/live2")
YT_KEY = os.environ.get("YT_KEY", "")
YT_BACKUP = os.environ.get("YT_BACKUP", "")
RESTART_INTERVAL = int(os.environ.get("RESTART_INTERVAL", 14400))  # seconds

FFMPEG_CMD = [
    "ffmpeg", "-hide_banner", "-loglevel", "info",
    "-f", "rawvideo", "-pix_fmt", "rgb24",
    "-s", f"{WIDTH}x{HEIGHT}",
    "-r", str(FPS), "-i", "-", 
    "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
    "-c:v", "libx264", "-preset", "ultrafast", "-b:v", BITRATE,
    "-pix_fmt", "yuv420p", "-g", str(int(FPS*2)),
    "-c:a", "aac", "-b:a", "128k", "-ar", "44100",
    "-f", "flv", f"{YT_PRIMARY}/{YT_KEY}"
]

if YT_BACKUP:
    FFMPEG_CMD += ["-f", "flv", f"{YT_BACKUP}/{YT_KEY}"]

def fetch_snapshot(timeout=2.0):
    try:
        r = requests.get(SNAPSHOT_URL, timeout=timeout)
        if r.status_code == 200 and r.content:
            img = Image.open(io.BytesIO(r.content)).convert("RGB")
            img = img.resize((WIDTH, HEIGHT))
            return img
    except Exception as e:
        # print to stderr for logs
        print("snapshot error:", e, file=sys.stderr)
    return None

def send_loop():
    while True:
        try:
            proc = subprocess.Popen(FFMPEG_CMD, stdin=subprocess.PIPE)
            start = time.time()
            while True:
                img = fetch_snapshot(timeout=3)
                if img is None:
                    # if snapshot failing, sleep short and retry
                    time.sleep(0.5)
                    continue
                # write raw RGB bytes to ffmpeg stdin
                proc.stdin.write(img.tobytes())
                # flush occasionally
                # enforce FPS
                time.sleep(1.0 / FPS)
                # restart ffmpeg every RESTART_INTERVAL seconds to satisfy yt 4-hour behaviors
                if RESTART_INTERVAL and (time.time() - start) > RESTART_INTERVAL:
                    print("Restarting ffmpeg after interval", file=sys.stderr)
                    try:
                        proc.stdin.close()
                    except:
                        pass
                    proc.terminate()
                    proc.wait(timeout=5)
                    break
            # small pause before restarting
            time.sleep(1)
        except Exception as ex:
            print("Stream loop error:", ex, file=sys.stderr)
            time.sleep(2)

if __name__ == "__main__":
    if not YT_KEY:
        print("YT_KEY not set in env. Exiting.", file=sys.stderr)
        sys.exit(2)
    send_loop()
