#!/usr/bin/env python3
import os, sys, time, subprocess, io, signal, shutil
from PIL import Image
import requests

# config from env
WIDTH = int(os.environ.get("STREAM_WIDTH", 640))
HEIGHT = int(os.environ.get("STREAM_HEIGHT", 360))
FPS = int(os.environ.get("STREAM_FPS", 10))
BITRATE = os.environ.get("STREAM_BITRATE", "600k")
YT_KEY = os.environ.get("YT_KEY")
YT_PRIMARY = os.environ.get("YT_PRIMARY", "rtmp://a.rtmp.youtube.com/live2")
YT_BACKUP  = os.environ.get("YT_BACKUP", "rtmp://b.rtmp.youtube.com/live2?backup=1")
SERVER_URL = os.environ.get("SERVER_URL", "http://127.0.0.1:8050")
FRAME_INTERVAL = 1.0 / max(1, FPS)

if not YT_KEY:
    print("YT_KEY not set in environment. Exiting.", file=sys.stderr)
    sys.exit(2)

RTMP_PRIMARY = f"{YT_PRIMARY.rstrip('/')}/{YT_KEY}"
RTMP_BACKUP = f"{YT_BACKUP.rstrip('/')}/{YT_KEY}"

FFMPEG_CMD = [
    "ffmpeg", "-hide_banner", "-loglevel", "info",
    "-f", "rawvideo", "-pix_fmt", "rgb24",
    "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", "-",
    "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
    "-c:v", "libx264", "-preset", "veryfast",
    "-b:v", BITRATE, "-pix_fmt", "yuv420p", "-g", str(max(2, int(FPS*2))),
    "-c:a", "aac", "-b:a", "64k", "-ar", "44100",
    "-f", "flv", RTMP_PRIMARY, "-f", "flv", RTMP_BACKUP
]

proc = None
def start_ffmpeg():
    global proc
    if proc and proc.poll() is None:
        return proc
    try:
        proc = subprocess.Popen(FFMPEG_CMD, stdin=subprocess.PIPE)
        print("Started ffmpeg pid=", proc.pid)
    except Exception as e:
        print("Failed to start ffmpeg:", e)
        proc = None
    return proc

def get_snapshot(timeout=2.0):
    try:
        url = SERVER_URL.rstrip("/") + "/snapshot.png"
        r = requests.get(url, timeout=timeout)
        if r.status_code == 200:
            return Image.open(io.BytesIO(r.content)).convert("RGB")
    except Exception as e:
        # print("snapshot error", e)
        return None

def main_loop():
    last_start = time.time()
    ff = start_ffmpeg()
    backoff = 0.1
    while True:
        t0 = time.time()
        shot = get_snapshot(timeout=max(0.5, FRAME_INTERVAL*0.8))
        if shot is None:
            # if no snapshot, draw placeholder
            img = Image.new("RGB", (WIDTH, HEIGHT), (0,0,0))
        else:
            img = shot.resize((WIDTH, HEIGHT))
        # write raw to ffmpeg stdin
        if ff and ff.poll() is None:
            try:
                ff.stdin.write(img.tobytes())
                ff.stdin.flush()
                backoff = 0.1
            except Exception as e:
                print("ffmpeg write error:", e, "restarting ffmpeg")
                try:
                    ff.kill()
                except:
                    pass
                ff = start_ffmpeg()
        else:
            ff = start_ffmpeg()
        # sleep to maintain FPS
        elapsed = time.time() - t0
        to_sleep = FRAME_INTERVAL - elapsed
        if to_sleep > 0:
            time.sleep(to_sleep)
        else:
            # if behind schedule, small sleep to avoid busy-loop
            time.sleep(0.01)

if __name__ == "__main__":
    try:
        main_loop()
    except KeyboardInterrupt:
        print("Interrupted")
    finally:
        if proc:
            try:
                proc.terminate()
            except:
                pass
