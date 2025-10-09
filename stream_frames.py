#!/usr/bin/env python3
# stream_frames.py — simple, robust frame grabber -> ffmpeg -> YouTube
import os, sys, time, subprocess, io, signal
from PIL import Image
import requests

# config from env (with defaults)
WIDTH = int(os.getenv("STREAM_WIDTH", "640"))
HEIGHT = int(os.getenv("STREAM_HEIGHT", "360"))
FPS = float(os.getenv("STREAM_FPS", "10"))
BITRATE = os.getenv("STREAM_BITRATE", "600k")
YT_KEY = os.getenv("YT_KEY")
YT_PRIMARY = os.getenv("YT_PRIMARY", "rtmp://a.rtmp.youtube.com/live2")
YT_BACKUP = os.getenv("YT_BACKUP", "rtmp://b.rtmp.youtube.com/live2?backup=1")
SERVER_URL = os.getenv("SERVER_URL", "http://127.0.0.1:8050")
LIVE_PY_CMD = os.getenv("LIVE_PY_CMD", "python3 live.py")
MAX_DURATION = int(os.getenv("STREAM_MAX_DURATION", "0"))  # seconds; 0 = disabled

if not YT_KEY:
    print("YT_KEY not set. Exiting.", file=sys.stderr)
    sys.exit(2)

FRAME_INTERVAL = 1.0 / max(1.0, FPS)

def get_snapshot(timeout=2.0):
    url = SERVER_URL.rstrip("/") + "/snapshot.png"
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code == 200:
            return Image.open(io.BytesIO(r.content)).convert("RGB")
    except Exception as e:
        #print("snapshot error:", e, file=sys.stderr)
        return None
    return None

def start_ffmpeg_process():
    rtmp1 = f"{YT_PRIMARY}/{YT_KEY}"
    rtmp2 = f"{YT_BACKUP}/{YT_KEY}"
    ffmpeg_cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "info",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{WIDTH}x{HEIGHT}", "-r", str(int(FPS)),
        "-i", "-",  # stdin raw frames
        "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
        "-c:v", "libx264", "-preset", "ultrafast", "-b:v", BITRATE,
        "-pix_fmt", "yuv420p", "-g", "50",
        "-c:a", "aac", "-b:a", "128k", "-ar", "44100",
        "-f", "flv", rtmp1, "-f", "flv", rtmp2
    ]
    print("Starting ffmpeg:", " ".join(ffmpeg_cmd))
    return subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

def main_loop():
    start_time = time.time()
    ff = start_ffmpeg_process()
    try:
        while True:
            if MAX_DURATION and (time.time() - start_time) > MAX_DURATION:
                print("Maximum stream duration reached; restarting ffmpeg.")
                try:
                    ff.stdin.close()
                    ff.terminate()
                except:
                    pass
                ff.wait(timeout=5)
                start_time = time.time()
                ff = start_ffmpeg_process()
                continue

            shot = get_snapshot(timeout=min(2.0, FRAME_INTERVAL*0.8))
            if shot is None:
                # fallback: black frame
                img = Image.new("RGB", (WIDTH, HEIGHT), color=(0,0,0))
            else:
                img = shot.resize((WIDTH, HEIGHT))

            # write raw frame to ffmpeg stdin
            try:
                ff.stdin.write(img.tobytes())
            except Exception as e:
                print("ffmpeg pipe error, restarting ffmpeg:", e, file=sys.stderr)
                try:
                    ff.terminate()
                except:
                    pass
                ff.wait(timeout=5)
                ff = start_ffmpeg_process()
            time.sleep(FRAME_INTERVAL)
    except KeyboardInterrupt:
        print("Interrupted, exiting.")
    finally:
        try:
            ff.stdin.close()
            ff.terminate()
            ff.wait(timeout=5)
        except:
            pass

if __name__ == "__main__":
    main_loop()
