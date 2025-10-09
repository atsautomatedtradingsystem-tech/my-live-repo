#!/usr/bin/env bash
set -euo pipefail

# load .env safely (if exists)
if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
fi

# create logs
mkdir -p logs

# create venv
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
# activate
# shellcheck disable=SC1091
. .venv/bin/activate

# upgrade pip
python -m pip install --upgrade pip setuptools wheel

# install system deps (may require sudo in codespace)
if command -v sudo >/dev/null 2>&1; then
  sudo apt update -y
  sudo apt install -y ffmpeg
else
  echo "WARNING: cannot run apt (no sudo). Make sure ffmpeg is installed."
fi

# install python deps for stream component
python -m pip install -r requirements-stream.txt

# install torch CPU wheel (if live.py needs torch) — optional
# uncomment/modify for your python version when required:
# python -m pip install "torch" --index-url https://download.pytorch.org/whl/cpu

# start live.py (if present)
if [ -f live.py ]; then
  echo "Starting live.py ..."
  nohup python3 live.py > logs/live_stdout.log 2>&1 &
  echo $! > logs/live.pid
else
  echo "live.py not found in repo root"
fi

# start stream_frames.py
if [ -f stream_frames.py ]; then
  echo "Starting stream_frames.py ..."
  chmod +x stream_frames.py
  nohup python3 stream_frames.py > logs/stream_run.log 2>&1 &
  echo $! > logs/stream.pid
else
  echo "stream_frames.py not found"
fi

echo "Started (or already running). Check logs/live_stdout.log and logs/stream_run.log"
