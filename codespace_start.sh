#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# load .env if present
if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

# create logs dir
mkdir -p logs

# create venv if missing
if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
. .venv/bin/activate

# install requirements
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt || true

# ensure ffmpeg exists (attempt)
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg not found. Trying to apt-get install (may fail in Codespaces without sudo)."
  if command -v sudo >/dev/null 2>&1; then
    sudo apt-get update && sudo apt-get install -y ffmpeg || true
  fi
fi

# start live.py (Dash app) in background if not running
if pgrep -f "python.*live.py" >/dev/null 2>&1; then
  echo "live.py already running"
else
  nohup python3 live.py > logs/live_stdout.log 2>&1 &
  sleep 2
fi

# start stream_frames.py in background
if pgrep -f "python.*stream_frames.py" >/dev/null 2>&1; then
  echo "stream_frames.py already running"
else
  nohup python3 stream_frames.py > logs/stream_run.log 2>&1 &
fi

echo "Started (or already running). Check logs/live_stdout.log and logs/stream_run.log"
