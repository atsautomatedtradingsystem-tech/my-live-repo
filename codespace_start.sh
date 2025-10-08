#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# load env file if present
if [ -f ".env" ]; then
  set -a
  source .env
  set +a
fi

# create logs dir
mkdir -p logs

# create venv if needed
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate

pip install --upgrade pip setuptools wheel
# install requirements (ignore missing torch issues)
pip install -r requirements.txt || true

# optional: install CPU torch if needed (uncomment)
# python -m pip install "torch" --index-url https://download.pytorch.org/whl/cpu || true

# stop old ones
pkill -f "live.py" || true
pkill -f "stream_frames.py" || true
pkill -f ffmpeg || true
sleep 1

# start live.py (background)
nohup $LIVE_PY_CMD > logs/live_stdout.log 2>&1 &
sleep 3

# start stream process
nohup python3 stream_frames.py > logs/stream_run.log 2>&1 &

echo "Started (or already running). Check logs/live_stdout.log and logs/stream_run.log"
