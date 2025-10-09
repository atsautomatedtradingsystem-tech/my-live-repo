#!/usr/bin/env bash
set -euo pipefail

# load .env if exists
if [ -f .env ]; then
  # shellcheck disable=SC1091
  export $(grep -v '^#' .env | xargs)
fi

# create logs dir
mkdir -p logs

# create venv if missing
if [ ! -d .venv ]; then
  python3 -m venv .venv
fi

# activate and install deps
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

# install base requirements (adjust files if you have different names)
if [ -f requirements.txt ]; then
  pip install -r requirements.txt || true
fi

# install stream-specific minimal libs
pip install requests pillow numpy

# (optional) install CPU torch wheel if live.py needs torch:
# python -m pip install "torch" --index-url https://download.pytorch.org/whl/cpu || true

# start live.py
nohup ${LIVE_PY_CMD:-"python3 live.py"} > logs/live_stdout.log 2>&1 &

sleep 1

# start streaming helper
nohup python3 stream_frames.py > logs/stream_run.log 2>&1 &

echo "Started (or already running). Check logs/live_stdout.log and logs/stream_run.log"
