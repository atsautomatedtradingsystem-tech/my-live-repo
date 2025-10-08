#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")"

# create logs dir
mkdir -p logs

# create venv (idempotent)
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
# activate
# shellcheck disable=SC1091
source .venv/bin/activate

# upgrade pip/wheel
python -m pip install --upgrade pip setuptools wheel

# install normal requirements (without torch)
python -m pip install -r requirements.txt

# install CPU torch wheel (explicit index) — works for many linux setups.
python -m pip install "torch" --index-url https://download.pytorch.org/whl/cpu

# make sure ffmpeg exists (if not, install via apt; Codespaces usually has ffmpeg)
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg not found — attempting apt-get install (may require root)"
  sudo apt-get update && sudo apt-get install -y ffmpeg
fi

# export .env variables if file exists (do NOT commit .env)
if [ -f .env ]; then
  set -o allexport
  # shellcheck disable=SC1090
  source .env
  set +o allexport
fi

# start live.py in background with log
pkill -f live.py || true
nohup ${LIVE_PY_CMD:-python3 live.py} > logs/live_stdout.log 2>&1 &

sleep 2

# start streaming wrapper (this script restarts stream every 4h by default)
pkill -f start_stream.sh || true
nohup ./start_stream.sh > logs/stream_run.log 2>&1 &
