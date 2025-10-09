#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# load .env if exists
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

# create venv
python3 -m venv .venv
source .venv/bin/activate

# upgrade pip and install requirements (torch separately if needed)
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt || true

# If live.py needs torch, install CPU wheel (Uncomment if needed)
# python -m pip install "torch" --index-url https://download.pytorch.org/whl/cpu

# ensure log dirs
mkdir -p logs

# start live.py (dash) in background with nohup
nohup ${LIVE_PY_CMD} > logs/live_stdout.log 2>&1 & echo $! > logs/live.pid

# wait short time for server to come up
sleep 3

# check snapshot endpoint
curl -sS ${SERVER_URL:-http://127.0.0.1:8050}/snapshot.png -o /tmp/test_snapshot.png || true

# start streaming process
nohup python3 stream_frames.py > logs/stream_run.log 2>&1 & echo $! > logs/stream.pid

echo "Started (or already running). Check logs/live_stdout.log and logs/stream_run.log"
