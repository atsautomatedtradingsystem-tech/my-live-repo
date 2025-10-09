#!/usr/bin/env bash
set -euo pipefail
# start everything from repo root
# assumes .venv exists and .env present

# activate venv
if [ -f .venv/bin/activate ]; then
  . .venv/bin/activate
fi

# load .env variables (naive)
set -a
[ -f .env ] && . .env
set +a

mkdir -p logs

# start server (live.py)
nohup ${LIVE_PY_CMD:-"python3 live.py"} > logs/live_stdout.log 2>&1 &
echo "started live.py pid=$(pgrep -f live.py | tr '\n' ' ')"

# small wait
sleep 2

# start stream_frames.py
nohup python3 stream_frames.py > logs/stream_run.log 2>&1 &
echo "started stream_frames.py pid=$(pgrep -f stream_frames.py | tr '\n' ' ')"

echo "logs: logs/live_stdout.log logs/stream_run.log"
