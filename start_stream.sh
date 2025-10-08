#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")"

# timeout duration (4h by default) — змінити тут на іншу величину (e.g. 24h)
DURATION=${STREAM_RUN_DURATION:-4h}

# path to python and script
PY_BIN="${VENV_PY:-./.venv/bin/python}"
STREAM_SCRIPT="${STREAM_SCRIPT:-./stream_frames.py}"

# make sure logs dir exists
mkdir -p logs

# loop forever (24/7), but кожен запуск закінчується через DURATION
while true; do
  echo "Starting stream_frames.py at $(date) — duration ${DURATION}" >> logs/stream_run.log
  # Use timeout to force-stop after DURATION. --foreground ensures child signals passed.
  timeout --foreground ${DURATION} ${PY_BIN} ${STREAM_SCRIPT}
  echo "stream_frames.py exited at $(date). Sleeping 5s and restart." >> logs/stream_run.log
  sleep 5
done
