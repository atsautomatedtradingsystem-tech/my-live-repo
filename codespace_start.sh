#!/usr/bin/env bash
set -euo pipefail

# 1) віртуальне оточення
python3 -m venv .venv
. .venv/bin/activate

# 2) оновити pip
python -m pip install --upgrade pip setuptools wheel

# 3) встановити звичайні requirements
pip install -r requirements.txt

# 4) якщо потрібно torch (CPU) – інсталюємо окремо (не завжди у requirements)
python -m pip install "torch" --index-url https://download.pytorch.org/whl/cpu || true

# 5) системні перевірки
which ffmpeg >/dev/null 2>&1 || echo "WARNING: ffmpeg not found. Install system package."

# 6) створити директорії логів
mkdir -p logs

# 7) виконай live.py у фоні (лог в logs/live_stdout.log)
nohup .venv/bin/python3 live.py > logs/live_stdout.log 2>&1 &

# 8) почекай трохи, потім запусти stream (лог в logs/stream_run.log)
sleep 3
nohup .venv/bin/python3 stream_frames.py > logs/stream_run.log 2>&1 &
echo "Started live.py and stream_frames.py; check logs/ directory"
