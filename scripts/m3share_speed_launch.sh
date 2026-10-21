#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8
out=/root/autodl-tmp/version10-12/M3SHARE-speed-v1
mkdir -p "$out"
exec 9>"$out/.run.lock"
flock -n 9 || { echo 'M3SHARE speed task is already active'; exit 1; }
python_bin=/root/miniconda3/bin/python
"$python_bin" - <<'PY'
import json
from pathlib import Path
p=Path('registry/active_task.json')
if p.exists():
    task=json.loads(p.read_text())['task']
    assert task in ('M3SHARE-speed-v1','M3SHARE-speed-v1-prepare','InsPLAD-sharedROI-report-sync-v1'), 'Unrelated active task; resolve registration first'
PY
"$python_bin" scripts/start_task.py scripts/m3share_speed.task.yaml
"$python_bin" -u scripts/m3share_speed.py --mode run
"$python_bin" scripts/accept_check.py scripts/m3share_speed.task.yaml
"$python_bin" scripts/log_run.py --task M3SHARE-speed-v1 --status PASS \
  --note 'Actual local native-crop inputs only; matched all-box banks; alternating synchronized calls, separate profiles, conditional accuracy; raw-scene speed remains unknown'
"$python_bin" -u scripts/m3share_speed_publish.py
