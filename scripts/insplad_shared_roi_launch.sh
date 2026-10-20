#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8
python_bin=/root/miniconda3/bin/python
out=/root/autodl-tmp/version10-12/InsPLAD-sharedROI-ABC-ref3001-v1
mkdir -p "$out"
exec 9>"$out/.run.lock"
flock -n 9 || { echo 'Shared-ROI experiment is already running'; exit 1; }
"$python_bin" - <<'PY'
import json
from pathlib import Path
p=Path('registry/active_task.json')
if p.exists():
    active=json.loads(p.read_text())['task']
    assert active in ('InsPLAD-frozenP3-14class-v1','InsPLAD-sharedROI-ABC-ref3001-v1',
                      'InsPLAD-sharedROI-ABC-ref3001-v1-prepare'), 'Unrelated active task; resolve registration first'
    if active == 'InsPLAD-frozenP3-14class-v1':
        root=Path('/root/autodl-tmp/version10-12')/active
        assert json.loads((root/'DETECTION_RESULTS.json').read_text())['completed']
        assert json.loads((root/'FREEZE_CORRECTNESS.json').read_text())['passed']
    if active.endswith('-prepare'):
        root=Path('/root/autodl-tmp/version10-12/InsPLAD-sharedROI-ABC-ref3001-v1')
        assert json.loads((root/'PREPARE.json').read_text())['ready']
        assert json.loads((root/'CPU_PREFLIGHT.json').read_text())['passed']
PY
"$python_bin" scripts/start_task.py scripts/insplad_shared_roi.task.yaml
"$python_bin" -u scripts/insplad_shared_roi.py --mode run "$@"
"$python_bin" scripts/accept_check.py scripts/insplad_shared_roi.task.yaml
"$python_bin" scripts/log_run.py --task InsPLAD-sharedROI-ABC-ref3001-v1 --status PASS \
  --note 'Small-image 10N ref3001 A/B/C: 7 pairs, 21 score records; full query binding and fallback rates'
