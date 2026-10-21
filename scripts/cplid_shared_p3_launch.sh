#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8
PYTHON=/root/miniconda3/bin/python
RESULT=/root/autodl-tmp/version10-12/CPLID-sharedP3-R0R1R2-v1
mkdir -p "$RESULT"
exec 9>"$RESULT/.run.lock"
flock -n 9 || { echo 'Another R0/R1/R2 process is active'; exit 1; }
"$PYTHON" scripts/cplid_shared_p3.py --mode prepare
"$PYTHON" scripts/start_task.py "$RESULT/TASK.yaml"
"$PYTHON" -u scripts/cplid_shared_p3.py --mode run "$@"
"$PYTHON" scripts/accept_check.py "$RESULT/TASK.yaml"
"$PYTHON" scripts/log_run.py --task CPLID-sharedP3-R0R1R2-v1 --status PASS --note 'Frozen R0/R1/R2 completed; see result directory; large cache/weights not committed'
echo "[complete] $RESULT/REPORT.md"
