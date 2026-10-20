#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export YOLO_AUTOINSTALL=false
exec 9>/root/autodl-tmp/version10-12/.insplad-frozen-p3.lock
flock -n 9 || { echo 'An InsPLAD frozen-P3 process is already active'; exit 1; }
exec /root/miniconda3/bin/python -u scripts/insplad_frozen_p3_recover.py "$@"
