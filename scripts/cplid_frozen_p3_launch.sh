#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p /root/autodl-tmp/version10-12/CPLID-frozenP3-det-v1
exec 9>/root/autodl-tmp/version10-12/CPLID-frozenP3-det-v1/.train.lock
flock -n 9 || { echo 'CPLID frozen-P3 training is already active'; exit 1; }
exec /root/miniconda3/bin/python -u scripts/cplid_frozen_p3.py --mode train "$@"
