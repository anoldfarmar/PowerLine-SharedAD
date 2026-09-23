#!/usr/bin/env bash
set -euo pipefail
cd /root/auto-exp
if [[ $# != 2 || "$1" != '--gpu-budget-hours' ]]; then
    echo 'Review outputs/M3nve-residual/HANDOFF.md. Usage: bash scripts/m3nve_launch.sh --gpu-budget-hours 8' >&2
    exit 2
fi
PY=/root/miniconda3/bin/python3
"$PY" scripts/m3nve_preflight.py
"$PY" scripts/start_task.py plan/tasks/M3NVE-residual-v1.yaml
"$PY" scripts/m3nve_run.py full --gpu-budget-hours "$2"
"$PY" scripts/m3nve_report.py
"$PY" scripts/accept_check.py plan/tasks/M3NVE-residual-v1.yaml
echo 'Evaluation, report, and acceptance finished. No registry PASS, git commit, or push was performed by this launcher.'
