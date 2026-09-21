#!/bin/bash
# M3DIAG2 pipeline: BN recal -> repr diag -> C/C2 eval (no training, ~15 min)
set -euo pipefail
cd /root/auto-exp
nvidia-smi -L || { echo "no GPU"; exit 1; }
FREEKB=$(df -k / | tail -1 | awk '{print $4}')
if [ "$FREEKB" -lt 2097152 ]; then echo "disk low"; exit 1; fi
if pgrep -f "python3 scripts/m3diag" >/dev/null; then echo "m3diag running"; exit 1; fi
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
mkdir -p outputs/M3diag2
echo "=== [1/3] BN recal ==="
python3 scripts/m3diag2_bn_recal.py 2>&1 | tee -a outputs/M3diag2/run.log
echo "=== [2/3] repr diag ==="
python3 scripts/m3diag2_repr_diag.py 2>&1 | tee -a outputs/M3diag2/run.log
echo "=== [3/3] C/C2 eval ==="
python3 scripts/m3diag2_bn_eval.py 2>&1 | tee -a outputs/M3diag2/run.log
echo "=== done ==="