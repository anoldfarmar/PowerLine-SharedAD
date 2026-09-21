#!/bin/bash
# M3DIAG 训练+评测 启动器 (提取 -> B -> C -> 三变体评测)
set -e
cd /root/auto-exp
echo "[guard] GPU:"
nvidia-smi -L || { echo "无 GPU"; exit 1; }
FREEKB=$(df -k / | tail -1 | awk '{print $4}')
if [ "$FREEKB" -lt 2097152 ]; then echo "磁盘不足 2G (${FREEKB}KB)"; exit 1; fi
if pgrep -f "python3 scripts/m3diag_" >/dev/null; then echo "已有 m3diag 进程在跑"; exit 1; fi
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export XDG_CACHE_HOME=/root/auto-exp/.cache
mkdir -p outputs/M3diag
echo "=== [1/4] 特征提取 ==="
python3 scripts/m3diag_extract.py 2>&1 | tee -a outputs/M3diag/run.log
echo "=== [2/4] 训练 B (local-supcon) ==="
python3 scripts/m3diag_train.py --variant B 2>&1 | tee -a outputs/M3diag/run.log
echo "=== [3/4] 训练 C (local-correspondence) ==="
python3 scripts/m3diag_train.py --variant C 2>&1 | tee -a outputs/M3diag/run.log
echo "=== [4/4] 评测 A/B/C + 汇总 ==="
python3 scripts/m3diag_bank_eval.py 2>&1 | tee -a outputs/M3diag/run.log
echo "=== 全部完成 ==="