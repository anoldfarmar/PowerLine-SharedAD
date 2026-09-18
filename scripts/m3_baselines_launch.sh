#!/bin/bash
# M3 anomalib 基线一键启动（有卡模式开机后运行）
#   用法: bash scripts/m3_baselines_launch.sh
#   内容: patchcore/padim/efficientad × 5资产 × k∈{1,5,10} × 3种子 = 135 个配置
#   预计: 3–5 小时（RTX 4080 SUPER）
set -euo pipefail
cd /root/auto-exp
export OMP_NUM_THREADS=8  # 交互环境继承的值非法(libgomp 告警), 固定为安全值

# 1) GPU 断言：无卡模式下直接退出，防止误用 CPU 跑十几个小时
python3 - <<'EOF'
import torch
assert torch.cuda.is_available(), "CUDA 不可用：疑似无卡模式开机，请关机后用有卡模式重新开机再运行本脚本"
print("[launch] GPU OK:", torch.cuda.get_device_name(0))
EOF

# 2) imagenette 守护：/tmp 被清时自动从 download/ 重新解压，并核对图片总数
if [ ! -d /tmp/imagenette2/imagenette2/train ]; then
  echo "[launch] /tmp/imagenette2 缺失，重新解压 download/imagenette2.tgz ..."
  mkdir -p /tmp/imagenette2
  tar xzf download/imagenette2.tgz -C /tmp/imagenette2
fi
n=$(find /tmp/imagenette2 -name '*.JPEG' | wc -l)
if [ "$n" -ne 13394 ]; then
  echo "[launch] imagenette 图片数异常: $n (期望 13394)"; exit 1
fi
echo "[launch] imagenette OK: 13394 张"

# 3) 磁盘余量 >= 2G (df -k + 算术换算, 兼容 autodl 镜像无 -G 选项的 df)
avail_KB=$(df -k / | tail -1 | awk '{print $4}')
if [ -z "$avail_KB" ] || [ "$avail_KB" -lt 2097152 ]; then
  echo "[launch] 根分区可用空间不足 2G (剩 ${avail_KB}KB)"; exit 1
fi

# 4) 残留进程检查 (只匹配真正的 python 训练进程, 避免把包含脚本名的其他命令误判为残留)
if pgrep -f "python scripts/m3_run_anomalib_baselines\.py" > /dev/null; then
  echo "[launch] 已有基线进程在跑，先 kill 或确认后再启动"; exit 1
fi

# 5) 启动（后台 + 日志落盘）
export XDG_CACHE_HOME=/root/auto-exp/.cache
export HF_ENDPOINT=https://hf-mirror.com
# setsid: 脱离当前会话/进程组, 防止父 shell 退出时连带回收训练进程
setsid nohup python scripts/m3_run_anomalib_baselines.py > outputs/M3/baselines_run.log 2>&1 &
echo "[launch] baselines 已启动 pid=$!"
echo "[launch] 日志: outputs/M3/baselines_run.log  （tail -f 可观察进度）"
