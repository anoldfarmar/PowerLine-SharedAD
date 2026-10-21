# CPLID 严格冻结 P3 的绝缘子定位训练

与当前 InsPLAD 同一思路：COCO YOLOv8n 初始化，模块0–4（含主干P3）参数及BN完全冻结，训练后续P4/P5、Neck和单类别检测头。
不是只训练最后一层，也不是解冻全主干；不加载 InsPLAD best.pt 或旧 CPLID T1/T2 作为初始化。

数据根目录 `/root/autodl-tmp/data/CPLID`。沿用已冻结的来源组划分，不重新抽样：

- 正常训练419图、验证61图、保留正常测试120图；对应来源组分别418/60/119。
- 用原图与VOC绝缘子框做单类别 `insulator` 目标检测，不用故障框训练二分类检测头。
- 248张合成故障图不参与训练、验证选模型或建库；本任务不做异常检测评测。
- 旧清单路径 `/root/data_cplid_local/...` 只在图片SHA256和XML框核验后搬到实际数据根目录，绝不按编号猜身份。
- 原数据、XML和旧T0/T1/T2实验不改动；派生图片为软链接，YOLO标签另存结果目录。

配置沿用 InsPLAD：60轮，imgsz1280，batch8，AdamW lr0=.001，seed0，FP32，关闭TF32；增强也沿用原配方。
旧CPLID全量微调T1/T2用了不同训练配方；不能把它们当作同配方严格冻结消融。
每轮审计参数/BN/EMA及固定真实输入的P3，训练结束验证真实best.pt和last.pt。

## 用户启动

```bash
cd /root/auto-exp
nohup bash scripts/cplid_frozen_p3_launch.sh \
  > /root/autodl-tmp/version10-12/CPLID-frozenP3-det-v1.log 2>&1 &
tail -f /root/autodl-tmp/version10-12/CPLID-frozenP3-det-v1.log
```

明确使用已配置的 `/root/miniconda3/bin/python`，无需安装环境或下载模型。
中断的训练可用同一启动指令追加 `--resume`，建议日志用 `>>` 追加；之前的冻结审计会另行归档。
完成60轮后若仅最终评测失败，不重训，执行：

```bash
OMP_NUM_THREADS=4 /root/miniconda3/bin/python -u scripts/cplid_frozen_p3.py --mode evaluate
```

不要修改已冻结目录里的epoch、batch、imgsz或代码；改变配方需新输出目录并重新prepare。
完成后先用normal-val选择best，再在normal-test上评估，不按测试精度选轮次/阈值。
自动验收、登记并同步允许的报告/清单；远程冲突或索引有其他内容则停止同步，不强推。

权重：`/root/autodl-tmp/version10-12/CPLID-frozenP3-det-v1/runs/frozen_p3/weights/best.pt`、`last.pt`。
训练曲线：上述run的 `results.csv`；最终结果：`DETECTION_REPORT.md`、`DETECTION_RESULTS.json`、`FREEZE_CORRECTNESS.json`。
保留模块0–4只保证相同输入下P3保持，不保证检测mAP或共享ROI异常AUROC；本轮没有运行这些异常评测。
