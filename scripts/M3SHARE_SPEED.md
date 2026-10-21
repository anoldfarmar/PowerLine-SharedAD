# M3SHARE-speed-v1：用户启动

本轮正式GPU门、探针、建库、测速与辅助精度评测由用户运行。准备阶段只做CPU单元测试及清单/身份检查。

## 实际完成范围

当前没有可靠的原始航拍大图—异常小图映射，不猜DJI身份。
所以本次是 **原有小图的原始分辨率文件→检测→裁剪或共享ROI→诊断输出** 的局部完整调用测速，不是1920×1080原始场景端到端加速验证。
旧A没有检测，旧C是top1框；不改它们的定义。新增 `T-CROP-small` 和 `T-SHARE-small` 保留全部指定类别框，用于框数成本对照。
两臂无框时都回退全图并诊断，绝不把无框当正常或跳过。每个原图batch的裁剪/ROI均批量处理。

固定10N/ref3001的50正常支持文件；各自路径重新建库，vari-grip有13个正常候选ROI但仍只有10个文件预算。
计时固定每资产路径哈希前64张，共320张；不看标签、框数、分数或速度选样。
辅助精度使用全部5215个原冻结查询。新多框路径没有图像聚合规则，因此新臂只报同一恰好1框、无框回退子集的图像级AUROC；旧A/C全量精度另列。

## 启动

```bash
cd /root/auto-exp
nohup bash scripts/m3share_speed_launch.sh \
  > /root/autodl-tmp/version10-12/M3SHARE-speed-v1.log 2>&1 &
tail -f /root/autodl-tmp/version10-12/M3SHARE-speed-v1.log
```

显式使用 `/root/miniconda3/bin/python`，不安装环境、不下载模型、不训练。
运行期间不要启动其他GPU任务；发现其他CUDA计算进程会停止测速，不会擅自杀进程。
开始先实际建匹配库和执行GPU正确性门。容差atol=rtol=1e-4；只排除未通过的原图batch候选，不放宽精度门。
Crop内部最多32裁剪批量，batch1/32门失败则停止；原图候选1/8/16由门和短探针决定共同吞吐batch。
探针按预声明规则冻结预热与重复次数后，交替运行全部正式轮次；不会挑最快轮。
每条主计时调用必须真的前向，不能用旧特征缓存替代；模型加载和建库单列成本，I/O单列。

正式计时中断后**不自动续写或重新抽轮次**：保留 `TIMING_PROTOCOL.json` 和 `TIMING_RAW.jsonl`，先交回诊断。
完成后自动验收、登记、将本任务代码对应的报告/清单/原始计时等小型结果同步到main。
若远程领先、认证失败或索引有其他人的暂存内容，只停止同步并保留本地结果，不强推、不自动合并。

## 输出

目录 `/root/autodl-tmp/version10-12/M3SHARE-speed-v1/`：

- `METHOD_SPEC.md`、`DATA_AUDIT.json`：已有A/C核实、局部范围与坐标/计时边界。
- `FROZEN_INPUTS.json`：全部对象路径/哈希/尺寸、框数/面积、标签可用性，及320测速名单。
- `REPRO_AND_CORRECTNESS.json`：状态、坐标、零/多框、边界、非方形、批量与检索门实际数值。
- `PROBE.json`、`TIMING_PROTOCOL.json`：短探针及正式计时前冻结参数。
- `TIMING_RAW.jsonl`：单帧、吞吐和单独profiling，各条绑定方法/输入/框数/batch/轮次/环境/实际前向计数。
- `DISK_IO.json`、`RESOURCE_COST.json`、`BANK_MANIFEST.json`：读盘、显存/RSS、启动及建库时间/库身份。
- `ACCURACY_SUMMARY.json`：旧A/C和新局部参照的有定义精度结果；逐框原分数另在 `accuracy/`，不混作整图AUROC。
- `SPEED_METRICS.json`、`SPEED_REPORT.md`：实际成本与适用边界；准备阶段不会生成虚构速度。

大特征缓存、权重、图片不提交Git。需要真实大场景结论，必须先取得可靠原图对应；本轮不自动进入其他方向或追加训练。
