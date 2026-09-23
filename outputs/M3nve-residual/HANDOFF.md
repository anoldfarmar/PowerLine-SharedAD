# M3NVE-residual-v1 用户运行交接

> 完成更新（2026-09-23）：用户已运行全量；60配置/180臂完成，独立审计通过，机械验收8/8 PASS。以下准备状态和启动指令为历史交接记录，请勿再次启动。当前结果见NVE_REPORT.md，交付范围见COMPLETION.md。

状态：脚本、名单、数据及缓存已准备；全量实验未启动。未宣称实验PASS，未登记完成或推送未完成结果。

已完成：60配置A逐路径复现（max差约5e-9，容差1e-4）、人工向量、批量/单图一致性、模型参数及BN buffer不变检查。正式缓存有2691张图的FP32映射特征、20个N库、60个F库；两种预算共享历史名单；不需要重新下载数据、重提查询特征或重跑prepare。

工作目录 `/root/auto-exp`；原图和骨干权重位于 `/root/autodl-tmp`；输入路径均为原绝对路径。缓存位于 `outputs/M3nve-residual/cache`。必须使用 `/root/miniconda3/bin/python3`（torch2.8.0+cu128），不要使用exp-01。初次环境错误产物在quarantine，不参与正式结果；详见RECOVERY.md。

## 成本与预算

四个固定32图探针耗时21.61–24.63秒；共20,120图次、约3219万次局部patch PCA。按探针吞吐约4–4.4小时，加入50%余量及准备成本后保守估计6.49小时。主要成本为每patch的128×12薄SVD，非骨干提取。峰值显存准备阶段约346MiB，耗时不代表完整推理速度。预计额外产物小于2GiB，预检要求至少2GiB剩余空间。

原合同1GPU小时门保持生效。下面的命令含显式 `--gpu-budget-hours 8`：**由你手动执行即授权本次运行扩展到8GPU小时、10小时墙钟预算**，启动时写入USER_RUN_AUTHORIZATION.json及活动任务预算。没有该参数不会自动放行；历史cost_estimate.json中full_authorized=false保留，表示原1小时预算下不获准。

可选优化方向（本次未实施）：同一正常库下对相同邻居集合缓存局部子空间；或验证更适合小矩阵的批量薄SVD实现。需独立数值一致性检查及新的计算实现冻结。当前交接保留已检查的逐patch薄SVD，不改参数、不用全库PCA。

## 启动

建议在tmux中执行，避免SSH断开中止。不要再次执行prepare。

```bash
cd /root/auto-exp
bash scripts/m3nve_launch.sh --gpu-budget-hours 8
```

该入口先只读验证全部冻结输入/缓存、环境和空间，再激活同任务，执行full、生成图表报告、机械验收。**不会自动执行下一轮实验、登记PASS、提交或推送。** 最后由你审核验收输出和报告，再按仓库流程登记与同步。

仅检查准备状态，不运行实验：

```bash
cd /root/auto-exp
/root/miniconda3/bin/python3 scripts/m3nve_preflight.py
```

查看运行进度：

```bash
tail -f /root/auto-exp/outputs/M3nve-residual/full.log
```

每完成一个正常参照配置（含3个故障支持种子）输出一行，约数分钟至十几分钟。输出目录会出现raw_scores、paired_configs.csv、nve_eval.json、MECHANISMS.json、RESOURCE_COST.json、vis和NVE_REPORT.md。缓存及临时日志不入Git。

当前全量执行不支持中断续跑；请勿重复并发启动。若失败或中断，保留日志及现场，先诊断，不反复重跑。已记录1次prepare失败；再有正式失败达到2次上限应停止。独立预检失败不启动评分。

## 冻结记录与待完成项

本次交接只增加手动预算入口与预检，不改科学计算：原冻结manifest和原run源码归档于quarantine/pre_handoff_*，EXECUTION_AMENDMENT.json记录前后SHA256；预检按AST验证full中t0之后的全部科学执行语句及其他计算函数完全一致。当前manifest更新代码hash并绑定新增入口；名单、参数、权重、库与已通过复现结果不变。

待你运行后才能完成：B/C全量、独立AUROC审计、可视化及六项研究裁决、最终验收、登记、代码提交和远程同步。报告生成脚本已写好，尚未在全量产物上执行验证。
