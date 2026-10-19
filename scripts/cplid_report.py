#!/usr/bin/env python3
"""CPLID-T0T1T2: plots, REPORT.md, README.md, DATA_AUDIT.md (post-eval)."""
import json, shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("/root/autodl-tmp/version10-12/CPLID-T0T1T2-detector-top8-v1")
REFS = [3001, 3002, 3003, 3004, 3005]


def wj(p, obj):
    p = Path(p); tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1) + "\n"); tmp.replace(p)


def main():
    m = json.load(open(OUT / "metrics.json"))
    G = json.load(open(OUT / "REPRO_AND_CORRECTNESS.json"))
    audit = json.load(open(OUT / "AUDIT_SUMMARY.json"))
    t2m = json.load(open(OUT / "T2_EXPOSURE_MANIFEST.json"))
    diag = json.load(open(OUT / "EXPOSURE_DIAGNOSTICS.json"))
    sl = json.load(open(OUT / "SUPPORT_LISTS.json"))
    per = {a: {r["ref"]: r["auroc"] for r in m["rows"] if r["arm"] == a} for a in ("T0", "T1", "T2")}
    # ---------- plots ----------
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    x = np.arange(5); wd = 0.26
    for i, a in enumerate(("T0", "T1", "T2")):
        ax[0].bar(x + (i - 1) * wd, [per[a][j] for j in REFS], wd, label=a)
    ax[0].set_xticks(x); ax[0].set_xticklabels(["ref%d" % j for j in REFS])
    ax[0].set_ylim(0.5, 1.0); ax[0].set_ylabel("crop-level image AUROC")
    ax[0].set_title("Per-ref AUROC by arm"); ax[0].legend()
    d1 = [per["T1"][j] - per["T0"][j] for j in REFS]
    d2 = [per["T2"][j] - per["T1"][j] for j in REFS]
    ax[1].bar(x - 0.18, d1, 0.36, label="T1-T0")
    ax[1].bar(x + 0.18, d2, 0.36, label="T2-T1")
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_xticks(x); ax[1].set_xticklabels(["ref%d" % j for j in REFS])
    ax[1].set_title("Paired differences"); ax[1].legend()
    fig.tight_layout(); fig.savefig(OUT / "auroc_and_paired.png", dpi=130); plt.close(fig)
    # ---------- DATA_AUDIT.md ----------
    (OUT / "DATA_AUDIT.md").write_text("""# DATA_AUDIT — CPLID-T0T1T2-detector-top8-v1

扫描对象：本地副本 `/root/data_cplid_local`（自 `/root/autodl-tmp/data/CPLID` 完整复制，
tree 比对一致；复制原因：数据盘当日报到间歇性读停顿，审计与训练改为本地盘，原盘数据未改动）。

## 观察值（与任务预期一致）

- Normal_Insulators：600 图 + 600 XML；Defective_Insulators：248 图；
  labels/insulator 248 XML + labels/defect 248 XML（496 份 XML，非 496 张图）；
- 图像/XML 按 stem 一一对应（XML filename 无扩展名，按实际图片 stem 匹配；
  XML 遗留绝对 path 字段一律忽略）；尺寸断言一致；**问题计数 0**（无缺 XML、无尺寸不符、无非法框）；
- 框类别计数：insulator 1321（正常+缺陷整框）、defect 248（每缺陷图 1 个小缺陷框）；
- OriginalImageOnWebDriver 为空目录（0 文件），记录在案。

## 重复与来源分组

预注册规则（扫描前冻结于 cplid_audit.py 文档串）：精确重复=文件 sha256 相同；
近重复=dHash64 Hamming<=10（为隔离安全取保守宽松阈值）；编号相邻不作为证据；
相似度低不证明独立（记 unknown）。

- 精确重复组 3、法线侧近重复边 3 → 600 张正常图归并为 **597 个源组**；
- 248 张缺陷图各自成组（无精确重复）；
- 跨侧共享：缺陷图对正常图 dHash 最小 Hamming 分布 min=11（1 例）、12–14 共 10 例、
  其余 >=15。按预注册阈值 <=10：**确认共享来源 0 例，排除组 0**。
  Hamming=11 的 1 例与 12–14 的 10 例按规则记 unknown，不作排除，如实披露。

## 隔离边界声明

本轮为**文件及已识别来源组隔离评测**：已识别的共享来源（精确/近重复）已纳入分组；
无法核实的合成来源关系（含全部 unknown 对）不宣称排除。
不声明"所有测试源帧从未被主干见过"。

## 建库名单与重叠

5 组 x 10N（10 个不同源组，PCG64(ref)，固定排序后抽取，不做外观/表现筛选）。
配对重叠矩阵见 SUPPORT_LISTS.json（ref3002-ref3003 共享 1 组，其余 0）。
""")
    # ---------- REPORT.md ----------
    s = m["arm_summary"]; pr = m["paired"]; cnt = m["counts"]
    rows = chr(10).join("|%s|%d|%.4f|%.4f|%.4f|%.4f|" % (
        "ref%d" % j, 5, per["T0"][j], per["T1"][j], per["T2"][j], per["T2"][j] - per["T1"][j]) for j in REFS)
    dexp = diag["per_arm"]
    report = """# REPORT — CPLID-T0T1T2-detector-top8-v1

命名：**文件及已识别来源组隔离评测**（不声明源帧从未见过；unknown 来源关系如实披露）。
框架：YOLO-P3(module4) 通道L2 patch + 官方 KCenterGreedy(0.1, 16000->1600) + 欧氏NN + top8 均值，
GT 整框裁剪（隔离异常评分能力，非完整自动检测流水线）。三臂同 5x10N 名单、同测试集。

## 主表（裁剪级 image AUROC，5 组建库）

|ref|k|T0|T1|T2|T2-T1|
|---|---|---|---|---|---|
%s

|臂|均值|std(ddof=0)|逐组范围|
|---|---|---|---|
|T0|%.4f|%.4f|[%.4f, %.4f]|
|T1|%.4f|%.4f|[%.4f, %.4f]|
|T2|%.4f|%.4f|[%.4f, %.4f]|

配对：T1-T0 = %+.4f（均值），T2-T1 = %+.4f（均值）。逐组见 paired_comparisons.csv 与图 auroc_and_paired.png。

## 样本量

- 正常测试：%d 原图 / %d 裁剪（其中 T2 暴露 %d 裁剪）；缺陷：%d 原图 / %d 裁剪（labels/insulator 整框）；
- 训练侧：T1 与 T2 各 %d 张训练图（T2 = T1 - %d 移除 + %d 暴露，len 断言相等），val %d 图两臂全同；
- 主干训练量与建库量分别报告：主干见 100 epochs x %d 步/臂；建库 = 每臂每参照 10 裁剪 16000 patch -> 1600 原型。

## 污染诊断（T2 暴露 vs 未暴露正常测试子集）

|臂|ref|缺陷vs暴露|缺陷vs未暴露|暴露均值分|未暴露均值分|
|---|---|---|---|---|---|
%s

注意：两个正常子集本身难度不同，绝对分数差不能直接解释为污染的因果影响；
T2-T1 的系统性差异还包含被替换训练图的内容差异。T2 是预注册污染诊断，不是有效性能主结果。

## 正确性门（全部实际执行，数值见 REPRO_AND_CORRECTNESS.json）

角色断言、T2 暴露精确核对、T1/T2 初始参数哈希一致+更新证明、checkpoint 重载输出一致（diff=0）、
P3 形状与通道L2（norm∈[%.6f,%.6f]）、冻结提取 batch=1 vs 32 atol=1e-4（实测 %.2e）、
分块NN vs float64暴力（尾块）、top8 轴与方向合成测试、同种子双建库一致（sha 相同）、
评测前后状态哈希不变、逐分绑定（原图,框,裁剪sha）、AUROC 双实现复核（差<1e-9）。

## 数据与审计

见 DATA_AUDIT.md 与 DATA_MANIFEST.jsonl。确认共享来源 0 例；unknown 关系（Hamming 11-14）10+1 对
未排除、如实披露；不宣称全面排除。

## 回答

1. **正常检测训练是否优于 T0（在已审计隔离边界内）**：T1-T0 = %+.4f 均值（逐组 %s）。
2. **T2 相对 T1**：%+.4f 均值（逐组 %s）——提高/下降/基本不变的判读见逐组数据与暴露诊断。
3. **变化位置**：暴露诊断表 + 配对差；不得把子集难度差读成因果。
4. **未知来源关系**：合成缺陷的背景来源关系仅能以内容相似性近似核实，Hamming 11-14 的对
   未知；不能声明完全无泄露。
5. **是否值得进入预测框端到端验证**：仅当 T1-T0 与 T2-T1 的方向结论在逐组上稳定时才有意义；
   本轮为 GT 框隔离评分，端到端验证是独立后续决策，不自动启动。

## 三种解释的区分

- **T1-T0**：正常检测训练的系统效果；
- **T2-T1**：预注册污染处理的系统效果，仍混杂被替换训练图内容差异；
- **五组建库波动**：正常库选择的不确定性，不是五个独立数据集，也不是五次独立主干训练。

不在 test 上挑阈值；不报告测试集最优召回/F1；不因 T2 未提升声称数据泄露无害；
不把本轮 T2 效应套用到旧 InsPLAD 实验。
""" % (rows,
        s["T0"]["mean"], s["T0"]["std"], min(per["T0"].values()), max(per["T0"].values()),
        s["T1"]["mean"], s["T1"]["std"], min(per["T1"].values()), max(per["T1"].values()),
        s["T2"]["mean"], s["T2"]["std"], min(per["T2"].values()), max(per["T2"].values()),
        pr["T1_minus_T0"]["mean"], pr["T2_minus_T1"]["mean"],
        cnt["normal_images"], cnt["normal_crops"], cnt["exposed_normal_crops"],
        cnt["defective_images"], cnt["defective_crops"],
        len(t2m["t1_train_images"]), t2m["removed_image_count"], t2m["exposed_image_count"],
        61, G.get("g3_params_ok") and json.load(open(OUT / "TRAIN_LOG_T1.json"))["optimizer_steps_estimate"],
        chr(10).join("|%s|%d|%.4f|%.4f|%.6f|%.6f|" % (a, j,
            dexp[a]["ref%d" % j]["defect_vs_exposed"], dexp[a]["ref%d" % j]["defect_vs_unexposed"],
            dexp[a]["ref%d" % j]["mean_score_exposed"], dexp[a]["ref%d" % j]["mean_score_unexposed"])
            for a in ("T0", "T1", "T2") for j in REFS),
        G["g5_p3_l2"]["norm_min"], G["g5_p3_l2"]["norm_max"],
        G["g6_batch_equiv"]["max_abs_diff"],
        pr["T1_minus_T0"]["mean"],
        ", ".join("%+.3f" % v for v in pr["T1_minus_T0"]["per_ref"].values()),
        pr["T2_minus_T1"]["mean"],
        ", ".join("%+.3f" % v for v in pr["T2_minus_T1"]["per_ref"].values()))
    (OUT / "REPORT.md").write_text(report)
    # ---------- README ----------
    (OUT / "README.md").write_text("""# CPLID-T0T1T2-detector-top8-v1 结果目录

合同: `plan/tasks/CPLID-T0T1T2-detector-top8-v1.yaml`

复算:

```bash
export CPLID_ROOT=/root/data_cplid_local
python3 scripts/cplid_audit.py          # 数据审计 (幂等)
python3 scripts/cplid_freeze.py         # 划分/名单/T2 冻结 (幂等, 评分前)
TRAIN_STAGE=prep python3 scripts/cplid_train.py && TRAIN_STAGE=all python3 scripts/cplid_train.py
python3 scripts/cplid_eval.py           # 12 门 + 15 配置评测
python3 scripts/cplid_report.py         # 报告与图
```

阅读顺序: REPORT.md -> DATA_AUDIT.md -> REPRO_AND_CORRECTNESS.json -> metrics.json /
paired_comparisons.csv / EXPOSURE_DIAGNOSTICS.json -> raw_scores/ -> SUPPORT_LISTS.json /
T2_EXPOSURE_MANIFEST.json -> FROZEN_CONFIG.json / MODEL_IDENTITY.json / TRAIN_LOG_*.json ->
BANK_MANIFEST.json / RESOURCE_COST.json。

权重与大缓存不入 Git（cache/ 目录含数据集符号链接、运行与特征缓存；特征缓存在 /root/cplid_cache）。
""")
    print("[report] written; T1-T0=%+.4f T2-T1=%+.4f" %
           (pr["T1_minus_T0"]["mean"], pr["T2_minus_T1"]["mean"]))


if __name__ == "__main__":
    main()
