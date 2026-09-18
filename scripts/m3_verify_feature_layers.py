#!/usr/bin/env python3
"""M3 Step-1: pre-flight + 特征层审计 + actual-M2-v0 复现.

产出:
  outputs/M3/preflight.json        —— 划分哈希核验 + 全部固化决策 (先于任何 query 评测)
  outputs/M3/feature_layer_audit.json —— 各层输出形状/stride 审计, P3 定位
  outputs/M3/actual_m2_repro.json  —— 用 M2 原管线 (model[:10] GAP) 复现 M2 metrics

用法: python3 scripts/m3_verify_feature_layers.py \
        --weights /root/autodl-tmp/runs/yolov8n_det/weights/best.pt
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m3_common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="/root/autodl-tmp/runs/yolov8n_det/weights/best.pt")
    ap.add_argument("--splits", default="/root/auto-exp/data/splits/insplad-fd")
    ap.add_argument("--m2-metrics", default="/root/auto-exp/outputs/M2/metrics.json")
    ap.add_argument("--out", default="/root/auto-exp/outputs/M3")
    ap.add_argument("--fd-root", default="/root/autodl-tmp/data/insplad/defect_supervised")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # ---------- 1. 权重与划分哈希 ----------
    wsha = C.sha256_file(args.weights)
    n_ok, bad = C.verify_split_hashes(args.splits, C.MANIFEST_HASHES)
    print(f"[preflight] best.pt sha256={wsha}")
    print(f"[preflight] split hashes verified: {n_ok}/12, bad={bad}")

    # ---------- 2. 特征层审计 ----------
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    bb = C.Backbone(args.weights, dev)
    x = torch.zeros(1, 3, C.IMGSZ, C.IMGSZ, device=dev)
    with torch.no_grad():
        feats = bb.forward_layers(x)
    layers_audit = []
    p3_found = None
    for i in range(10):
        f = feats[i]
        stride = C.IMGSZ // f.shape[-1]
        rec = {"layer": i, "type": type(bb.layers[i]).__name__,
               "channels": int(f.shape[1]), "spatial": [int(f.shape[2]), int(f.shape[3])],
               "stride": int(stride)}
        layers_audit.append(rec)
        if stride == 8 and p3_found is None:
            p3_found = rec
    audit = {
        "weights": args.weights, "weights_sha256": wsha,
        "input_size": C.IMGSZ,
        "model10_output_layer": C.P5_SPPF_LAYER,
        "model10_output": layers_audit[C.P5_SPPF_LAYER],
        "actual_m2_v0_feature": {
            "layer": C.P5_SPPF_LAYER, "name": "SPPF (P5)",
            "stride": layers_audit[C.P5_SPPF_LAYER]["stride"],
            "channels": layers_audit[C.P5_SPPF_LAYER]["channels"],
            "note": "M2 v0 实际用 model.model[:10] 末层输出做 GAP。"
                    "M2 报告的 'P3 (stride-8)' 标注与实现不符——实际为 stride-32 P5。"
                    "M2 产物不修改, 以 actual-M2-v0 登记并在 M3 报告中披露。",
        },
        "p3_layer": p3_found,
        "p3_stride_verified": 1 if (p3_found and p3_found["stride"] == 8
                                    and p3_found["channels"] == 64) else 0,
        "all_layers": layers_audit,
    }
    C.save_json(audit, f"{args.out}/feature_layer_audit.json")

    # ---------- 3. preflight 决策固化 (必须在 query 评测前) ----------
    import ultralytics
    import numpy as np
    preflight = {
        "task": "M3-proto-head-v1",
        "weights_sha256": wsha,
        "expected_m2_sha256_prefix": "ea1526c29d4afcfb",
        "weights_match_m2": wsha.startswith("ea1526c29d4afcfb"),
        "split_hashes_verified": n_ok,
        "split_hashes_total": 12,
        "split_hash_mismatches": bad,
        "env": {"torch": torch.__version__, "ultralytics": ultralytics.__version__,
                "cuda": torch.cuda.is_available(),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                "numpy": np.__version__},
        "data_paths": {"fd_root": args.fd_root,
                       "splits": args.splits,
                       "fd_root_exists": os.path.isdir(args.fd_root)},
        "leakage_registry": {
            "normal_query_src_in_det_train": "2500 裁剪中 2452 个源组出现在 yolo_det train "
                                             "(M2 冻结协议既定: 基类检测训练边界, 非本次新引入)",
            "defect_query_src_in_det_train": "100% 缺陷查询源组出现在 det train 帧 "
                                             "(同上, 协议既定; M3 禁止的是目标故障 query/k-shot "
                                             "实例特征参与任何反向传播, 本任务遵守)",
        },
        "frozen_decisions_before_query_eval": {
            "imgsz": C.IMGSZ,
            "variants": ["actual-M2-v0", "P3-GAP", "P3-patch-pre", "P3-patch-metric-v1"],
            "feature_layers": {"actual-M2-v0": "model[:10] 末层 (SPPF, stride32, 256d) GAP",
                               "P3-GAP": "layer4 (C2f, stride8, 64d) GAP",
                               "P3-patch-pre": "layer4 P3 patch 64d -> 固定随机正交投影 128d",
                               "P3-patch-metric-v1": "layer4 P3 patch 64d -> 度量学习投影头 128d"},
            "patch": {"dim": C.PATCH_DIM, "proj": "fixed random orthogonal 64->128",
                      "proj_seed": C.PATCH_PROJ_SEED,
                      "l2_normalize": True,
                      "image_aggregation": f"mean of top-{C.PATCH_TOPK} patch-bank max cosine sims",
                      "topk": C.PATCH_TOPK,
                      "coreset": C.CORESET_METHOD, "coreset_cap": C.CORESET_CAP,
                      "coreset_seed": C.CORESET_SEED},
            "metric_v1": C.METRIC_CFG,
            "recall_threshold_rule": "同资产正常查询打分 mean+2std (M2 固定规则, 零调参)",
            "seeds": C.SEEDS, "shots": C.SHOTS,
            "reporting_rule": "全部 1/5/10-shot × 3 种子 × 7 组合完整报告, 不挑选",
        },
        "anomalib_baseline_cfg": {
            "methods": ["PatchCore", "PaDiM", "EfficientAD"],
            "support_source": "非查询的 train/good 池 (defect_supervised/<asset>/train/good), "
                              "禁止使用 normal_query 建模或调参",
            "support_selection_rule": "每资产 train/good 池按固定种子随机抽 1/5/10 张, "
                                      "固定种子=sha256('insplad-fd-ad-support-k{k}-s{seed}') 派生; "
                                      "消费清单与哈希写入 baselines_eval.json",
            "support_shots": C.SHOTS, "support_seeds": C.SEEDS,
            "imgsz": C.IMGSZ,
            "supervision_note": "AD 基线只用正常支持样本(单类), 本方案用目标故障 k-shot 原型 "
                                "(多类判别) —— 监督信息不同, 报告中明确标注",
        },
        "rust_analysis_scope": C.RUST_COMBOS,
        "statistical_limitation_flags": {
            "polymer-insulator-upper-shackle/rust": "33 实例, k=10 查询侧仅 3 实例, 统计意义弱 "
                                                    "(划分清单使用纪律要求标注)"},
    }
    C.save_json(preflight, f"{args.out}/preflight.json")

    # ---------- 4. actual-M2-v0 复现 (M2 原管线) ----------
    tf = C.make_transform(C.IMGSZ)

    def p5gap(paths):
        vs = []
        for xb in C.load_batches(paths, tf):
            _, p5 = bb.p3_and_p5gap(xb)
            vs.append(torch.nn.functional.normalize(p5, dim=1).cpu())
        return torch.cat(vs)

    results = {}
    for k in C.SHOTS:
        per_seed = []
        for seed in C.SEEDS:
            with open(f"{args.splits}/kshot_k{k}_seed{seed}.json", encoding="utf-8") as f:
                split = json.load(f)
            with open(f"{args.splits}/query_k{k}.json", encoding="utf-8") as f:
                query = json.load(f)
            protos = {c: torch.nn.functional.normalize(p5gap(p).mean(0, keepdim=True), dim=1)
                      for c, p in split["prototype_pools"].items()}
            asset_norm = {}
            for akey, paths in query["normal_query"].items():
                asset_norm.setdefault(akey.split("/")[0], p5gap(paths))
            aurocs = {}
            for c in C.COMBOS:
                qd = p5gap(query["defect_query"][c])
                sn = asset_norm[C.ASSET_OF[c]]
                s_d = (qd @ protos[c].T).flatten().numpy()
                s_n = (sn @ protos[c].T).flatten().numpy()
                aurocs[c] = C.auroc_recall(s_d, s_n)[0]
            per_seed.append(aurocs)
        results[f"k{k}"] = per_seed

    with open(args.m2_metrics, encoding="utf-8") as f:
        m2 = json.load(f)
    diffs = []
    for k in C.SHOTS:
        for si in range(3):
            for c in C.COMBOS:
                old = m2["results"][f"k{k}"][si]["auroc"].get(c)
                new = results[f"k{k}"][si][c]
                if old is not None:
                    diffs.append(abs(old - new))
    repro = {"weights": args.weights, "max_abs_auroc_diff_vs_m2": max(diffs),
             "mean_abs_auroc_diff_vs_m2": sum(diffs) / len(diffs),
             "reproduced": max(diffs) < 0.02,
             "results_per_k_per_seed_per_combo": results}
    C.save_json(repro, f"{args.out}/actual_m2_repro.json")
    print(f"[repro] max |ΔAUROC| vs M2 metrics.json = {max(diffs):.5f} "
          f"(reproduced={repro['reproduced']})")


if __name__ == "__main__":
    main()
