#!/usr/bin/env python3
"""M3MECH-followup-v2 §七: AnomalyDINO 适配入口 — 只准备, 不运行.
--check: 静态检查 (冻结清单存在性/哈希, 依赖可见性, 克隆状态) + 生成 baseline_spec.json.
明确禁止: 模型下载, 特征提取, 建库, CUDA预热, 评测. 待用户明确启动后再做冒烟/精度门/成本探针."""
import argparse, hashlib, json, os, subprocess, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m3diag_common as D

DEFAULT_REPO = "https://github.com/dammsi/AnomalyDINO"
HEAD_SHA = "b9d1c2648e3a5247437d4d953d907a8f3d994457"  # main HEAD, 记录于 2025-09-30
SPEC = {
    "repo": DEFAULT_REPO, "commit": HEAD_SHA,
    "model": {"name": "dinov2_vits14", "resolution": 448, "note": "官方默认; 不使用 vitg14 (范围取消 SubspaceAD/DINOv2-giant)"},
    "scoring": {"patch_metric": "L2_normalized (1 - cosine)", "knn_neighbors": 1,
                 "image_score": "mean_top1p(patch distances) 官方函数, 不修改",
                 "library": "faiss (可选 --faiss_on_cpu)"},
    "preprocessing": {"official": "agnostic = PCA 前景掩码 + 参考图旋转增广 (8角度)",
                       "adapter_rule": "保留官方特征/评分/预处理逻辑; 增广与否按官方 agnostic 口径, 在启动前冻结记录"},
    "data_binding": {"supports": "必须消费 M3FULL/M3MECH 冻结 10N 支持名单 (shots=10 恰为名单, 禁官方自动抽样)",
                      "queries": "同 M3MECH 查询与标签; 逐图原始分数可追溯 (官方 save_patch_dists=True 默认)",
                      "forbidden_modes": ["官方自动抽样正常支持", "整查询集建库的批量零样本模式"]},
    "env": {"torch": "2.8.0+cu128 (现有)", "timm": "1.0.29 (现有)",
             "anomalydino_env": "用户准备中, 本任务不创建不改动", "weight_cache": "~/.cache/torch/hub/checkpoints/ (dinov2_vits14_pretrain.pth, 启动时才允许下载)",
             "repo_clone_hint": "/root/autodl-tmp/AnomalyDINO (待用户放置, 本任务不克隆)"},
    "planned_artifacts": {
        "scores": "<result>/anomalydino_scores/{combo}_AD-{arm}_10N_ref{3001..3005}.json (逐图 official 原始分数 + patch 距离可选)",
        "bank": "<result>/anomalydino_scores/BANK_MANIFEST.json", "report": "<result>/BASELINE_REPORT.md"},
    "gates_when_started": ["环境冒烟 (import+权重缓存检查)", "批量一致性门 (bs1 vs bsN 特征)",
                            "锚点双建库门", "成本探针 (wall/VRAM)"]}


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:32]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--out", default="/root/autodl-tmp/version10-12/M3MECH-followup-v2")
    args = ap.parse_args()
    if not args.check:
        print("本入口只提供 --check 静态检查; 运行 AnomalyDINO 需用户明确启动 (另行冒烟/门/探针)")
        return
    st = {"frozen_lists": {}, "deps": {}, "clone": None}
    SL = D.ROOT + "/outputs/M3full-frozen/SUPPORT_LISTS.json"
    st["frozen_lists"]["support_lists"] = {"path": SL, "sha256": sha(SL), "ok": os.path.exists(SL)}
    st["frozen_lists"]["query_k5"] = {"path": D.SPLITS + "/query_k5.json", "sha256": sha(D.SPLITS + "/query_k5.json"), "ok": True}
    for mod in ("torch", "faiss", "cv2", "anomalydino_src"):
        try:
            __import__(mod)
            st["deps"][mod] = "available"
        except Exception as e:
            st["deps"][mod] = "missing (%s)" % type(e).__name__
    for cand in ("/root/autodl-tmp/AnomalyDINO", "/root/AnomalyDINO"):
        if os.path.isdir(cand):
            try:
                h = subprocess.check_output(["git", "-C", cand, "rev-parse", "HEAD"]).decode().strip()
                st["clone"] = {"path": cand, "head": h, "matches_recorded": h.startswith(HEAD_SHA[:12])}
            except Exception:
                st["clone"] = {"path": cand, "head": None}
    st["cuda_touched"] = False
    st["note"] = "静态检查未加载任何模型/权重, 未触碰 CUDA, 未下载任何文件"
    with open(os.path.join(args.out, "baseline_spec.json"), "w") as f:
        json.dump({"spec": SPEC, "check": st}, f, ensure_ascii=False, indent=1)
    print(json.dumps({"deps": st["deps"], "clone": st["clone"], "support_lists_ok": st["frozen_lists"]["support_lists"]["ok"]}, ensure_ascii=False))
    print("baseline_spec.json 已生成 (仅准备; 未运行任何 AnomalyDINO 组件)")


if __name__ == "__main__":
    main()
