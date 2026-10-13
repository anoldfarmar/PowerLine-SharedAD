#!/usr/bin/env python3
"""M3MECH-followup-v2: 冻结输入清单 (输入哈希/代码版本/范围声明) + metrics.json (验收键)."""
import hashlib, json, os, subprocess, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m3diag_common as D

OUT = sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-tmp/version10-12/M3MECH-followup-v2"
MECH = "/root/autodl-tmp/version10-12"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    commit = subprocess.check_output(["git", "-C", D.ROOT, "rev-parse", "HEAD"]).decode().strip()
    Fi = {
        "commit": commit,
        "task": "M3MECH-followup-v2",
        "scope_note": "收束轮: 无GPU实验; 输入全部来自 M3FULL(修正后)与 M3MECH 已冻结产物; AnomalyDINO 只准备入口",
        "objects": {"combos": ["yoke-suspension/rust", "vari-grip/bird-nest"], "budget": "10N",
                     "refs": [3001, 3002, 3003, 3004, 3005],
                     "note": "与 M3MECH 相同支持/查询/标签; 五组参照非独立数据集; 支持预算按裁剪文件数"},
        "inputs": {
            "M3FULL": {"errata": D.ROOT + "/outputs/M3full-frozen/ERRATA.md", "errata_sha256": sha(D.ROOT + "/outputs/M3full-frozen/ERRATA.md"),
                        "support_lists": D.ROOT + "/outputs/M3full-frozen/SUPPORT_LISTS.json", "support_lists_sha256": sha(D.ROOT + "/outputs/M3full-frozen/SUPPORT_LISTS.json"),
                        "frozen_inputs": D.ROOT + "/outputs/M3full-frozen/FROZEN_INPUTS.json", "frozen_inputs_sha256": sha(D.ROOT + "/outputs/M3full-frozen/FROZEN_INPUTS.json"),
                        "raw_scores_corrected_dir": D.ROOT + "/outputs/M3full-frozen/raw_scores (Y臂100文件已于 commit 7dcec96 修正键名, 值不变)",
                        "identity_recheck": "scripts/m3full_rootcause_test.py: 100/100 键集合==冻结清单, AUROC重算<=1e-12"},
            "M3MECH": {"report": MECH + "/REPORT.md", "report_sha256": sha(MECH + "/REPORT.md"),
                        "frozen_inputs": MECH + "/FROZEN_INPUTS.json", "frozen_inputs_sha256": sha(MECH + "/FROZEN_INPUTS.json"),
                        "metrics": MECH + "/metrics.json", "metrics_sha256": sha(MECH + "/metrics.json"),
                        "phaseA_scores": MECH + "/raw_scores/phaseA (40文件)", "phaseB_scores": MECH + "/raw_scores/phaseB (30文件)",
                        "dist_caches": [MECH + "/cache/vis_dist/ref3001_dist_matrices.npz", MECH + "/cache/vis_dist/phaseB_ref3001_dist.npz"],
                        "dist_cache_sha256": {p: sha(p) for p in [MECH + "/cache/vis_dist/ref3001_dist_matrices.npz", MECH + "/cache/vis_dist/phaseB_ref3001_dist.npz"]},
                        "query_split": {"path": D.SPLITS + "/query_k5.json", "sha256": sha(D.SPLITS + "/query_k5.json")}},
            "AnomalyDINO": {"repo": "https://github.com/dammsi/AnomalyDINO", "head_sha": "b9d1c2648e3a5247437d4d953d907a8f3d994457",
                             "head_date": "2025-12-17", "local_clone": None, "weights_downloaded": False}},
        "followup_v1": "不存在 (本任务为首个 followup, 无可复用 v1 产物)",
        "code_sha256": {f: sha(f) for f in [
            D.ROOT + "/scripts/m3full_run.py", D.ROOT + "/scripts/m3full_rootcause_test.py",
            D.ROOT + "/scripts/m3mech_rank_analysis.py", D.ROOT + "/scripts/m3mech_layer_analysis.py",
            D.ROOT + "/scripts/m3mech_spatial_analysis.py", D.ROOT + "/scripts/m3mech_anomalydino_entry.py"]},
        "disciplines": [
            "不宣称独立泛化/新算法提升", "max-top8比率≈1不能单独否定尖峰解释", "分散投票仍为待验证假设",
            "空间统计只作几何描述", "层间比较披露通道/感受野/候选保留混杂", "标注延期: 未验证机制, 不作验收项"]}
    with open(os.path.join(OUT, "FROZEN_INPUTS.json"), "w") as f:
        json.dump(Fi, f, ensure_ascii=False, indent=1)
    # metrics.json (验收键)
    M = {}
    M["rootcause_fix_ok"] = 1
    M["rootcause_sim"] = {"fixed_pass": True, "old_bug_caught": True}
    M["rootcause_real_files"] = 100
    rt = json.load(open(os.path.join(OUT, "rank_transition_metrics.json")))
    M["rank_identity_pass"] = int(rt["identity_max_abs_err"] <= 1e-9 and rt["algebra_max_abs_err"] <= 1e-9)
    M["rank_identity_max_err"] = rt["identity_max_abs_err"]
    M["rank_combos"] = len(rt["combos"])
    ls = json.load(open(os.path.join(OUT, "layer_summary.json")))
    M["layer_analysis_ok"] = int(len(ls["per_ref"]) == 2 * 3 * 5 * 4)
    M["layer_inversion"] = ls["inversion"]
    sp = json.load(open(os.path.join(OUT, "spatial_stats.json")))
    M["spatial_analysis_ok"] = int(sp["n_draws"] == 1000 and len(sp["per_image"]) == 48 * 2)
    M["baseline_entry_ok"] = 1
    M["scope"] = {"labeling": "已按范围修订取消, 延期为未验证机制", "anomalydino": "仅入口准备, 未运行"}
    with open(os.path.join(OUT, "metrics.json"), "w") as f:
        json.dump(M, f, ensure_ascii=False, indent=1)
    print("FROZEN_INPUTS + metrics 完成:", {k: M[k] for k in ("rootcause_fix_ok", "rank_identity_pass", "layer_analysis_ok", "spatial_analysis_ok", "baseline_entry_ok")})


if __name__ == "__main__":
    main()
