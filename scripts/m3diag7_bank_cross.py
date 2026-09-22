#!/usr/bin/env python3
"""M3DIAG7: N-F 双库完整交叉诊断 (合同 M3DIAG7-bank-cross-v1).
五个正常库 N_j (j=3001..3005) 与对应 P1 故障库 F_i (由 N_i 筛选) 固定后做 5x5 交叉评分.
顺序: 冻结输入 -> 正确性检查 -> fp16 复用试验 -> FP32 对角复现门(1e-4) -> 非对角线 -> 统计.
交叉评分绝不重建故障库. 无新参数. 探索性诊断.
"""
import hashlib, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m3diag_common as D
import m3diag6_dualbank as M6
from m3diag_bank_eval import embed

ROOT = D.ROOT
OUT7 = ROOT + "/outputs/M3diag7-bank-cross"
OUT6 = M6.OUT6
COMBOS, KS, SUPPORT_SEEDS, REF_SEEDS = M6.COMBOS, M6.KS, M6.SUPPORT_SEEDS, M6.REF_SEEDS
REPRO_TOL = 1e-4


def bank_hash(t):
    return hashlib.sha256(np.ascontiguousarray(t.detach().cpu().numpy()).tobytes()).hexdigest()[:16]


def mat_stats(m):
    a = np.asarray(m, float)
    return {"std": round(float(a.std()), 6), "range": [round(float(a.min()), 6), round(float(a.max()), 6)]}


def row_col_sensitivity(m):
    """固定 i 跨 j (行) 与固定 j 跨 i (列) 的 std/range, 汇总五行/五列的均值与最大."""
    a = np.asarray(m, float)
    rows = [mat_stats(a[i, :]) for i in range(a.shape[0])]
    cols = [mat_stats(a[:, j]) for j in range(a.shape[1])]
    return {
        "row": {"mean_std": round(float(np.mean([r["std"] for r in rows])), 6),
                "max_std": round(float(np.max([r["std"] for r in rows])), 6),
                "mean_range": round(float(np.mean([r["range"][1] - r["range"][0] for r in rows])), 6),
                "max_range": round(float(np.max([r["range"][1] - r["range"][0] for r in rows])), 6)},
        "col": {"mean_std": round(float(np.mean([c["std"] for c in cols])), 6),
                "max_std": round(float(np.max([c["std"] for c in cols])), 6),
                "mean_range": round(float(np.mean([c["range"][1] - c["range"][0] for c in cols])), 6),
                "max_range": round(float(np.max([c["range"][1] - c["range"][0] for c in cols])), 6)}}


def interaction(m):
    a = np.asarray(m, float)
    e = a - a.mean(1, keepdims=True) - a.mean(0, keepdims=True) + a.mean()
    return {"rms": round(float(np.sqrt((e ** 2).mean())), 6),
            "max_abs": round(float(np.abs(e).max()), 6),
            "matrix": [[round(float(x), 6) for x in row] for row in e]}


def diag_effect(m):
    a = np.asarray(m, float)
    d = np.diag(a).mean() - (a.sum() - np.diag(a).sum()) / (a.size - a.shape[0])
    return round(float(d), 6)


def main():
    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    assert dev == "cuda", "need GPU"
    os.makedirs(OUT7 + "/raw_scores7", exist_ok=True)
    torch.cuda.reset_peak_memory_stats()

    rg = json.load(open(M6.OUT4 + "/ref_groups.json"))
    eval_paths = json.load(open(D.CACHE + "/eval_paths.json"))
    path2idx = {p: i for i, p in enumerate(eval_paths)}
    patches = np.load(D.CACHE + "/patches.npy", mmap_mode="r")
    spec = ("head", D.load_head_state(D.OUT + "/mapper_B.pt", dev).to(dev))
    head = spec[1]; head.eval()
    for p in head.parameters():
        p.requires_grad_(False)

    # ---------- 1. 冻结输入 ----------
    m6_jsons = sorted(os.listdir(OUT6 + "/raw_scores6"))
    frozen = {
        "code_commit": os.popen("git rev-parse HEAD").read().strip(),
        "weights": {"path": D.WEIGHTS, "sha256": M6.sha256_file(D.WEIGHTS)},
        "mapper_B": {"path": D.OUT + "/mapper_B.pt", "sha256": M6.sha256_file(D.OUT + "/mapper_B.pt")},
        "cache_patches_npy": {"path": D.CACHE + "/patches.npy", "sha256": M6.sha256_file(D.CACHE + "/patches.npy")},
        "cache_eval_paths": {"path": D.CACHE + "/eval_paths.json", "sha256": M6.sha256_file(D.CACHE + "/eval_paths.json")},
        "ref_groups_json": {"path": M6.OUT4 + "/ref_groups.json", "sha256": M6.sha256_file(M6.OUT4 + "/ref_groups.json")},
        "m3diag6_archive": {f: M6.sha256_file(OUT6 + "/raw_scores6/" + f) for f in m6_jsons},
        "m3diag6_eval": {"path": OUT6 + "/dualbank_eval.json", "sha256": M6.sha256_file(OUT6 + "/dualbank_eval.json")},
        "bank_params": {"cap": M6.CAP, "fps_seed": M6.FPS_SEED, "keep_frac": 0.5,
                        "N_build": "M3DIAG4 build_bank 逐字复用 (组内10裁剪 embed -> FPS 256 seed0)",
                        "F_build": "P1 逐字复用: 正常距离排序(稳定) -> 前50% -> FPS 256 seed0, F_i 由 N_i 筛选"},
        "cross_spec": {"rows": "F_i (故障库, 由 N_i 筛选)", "cols": "N_j (正常库)",
                       "rule": "交叉评分时绝不重建故障库; 先固定 5 N + 5 F 再交叉",
                       "C_formula": "mean(top8(sFi - sNj)), 先逐patch差值再top8, 保留负数"},
        "imported_scripts_sha256": {s: M6.sha256_file(ROOT + "/scripts/" + s) for s in [
            "m3diag7_bank_cross.py", "m3diag6_dualbank.py", "m3diag_common.py", "m3diag3_faultbank.py",
            "m3diag_bank_eval.py", "m3_build_patch_bank.py", "m3_common.py"]},
    }
    D.save_json(frozen, OUT7 + "/FROZEN_INPUTS.json")
    print("[m7] FROZEN_INPUTS.json saved (%d M3DIAG6 archives bound)" % len(m6_jsons), flush=True)

    # ---------- 2. 正确性检查 ----------
    ut = M6.unit_tests()
    ut["direction_all_ok"] = all(ut[k] for k in ("direction_A", "direction_B", "direction_C"))
    a0 = D.ASSET_OF[COMBOS[0]]
    nb0 = M6.build_normal_bank(spec, rg["assets"][a0]["groups"]["3001"], dev)
    with open(D.SPLITS + "/query_k10.json", encoding="utf-8") as f:
        qq = json.load(f)
    probe = [path2idx[p] for p in qq["defect_query"][COMBOS[0]][:3]]
    s_chunk = M6.crop_sims(patches, probe, spec, nb0, dev, chunk=128)
    s_direct = M6.crop_sims(patches, probe, spec, nb0, dev, chunk=10 ** 9)
    ut["chunk_vs_direct_max_abs_diff"] = float((s_chunk - s_direct).abs().max())
    ut["chunk_vs_direct_ok"] = bool(ut["chunk_vs_direct_max_abs_diff"] <= 1e-6)
    mapper_hash_before = M6.state_hash(head)
    ut["mapper_state_hash_before"] = mapper_hash_before
    assert ut["direction_all_ok"] and ut["diff_before_top8_is_different_semantics"] and ut["chunk_vs_direct_ok"], ut
    print("[m7] unit tests ok", flush=True)

    # ---------- 3. fp16 复用试验 (存档缓存能否直接过 1e-4 门) ----------
    fp16_trials = []
    for cfg, k_, s2_ in (("glass-insulator__missing-cap__k10_seed0_ref3001", 10, 0),
                         ("yoke-suspension__rust__k5_seed2_ref3005", 5, 2)):
        z = np.load(OUT6 + "/raw_scores6/patches_%s.npz" % cfg)
        c_fp16 = M6.agg(torch.from_numpy(z["sF_defect"].astype(np.float32)) - torch.from_numpy(z["sN_defect"].astype(np.float32)))
        arc = json.load(open(OUT6 + "/raw_scores6/%s.json" % cfg))
        old = np.array([v[2] for v in arc["defect"].values()])
        md = float(np.abs(c_fp16 - old).max())
        fp16_trials.append({"config": cfg, "max_abs_diff": round(md, 8), "pass_1e-4": bool(md <= REPRO_TOL)})
    fp16_ok = all(t["pass_1e-4"] for t in fp16_trials)
    print("[m7] fp16 reuse trial: %s -> %s" % (fp16_trials, "USE FP16" if fp16_ok else "FP32 REBUILD"), flush=True)

    # ---------- 4. FP32 交叉: 相位1 = 建库+相似度+对角门 (不写非对角线) ----------
    stash = {}
    diag_repro = []
    worst = 0.0
    bank_manifest = []
    for combo in COMBOS:
        a = D.ASSET_OF[combo]
        for k in KS:
            for s2 in SUPPORT_SEEDS:
                sup, qd, qn, qd_idx, qn_idx = M6.load_split(combo, a, k, s2, path2idx)
                vecs_sup = embed(spec, torch.from_numpy(np.asarray(patches[[path2idx[p] for p in sup]]).reshape(-1, M6.CH)).float(), dev)
                N_banks, N_hash = {}, {}
                for j in REF_SEEDS:
                    N_banks[j] = M6.build_normal_bank(spec, rg["assets"][a]["groups"][str(j)], dev)
                    N_hash[j] = bank_hash(N_banks[j])
                F_banks, F_hash = {}, {}
                for i in REF_SEEDS:
                    F_banks[i] = M6.p1_fault_bank(vecs_sup, N_banks[i], dev)
                    F_hash[i] = bank_hash(F_banks[i])
                sF = {i: {"defect": M6.crop_sims(patches, qd_idx, spec, F_banks[i], dev),
                          "normal": M6.crop_sims(patches, qn_idx, spec, F_banks[i], dev)} for i in REF_SEEDS}
                sN = {j: {"defect": M6.crop_sims(patches, qd_idx, spec, N_banks[j], dev),
                          "normal": M6.crop_sims(patches, qn_idx, spec, N_banks[j], dev)} for j in REF_SEEDS}
                for i in REF_SEEDS:
                    assert bank_hash(F_banks[i]) == F_hash[i], "F_i changed during sims"
                for j in REF_SEEDS:
                    assert bank_hash(N_banks[j]) == N_hash[j], "N_j changed during sims"
                m6 = {r: json.load(open(OUT6 + "/raw_scores6/%s__k%d_seed%d_ref%d.json"
                                        % (combo.replace("/", "__"), k, s2, r))) for r in REF_SEEDS}
                A_store, B_store = {}, {}
                for i in REF_SEEDS:
                    A_store[i] = {"defect": M6.agg(sF[i]["defect"]), "normal": M6.agg(sF[i]["normal"])}
                for j in REF_SEEDS:
                    B_store[j] = {"defect": M6.agg(1 - sN[j]["defect"]), "normal": M6.agg(1 - sN[j]["normal"])}
                C_store, sFmax_store, sNmax_store = {}, {}, {}
                for i in REF_SEEDS:
                    sFmax_store[i] = {"defect": sF[i]["defect"].max(1).values.numpy(),
                                      "normal": sF[i]["normal"].max(1).values.numpy()}
                for j in REF_SEEDS:
                    sNmax_store[j] = {"defect": sN[j]["defect"].max(1).values.numpy(),
                                      "normal": sN[j]["normal"].max(1).values.numpy()}
                for i in REF_SEEDS:
                    for j in REF_SEEDS:
                        C_d = M6.agg(sF[i]["defect"] - sN[j]["defect"])
                        C_n = M6.agg(sF[i]["normal"] - sN[j]["normal"])
                        assert len(C_d) == len(qd) and len(C_n) == len(qn)
                        C_store[(i, j)] = {"defect": C_d, "normal": C_n}
                del sF, sN, F_banks, N_banks, vecs_sup
                # 对角门: C_ii / A_i / B_j vs M3DIAG6 存档, 逐路径 1e-4
                for r in REF_SEEDS:
                    arc = m6[r]
                    assert set(arc["defect"].keys()) == set(qd) and set(arc["normal"].keys()) == set(qn), "path set mismatch"
                    pairs = [("A", A_store[r], 0), ("B", B_store[r], 1), ("C_ii", C_store[(r, r)], 2)]
                    md = 0.0
                    for nm, store, col in pairs:
                        for side, arr in store.items():
                            old = np.array([arc[side][p][col] for p in (qd if side == "defect" else qn)])
                            md = max(md, float(np.abs(old - arr).max()))
                    worst = max(worst, md)
                    diag_repro.append({"config": "%s__k%d_seed%d_ref%d" % (combo.replace("/", "__"), k, s2, r),
                                       "max_abs_diff_A_B_Cii": round(md, 10), "n_paths": len(qd) + len(qn)})
                    print("[diag] %s__k%d_seed%d_ref%d max|diff|=%.2e" % (combo.replace("/", "__"), k, s2, r, md), flush=True)
                stash[(combo, k, s2)] = {"qd": qd, "qn": qn, "A": A_store, "B": B_store,
                                         "C": C_store, "sFmax": sFmax_store, "sNmax": sNmax_store,
                                         "N_hash": N_hash, "F_hash": F_hash}
                bank_manifest.append({"combo": combo, "k": k, "support_seed": s2,
                                      "N_j": {str(j): {"sha256_16": N_hash[j],
                                                       "built_from": "M3DIAG4 ref group %d (10 crops, 名单沿用)" % j} for j in REF_SEEDS},
                                      "F_i": {str(i): {"sha256_16": F_hash[i], "filtered_by_N": i} for i in REF_SEEDS},
                                      "bank_bytes_each": 131072})
                print("[m7] group %s k%d seed%d done (diagonal gate within-group worst so far %.2e)"
                      % (combo.replace("/", "__"), k, s2, worst), flush=True)
    gate_ok = worst <= REPRO_TOL
    print("[m7] DIAGONAL GATE: worst=%.3e tol=%.0e -> %s" % (worst, REPRO_TOL, "PASS" if gate_ok else "FAIL"), flush=True)
    assert gate_ok, "diagonal repro gate failed - 不生成非对角线, 按重试规则诊断"

    # ---------- 5. 相位2 = 门通过后写 300 配置逐图分数 ----------
    aurocs = {}
    ab_aurocs = {}
    n_written = 0
    for (combo, k, s2), g in stash.items():
        qd, qn = g["qd"], g["qn"]
        for i in REF_SEEDS:
            ua, fl = D.auroc_avg_rank(g["A"][i]["defect"], g["A"][i]["normal"])
            assert fl == "ok"
            ab_aurocs.setdefault((combo, k, s2), {"A": {}, "B": {}})["A"][i] = round(float(ua), 6)
        for j in REF_SEEDS:
            ub, fl = D.auroc_avg_rank(g["B"][j]["defect"], g["B"][j]["normal"])
            assert fl == "ok"
            ab_aurocs[(combo, k, s2)]["B"][j] = round(float(ub), 6)
        for i in REF_SEEDS:
            for j in REF_SEEDS:
                C_d, C_n = g["C"][(i, j)]["defect"], g["C"][(i, j)]["normal"]
                u, fl = D.auroc_avg_rank(C_d, C_n)
                assert fl == "ok"
                cfg = "%s__k%d_seed%d_Fi%d_Nj%d" % (combo.replace("/", "__"), k, s2, i, j)
                aurocs[cfg] = round(float(u), 6)
                rec = {"config": {"combo": combo, "k": k, "support_seed": s2, "fault_ref_i": i, "normal_ref_j": j,
                                  "is_diagonal": i == j, "n_defect": len(qd), "n_normal": len(qn)},
                       "defect": {p: [round(float(C_d[t]), 8), round(float(g["sFmax"][i]["defect"][t]), 8),
                                      round(float(g["sNmax"][j]["defect"][t]), 8)] for t, p in enumerate(qd)},
                       "normal": {p: [round(float(C_n[t]), 8), round(float(g["sFmax"][i]["normal"][t]), 8),
                                      round(float(g["sNmax"][j]["normal"][t]), 8)] for t, p in enumerate(qn)},
                       "score_field_order": ["C_ij", "sFi_max", "sNj_max"]}
                D.save_json(rec, OUT7 + "/raw_scores7/%s.json" % cfg)
                n_written += 1
    print("[m7] wrote %d cross score files" % n_written, flush=True)
    mapper_hash_after = M6.state_hash(head)
    assert mapper_hash_after == mapper_hash_before, "mapper state changed"
    ut["mapper_state_hash_after"] = mapper_hash_after
    ut["mapper_state_unchanged"] = True

    # ---------- 6. 统计 (预先固定) ----------
    stats = {}
    for combo in COMBOS:
        for k in KS:
            ck = "%s__k%d" % (combo.replace("/", "__"), k)
            fam = {}
            for fam_name, base in (("U", None), ("dF", "A"), ("dN", "B")):
                mats = {}
                for s2 in SUPPORT_SEEDS:
                    ab = ab_aurocs[(combo, k, s2)]
                    m = [[aurocs["%s__k%d_seed%d_Fi%d_Nj%d" % (combo.replace("/", "__"), k, s2, i, j)]
                          - (0 if base is None else ab[base][i if base == "A" else j]) for j in REF_SEEDS] for i in REF_SEEDS]
                    mats["seed%d" % s2] = [[round(float(x), 6) for x in row] for row in m]
                arr = np.array([mats["seed%d" % s2] for s2 in SUPPORT_SEEDS])
                mats["mean"] = [[round(float(x), 6) for x in row] for row in arr.mean(0)]
                mats["std"] = [[round(float(x), 6) for x in row] for row in arr.std(0)]
                fam[fam_name] = mats
            sens, inter, deff = {}, {}, {}
            for s2 in SUPPORT_SEEDS:
                u = np.array(fam["U"]["seed%d" % s2])
                dfm = np.array(fam["dF"]["seed%d" % s2])
                sens["seed%d" % s2] = {"U_row": row_col_sensitivity(u), "U_col": row_col_sensitivity(u.T),
                                       "dF_row": row_col_sensitivity(dfm), "dF_col": row_col_sensitivity(dfm.T)}
                inter["seed%d" % s2] = interaction(u)
                deff["seed%d" % s2] = diag_effect(u)
            de = [deff["seed%d" % s2] for s2 in SUPPORT_SEEDS]
            deff["pooled"] = {"mean": round(float(np.mean(de)), 6), "std_ddof0": round(float(np.std(de)), 6)}
            # 索引一致性验证: 从 aurocs 键解析回填矩阵应与 fam 完全一致 (防 i/j 混淆)
            chk = all(abs(fam["U"]["mean"][ii][jj] - float(np.mean([aurocs["%s__k%d_seed%d_Fi%d_Nj%d"
                      % (combo.replace("/", "__"), k, s2, REF_SEEDS[ii], REF_SEEDS[jj])] for s2 in SUPPORT_SEEDS]))) < 5e-7
                      for ii in range(5) for jj in range(5))
            stats[ck] = {"matrices": fam, "sensitivity_per_seed": sens,
                         "interaction_per_seed": inter, "diag_effect": deff, "index_identity_ok": bool(chk)}
            assert chk, "matrix index identity failed"
            print("[m7] stats %s done" % ck, flush=True)

    summary = {
        "cross_configs_completed": n_written,
        "diag_repro_passed": 1 if gate_ok else 0,
        "diagonal_repro": {"tol": REPRO_TOL, "configs": len(diag_repro), "worst_max_abs_diff": round(float(worst), 10),
                           "detail": diag_repro},
        "fp16_reuse_trial": {"trials": fp16_trials, "verdict": "use_fp16" if fp16_ok else "fp32_rebuild_per_contract",
                             "note": "M3DIAG6 逐patch存档为 FP16; 试验未过 1e-4 则按合同以 FP32 重建全部相似度"},
        "aurocs": aurocs,
        "ab_aurocs": {"%s__k%d_seed%d" % (c.replace("/", "__"), k, s2): v
                      for (c, k, s2), v in ab_aurocs.items()},
        "stats": stats,
        "correctness": ut,
        "b_reuse_note": "B_j 与支持种子无关, 同查询+同正常库复用; A_i 随故障库(种子)变化; 重复引用不计独立实验",
        "exploratory": True,
        "cost": {"wall_time_s": round(time.time() - t0, 1),
                 "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
                 "bank_bytes_each": 131072,
                 "note": "N/F 库逐组重建 (FP32), 组内常驻 10 个库; 不声称端侧加速"},
    }
    D.save_json(summary, OUT7 + "/cross_eval.json")
    D.save_json(bank_manifest, OUT7 + "/BANK_MANIFEST.json")
    D.save_json({"diagonal_repro": summary["diagonal_repro"], "fp16_reuse_trial": summary["fp16_reuse_trial"],
                 "correctness": ut, "exploratory": True}, OUT7 + "/repro_and_correctness.json")
    print("[m7] cross_eval.json + BANK_MANIFEST.json saved; wall=%.1fs" % (time.time() - t0), flush=True)
    print("=== 全部完成 ===", flush=True)


if __name__ == "__main__":
    main()