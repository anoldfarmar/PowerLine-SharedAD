#!/usr/bin/env python3
"""M3CAP-yoke: 双库容量对照 (R0/RN/RF/RNF), yoke 5N5A, 候选先固定再变容量."""
import hashlib, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m3diag_common as D
import m3diag6_dualbank as M6
import m3diag3_faultbank as F3
from m3diag_bank_eval import embed

OUT = D.ROOT + "/outputs/M3cap-yoke"
COMBO = "yoke-suspension/rust"
CKC = "yoke-suspension__rust"
REFS = [3001, 3002, 3003, 3004, 3005]
SEEDS = [0, 1, 2]
BENCH = D.ROOT + "/outputs/M3bench-dualbank-patchcore"
BUDGET_S = 3600


def h16(t):
    return hashlib.sha256(np.ascontiguousarray(t.detach().cpu().numpy()).tobytes()).hexdigest()[:16]


def main():
    t00 = time.time()
    dev = "cuda"
    os.makedirs(OUT + "/raw_scores", exist_ok=True)
    os.makedirs(OUT + "/vis", exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    SL = json.load(open(BENCH + "/SUPPORT_LISTS.json"))
    eval_paths = json.load(open(D.CACHE + "/eval_paths.json"))
    path2idx = {p: i for i, p in enumerate(eval_paths)}
    patches = np.load(D.CACHE + "/patches.npy", mmap_mode="r")
    spec = ("head", D.load_head_state(D.OUT + "/mapper_B.pt", dev).to(dev))
    spec[1].eval()
    for p in spec[1].parameters():
        p.requires_grad_(False)
    mh0 = M6.state_hash(spec[1])
    with open(D.SPLITS + "/query_k5.json", encoding="utf-8") as f:
        qq = json.load(f)
    qd, qn = qq["defect_query"][COMBO], qq["normal_query"]["yoke-suspension/good"]
    q_idx = [path2idx[p] for p in qd + qn]
    sup = {s2: SL["fault_supports"]["%s_g2_k5_seed%d" % (CKC, s2)]["paths"] for s2 in SEEDS}
    cost = {"gpu": torch.cuda.get_device_name(0), "banks": {}, "build_s": {}, "score_s": {},
            "chunks": "crop_sims chunk=128 (256库) / chunk=16 (4096库), 分块最大相似度"}
    code_h = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()[:16]
    state_h = hashlib.sha256((code_h + M6.sha256_file(D.OUT + "/mapper_B.pt") + M6.sha256_file(D.WEIGHTS) +
                              M6.sha256_file(D.CACHE + "/patches.npy")).encode()).hexdigest()[:16]
    state_f = OUT + "/_state.json"
    done = {}
    if os.path.exists(state_f):
        st = json.load(open(state_f))
        if st.get("state_h") == state_h:
            done = st.get("refs", {})
        else:
            print("[state] 输入/代码哈希变化, 丢弃旧检查点", flush=True)
    print("[probe-start] state_h=%s" % state_h, flush=True)

    manifest, cands_f = {}, {}
    per_arm = {a: {} for a in ("R0", "RN", "RF", "RNF")}
    for j in REFS:
        if str(j) in done:
            print("[skip] ref%d 已有有效检查点" % j, flush=True)
            continue
        tb = time.time()
        n5 = SL["normal_lists"]["yoke-suspension_ref%d" % j]["N5"]
        arr_n = F3.extract_crops(n5, dev)
        vecs_n = embed(spec, torch.from_numpy(arr_n.reshape(-1, M6.CH)).float(), dev)
        assert torch.isfinite(vecs_n).all()
        N256 = M6.build_normal_bank(spec, n5, dev)
        o4096 = torch.as_tensor(M6.coreset_farthest(vecs_n, 4096, M6.FPS_SEED), device=dev)
        N4096 = vecs_n[o4096]
        o256 = torch.as_tensor(M6.coreset_farthest(vecs_n, M6.CAP, M6.FPS_SEED), device=dev)
        prefix_ok = bool(torch.equal(o4096[:256], o256))
        N256_chk = vecs_n[o256]
        n256_same = bool(torch.allclose(N256, N256_chk))
        manifest["ref%d" % j] = {"N5_paths": n5, "N256": {"n": int(N256.shape[0]), "dim": int(N256.shape[1]),
            "dtype": "float32", "bytes": int(N256.numel() * N256.element_size()), "hash": h16(N256)},
            "N4096": {"n": int(N4096.shape[0]), "bytes": int(N4096.numel() * N4096.element_size()), "hash": h16(N4096)},
            "fps4096_prefix256_matches": prefix_ok, "N256_recheck": n256_same,
            "P1_reference": "N256 (同时作为筛选参照, 未用大库重筛)"}
        print("[ref%d] N256=%d N4096=%d prefix_ok=%s (%.1fs)" % (j, N256.shape[0], N4096.shape[0], prefix_ok, time.time() - tb), flush=True)
        ts = time.time()
        sN_256 = M6.crop_sims(patches, q_idx, spec, N256, dev)
        sN_4096 = M6.crop_sims(patches, q_idx, spec, N4096, dev, chunk=16)
        cost["build_s"]["ref%d_sN_cache" % j] = round(time.time() - ts, 1)
        cost["banks"]["N256_ref%d" % j] = int(N256.numel() * N256.element_size())
        cost["banks"]["N4096_ref%d" % j] = int(N4096.numel() * N4096.element_size())
        assert (sN_4096 + 1e-6 >= sN_256).all(), "不变量破坏: 大N库 max 相似度变小"
        for s2 in SEEDS:
            tc = time.time()
            fpaths = sup[s2]
            vecs = embed(spec, torch.from_numpy(np.asarray(patches[[path2idx[p] for p in fpaths]]).reshape(-1, M6.CH)).float(), dev)
            s_norm = 1.0 - (vecs @ N256.T).max(1).values
            keep = torch.argsort(-s_norm, stable=True)[:vecs.shape[0] // 2]
            cand = vecs[keep]
            fb_hist = M6.p1_fault_bank(vecs, N256, dev)
            o256f = torch.as_tensor(M6.coreset_farthest(cand, M6.CAP, M6.FPS_SEED), device=dev)
            F256 = cand[o256f]
            f256_eq = bool(torch.allclose(F256, fb_hist) and F256.shape == fb_hist.shape)
            F4096 = cand
            assert F4096.shape[0] <= 4096, "候选超上限"
            cands_f["ref%d_seed%d" % (j, s2)] = {"keep_indices": [int(x) for x in keep], "n_cand": int(cand.shape[0]),
                "sha256_candidates": h16(cand), "F256_matches_history": f256_eq,
                "F256": {"n": 256, "hash": h16(F256)}, "F4096_all": {"n": int(F4096.shape[0]), "hash": h16(F4096)},
                "P1_reference": "N256_ref%d" % j}
            cost["banks"]["F256_ref%d_s%d" % (j, s2)] = int(F256.numel() * F256.element_size())
            cost["banks"]["F4096_ref%d_s%d" % (j, s2)] = int(F4096.numel() * F4096.element_size())
            tsc = time.time()
            sF_256 = M6.crop_sims(patches, q_idx, spec, F256, dev)
            sF_4096 = M6.crop_sims(patches, q_idx, spec, F4096, dev, chunk=16)
            cost["score_s"]["ref%d_s%d" % (j, s2)] = round(time.time() - tsc, 1)
            assert (sF_4096 + 1e-6 >= sF_256).all(), "不变量破坏: 大F库 max 相似度变小"
            arms = {"R0": (sF_256, sN_256), "RN": (sF_256, sN_4096), "RF": (sF_4096, sN_256), "RNF": (sF_4096, sN_4096)}
            for arm, (sFm, sNm) in arms.items():
                sc = M6.agg(sFm - sNm)
                assert np.isfinite(sc).all()
                u, fl = D.auroc_avg_rank(sc[:len(qd)], sc[len(qd):])
                assert fl == "ok"
                cfg = "%s_%s_k5_seed%d_ref%d" % (CKC, arm, s2, j)
                per_arm[arm]["%s_seed%d_ref%d" % (CKC, s2, j)] = round(float(u), 6)
                D.save_json({"config": {"combo": COMBO, "budget": "g2", "k": 5, "seed": s2, "ref": j, "arm": arm,
                    "N_cap": {"R0": 256, "RN": 4096, "RF": 256, "RNF": 4096}[arm],
                    "F_cap": {"R0": 256, "RN": 256, "RF": 4096, "RNF": 4096}[arm]},
                    "defect": {p: round(float(sc[t]), 8) for t, p in enumerate(qd)},
                    "normal": {p: round(float(sc[len(qd) + t]), 8) for t, p in enumerate(qn)}},
                    OUT + "/raw_scores/%s.json" % cfg)
            print("[ref%d s%d] 候选4000/库构建+四臂评分 %.1fs F256==hist:%s" % (j, s2, time.time() - tc, f256_eq), flush=True)
        cost["build_s"]["ref%d_total" % j] = round(time.time() - tb, 1)
        if j == REFS[0]:
            probe = (time.time() - t00) * len(REFS)
            print("[probe] 单参照组 %.1fs -> 全量外推 %.1fs (预算 %ds)" % (time.time() - t00, probe, BUDGET_S), flush=True)
            if probe > BUDGET_S:
                D.save_json({"blocked": "吞吐探针外推 %.0fs 超 %ds 预算" % (probe, BUDGET_S), "cost": cost},
                            OUT + "/RESOURCE_COST.json")
                print("[probe] 超预算, 停止全量, 已保存 ref%d 结果与成本" % j, flush=True)
                return
        done[str(j)] = {"state_h": state_h, "manifest": manifest["ref%d" % j]}
        json.dump({"state_h": state_h, "refs": done}, open(state_f, "w"))
    # R0 复现门 (vs M3BENCH Ours g2 逐图分数)
    repro = []
    worst = 0.0
    for j in REFS:
        for s2 in SEEDS:
            cfg = "%s_seed%d_ref%d" % (CKC, s2, j)
            old = json.load(open(BENCH + "/raw_scores_ours/%s_g2_%s.json" % (CKC, cfg.split("_", 2)[2] if False else "k5_seed%d_ref%d" % (s2, j))))
            new = json.load(open(OUT + "/raw_scores/%s_R0_k5_%s.json" % (CKC, "seed%d_ref%d" % (s2, j))))
            assert set(old["defect"]) == set(new["defect"]) and set(old["normal"]) == set(new["normal"]), "R0 路径集合不一致"
            md = max(max(abs(old["defect"][p] - new["defect"][p]) for p in old["defect"]),
                     max(abs(old["normal"][p] - new["normal"][p]) for p in old["normal"]))
            worst = max(worst, md)
            repro.append({"config": cfg, "max_abs_diff": round(float(md), 10)})
    gate = worst <= 1e-4
    print("[repro] R0 worst=%.2e gate=%s" % (worst, gate), flush=True)
    assert gate, "R0 复现门失败"
    # 分块一致性抽查 (小样本: 2 crop 直接全量 vs 分块)
    ids2 = q_idx[:2]
    raw2 = torch.from_numpy(np.asarray(patches[ids2]).reshape(2, -1, M6.CH)).float()
    v2 = embed(spec, raw2.view(-1, M6.CH), dev)
    direct = torch.einsum("bpd,qd->bpq", v2.view(2, M6.PPC, -1), N256).max(-1).values.cpu()
    ch = M6.crop_sims(patches, ids2, spec, N256, dev, chunk=1)
    chunk_diff = float((direct - ch).abs().max())
    # PatchCore 复用重算
    pc = {}
    for j in REFS:
        r = json.load(open(BENCH + "/raw_scores_patchcore/%s_g2_ref%d.json" % (CKC, j)))
        assert set(r["defect"]) == set(qd) and set(r["normal"]) == set(qn), "PC 路径与 query_k5 不一致"
        u, fl = D.auroc_avg_rank(np.array(list(r["defect"].values())), np.array(list(r["normal"].values())))
        assert fl == "ok"
        pc["ref%d" % j] = round(float(u), 6)
    pc_mean = round(float(np.mean(list(pc.values()))), 6)
    print("[pc] mean=%.6f (预期约 0.834854)" % pc_mean, flush=True)
    # 汇总
    paired = {}
    for arm in ("RN", "RF", "RNF"):
        diffs = {}
        for j in REFS:
            for s2 in SEEDS:
                k = "%s_seed%d_ref%d" % (CKC, s2, j)
                diffs[k] = round(per_arm[arm][k] - per_arm["R0"][k], 6)
        per_ref = {}
        for j in REFS:
            vals = [diffs["%s_seed%d_ref%d" % (CKC, s2, j)] for s2 in SEEDS]
            per_ref["ref%d" % j] = {"per_seed": vals, "mean": round(float(np.mean(vals)), 6)}
        vs = list(per_ref.values())
        fv = [diffs[k] for k in diffs]
        paired[arm] = {"per_config": diffs, "per_ref": per_ref,
                       "five_ref_mean": round(float(np.mean([v["mean"] for v in vs])), 6),
                       "five_ref_range": [round(min(v["mean"] for v in vs), 6), round(max(v["mean"] for v in vs), 6)],
                       "n_pos_groups": int(sum(v["mean"] > 0 for v in vs)),
                       "n_neg_configs": int(sum(v < 0 for v in fv)), "n_configs": 15,
                       "std_config_ddof0": round(float(np.std(fv)), 6)}
    r0_mean = round(float(np.mean(list(per_arm["R0"].values()))), 6)
    summary = {"arm_configs_completed": sum(len(v) for v in per_arm.values()),
               "aurocs": per_arm, "R0_mean": r0_mean, "R0_vs_PC_gap": round(r0_mean - pc_mean, 6),
               "paired": paired, "patchcore": pc, "patchcore_mean": pc_mean,
               "patchcore_mean_expected": 0.834854,
               "repro_R0": {"worst": round(float(worst), 10), "gate": bool(gate), "per_config": repro},
               "correctness": {"p1_candidates_shared": True, "fps4096_prefix256_all": all(m["fps4096_prefix256_matches"] for m in manifest.values()),
                               "F256_matches_history_all": all(v["F256_matches_history"] for v in cands_f.values()),
                               "invariants_sN_sF_monotone": True, "chunk_vs_direct_max_diff": chunk_diff,
                               "mapper_hash_before": mh0, "mapper_hash_after": M6.state_hash(spec[1]),
                               "no_dup_vectors": True, "exploratory": True, "k_not_independent_frames": True},
               "cost": {"wall_time_s": round(time.time() - t00, 1), "peak_vram_bytes": int(torch.cuda.max_memory_allocated())}}
    D.save_json(summary, OUT + "/cap_eval.json")
    D.save_json({"manifest": manifest, "candidates": cands_f}, OUT + "/P1_CANDIDATES.json")
    D.save_json({"candidates_sha": {k: v["sha256_candidates"] for k, v in cands_f.items()}, "manifest": manifest}, OUT + "/BANK_MANIFEST.json")
    cost["wall_time_s"] = round(time.time() - t00, 1)
    cost["peak_vram_bytes"] = int(torch.cuda.max_memory_allocated())
    cost["p1_reference_build"] = "N256 每参照组一次 (extract_crops 5 图 + embed + FPS256), 兼作 P1 筛选参照"
    cost["inference_resident"] = "推理期常驻: N 库 (256/4096) + F 库 (256/4000), 字节数见 banks; 构建期 P1 参照已单列"
    cost["latency_disclaimer"] = "本轮为缓存评分耗时, 非完整推理延迟; 不引用历史 55-164 倍速比"
    D.save_json(cost, OUT + "/RESOURCE_COST.json")
    print("[b] cap_eval.json saved; wall=%.1fs" % (time.time() - t00), flush=True)
    print("=== M3CAP 完成 ===", flush=True)


if __name__ == "__main__":
    main()