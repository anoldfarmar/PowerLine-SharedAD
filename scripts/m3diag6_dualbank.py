#!/usr/bin/env python3
"""M3DIAG6: 正常-故障双库三种评分 (A/B/C) 探索性诊断 (合同 M3DIAG6-dualbank-score-v1).
顺序: 冻结输入 -> 正确性检查 -> 复现门(1e-4, 60 配置逐路径 + seed3000 锚点) -> A/B/C 评测 -> 汇总.
库全部冻结: N = M3DIAG4 参照组建库, F = P1; 只换评分. 无新增可调参数. 负结果全保留.
"""
import hashlib, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m3diag_common as D
import m3_common as C
import m3diag3_faultbank as F3
from m3diag_bank_eval import embed
from m3_build_patch_bank import coreset_farthest

PATCH_TOPK = D.PATCH_TOPK

ROOT = D.ROOT
OUT6 = ROOT + "/outputs/M3diag6-dualbank"
OUT4 = ROOT + "/outputs/M3diag4"
OUT3 = ROOT + "/outputs/M3diag3"
COMBOS = F3.COMBOS
KS = [10, 5]
SUPPORT_SEEDS = [0, 1, 2]
REF_SEEDS = [3001, 3002, 3003, 3004, 3005]
REPRO_TOL = 1e-4
FPS_SEED = F3.FPS_SEED
HP, CH, PPC = 40, 64, 1600
CAP = 256


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def state_hash(head):
    h = hashlib.sha256()
    for k in sorted(head.state_dict()):
        h.update(k.encode())
        h.update(head.state_dict()[k].detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def crop_sims(patches, idxs, spec, bank, dev, chunk=128):
    """每 crop 每 patch 对库的最大余弦, 返回 (n_crops, 1600) fp32 (CPU)."""
    bk = torch.as_tensor(bank, dtype=torch.float32, device=dev)
    outs = []
    for i in range(0, len(idxs), chunk):
        ids = list(idxs[i:i + chunk])
        raw = torch.from_numpy(np.asarray(patches[ids]).reshape(len(ids), -1, CH)).float()
        nb, npn = raw.shape[0], raw.shape[1]
        v = embed(spec, raw.view(-1, CH), dev).view(nb, npn, -1)
        with torch.no_grad():
            sims = torch.einsum("bpd,qd->bpq", v, bk).max(-1).values
        assert torch.isfinite(sims).all()
        outs.append(sims.cpu())
    return torch.cat(outs, 0) if outs else torch.zeros(0, PPC)


def agg(sim):
    """图像分 = top-k 最大余弦均值 (k=PATCH_TOPK=8), 越高越异常."""
    return torch.topk(sim, PATCH_TOPK, dim=1).values.mean(1).numpy()


def build_normal_bank(spec, paths, dev):
    arr = F3.extract_crops(paths, dev)
    vecs = embed(spec, torch.from_numpy(arr.reshape(-1, CH)).float(), dev)
    assert torch.isfinite(vecs).all()
    return vecs[torch.as_tensor(coreset_farthest(vecs, CAP, FPS_SEED), device=dev)]


def p1_fault_bank(vecs, nb, dev):
    """P1: 正常距离排序 -> 保留前 50% -> FPS (规则与 M3DIAG3/4 逐字一致)."""
    s_norm = 1.0 - (vecs @ nb.T).max(1).values
    assert torch.isfinite(s_norm).all()
    keep = torch.argsort(-s_norm, stable=True)[:vecs.shape[0] // 2]
    v = vecs[keep]
    return v[torch.as_tensor(coreset_farthest(v, CAP, FPS_SEED), device=dev)]


def load_split(combo, a, k, s2, path2idx):
    with open(D.SPLITS + "/kshot_k%d_seed%d.json" % (k, s2), encoding="utf-8") as f:
        split = json.load(f)
    with open(D.SPLITS + "/query_k%d.json" % k, encoding="utf-8") as f:
        query = json.load(f)
    sup = split["prototype_pools"][combo]
    qd = query["defect_query"][combo]
    qn = query["normal_query"][a + "/good"]
    return sup, qd, qn, [path2idx[p] for p in qd], [path2idx[p] for p in qn]


def unit_tests():
    """最小正确性检查: 方向 / 先差值后top8 / 分块一致性 (合成向量)."""
    r = {}
    torch.manual_seed(6)
    e = torch.nn.functional.normalize(torch.randn(6, 8, dtype=torch.float64), dim=1)
    Fb, Nb = e[:2], e[2:3]
    qd = torch.nn.functional.normalize(Fb[0] + 0.05 * Nb[0], dim=0)
    qn = torch.nn.functional.normalize(Nb[0] + 0.05 * Fb[0], dim=0)
    sF_d = (qd @ Fb.T).max(-1).values.item(); sN_d = (qd @ Nb.T).max(-1).values.item()
    sF_n = (qn @ Fb.T).max(-1).values.item(); sN_n = (qn @ Nb.T).max(-1).values.item()
    r["direction_A"] = bool(sF_d > sF_n)
    r["direction_B"] = bool((1 - sN_d) > (1 - sN_n))
    r["direction_C"] = bool((sF_d - sN_d) > (sF_n - sN_n))
    # 先差值后 top8 vs 先聚合后差: 构造 8 patch 情形验证二者不等且 C 等于参考实现
    np.random.seed(6)
    sF8 = np.random.rand(16); sN8 = np.random.rand(16)
    c_ref = np.sort(sF8 - sN8)[::-1][:8].mean()
    c_wrong = sF8.mean() - sN8.mean()
    r["diff_before_top8_value"] = round(float(c_ref), 6)
    r["aggregate_then_subtract_value"] = round(float(c_wrong), 6)
    r["diff_before_top8_is_different_semantics"] = bool(abs(c_ref - c_wrong) > 1e-9)
    # 分块 vs 直接 (真实 mapper, 真实缓存少量 crop)
    return r


def main():
    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    assert dev == "cuda", "need GPU"
    os.makedirs(OUT6 + "/raw_scores6", exist_ok=True)
    torch.cuda.reset_peak_memory_stats()

    # ---------- 1. 冻结全部实际输入 ----------
    rg = json.load(open(OUT4 + "/ref_groups.json"))
    nref3 = json.load(open(OUT3 + "/normal_ref.json"))
    eval_paths = json.load(open(D.CACHE + "/eval_paths.json"))
    path2idx = {p: i for i, p in enumerate(eval_paths)}
    patches = np.load(D.CACHE + "/patches.npy", mmap_mode="r")
    spec = ("head", D.load_head_state(D.OUT + "/mapper_B.pt", dev).to(dev))
    head = spec[1]; head.eval()
    for p in head.parameters():
        p.requires_grad_(False)
    frozen = {
        "code_commit": os.popen("git rev-parse HEAD").read().strip(),
        "weights": {"path": D.WEIGHTS, "sha256": sha256_file(D.WEIGHTS)},
        "mapper_B": {"path": D.OUT + "/mapper_B.pt", "sha256": sha256_file(D.OUT + "/mapper_B.pt")},
        "cache_patches_npy": {"path": D.CACHE + "/patches.npy", "sha256": sha256_file(D.CACHE + "/patches.npy")},
        "cache_eval_paths": {"path": D.CACHE + "/eval_paths.json", "sha256": sha256_file(D.CACHE + "/eval_paths.json")},
        "ref_groups_json": {"path": OUT4 + "/ref_groups.json", "sha256": sha256_file(OUT4 + "/ref_groups.json")},
        "normal_ref_json_seed3000": {"path": OUT3 + "/normal_ref.json", "sha256": sha256_file(OUT3 + "/normal_ref.json")},
        "split_files": {os.path.basename(f): sha256_file(f) for f in [
            D.SPLITS + "/query_k%d.json" % k for k in KS] + [
            D.SPLITS + "/kshot_k%d_seed%d.json" % (k, s) for k in KS for s in SUPPORT_SEEDS]},
        "archive_raw_scores4": {f: sha256_file(OUT4 + "/raw_scores4/" + f)
                                for f in sorted(os.listdir(OUT4 + "/raw_scores4"))},
        "archive_raw_scores3_P1": {f: sha256_file(OUT3 + "/raw_scores3/" + f) for f in
                                   sorted(os.listdir(OUT3 + "/raw_scores3")) if f.startswith("P1_")},
        "bank_params": {"cap": CAP, "fps_seed": FPS_SEED, "keep_frac": 0.5,
                        "normal_bank_build": "组内裁剪 embed -> FPS(256, seed0), 与 M3DIAG4 build_bank 一致",
                        "fault_bank_build": "P1: 正常距离排序(稳定) -> 前50% -> FPS(256, seed0)"},
        "scoring": {"A": "mean(top8(sF))", "B": "mean(top8(1-sN))", "C": "mean(top8(sF-sN))",
                    "topk": PATCH_TOPK, "ppc": PPC, "l2": True},
        "eval_mode": {"mapper": "eval + requires_grad False", "backbone": "不参与前向(特征来自冻结缓存)"},
    }
    ref_lists = {}
    for a, obj in rg["assets"].items():
        for sd in REF_SEEDS:
            ref_lists["%s_ref%d" % (a, sd)] = obj["groups"][str(sd)]
    for a in nref3["assets"]:
        ref_lists["%s_ref3000" % a] = nref3["assets"][a]["selected"]
    frozen["normal_ref_lists_sha256"] = {k: [sha256_file(p) for p in v] for k, v in sorted(ref_lists.items())}
    frozen["imported_scripts_sha256"] = {s: sha256_file(ROOT + "/scripts/" + s) for s in [
        "m3diag6_dualbank.py", "m3diag_common.py", "m3diag3_faultbank.py",
        "m3diag_bank_eval.py", "m3_build_patch_bank.py", "m3_common.py"]}
    D.save_json(frozen, OUT6 + "/FROZEN_INPUTS.json")
    print("[m6] FROZEN_INPUTS.json saved (%d archived score files bound)" % len(frozen["archive_raw_scores4"]), flush=True)

    # ---------- 2. 正确性检查 ----------
    ut = unit_tests()
    ut["direction_all_ok"] = all(ut[k] for k in ("direction_A", "direction_B", "direction_C"))
    # 分块 vs 直接: 真实特征小样
    a = D.ASSET_OF[COMBOS[0]]
    nb0 = build_normal_bank(spec, rg["assets"][a]["groups"]["3001"], dev)
    with open(D.SPLITS + "/query_k10.json", encoding="utf-8") as f:
        qq = json.load(f)
    probe = [path2idx[p] for p in qq["defect_query"][COMBOS[0]][:3]]
    s_chunk = crop_sims(patches, probe, spec, nb0, dev, chunk=128)
    s_direct = crop_sims(patches, probe, spec, nb0, dev, chunk=10 ** 9)
    ut["chunk_vs_direct_max_abs_diff"] = float((s_chunk - s_direct).abs().max())
    ut["chunk_vs_direct_ok"] = bool(ut["chunk_vs_direct_max_abs_diff"] <= 1e-6)
    mapper_hash_before = state_hash(head)
    ut["mapper_state_hash_before"] = mapper_hash_before
    print("[m6] unit tests:", json.dumps({k: ut[k] for k in ut if not k.startswith("mapper")}), flush=True)
    assert ut["direction_all_ok"] and ut["diff_before_top8_is_different_semantics"] and ut["chunk_vs_direct_ok"], ut

    # ---------- 3. 复现门: A vs 存档 (先于任何 B/C 生成) ----------
    normal_banks = {}
    for a in rg["assets"]:
        for sd in REF_SEEDS:
            normal_banks[(a, sd)] = build_normal_bank(spec, rg["assets"][a]["groups"][str(sd)], dev)
    for a in nref3["assets"]:
        normal_banks[(a, 3000)] = build_normal_bank(spec, nref3["assets"][a]["selected"], dev)
    nb_bytes = sum(int(v.nbytes) for v in normal_banks.values())
    print("[m6] %d normal banks built" % len(normal_banks), flush=True)

    repro = []
    worst = 0.0
    src_groups = {}
    for combo in COMBOS:
        a = D.ASSET_OF[combo]
        for k in KS:
            for s2 in SUPPORT_SEEDS:
                sup, qd, qn, qd_idx, qn_idx = load_split(combo, a, k, s2, path2idx)
                gk = D.frame_key(sup[0])
                src_groups["%s__k%d_seed%d" % (combo.replace("/", "__"), k, s2)] = len({D.frame_key(p) for p in sup})
                vecs_sup = embed(spec, torch.from_numpy(np.asarray(patches[[path2idx[p] for p in sup]]).reshape(-1, CH)).float(), dev)
                for sd in REF_SEEDS:
                    nb = normal_banks[(a, sd)]
                    fb = p1_fault_bank(vecs_sup, nb, dev)
                    fb_bytes_now = int(fb.nbytes)
                    sF_d = agg(crop_sims(patches, qd_idx, spec, fb, dev))
                    sF_n = agg(crop_sims(patches, qn_idx, spec, fb, dev))
                    arc = json.load(open(OUT4 + "/raw_scores4/P1_ref%d_%s__k%d_seed%d.json"
                                        % (sd, combo.replace("/", "__"), k, s2)))
                    cfg = "%s__k%d_seed%d_ref%d" % (combo.replace("/", "__"), k, s2, sd)
                    md = 0.0
                    for side, new in (("defect", dict(zip(qd, sF_d))), ("normal", dict(zip(qn, sF_n)))):
                        old = {p: float(x) for p, x in arc[side]}
                        assert set(old) == set(new), (cfg, side, "path set mismatch")
                        md = max(md, max(abs(old[p] - new[p]) for p in old))
                    worst = max(worst, md)
                    repro.append({"config": cfg, "max_abs_diff": round(float(md), 10),
                                  "n_paths": len(qd) + len(qn), "bank_bytes": fb_bytes_now})
                    print("[repro] %s max|diff|=%.2e" % (cfg, md), flush=True)
    # seed3000 锚点 (单独报告, 不入主汇总)
    anchor = []
    worst3 = 0.0
    for combo in COMBOS:
        a = D.ASSET_OF[combo]
        for k in KS:
            for s2 in SUPPORT_SEEDS:
                sup, qd, qn, qd_idx, qn_idx = load_split(combo, a, k, s2, path2idx)
                vecs_sup = embed(spec, torch.from_numpy(np.asarray(patches[[path2idx[p] for p in sup]]).reshape(-1, CH)).float(), dev)
                fb = p1_fault_bank(vecs_sup, normal_banks[(a, 3000)], dev)
                sF_d = agg(crop_sims(patches, qd_idx, spec, fb, dev))
                sF_n = agg(crop_sims(patches, qn_idx, spec, fb, dev))
                arc = json.load(open(OUT3 + "/raw_scores3/P1_%s__k%d_seed%d.json" % (combo.replace("/", "__"), k, s2)))
                cfg = "anchor3000_%s__k%d_seed%d" % (combo.replace("/", "__"), k, s2)
                md = 0.0
                for side, new in (("defect", dict(zip(qd, sF_d))), ("normal", dict(zip(qn, sF_n)))):
                    old = {p: float(x) for p, x in arc[side]}
                    assert set(old) == set(new), (cfg, side)
                    md = max(md, max(abs(old[p] - new[p]) for p in old))
                worst3 = max(worst3, md)
                anchor.append({"config": cfg, "max_abs_diff": round(float(md), 10), "n_paths": len(qd) + len(qn)})
    gate_ok = worst <= REPRO_TOL and worst3 <= REPRO_TOL
    print("[m6] REPRO GATE: main worst=%.3e anchor3000 worst=%.3e tol=%.0e -> %s"
          % (worst, worst3, REPRO_TOL, "PASS" if gate_ok else "FAIL"), flush=True)
    assert gate_ok, "repro gate failed - 不进入 B/C, 按重试规则诊断"

    # ---------- 4. A/B/C 评测 (同一特征同一套库, sF/sN 一次计算) ----------
    aurocs, per_img_meta = {}, {}
    fb_bytes_total, fb_bytes_max = 0, 0
    for combo in COMBOS:
        a = D.ASSET_OF[combo]
        for k in KS:
            for sd in REF_SEEDS:
                nb = normal_banks[(a, sd)]
                for s2 in SUPPORT_SEEDS:
                    sup, qd, qn, qd_idx, qn_idx = load_split(combo, a, k, s2, path2idx)
                    vecs_sup = embed(spec, torch.from_numpy(np.asarray(patches[[path2idx[p] for p in sup]]).reshape(-1, CH)).float(), dev)
                    fb = p1_fault_bank(vecs_sup, nb, dev)
                    fb_bytes_total += int(fb.nbytes); fb_bytes_max = max(fb_bytes_max, int(fb.nbytes))
                    sF_d = crop_sims(patches, qd_idx, spec, fb, dev)
                    sF_n = crop_sims(patches, qn_idx, spec, fb, dev)
                    sN_d = crop_sims(patches, qd_idx, spec, nb, dev)
                    sN_n = crop_sims(patches, qn_idx, spec, nb, dev)
                    A_d, A_n = agg(sF_d), agg(sF_n)
                    B_d, B_n = agg(1 - sN_d), agg(1 - sN_n)
                    C_d, C_n = agg(sF_d - sN_d), agg(sF_d_n if False else sF_d - sN_d), 
                    cfg = "%s__k%d_seed%d_ref%d" % (combo.replace("/", "__"), k, s2, sd)
                    ad, fl1 = D.auroc_avg_rank(A_d, A_n)
                    bd, fl2 = D.auroc_avg_rank(B_d, B_n)
                    cd, fl3 = D.auroc_avg_rank(C_d, C_n)
                    fls = {}
                    for nm, fl in (("A", fl1), ("B", fl2), ("C", fl3)):
                        if fl != {"ok"}:
                            fls[nm] = sorted(fl)
                    aurocs[cfg] = {"A": round(float(ad), 6), "B": round(float(bd), 6), "C": round(float(cd), 6),
                                   "flags": fls if fls else "ok"}
                    np.savez_compressed(OUT6 + "/raw_scores6/patches_%s.npz" % cfg,
                                        sF_defect=sF_d.numpy().astype(np.float16), sN_defect=sN_d.numpy().astype(np.float16),
                                        sF_normal=sF_n.numpy().astype(np.float16), sN_normal=sN_n.numpy().astype(np.float16))
                    rec = {"config": {"combo": combo, "asset": a, "k": k, "support_seed": s2, "ref_seed": sd,
                                      "bank_N": int(nb.shape[0]), "bank_F": int(fb.shape[0]),
                                      "source_groups_support": src_groups["%s__k%d_seed%d" % (combo.replace("/", "__"), k, s2)]},
                           "defect": {p: [round(float(A_d[i]), 8), round(float(B_d[i]), 8), round(float(C_d[i]), 8),
                                          round(float(sF_d[i].max()), 8), round(float(sN_d[i].max()), 8)] for i, p in enumerate(qd)},
                           "normal": {p: [round(float(A_n[i]), 8), round(float(B_n[i]), 8), round(float(C_n[i]), 8),
                                          round(float(sF_n[i].max()), 8), round(float(sN_n[i].max()), 8)] for i, p in enumerate(qn)},
                           "score_field_order": ["score_A", "score_B", "score_C", "sF_max", "sN_max"]}
                    D.save_json(rec, OUT6 + "/raw_scores6/%s.json" % cfg)
                    print("[eval] %s A=%.4f B=%.4f C=%.4f" % (cfg, ad, bd, cd), flush=True)

    mapper_hash_after = state_hash(head)
    assert mapper_hash_after == mapper_hash_before, "mapper state changed during eval"
    ut["mapper_state_hash_after"] = mapper_hash_after
    ut["mapper_state_unchanged"] = True

    # ---------- 5. 汇总 ----------
    contrasts = ["C-A", "C-B", "B-A"]
    def g(c, arm):
        return aurocs[c][arm[0]]
    paired = {}
    for combo in COMBOS:
        for k in KS:
            for ct in contrasts:
                left, right = ct[0], ct[2]
                per_ref = {}
                for sd in REF_SEEDS:
                    ds = [g("%s__k%d_seed%d_ref%d" % (combo.replace("/", "__"), k, s2, sd), left)
                          - g("%s__k%d_seed%d_ref%d" % (combo.replace("/", "__"), k, s2, sd), right)
                          for s2 in SUPPORT_SEEDS]
                    per_ref["ref%d" % sd] = {"mean_over_seeds": round(float(np.mean(ds)), 6),
                                             "per_seed": [round(float(x), 6) for x in ds]}
                vals = [per_ref["ref%d" % sd]["mean_over_seeds"] for sd in REF_SEEDS]
                paired.setdefault("%s__k%d" % (combo.replace("/", "__"), k), {})[ct] = {
                    "per_ref": per_ref,
                    "mean_over_groups": round(float(np.mean(vals)), 6),
                    "median_over_groups": round(float(np.median(vals)), 6),
                    "range": [round(float(min(vals)), 6), round(float(max(vals)), 6)],
                    "n_pos_groups": int(sum(v > 0 for v in vals)), "n_groups": len(vals)}
    summary = {
        "scoring_variants_completed": 3,
        "repro_gate_passed": 1 if gate_ok else 0,
        "repro": {"tol": REPRO_TOL, "main_configs": len(repro), "main_worst_max_abs_diff": round(float(worst), 10),
                  "anchor3000_configs": len(anchor), "anchor3000_worst_max_abs_diff": round(float(worst3), 10),
                  "main": repro, "anchor3000": anchor},
        "correctness": ut,
        "source_groups_support": src_groups,
        "aurocs": aurocs,
        "paired_diffs": paired,
        "cost": {"wall_time_s": round(time.time() - t0, 1),
                 "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
                 "normal_bank_bytes_each": int(next(iter(normal_banks.values())).nbytes),
                 "normal_banks_total_bytes": nb_bytes,
                 "fault_bank_bytes_max": fb_bytes_max,
                 "fault_bank_bytes_total_over_configs": fb_bytes_total,
                 "note": "N 库 12 个常驻 (5 新组+seed3000, 每资产), F 库逐配置重建; 不声称双库与单库同内存"},
        "exploratory": True,
        "b_reuse_note": "B 与故障支持种子无关, 同查询+同正常库分数相同; 聚合中按 (combo,k,ref) 复用, 不计独立实验",
    }
    D.save_json(summary, OUT6 + "/dualbank_eval.json")
    print("[m6] dualbank_eval.json saved; wall=%.1fs" % (time.time() - t0), flush=True)
    print("=== 全部完成 ===", flush=True)


if __name__ == "__main__":
    main()