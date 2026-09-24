#!/usr/bin/env python3
"""M3TRANSFER-wrnfix: 官方池化路径修正 WRN 重评. 审计->门(assert)->建库->评分->汇总->勘误. 20 配置."""
import csv, hashlib, json, os, sys, time
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m3diag_common as D
import m3score_extract as X
import m3score_pcverify as PV
import m3transfer_lib as T

OUT = D.ROOT + "/outputs/M3transfer-wrnfix"
BENCH = D.ROOT + "/outputs/M3bench-dualbank-patchcore"
M3T = D.ROOT + "/outputs/M3transfer-top8"
REFS = [3001, 3002, 3003, 3004, 3005]
ASSETS = {"yoke": ("yoke-suspension", "rust", "g2"), "glass": ("glass-insulator", "missing-cap", "g2")}
ARMS = ["W-PC-FIX", "W-T8-FIX"]
TOL_A, TOL_R = 1e-4, 1e-4
LEDGER = {}
PM_W = None


def th(t):
    return hashlib.sha256(np.ascontiguousarray(t.detach().cpu().numpy()).tobytes()).hexdigest()[:32]


def state_hash(m):
    h = hashlib.sha256()
    for k in sorted(m.state_dict()):
        h.update(k.encode())
        h.update(np.ascontiguousarray(m.state_dict()[k].detach().cpu().numpy()).tobytes())
    return h.hexdigest()[:32]


def combo(a):
    cat, deft, gk = ASSETS[a]
    return "%s__%s" % (cat, deft), cat, deft, gk


def std_extract_flat(pm, xb):
    """共享标准提取: feature_extractor -> feature_pooler(逐层, 官方) -> generate_embedding -> reshape_embedding."""
    feats = pm.feature_extractor(xb)
    feats = {k: pm.feature_pooler(v) for k, v in feats.items()}
    emb = pm.generate_embedding(feats)
    return pm.reshape_embedding(emb)


@torch.inference_mode()
def std_extract(pm, paths, bs=10):
    tf = T.wrn_tf()
    outs = []
    for i in range(0, len(paths), bs):
        xb = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths[i:i + bs]]).to("cuda")
        outs.append(std_extract_flat(pm, xb).cpu())
    return torch.cat(outs, 0)


def wrn_ffn(paths):
    n = len(paths)
    return std_extract(PM_W, paths).reshape(n, -1, 1536).float()


def save_cfg(name, meta, sd, sn, qd, qn):
    u, _ = D.auroc_avg_rank(np.array(sd, dtype=np.float64), np.array(sn, dtype=np.float64))
    rec = {"defect": {p: float(v) for p, v in zip(qd, sd)},
           "normal": {p: float(v) for p, v in zip(qn, sn)}, "auroc": float(u)}
    rec.update(meta)
    with open("%s/raw_scores/%s.json" % (OUT, name), "w") as f:
        json.dump(rec, f)
    return float(u)


def main():
    global PM_W
    t0 = time.time()
    T.deterministic()
    os.makedirs(OUT + "/raw_scores", exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    # ---------------- 审计 (评分前) ----------------
    t = time.time()
    import timm
    import anomalib
    A = "/root/miniconda3/lib/python3.12/site-packages/anomalib"
    srcs = {os.path.basename(p): T.file_sha(p) for p in
            [A + "/models/image/patchcore/torch_model.py",
             A + "/models/components/sampling/k_center_greedy.py",
             A + "/models/components/feature_extractors/timm.py"]}
    m = timm.create_model("wide_resnet50_2", pretrained=False)
    pcfg = dict(m.pretrained_cfg) if hasattr(m, "pretrained_cfg") else {}
    PM_W, sdh = T.wrn_model()  # 进程内清代理 + HF offline; timm 默认 pretrained=True
    sdh2 = state_hash(PM_W.feature_extractor)
    hf = "/root/.cache/huggingface/hub/models--timm--wide_resnet50_2.racm_in1k/snapshots/30f73aceaaa1911830a9795b83ab1908dba18719/model.safetensors"
    tv = "/root/.cache/torch/hub/checkpoints/wide_resnet50_2-95faca4d.pth"
    tv_note = "未对比"
    if os.path.exists(tv):
        sd_tv = torch.load(tv, map_location="cpu", weights_only=True)
        sd_ex = PM_W.feature_extractor.state_dict()
        k = "conv1.weight"
        kex = [kk for kk in sd_ex if kk.endswith("conv1.weight")][0]
        tv_note = "conv1.weight max|diff|=%.3e (不同训练产物, timm 路径不加载 torchvision 文件)" % float(
            (sd_tv[k].float() - sd_ex[kex].cpu()).abs().max())
    aud = {"anomalib": anomalib.__version__, "torch": torch.__version__, "timm": timm.__version__,
           "sources_sha256": srcs, "timm_pretrained_cfg": pcfg,
           "hf_checkpoint": hf, "hf_sha256": T.file_sha(hf) if os.path.exists(hf) else "missing",
           "loaded_state_dict_sha256": sdh2, "torchvision_ckpt": tv, "torchvision_compare": tv_note,
           "pooling": "官方 forward 含 feature_pooler=AvgPool2d(3,1,1); M3TRANSFER 薄适配遗漏 -> 旧结果=未池化WRN变体"}
    X.atomic_json(aud, OUT + "/audit_evidence.json")
    with open(OUT + "/OFFICIAL_PATH_AUDIT.md", "w") as f:
        f.write("# OFFICIAL_PATH_AUDIT (安装版 2.6.2, 非网上 main)\n\n"
                "实际安装 forward (torch_model.py sha=" + srcs["torch_model.py"][:16] + ") 含:\n\n"
                "    features = self.feature_extractor(input_tensor)\n"
                "    features = {layer: self.feature_pooler(feature) for ...}  # AvgPool2d(3,1,1)\n"
                "    embedding = self.generate_embedding(features)\n"
                "    embedding = self.reshape_embedding(embedding)\n"
                "    patch_scores, locations = self.nearest_neighbors(embedding, n_neighbors=1)\n"
                "    pred_score = self.compute_anomaly_score(...)\n\n"
                "M3TRANSFER-top8-v1 薄适配 = generate_embedding(feature_extractor(x)), **遗漏池化** -> 登记为特征路径偏离; "
                "旧 WRN 结果命名「未池化WRN变体」, 原始文件不动。本轮唯一提取函数 std_extract = extractor -> pooler -> "
                "generate_embedding -> reshape_embedding, 支持/查询/验证共用。\n")
    with open(OUT + "/WEIGHT_IDENTITY_AUDIT.md", "w") as f:
        f.write("# WEIGHT_IDENTITY_AUDIT\n\n"
                "- M3BENCH 冻结记录只存 YOLO 权重哈希, WRN state 哈希未留存 -> **历史 WRN 权重身份未完全核实**。\n"
                "- 但历史与本轮同机/同环境 (torch 2.8.0+cu128, anomalib 2.6.2) 同代码路径: "
                "Patchcore(pre_trained=True, backbone=wide_resnet50_2) -> anomalib TimmFeatureExtractor -> timm create_model(pretrained=True) -> "
                "timm 默认 pretrained_cfg=" + json.dumps(pcfg)[:220] + " -> HF 缓存 model.safetensors (sha256=" + aud["hf_sha256"][:16] + ")。\n"
                "- 本轮实际加载 feature_extractor state_dict sha256=" + sdh2[:16] + " (冻结, 不按查询结果挑选)。\n"
                "- torchvision wide_resnet50_2-95faca4d.pth 与 timm racm_in1k 为不同训练产物: " + tv_note + "\n"
                "- 结论: 分支 B (证据链强一致但无历史哈希直证); 本轮结果称「权重明确的标准WRN配置重评」, 不宣称逐项复现历史。\n")
    print("[audit] 权重 sha=%s tag=%s" % (sdh2[:12], pcfg.get("tag", "?")), flush=True)
    LEDGER["audit_s"] = round(time.time() - t, 1)
    # ---------------- freeze ----------------
    SL = json.load(open(BENCH + "/SUPPORT_LISTS.json"))
    eval_paths = json.load(open(D.CACHE + "/eval_paths.json"))
    path2idx = {p: i for i, p in enumerate(eval_paths)}
    qq = json.load(open(D.SPLITS + "/query_k5.json"))
    data = {}
    for a in ("yoke", "glass"):
        c, cat, deft, gk = combo(a)
        qd = qq["defect_query"]["%s/%s" % (cat, deft)]
        qn = qq["normal_query"]["%s/good" % cat]
        assert set(qd).isdisjoint(set(qn)) and all(p in path2idx for p in qd + qn)
        n10of = {j: SL["normal_lists"]["%s_ref%d" % (cat, j)]["N10"] for j in REFS}
        data[a] = {"cat": cat, "deft": deft, "gk": gk, "qd": qd, "qn": qn, "n10": n10of}
        print("[freeze] %s qd=%d qn=%d" % (a, len(qd), len(qn)), flush=True)
    X.atomic_json({"weights_sha256": sdh2, "hf_sha256": aud["hf_sha256"],
                   "sources_sha256": srcs, "timm_pretrained_cfg": pcfg, "pooling": aud["pooling"],
                   "splits_query_k5": X.sha256_file(D.SPLITS + "/query_k5.json"),
                   "support_lists": X.sha256_file(BENCH + "/SUPPORT_LISTS.json"),
                   "rules": {"pc_seed": {str(j): 7000 + (j - 3000) for j in REFS}, "ratio": 0.1, "nn": 9, "topk": 8,
                             "arms": {"W-PC-FIX": "官方 nn=9 评分 (池化路径)", "W-T8-FIX": "同库同距离 patch 轴 top8"},
                             "tol": {"atol": TOL_A, "rtol": TOL_R}},
                   "configs_expected": sorted(["%s__%s_%s_ref%d" % (data[a]["cat"], data[a]["deft"], arm, j)
                                               for a in data for arm in ARMS for j in REFS]),
                   "budget": {"gpu_hours": 1, "probe_first": True}}, OUT + "/FROZEN_INPUTS.json")
    # ---------------- 门 (正式评分前立即 assert) ----------------
    t = time.time()
    g = {}
    x4 = torch.stack([T.wrn_tf()(Image.open(p).convert("RGB")) for p in data["yoke"]["n10"][3001][:4]]).to("cuda")
    sd_saved = {k: v.detach().cpu().clone() for k, v in PM_W.state_dict().items()}
    h_before = state_hash(PM_W)
    store_before = len(PM_W.embedding_store)
    PM_W.train()
    with torch.no_grad():
        path1 = PM_W(x4)  # 官方 train 分支返回 embedding (含池化+融合+展平, 独立官方路径)
    PM_W.load_state_dict(sd_saved)  # BN running stats 还原
    PM_W.embedding_store = PM_W.embedding_store[:store_before]
    PM_W.eval()
    assert state_hash(PM_W) == h_before, "train 前后状态哈希不一致"
    assert len(PM_W.embedding_store) == store_before, "embedding_store 未清理"
    assert PM_W.training is False
    path2 = std_extract_flat(PM_W, x4)
    d12 = (path1.float() - path2.float()).abs()
    ok = bool((d12 <= TOL_A + TOL_R * path2.float().abs()).all())
    g["e2e_path"] = {"max_abs_diff": float(d12.max()), "pass": ok, "shape": list(path2.shape)}
    assert ok, "端到端路径门失败: %s" % g["e2e_path"]
    pe1 = std_extract_flat(PM_W, x4[:1])
    b1d = float((pe1 - path2[:1600]).abs().max())
    g["batch_consistency"] = {"b1_vs_b4_max_diff": b1d, "pass": bool(b1d <= TOL_A + TOL_R * float(path2[:1600].abs().max()))}
    assert g["batch_consistency"]["pass"], g["batch_consistency"]
    assert path2.dtype == torch.float32 and bool(torch.isfinite(path2).all())
    top8v = T.verify_top8_axis()
    assert top8v["axis_ok"] and top8v["wrong_axis_differs"]
    g["top8_axis"] = top8v
    print("[gates] e2e_path max_diff=%.2e batch=%.2e" % (g["e2e_path"]["max_abs_diff"], b1d), flush=True)
    # ---------------- 探针 ----------------
    t = time.time()
    std_extract(PM_W, data["glass"]["qd"][:8])
    te = (time.time() - t) / 8
    PM_W.memory_bank = torch.randn(1600, 1536).to("cuda")
    pe = torch.randn(8 * 1600, 1536).to("cuda")
    t = time.time()
    with torch.no_grad():
        PM_W.nearest_neighbors(pe, 1)
    td = (time.time() - t) / 8
    n_q = sum(len(data[a]["qd"]) + len(data[a]["qn"]) for a in data)
    proj = 20 * 10 * te + n_q * te + n_q * 5 * td + 30
    print("[probe] extract=%.2fs/img dist=%.2fs/img -> proj=%.0fs / 3600s" % (te, td, proj), flush=True)
    assert proj < 3600, "预算投影超限"
    # ---------------- 建库 (种子固定, 双建库核验) ----------------
    t = time.time()
    banks, bmeta = {}, {}
    for a in ("yoke", "glass"):
        d_ = data[a]
        for j in REFS:
            seed = 7000 + (j - 3000)
            pm, meta = PV.make_pm(wrn_ffn, d_["n10"][j], seed)
            if j == 3001:
                pm2, _ = PV.make_pm(wrn_ffn, d_["n10"][j], seed)
                assert torch.equal(pm.memory_bank, pm2.memory_bank), "%s ref3001 双建库不一致" % a
                g.setdefault("bank_rebuild", {})["%s_ref3001" % a] = True
            banks[(a, j)] = pm.memory_bank
            bmeta["%s_ref%d" % (a, j)] = {"n_bank": int(pm.memory_bank.shape[0]), "dtype": "float32",
                                          "bytes": int(pm.memory_bank.numel() * 4), "tensor_hash": th(pm.memory_bank),
                                          "coreset_indices": meta["coreset_indices"], "pc_seed": seed,
                                          "support_list": d_["n10"][j]}
            print("[bank] %s ref%d protos=%d" % (a, j, meta["n_bank"]), flush=True)
    LEDGER["banks_s"] = round(time.time() - t, 1)
    # ---------------- 评分门: 官方完整推理 pred_score vs 适配 ----------------
    PM_W.memory_bank = banks[("yoke", 3001)]
    PM_W.eval()
    with torch.no_grad():
        out = PM_W(x4)
        official_pred = out.pred_score
        pe = std_extract_flat(PM_W, x4)
        ps, loc = PM_W.nearest_neighbors(pe, 1)
        pred2 = PM_W.compute_anomaly_score(ps.reshape(4, -1), loc.reshape(4, -1), pe.reshape(4, -1, 1536))
    dps = (official_pred.float() - pred2.float()).abs()
    ok = bool((dps <= TOL_A + TOL_R * official_pred.float().abs()).all())
    g["pred_score"] = {"max_abs_diff": float(dps.max()), "pass": ok, "n": 4}
    assert ok, "官方 pred_score 门失败: %s" % g["pred_score"]
    print("[gates] pred_score max_diff=%.2e (n=4)" % g["pred_score"]["max_abs_diff"], flush=True)
    LEDGER["gates_s"] = round(time.time() - t, 1)
    # ---------------- 正式评分 (每查询单次前向 x 5 库) ----------------
    t = time.time()
    aurocs, dist_files = {}, {}
    for a in ("yoke", "glass"):
        d_ = data[a]
        c = "%s__%s" % (d_["cat"], d_["deft"])
        qpaths = d_["qd"] + d_["qn"]
        dists = {j: [] for j in REFS}
        pcs = {j: [] for j in REFS}
        for i in range(0, len(qpaths), 16):
            xb = torch.stack([T.wrn_tf()(Image.open(p).convert("RGB")) for p in qpaths[i:i + 16]]).to("cuda")
            pe = std_extract_flat(PM_W, xb)
            B = xb.shape[0]
            for j in REFS:
                PM_W.memory_bank = banks[(a, j)]
                with torch.no_grad():
                    ps, loc = PM_W.nearest_neighbors(pe, 1)
                    pred = PM_W.compute_anomaly_score(ps.reshape(B, -1), loc.reshape(B, -1), pe.reshape(B, -1, 1536))
                pcs[j].extend(float(x) for x in pred)
                dists[j].append(ps.reshape(B, -1).cpu())
        npz = "/root/autodl-tmp/m3wrnfix-dists-%s.npz" % a
        np.savez_compressed(npz, **{"ref%d" % j: torch.cat(dists[j], 0).numpy().astype(np.float32) for j in REFS})
        dist_files[a] = {"path": npz, "sha256": T.file_sha(npz)}
        k = len(d_["qd"])
        for j in REFS:
            dmat = torch.cat(dists[j], 0)
            assert dmat.shape[0] == len(qpaths) and dmat.shape[1] == bmeta["%s_ref%d" % (a, j)]["n_bank"]
            bh = bmeta["%s_ref%d" % (a, j)]["tensor_hash"]
            u1 = save_cfg("%s_W-PC-FIX_ref%d" % (c, j), {"asset": a, "arm": "W-PC-FIX", "ref": j, "bank_hash": bh},
                          pcs[j][:k], pcs[j][k:], d_["qd"], d_["qn"])
            t8 = T.top8_mean(dmat).tolist()
            u2 = save_cfg("%s_W-T8-FIX_ref%d" % (c, j), {"asset": a, "arm": "W-T8-FIX", "ref": j, "bank_hash": bh},
                          t8[:k], t8[k:], d_["qd"], d_["qn"])
            aurocs["%s_%s_ref%d" % (c, "W-PC-FIX", j)] = u1
            aurocs["%s_%s_ref%d" % (c, "W-T8-FIX", j)] = u2
        print("[score] %s 完成" % a, flush=True)
    LEDGER["score_s"] = round(time.time() - t, 1)
    # ---------------- 历史集合断言 (g2) 与对比 ----------------
    hist = {}
    for a in ("yoke", "glass"):
        d_ = data[a]
        c = "%s__%s" % (d_["cat"], d_["deft"])
        for j in REFS:
            h = json.load(open(BENCH + "/raw_scores_patchcore/%s_%s_ref%d.json" % (c, d_["gk"], j)))
            assert set(h["defect"]) == set(d_["qd"]), "%s g2 ref%d 故障查询集合不一致" % (a, j)
            assert set(h["normal"]) == set(d_["qn"]), "%s g2 ref%d 正常查询集合不一致" % (a, j)
            hu, _ = D.auroc_avg_rank(np.array(list(h["defect"].values()), np.float64),
                                     np.array(list(h["normal"].values()), np.float64))
            hist["%s_ref%d" % (a, j)] = {"hist_g2_W-PC": float(hu),
                                         "new_W-PC-FIX": aurocs["%s_W-PC-FIX_ref%d" % (c, j)],
                                         "diff": float(aurocs["%s_W-PC-FIX_ref%d" % (c, j)] - hu)}
    g["hist_set_assert"] = {"pass": True, "note": "g2 文件路径集合/标签侧全部一致; glass g1 弃用"}
    # ---------------- 汇总 ----------------
    names = sorted(aurocs)
    expected = sorted(["%s__%s_%s_ref%d" % (data[a]["cat"], data[a]["deft"], arm, j)
                       for a in data for arm in ARMS for j in REFS])
    set_match = int(names == expected)
    assert set_match == 1 and len(names) == 20, "配置集合与预注册不一致"

    def stats(vs):
        v = np.array(vs, np.float64)
        return {"mean": float(v.mean()), "min": float(v.min()), "max": float(v.max()), "std_across_refs": float(v.std(ddof=0))}

    arm_means = {arm: {a: stats([aurocs["%s__%s_%s_ref%d" % (data[a]["cat"], data[a]["deft"], arm, j)] for j in REFS])
                       for a in data} for arm in ARMS}

    def paired(k1, k2, key):
        out = {}
        for a in data:
            d_ = data[a]
            c = "%s__%s" % (d_["cat"], d_["deft"])
            per = {"ref%d" % j: aurocs["%s_%s_ref%d" % (c, k1, j)] - aurocs["%s_%s_ref%d" % (c, k2, j)] for j in REFS}
            out["%s_%s" % (key, a)] = {"per_ref": per, **stats(list(per.values())),
                                       "n_pos": sum(1 for v in per.values() if v > 0), "n_neg": sum(1 for v in per.values() if v < 0)}
        return out

    comps = {}
    comps.update(paired("W-T8-FIX", "W-PC-FIX", "WT8FIX-WPCFIX"))
    old = {}
    for a in data:
        d_ = data[a]
        c = "%s__%s" % (d_["cat"], d_["deft"])
        for j in REFS:
            old["%s_Y-T8_ref%d" % (c, j)] = float(json.load(open(M3T + "/raw_scores/%s_Y-T8_k5_ref%d.json" % (c, j)))["auroc"])
            old["%s_W-PC-UNPOOL_ref%d" % (c, j)] = float(json.load(open(M3T + "/raw_scores/%s_W-PC_k5_ref%d.json" % (c, j)))["auroc"])
            old["%s_W-T8-UNPOOL_ref%d" % (c, j)] = float(json.load(open(M3T + "/raw_scores/%s_W-T8_k5_ref%d.json" % (c, j)))["auroc"])
    for k1, k2, key in (("Y-T8", "W-PC-FIX", "YT8-WPCFIX"), ("Y-T8", "W-T8-FIX", "YT8-WT8FIX"),
                        ("W-PC-FIX", "W-PC-UNPOOL", "FIX-UNPOOL_PC"), ("W-T8-FIX", "W-T8-UNPOOL", "FIX-UNPOOL_T8")):
        for a in data:
            d_ = data[a]
            c = "%s__%s" % (d_["cat"], d_["deft"])
            per = {}
            for j in REFS:
                v1 = aurocs["%s_%s_ref%d" % (c, k1, j)] if k1 in ARMS else old["%s_%s_ref%d" % (c, k1, j)]
                v2 = aurocs["%s_%s_ref%d" % (c, k2, j)] if k2 in ARMS else old["%s_%s_ref%d" % (c, k2, j)]
                per["ref%d" % j] = v1 - v2
            comps["%s_%s" % (key, a)] = {"per_ref": per, **stats(list(per.values())),
                                         "n_pos": sum(1 for v in per.values() if v > 0), "n_neg": sum(1 for v in per.values() if v < 0)}
    ev = {"arm_configs_completed": len(names), "pre_registered_set_match": set_match,
          "gates": g, "gates_all_passed": 1, "official_path_ok": 1, "official_pred_score_ok": 1,
          "same_bank_identity": 1, "hist_set_assert_ok": 1,
          "arm_means": arm_means, "paired": comps, "hist_g2_vs_new": hist,
          "bank_meta": {k: {kk: vv for kk, vv in v.items() if kk != "support_list"} | {"n_support": 10} for k, v in bmeta.items()},
          "wrn_dist_files": dist_files, "yolo_reuse": "M3TRANSFER-top8-v1 raw_scores (锚点已验证)"}
    X.atomic_json(ev, OUT + "/wrnfix_eval.json")
    with open(OUT + "/paired_comparisons.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["comparison", "asset", "group", "diff"])
        for key in ("WT8FIX-WPCFIX", "YT8-WPCFIX", "YT8-WT8FIX", "FIX-UNPOOL_PC", "FIX-UNPOOL_T8"):
            for a in data:
                for r, v in comps["%s_%s" % (key, a)]["per_ref"].items():
                    w.writerow([key, a, r, v])
    with open(OUT + "/BANK_MANIFEST.json", "w") as f:
        json.dump(bmeta, f, ensure_ascii=False, indent=1)
    X.atomic_json({"gates": g, "bank_meta": bmeta, "dist_files": dist_files, "audit": aud,
                   "wrn_tol": {"atol": TOL_A, "rtol": TOL_R},
                   "note": "全部正确性门在正式评分前立即 assert; 状态哈希前后一致; embedding_store 清理核查"},
                  OUT + "/REPRO_AND_CORRECTNESS.json")
    X.atomic_json({"stages_s": LEDGER, "total_wall_s": round(time.time() - t0, 1),
                   "peak_vram_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                   "peak_vram_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                   "wrn_dist_bytes": {a: os.path.getsize(dist_files[a]["path"]) for a in dist_files},
                   "note": "评分为缓存特征计算, 非端到端推理延迟; 查询单次前向对 5 库评分"}, OUT + "/RESOURCE_COST.json")
    print("[eval] 20 配置 set_match=%d wall=%.0fs" % (set_match, time.time() - t0), flush=True)
    print("=== M3TRANSFER-WRNFIX 完成 ===", flush=True)


if __name__ == "__main__":
    main()
