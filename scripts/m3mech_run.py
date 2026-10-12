#!/usr/bin/env python3
"""M3MECH-top8-v1: top8 收益机制诊断.
阶段A: 4 特征臂 (Y-RAW/Y-POOL/W-STD/W-NOPOOL) x 2 资产 x 5 参照, 7 种评分, G=top8-max.
阶段B: P3/P4/P5 单层, 共同原型数 M, max/top8/比例评分.
结果目录: /root/autodl-tmp/version10-12 (参数化 --out)."""
import argparse, hashlib, json, os, random as pyrandom, subprocess, sys, time
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m3diag_common as D
import m3score_extract as X
import m3score_pcverify as PV
import m3transfer_lib as T
from anomalib.models.image.patchcore.torch_model import KCenterGreedy

AP = argparse.ArgumentParser()
AP.add_argument("--out", default="/root/autodl-tmp/version10-12")
AP.add_argument("--phase", default="all", choices=["all", "A", "B"])
ARGS = AP.parse_args()
OUT = ARGS.out

REFS = [3001, 3002, 3003, 3004, 3005]
ASSETS = ["yoke-suspension", "vari-grip"]
FAULT = {"yoke-suspension": "rust", "vari-grip": "bird-nest"}
COMBO = {"yoke-suspension": "yoke-suspension__rust", "vari-grip": "vari-grip__bird-nest"}
TOPKS = [4, 8, 16, 32]
SAMPLE_SEED = 20250928
LEDGER = {}
PM_W = None
Y_BS = 32
W_BS = 32


def th(t):
    return hashlib.sha256(np.ascontiguousarray(t.detach().cpu().numpy()).tobytes()).hexdigest()[:32]


def hash_eq(mine, src):
    return str(src) == str(mine)[:len(str(src))]


def state_hash(m):
    h = hashlib.sha256()
    for k in sorted(m.state_dict()):
        h.update(k.encode())
        h.update(np.ascontiguousarray(m.state_dict()[k].detach().cpu().numpy()).tobytes())
    return h.hexdigest()[:32]


def stats(vs):
    v = np.array(vs, np.float64)
    return {"mean": float(v.mean()), "min": float(v.min()), "max": float(v.max()), "std": float(v.std(ddof=0))}


def atomic_json(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def dji_key(p):
    import re
    m = re.search(r"_DJI_(\d{4})_", "_" + os.path.basename(p))
    return m.group(1) if m else os.path.basename(p)[:24]


# ---------- 特征函数 ----------
B1 = None  # YOLO backbone
POOL3 = nn.AvgPool2d(3, stride=1, padding=1)


def yolo_layers(paths, layer):
    from m3score_extract import MC
    tf = MC.make_transform(D.IMGSZ)
    outs = []
    with torch.inference_mode():
        for i in range(0, len(paths), Y_BS):
            xb = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths[i:i + Y_BS]]).to("cuda")
            f = B1.forward_layers(xb)[layer]
            outs.append(f.cpu())
    return torch.cat(outs, 0)


def l2_map(f):
    B, C, H, W = f.shape
    v = f.reshape(B, C, H * W).transpose(1, 2)
    return torch.nn.functional.normalize(v, dim=2)


def std_extract_flat(pm, xb, pool=True):
    feats = pm.feature_extractor(xb)
    if pool:
        feats = {k: pm.feature_pooler(v) for k, v in feats.items()}
    emb = pm.generate_embedding(feats)
    return pm.reshape_embedding(emb)


def wrn_feats(paths, pool):
    outs = []
    with torch.inference_mode():
        for i in range(0, len(paths), W_BS):
            xb = torch.stack([T.wrn_tf()(Image.open(p).convert("RGB")) for p in paths[i:i + W_BS]]).to("cuda")
            outs.append(std_extract_flat(PM_W, xb, pool=pool).cpu())
    return torch.cat(outs, 0)


def wrn_ffn_pool(ps):
    return wrn_feats(ps, True).reshape(len(ps), -1, 1536).float()


def wrn_ffn_nopool(ps):
    return wrn_feats(ps, False).reshape(len(ps), -1, 1536).float()


# ---------- 建库 ----------
def make_bank(feat_fn, paths, seed, M=None, nn_=9):
    """复刻 PV.make_pm 种子语义; M 给定时按目标数 KCenter (取序列前 M)."""
    pyrandom.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pm = PV._cls()(layers=["layer2"], backbone="wide_resnet50_2", pre_trained=False, num_neighbors=nn_).to(PV.DEVICE)
    pm.train()
    store = []
    with torch.no_grad():
        for i in range(0, len(paths), 32):
            q = feat_fn(paths[i:i + 32])
            store.append(PV._flat(pm.reshape_embedding(PV._to4d(q))))
    allp = torch.cat(store, 0).float().to(PV.DEVICE)
    if M is None:
        kg = KCenterGreedy(embedding=allp, sampling_ratio=0.1)
    else:
        assert len(paths) * 1 > 0
        nc = allp.shape[0]
        if nc <= M:
            idxs = list(range(nc))
            pm.memory_bank = allp
            pm.eval()
            meta = {"n_support_patches": int(nc), "n_bank": int(nc), "bank_dim": int(allp.shape[1]),
                    "bank_dtype": str(allp.dtype), "bank_bytes": int(allp.numel() * allp.element_size()),
                    "coreset_indices": idxs, "seed": seed, "ratio": 1.0, "nn": nn_, "all_kept": True}
            return pm, meta
        kg = KCenterGreedy(embedding=allp, sampling_ratio=M / nc)
    idxs = [int(i) for i in kg.select_coreset_idxs()]
    if M is not None and len(idxs) > M:
        idxs = idxs[:M]
    pm.memory_bank = kg.embedding[torch.as_tensor(idxs, device=PV.DEVICE)]
    pm.eval()
    meta = {"n_support_patches": int(allp.shape[0]), "n_bank": int(pm.memory_bank.shape[0]),
            "bank_dim": int(pm.memory_bank.shape[1]), "bank_dtype": str(pm.memory_bank.dtype),
            "bank_bytes": int(pm.memory_bank.numel() * pm.memory_bank.element_size()),
            "coreset_indices": idxs, "seed": seed, "ratio": (M / allp.shape[0]) if M else 0.1, "nn": nn_}
    return pm, meta


# ---------- 评分 ----------
def score_arm(feat_fn, bank, paths, P, bs_img=32):
    """返回 (N,P) 最小距离 D (cpu fp32) 与 official PC 分数列表. 多种评分共享 D."""
    pm = PM_SHELL
    pm.memory_bank = bank
    N = len(paths)
    Dm = np.zeros((N, P), np.float32)
    pc = np.zeros(N, np.float32)
    with torch.inference_mode():
        for i in range(0, N, bs_img):
            q = feat_fn(paths[i:i + bs_img])
            B = q.shape[0]
            pe = q.reshape(B * P, q.shape[2]).to(PV.DEVICE)
            ps, loc = pm.nearest_neighbors(pe, 1)
            Dm[i:i + B] = ps.reshape(B, P).cpu().numpy()
            pred = pm.compute_anomaly_score(ps.reshape(B, -1), loc.reshape(B, -1), pe.reshape(B, -1, q.shape[2]))
            pc[i:i + B] = pred.cpu().numpy()
    return Dm, pc


def scores_from_D(Dm, pc):
    s = {"official_pc": pc.astype(np.float64)}
    s["pure_max"] = Dm.max(1).astype(np.float64)
    for k in TOPKS:
        if Dm.shape[1] >= k:
            s["top%d" % k] = np.sort(Dm, axis=1)[:, -k:].mean(1).astype(np.float64)
    s["mean_all"] = Dm.mean(1).astype(np.float64)
    return s


def auroc(sd, sn):
    u, _ = D.auroc_avg_rank(np.asarray(sd, np.float64), np.asarray(sn, np.float64))
    return float(u)


# ---------- 主流程 ----------
def main():
    global B1, PM_W, PM_SHELL, FFNS
    t00 = time.time()
    T.deterministic()
    for sub in ("raw_scores/phaseA", "raw_scores/phaseB", "vis", "cache", "cache/vis_dist"):
        os.makedirs(os.path.join(OUT, sub), exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    commit = subprocess.check_output(["git", "-C", D.ROOT, "rev-parse", "HEAD"]).decode().strip()
    qq = json.load(open(D.SPLITS + "/query_k5.json"))
    SL = json.load(open(D.ROOT + "/outputs/M3full-frozen/SUPPORT_LISTS.json"))
    data = {}
    for a in ASSETS:
        qn = qq["normal_query"]["%s/good" % a]
        qd = qq["defect_query"]["%s/%s" % (a, FAULT[a])]
        n10 = {j: SL["%s_ref%d" % (a, j)]["N10"] for j in REFS}
        allsup = set(p for j in REFS for p in n10[j])
        assert allsup.isdisjoint(set(qn) | set(qd))
        data[a] = {"qn": qn, "qd": qd, "n10": n10, "allq": list(qn) + list(qd), "qn_n": len(qn)}
    # --- 冻结固定样本组 (查看任何新增结果之前) ---
    frozen_samples = {}
    for a in ASSETS:
        for role, pool in (("normal", data[a]["qn"]), ("fault", data[a]["qd"])):
            paths = sorted(set(pool))
            rng = np.random.Generator(np.random.PCG64(SAMPLE_SEED))
            perm = rng.permutation(len(paths))
            picked, seen = [], set()
            for i in perm:
                k = dji_key(paths[i])
                if k in seen:
                    continue
                seen.add(k)
                picked.append(paths[i])
                if len(picked) == 12:
                    break
            frozen_samples["%s_%s" % (a, role)] = picked
    # --- FROZEN_INPUTS ---
    wfa = json.load(open(D.ROOT + "/outputs/M3transfer-wrnfix/audit_evidence.json"))
    yolo_sha = T.file_sha(D.WEIGHTS)
    file_hashes = {}
    for a in ASSETS:
        for p in data[a]["allq"] + [p for j in REFS for p in data[a]["n10"][j]]:
            if p not in file_hashes:
                file_hashes[p] = T.file_sha(p)
    Fi = {"commit": commit, "torch": torch.__version__, "cuda": torch.version.cuda,
          "device": torch.cuda.get_device_name(0), "yolo_weights_sha256": yolo_sha,
          "wrn_loaded_state_hash": wfa["loaded_state_dict_sha256"],
          "support_lists_source": "M3FULL SUPPORT_LISTS.json (冻结)",
          "queries_source": "data/splits/insplad-fd/query_k5.json (冻结, 10N/5N 同查询)",
          "supports": {a: {str(j): data[a]["n10"][j] for j in REFS} for a in ASSETS},
          "queries": {a: {"defect": data[a]["qd"], "normal": data[a]["qn"]} for a in ASSETS},
          "file_sha256": file_hashes,
          "code_sha256": {f: T.file_sha(f) for f in (
              D.ROOT + "/scripts/m3mech_run.py", D.ROOT + "/scripts/m3score_pcverify.py",
              D.ROOT + "/scripts/m3score_extract.py", D.ROOT + "/scripts/m3transfer_lib.py",
              D.ROOT + "/scripts/m3diag_common.py")},
          "modules": {
              "A1_Y-RAW": {"module": "YOLOv8n backbone layer4 (C2f)", "out": "64ch 40x40 -> 1600 pos",
                            "norm": "逐位置 L2", "pool": "无", "preprocess": "MC.make_transform(320)", "cache": "M3FULL 历史缓存 (身份绑定复用)"},
              "A2_Y-POOL": {"module": "同 A1 P3", "pool": "AvgPool2d(3,1,1) 在 L2 之前", "norm": "池化后逐位置 L2",
                             "out": "64ch 40x40 -> 1600 pos", "cache": "本任务新提取 (同确定性口径)"},
              "A3_W-STD": {"module": "timm wide_resnet50_2 layer2+layer3 -> feature_pooler(AvgPool2d(3,1,1)/层) -> 融合展平",
                            "out": "1536d x 2000 pos (1600+400)", "norm": "无额外 L2", "preprocess": "T.wrn_tf() 320"},
              "A4_W-NOPOOL": {"module": "同 A3 但去掉 feature_pooler", "out": "1536d x 2000 pos", "norm": "无额外 L2"}},
          "coreset": {"algo": "anomalib KCenterGreedy", "ratio": 0.1, "seeds": {str(j): 7000 + (j - 3000) for j in REFS},
                       "phaseB_M": 1000, "phaseB_note": "M=min(历史P3 10N容量1600, 层候选最小 10x100=1000); P5 候选恰=1000 全保留"},
          "scorings_phaseA": ["official_pc", "pure_max", "top4", "top8", "top16", "top32", "mean_all"],
          "scorings_phaseB": ["pure_max", "top8", "top0.5pct", "top1pct"],
          "G_definition": "G = AUROC(top8) - AUROC(pure_max); 比较 G(Y-RAW)vsG(Y-POOL), G(W-STD)vsG(W-NOPOOL), 逐参照配对",
          "frozen_samples": frozen_samples,
          "frozen_sample_rule": "每资产正常/故障各12张; PCG64(20250928) 置乱排序后路径, 裸DJI源组去重, 不足则披露; 冻结于任何新增结果计算之前",
          "failure_case_rules": {
              "F1_正常最高分": "每资产正常查询中 Y-RAW pure_max 最高的 5 张 (10N ref3001)",
              "F2_故障最低分": "每资产故障查询中 Y-RAW pure_max 最低的 5 张",
              "F3_方法分歧": "max 分数在各自臂内全查询分位的 |pct_Y - pct_W| 最大的 5 张",
              "label": "结果驱动失败分析, 非随机代表样本"},
          "data_identity": {
              "rule": "裸DJI号碰撞不作为同源确认; 号级100%碰撞是按号统计的保守上界, 不等于帧级包含",
              "known": "M3DIAG5 审计: 部分训练画面与查询裁剪源号重叠; 具体帧级包含关系未知",
              "spot_check": "2025-09-29 抽验3例同号全帧 (DJI_0089/0183/0014), SIFT+目视均未见对应裁剪内容; 详见会话记录",
              "contradiction_rule": "若文字与机读数字矛盾, 记录矛盾, 不沿用100%重叠表述"},
          "top8_note": "topK 指图像内最高异常距离位置数量; 最近邻数 nn=9 是另一参数; top8 无显式空间感知"}
    atomic_json(Fi, os.path.join(OUT, "FROZEN_INPUTS.json"))
    spec = ["# EXPERIMENT_SPEC — M3MECH-top8-v1 (预注册)", "",
            "## 阶段A 复现门 (新增结果之前)",
            "- A1 重建 Y-RAW 10N 库 ref3001 (yoke) 哈希 == M3BUDGET yoke_Y_10N_ref3001; A3 重建 W-STD 10N 库 ref3001 == yoke_W_10N_ref3001 / wrnfix yoke_ref3001",
            "- A1 official_pc/top8 对 M3FULL Y-PC/Y-T8 逐图 (两资产 x 5 参照, atol=rtol=1e-4); A3 对 W-PC/W-T8 同口径",
            "- 路径集合、标签侧、支持名单逐项一致; 不以 AUROC 替代",
            "## 阶段A 主分析",
            "- 每臂自有特征建库 (ratio 0.1, 冻结种子); 同一 patch 距离派生 7 种评分; G=top8-max 逐参照配对",
            "- A2 batch 门 (bs1 vs 32), A4 batch 门; 代表库双建库; 距离分块 vs 直接抽查; 状态哈希前后一致",
            "## 阶段B",
            "- P3=layer4/P4=layer6/P5=layer9(SPPF) 单层独立 L2 建库; M=1000; max/top8/top0.5%/top1% (k=max(1,ceil(q*H*W)))",
            "- P3 共同容量臂与历史 ratio=0.1 臂分开报告",
            "## 可视化",
            "- 固定随机组 12+12/资产 (已冻结于 FROZEN_INPUTS), 失败案例组按预注册规则; 固定颜色规则 (臂内正常样本 99.5 分位为 vmax), 不逐图 min-max",
            "## 纪律",
            "- 5 参照非独立数据集; 无显著性声明; 不在结果上挑最佳层/k 宣称提升; 机制解释区分已验证/待确证/被否定"]
    with open(os.path.join(OUT, "EXPERIMENT_SPEC.md"), "w") as f:
        f.write("\n".join(spec) + "\n")
    print("[freeze] FROZEN_INPUTS + SPEC 完成", flush=True)
    # --- 特征基础设施 ---
    from m3score_extract import MC
    B1 = MC.Backbone(D.WEIGHTS, "cuda")
    h0y = state_hash(B1.model)
    PM_W, _ = T.wrn_model()
    assert state_hash(PM_W.feature_extractor) == wfa["loaded_state_dict_sha256"], "WRN 加载态哈希与 wrnfix 冻结不符"
    h0w = state_hash(PM_W.feature_extractor)
    PM_SHELL = PM_W
    # --- A2/A4/阶段B 的 YOLO 新缓存 (P3-pool, P4, P5) ---
    FFNS = {}
    t = time.time()
    for a in ASSETS:
        uniq = sorted(set(p for j in REFS for p in data[a]["n10"][j]) | set(data[a]["allq"]))
        # A1 复用历史缓存
        man, arr = X.build_cache(uniq, cache_dir="/root/autodl-tmp/m3budget-%s-cache" % a)
        FFNS["yraw_" + a] = (arr, {p: i for i, p in enumerate(man["paths"])})
        # A2 Y-POOL
        cdir = os.path.join(OUT, "cache", "ypool_%s" % a)
        fp = cdir + "/feats.npy"
        mp = cdir + "/manifest.json"
        if not (os.path.exists(fp) and os.path.exists(mp) and json.load(open(mp))["paths"] == uniq):
            os.makedirs(cdir, exist_ok=True)
            raw = yolo_layers(uniq, 4)
            pooled = POOL3(raw)
            feats = l2_map(pooled).reshape(len(uniq), -1, 64).numpy()
            np.save(fp, feats)
            atomic_json({"paths": uniq, "norm": "AvgPool2d(3,1,1) 后逐位置 L2", "layer": "backbone layer4",
                          "shape": list(feats.shape), "code": T.file_sha(__file__)}, mp)
        man2 = json.load(open(mp))
        arr2 = np.load(fp, mmap_mode="r")
        FFNS["ypool_" + a] = (arr2, {p: i for i, p in enumerate(man2["paths"])})
        FFNS["p3_" + a] = FFNS["yraw_" + a]  # 阶段B P3 = A1 特征 (1600 pos, 64d, L2)
        # 阶段B P4/P5
        for li, lname, P, C_ in ((6, "p4", 400, 128), (9, "p5", 100, 256)):
            cdir = os.path.join(OUT, "cache", "%s_%s" % (lname, a))
            fp = cdir + "/feats.npy"
            mp = cdir + "/manifest.json"
            if not (os.path.exists(fp) and os.path.exists(mp) and json.load(open(mp))["paths"] == uniq):
                os.makedirs(cdir, exist_ok=True)
                raw = yolo_layers(uniq, li)
                feats = l2_map(raw).reshape(len(uniq), P, C_).numpy()
                np.save(fp, feats)
                atomic_json({"paths": uniq, "norm": "逐位置 L2", "layer": "backbone layer%d" % li,
                              "shape": list(feats.shape), "code": T.file_sha(__file__)}, mp)
            man3 = json.load(open(mp))
            arr3 = np.load(fp, mmap_mode="r")
            FFNS["%s_%s" % (lname, a)] = (arr3, {p: i for i, p in enumerate(man3["paths"])})
        print("[cache] %s 完成" % a, flush=True)
    LEDGER["caches_s"] = round(time.time() - t, 1)
    # --- 复现门 A1/A3 ---
    t = time.time()
    gates = {"repro": {}, "batch": {}, "double_build": {}, "state_hash": {"yolo": h0y == state_hash(B1.model)}}
    mb_banks = json.load(open(D.ROOT + "/outputs/M3budget-aligned/BANK_MANIFEST.json"))
    maxdiff = {"A1_pc": 0.0, "A1_t8": 0.0, "A3_pc": 0.0, "A3_t8": 0.0}
    def yraw_fn(a):
        arr, r2 = FFNS["yraw_" + a]
        return lambda ps: torch.from_numpy(np.ascontiguousarray(arr[[r2[p] for p in ps]])).reshape(len(ps), 1600, 64).float()
    def ypool_fn(a):
        arr, r2 = FFNS["ypool_" + a]
        return lambda ps: torch.from_numpy(np.ascontiguousarray(arr[[r2[p] for p in ps]])).reshape(len(ps), 1600, 64).float()
    def p_fn(a, lname):
        arr, r2 = FFNS["%s_%s" % (lname, a)]
        PD = {"p3": (1600, 64), "p4": (400, 128), "p5": (100, 256)}[lname]
        return lambda ps: torch.from_numpy(np.ascontiguousarray(arr[[r2[p] for p in ps]])).reshape(len(ps), PD[0], PD[1]).float()
    for a in ASSETS:
        d_ = data[a]
        for ref_gate in [3001]:
            seed = 7000 + (ref_gate - 3000)
            pm, meta = make_bank(yraw_fn(a), d_["n10"][ref_gate], seed)
            if a == "yoke-suspension" and ref_gate == 3001:
                assert hash_eq(th(pm.memory_bank), mb_banks["yoke_Y_10N_ref3001"]["tensor_hash"]), "A1 重建库哈希不符"
                gates["repro"]["A1_bank_hash_yoke"] = True
            Dm, pc = score_arm(yraw_fn(a), pm.memory_bank, d_["allq"], 1600)
            sc = scores_from_D(Dm, pc)
            for arm_m3, key_pc, key_t8 in (("Y-PC", "official_pc", "top8"),):
                src = json.load(open("%s/raw_scores/%s_%s_10N_ref%d.json" % (D.ROOT + "/outputs/M3full-frozen", COMBO[a], arm_m3, ref_gate)))
                qset = list(src["defect"].keys()) + list(src["normal"].keys())
                assert set(qset) == set(d_["allq"]), "路径集合不一致"
                idx = {p: i for i, p in enumerate(d_["allq"])}
                for p, v in src["defect"].items():
                    maxdiff["A1_pc"] = max(maxdiff["A1_pc"], abs(sc[key_pc][idx[p]] - v))
                for p, v in src["normal"].items():
                    maxdiff["A1_pc"] = max(maxdiff["A1_pc"], abs(sc[key_pc][idx[p]] - v))
                src8 = json.load(open("%s/raw_scores/%s_%s_10N_ref%d.json" % (D.ROOT + "/outputs/M3full-frozen", COMBO[a], "Y-T8", ref_gate)))
                for p, v in list(src8["defect"].items()) + list(src8["normal"].items()):
                    maxdiff["A1_t8"] = max(maxdiff["A1_t8"], abs(sc[key_t8][idx[p]] - v))
    for a in ASSETS:
        d_ = data[a]
        seed = 7001
        pm, meta = make_bank(wrn_ffn_pool, d_["n10"][3001], seed)
        if a == "yoke-suspension":
            assert hash_eq(th(pm.memory_bank), mb_banks["yoke_W_10N_ref3001"]["tensor_hash"]), "A3 重建库哈希不符"
            assert hash_eq(th(pm.memory_bank), wfa and json.load(open(D.ROOT + "/outputs/M3transfer-wrnfix/BANK_MANIFEST.json"))["yoke_ref3001"]["tensor_hash"]), "A3 vs wrnfix 哈希不符"
            gates["repro"]["A3_bank_hash_yoke"] = True
        Dm, pc = score_arm(wrn_ffn_pool, pm.memory_bank, d_["allq"], 2000)
        sc = scores_from_D(Dm, pc)
        src = json.load(open("%s/raw_scores/%s_W-PC_10N_ref3001.json" % (D.ROOT + "/outputs/M3full-frozen", COMBO[a])))
        idx = {p: i for i, p in enumerate(d_["allq"])}
        for p, v in list(src["defect"].items()) + list(src["normal"].items()):
            maxdiff["A3_pc"] = max(maxdiff["A3_pc"], abs(sc["official_pc"][idx[p]] - v))
        src8 = json.load(open("%s/raw_scores/%s_W-T8_10N_ref3001.json" % (D.ROOT + "/outputs/M3full-frozen", COMBO[a])))
        for p, v in list(src8["defect"].items()) + list(src8["normal"].items()):
            maxdiff["A3_t8"] = max(maxdiff["A3_t8"], abs(sc["top8"][idx[p]] - v))
    TOLA = 1e-4
    for k, v in maxdiff.items():
        print("[gate] %s maxdiff=%.2e" % (k, v), flush=True)
    assert max(maxdiff.values()) <= TOLA, "复现门失败: %s" % maxdiff
    gates["repro"]["max_diffs"] = maxdiff
    gates["repro"]["ok"] = True
    print("[gates] A1/A3 复现门通过", flush=True)
    LEDGER["gates_s"] = round(time.time() - t, 1)
    json.dump({"stage": "gates_done", "wall": round(time.time() - t00, 1)}, open(os.path.join(OUT, "run_progress.json"), "w"))
    if ARGS.phase == "A":
        return
    # --- batch 门 (A2/A4) ---
    t = time.time()
    a = "yoke-suspension"
    d_ = data[a]
    sub = d_["n10"][3001][:2] + d_["qd"][:3] + d_["qn"][:3]
    a2 = ypool_fn(a)(sub)
    a2_1 = torch.cat([ypool_fn(a)([p]) for p in sub], 0)
    wsub = d_["n10"][3001][:2] + d_["qd"][:2] + d_["qn"][:2]
    w32p = wrn_feats(wsub, True).reshape(len(wsub), -1, 1536)
    w32n = wrn_feats(wsub, False).reshape(len(wsub), -1, 1536)
    w1p = torch.cat([wrn_feats([p], True).reshape(1, -1, 1536) for p in wsub], 0)
    w1n = torch.cat([wrn_feats([p], False).reshape(1, -1, 1536) for p in wsub], 0)
    assert w32p.shape[1] == 2000 and w32n.shape[1] == 2000, "WRN 位置数与预期 2000 不符"
    bg = {"A2_feat_bs1_vs_32": float(np.abs(a2.numpy() - a2_1.numpy()).max()),
           "A3_feat_bs1_vs_32": float(np.abs(w32p.numpy() - w1p.numpy()).max()),
           "A4_feat_bs1_vs_32": float(np.abs(w32n.numpy() - w1n.numpy()).max()),
           "A2_vs_A1_feat_shift": float(np.abs(a2.numpy() - yraw_fn(a)(sub).numpy()).max()),
           "A4_vs_A3_feat_shift": float(np.abs(w32n.numpy() - w32p.numpy()).max()),
           "wrn_positions": 2000, "yolo_p3_positions": 1600}
    assert bg["A2_feat_bs1_vs_32"] <= 1e-4 + 1e-4 and bg["A4_feat_bs1_vs_32"] <= 1e-4 + 1e-4 and bg["A3_feat_bs1_vs_32"] <= 1e-4 + 1e-4
    # 固定库两 batch 分数门 (A2)
    pm2, _ = make_bank(ypool_fn(a), d_["n10"][3001], 7001)
    q6 = d_["qn"][:6]
    D64, pc64 = score_arm(ypool_fn(a), pm2.memory_bank, q6, 1600, bs_img=6)
    D1, pc1 = score_arm(ypool_fn(a), pm2.memory_bank, q6, 1600, bs_img=1)
    bg["A2_score_b1_vs_b6"] = float(np.abs(D64 - D1).max())
    assert bg["A2_score_b1_vs_b6"] <= 1e-4
    # 距离分块 vs 直接 (A2 抽样)
    q = ypool_fn(a)(q6).reshape(-1, 64).double()
    direct = torch.cdist(q, pm2.memory_bank.detach().cpu().double()).min(1).values.numpy()
    ch = D64.reshape(-1).astype(np.float64)
    bg["A2_chunked_vs_direct"] = float(np.abs(ch - direct).max())
    assert bg["A2_chunked_vs_direct"] <= 1e-4 + 1e-4 * float(np.abs(direct).max())
    gates["batch"] = bg
    gates["batch"]["ok"] = True
    print("[gates] batch 门通过", json.dumps({k: ("%.2e" % v) for k, v in bg.items() if isinstance(v, float)}), flush=True)
    # 双建库
    pm2b, _ = make_bank(ypool_fn(a), d_["n10"][3001], 7001)
    gates["double_build"]["A2_yoke_ref3001"] = bool(torch.equal(pm2.memory_bank, pm2b.memory_bank))
    pm4, _ = make_bank(wrn_ffn_nopool, d_["n10"][3001], 7001)
    pm4b, _ = make_bank(wrn_ffn_nopool, d_["n10"][3001], 7001)
    gates["double_build"]["A4_yoke_ref3001"] = bool(torch.equal(pm4.memory_bank, pm4b.memory_bank))
    gates["state_hash"]["wrn"] = True
    assert all(gates["double_build"].values())
    gates["double_build"]["ok"] = True
    LEDGER["batch_gates_s"] = round(time.time() - t, 1)
    gates["state_hash"]["yolo_after_gates"] = h0y == state_hash(B1.model)
    gates["state_hash"]["wrn_after_gates"] = h0w == state_hash(PM_W.feature_extractor)
    assert gates["state_hash"]["yolo_after_gates"] and gates["state_hash"]["wrn_after_gates"]
    atomic_json(gates, os.path.join(OUT, "repro_gates.json"))
    json.dump({"stage": "batch_gates_done", "wall": round(time.time() - t00, 1)}, open(os.path.join(OUT, "run_progress.json"), "w"))

    # ========== 阶段A 正式: 4 臂 ==========
    t = time.time()
    ARMS = [("Y-RAW", lambda a_: yraw_fn(a_), 1600, "phaseA"),
            ("Y-POOL", lambda a_: ypool_fn(a_), 1600, "phaseA"),
            ("W-STD", lambda a_: wrn_ffn_pool, 2000, "phaseA"),
            ("W-NOPOOL", lambda a_: wrn_ffn_nopool, 2000, "phaseA")]
    bank_manifest = {}
    Dsave = {}
    for arm_name, ffn, P, ph in ARMS:
        for a in ASSETS:
            d_ = data[a]
            fn = ffn(a)
            # W 臂: 每资产一次流式提取服务 5 库
            if arm_name.startswith("W-"):
                banks = {}
                for j in REFS:
                    pm, meta = make_bank(fn, d_["n10"][j], 7000 + (j - 3000))
                    banks[j] = (pm, meta)
                N = len(d_["allq"])
                Dall = {j: np.zeros((N, P), np.float32) for j in REFS}
                pcall = {j: np.zeros(N, np.float32) for j in REFS}
                PM_W.memory_bank = None
                with torch.inference_mode():
                    for i in range(0, N, W_BS):
                        ps_paths = d_["allq"][i:i + W_BS]
                        q = fn(ps_paths)
                        B = q.shape[0]
                        pe = q.reshape(B * P, q.shape[2]).to(PV.DEVICE)
                        for j in REFS:
                            PM_SHELL = None
                            pmj, _ = banks[j]
                            PM_W.memory_bank = pmj.memory_bank
                            ps, loc = PM_W.nearest_neighbors(pe, 1)
                            Dall[j][i:i + B] = ps.reshape(B, P).cpu().numpy()
                            pred = PM_W.compute_anomaly_score(ps.reshape(B, -1), loc.reshape(B, -1), pe.reshape(B, -1, q.shape[2]))
                            pcall[j][i:i + B] = pred.cpu().numpy()
                for j in REFS:
                    sc = scores_from_D(Dall[j], pcall[j])
                    save_arm_scores(arm_name, a, j, sc, d_, banks[j][1])
                    bank_manifest["%s_%s_ref%d" % (arm_name, a, j)] = banks[j][1]
                Dsave["%s_%s" % (arm_name, a)] = (Dall[3001], list(d_["allq"]))
                print("[arm] %s %s 完成 (5 refs)" % (arm_name, a), flush=True)
            else:
                for j in REFS:
                    pm, meta = make_bank(fn, d_["n10"][j], 7000 + (j - 3000))
                    Dm, pc = score_arm(fn, pm.memory_bank, d_["allq"], P)
                    sc = scores_from_D(Dm, pc)
                    save_arm_scores(arm_name, a, j, sc, d_, meta)
                    bank_manifest["%s_%s_ref%d" % (arm_name, a, j)] = meta
                    if j == 3001:
                        Dsave["%s_%s" % (arm_name, a)] = (Dm, list(d_["allq"]))
                print("[arm] %s %s 完成 (5 refs)" % (arm_name, a), flush=True)
    np.savez_compressed(os.path.join(OUT, "cache", "vis_dist", "ref3001_dist_matrices.npz"),
                        **{"%s__%s__D" % (k.split("_")[0], k.split("__")[0]) if False else k: v[0] for k, v in Dsave.items()},
                        **{k + "__order": json.dumps(v[1]) for k, v in Dsave.items()})
    LEDGER["phaseA_s"] = round(time.time() - t, 1)
    json.dump({"stage": "phaseA_done", "wall": round(time.time() - t00, 1)}, open(os.path.join(OUT, "run_progress.json"), "w"))

    # ========== 阶段B ==========
    t = time.time()
    M = 1000
    B_results = {}
    Db_save = {}
    for a in ASSETS:
        d_ = data[a]
        for lname, P, C_ in (("p3", 1600, 64), ("p4", 400, 128), ("p5", 100, 256)):
            fn = p_fn(a, lname)
            for j in REFS:
                seed = 7000 + (j - 3000)
                pm, meta = make_bank(fn, d_["n10"][j], seed, M=M)
                Dm, pc = score_arm(fn, pm.memory_bank, d_["allq"], P)
                s = {"pure_max": Dm.max(1).astype(np.float64)}
                s["top8"] = np.sort(Dm, 1)[:, -8:].mean(1).astype(np.float64)
                k05 = max(1, int(np.ceil(0.005 * P)))
                k1 = max(1, int(np.ceil(0.01 * P)))
                s["top0.5pct"] = np.sort(Dm, 1)[:, -k05:].mean(1).astype(np.float64)
                s["top1pct"] = np.sort(Dm, 1)[:, -k1:].mean(1).astype(np.float64)
                rec = {"asset": a, "layer": lname, "ref": j, "M": M, "k": {"top8": 8, "top0.5pct": k05, "top1pct": k1},
                        "defect": {p: {kk: float(vv[i]) for kk, vv in s.items()} for i, p in enumerate(d_["qd"])},
                        "normal": {p: {kk: float(vv[i]) for kk, vv in s.items()} for i, p in enumerate(d_["qn"])},
                        "bank": meta, "duplicate_k": k05 == k1}
                atomic_json(rec, os.path.join(OUT, "raw_scores", "phaseB", "%s_%s_%s_ref%d.json" % (COMBO[a], lname.upper(), a, j)).replace(a + "_", "", 1) if False else os.path.join(OUT, "raw_scores", "phaseB", "%s_%s_ref%d.json" % (COMBO[a], lname, j)))
                B_results["%s_%s_ref%d" % (COMBO[a], lname, j)] = rec
                bank_manifest["%s_%s_ref%d" % (lname, a, j)] = meta
                if j == 3001:
                    Db_save["%s_%s" % (lname, a)] = (Dm, list(d_["allq"]))
            print("[phaseB] %s %s 完成" % (a, lname), flush=True)
    for k, v in Db_save.items():
        Dsave[k] = v
    np.savez_compressed(os.path.join(OUT, "cache", "vis_dist", "phaseB_ref3001_dist.npz"),
                        **{k: v[0] for k, v in Db_save.items()},
                        **{k + "__order": json.dumps(v[1]) for k, v in Db_save.items()})
    LEDGER["phaseB_s"] = round(time.time() - t, 1)
    atomic_json(bank_manifest, os.path.join(OUT, "BANK_MANIFEST.json"))
    expA = sorted("%s_%s_ref%d.json" % (COMBO[a], ar, j) for ar in ("Y-RAW", "Y-POOL", "W-STD", "W-NOPOOL") for a in ASSETS for j in REFS)
    expB = sorted("%s_%s_ref%d.json" % (COMBO[a], ln, j) for ln in ("p3", "p4", "p5") for a in ASSETS for j in REFS)
    gotA = sorted(os.listdir(os.path.join(OUT, "raw_scores", "phaseA")))
    gotB = sorted(os.listdir(os.path.join(OUT, "raw_scores", "phaseB")))
    set_match = int(gotA == expA and gotB == expB and len(gotA) == 40 and len(gotB) == 30)
    gates["state_hash"]["yolo_final"] = h0y == state_hash(B1.model)
    gates["state_hash"]["wrn_final"] = h0w == state_hash(PM_W.feature_extractor)
    gates["set_match"] = set_match
    atomic_json(gates, os.path.join(OUT, "repro_gates.json"))
    feat_man = {"A1_Y-RAW": {"positions": 1600, "dim": 64, "norm": "逐位置L2", "source": "M3FULL 历史缓存"},
                "A2_Y-POOL": {"positions": 1600, "dim": 64, "norm": "AvgPool2d(3,1,1)后逐位置L2", "source": "cache/ypool_*"},
                "A3_W-STD": {"positions": 2000, "dim": 1536, "norm": "无额外L2 (官方池化)", "source": "流式"},
                "A4_W-NOPOOL": {"positions": 2000, "dim": 1536, "norm": "无额外L2 (无池化)", "source": "流式"},
                "phaseB": {"p3": {"pos": 1600, "ch": 64}, "p4": {"pos": 400, "ch": 128},
                            "p5": {"pos": 100, "ch": 256, "module": "backbone layer9 SPPF (含SPPF, 实证 shape [1,256,10,10])"}}}
    atomic_json(feat_man, os.path.join(OUT, "FEATURE_MANIFEST.json"))
    caches = {}
    for root2, _, files in os.walk(os.path.join(OUT, "cache")):
        for f2 in files:
            p2 = os.path.join(root2, f2)
            caches[os.path.relpath(p2, OUT)] = os.path.getsize(p2)
    atomic_json({"stages_s": LEDGER, "total_wall_s": round(time.time() - t00, 1),
                 "peak_vram_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                 "cache_bytes": caches, "gpu": torch.cuda.get_device_name(0),
                 "note": "缓存评分耗时不代表端到端推理延迟"}, os.path.join(OUT, "RESOURCE_COST.json"))
    json.dump({"stage": "complete", "wall": round(time.time() - t00, 1), "set_match": set_match}, open(os.path.join(OUT, "run_progress.json"), "w"))
    print("=== RUNNER COMPLETE wall=%.0fs set_match=%d ===" % (time.time() - t00, set_match), flush=True)


def save_arm_scores(arm_name, a, j, sc, d_, meta):
    rec = {"arm": arm_name, "asset": a, "ref": j, "budget": "10N", "bank": meta,
            "defect": {p: {k: float(v[i]) for k, v in sc.items()} for i, p in enumerate(d_["qd"])},
            "normal": {p: {k: float(v[i]) for k, v in sc.items()} for i, p in enumerate(d_["qn"])}}
    atomic_json(rec, os.path.join(OUT, "raw_scores", "phaseA", "%s_%s_%s_ref%d.json" % (COMBO[a], arm_name, a, j)).replace(a + "_", "", 1) if False else os.path.join(OUT, "raw_scores", "phaseA", "%s_%s_ref%d.json" % (COMBO[a], arm_name, j)))


if __name__ == "__main__":
    main()
