#!/usr/bin/env python3
"""M3TRANSFER 共享库: WRN 官方特征/库/距离 + YOLO 复用 (M3SCORE 组件) + top8."""
import hashlib, json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as Fn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m3score_pcverify as PV
import m3score_extract as X

DEV = "cuda"
TOPK = 8


def file_sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def deterministic():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def top8_mean(dist2d):
    """dist2d (B,P) -> (B,) 每图 patch 轴 top8 均值 (合成验证轴正确性见 verify_top8_axis)."""
    return torch.topk(dist2d, TOPK, dim=1).values.mean(1)


def verify_top8_axis():
    d = torch.arange(400, dtype=torch.float64).reshape(4, 100)
    got = top8_mean(d)
    ref = torch.stack([row.topk(TOPK).values.mean() for row in d])
    wrong = torch.stack([col.topk(TOPK).values.mean() for col in d.T])  # 错误轴 (原型/列)
    return {"axis_ok": bool(torch.equal(got, ref)), "wrong_axis_differs": bool(not torch.equal(got, wrong))}


def wrn_model():
    """官方 PatchcoreModel (pre_trained WRN50, layer2+3). 返回 (pm, state_hash)."""
    pm = PV._cls()(layers=["layer2", "layer3"], backbone="wide_resnet50_2", pre_trained=True, num_neighbors=9).to(DEV)
    pm.eval()
    sd = pm.feature_extractor.state_dict()
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode())
        h.update(sd[k].detach().cpu().numpy().tobytes())
    return pm, h.hexdigest()


def wrn_tf():
    import torchvision.transforms as T2
    return T2.Compose([T2.Resize((320, 320), antialias=True),
                       T2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


@torch.inference_mode()
def wrn_extract(pm, paths, bs=16):
    """官方 generate_embedding(feature_extractor) -> (N,1536,40,40) fp32 cpu."""
    from PIL import Image
    tf = wrn_tf()
    outs, exp = [], None
    for i in range(0, len(paths), bs):
        xb = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths[i:i + bs]]).to(DEV)
        feats = pm.feature_extractor(xb)
        emb = pm.generate_embedding(feats)
        if exp is None:
            exp = list(emb.shape)
            assert emb.shape[1] == 1536, "WRN 融合通道异常: %s" % list(emb.shape)
        assert torch.isfinite(emb).all()
        outs.append(emb.cpu())
    return torch.cat(outs, 0), exp


def build_bank(feats_flat, seed, ratio=0.1):
    """feats_flat (N,D) -> seeded KCenterGreedy -> (bank, indices). 建库前重置 RNG."""
    import random as _r
    _r.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    fb = feats_flat.float().to(DEV)
    kg = PV.KCenterGreedy(embedding=fb, sampling_ratio=ratio)
    idxs = kg.select_coreset_idxs()
    bank = kg.embedding[torch.as_tensor(idxs, device=DEV)]
    return bank, [int(i) for i in idxs]


@torch.inference_mode()
def wrn_scores_and_dists(pm, bank, paths, bs=16):
    """流式: 每 batch 官方 reshape_embedding -> nearest_neighbors(1) -> PC 分数 + T8 距离. 返回 (pc_vals, dists(n,P))."""
    from PIL import Image
    tf = wrn_tf()
    vals, dist_rows = [], []
    for i in range(0, len(paths), bs):
        xb = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths[i:i + bs]]).to(DEV)
        emb = pm.generate_embedding(pm.feature_extractor(xb))
        B, C = emb.shape[0], emb.shape[1]
        pe = PV._flat(pm.reshape_embedding(emb)).float()
        ps, loc = pm.nearest_neighbors(pe, 1)  # 官方分块 (chunk 1024) 跨块+尾块
        pred = pm.compute_anomaly_score(ps.reshape(B, -1), loc.reshape(B, -1), pe.reshape(B, -1, C))
        vals.extend(float(x) for x in pred)
        dist_rows.append(ps.reshape(B, -1).cpu())
    assert all(np.isfinite(vals))
    return vals, torch.cat(dist_rows, 0)


def verify_wrn_distance(pm, seed=23):
    """官方距离 vs float64 直接 (含跨块: >1024 行触发内部分块, 尾块)."""
    torch.manual_seed(seed)
    bank = torch.randn(300, 1536).cuda()
    q = torch.randn(2500, 1536).cuda()  # 2500 = 2 chunk + 尾块 452
    pm.memory_bank = bank
    with torch.no_grad():
        ps, loc = pm.nearest_neighbors(q, 1)
    rows = torch.linspace(0, q.shape[0] - 1, 40).long()
    ref = torch.cdist(q.double()[rows], bank.double()).min(1).values
    err = float((ps.double()[rows] - ref).abs().max())
    idx_ok = bool(torch.allclose(bank.double()[loc[rows]].sub(q.double()[rows]).norm(dim=-1), ref, atol=1e-3))
    n_chunks = (q.shape[0] + 1023) // 1024
    tail = q.shape[0] % 1024 != 0
    return {"max_abs_err": err, "nn_index_ok": idx_ok, "chunks": n_chunks, "tail": tail,
            "pass": bool(err <= 1e-3 and idx_ok and n_chunks >= 2 and tail)}


def verify_reference_composition(pm, seed=29, d=1536):
    """独立公式组合 (M3SCORE reference) vs 官方 compute_anomaly_score, atol=rtol=1e-4."""
    torch.manual_seed(seed)
    bank = Fn.normalize(torch.randn(100, d), dim=-1).float().cuda()
    q = Fn.normalize(torch.randn(4, 25, d), dim=-1).float().cuda()
    pm.memory_bank = bank
    with torch.no_grad():
        pe = PV._flat(pm.reshape_embedding(q.permute(0, 2, 1).reshape(4, d, 5, 5))).float().cuda()
        ps, loc = pm.nearest_neighbors(pe, 1)
        official = pm.compute_anomaly_score(ps.reshape(4, 25), loc.reshape(4, 25), pe.reshape(4, 25, d))
        ref = PV.reference_image_score(pm, ps.reshape(4, 25), loc.reshape(4, 25), pe.reshape(4, 25, d))
    diff = (official.double() - ref.double()).abs()
    ok = bool((diff <= 1e-4 + 1e-4 * ref.double().abs()).all())
    return {"max_abs_diff": float(diff.max()), "pass": ok}


def verify_batch_score(feat_batch_fn, rows, pm, bs_list=(1, 16)):
    """固定库 batch=1 vs 正式 batch 评分一致性 (atol=rtol=1e-4). feat_batch_fn(rows)->(n,P,d)."""
    outs = {}
    with torch.no_grad():
        for bs in bs_list:
            vals = []
            for i in range(0, len(rows), bs):
                q = feat_batch_fn(rows[i:i + bs])
                B, C = q.shape[0], q.shape[-1]
                pe = PV._flat(pm.reshape_embedding(q)).float()
                ps, loc = pm.nearest_neighbors(pe, 1)
                pred = pm.compute_anomaly_score(ps.reshape(B, -1), loc.reshape(B, -1), pe.reshape(B, -1, C))
                vals.extend(float(x) for x in pred)
            outs[bs] = np.array(vals)
    d = np.abs(outs[1] - outs[bs_list[1]])
    ok = bool(np.all(d <= 1e-4 + 1e-4 * np.abs(outs[bs_list[1]])))
    return {"max_abs": float(d.max()), "pass": ok}
