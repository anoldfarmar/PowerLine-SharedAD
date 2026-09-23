#!/usr/bin/env python3
"""M3FEAT 特征提取: E1 (P3 原始 L2, 缓存), E2 (P3/P4/P5 融合, 新鲜 FP32 缓存). E0 走历史路径见 heads/run."""
import hashlib, json, os, sys, tempfile
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m3diag_common as D
import m3_common as MC

OUT = D.ROOT + "/outputs/M3feat-yoke"
CACHE = OUT + "/cache"


def l2_pos(x):
    """逐位置 L2 归一化 (..., C) -> 同形."""
    return F.normalize(x, dim=-1)


def atomic_save_npy(arr, path):
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path))
    os.close(fd)
    np.save(tmp, arr)
    os.replace(tmp + ".npy" if not tmp.endswith(".npy") else tmp, path)


def atomic_save_json(obj, path):
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path))
    with os.fdopen(fd, "w") as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp, path)


def e1_features(rows, patches):
    """E1: 缓存 P3 (fp16 历史) -> fp32 -> 逐位置 L2. 返回 (n, P, d) torch fp32."""
    raw = torch.from_numpy(np.asarray(patches[rows])).float()
    n = raw.shape[0]
    f = l2_pos(raw.reshape(n, -1, raw.shape[-1]))
    assert torch.isfinite(f).all()
    return f


class Backbone3:
    """同一冻结 YOLOv8n, 一次前向取 P3/P4/P5. 节点以 shape 断言确认, 不凭名称猜."""
    def __init__(self, device):
        self.bb = MC.Backbone(D.WEIGHTS, device)
        self.device = device
        h0 = self.param_hash()
        self.hash_before = h0
    def param_hash(self):
        h = hashlib.sha256()
        for k in sorted(self.bb.model.state_dict()):
            h.update(k.encode())
            h.update(self.bb.model.state_dict()[k].detach().cpu().numpy().tobytes())
        return h.hexdigest()[:16]
    @torch.no_grad()
    def forward(self, xb):
        feats = self.bb.forward_layers(xb)
        return feats[4], feats[6], feats[9]


def extract_e2(paths, bb, tf, bs=32):
    """E2: P3/P4/P5 融合特征, FP32. 返回 (n, 40, 40, 448) 与层结构记录."""
    mods = {i: type(bb.bb.model.model[i]).__name__ for i in (4, 6, 9)}
    outs = []
    exp = None
    for i in range(0, len(paths), bs):
        xb = torch.stack([tf(p) for p in paths[i:i + bs]]).to(bb.device)
        p3, p4, p5 = bb.forward(xb)
        if exp is None:
            exp = {"P3": list(p3.shape), "P4": list(p4.shape), "P5": list(p5.shape)}
            assert p3.shape[1] == 64 and p4.shape[1] == 128 and p5.shape[1] == 256, "通道数与预期不符"
            assert p3.shape[-1] == 40 and p4.shape[-1] == 20 and p5.shape[-1] == 10, "空间尺寸与预期不符"
        grid = p3.shape[-2:]
        p4u = F.interpolate(p4, size=grid, mode="bilinear", align_corners=False)
        p5u = F.interpolate(p5, size=grid, mode="bilinear", align_corners=False)
        z = torch.cat([l2_pos(p3), l2_pos(p4u), l2_pos(p5u)], dim=1)
        z = l2_pos(z)
        assert torch.isfinite(z).all()
        outs.append(z.permute(0, 2, 3, 1).cpu())
    A = torch.cat(outs, 0).numpy().astype(np.float32)
    return A, {"layers": mods, "shapes": exp, "dim": int(A.shape[-1]), "grid": [int(A.shape[1]), int(A.shape[2])],
               "norm": "逐层逐位置 L2 后等权拼接再整体 L2", "interp": "bilinear align_corners=False", "dtype": "float32"}


def build_e2_cache(unique_paths, force=False):
    """唯一路径提取一次, 原子写缓存; 返回 (rows_map, manifest)."""
    os.makedirs(CACHE, exist_ok=True)
    fp = CACHE + "/e2_feats.npy"
    lp = CACHE + "/e2_paths.json"
    if not force and os.path.exists(fp) and os.path.exists(lp):
        paths = json.load(open(lp))["paths"]
        if paths == list(unique_paths):
            return {p: i for i, p in enumerate(paths)}, json.load(open(lp))["meta"], None
    bb = Backbone3("cuda")
    tf = MC.make_transform(D.IMGSZ)
    A, meta = extract_e2(list(unique_paths), bb, tf)
    hash_after = bb.param_hash()
    atomic_save_npy(A, fp)
    atomic_save_json({"paths": list(unique_paths), "meta": meta, "backbone_hash": hash_after}, lp)
    return {p: i for i, p in enumerate(unique_paths)}, meta, hash_after


def e2_features(rows, e2map, e2arr):
    raw = torch.from_numpy(np.ascontiguousarray(e2arr[[e2map[p] for p in rows]]))
    return raw.reshape(raw.shape[0], -1, raw.shape[-1])


if __name__ == "__main__":
    print("module only")