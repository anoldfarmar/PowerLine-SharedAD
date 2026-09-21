#!/usr/bin/env python3
"""M3DIAG2 阶段二: A/B/C(/C2) 推理态表示诊断 (简报四).
只用固定可追溯训练子集 (框 0-1023, 16 位/框), 不触碰查询数据.
输出 outputs/M3diag2/repr_diag.json.
注意: 框内 16 patch 高度相关, 统计置信度不能按独立图片数计.
"""
import json, os, sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m3diag_common as D
import m3_common as C

NB = 1024

def bn_report(head):
    out = []
    for name, m in head.named_modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            rm, rv, nb = m.running_mean, m.running_var, int(m.num_batches_tracked)
            out.append({"layer": name, "num_batches_tracked": nb,
                        "rm_std": round(float(rm.std()), 6),
                        "rm_absmax": round(float(rm.abs().max()), 6),
                        "rv_min": round(float(rv.min()), 6),
                        "rv_max": round(float(rv.max()), 6),
                        "finite": bool(torch.isfinite(rm).all() and torch.isfinite(rv).all())})
    return out

def diag_variant(tag, spec, heads, Xb, Xc):
    dev = "cuda"
    with torch.no_grad():
        kind, obj = spec
        raw = Xb.view(-1, 64).to(dev)
        if kind == "proj":
            z = raw @ obj.to(dev)
        else:
            z = obj(raw)
        zn = F.normalize(z, dim=1)
    z = z.cpu().numpy(); zn = zn.cpu().numpy()
    cov = np.cov(z.T.astype(np.float64))
    ev = np.linalg.eigvalsh(cov)[::-1]
    ev = np.clip(ev, 0, None)
    pr = float(ev.sum() ** 2 / max(1e-12, (ev ** 2).sum()))
    p = ev / ev.sum(); p = p[p > 1e-12]
    eff = float(np.exp(-(p * np.log(p)).sum()))
    NB, PP = Xb.shape[0], Xb.shape[1]
    zt = torch.from_numpy(zn)
    same = (zt.view(NB, PP, -1)[:, :, None, :] * zt.view(NB, 1, PP, -1)).sum(-1)
    iu = torch.triu_indices(PP, PP, offset=1)
    same_pairs = same[:, iu[0], iu[1]].reshape(-1)
    g = torch.Generator().manual_seed(0)
    a = torch.randint(0, zt.shape[0], (20000,), generator=g)
    b = torch.randint(0, zt.shape[0], (20000,), generator=g)
    keep = a != b
    cross = (zt[a[keep]] * zt[b[keep]]).sum(-1)
    res = {"n_patches": int(z.shape[0]),
           "pre_dim_std_mean": round(float(z.std(0, ddof=0).mean()), 5),
           "pre_dim_std_min": round(float(z.std(0, ddof=0).min()), 5),
           "vec_norm_mean": round(float(np.linalg.norm(z, axis=1).mean()), 4),
           "vec_norm_std": round(float(np.linalg.norm(z, axis=1).std()), 4),
           "post_dim_std_mean": round(float(zn.std(0, ddof=0).mean()), 5),
           "post_dim_std_min": round(float(zn.std(0, ddof=0).min()), 5),
           "eig_top8": [round(float(x), 5) for x in ev[:8]],
           "participation_ratio": round(pr, 3), "effective_rank_expent": round(eff, 3),
           "cos_samebox_crosspos": {"mean": round(float(same_pairs.mean()), 4),
                                    "std": round(float(same_pairs.std()), 4)},
           "cos_crossbox": {"mean": round(float(cross.mean()), 4),
                            "std": round(float(cross.std()), 4)},
           "bn": bn_report(heads[tag]) if tag in heads else []}
    if tag in ("C", "C2") and Xc is not None:
        with torch.no_grad():
            v1 = obj(Xc[:, 0].reshape(-1, 64).to(dev)) if kind == "head" else Xc[:, 0].reshape(-1, 64).to(dev) @ obj.to(dev)
            v2 = obj(Xc[:, 1].reshape(-1, 64).to(dev)) if kind == "head" else Xc[:, 1].reshape(-1, 64).to(dev) @ obj.to(dev)
        n1 = F.normalize(v1, dim=1); n2 = F.normalize(v2, dim=1)
        cs = (n1 * n2).sum(1)
        res["view_pair_samepos_cos"] = {"mean": round(float(cs.mean()), 4),
                                        "std": round(float(cs.std()), 4), "n": int(cs.shape[0])}
    return res

def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    assert dev == "cuda", "需要 GPU"
    Xb = torch.from_numpy(np.load(D.CACHE + "/trainB_patches.npy")[:NB]).float()
    Xc = torch.from_numpy(np.asarray(
        np.load(D.CACHE + "/trainC_patches.npy", mmap_mode="r")[:NB])).float()
    heads = {}
    heads["B"] = D.load_head_state(D.OUT + "/mapper_B.pt", dev).to(dev).eval()
    heads["C"] = D.load_head_state(D.OUT + "/mapper_C.pt", dev).to(dev).eval()
    p2 = D.OUT + "/mapper_C_bnrecal.pt"
    if os.path.exists(p2):
        heads["C2"] = D.load_head_state(p2, dev).to(dev).eval()
    proj = torch.from_numpy(np.asarray(C.patch_projection_matrix())).float()
    specs = {"A": ("proj", proj), "B": ("head", heads["B"]), "C": ("head", heads["C"])}
    if "C2" in heads:
        specs["C2"] = ("head", heads["C2"])
    out = {"fixed_subset": {"boxes": "0..%d (index order)" % (NB - 1), "patches_per_box": 16,
                            "note": "框内 patch 相关, 不能当独立图片计置信度"},
           "std_ddof": 0, "variants": {}}
    for tag in ("A", "B", "C", "C2"):
        if tag not in specs:
            continue
        out["variants"][tag] = diag_variant(tag, specs[tag], heads, Xb, Xc)
        print("[repr]", tag, "pre_std_mean=%.4f post_std_mean=%.4f PR=%.2f effrank=%.2f" % (
            out["variants"][tag]["pre_dim_std_mean"], out["variants"][tag]["post_dim_std_mean"],
            out["variants"][tag]["participation_ratio"], out["variants"][tag]["effective_rank_expent"]))
    D.save_json(out, D.OUT + "/repr_diag.json")

if __name__ == "__main__":
    main()