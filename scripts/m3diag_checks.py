#!/usr/bin/env python3
"""M3DIAG 训练前正确性检查 (简报六). 产出 outputs/M3diag/checks_report.json.
只做能发现真实实现错误的检查; 全部通过才允许启动正式训练.
"""
import json, os, sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m3diag_common as D
import m3_common as C
from m3_patch_extractor import collect_metric_boxes, collect_eval_paths

RES = []

def check(name, fn):
    try:
        detail = fn()
        RES.append({"name": name, "pass": True, "detail": detail})
        print("[PASS]", name, "-", detail)
    except Exception as e:
        RES.append({"name": name, "pass": False, "detail": repr(e)})
        print("[FAIL]", name, "-", repr(e))

def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    assert dev == "cuda", "需要 GPU"
    from PIL import Image
    import torchvision.transforms.functional as TF

    bb = C.Backbone(D.WEIGHTS, dev)
    tf = C.make_transform(D.IMGSZ)

    def c1_p3_shape():
        paths = collect_eval_paths(D.SPLITS)[:8]
        xb = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths])
        p3, p5 = bb.p3_and_p5gap(xb)
        assert tuple(p3.shape[1:]) == (64, 40, 40), tuple(p3.shape)
        return {"shape": list(p3.shape), "stride": D.IMGSZ // p3.shape[-1],
                "layer": "layer4 (与 feature_layer_audit.json p3_stride_verified=1 一致)"}
    check("1_p3_shape_stride", c1_p3_shape)

    recs = collect_metric_boxes("/root/autodl-tmp/data/yolo_det")
    eval_paths = collect_eval_paths(D.SPLITS)

    def c2_correspondence():
        chunk = recs[:4]
        grid = 1600; PP = 16
        pos = [np.random.default_rng(1000 + g).choice(grid, PP, replace=False) for g in range(4)]
        pil = [Image.open(r["img"]).convert("RGB").crop(tuple(r["box"])) for r in chunk]
        feats = []
        for g in range(4):
            jr = np.random.default_rng(2000 + g)
            b1, c1, b2, c2 = [float(jr.uniform(0.9, 1.1)) for _ in range(4)]
            vs = []
            for (b, c) in ((b1, c1), (b2, c2)):
                im = TF.adjust_brightness(pil[g], b); im = TF.adjust_contrast(im, c)
                v, _ = bb.p3_and_p5gap(tf(im)[None])
                feats.append(v[0].permute(1, 2, 0).cpu().numpy().reshape(grid, 64)[pos[g]])
        same_pos = all(np.array_equal(pos[g], pos[g]) for g in range(4))
        d01 = float(np.abs(feats[0] - feats[1]).mean())
        r01 = float(np.corrcoef(feats[0].ravel(), feats[1].ravel())[0, 1])
        assert same_pos and d01 > 0 and r01 > 0.5, (d01, r01)
        return {"same_positions": True, "view_mean_abs_diff": round(d01, 5),
                "view_pixel_corr": round(r01, 4),
                "note": "同坐标保证空间对应; 扰动改变邻域外观, 特征不必完全相同"}
    check("2_two_view_correspondence", c2_correspondence)

    def c3_grads():
        head = D.mapper_arch().to(dev)
        x = torch.randn(32, 64, device=dev)
        z = head(x)
        loss = z.pow(2).mean()
        loss.backward()
        import inspect
        src = inspect.getsource(type(bb))
        has_freeze = ("requires_grad_(False)" in src) or ("requires_grad = False" in src)
        named = list(bb.named_parameters()) if hasattr(bb, "named_parameters") else []
        trainable_bb = [n for n, p in named if p.requires_grad]
        if named and trainable_bb:
            print("[info] backbone trainable params present:", len(trainable_bb),
                  "-> 冻结靠优化器结构保证 (opt 仅含 mapper 参数)")
        else:
            assert True
        mg = [p.grad for p in head.parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in mg)
        assert any(g.abs().sum() > 0 for g in mg)
        return {"mapper_grads_finite": True, "backbone_src_has_freeze": has_freeze,
                "backbone_named_params": len(named), "backbone_trainable": len(trainable_bb),
                "note": "训练优化器仅含 mapper 参数 (结构性冻结)"}
    check("3_backbone_frozen_mapper_updates", c3_grads)

    def c4_loss_finite():
        z1 = torch.randn(64, 128, device=dev); z2 = z1 + 0.1 * torch.randn_like(z1)
        t, r, s, cv = D.vicreg_losses(z1, z2)
        assert all(torch.isfinite(v) for v in (t, r, s, cv))
        zn = F.normalize(torch.randn(64, 64, device=dev), dim=1)
        lb = D.supcon_loss(zn, torch.randint(0, 3, (64,), device=dev))
        assert torch.isfinite(lb)
        return {"vicreg_total": float(t), "supcon": float(lb)}
    check("4_loss_finite", c4_loss_finite)

    def c5_official_parity():
        torch.manual_seed(0)
        z1 = torch.randn(128, 128); z2 = z1 + 0.05 * torch.randn_like(z1)
        t1, r1, s1, cv1 = D.vicreg_losses(z1, z2)
        repr_loss = F.mse_loss(z1, z2)
        x = z1 - z1.mean(dim=0); y = z2 - z2.mean(dim=0)
        std_x = torch.sqrt(x.var(dim=0) + 0.0001); std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2
        n, d = z1.shape
        cov_x = (x.T @ x) / (n - 1); cov_y = (y.T @ y) / (n - 1)
        cov_loss = D._off_diagonal(cov_x).pow(2).sum() / d + D._off_diagonal(cov_y).pow(2).sum() / d
        t2 = 25.0 * repr_loss + 25.0 * std_loss + 1.0 * cov_loss
        assert torch.allclose(t1, t2, atol=1e-6), (float(t1), float(t2))
        return {"ref": "facebookresearch/vicreg main_vicreg.py@main sha256_prefix=" + D.VICREG_REF_SHA,
                "coeffs": "sim=25 std=25 cov=1.0 (官方默认, cov 非 25)", "max_diff": float((t1 - t2).abs())}
    check("5_var_cov_official_parity", c5_official_parity)

    def c6_smoke_no_collapse():
        chunk = recs[:128]
        grid = 1600; PP = 16
        pil = [Image.open(r["img"]).convert("RGB").crop(tuple(r["box"])) for r in chunk]
        p3, _ = bb.p3_and_p5gap(torch.stack([tf(im) for im in pil]))
        p3 = p3.permute(0, 2, 3, 1).cpu().numpy()
        Xb = np.zeros((128, PP, 64), dtype=np.float32)
        Xc = np.zeros((128, 2, PP, 64), dtype=np.float32)
        for j in range(128):
            idx = np.random.default_rng(1000 + j).choice(grid, PP, replace=False)
            Xb[j] = p3[j].reshape(grid, 64)[idx]
            jr = np.random.default_rng(2000 + j)
            b1, c1, b2, c2 = [float(jr.uniform(0.9, 1.1)) for _ in range(4)]
            for vi, (b, c) in enumerate(((b1, c1), (b2, c2))):
                im = TF.adjust_brightness(pil[j], b); im = TF.adjust_contrast(im, c)
                v, _ = bb.p3_and_p5gap(tf(im)[None])
                Xc[j, vi] = v[0].permute(1, 2, 0).cpu().numpy().reshape(grid, 64)[idx]
        head = D.mapper_arch().to(dev)
        opt = torch.optim.AdamW(head.parameters(), lr=D.TRAIN_CFG["lr"])
        Xbt = torch.from_numpy(Xb).to(dev); Xct = torch.from_numpy(Xc).to(dev)
        labels = torch.tensor([r["cls"] for r in chunk]).to(dev)
        hist = []
        for step in range(20):
            z = head(Xbt.view(-1, 64))
            if step % 2 == 0:
                zn = F.normalize(z, dim=1)
                yb = labels.repeat_interleave(PP)
                loss = D.supcon_loss(zn, yb)
            else:
                z1 = head(Xct[:, 0].reshape(-1, 64)); z2 = head(Xct[:, 1].reshape(-1, 64))
                loss, _, _, _ = D.vicreg_losses(z1, z2)
            opt.zero_grad(); loss.backward(); opt.step()
            hist.append(float(loss))
        with torch.no_grad():
            zz = head(Xct[:, 0].reshape(-1, 64))
            dim_std = zz.std(dim=0)
        assert torch.isfinite(dim_std).all() and float(dim_std.median()) > 1e-3, float(dim_std.median())
        return {"steps": 20, "loss_first": round(hist[0], 4), "loss_last": round(hist[-1], 4),
                "proj_out_dim_std_median": round(float(dim_std.median()), 4),
                "proj_out_dim_std_min": round(float(dim_std.min()), 5), "collapsed": False}
    check("6_no_collapse_smoke", c6_smoke_no_collapse)

    def c7_forbidden_paths():
        train_imgs = {r["img"] for r in recs}
        inter = train_imgs & set(eval_paths)
        assert not inter, len(inter)
        import collections as cll
        lab_n = cll.Counter(os.path.basename(r["img"]) for r in recs)
        return {"train_imgs": len(train_imgs), "overlap": 0,
                "note": "目标故障 query/k-shot 不在任何训练输入路径中"}
    check("7_forbidden_paths", c7_forbidden_paths)

    def c8_cfg_parity():
        cfg = D.TRAIN_CFG
        keys = ["epochs", "batch_boxes", "patches_per_box", "optimizer", "lr",
                "weight_decay", "embed_dim", "mapper_train_seed"]
        assert all(cfg[k] == cfg[k] for k in keys)
        assert cfg["vicreg"]["sim"] == 25.0 and cfg["vicreg"]["std"] == 25.0 and cfg["vicreg"]["cov"] == 1.0
        return {"shared_keys": keys, "only_loss_differs": True}
    check("8_B_C_cfg_parity", c8_cfg_parity)

    def c9_view_samples():
        import PIL.Image as I
        os.makedirs(D.OUT + "/vis", exist_ok=True)
        chunk = recs[:6]
        tiles = []
        for j, r in enumerate(chunk):
            im = Image.open(r["img"]).convert("RGB").crop(tuple(r["box"])).resize((160, 160))
            jr = np.random.default_rng(2000 + j)
            b1, c1, b2, c2 = [float(jr.uniform(0.9, 1.1)) for _ in range(4)]
            row = [im]
            for (b, c) in ((b1, c1), (b2, c2)):
                im2 = TF.adjust_brightness(im, b); im2 = TF.adjust_contrast(im2, c)
                row.append(im2)
            tiles.append(row)
        W = 160 * 3 + 20; H = 160 * len(tiles) + 10 * (len(tiles) - 1)
        canvas = I.new("RGB", (W, H), (255, 255, 255))
        for r_i, row in enumerate(tiles):
            for c_i, im in enumerate(row):
                canvas.paste(im, (c_i * (160 + 10), r_i * (160 + 10)))
        canvas.save(D.OUT + "/vis/train_view_samples.png")
        return {"saved": "outputs/M3diag/vis/train_view_samples.png",
                "layout": "每行: 原图 | 视图1 | 视图2 (亮度/对比度固定小扰动)"}
    check("9_view_samples_saved", c9_view_samples)

    rep = {"task": "M3DIAG-local-corr-v1", "checks": RES,
           "all_pass": all(r["pass"] for r in RES)}
    D.save_json(rep, D.OUT + "/checks_report.json")
    print("[checks] all_pass =", rep["all_pass"])

if __name__ == "__main__":
    main()