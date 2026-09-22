#!/usr/bin/env python3
"""M3BENCH: 双库方法 (1N1A/5N5A) 对 PatchCore (1N/10N) 探索性直接比较."""
import hashlib, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m3diag_common as D
import m3diag6_dualbank as M6
from m3diag_bank_eval import embed
from m3diag3_faultbank import extract_crops
from m3_common import C as MC

ROOT = D.ROOT
OUT = ROOT + "/outputs/M3bench-dualbank-patchcore"
REF_SEEDS = [3001, 3002, 3003, 3004, 3005]
SEEDS = [0, 1, 2]
BUDGETS = {"g1": {"k": 1, "n_sub": "N1", "pc_n": "N1"}, "g2": {"k": 5, "n_sub": "N5", "pc_n": "N10"}}
COMBOS = M6.COMBOS
TOL = 1e-4

def sf(p):
    return M6.sha256_file(p)

def h16(t):
    return hashlib.sha256(np.ascontiguousarray(t.detach().cpu().numpy()).tobytes()).hexdigest()[:16]

def main():
    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    assert dev == "cuda"
    for d_ in ("raw_scores_ours", "raw_scores_patchcore"):
        os.makedirs(OUT + "/" + d_, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    rg = json.load(open(M6.OUT4 + "/ref_groups.json"))
    eval_paths = json.load(open(D.CACHE + "/eval_paths.json"))
    path2idx = {p: i for i, p in enumerate(eval_paths)}
    patches = np.load(D.CACHE + "/patches.npy", mmap_mode="r")
    spec = ("head", D.load_head_state(D.OUT + "/mapper_B.pt", dev).to(dev))
    head = spec[1]; head.eval()
    for p in head.parameters():
        p.requires_grad_(False)
    mh0 = M6.state_hash(head)

    # ---------- F. 冻结名单 ----------
    lists = {}
    for a in rg["assets"]:
        for j in REF_SEEDS:
            p10 = sorted(rg["assets"][a]["groups"][str(j)])
            perm = np.random.Generator(np.random.PCG64(seed=j)).permutation(len(p10))
            ordered = [p10[int(t)] for t in perm]
            lists["%s_ref%d" % (a, j)] = {"sorted_paths": p10, "perm": [int(x) for x in perm],
                "ordered_paths": ordered, "N1": ordered[:1], "N5": ordered[:5], "N10": ordered,
                "sha256_ordered": [sf(p) for p in ordered],
                "src_groups": {p: list(D.frame_key(p)) for p in ordered}}
    sup, qry = {}, {}
    for g, b in BUDGETS.items():
        for combo in COMBOS:
            a = D.ASSET_OF[combo]
            for s2 in SEEDS:
                with open(D.SPLITS + "/kshot_k%d_seed%d.json" % (b["k"], s2), encoding="utf-8") as f:
                    sp = json.load(f)
                sup["%s_%s_k%d_seed%d" % (combo.replace("/", "__"), g, b["k"], s2)] = {
                    "paths": sp["prototype_pools"][combo],
                    "src_groups": sorted({tuple(D.frame_key(p)) for p in sp["prototype_pools"][combo]})}
            with open(D.SPLITS + "/query_k%d.json" % b["k"], encoding="utf-8") as f:
                qq = json.load(f)
            qry["%s_%s" % (combo.replace("/", "__"), g)] = {
                "defect": qq["defect_query"][combo], "normal": qq["normal_query"][a + "/good"],
                "defect_src_groups_n": len({tuple(D.frame_key(p)) for p in qq["defect_query"][combo]})}
    env = {"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)}
    try:
        import anomalib; env["anomalib"] = anomalib.__version__
    except Exception:
        env["anomalib"] = "missing"
    frozen = {"code_commit": os.popen("git rev-parse HEAD").read().strip(),
              "weights": {"path": D.WEIGHTS, "sha256": sf(D.WEIGHTS)},
              "mapper_B": {"path": D.OUT + "/mapper_B.pt", "sha256": sf(D.OUT + "/mapper_B.pt")},
              "cache": {"patches": sf(D.CACHE + "/patches.npy"), "eval_paths": sf(D.CACHE + "/eval_paths.json")},
              "ref_groups": sf(M6.OUT4 + "/ref_groups.json"),
              "env": env,
              "budgets": BUDGETS,
              "disclosure": {"k_semantics": "历史 k-crop 支持预算, 非独立源帧数",
                             "identity": "源身份未知记未知不记零重叠", "queries_g1_g2_differ": True}}
    D.save_json(frozen, OUT + "/FROZEN_INPUTS.json")
    D.save_json({"normal_lists": lists, "fault_supports": sup, "queries": qry}, OUT + "/SUPPORT_LISTS.json")
    print("[b] frozen lists saved", flush=True)

    # ---------- A. 锚点 (原历史预算 10 张) ----------
    anchor = []
    aw = 0.0
    for combo in COMBOS:
        a = D.ASSET_OF[combo]
        nb = M6.build_normal_bank(spec, rg["assets"][a]["groups"]["3001"], dev)
        with open(D.SPLITS + "/kshot_k5_seed0.json", encoding="utf-8") as f:
            sp = json.load(f)
        sup0 = sp["prototype_pools"][combo]
        vecs = embed(spec, torch.from_numpy(np.asarray(patches[[path2idx[p] for p in sup0]]).reshape(-1, M6.CH)).float(), dev)
        fb = M6.p1_fault_bank(vecs, nb, dev)
        _, qd, qn, qd_idx, qn_idx = M6.load_split(combo, a, 5, 0, path2idx)
        A_d, A_n = M6.agg(M6.crop_sims(patches, qd_idx, spec, fb, dev)), M6.agg(M6.crop_sims(patches, qn_idx, spec, fb, dev))
        sN_d, sN_n = M6.crop_sims(patches, qd_idx, spec, nb, dev), M6.crop_sims(patches, qn_idx, spec, nb, dev)
        C_d, C_n = M6.agg(M6.crop_sims(patches, qd_idx, spec, fb, dev) - sN_d), M6.agg(M6.crop_sims(patches, qn_idx, spec, fb, dev) - sN_n)
        arc = json.load(open(M6.OUT6 + "/raw_scores6/glass-insulator__missing-cap__k5_seed0_ref3001.json")) if "glass" in combo else json.load(open(M6.OUT6 + "/raw_scores6/yoke-suspension__rust__k5_seed0_ref3001.json"))
        md = 0.0
        for side, arrs, col in (("defect", [A_d, C_d], None), ("normal", [A_n, C_n], None)):
            old = arc[side]
            for p, v in old.items():
                pass
        # A/B/C 三分数比对 (field 0/1/2)
        B_d = M6.agg(1 - sN_d); B_n = M6.agg(1 - sN_n)
        for side, qpaths, ours in (("defect", qd, [A_d, B_d, C_d]), ("normal", qn, [A_n, B_n, C_n])):
            for col, arr in enumerate(ours):
                old = np.array([arc[side][p][col] for p in qpaths])
                md = max(md, float(np.abs(old - arr).max()))
        aw = max(aw, md)
        anchor.append({"config": combo, "max_abs_diff": round(md, 10)})
        print("[anchor] %s max|diff|=%.2e" % (combo, md), flush=True)
    gate_ok = aw <= TOL
    assert gate_ok, "anchor gate failed"

    # ---------- O. Ours 60 配置 ----------
    ours_auroc = {}
    for g, b in BUDGETS.items():
        k = b["k"]
        for combo in COMBOS:
            a = D.ASSET_OF[combo]
            for j in REF_SEEDS:
                nl = lists["%s_ref%d" % (a, j)]
                nb = M6.build_normal_bank(spec, nl[b["n_sub"]], dev)
                for s2 in SEEDS:
                    sup0 = sup["%s_%s_k%d_seed%d" % (combo.replace("/", "__"), g, k, s2)]["paths"]
                    vecs = embed(spec, torch.from_numpy(np.asarray(patches[[path2idx[p] for p in sup0]]).reshape(-1, M6.CH)).float(), dev)
                    fb = M6.p1_fault_bank(vecs, nb, dev)
                    _, qd, qn, qd_idx, qn_idx = M6.load_split(combo, a, k, s2, path2idx)
                    sF_d, sF_n = M6.crop_sims(patches, qd_idx, spec, fb, dev), M6.crop_sims(patches, qn_idx, spec, fb, dev)
                    sN_d, sN_n = M6.crop_sims(patches, qd_idx, spec, nb, dev), M6.crop_sims(patches, qn_idx, spec, nb, dev)
                    C_d, C_n = M6.agg(sF_d - sN_d), M6.agg(sF_n - sN_n)
                    assert len(C_d) == len(qd) and len(C_n) == len(qn)
                    u, fl = D.auroc_avg_rank(C_d, C_n)
                    assert fl == "ok"
                    cfg = "%s_%s_k%d_seed%d_ref%d" % (combo.replace("/", "__"), g, k, s2, j)
                    ours_auroc[cfg] = round(float(u), 6)
                    D.save_json({"config": {"combo": combo, "budget": g, "k": k, "seed": s2, "ref": j,
                                  "N_size": b["n_sub"], "fault_bank_bytes": int(fb.nbytes), "normal_bank_bytes": int(nb.nbytes)},
                                  "defect": {p: round(float(C_d[t]), 8) for t, p in enumerate(qd)},
                                  "normal": {p: round(float(C_n[t]), 8) for t, p in enumerate(qn)}},
                                OUT + "/raw_scores_ours/%s.json" % cfg)
        print("[ours] %s done" % g, flush=True)

    # ---------- P. PatchCore 20 配置 ----------
    import lightning.pytorch as pl
    import torchvision.transforms.v2 as T2
    from PIL import Image
    from anomalib.models import Patchcore
    from anomalib.pre_processing import PreProcessor
    from anomalib.engine import Engine
    tr = T2.Compose([T2.Resize((320, 320), antialias=True),
                     T2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    pp = PreProcessor(transform=tr)
    class ImgDS(torch.utils.data.Dataset):
        def __init__(self, paths):
            self.paths = paths
        def __len__(self):
            return len(self.paths)
        def __getitem__(self, i):
            return {"image": Image.open(self.paths[i]).convert("RGB"), "image_path": self.paths[i]}
    class DM(pl.LightningDataModule):
        def __init__(self, paths):
            super().__init__(); self.paths = paths
        def train_dataloader(self):
            return torch.utils.data.DataLoader(ImgDS(self.paths), batch_size=32, num_workers=4)
    def pc_run(paths_n, qd, qn):
        model = Patchcore(backbone="wide_resnet50_2", layers=["layer2", "layer3"], pre_trained=True,
                          coreset_sampling_ratio=0.1, num_neighbors=9, pre_processor=pp)
        eng = Engine()
        tb = time.time()
        try:
            eng.fit(model, datamodule=DM(paths_n))
            idiom = "datamodule"
        except Exception as e1:
            eng = Engine()
            model = Patchcore(backbone="wide_resnet50_2", layers=["layer2", "layer3"], pre_trained=True,
                              coreset_sampling_ratio=0.1, num_neighbors=9, pre_processor=pp)
            eng.fit(model, train_dataloaders=torch.utils.data.DataLoader(ImgDS(paths_n), batch_size=32, num_workers=4))
            idiom = "train_dataloaders"
        build_s = round(time.time() - tb, 1)
        eng_ = Engine() if idiom == "datamodule" else eng
        outs = eng_.predict(model, dataloaders=torch.utils.data.DataLoader(ImgDS(qd + qn), batch_size=32, num_workers=4))
        sd, sn = {}, {}
        for it in outs:
            p = it["image_path"] if hasattr(it, "keys") and "image_path" in it else getattr(it, "image_path", None)
            v = it["pred_score"] if hasattr(it, "keys") and "pred_score" in it else getattr(it, "pred_score", None)
            if v is None and hasattr(it, "pred_score"):
                v = it.pred_score
            v = float(v)
            assert np.isfinite(v)
            (sd if p in set(qd) else sn)[p] = round(v, 8)
        mb = getattr(model, "memory_bank", None)
        shape = tuple(mb.memory_bank.shape) if hasattr(mb, "memory_bank") else "n/a"
        return sd, sn, {"idiom": idiom, "build_s": build_s, "memory_bank_shape": list(shape) if shape != "n/a" else "n/a"}
    pc_auroc = {}
    pc_meta = {}
    for g, b in BUDGETS.items():
        k = b["k"]
        for combo in COMBOS:
            a = D.ASSET_OF[combo]
            for j in REF_SEEDS:
                nl = lists["%s_ref%d" % (a, j)]
                q = qry["%s_%s" % (combo.replace("/", "__"), g)]
                sd, sn, meta = pc_run(nl[b["pc_n"]], q["defect"], q["normal"])
                assert set(sd.keys()) == set(q["defect"]) and set(sn.keys()) == set(q["normal"]), "PC path set mismatch"
                u, fl = D.auroc_avg_rank(np.array(list(sd.values())), np.array(list(sn.values())))
                assert fl == "ok"
                cfg = "%s_%s_ref%d" % (combo.replace("/", "__"), g, j)
                pc_auroc[cfg] = round(float(u), 6)
                pc_meta[cfg] = meta
                D.save_json({"defect": sd, "normal": sn, "meta": meta}, OUT + "/raw_scores_patchcore/%s.json" % cfg)
        print("[pc] %s done" % g, flush=True)

    # ---------- S. 汇总 ----------
    per_ref = {}
    for g in BUDGETS:
        for combo in COMBOS:
            ckc = combo.replace("/", "__")
            for j in REF_SEEDS:
                ds = []
                pc = pc_auroc["%s_%s_ref%d" % (ckc, g, j)]
                for s2 in SEEDS:
                    k = BUDGETS[g]["k"]
                    o = ours_auroc["%s_%s_k%d_seed%d_ref%d" % (ckc, g, k, s2, j)]
                    ds.append(round(o - pc, 6))
                per_ref.setdefault("%s_%s" % (ckc, g), {})["ref%d" % j] = {"per_seed": ds, "mean": round(float(np.mean(ds)), 6), "pc": pc}
    paired = {}
    for key, pr in per_ref.items():
        vals = [v["mean"] for v in pr.values()]
        paired[key] = {"mean": round(float(np.mean(vals)), 6), "range": [round(min(vals), 6), round(max(vals), 6)],
                       "n_pos_groups": int(sum(v > 0 for v in vals)), "n_groups": 5, "per_ref": pr}
    summary = {"method_configs_completed": len(ours_auroc) + len(pc_auroc),
               "ours_configs": len(ours_auroc), "patchcore_configs": len(pc_auroc),
               "anchor_repro_passed": 1 if gate_ok else 0, "anchor": anchor, "anchor_worst": round(aw, 10),
               "ours_auroc": ours_auroc, "patchcore_auroc": pc_auroc, "patchcore_meta": pc_meta,
               "paired": paired, "correctness": {"mapper_hash_before": mh0, "mapper_hash_after": M6.state_hash(head),
               "b_reuse": "PatchCore 不依赖故障种子, 同查询复用", "exploratory": True},
               "cost": {"wall_time_s": round(time.time() - t0, 1), "peak_vram_bytes": int(torch.cuda.max_memory_allocated())}}
    D.save_json(summary, OUT + "/bench_eval.json")
    D.save_json({"anchor": anchor, "anchor_worst": round(aw, 10), "correctness": summary["correctness"], "exploratory": True},
                OUT + "/repro_and_correctness.json")
    print("[b] bench_eval.json saved; wall=%.1fs" % (time.time() - t0), flush=True)
    print("=== 全部完成 ===", flush=True)


if __name__ == "__main__":
    main()