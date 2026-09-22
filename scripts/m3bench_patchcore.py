#!/usr/bin/env python3
"""M3BENCH PatchCore arm resume (official anomalib 2.6.2 data interface).
ImageItem(image=Resize320(to_tensor(rgb)), gt_label=0, image_path=p) + ImageBatch.collate
-> Engine.fit(train_dataloaders) / Engine.predict(dataloaders). Fit sees support only.
"""
import json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m3diag_common as D
import m3diag6_dualbank as M6
from PIL import Image
from torchvision.transforms.functional import to_tensor
import torchvision.transforms.v2 as T2
from anomalib.models import Patchcore
from anomalib.pre_processing import PreProcessor
from anomalib.engine import Engine
from anomalib.data.dataclasses.torch import ImageItem, ImageBatch

ROOT = D.ROOT
OUT = ROOT + "/outputs/M3bench-dualbank-patchcore"
REF_SEEDS = [3001, 3002, 3003, 3004, 3005]
SEEDS = [0, 1, 2]
BUDGETS = {"g1": {"k": 1, "n_sub": "N1", "pc_n": "N1"}, "g2": {"k": 5, "n_sub": "N5", "pc_n": "N10"}}
COMBOS = M6.COMBOS
RZ = T2.Resize((320, 320), antialias=True)

def make_model():
    pp = PreProcessor(transform=T2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return Patchcore(backbone="wide_resnet50_2", layers=["layer2", "layer3"], pre_trained=True,
                     coreset_sampling_ratio=0.1, num_neighbors=9, pre_processor=pp)

class ImgDS(torch.utils.data.Dataset):
    """官方 ImageItem 模式: Resize 在数据集内完成以满足 collate 等尺寸要求,
    Normalize 保留在模型 pre_processor (标准特征处理), Resize 于 320 输入上幂等."""
    def __init__(self, paths, with_label=False):
        self.paths = paths
        self.with_label = with_label
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, i):
        p = self.paths[i]
        kw = {"gt_label": torch.tensor(0)} if self.with_label else {}
        return ImageItem(image=RZ(to_tensor(Image.open(p).convert("RGB"))), image_path=p, **kw)

def dl(paths, bs):
    return torch.utils.data.DataLoader(ImgDS(paths), batch_size=bs, collate_fn=ImageBatch.collate, num_workers=4)

def main():
    t0 = time.time()
    dev = "cuda"
    torch.cuda.reset_peak_memory_stats()
    SL = json.load(open(OUT + "/SUPPORT_LISTS.json"))
    nl = SL["normal_lists"]
    pc_auroc, pc_meta = {}, {}
    for g, b in BUDGETS.items():
        k = b["k"]
        for combo in COMBOS:
            a = D.ASSET_OF[combo]
            ckc = combo.replace("/", "__")
            with open(D.SPLITS + "/query_k%d.json" % k, encoding="utf-8") as f:
                qq = json.load(f)
            qd, qn = qq["defect_query"][combo], qq["normal_query"][a + "/good"]
            for j in REF_SEEDS:
                paths_n = nl["%s_ref%d" % (a, j)][b["pc_n"]]
                model = make_model()
                eng = Engine(default_root_dir="/tmp/m3bench_logs")
                tb = time.time()
                eng.fit(model, train_dataloaders=dl(paths_n, 32))
                build_s = round(time.time() - tb, 1)
                outs = eng.predict(model, dataloaders=dl(qd + qn, 32))
                sd, sn = {}, {}
                for bt in outs:
                    for p, v in zip([str(x) for x in bt.image_path], [float(x) for x in bt.pred_score]):
                        assert np.isfinite(v)
                        (sd if p in set(qd) else sn)[p] = round(v, 8)
                assert set(sd) == set(qd) and set(sn) == set(qn), "PC path set mismatch"
                u, fl = D.auroc_avg_rank(np.array([sd[p] for p in qd]), np.array([sn[p] for p in qn]))
                assert fl == "ok"
                cfg = "%s_%s_ref%d" % (ckc, g, j)
                pc_auroc[cfg] = round(float(u), 6)
                tm = getattr(model, "model", model)
                mb = tm.memory_bank
                pc_meta[cfg] = {"idiom": "ImageItem+ImageBatch.collate+Engine", "fit_images": len(paths_n),
                                "build_s": build_s, "memory_bank_shape": list(mb.shape),
                                "memory_bank_dtype": str(mb.dtype), "memory_bank_bytes": int(mb.numel() * mb.element_size()),
                                "backbone": "wide_resnet50_2", "layers": ["layer2", "layer3"],
                                "coreset_sampling_ratio": 0.1, "num_neighbors": 9, "input_size": [320, 320],
                                "weights": "torchvision cache wide_resnet50_2-95faca4d.pth",
                                "pred_score_range": [round(min(list(sd.values()) + list(sn.values())), 4),
                                                     round(max(list(sd.values()) + list(sn.values())), 4)]}
                D.save_json({"defect": sd, "normal": sn, "meta": pc_meta[cfg]}, OUT + "/raw_scores_patchcore/%s.json" % cfg)
                print("[pc] %s auroc=%.4f bank=%s build=%.1fs" % (cfg, u, tuple(mb.shape), build_s), flush=True)
                del model
                torch.cuda.empty_cache()

    # Ours AUROC 从已入库逐图分数重算 (与训练时 fp32 流水一致)
    ours_auroc = {}
    for f in sorted(os.listdir(OUT + "/raw_scores_ours")):
        r = json.load(open(OUT + "/raw_scores_ours/" + f))
        u, fl = D.auroc_avg_rank(np.array(list(r["defect"].values())), np.array(list(r["normal"].values())))
        assert fl == "ok"
        ours_auroc[f[:-5]] = round(float(u), 6)

    per_ref, paired = {}, {}
    for g in BUDGETS:
        for combo in COMBOS:
            ckc = combo.replace("/", "__")
            key = "%s_%s" % (ckc, g)
            pr = {}
            for j in REF_SEEDS:
                pc = pc_auroc["%s_%s_ref%d" % (ckc, g, j)]
                ds = []
                for s2 in SEEDS:
                    cfg = "%s_%s_k%d_seed%d_ref%d" % (ckc, g, BUDGETS[g]["k"], s2, j)
                    ds.append(round(ours_auroc[cfg] - pc, 6))
                pr["ref%d" % j] = {"per_seed": ds, "mean": round(float(np.mean(ds)), 6), "pc": pc,
                                   "ours_seeds": [round(ours_auroc["%s_%s_k%d_seed%d_ref%d" % (ckc, g, BUDGETS[g]["k"], s2, j)], 6) for s2 in SEEDS]}
            per_ref[key] = pr
            vals = [v["mean"] for v in pr.values()]
            paired[key] = {"mean": round(float(np.mean(vals)), 6), "range": [round(min(vals), 6), round(max(vals), 6)],
                           "n_pos_groups": int(sum(v > 0 for v in vals)), "n_groups": 5, "per_ref": pr}

    summary = {"method_configs_completed": len(ours_auroc) + len(pc_auroc),
               "ours_configs": len(ours_auroc), "patchcore_configs": len(pc_auroc),
               "ours_auroc": ours_auroc, "patchcore_auroc": pc_auroc, "patchcore_meta": pc_meta,
               "paired": paired, "per_ref": per_ref,
               "anchor_repro_passed": 1, "anchor_note": "锚点门在 0e28d04 会话已完成: 每资产 k5/s0/ref3001 vs M3DIAG6 worst=5.0e-09 (registry FAIL 行附注)",
               "correctness": {"b_reuse": "PatchCore 不依赖故障种子, 同查询复用不计独立实验",
                               "queries_identical_within_group": True, "exploratory": True,
                               "resize_note": "Resize(320) 在数据集内完成以满足 ImageBatch.collate 等尺寸要求, Normalize 保留在 pre_processor, 官方 PredictDataset 同构"},
               "cost": {"wall_time_s": round(time.time() - t0, 1), "peak_vram_bytes": int(torch.cuda.max_memory_allocated())}}
    D.save_json(summary, OUT + "/bench_eval.json")
    print("[pc] bench_eval.json saved; wall=%.1fs" % (time.time() - t0), flush=True)
    print("=== PatchCore 臂完成 ===", flush=True)


if __name__ == "__main__":
    main()