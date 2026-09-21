#!/usr/bin/env python3
"""M3DIAG2 修正版映射器训练 (对应简报三.A). 本轮不运行, 仅供合同授权后的干净重训.
唯一改动 vs scripts/m3diag_train.py: 原始特征前向从分支前移入 B 分支,
使 C 每步只有两次增强视图前向, BN running 统计不再混入 70% 增强分布.
其余 (采样器/损失/预算/裁剪/批量126框) 与 m3diag_train.py 逐字一致, 保证"一次修正、其余不变".
注意: 修复方式是移动前向, 不是 no_grad 包裹 (no_grad 不阻止 train 模式 BN running 统计更新).
"""
import argparse, collections, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m3diag_common as D

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["B", "C"], required=True)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = D.TRAIN_CFG
    seed = cfg["mapper_train_seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    Xb = torch.from_numpy(np.load(D.CACHE + "/trainB_patches.npy")).float()
    lab = json.load(open(D.CACHE + "/trainB_labels.json"))["labels"]
    y = torch.tensor(lab)
    M, PP, CH = Xb.shape
    head = D.mapper_arch().to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=cfg["lr"],
                            weight_decay=cfg["weight_decay"])

    by_cls = collections.defaultdict(list)
    for i, c in enumerate(lab):
        by_cls[int(c)].append(i)
    n_cls = len(by_cls)
    BB = cfg["batch_boxes"]
    per_cls = max(1, BB // n_cls)
    steps_per_epoch = M // BB
    g = torch.Generator().manual_seed(seed)

    seen_boxes = set()
    cls_counts = collections.Counter()
    log = {"variant": args.variant, "cfg": cfg, "train_boxes": M,
           "steps_per_epoch": steps_per_epoch, "epochs": [],
           "sampler": "per-step per-class balanced, same seed sequence for B/C",
           "fix": "raw-feature forward moved into B branch (BN stats: B/C each see only their own loss inputs)"}

    for ep in range(cfg["epochs"]):
        perm = {c: torch.randperm(len(v), generator=g).tolist() for c, v in by_cls.items()}
        ptr = {c: 0 for c in by_cls}
        ep_loss = []
        comp = collections.defaultdict(float)
        for _ in range(steps_per_epoch):
            idx = []
            for c in sorted(by_cls):
                take = []
                while len(take) < per_cls:
                    if ptr[c] >= len(perm[c]):
                        perm[c] = torch.randperm(len(by_cls[c]), generator=g).tolist()
                        ptr[c] = 0
                    need = per_cls - len(take)
                    seg = perm[c][ptr[c]:ptr[c] + need]
                    take += [by_cls[c][k] for k in seg]
                    ptr[c] += len(seg)
                idx += take[:per_cls]
            idx = torch.tensor(idx[:BB])
            seen_boxes.update(idx.tolist())
            cls_counts.update([lab[i] for i in idx.tolist()])
            xb = Xb[idx].to(dev)
            if args.variant == "B":
                z = head(xb.view(-1, CH))
                zn = torch.nn.functional.normalize(z, dim=1)
                yb = y[idx].repeat_interleave(PP).to(dev)
                loss = D.supcon_loss(zn, yb, cfg["supcon_temperature"])
                ep_loss.append(float(loss.detach()))
                comp["supcon"] += float(loss.detach())
            else:
                Xc = np.load(D.CACHE + "/trainC_patches.npy", mmap_mode="r")[idx.numpy()]
                Xc = torch.from_numpy(np.asarray(Xc)).float()
                z1 = head(Xc[:, 0].reshape(-1, CH).to(dev))
                z2 = head(Xc[:, 1].reshape(-1, CH).to(dev))
                total, rl, sl, cl = D.vicreg_losses(z1, z2)
                loss = total
                ep_loss.append(float(total.detach()))
                comp["inv"] += float(rl.detach()); comp["var"] += float(sl.detach()); comp["cov"] += float(cl.detach())
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)
            assert torch.isfinite(loss), "loss NaN/Inf at epoch %d" % ep
            opt.step()
        row = {"epoch": ep, "loss": float(np.mean(ep_loss)),
               "components_mean": {k: v / steps_per_epoch for k, v in comp.items()}}
        log["epochs"].append(row)
        print("[train2 %s] epoch %d loss %.5f" % (args.variant, ep, row["loss"]))

    log["actual_steps"] = cfg["epochs"] * steps_per_epoch
    log["coverage"] = {"unique_boxes_seen": len(seen_boxes), "total_boxes": M,
                       "passes": cfg["epochs"], "class_sample_counts": dict(cls_counts),
                       "actual_batch_boxes": int(n_cls * per_cls),
                       "actual_batch_patches": int(n_cls * per_cls * PP)}
    out = D.OUT + "/mapper_%s_fixed.pt" % args.variant
    torch.save({"state_dict": head.state_dict(), "variant": args.variant, "cfg": cfg,
                "parent_code_sha256": "ed51bd5df3bb1605d689826b556f2d7033b1db11546b096c2cd72f468989d7b7"}, out)
    D.save_json(log, D.OUT + "/metric_train_log_%s_fixed.json" % args.variant)
    print("[done] %s saved" % out)

if __name__ == "__main__":
    main()