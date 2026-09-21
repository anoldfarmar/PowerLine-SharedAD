#!/usr/bin/env python3
"""M3DIAG 映射器训练: --variant B (local-supcon) / C (local-correspondence).
B/C 同代码路径: 相同框清单/采样器序列(同种子)/优化器/步数(30x176=5280)/批量(128框).
区别仅在损失: B = SupCon(框级资产标签广播至16 patch, 作用于L2归一化后);
C = VICReg 式局部对应 (双视图对应位一致性+方差+协方差, 作用于归一化前).
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
           "sampler": "per-step per-class balanced, same seed sequence for B/C"}

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
            z = head(xb.view(-1, CH))
            if args.variant == "B":
                zn = torch.nn.functional.normalize(z, dim=1)
                yb = y[idx].repeat_interleave(PP).to(dev)
                loss = D.supcon_loss(zn, yb, cfg["supcon_temperature"])
                ep_loss.append(float(loss))
                comp["supcon"] += float(loss)
            else:
                Xc = np.load(D.CACHE + "/trainC_patches.npy", mmap_mode="r")[idx.numpy()]
                Xc = torch.from_numpy(np.asarray(Xc)).float()
                z1 = head(Xc[:, 0].reshape(-1, CH).to(dev))
                z2 = head(Xc[:, 1].reshape(-1, CH).to(dev))
                total, rl, sl, cl = D.vicreg_losses(z1, z2)
                loss = total
                ep_loss.append(float(total))
                comp["inv"] += float(rl); comp["var"] += float(sl); comp["cov"] += float(cl)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)
            assert torch.isfinite(loss), "loss NaN/Inf at epoch %d" % ep
            opt.step()
        row = {"epoch": ep, "loss": float(np.mean(ep_loss)),
               "components_mean": {k: v / steps_per_epoch for k, v in comp.items()}}
        log["epochs"].append(row)
        print("[train %s] epoch %d loss %.5f" % (args.variant, ep, row["loss"]))

    log["actual_steps"] = cfg["epochs"] * steps_per_epoch
    log["coverage"] = {"unique_boxes_seen": len(seen_boxes), "total_boxes": M,
                       "passes": cfg["epochs"], "class_sample_counts": dict(cls_counts)}
    os.makedirs(D.OUT, exist_ok=True)
    torch.save({"state_dict": head.state_dict(), "variant": args.variant, "cfg": cfg},
               D.OUT + "/mapper_%s.pt" % args.variant)
    D.save_json(log, D.OUT + "/metric_train_log_%s.json" % args.variant)
    print("[done] mapper_%s.pt actual_steps=%d unique_boxes=%d" %
          (args.variant, log["actual_steps"], len(seen_boxes)))

if __name__ == "__main__":
    main()