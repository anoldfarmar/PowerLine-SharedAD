#!/usr/bin/env python3
"""M3DIAG2 阶段三: mapper_C 的 BatchNorm running 统计重校准 (简报五).
预注册单一方案: 校准数据 = trainB (未增强, 与推理一致) 全量 22,635 框按索引序;
批 = 126 框 (与训练实际批量一致), 179 个整批 (尾批丢弃); BN momentum=None (累计均值);
可学习参数全部冻结; 原权重不覆盖, 存 mapper_C_bnrecal.pt.
不使用查询数据选择任何校准选项; 这是 BN 敏感性诊断, 不是干净重训.
"""
import copy, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m3diag_common as D

OUT2 = D.ROOT + "/outputs/M3diag2"
os.makedirs(OUT2, exist_ok=True)

BB = 126

def bn_modules(head):
    return [(n, m) for n, m in head.named_modules()
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]

def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    assert dev == "cuda", "需要 GPU"
    ck = torch.load(D.OUT + "/mapper_C.pt", map_location=dev)
    head = D.mapper_arch().to(dev)
    head.load_state_dict(ck["state_dict"])
    before = [{"layer": n, "rm_std": round(float(m.running_mean.std()), 6),
               "rv_mean": round(float(m.running_var.mean()), 6)}
              for n, m in bn_modules(head)]
    for p in head.parameters():
        p.requires_grad_(False)
    for _, m in bn_modules(head):
        m.reset_running_stats()
        m.momentum = None
    head.train()
    Xb = np.load(D.CACHE + "/trainB_patches.npy")
    M = Xb.shape[0]
    nb = M // BB
    with torch.no_grad():
        for i in range(nb):
            xb = torch.from_numpy(Xb[i * BB:(i + 1) * BB]).float().view(-1, 64).to(dev)
            head(xb)
            if i % 40 == 0:
                print("[recal] batch %d/%d" % (i, nb))
    head.eval()
    after = [{"layer": n, "rm_std": round(float(m.running_mean.std()), 6),
              "rv_mean": round(float(m.running_var.mean()), 6),
              "num_batches_tracked": int(m.num_batches_tracked)}
             for n, m in bn_modules(head)]
    torch.save({"state_dict": head.state_dict(), "variant": "C2",
                "source": "mapper_C.pt (dc0b9cdb...)",
                "recal": {"data": "trainB non-augmented, index order", "batch_boxes": BB,
                          "batches": nb, "momentum": None, "mode": "cumulative moving average",
                          "frozen_params": True}},
               OUT2 + "/mapper_C_bnrecal.pt")
    D.save_json({"before": before, "after": after, "batches": nb,
                 "note": "BN 敏感性诊断; 单一预注册方案; 不按查询成绩挑选"},
                OUT2 + "/bn_recal_config.json")
    print("[done] mapper_C_bnrecal.pt saved (%d batches)" % nb)

if __name__ == "__main__":
    main()