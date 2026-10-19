#!/usr/bin/env python3
"""CPLID-T0T1T2: freeze splits, support lists, T2 exposure design — BEFORE any training/scoring.
Pre-registered rules (all deterministic):
  - split: group-level, PCG64(20261019), groups sorted by representative path,
    n_tr=round(.7n) n_val=round(.1n) n_test=rest
  - support lists: per ref r, PCG64(r) over normal_train groups (sorted), 10 distinct groups,
    per group 1 image (sorted list, rng choice), per image 1 valid insulator box
    (sorted by (x1,y1,x2,y2), rng choice); crop sha = PNG-encoded bytes
  - T2: PCG64(20261020) permutation of eligible normal_test groups (excludes shared-source
    groups and any bank-list source group), take first ceil(.5*n) as exposed (complete groups);
    removals from normal_train: PCG64(20261021) permutation of eligible groups (not in bank
    lists, not exposed-source), whole groups first, then partial group to reach exact image count.
"""
import hashlib, io, json, os
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(os.environ.get("CPLID_ROOT", "/root/data_cplid_local"))
OUT = Path("/root/autodl-tmp/version10-12/CPLID-T0T1T2-detector-top8-v1")
REFS = [3001, 3002, 3003, 3004, 3005]
SPLIT_SEED = 20261019
T2_SEED = 20261020
RM_SEED = 20261021


def wj(p, obj):
    p = Path(p); tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1) + "\n"); tmp.replace(p)


def crop_png_sha(img_path, box):
    im = Image.open(img_path).convert("RGB")
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    c = im.crop((x1, y1, x2, y2))
    buf = io.BytesIO(); c.save(buf, "PNG")
    return hashlib.sha256(buf.getvalue()).hexdigest(), (x2 - x1, y2 - y1)


def main():
    man = [json.loads(l) for l in (OUT / "DATA_MANIFEST.jsonl").read_text().splitlines() if '"side"' in l]
    normal = sorted([m for m in man if m["side"] == "normal"], key=lambda m: m["path"])
    defective = [m for m in man if m["side"] == "defective"]
    audit = json.load(open(OUT / "AUDIT_SUMMARY.json"))
    assert audit["audit_ok"] and audit["observed"]["normal_images"] == 600
    # groups among normals
    g2imgs = {}
    for m in normal:
        g2imgs.setdefault(m["group_id"], []).append(m)
    gids = sorted(g2imgs)  # group_id assignment was by sorted representative, deterministic
    excluded = set(audit["shared_normal_paths"])  # confirmed shared-source images (none expected)
    n = len(gids)
    rng = np.random.Generator(np.random.PCG64(SPLIT_SEED))
    perm = rng.permutation(n)
    n_tr = int(round(0.7 * n)); n_val = int(round(0.1 * n))
    tr_g = [gids[i] for i in perm[:n_tr]]
    va_g = [gids[i] for i in perm[n_tr:n_tr + n_val]]
    te_g = [gids[i] for i in perm[n_tr + n_val:]]
    split = dict(seed=SPLIT_SEED, rule="group-level PCG64(20261019), groups sorted by representative",
                  n_groups=n, train_groups=tr_g, val_groups=va_g, test_groups=te_g,
                  train_images=sorted(p for g in tr_g for p in [m["path"] for m in g2imgs[g]]),
                  val_images=sorted(p for g in va_g for p in [m["path"] for m in g2imgs[g]]),
                  test_images=sorted(p for g in te_g for p in [m["path"] for m in g2imgs[g]]),
                  excluded_shared_source_images=sorted(excluded))
    assert not (set(split["train_images"]) & set(split["val_images"]))
    assert not (set(split["train_images"]) & set(split["test_images"]))
    wj(OUT / "SPLIT_MANIFEST.json", split)
    # ---------- support lists ----------
    tr_img2g = {p: g for g in tr_g for p in [m["path"] for m in g2imgs[g]]}
    img_boxes = {m["path"]: m["ins_boxes"] for m in normal}
    img_sha = {m["path"]: m["sha256"] for m in normal}
    avail = sorted(tr_g)
    lists = {}
    used_groups = set()
    for r in REFS:
        g = np.random.Generator(np.random.PCG64(r))
        pick = [avail[i] for i in sorted(g.choice(len(avail), size=10, replace=False))]
        assert len(set(pick)) == 10
        entries = []
        for gid in pick:
            imgs = sorted(m["path"] for m in g2imgs[gid])
            ip = imgs[int(g.integers(len(imgs)))]
            boxes = sorted(img_boxes[ip])
            bx = boxes[int(g.integers(len(boxes)))]
            csha, csize = crop_png_sha(ROOT / "Normal_Insulators/images" / Path(ip).name, bx)
            entries.append(dict(source_group=gid, image_path=ip, image_sha256=img_sha[ip],
                                 box=[float(v) for v in bx], crop_png_sha256=csha, crop_size=csize))
        lists["ref%d" % r] = dict(ref=r, seed=r, entries=entries,
                                   source_groups=[e["source_group"] for e in entries])
        used_groups.update(pick)
    wj(OUT / "SUPPORT_LISTS.json", dict(rule="10 distinct normal_train source groups per ref, PCG64(ref), "
                                         "no appearance/performance filtering", refs=REFS, lists=lists,
                                         overlap_matrix_note="pairwise shared group counts below",
                                         overlap_matrix={
                                             "ref%d-ref%d" % (a, b): len(set(lists["ref%d" % a]["source_groups"]) &
                                                                          set(lists["ref%d" % b]["source_groups"]))
                                             for a in REFS for b in REFS if a < b}))
    # ---------- T2 exposure ----------
    te_img2g = {p: g for g in te_g for p in [m["path"] for m in g2imgs[g]]}
    eligible_te = sorted(g for g in te_g if g not in used_groups and
                          not any(x in excluded for x in [m["path"] for m in g2imgs[g]]))
    g2 = np.random.Generator(np.random.PCG64(T2_SEED))
    order = [eligible_te[i] for i in g2.permutation(len(eligible_te))]
    k = int(np.ceil(0.5 * len(eligible_te)))
    exposed = sorted(order[:k])
    exposed_imgs = sorted(p for g in exposed for p in [m["path"] for m in g2imgs[g]])
    # removals: eligible train groups = not used by bank lists, not excluded
    eligible_tr = sorted(g for g in tr_g if g not in used_groups and
                          not any(x in excluded for x in [m["path"] for m in g2imgs[g]]))
    g3 = np.random.Generator(np.random.PCG64(RM_SEED))
    rm_order = [eligible_tr[i] for i in g3.permutation(len(eligible_tr))]
    removed, need = [], len(exposed_imgs)
    for gid in rm_order:
        if need <= 0:
            break
        imgs = sorted(m["path"] for m in g2imgs[gid])
        take = imgs if len(imgs) <= need else imgs[:need]
        removed.extend(take); need -= len(take)
    assert len(removed) == len(exposed_imgs), (len(removed), len(exposed_imgs))
    t2_train = sorted((set(split["train_images"]) - set(removed)) | set(exposed_imgs))
    assert len(t2_train) == len(split["train_images"]), (len(t2_train), len(split["train_images"]))
    bank_source_images = {e["image_path"] for l in lists.values() for e in l["entries"]}
    for r in removed:
        assert tr_img2g[r] not in used_groups, ("removed from bank source group", r)
    t2m = dict(seed=T2_SEED, rm_seed=RM_SEED, target_fraction=0.5,
                eligible_test_groups=len(eligible_te), exposed_groups=len(exposed),
                exposed_images=exposed_imgs, exposed_image_count=len(exposed_imgs),
                exposed_fraction_of_test=len(exposed_imgs) / max(len(split["test_images"]), 1),
                removed_train_images=removed, removed_image_count=len(removed),
                bank_source_groups=sorted(used_groups),
                t2_train_images=t2_train, t1_train_images=split["train_images"],
                equal_len_asserted=len(t2_train) == len(split["train_images"]),
                val_identical=True, test_identical=True)
    wj(OUT / "T2_EXPOSURE_MANIFEST.json", t2m)
    wj(OUT / "T2_TRAIN.json", dict(images=t2_train, boxes_from="Normal_Insulators/labels VOC xml",
                                    class_count=1, note="exposed normal_test images included with insulator boxes"))
    wj(OUT / "T2_EXPOSED_TEST.json", dict(images=exposed_imgs, groups=exposed))
    wj(OUT / "T2_REMOVED_TRAIN.json", dict(images=removed))
    wj(OUT / "T1_TRAIN.json", dict(images=split["train_images"]))
    # ---------- frozen config ----------
    import ultralytics, torch
    wj(OUT / "FROZEN_CONFIG.json", dict(
        imgsz=640, epochs=100, optimizer="AdamW", lr0=1e-3, lrf=0.01, weight_decay=5e-4,
        warmup_epochs=3, seed=0, patience=0, amp=False, tf32=False,
        mosaic=0.0, mixup=0.0, copy_paste=0.0, fliplr=0.5, flipud=0.0, degrees=0.0,
        translate=0.1, scale=0.2, hsv_h=0.015, hsv_s=0.4, hsv_v=0.4,
        classes=1, batch_candidates=[32, 64],
        init_weights="/root/autodl-tmp/third_party/yolov8n.pt",
        init_sha256="f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36",
        ultralytics=ultralytics.__version__, torch=torch.__version__,
        checkpoint_rule="final epoch (100) EMA weights for all arms; no AUROC-based selection",
        eval_preprocessing="RGB Resize(320,320) ToTensor (historical m3_common.make_transform)",
        feature="module4 P3 [B,64,40,40], channel-L2 patches",
        coreset="official anomalib KCenterGreedy ratio 0.1 (16000->1600)",
        scoring="Euclidean NN patch distances, top8 mean, higher=more anomalous",
        audit_summary_sha256=hashlib.sha256((OUT / "AUDIT_SUMMARY.json").read_bytes()).hexdigest()))
    print("frozen: split %d/%d/%d groups, exposed %d imgs (%.3f of test), removed %d train imgs" %
           (len(tr_g), len(va_g), len(te_g), len(exposed_imgs), t2m["exposed_fraction_of_test"], len(removed)))
    print("bank group overlap matrix:", json.load(open(OUT / "SUPPORT_LISTS.json"))["overlap_matrix"])


if __name__ == "__main__":
    main()
