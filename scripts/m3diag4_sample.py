#!/usr/bin/env python3
"""M3DIAG4 抽样冻结: 5 个新参照组 (seed 3001-3005), 名单+哈希+组间重叠矩阵落盘, 先于任何查询评测."""
import glob, hashlib, json, os, re
import numpy as np

SPL = "/root/auto-exp/data/splits/insplad-fd"
ROOT = "/root/autodl-tmp/data/insplad/defect_supervised"
ASSETS = ["glass-insulator", "yoke-suspension"]
SEEDS = [3001, 3002, 3003, 3004, 3005]
OUT4 = "/root/auto-exp/outputs/M3diag4"

def fk(p):
    m = re.match(r"^(.*)_DJI_([0-9]+)(_.*)?$", os.path.splitext(os.path.basename(p))[0])
    return (m.group(1), m.group(2)) if m else (os.path.splitext(os.path.basename(p))[0], None)

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def main():
    evalp = set()
    for k in (1, 5, 10):
        q = json.load(open(SPL + "/query_k%d.json" % k))
        for dd in (q["defect_query"], q["normal_query"]):
            for ps in dd.values():
                evalp.update(ps)
        for s2 in (0, 1, 2):
            sp = json.load(open(SPL + "/kshot_k%d_seed%d.json" % (k, s2)))
            for ps in sp["prototype_pools"].values():
                evalp.update(ps)
    evalf = {fk(p) for p in evalp}
    prev = json.load(open("/root/auto-exp/outputs/M3diag3/normal_ref.json"))
    out = {"new_seeds": SEEDS, "algorithm": "与 M3DIAG3 相同: 严格隔离池排序后 rng(seed) 无放回抽 10",
           "assets": {}}
    for a in ASSETS:
        fs = sorted(f for f in glob.glob(ROOT + "/" + a + "/*/good/*") if os.path.isfile(f))
        iso = [f for f in fs if f not in evalp and fk(f) not in evalf]
        assert not (set(iso) & evalp) and not ({fk(f) for f in iso} & evalf)
        groups = {"3000": prev["assets"][a]["selected"]}
        for sd in SEEDS:
            rng = np.random.default_rng(sd)
            groups[str(sd)] = sorted(np.array(iso)[rng.choice(len(iso), 10, replace=False)].tolist())
        names = sorted(groups)
        ov = {"file": {}, "frame": {}}
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                A, B = names[i], names[j]
                fa = {fk(f) for f in groups[A]}
                fb = {fk(f) for f in groups[B]}
                ov["file"]["%s-%s" % (A, B)] = len(set(groups[A]) & set(groups[B]))
                ov["frame"]["%s-%s" % (A, B)] = len(fa & fb)
        rec = {"isolated_pool": len(iso), "groups": {}, "group_meta": {},
               "overlap_matrix": ov,
               "disclosure": "glass 隔离池仅 %d 张, 组间重叠不可避免, 多组不可视为独立数据集" % len(iso)}
        for g, lst in groups.items():
            frames = [fk(f) for f in lst]
            rec["groups"][g] = lst
            rec["group_meta"][g] = {"n": len(lst), "independent_source_frames": len(set(frames)),
                                    "sha256": {os.path.basename(f): sha256(f) for f in lst}}
        out["assets"][a] = rec
        print("[sample] %s: pool=%d, groups=%s, 3000-3001 file_ovl=%d frame_ovl=%d" % (
            a[:12], len(iso), names, ov["file"]["3000-3001"], ov["frame"]["3000-3001"]))
    json.dump(out, open(OUT4 + "/ref_groups.json", "w"), ensure_ascii=False, indent=1)
    print("[done] ref_groups.json frozen (lists + hashes + overlap matrix)")

if __name__ == "__main__":
    main()