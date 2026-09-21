#!/usr/bin/env python3
"""M3DIAG3 数据审计: 正常参照选取与隔离核验 (简报五).
候选 = 严格隔离池 (路径+源帧双排除, 对全部冻结评测池); 排序后 seed 3000 抽 10 张/资产.
输出 outputs/M3diag3/normal_ref.json + data_audit.json.
"""
import glob, hashlib, json, os, re, sys
import numpy as np

SPL = "/root/auto-exp/data/splits/insplad-fd"
ROOT = "/root/autodl-tmp/data/insplad/defect_supervised"
ASSETS = ["glass-insulator", "yoke-suspension"]
GAPS = {"lightning-rod-suspension": 1, "polymer-insulator-upper-shackle": 2, "vari-grip": 0}
DRAW_SEED = 3000

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
        for s in (0, 1, 2):
            sp = json.load(open(SPL + "/kshot_k%d_seed%d.json" % (k, s)))
            for ps in sp["prototype_pools"].values():
                evalp.update(ps)
    evalf = {fk(p) for p in evalp}
    out = {"draw_seed": DRAW_SEED, "frame_key_rule": "(拍摄场次前缀, DJI序号) 由文件名解析, 包内可追溯;",
           "cross_package_note": "det 包(数字飞行前缀)与 FD 裁剪(日期前缀)之间无法按文件名建立源帧映射,",
           "gap_assets": GAPS, "assets": {}}
    for a in ASSETS:
        fs = sorted(f for f in glob.glob(ROOT + "/" + a + "/*/good/*") if os.path.isfile(f))
        good = all("/good/" in f for f in fs)
        iso = [f for f in fs if f not in evalp and fk(f) not in evalf]
        rng = np.random.default_rng(DRAW_SEED)
        sel = sorted(np.array(iso)[rng.choice(len(iso), 10, replace=False)].tolist())
        frames = [fk(f) for f in sel]
        rec = {"pool_all_good": len(fs), "good_label_all": good,
               "isolated_pool": len(iso), "selected_n": len(sel),
               "selected": sel,
               "selected_sha256": {os.path.basename(f): sha256(f) for f in sel},
               "selected_source_frames": frames,
               "independent_source_frames": len(set(frames)),
               "multi_crop_frames": {str(f): frames.count(f) for f in set(frames) if frames.count(f) > 1},
               "file_overlap_with_eval": 0, "frame_overlap_with_eval": 0}
        assert not (set(sel) & evalp), "file overlap!"
        assert not (set(frames) & evalf), "frame overlap!"
        out["assets"][a] = rec
        print("[audit] %s: good=%s pool=%d isolated=%d selected=10 indep_frames=%d" % (
            a, good, len(fs), len(iso), rec["independent_source_frames"]))
    json.dump(out, open("/root/auto-exp/outputs/M3diag3/normal_ref.json", "w"), ensure_ascii=False, indent=1)
    json.dump({"verified": True, "note": "隔离核验通过: good标签/文件零重叠/源帧零重叠; 跨包映射不可追溯已披露;"
               "独立源帧数见 normal_ref.json; 抽样方差不估计(仅一组参照)"},
              open("/root/auto-exp/outputs/M3diag3/data_audit.json", "w"), ensure_ascii=False, indent=1)
    print("[done] normal_ref.json + data_audit.json")

if __name__ == "__main__":
    main()