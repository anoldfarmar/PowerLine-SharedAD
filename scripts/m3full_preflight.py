#!/usr/bin/env python3
"""M3FULL preflight: 正常候选池/查询/交集/源组身份/可行性. 只读, 不评分."""
import json, os, re, sys
sys.path.insert(0, "/root/auto-exp/scripts")

OUT = "/root/auto-exp/outputs/M3full-frozen"
COMBOS = [("glass-insulator", "missing-cap"), ("lightning-rod-suspension", "rust"),
          ("polymer-insulator-upper-shackle", "corrosion"), ("polymer-insulator-upper-shackle", "rust"),
          ("vari-grip", "bird-nest"), ("vari-grip", "rust"), ("yoke-suspension", "rust")]
ASSETS = sorted(set(a for a, _ in COMBOS))
GOOD_DIRS = ("good", "normal")
SRC_RE = re.compile(r"(\d{2}-\d{2}-\d{4}_DJI_\d{4}|\d{8}_\d{6}_DJI_\d{4})")


def dji_key(p):
    m = SRC_RE.search(os.path.basename(p))
    return m.group(1) if m else os.path.basename(p).rsplit(".", 1)[0][:24]


def main():
    os.makedirs(OUT, exist_ok=True)
    qq = json.load(open("/root/auto-exp/data/splits/insplad-fd/query_k5.json"))
    SL = json.load(open("/root/auto-exp/outputs/M3bench-dualbank-patchcore/SUPPORT_LISTS.json"))
    base = "/root/autodl-tmp/data/insplad/defect_supervised"
    pf = {"combos": {}, "assets": {}, "notes": []}
    all_query = set()
    for c, (a, f) in enumerate(COMBOS):
        qd = qq["defect_query"]["%s/%s" % (a, f)]
        qn = qq["normal_query"]["%s/good" % a]
        all_query.update(qd)
        all_query.update(qn)
        pf["combos"]["%s/%s" % (a, f)] = {"n_defect_query": len(qd), "n_normal_query": len(qn),
                                          "in_query_k5": True}
    pools, rows = {}, []
    for a in ASSETS:
        cand = []
        for split in ("train", "val"):
            for gd in GOOD_DIRS:
                d = os.path.join(base, a, split, gd)
                if os.path.isdir(d):
                    cand += [os.path.join(d, x) for x in sorted(os.listdir(d)) if x.lower().endswith((".jpg", ".png", ".jpeg"))]
        cand = sorted(set(cand))
        qn = qq["normal_query"]["%s/good" % a]
        qd_all = [p for c2, (a2, f2) in enumerate(COMBOS) if a2 == a for p in qq["defect_query"]["%s/%s" % (a2, f2)]]
        excl = set(qn) | set(qd_all)
        pool = [p for p in cand if p not in excl]
        pools[a] = pool
        qn_dji = set(dji_key(p) for p in qn)
        qd_dji = set(dji_key(p) for p in qd_all)
        pool_dji = set(dji_key(p) for p in pool)
        pf["assets"][a] = {
            "normal_candidates_raw": len(cand), "excluded_query_overlap": len(cand) - len(pool),
            "normal_candidate_pool": len(pool), "n_normal_query": len(qn),
            "n_defect_query_total": len(qd_all),
            "pool_dji_total": len(pool_dji),
            "pool_dji_also_in_normal_query_src": len(pool_dji & qn_dji),
            "pool_dji_also_in_defect_query_src": len(pool_dji & qd_dji),
            "satisfy_10N_5refs_with_overlap": len(pool) >= 10,
            "source_identity": "未知/可能同源 (裸DJI重号严重, 不作独立性声明; 仅路径级排除)",
            "reuse_m3budget_lists": a in ("yoke-suspension", "glass-insulator")}
        rows.append([a, len(cand), len(pool), len(qn), len(qd_all), len(pool_dji & qn_dji), len(pool_dji & qd_dji)])
    feasible = [c for c in COMBOS if pf["assets"][c[0]]["satisfy_10N_5refs_with_overlap"]]
    pf["feasible_combos"] = ["%s/%s" % c for c in feasible]
    pf["blocked_combos"] = ["%s/%s" % c for c in COMBOS if c not in feasible]
    pf["expected_configs"] = len(feasible) * 2 * 4 * 5
    pf["reuse_from_m3budget"] = {"configs": 80, "combos": ["yoke-suspension/rust", "glass-insulator/missing-cap"]}
    pf["notes"] = [
        "隔离口径: 仅路径级 支持∩查询=空; 源组身份按 M3DIAG5 审计为未知/可能同源 (裸DJI号跨航次重号), 不宣称独立未见或骨干未见",
        "既有 60 张正常参照 (M3DIAG3/4) 与 det-train 全部 DJI 碰撞; 本轮支持池沿用 defect_supervised good/normal 池, 属评测池隔离",
        "正常支持以裁剪计数, 非独立拍摄源帧数"]
    json.dump(pf, open(OUT + "/DATA_PREFLIGHT.json", "w"), ensure_ascii=False, indent=1)
    with open(OUT + "/DATA_PREFLIGHT.md", "w") as f:
        f.write("# DATA_PREFLIGHT — M3FULL-frozen-benchmark-v1\n\n")
        f.write("| 资产 | 原始候选 | 可用池 | 正常查询 | 故障查询合计 | 池DJI∩正常查询源 | 池DJI∩故障查询源 |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write("| %s | %d | %d | %d | %d | %d | %d |\n" % tuple(r))
        f.write("\n- 可行组合: %s\n- 阻塞组合: %s\n- 预计配置: %d (复用 80 + 新增 %d)\n\n## 口径披露\n\n" % (
            ", ".join(pf["feasible_combos"]) or "无", ", ".join(pf["blocked_combos"]) or "无",
            pf["expected_configs"], len(feasible) * 2 * 4 * 5))
        for n in pf["notes"]:
            f.write("- %s\n" % n)
    print("feasible:", len(feasible), "combos; expected configs:", pf["expected_configs"])
    for a in ASSETS:
        print(a, "pool=%d" % pf["assets"][a]["normal_candidate_pool"])


if __name__ == "__main__":
    main()
