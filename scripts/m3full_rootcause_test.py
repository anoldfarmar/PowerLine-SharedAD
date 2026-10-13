#!/usr/bin/env python3
"""M3FULL E2-FIX 根因验证.
1) 模拟: 两个正常样本数相同、路径不同的资产连续保存 —
   固定版断言全部通过且逐路径映射准确; 故意复刻旧 bug (沿用前一资产 qn) 必须被断言拦下.
2) 真实产物: 修正后 M3FULL raw_scores (Y 臂 100 文件) 逐文件核对
   normal/defect 键集合 == 冻结清单, 长度一致, AUROC 重算一致. 只读, 不改写."""
import json, sys
import numpy as np
sys.path.insert(0, "/root/auto-exp/scripts")
import m3diag_common as D

FIX_OK = {"sim_fixed": 0, "sim_broken_caught": 0, "real_files": 0, "real_fail": []}


def save_like(asset, qn_used, qd, scores_n, scores_d, store):
    """复刻 m3full_run.py 保存逻辑的断言部分 (E2-FIX 后口径)."""
    allq = list(qn_used) + list(qd)
    assert len(set(allq)) == len(allq), "路径不唯一"
    assert set(qn_used).isdisjoint(set(qd)), "正常/故障相交"
    assert all(("/%s/" % asset) in p for p in allq), "出现他资产路径"
    assert len(scores_n) == len(qn_used) and len(scores_d) == len(qd), "分数数与路径数不一致"
    store[asset] = {"normal": {p: float(v) for p, v in zip(qn_used, scores_n)},
                     "defect": {p: float(v) for p, v in zip(qd, scores_d)}}


def sim():
    n = 5
    A_paths = ["/mock/assetA/good/img_%d.jpg" % i for i in range(n)]
    B_paths = ["/mock/assetB/good/other_%d.jpg" % i for i in range(n)]
    qd = ["/mock/assetA/fault/f_%d.jpg" % i for i in range(3)]
    qd_B = ["/mock/assetB/fault/g_%d.jpg" % i for i in range(3)]
    sA = np.arange(n, dtype=np.float64) + 0.10
    sB = np.arange(n, dtype=np.float64) + 0.90
    store = {}
    # --- 固定版: 每资产自己的 qn ---
    save_like("assetA", A_paths, qd, sA, np.arange(3.0), store)
    save_like("assetB", B_paths, qd_B, sB, np.arange(3.0) + 5, store)
    assert all(p.startswith("/mock/assetB/") for p in store["assetB"]["normal"]), "assetB 继承了 assetA 路径"
    assert abs(store["assetB"]["normal"][B_paths[2]] - 2.90) < 1e-12, "assetB 路径-分数映射错误"
    assert abs(store["assetA"]["normal"][A_paths[2]] - 2.10) < 1e-12, "assetA 路径-分数映射错误"
    FIX_OK["sim_fixed"] = 1
    # --- 旧 bug 复刻: assetB 误用 assetA 的 qn (其余正确) 必须被拦 ---
    try:
        save_like("assetB", A_paths, qd_B, sB, np.arange(3.0) + 5, store)
        raise SystemExit("旧 bug 未被拦截 — 断言失效!")
    except AssertionError:
        FIX_OK["sim_broken_caught"] = 1
    # --- 长度同但分数错位 (仅检查长度会漏) 的映射抽查 ---
    try:
        store2 = {}
        save_like("assetB", B_paths, qd_B, sA, np.arange(3.0), store2)  # 长度合法但值是 A 的
        assert abs(store2["assetB"]["normal"][B_paths[0]] - 0.90) < 1e-9, "应检出值错位"
        raise SystemExit("值错位未被映射抽查发现!")
    except AssertionError:
        pass  # 期望: 映射抽查断言 (基于注入值核对) 拦下
    print("[sim] 固定版通过; 旧bug被拦截=%s" % FIX_OK["sim_broken_caught"])


def real():
    qq = json.load(open(D.SPLITS + "/query_k5.json"))
    ASSETS = {"lightning-rod-suspension": ["lightning-rod-suspension__rust"],
               "polymer-insulator-upper-shackle": ["polymer-insulator-upper-shackle__corrosion", "polymer-insulator-upper-shackle__rust"],
               "vari-grip": ["vari-grip__bird-nest", "vari-grip__rust"]}
    base = D.ROOT + "/outputs/M3full-frozen/raw_scores"
    for a, combos in ASSETS.items():
        qn = set(qq["normal_query"]["%s/good" % a])
        for c in combos:
            qd = set(qq["defect_query"][c.replace(a + "__", a + "/")])
            for arm in ("Y-PC", "Y-T8"):
                for b in ("5N", "10N"):
                    for j in range(3001, 3006):
                        r = json.load(open("%s/%s_%s_%s_ref%d.json" % (base, c, arm, b, j)))
                        assert set(r["normal"]) == qn, (c, arm, b, j, "normal键不符")
                        assert set(r["defect"]) == qd, (c, arm, b, j, "defect键不符")
                        assert len(r["normal"]) == len(qn) and len(r["defect"]) == len(qd)
                        u, _ = D.auroc_avg_rank(np.array(list(r["defect"].values())), np.array(list(r["normal"].values())))
                        assert abs(u - r["auroc"]) <= 1e-12, (c, arm, b, j)
                        FIX_OK["real_files"] += 1
    print("[real] Y 臂身份复核通过文件数=%d/100" % FIX_OK["real_files"])


if __name__ == "__main__":
    sim()
    real()
    print("E2FIX_RESULT:", json.dumps(FIX_OK))
