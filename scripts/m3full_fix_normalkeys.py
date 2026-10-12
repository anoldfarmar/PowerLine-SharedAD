#!/usr/bin/env python3
"""M3FULL 数据修正: 新资产 Y 臂 raw_scores 的 normal 键错标为 yoke 正常查询路径.
值未变 (按 allq 顺序 = 该资产自身 qn 顺序), 仅键重建; AUROC 必须逐文件不变."""
import json, os, sys
import numpy as np
sys.path.insert(0, '/root/auto-exp/scripts')
import m3diag_common as D

qq = json.load(open(D.SPLITS + '/query_k5.json'))
ASSETS = {'lightning-rod-suspension': [('lightning-rod-suspension__rust', 'lightning-rod-suspension/rust')],
          'polymer-insulator-upper-shackle': [('polymer-insulator-upper-shackle__corrosion', 'polymer-insulator-upper-shackle/corrosion'),
                                               ('polymer-insulator-upper-shackle__rust', 'polymer-insulator-upper-shackle/rust')],
          'vari-grip': [('vari-grip__bird-nest', 'vari-grip/bird-nest'), ('vari-grip__rust', 'vari-grip/rust')]}
BUDGETS = ['5N', '10N']
REFS = [3001, 3002, 3003, 3004, 3005]
base = '/root/auto-exp/outputs/M3full-frozen/raw_scores'
yoke_qn = set(qq['normal_query']['yoke-suspension/good'])
nfix, nauroc = 0, 0
for a, combos in ASSETS.items():
    qn_correct = list(dict.fromkeys(qq['normal_query']['%s/good' % a]))
    for c, ck in combos:
        qd = qq['defect_query'][ck]
        for arm in ('Y-PC', 'Y-T8'):
            for b in BUDGETS:
                for j in REFS:
                    p = '%s/%s_%s_%s_ref%d.json' % (base, c, arm, b, j)
                    r = json.load(open(p))
                    vals = list(r['normal'].values())
                    assert len(vals) == len(qn_correct) == 500, (p, len(vals))
                    assert set(r['normal'].keys()) <= yoke_qn, p
                    u_old, _ = D.auroc_avg_rank(np.array(list(r['defect'].values()), np.float64), np.array(vals, np.float64))
                    assert abs(u_old - r['auroc']) <= 1e-12, (p, u_old, r['auroc'])
                    r['normal'] = {p2: v for p2, v in zip(qn_correct, vals)}
                    u_new, _ = D.auroc_avg_rank(np.array(list(r['defect'].values()), np.float64), np.array(list(r['normal'].values()), np.float64))
                    assert abs(u_new - r['auroc']) <= 1e-12, (p, u_new, r['auroc'])
                    assert set(r['normal'].keys()) == set(qn_correct)
                    with open(p, 'w') as f:
                        json.dump(r, f)
                    nfix += 1
                    if abs(u_new - u_old) <= 1e-12:
                        nauroc += 1
print('fixed files:', nfix, '/ 100; auroc unchanged:', nauroc)
