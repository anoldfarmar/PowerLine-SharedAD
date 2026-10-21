#!/usr/bin/env python3
"""CPU-only result verification and allowlisted publication. No evaluation changes."""
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil

import numpy as np
from sklearn.metrics import roc_auc_score

from insplad_freeze_utils import json_write

ROOT=Path(__file__).resolve().parents[1]
BASE=Path('/root/autodl-tmp/version10-12')


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as fp:
        for chunk in iter(lambda:fp.read(4*1024**2),b''):h.update(chunk)
    return h.hexdigest()


def main():
    coco=BASE/'InsPLAD-COCO-top8-full-v1'
    cplid=BASE/'CPLID-sharedP3-R0R1R2-v1'
    evidence=dict(scope='CPU raw-score/statistics verification and small-result publication only',source_files={})
    from cplid_shared_p3 import validate
    validate(read(cplid/'FROZEN_INPUTS.json'))
    evidence['cplid_frozen_input_code_weight_hashes_verified']=True
    rows=list(csv.DictReader((coco/'per_reference_metrics.csv').open()))
    assert len(rows)==210
    sets={}
    for row in rows:
        p=coco/'raw_scores'/f"{row['combination']}_{row['method']}_{row['budget']}_ref{row['ref']}.json"
        obj=read(p);normal=obj['normal'];fault=obj['defect']
        assert len(normal)==int(row['normal']) and len(fault)==int(row['defect'])
        assert set(normal).isdisjoint(fault)
        v=roc_auc_score([0]*len(normal)+[1]*len(fault),list(normal.values())+list(fault.values()))
        assert abs(v-float(row['auroc']))<1e-12
        key=(row['combination'],row['budget'],row['ref'])
        pair=(set(normal),set(fault))
        if key in sets:assert sets[key]==pair
        else:sets[key]=pair
        evidence['source_files'][str(p)]=sha(p)
    assert read(coco/'REPRO_AND_CORRECTNESS.json')['passed'] and read(coco/'metrics.json')['all_complete']
    evidence['coco_raw_AUROC_verified']=len(rows)
    evidence['coco_method_query_label_sets_equal']=True
    expected=read(coco/'metrics.json')['summary']
    for (comb,budget,method) in {(r['combination'],r['budget'],r['method']) for r in rows}:
        vals=[float(r['auroc']) for r in rows if (r['combination'],r['budget'],r['method'])==(comb,budget,method)]
        assert len(vals)==5
        assert abs(float(np.mean(vals))-expected[f'{comb}_{budget}'][method]['mean'])<1e-12
        assert abs(float(np.std(vals))-expected[f'{comb}_{budget}'][method]['std'])<1e-12
    cov=read(cplid/'COVERAGE.json');matches={r['id']:r['matched_GT_indices'] for r in cov['frames']}
    cm=read(cplid/'metrics.json');assert cm['completed']
    checks=read(cplid/'REPRO_AND_CORRECTNESS.json')
    assert checks['passed'] and checks['state_unchanged_after_all_inference']
    count=0
    for path in sorted((cplid/'raw_scores').glob('*.json')):
        rs=read(path);by={a:[r for r in rs if r['arm']==a and r['label'] is not None] for a in ['R0','R1','R2']}
        assert {(r['image_id'],r['GT_index'],r['label']) for r in by['R1']}=={(r['image_id'],r['GT_index'],r['label']) for r in by['R2']}
        for arm,values in by.items():
            m=next(m for m in cm['metrics'] if m['reference']==path.stem and m['arm']==arm)
            v=roc_auc_score([r['label'] for r in values],[r['score'] for r in values]);assert abs(v-m['AUROC'])<1e-12
            pair=[r for r in values if r['GT_index'] in matches[r['image_id']]]
            v=roc_auc_score([r['label'] for r in pair],[r['score'] for r in pair]);assert abs(v-m['paired_AUROC'])<1e-12
            count+=1
        evidence['source_files'][str(path)]=sha(path)
    assert count==15;evidence['cplid_raw_AUROC_verified']=count
    timing=[json.loads(l) for l in (cplid/'TIMING_RAW.jsonl').read_text().splitlines()]
    assert len(timing)==368*3*2
    assert len({(r['method'],r['image_id'],r['round']) for r in timing})==len(timing)
    speed=read(cplid/'SPEED_METRICS.json')['groups']['all']
    counters={}
    for arm in ['R1','R2']:
        ts=[r for r in timing if r['method']==arm]
        assert abs(float(np.mean([r['ms'] for r in ts]))-speed[arm]['mean_ms'])<1e-10
        counters[arm]={k:sum(r['counters'][k] for r in ts) for k in ts[0]['counters']}
        assert counters[arm]['full']==1104 and counters[arm]['full_images']==1104
        assert counters[arm]['crop']==(1104 if arm=='R1' else 0)
        assert counters[arm]['crop_objects']==(1491 if arm=='R1' else 0)
    evidence['cplid_timing_observations_verified']=len(timing);evidence['cplid_forward_counters']=counters
    allow={
        'InsPLAD-COCO-top8-full-v1':['PERFORMANCE_TABLE.md','metrics.json','per_reference_metrics.csv',
            'performance_comparison.csv','paired_comparisons.csv','paired_summary.json','REPRO_AND_CORRECTNESS.json',
            'RESOURCE_COST.json','FROZEN_INPUTS.json','REUSE_MANIFEST.json','BANK_MANIFEST.json'],
        'CPLID-sharedP3-R0R1R2-v1':['REPORT.md','metrics.json','per_reference_metrics.csv','REPRO_AND_CORRECTNESS.json',
            'RESOURCE_COST.json','COVERAGE.json','SPEED_METRICS.json','TIMING_RAW.jsonl','STAGE_PROFILE.json',
            'DECODE_TIMING.json','BANK_MANIFEST.json','CACHE_MANIFEST.json'],
    }
    evidence['published_files']={}
    for task,names in allow.items():
        for name in names:
            src=BASE/task/name;dst=ROOT/'outputs/reports'/task/name
            dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
            assert sha(src)==sha(dst)
            evidence['published_files'][str(dst.relative_to(ROOT))]=sha(dst)
    for src in sorted((cplid/'raw_scores').glob('*.json')):
        dst=ROOT/'outputs/reports'/cplid.name/'raw_scores'/src.name
        dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
        assert sha(src)==sha(dst);evidence['published_files'][str(dst.relative_to(ROOT))]=sha(dst)
    evidence['coco_raw_scores_policy']='29MiB raw scores remain local; all210 source file hashes recorded here; compact metrics/manifests published'
    doc=ROOT/'README.md';text=doc.read_text()
    links=re.findall(r'\]\(([^)]+)\)',text)
    bad=[v for v in links if '://' not in v and not (doc.parent/v.split('#')[0]).exists()]
    assert not bad,bad
    evidence['README_relative_links_verified']=len(links)
    for arm in ['R0','R1','R2']:
        mean=float(np.mean([m['AUROC'] for m in cm['metrics'] if m['arm']==arm]))
        assert f'{mean:.6f}' in text
    for arm in ['R1','R2']:assert f"{speed[arm]['mean_ms']:.3f}" in text
    evidence['README_sha256']=sha(doc)
    evidence['no_training_or_GPU_inference']=True;evidence['passed']=True
    json_write(ROOT/'outputs/reports/README-results-sync-v1/VERIFICATION.json',evidence)
    print(json.dumps({k:v for k,v in evidence.items() if k not in ['source_files','published_files']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
