#!/usr/bin/env python3
"""Export allowed metadata/results and safely sync only this completed task."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

import insplad_shared_roi as old

TASK='M3SHARE-speed-v1'
EXPORT=['METHOD_SPEC.md','DATA_AUDIT.json','FROZEN_INPUTS.json','REPRO_AND_CORRECTNESS.json',
        'TIMING_PROTOCOL.json','PROBE.json','TIMING_RAW.jsonl','DISK_IO.json','RESOURCE_COST.json',
        'ACCURACY_SUMMARY.json','SPEED_METRICS.json','SPEED_REPORT.md','COMPLETED.json','BANK_MANIFEST.json']


def git(*args):
    return subprocess.check_output(['git',*args],cwd=old.ROOT).decode().strip()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=old.BASE/TASK)
    args=parser.parse_args()
    assert old.read(args.out/'COMPLETED.json')['complete']
    assert old.read(args.out/'REPRO_AND_CORRECTNESS.json')['passed']
    frozen=old.read(args.out/'FROZEN_INPUTS.json')
    for name,sha in frozen['code_sha256'].items():
        assert old.sha(old.ROOT/'scripts'/name)==sha, 'Code changed after freeze'
        tracked=subprocess.check_output(['git','show',f'HEAD:scripts/{name}'],cwd=old.ROOT)
        import hashlib
        assert hashlib.sha256(tracked).hexdigest()==sha, 'Commit the exact prepared code before publishing its results'
    assert git('branch','--show-current')=='main'
    assert git('remote','get-url','origin')=='https://gitcode.com/gcw_95nPugfk/bounded-proto-det.git'
    assert not git('diff','--cached','--name-only'),'Unrelated staged files: stop rather than commit them'
    subprocess.run(['git','fetch','origin','main'],cwd=old.ROOT,check=True)
    _,remote_ahead=map(int,git('rev-list','--left-right','--count','HEAD...origin/main').split())
    assert remote_ahead==0,'Remote ahead/diverged: keep local results; do not merge, force or overwrite'
    destination=old.ROOT/'outputs/reports'/TASK
    destination.mkdir(parents=True,exist_ok=True)
    for name in EXPORT:
        src=args.out/name
        assert src.is_file() and src.stat().st_size<25*1024*1024, f'Not a small report/metadata artifact: {name}'
        shutil.copyfile(src,destination/name)
    manifest=dict(task=TASK,exported={n:old.sha(destination/n) for n in EXPORT},
                  omitted=['cache/','accuracy/raw_scores.jsonl','images','weights'],
                  original_result_dir=str(args.out),scope=frozen['scope'])
    old.write(destination/'EXPORT_MANIFEST.json',manifest)
    relative=[str((destination/n).relative_to(old.ROOT)) for n in EXPORT+['EXPORT_MANIFEST.json']]
    subprocess.run(['git','add','--',*relative],cwd=old.ROOT,check=True)
    # The worktree registry can contain unrelated, still-uncommitted runs.
    base=git('show','HEAD:registry/runs.jsonl')
    existing=[json.loads(line) for line in base.splitlines() if line.strip()]
    known={(r['task'],r.get('ts')) for r in existing}
    new=[line for line in (old.ROOT/'registry/runs.jsonl').read_text().splitlines()
         if line.strip() and json.loads(line)['task']==TASK and (json.loads(line)['task'],json.loads(line).get('ts')) not in known]
    assert new,'No new PASS registry record for this task'
    assert all(json.loads(line)['status']=='PASS' for line in new)
    blob=subprocess.check_output(['git','hash-object','-w','--stdin'],cwd=old.ROOT,
                                 input=(base+'\n'+'\n'.join(new)+'\n').encode()).decode().strip()
    subprocess.run(['git','update-index','--cacheinfo',f'100644,{blob},registry/runs.jsonl'],cwd=old.ROOT,check=True)
    staged=git('diff','--cached','--name-only').splitlines()
    assert all(n in relative+['registry/runs.jsonl'] for n in staged)
    subprocess.run(['git','diff','--cached','--check'],cwd=old.ROOT,check=True)
    subprocess.run(['git','diff','--cached','--stat'],cwd=old.ROOT,check=True)
    subprocess.run(['git','commit','-m','Publish M3SHARE local crop-vs-share actual timing and bounded accuracy'],cwd=old.ROOT,check=True)
    subprocess.run(['git','push','origin','main'],cwd=old.ROOT,check=True)
    assert git('rev-parse','HEAD')==git('rev-parse','origin/main')
    print('[published]',git('rev-parse','HEAD'),flush=True)


if __name__=='__main__':
    main()
