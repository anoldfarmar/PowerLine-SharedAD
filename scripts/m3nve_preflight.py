#!/usr/bin/env python3
"""Read-only handoff preflight. Does not score queries or activate a task."""
import ast
import json
import shutil
from pathlib import Path
import sys

import m3nve_run as E


def main():
    assert sys.executable == '/root/miniconda3/bin/python3' or Path(sys.executable).resolve() == Path('/root/miniconda3/bin/python').resolve(), 'use base Python'
    assert E.torch.__version__=='2.8.0+cu128'
    import scipy.stats
    import matplotlib
    import PIL.Image
    import yaml
    frozen=E.verify_frozen()
    amendment=E.read(E.OUT/'EXECUTION_AMENDMENT.json')
    assert E.sha(E.ROOT/'scripts/m3nve_run.py')==amendment['new_run_sha256']
    assert E.sha(E.OUT/'quarantine/pre_handoff_m3nve_run.py')==amendment['old_run_sha256']
    assert E.sha(E.OUT/'quarantine/pre_handoff_FROZEN_INPUTS.json')==amendment['old_manifest_sha256']
    original=ast.parse((E.OUT/'quarantine/pre_handoff_m3nve_run.py').read_text())
    current=ast.parse((E.ROOT/'scripts/m3nve_run.py').read_text())
    funcs=lambda tree:{n.name:ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
    before,after=funcs(original),funcs(current)
    assert before.keys()==after.keys()
    assert {k for k in before if before[k]!=after[k]}=={'full','main'}
    # Everything in full from t0 onwards (all scientific computation) must be identical.
    bodies=lambda tree:next(n.body for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='full')
    tails=[]
    for tree in [original,current]:
        body=bodies(tree)
        idx=next(i for i,n in enumerate(body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='t0' for t in n.targets))
        tails.append([ast.dump(n,include_attributes=False) for n in body[idx:]])
    assert tails[0]==tails[1], 'scientific computation changed'
    for name,record in E.read(E.OUT/'CACHE_MANIFEST.json').items():
        assert E.sha(E.OUT/'cache'/name)==record['sha256'],name
    checks=E.read(E.OUT/'repro_and_correctness.json')
    assert checks['passed'] and checks['checks']['A_configs']==60
    assert checks['checks']['A_repro_max_diff']<=1e-4
    assert all(checks['checks']['synthetic'].values())
    lists=E.read(E.HIST/'SUPPORT_LISTS.json')
    ps=E.read(E.OUT/'cache/paths.json'); covered=set(ps)
    banks=E.np.load(E.OUT/'cache/banks.npz')
    mapped=E.np.load(E.OUT/'cache/mapped.npy',mmap_mode='r')
    assert mapped.shape==(len(ps),1600,128) and mapped.dtype==E.np.float32
    count=0
    for c in E.configurations(lists):
        assert set(E.query_paths(c)+c['normal'])<=covered
        assert banks[c['key']+'_N'].shape==(256,128)
        pc=E.read(E.HIST/'raw_scores_patchcore'/f"{c['key']}.json")
        assert all(set(pc[s])==set(c['query'][s]) for s in ['normal','defect'])
        for seed in range(3):
            name=E.cfg_name(c,seed)
            assert banks[name+'_F'].shape==(256,128)
            assert (E.OUT/'cache'/f'A_{name}.json').exists()
            count+=1
    assert count==60
    assert not (E.ROOT/'registry/STOP').exists()
    assert E.read(E.ROOT/'registry/active_task.json')['task']==E.TASK
    assert (E.ROOT/'registry/gates/M3_PASSED').exists()
    assert not (E.OUT/'nve_eval.json').exists(), 'experiment already completed'
    failures=E.read(E.OUT/'FAILURES.json')
    assert len(failures)<2
    free=shutil.disk_usage(E.OUT).free
    assert free>2*1024**3, 'need at least2GiB free for outputs'
    print(json.dumps(dict(status='READY_FOR_USER_RUN',configs=count,mapped_images=len(ps),
          historical_A_max_diff=checks['checks']['A_repro_max_diff'],
          frozen_inputs_verified=len(frozen['sha256']),cache_hashes_verified=True,
          scientific_code_unchanged=True,remaining_disk_bytes=free,
          conservative_hours=E.read(E.OUT/'cost_estimate.json')['estimated_total_gpu_reservation_seconds']/3600,
          original_1h_gate='blocked; explicit manual budget required',full_started=False),indent=2))


if __name__=='__main__': main()
