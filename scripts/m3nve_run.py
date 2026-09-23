#!/usr/bin/env python3
"""Frozen, gated M3NVE experiment. No training, parameter search or PatchCore execution."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

import m3diag_common as D
import m3diag6_dualbank as M6
import m3_common as MC
from m3diag_bank_eval import embed
from m3_build_patch_bank import coreset_farthest

ROOT = Path(D.ROOT)
OUT = ROOT / 'outputs/M3nve-residual'
HIST = ROOT / 'outputs/M3bench-dualbank-patchcore'
TASK = 'M3NVE-residual-v1'
PARAMS = dict(k=12, r=4, alpha_B=0., alpha_C=.8, eps=1e-8,
              rank_tol=1e-6, batch=32, svd_chunk=2048, top8=8,
              svd_dtype='float32', svd_driver='gesvdj', repro_tol=1e-4,
              consistency_tol=1e-5, cap=256, fps_seed=0, keep_frac=.5)
DEV = 'cuda'


def read(p):
    return json.loads(Path(p).read_text())


def save(name, obj):
    p = OUT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def sha(p):
    return M6.sha256_file(str(p))


def tensor_sha(x):
    return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def state_sha(model):
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def sync_time():
    torch.cuda.synchronize()
    return time.perf_counter()


def unit(x):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(PARAMS['eps'])


def clean(r, projection, alpha):
    return r-alpha*projection


@torch.no_grad()
def local(x, n, chunk=None):
    """One stable normal search shared by nearest residual and local PCA."""
    chunk = chunk or PARAMS['svd_chunk']
    output = [[] for _ in range(5)]
    for start in range(0, len(x), chunk):
        v = x[start:start+chunk]
        sim = v @ n.T
        ids = torch.argsort(sim, dim=-1, descending=True, stable=True)[:, :PARAMS['k']]
        residual = v-n[ids[:, 0]]
        neighbours = n[ids]
        centered = neighbours-neighbours.mean(1, keepdim=True)
        # Tall thin SVD avoids a 128x128 covariance or projector.
        u, s, _ = torch.linalg.svd(centered.transpose(1, 2), full_matrices=False,
                                   driver=PARAMS['svd_driver'])
        rank = (s > PARAMS['rank_tol']*s[:, :1]).sum(-1).clamp_max(PARAMS['r'])
        u = u[:, :, :PARAMS['r']]
        mask = torch.arange(PARAMS['r'], device=x.device)[None, :] < rank[:, None]
        coeff = torch.einsum('ndr,nd->nr', u, residual)*mask
        proj = torch.einsum('ndr,nr->nd', u, coeff)
        for dst, val in zip(output, (1-sim.gather(1, ids[:, :1])[:, 0], residual, proj, rank, ids[:, 0])):
            dst.append(val)
    return tuple(torch.cat(v) for v in output)


def residual_score(dn, rq, pq, rf, pf, alpha):
    return .5*(dn+(unit(clean(rq, pq, alpha)) @ unit(clean(rf, pf, alpha)).T).max(-1).values)


def image_score(v):
    return v.reshape(-1, 1600).topk(8, dim=1).values.mean(1)


def synthetic_checks():
    torch.manual_seed(923)
    d = DEV
    u = torch.linalg.qr(torch.randn(16, 4, device=d)).Q
    r = torch.randn(50, 16, device=d)
    p = (r@u)@u.T
    inside = torch.randn(50, 4, device=d)@u.T
    outside = r-p
    results = {
        'alpha0_identity': torch.equal(clean(r, p, 0), r),
        'in_subspace_scales_point2': torch.allclose(clean(inside, (inside@u)@u.T, .8), .2*inside, atol=2e-6),
        'orthogonal_unchanged': torch.allclose(clean(outside, (outside@u)@u.T, .8), outside, atol=2e-6),
        'norm_nonincrease': all(bool((clean(r, p, a).norm(dim=1) <= r.norm(dim=1)+2e-6).all()) for a in [0., .2, .8, 1.]),
        'zero_cosine': bool((unit(torch.zeros_like(r))@unit(r).T == 0).all()),
    }
    n = unit(torch.randn(20, 16, device=d)); x = unit(torch.randn(43, 16, device=d))
    a = local(x, n, 43); b = local(x, n, 7)
    results['chunk_vs_direct'] = all(torch.allclose(v, w, atol=2e-6) for v, w in zip(a, b))
    z = local(n[:1].expand(3, -1), n[:1].expand(20, -1))
    # Exactly representable repeated neighbours test zero rank without centering round-off.
    e = torch.zeros(20, 16, device=d); e[:, 0] = 1
    z = local(e[:3], e)
    results['degenerate_zero_rank'] = bool((z[3] == 0).all() and torch.equal(clean(z[1], z[2], .8), z[1]))
    results['zero_finite'] = bool(all(torch.isfinite(v).all() for v in z))
    results['stable_tie_first_index'] = bool((z[4] == 0).all())
    sb = residual_score(a[0], a[1], a[2], b[1], b[2], 0.)
    direct = .5*(a[0]+(unit(a[1])@unit(b[1]).T).max(-1).values)
    results['B_equals_C_alpha0'] = bool(torch.equal(sb, direct))
    patches = torch.arange(3200, dtype=torch.float32, device=d).reshape(2, 1600)
    results['top8_patch_dimension'] = bool(torch.equal(image_score(patches), patches[:, -8:].mean(1)))
    # Independent scalar SVD on non-identical neighbourhoods validates projection orientation.
    sim = x[0]@n.T; ids = torch.argsort(sim, descending=True, stable=True)[:12]
    c = n[ids]-n[ids].mean(0)
    _, s, vh = torch.linalg.svd(c, full_matrices=False)
    rank = min(4, int((s > s[0]*1e-6).sum()))
    rr = x[0]-n[ids[0]]; pp = (rr@vh[:rank].T)@vh[:rank]
    results['independent_projection'] = bool(torch.allclose(a[2][0], pp, atol=3e-6))
    assert all(results.values()), results
    return results


def configurations(lists):
    for g, k, ns in [('g1', 1, 'N1'), ('g2', 5, 'N5')]:
        for combo in M6.COMBOS:
            cc = combo.replace('/', '__')
            for ref in range(3001, 3006):
                key = f'{cc}_{g}_ref{ref}'
                yield dict(key=key, combo=combo, cc=cc, g=g, k=k, ref=ref,
                           normal=lists['normal_lists'][f'{D.ASSET_OF[combo]}_ref{ref}'][ns],
                           query=lists['queries'][f'{cc}_{g}'])


def cfg_name(c, seed):
    return f"{c['cc']}_{c['g']}_k{c['k']}_seed{seed}_ref{c['ref']}"


def query_paths(c):
    return c['query']['defect']+c['query']['normal']


def get_features(mapped, index, paths):
    return torch.from_numpy(np.asarray(mapped[[index[p] for p in paths]])).to(DEV)


def guard():
    assert read(ROOT/'registry/active_task.json')['task'] == TASK, 'task not activated'
    assert not (ROOT/'registry/STOP').exists(), 'human STOP'
    assert (ROOT/'registry/gates/M3_PASSED').exists()
    assert torch.cuda.is_available()
    assert torch.__version__ == '2.8.0+cu128', 'use historical /root/miniconda3/bin/python'
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    # Preserve historical convolution extraction; mapper/matmul/SVD scoring is FP32.
    torch.backends.cudnn.allow_tf32 = True
    torch.set_num_threads(4)


def freeze():
    frozen_path = OUT/'FROZEN_INPUTS.json'
    assert not frozen_path.exists(), 'refuse replacing frozen implementation'
    paths = [Path(D.WEIGHTS), Path(D.OUT)/'mapper_B.pt', Path(D.CACHE)/'patches.npy',
             Path(D.CACHE)/'eval_paths.json', HIST/'SUPPORT_LISTS.json', HIST/'FROZEN_INPUTS.json',
             ROOT/'paper/Beyond Normal References.pdf', ROOT/f'plan/tasks/{TASK}.yaml', OUT/'METHOD_SPEC.md']
    paths += sorted((ROOT/'scripts').glob('m3*.py'))
    paths += sorted((ROOT/'data/splits/insplad-fd').glob('*.json'))
    paths += sorted((HIST/'raw_scores_ours').glob('*.json'))
    paths += sorted((HIST/'raw_scores_patchcore').glob('*.json'))
    lists = read(HIST/'SUPPORT_LISTS.json')
    images = set()
    for c in configurations(lists):
        images.update(c['normal']); images.update(query_paths(c))
    for value in lists['fault_supports'].values():
        images.update(value['paths'])
    paths += [Path(p) for p in sorted(images)]
    hashes = {str(p): sha(p) for p in paths}
    old = read(HIST/'FROZEN_INPUTS.json')
    assert hashes[D.WEIGHTS] == old['weights']['sha256']
    assert hashes[str(Path(D.OUT)/'mapper_B.pt')] == old['mapper_B']['sha256']
    for name in ['patches', 'eval_paths']:
        p = Path(D.CACHE)/(name+('.npy' if name == 'patches' else '.json'))
        assert hashes[str(p)] == old['cache'][name]
    for value in lists['normal_lists'].values():
        for p, h in zip(value['ordered_paths'], value['sha256_ordered']):
            if p in hashes:
                assert hashes[p] == h
    selected = {}
    for c in configurations(lists):
        if c['ref'] == 3001:
            selected[c['key']] = {s: sorted(c['query'][s])[:2] for s in ['normal', 'defect']}
    save('FROZEN_INPUTS.json', dict(timestamp=time.time(), parameters=PARAMS,
         sha256=hashes, code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
         env=dict(python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
                  gpu=torch.cuda.get_device_name(), matmul_tf32=False, historical_cudnn_tf32=True),
         visual_selection=selected, source_lists='direct historical lists; no resampling',
         previous_task_completed='M3BENCH PASS 2026-09-22T16:21:38',
         historical_environment=old['env']))


def verify_frozen():
    frozen = read(OUT/'FROZEN_INPUTS.json')
    assert frozen['parameters'] == PARAMS
    for p, expected in frozen['sha256'].items():
        assert sha(p) == expected, f'frozen input changed: {p}'
    return frozen


def prepare():
    t0 = sync_time()
    freeze()
    checks = dict(synthetic=synthetic_checks())
    lists = read(HIST/'SUPPORT_LISTS.json')
    paths = set()
    for c in configurations(lists):
        paths.update(query_paths(c)); paths.update(c['normal'])
    for v in lists['fault_supports'].values():
        paths.update(v['paths'])
    paths = sorted(paths); index = {p: i for i, p in enumerate(paths)}
    cache = OUT/'cache'; cache.mkdir(exist_ok=True)
    save('cache/paths.json', paths)
    ep = read(Path(D.CACHE)/'eval_paths.json'); ei = {p:i for i,p in enumerate(ep)}
    raw = np.load(Path(D.CACHE)/'patches.npy', mmap_mode='r')
    missing = [p for p in paths if p not in ei]
    print(f'[prepare] paths={len(paths)}, missing normal refs={len(missing)}', flush=True)
    head = D.load_head_state(D.OUT+'/mapper_B.pt', DEV).to(DEV).eval()
    head.requires_grad_(False); hs = state_sha(head)
    spec = ('head', head)
    normal_raw = {}
    backbone_record = {'invoked': False}
    if missing:
        from PIL import Image
        bb = MC.Backbone(D.WEIGHTS, DEV)
        before = state_sha(bb.model); tf = MC.make_transform(D.IMGSZ)
        for p in missing:
            p3, _ = bb.p3_and_p5gap(tf(Image.open(p).convert('RGB')).unsqueeze(0))
            normal_raw[p] = p3.permute(0, 2, 3, 1)[0].cpu().numpy().astype(np.float16)
        after = state_sha(bb.model)
        assert before == after
        backbone_record = dict(invoked=True, missing_images=len(missing), before=before, after=after,
                               batch=1, reason='match historical reference extraction; only missing normal images')
        np.save(cache/'normal_raw.npy', np.stack([normal_raw[p] for p in missing]))
        save('cache/normal_paths.json', missing)
        del bb
    mapped = np.lib.format.open_memmap(cache/'mapped.npy', mode='w+', dtype=np.float32, shape=(len(paths), 1600, 128))
    for start in range(0, len(paths), PARAMS['batch']):
        ps = paths[start:start+PARAMS['batch']]
        arr = np.stack([np.asarray(raw[ei[p]]) if p in ei else normal_raw[p] for p in ps])
        z = embed(spec, torch.from_numpy(arr.reshape(-1, 64)).float(), DEV).reshape(-1, 1600, 128)
        mapped[start:start+len(ps)] = z.cpu().numpy()
    mapped.flush()
    assert state_sha(head) == hs
    checks['mapper'] = dict(before=hs, after=state_sha(head), eval=True, gradients=False)
    checks['backbone'] = backbone_record
    # Mapping numerical consistency before query scoring.
    ps = paths[:2]
    single = []
    for p in ps:
        arr = np.asarray(raw[ei[p]]) if p in ei else normal_raw[p]
        single.append(embed(spec, torch.from_numpy(arr.reshape(-1,64)).float(), DEV).cpu().numpy())
    checks['mapping_single_batch_diff'] = float(np.max(np.abs(mapped[:2]-np.stack(single))))
    assert checks['mapping_single_batch_diff'] <= 1e-6
    del head, normal_raw
    bank_manifest = {}; all_banks = {}
    for c in configurations(lists):
        v = get_features(mapped, index, c['normal']).reshape(-1,128)
        n = v[torch.as_tensor(coreset_farthest(v, 256, 0), device=DEV)]
        all_banks[c['key']+'_N'] = n.cpu().numpy()
        bank_manifest[c['key']+'_N'] = dict(sha256=tensor_sha(n), shape=list(n.shape), bytes=n.nbytes)
        for seed in range(3):
            sk = f"{c['cc']}_{c['g']}_k{c['k']}_seed{seed}"
            fp = lists['fault_supports'][sk]['paths']
            assert not set(fp)&set(query_paths(c))
            f = M6.p1_fault_bank(get_features(mapped,index,fp).reshape(-1,128),n,DEV)
            name=cfg_name(c,seed)
            all_banks[name+'_F']=f.cpu().numpy()
            bank_manifest[name+'_F']=dict(sha256=tensor_sha(f),shape=list(f.shape),bytes=f.nbytes)
    np.savez(cache/'banks.npz', **all_banks)
    save('BANK_MANIFEST.json', bank_manifest)
    save('CACHE_MANIFEST.json', {p.name:dict(sha256=sha(p), bytes=p.stat().st_size) for p in sorted(cache.iterdir()) if p.is_file()})
    # Whole A gate: no residual query scoring until all 60 configurations pass.
    repro=[]
    for c in configurations(lists):
        n=torch.as_tensor(all_banks[c['key']+'_N'],device=DEV)
        fs=[torch.as_tensor(all_banks[cfg_name(c,s)+'_F'],device=DEV) for s in range(3)]
        pathsq=query_paths(c); outputs=[[] for _ in fs]
        for start in range(0,len(pathsq),32):
            q=get_features(mapped,index,pathsq[start:start+32]).reshape(-1,128)
            sn=(q@n.T).max(-1).values
            for f, rows in zip(fs,outputs):
                rows.extend(image_score((q@f.T).max(-1).values-sn).cpu().tolist())
        nd=len(c['query']['defect'])
        pc=read(HIST/'raw_scores_patchcore'/f"{c['key']}.json")
        for side in ['normal','defect']:
            assert set(pc[side])==set(c['query'][side])
        for seed,vals in enumerate(outputs):
            name=cfg_name(c,seed); old=read(HIST/'raw_scores_ours'/f'{name}.json')
            for side in ['normal','defect']:
                assert set(old[side])==set(c['query'][side])
            expected=[old['defect' if i<nd else 'normal'][p] for i,p in enumerate(pathsq)]
            diff=float(np.max(np.abs(np.array(vals)-expected)))
            repro.append(dict(config=name,max_abs_diff=diff,n_paths=len(pathsq)))
            save(f'cache/A_{name}.json',dict(zip(pathsq,vals)))
            print(f'[A gate] {name}: {diff:.3g}',flush=True)
            if diff>PARAMS['repro_tol']:
                save('repro_and_correctness.json',dict(checks=checks,repro=repro,passed=False))
                raise AssertionError(f'A reproduction failed {name}: {diff} > 1e-4')
    checks['A_repro_max_diff']=max(v['max_abs_diff'] for v in repro)
    checks['A_configs']=len(repro)
    # Fixed first normal and defect images, 4 asset/budget combinations.
    consistency=[]; probes=[]
    for c in configurations(lists):
        if c['ref']!=3001: continue
        n=torch.as_tensor(all_banks[c['key']+'_N'],device=DEV)
        f=torch.as_tensor(all_banks[cfg_name(c,0)+'_F'],device=DEV)
        fl=local(f,n)
        ps=[sorted(c['query'][s])[0] for s in ['normal','defect']]
        q=get_features(mapped,index,ps).reshape(-1,128)
        full=local(q,n); one=[local(v,n) for v in q.reshape(2,1600,128)]
        err=max(float((v-torch.cat([z[i] for z in one])).abs().max()) for i,v in enumerate(full[:3]))
        scores=[]
        for alpha in [0.,.8]:
            bat=residual_score(*full[:3],fl[1],fl[2],alpha)
            singles=torch.cat([residual_score(*z[:3],fl[1],fl[2],alpha) for z in one])
            scores.append(float((bat-singles).abs().max()))
        assert max([err]+scores)<=PARAMS['consistency_tol'], (err,scores)
        consistency.append(dict(config=c['key'],paths=ps,residual_diff=err,score_diff=scores))
        # fixed lexicographic 32 paths; timing only, not performance-based selection
        ps=sorted(query_paths(c))[:32]; q=get_features(mapped,index,ps).reshape(-1,128)
        t=sync_time(); ql=local(q,n)
        for seed in range(3):
            f=torch.as_tensor(all_banks[cfg_name(c,seed)+'_F'],device=DEV); fl=local(f,n)
            for alpha in [0.,.8]:
                image_score(residual_score(*ql[:3],fl[1],fl[2],alpha))
        sec=sync_time()-t
        probes.append(dict(config=c['key'],images=32,seconds=sec,paths=ps))
        print(f'[probe] {c["key"]}: 32 images + 3 seeds {sec:.2f}s',flush=True)
    elapsed=sync_time()-t0
    # conservative worst throughput times all 20 normal configs, 50% margin + preparation
    total_queries=sum(len(query_paths(c)) for c in configurations(lists))
    estimate=elapsed+max(p['seconds']/p['images'] for p in probes)*total_queries*1.5
    save('cost_estimate.json',dict(probes=probes,prepare_seconds=elapsed,query_images_across_20_normal_configs=total_queries,
          estimated_total_gpu_reservation_seconds=estimate,limit_seconds=3600,full_authorized=estimate<=3600,
          method='prepare + worst fixed32 wall time/image * queries *1.5; includes3seed matching; no speed claim'))
    save('repro_and_correctness.json',dict(checks=checks,repro=repro,batch_consistency=consistency,passed=True))
    save('PREPARED.json',dict(seconds=elapsed,peak_bytes=torch.cuda.max_memory_allocated(),timestamp=time.time()))
    print(f'[prepared] elapsed={elapsed:.1f}s conservative estimate={estimate:.1f}s; allowed={estimate<=3600}',flush=True)


def stats(x):
    a=np.asarray(x,dtype=np.float64).reshape(-1)
    return dict(n=len(a),mean=float(a.mean()),std=float(a.std(ddof=0)),min=float(a.min()),max=float(a.max()),
                quantiles=dict(zip(['p0','p5','p25','p50','p75','p95','p100'],np.quantile(a,[0,.05,.25,.5,.75,.95,1]).tolist())))


def auroc(d,n):
    # Historical implementation rejects constants; the mathematically defined tied AUROC is .5.
    value,flag=D.auroc_avg_rank(d,n)
    if flag=='constant_score': return .5
    assert flag=='ok',flag
    return value


def full(gpu_budget_hours=None):
    frozen=verify_frozen(); assert read(OUT/'repro_and_correctness.json')['passed']
    estimate=read(OUT/'cost_estimate.json')
    if gpu_budget_hours is None:
        assert estimate['full_authorized'], 'estimate exceeds1GPUhour; user must explicitly supply --gpu-budget-hours after reviewing HANDOFF.md'
    else:
        assert np.isfinite(gpu_budget_hours) and gpu_budget_hours > 0
        assert gpu_budget_hours*3600 >= estimate['estimated_total_gpu_reservation_seconds'], 'explicit budget below conservative estimate'
        save('USER_RUN_AUTHORIZATION.json',dict(timestamp=time.time(),argv=sys.argv,
             gpu_hours=gpu_budget_hours,wall_clock_h=gpu_budget_hours+2,
             original_gpu_hours=1,estimate_seconds=estimate['estimated_total_gpu_reservation_seconds'],
             scope='manual user launch budget amendment only; scoring and frozen inputs unchanged'))
        active=read(ROOT/'registry/active_task.json')
        active['budget']=dict(gpu_hours=gpu_budget_hours,wall_clock_h=gpu_budget_hours+2,
             authorization='outputs/M3nve-residual/USER_RUN_AUTHORIZATION.json')
        active['started_at']=time.time()
        (ROOT/'registry/active_task.json').write_text(json.dumps(active,ensure_ascii=False,indent=2)+'\n')
    assert not (OUT/'nve_eval.json').exists(), 'refuse overwriting full results'
    t0=sync_time(); lists=read(HIST/'SUPPORT_LISTS.json')
    paths=read(OUT/'cache/paths.json'); index={p:i for i,p in enumerate(paths)}
    mapped=np.load(OUT/'cache/mapped.npy',mmap_mode='r'); banks=np.load(OUT/'cache/banks.npz')
    manifest=read(OUT/'BANK_MANIFEST.json')
    for name, record in read(OUT/'CACHE_MANIFEST.json').items():
        assert sha(OUT/'cache'/name)==record['sha256']
    records=[]; mechanisms={}; visuals=[]; group_times={}; bytes_extra=0
    for c in configurations(lists):
        assert not (ROOT/'registry/STOP').exists()
        gt=sync_time(); n=torch.as_tensor(banks[c['key']+'_N'],device=DEV)
        fs=[torch.as_tensor(banks[cfg_name(c,s)+'_F'],device=DEV) for s in range(3)]
        fls=[local(f,n) for f in fs]
        for seed,(f,fl) in enumerate(zip(fs,fls)):
            name=cfg_name(c,seed)
            # Both representations saved without dropping any vector.
            residuals=np.stack([fl[1].cpu().numpy(),clean(fl[1],fl[2],.8).cpu().numpy()])
            np.save(OUT/'cache'/f'residual_{name}.npy',residuals); bytes_extra+=residuals.nbytes
        pathsq=query_paths(c); nd=len(c['query']['defect']); vals=[[] for _ in fs]
        nr=[]; nc=[]; ranks=[]; fstats=[]
        for fl in fls:
            fstats.append(dict(raw_norm=stats(fl[1].norm(dim=1).cpu().numpy()),
                         clean_norm=stats(clean(fl[1],fl[2],.8).norm(dim=1).cpu().numpy()),
                         r_eff=np.bincount(fl[3].cpu().numpy(),minlength=5).tolist(),
                         zero_residuals=int((fl[1].norm(dim=1)==0).sum())))
        for start in range(0,len(pathsq),32):
            q=get_features(mapped,index,pathsq[start:start+32]).reshape(-1,128)
            dn,rq,pq,rank,_=local(q,n)
            raw_norm=rq.norm(dim=1); clean_norm=clean(rq,pq,.8).norm(dim=1)
            assert bool((clean_norm<=raw_norm+2e-6).all())
            nr.append(raw_norm.reshape(-1,1600).cpu().numpy()); nc.append(clean_norm.reshape(-1,1600).cpu().numpy())
            ranks.append(rank.reshape(-1,1600).cpu().numpy())
            for seed,(f,fl) in enumerate(zip(fs,fls)):
                a=(q@f.T).max(-1).values-(1-dn)
                b=residual_score(dn,rq,pq,fl[1],fl[2],0.)
                cc=residual_score(dn,rq,pq,fl[1],fl[2],.8)
                scores=torch.stack([image_score(v) for v in [a,b,cc]],dim=1).cpu().numpy()
                assert np.isfinite(scores).all()
                vals[seed].append(scores)
                if seed==0 and c['ref']==3001:
                    selected=frozen['visual_selection'][c['key']]
                    for offset,path in enumerate(pathsq[start:start+32]):
                        side=next((s for s in selected if path in selected[s]),None)
                        if side is not None:
                            maps=np.stack([v.reshape(-1,40,40)[offset].cpu().numpy() for v in [a,b,cc,raw_norm,clean_norm]])
                            visual_id=f"{c['cc']}_{c['g']}_{side}_{selected[side].index(path)}"
                            np.save(OUT/'cache'/f'vis_{visual_id}.npy',maps)
                            visuals.append(dict(id=visual_id,path=path,side=side,config=cfg_name(c,0),scores=scores[offset].tolist()))
        norms0=np.concatenate(nr); norms1=np.concatenate(nc); rankall=np.concatenate(ranks)
        mechanisms[c['key']]={'fault_banks':fstats}
        for side,sl in [('defect',slice(0,nd)),('normal',slice(nd,None))]:
            x=norms0[sl]; y=norms1[sl]; rr=rankall[sl]
            mechanisms[c['key']][side]=dict(raw_norm=stats(x),clean_norm=stats(y),norm_change=stats(y-x),
                  ratio_nonzero=stats((y/np.maximum(x,1e-8))[x>0]),
                  zero_residual_count=int((x==0).sum()),zero_clean_count=int((y==0).sum()),
                  zero_rank_count=int((rr==0).sum()),r_eff=np.bincount(rr.reshape(-1),minlength=5).tolist(),
                  patch_label='patches from '+side+' images, NOT pixel anomaly labels')
        np.savez(OUT/'cache'/f'norms_{c["key"]}.npz',raw=norms0,clean=norms1,rank=rankall)
        pc=read(HIST/'raw_scores_patchcore'/f"{c['key']}.json")
        assert all(set(pc[s])==set(c['query'][s]) for s in ['defect','normal'])
        upc=auroc(list(pc['defect'].values()),list(pc['normal'].values()))
        for seed,rows in enumerate(vals):
            name=cfg_name(c,seed); arr=np.concatenate(rows); assert arr.shape==(len(pathsq),3)
            oldA=read(OUT/'cache'/f'A_{name}.json')
            diff=float(np.max(np.abs(arr[:,0]-np.array([oldA[p] for p in pathsq]))))
            assert diff<=1e-5,('A full/prep mismatch',diff)
            record=dict(config=name,combo=c['combo'],budget=c['g'],seed=seed,ref=c['ref'],n_defect=nd,n_normal=len(pathsq)-nd,
                        A=auroc(arr[:nd,0],arr[nd:,0]),B=auroc(arr[:nd,1],arr[nd:,1]),C=auroc(arr[:nd,2],arr[nd:,2]),PatchCore=upc)
            for lhs,rhs in [('C','B'),('B','A'),('C','A'),('A','PatchCore'),('B','PatchCore'),('C','PatchCore')]:
                record[lhs+'-'+rhs]=record[lhs]-record[rhs]
            records.append(record)
            save(f'raw_scores/{name}.json',dict(config=record,columns=['A','B','C'],
                 defect={p:arr[i].tolist() for i,p in enumerate(pathsq[:nd])},
                 normal={p:arr[nd+i].tolist() for i,p in enumerate(pathsq[nd:])}))
            mechanisms[c['key']][f'seed{seed}_image_score_changes']={side:{label:stats(arr[sl,i]-arr[sl,j]) for label,i,j in [('C-B',2,1),('B-A',1,0),('C-A',2,0)]} for side,sl in [('defect',slice(0,nd)),('normal',slice(nd,None))]}
        assert tensor_sha(n)==manifest[c['key']+'_N']['sha256']
        assert all(tensor_sha(f)==manifest[cfg_name(c,s)+'_F']['sha256'] for s,f in enumerate(fs))
        group_times[c['key']]=sync_time()-gt
        print(f'[full] {c["key"]} 3 seeds complete, {group_times[c["key"]]:.1f}s',flush=True)
    assert len(records)==60
    with open(OUT/'paired_configs.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0])); w.writeheader(); w.writerows(records)
    summary={}
    for combo in M6.COMBOS:
        for g in ['g1','g2']:
            rs=[r for r in records if r['combo']==combo and r['budget']==g]
            cell={}
            for field in ['A','B','C','PatchCore','C-B','B-A','C-A','A-PatchCore','B-PatchCore','C-PatchCore']:
                refs={str(j):float(np.mean([r[field] for r in rs if r['ref']==j])) for j in range(3001,3006)}
                v=list(refs.values())
                cell[field]=dict(mean=float(np.mean(v)),std=float(np.std(v,ddof=0)),range=[min(v),max(v)],
                    positive_groups=sum(x>0 for x in v),negative_configs=sum(r[field]<0 for r in rs),per_ref=refs,
                    raw15_std=float(np.std([r[field] for r in rs],ddof=0)))
            summary[combo+'_'+g]=cell
    verify_frozen()
    elapsed=sync_time()-t0
    save('MECHANISMS.json',mechanisms); save('VISUAL_SELECTION.json',visuals)
    save('nve_eval.json',dict(configs_completed=60,arms_completed=180,all_checks_passed=1,
         per_config=records,summary=summary,exploratory=True,full_seconds=elapsed))
    prep=read(OUT/'PREPARED.json')
    cache_bytes=sum(p.stat().st_size for p in (OUT/'cache').iterdir() if p.is_file())
    save('RESOURCE_COST.json',dict(prepare_seconds=prep['seconds'],full_seconds=elapsed,
         measured_compute_workflow_seconds=prep['seconds']+elapsed,group_seconds=group_times,
         peak_allocated_bytes=max(prep['peak_bytes'],torch.cuda.max_memory_allocated()),
         peak_reserved_bytes=torch.cuda.max_memory_reserved(),cache_total_bytes=cache_bytes,
         mapped_feature_payload_bytes=mapped.nbytes,normal_fault_bank_payload_bytes=sum(v['bytes'] for v in manifest.values()),
         added_residual_bank_payload_bytes=bytes_extra,local_query_projection_reused_across_seeds=True,
         query_svd_normal_configurations=20,actual_image_batch=32,svd_patch_chunk=2048,
         timing_scope='cached feature preparation + scoring; NOT complete image inference or deployment benchmark'))
    print('[full] completed all 60 configurations',flush=True)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('stage',choices=['prepare','full'])
    ap.add_argument('--gpu-budget-hours',type=float,default=None,
                    help='Manual user authorization to extend the original 1h budget after reviewing cost_estimate.json')
    args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    failure_path=OUT/'FAILURES.json'
    failures=read(failure_path) if failure_path.exists() else []
    assert len(failures)<2, 'two failures; stop for human diagnosis'
    guard(); torch.cuda.reset_peak_memory_stats()
    log=open(OUT/f'{args.stage}.log','a',buffering=1)
    class Tee:
        def __init__(self,orig): self.orig=orig
        def write(self,s): self.orig.write(s); log.write(s)
        def flush(self): self.orig.flush(); log.flush()
    sys.stdout=Tee(sys.stdout); sys.stderr=Tee(sys.stderr)
    try:
        if args.stage=='prepare': prepare()
        else: full(args.gpu_budget_hours)
    except Exception:
        failure=dict(stage=args.stage,timestamp=time.time(),traceback=traceback.format_exc())
        failures.append(failure); save('FAILURES.json',failures)
        quarantine=OUT/'quarantine'/f'failure_{len(failures)}'; quarantine.mkdir(parents=True,exist_ok=True)
        (quarantine/'DIAGNOSIS.json').write_text(json.dumps(failure,indent=2))
        # A failed full stage never contributes official score or summary files.
        if args.stage=='full':
            for name in ['raw_scores','nve_eval.json','paired_configs.csv','MECHANISMS.json','VISUAL_SELECTION.json','RESOURCE_COST.json']:
                p=OUT/name
                if p.exists(): p.rename(quarantine/name)
        raise


if __name__=='__main__': main()
