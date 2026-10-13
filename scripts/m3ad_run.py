#!/usr/bin/env python3
"""M3BASELINE-AnomalyDINO-v1: frozen official components, resume by verified artifacts."""
import os
os.environ['OMP_NUM_THREADS']='4'
os.environ['FAISS_OPT_LEVEL']='generic'
os.environ['NVIDIA_TF32_OVERRIDE']='0'
os.environ['XFORMERS_DISABLED']='1'
import argparse
from collections import defaultdict
import csv
import hashlib
import importlib
import json
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time
import traceback

import cv2
import faiss
import numpy as np
from PIL import Image
import torch
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from m3ad_model import sha, statehash, load, independent_hf_forward

TASK='M3BASELINE-AnomalyDINO-v1'
ASSETS={'yoke-suspension':'rust','vari-grip':'bird-nest'}
PARAMS=dict(model='dinov2_vits14',short_edge=448,precision='FP32',tf32=False,
    angles=[0,45,90,135,180,225,270,315],masking=False,mask_ref_images=False,
    preprocessing='official get_dataset_info unknown-dataset agnostic_no_mask fallback',
    retrieval='exact IndexFlatL2/GpuIndexFlatL2, normalized then squaredL2/2',k=1,
    score='official mean_top1p: floor(num_total_patches*.01), fallback max if floor=0',
    seed=0,atol=1e-4,rtol=1e-4,batches=[8,16,32],faiss_cpu_threads=4)


def read(p): return json.loads(Path(p).read_text())


def write(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(p)


def arrsha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def now():torch.cuda.synchronize();return time.perf_counter()


def auc(d,n):
    x=np.r_[d,n];r=rankdata(x,method='average'); nd=len(d);nn=len(n)
    u=float((r[:nd].sum()-nd*(nd+1)/2)/(nd*nn))
    v=roc_auc_score(np.r_[np.ones(nd),np.zeros(nn)],x)
    assert abs(u-v)<1e-12
    return u


class Experiment:
    def __init__(self,args):
        self.a=args;self.root=Path(args.root);self.out=Path(args.out);self.out.mkdir(parents=True,exist_ok=True)
        self.cache=self.out/'cache';self.cache.mkdir(exist_ok=True)
        self.mech=Path(args.mech);self.source=read(self.mech/'FROZEN_INPUTS.json')
        self.support=read(self.root/'outputs/M3full-frozen/SUPPORT_LISTS.json')
        self.queries=self.source['queries'];self.weights=Path(args.model)
        self.start=now();self.cost=defaultdict(float);self.hits=0;self.new=0
        self.records={}; self.faiss_device='gpu'
        assert read(self.root/'registry/active_task.json')['task']==TASK
        assert not (self.root/'registry/STOP').exists()
        torch.set_grad_enabled(False);torch.manual_seed(0);np.random.seed(0);torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.cuda.reset_peak_memory_stats();faiss.omp_set_num_threads(4)
        self.wrapper,self.hf,self.identity=load(args.model,args.dino_repo,args.official)
        self.before=statehash(self.wrapper.model)
        from src.utils import augment_image, get_dataset_info
        from src.post_eval import mean_top1p
        from src.detection import run_anomaly_detection
        self.augment=augment_image;self.mean=mean_top1p;self.official_run=run_anomaly_detection
        self.get_info=get_dataset_info
        self.res=faiss.StandardGpuResources();self.res.setTempMemory(256<<20)
        self.identity['state_hash']=self.before

    def freeze(self):
        if (self.out/'FROZEN_INPUTS.json').exists():
            f=read(self.out/'FROZEN_INPUTS.json');assert f['parameters']==PARAMS
            for p,h in f['sha256'].items():assert sha(p)==h,p
            assert f['model_state_hash']==self.before
            return
        files=[self.mech/'FROZEN_INPUTS.json',self.mech/'repro_gates.json',self.mech/'metrics.json',
               self.root/'outputs/M3full-frozen/SUPPORT_LISTS.json',self.root/'data/splits/insplad-fd/query_k5.json',
               self.root/f'plan/tasks/{TASK}.yaml']
        files+=list((self.root/'scripts').glob('m3ad_*.py'))
        files+=list(self.weights.glob('*'))
        files+=[Path(self.a.official)/'run_anomalydino.py']+list((Path(self.a.official)/'src').glob('*.py'))
        files+=list((Path(self.a.dino_repo)/'dinov2').rglob('*.py'))
        images=set();binding={}
        qfile=read(self.root/'data/splits/insplad-fd/query_k5.json')
        for asset,fault in ASSETS.items():
            q=self.queries[asset]
            assert q['defect']==qfile['defect_query'][asset+'/'+fault]
            assert q['normal']==qfile['normal_query'][asset+'/good']
            assert not set(q['normal'])&set(q['defect'])
            images.update(q['normal']+q['defect'])
            for ref in range(3001,3006):
                supp=self.source['supports'][asset][str(ref)]
                assert len(supp)==len(set(supp))==10
                assert supp==self.support[f'{asset}_ref{ref}']['N10']
                assert not set(supp)&set(q['normal']+q['defect'])
                images.update(supp)
                for arm in ['Y-RAW','W-STD']:
                    p=self.mech/'raw_scores/phaseA'/f'{asset}__{fault}_{arm}_ref{ref}.json'
                    old=read(p);assert old['asset']==asset and old['arm']==arm and old['ref']==ref and old['budget']=='10N'
                    for side in ['normal','defect']:assert set(old[side])==set(q[side]),(p,side)
                    files.append(p)
                binding[f'{asset}_ref{ref}']=dict(support=supp,queries=q)
        files+=list(map(Path,sorted(images)))
        hashes={str(p):sha(p) for p in files if p.is_file()}
        for p in images:
            assert p in self.source['file_sha256']
            assert hashes[p].startswith(self.source['file_sha256'][p]),p
        env=dict(python=sys.executable,python_version=sys.version)
        for name in ['torch','torchvision','numpy','faiss','cv2','scipy','sklearn']:
            m=importlib.import_module(name);env[name]=getattr(m,'__version__','unknown')
        env.update(cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),faiss_gpu_count=faiss.get_num_gpus(),
                   FAISS_OPT_LEVEL='generic',OMP_NUM_THREADS=4,NVIDIA_TF32_OVERRIDE='0')
        commits={name:subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True).strip()
                 for name,p in [('AnomalyDINO',self.a.official),('DINOv2',self.a.dino_repo),('experiment',self.root)]}
        write(self.out/'MODEL_IDENTITY.json',self.identity)
        write(self.out/'FROZEN_INPUTS.json',dict(parameters=PARAMS,env=env,commits=commits,sha256=hashes,
              model_state_hash=self.before,binding=binding,source_provenance='M3MECH corrected phaseA, never old mislabeled M3FULL scores',
              timestamp=time.time(),environment_adaptation='FAISS_OPT_LEVEL=generic avoids numpy._core CPU autodetection incompatibility; no package installation',
              dirty_preexisting=['scripts/m3mech_run.py (not modified by this task)','baseline/','paper/']))

    def item(self,path,angle=None):
        tag=dict(path=path,image_sha256=sha(path),angle=angle,model=self.before,short_edge=448,
                 rgb='cv2 BGR2RGB',rotate='official cv2 warpAffine BORDER_DEFAULT' if angle is not None else None)
        key=hashlib.sha256(json.dumps(tag,sort_keys=True).encode()).hexdigest()
        return dict(key=key,tag=tag,path=path,angle=angle)

    def tensor(self,item):
        img=cv2.cvtColor(cv2.imread(item['path'],cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
        if item['angle'] is not None:
            from src.utils import rotate_image
            img=rotate_image(img,item['angle'])
        return self.wrapper.prepare_image(img)

    def file(self,item):return self.cache/'features'/f'{item["key"]}.npy'

    def ready(self,item):
        p=self.file(item);mp=p.with_suffix('.json')
        if not p.exists() or not mp.exists():return False
        meta=read(mp)
        assert meta['tag']==item['tag'] and sha(p)==meta['sha256'],p
        self.records[item['key']]=meta;self.hits+=1
        return True

    def extract(self,items,batch):
        pending=[i for i in items if not self.ready(i)]
        buckets=defaultdict(list)
        for item in pending:
            w,h=Image.open(item['path']).size
            if w<=h:ww=448;hh=int(448*h/w)
            else:hh=448;ww=int(448*w/h)
            buckets[(hh//14*14,ww//14*14)].append(item)
        t=now()
        for shape,group in sorted(buckets.items()):
            for start in range(0,len(group),batch):
                if (self.root/'registry/STOP').exists():raise RuntimeError('human STOP')
                subset=group[start:start+batch];ts=[self.tensor(i) for i in subset]
                assert all(tuple(v.shape[-2:])==shape for v,g in ts)
                a=torch.stack([v for v,g in ts]).cuda()
                ft=now()
                with torch.inference_mode():z=self.wrapper.model.get_intermediate_layers(a)[0].cpu().numpy()
                self.cost['feature_forward_gpu_synchronized_wall_seconds']+=now()-ft
                for item,v,(_,grid) in zip(subset,z,ts):
                    assert v.shape==(grid[0]*grid[1],384) and v.dtype==np.float32 and np.isfinite(v).all()
                    p=self.file(item);p.parent.mkdir(exist_ok=True,parents=True)
                    tmp=p.with_suffix('.tmp.npy');np.save(tmp,v);tmp.replace(p)
                    meta=dict(tag=item['tag'],grid=list(grid),shape=list(v.shape),dtype='float32',bytes=p.stat().st_size,sha256=sha(p))
                    write(p.with_suffix('.json'),meta);self.records[item['key']]=meta;self.new+=1
                print(f'[features] shape={shape}, {min(start+batch,len(group))}/{len(group)}, batch={batch}',flush=True)
        self.cost['feature_pipeline_wall_seconds']+=now()-t

    def features(self,item):return np.load(self.file(item))

    def support_items(self,asset,ref):
        return [self.item(p,a) for p in self.source['supports'][asset][str(ref)] for a in PARAMS['angles']]

    def bank(self,items):
        t=now();a=np.concatenate([self.features(i) for i in items]).astype('float32')
        faiss.normalize_L2(a)
        cfg=faiss.GpuIndexFlatConfig();cfg.device=0;cfg.useFloat16=False
        idx=faiss.GpuIndexFlatL2(self.res,384,cfg);idx.add(a)
        self.cost['bank_concat_normalize_upload_seconds']+=now()-t
        return a,idx

    def score(self,features,index):
        q=features.copy();faiss.normalize_L2(q)
        t=now();ds,_=index.search(q,1);self.cost['faiss_gpu_search_wall_seconds']+=now()-t
        # Exact official L2_normalized branch: FAISS returns squared L2, divide by2.
        dist=(ds/2).squeeze(1).astype(float)
        score=float(self.mean(dist.flatten()))
        assert np.isfinite(dist).all() and np.isfinite(score)
        return score,dist

    def gates(self):
        if (self.out/'REPRO_AND_CORRECTNESS.json').exists():
            c=read(self.out/'REPRO_AND_CORRECTNESS.json');assert c['passed'];return c
        c={};probes=[];bs_times=[]
        # Model equivalence, rectangle/square, all source keys covered before formal scoring.
        paths=sorted({p for a in ASSETS for p in self.source['supports'][a]['3001']})
        rect=next((p for a in ASSETS for p in self.queries[a]['normal'] if Image.open(p).size[0]!=Image.open(p).size[1]),paths[0])
        c['hf_patch_equivalence']=[]
        for p in [paths[0],rect]:
            item=self.item(p);t,grid=self.tensor(item);x=t.unsqueeze(0).cuda()
            with torch.inference_mode():v=self.wrapper.model.get_intermediate_layers(x)[0];r=independent_hf_forward(x,self.hf)
            err=float((v-r).abs().max());ok=torch.allclose(v,r,atol=1e-4,rtol=1e-4)
            c['hf_patch_equivalence'].append(dict(path=p,grid=grid,max_abs_diff=err,allclose=ok));assert ok
        del self.hf
        # Compare feasible image batches; no query AUROC or outcome used.
        t,grid=self.tensor(self.item(paths[0]));base=self.wrapper.extract_features(t)
        c['batch_features']=[]
        for b in [8,16,32]:
            x=t.unsqueeze(0).repeat(b,1,1,1).cuda()
            with torch.inference_mode():self.wrapper.model.get_intermediate_layers(x)
            tm=now()
            with torch.inference_mode():v=self.wrapper.model.get_intermediate_layers(x)[0].cpu().numpy()
            sec=now()-tm
            ok=np.allclose(v,np.broadcast_to(base,v.shape),atol=1e-4,rtol=1e-4)
            c['batch_features'].append(dict(batch=b,seconds=sec,seconds_per_image=sec/b,max_abs_diff=float(np.max(np.abs(v-base))),allclose=ok,
                  peak_allocated_bytes=torch.cuda.max_memory_allocated()))
            bs_times.append((sec/b,b))
            del x,v
        # Batch-consistency rule (frozen before formal run): a probed batch may serve the
        # formal run only if it reproduces batch1 features within atol=rtol=1e-4 on the
        # probe geometry; otherwise batch=1 (compute graph identical to official per-image
        # pipeline). Measured: batched cuBLAS kernels differ from batch1 by ~0.8-1.2e-4
        # (deterministic per shape; same-batch rerun self-diff 0.0) -> batch=1 for formal.
        passing=[r['batch'] for r in c['batch_features'] if r['allclose']]
        batch=passing[0] if passing else 1
        t0=now()
        with torch.inference_mode():one=self.wrapper.extract_features(t)
        c['batch1_seconds_per_image']=now()-t0
        c['batch_rectangle']=[]
        t,grid=self.tensor(self.item(rect));one=self.wrapper.extract_features(t)
        with torch.inference_mode():v=self.wrapper.model.get_intermediate_layers(t.unsqueeze(0).repeat(batch,1,1,1).cuda())[0].cpu().numpy()
        assert np.allclose(v,one,atol=1e-4,rtol=1e-4)
        c['batch_rectangle']=dict(path=rect,batch=batch,max_abs_diff=float(np.max(np.abs(v-one))))
        # Tail rounding/ties and score direction are tested using official functions.
        toy=np.r_[np.zeros(197),1.,2.,2.]
        assert self.mean(toy)==2 and self.mean(np.array([0.,.7]))==.7
        c['tail_rule']=dict(n=200,retained=2,score=float(self.mean(toy)),tiny_score=float(self.mean(np.array([0.,.7]))))
        c['official_parity']=[];c['repeat_bank']=[];c['retrieval']=[];c['batch_scores']=[]
        for asset,fault in ASSETS.items():
            supp=self.support_items(asset,3001)
            qps=[sorted(self.queries[asset][s])[0] for s in ['normal','defect']]
            qi=[self.item(p) for p in qps]
            self.extract(supp+qi,batch)
            bank,index=self.bank(supp);bank2,index2=self.bank(supp)
            assert arrsha(bank)==arrsha(bank2)
            diffs=[]
            for q in qi:
                s,d=self.score(self.features(q),index);s2,d2=self.score(self.features(q),index2)
                assert np.allclose(d,d2,atol=1e-4,rtol=1e-4);diffs.append(float(np.max(np.abs(d-d2))))
            c['repeat_bank'].append(dict(asset=asset,bank_sha256=arrsha(bank),repeat_sha256=arrsha(bank2),score_max_diff=max(diffs)))
            # Genuine batch1 and batchN feature/score gate on same query images.
            for qi0 in qi:
                t,_=self.tensor(qi0);one=self.wrapper.extract_features(t)
                with torch.inference_mode():many=self.wrapper.model.get_intermediate_layers(t.unsqueeze(0).repeat(batch,1,1,1).cuda())[0][0].cpu().numpy()
                s,d=self.score(one,index);s2,d2=self.score(many,index)
                assert np.allclose(one,many,atol=1e-4,rtol=1e-4) and np.allclose(d,d2,atol=1e-4,rtol=1e-4)
                c['batch_scores'].append(dict(path=qi0['path'],feature_diff=float(np.max(np.abs(one-many))),distance_diff=float(np.max(np.abs(d-d2))),score_diff=abs(s-s2)))
            cpu=faiss.IndexFlatL2(384);cpu.add(bank)
            q=self.features(qi[0]).copy();faiss.normalize_L2(q)
            t=now();dc,_=cpu.search(q,1);tc=now()-t
            t=now();dg,_=index.search(q,1);tg=now()-t
            assert np.allclose(dc,dg,atol=1e-4,rtol=1e-4)
            # independent brute-force direct distances, limited query subset, exact full bank
            brute=((q[:8,None,:]-bank[None,:,:])**2).sum(-1).min(-1)
            assert np.allclose(brute,dg[:8,0],atol=1e-4,rtol=1e-4)
            own,_=index.search(bank[:8].copy(),1)
            assert np.max(np.abs(own))<1e-4 and float(dg.mean())>float(own.mean())
            c['retrieval'].append(dict(asset=asset,cpu_seconds=tc,gpu_seconds=tg,cpu_gpu_max_diff=float(np.max(np.abs(dc-dg))),
                brute_force_max_diff=float(np.max(np.abs(brute-dg[:8,0]))),self_distance_max=float(np.max(np.abs(own))),query_mean_distance=float(dg.mean())))
            # Official end-to-end smoke on exactly frozen10 normal paths +2 queries.
            root=self.cache/'official_smoke'/asset
            for k,p in enumerate(self.source['supports'][asset]['3001']):
                dest=root/asset/'train/good'/f'{k:02d}.jpg';dest.parent.mkdir(parents=True,exist_ok=True)
                if not dest.exists():dest.symlink_to(p)
            for side,p in zip(['good',fault],qps):
                dest=root/asset/'test'/side/'query.jpg';dest.parent.mkdir(parents=True,exist_ok=True)
                if not dest.exists():dest.symlink_to(p)
            objs,anoms,masks,rots=self.get_info('InsPLAD','agnostic',str(root))
            assert objs==[asset] and not masks[asset] and rots[asset]
            np.random.seed(0);torch.manual_seed(0)
            scores,bt,it=self.official_run(self.wrapper,asset,str(root),10,anoms,str(root/'plots'),
                save_examples=False,masking=masks[asset],mask_ref_images=False,rotation=rots[asset],
                knn_metric='L2_normalized',knn_neighbors=1,faiss_on_cpu=False,seed=0,save_patch_dists=False,save_tiffs=False)
            errors=[]
            for side,item in zip(['good',fault],qi):
                v,_=self.score(self.features(item),index);old=scores[f'{side}/query.jpg']
                assert np.isclose(v,old,atol=1e-4,rtol=1e-4);errors.append(abs(v-old))
            c['official_parity'].append(dict(asset=asset,score_max_diff=max(errors),official_bank_seconds=bt,
                official_query_seconds=it,supports=10,rotations=8,reference_patch_count=len(bank),masking=False))
            probes.append(dict(asset=asset,bank_patches=len(bank),bank_bytes=bank.nbytes,
                 representative_query_patches=len(q),gpu_search_per_image_seconds=tg,cpu_search_per_image_seconds=tc))
            del bank,bank2,index,index2,cpu
        assert statehash(self.wrapper.model)==self.before
        c['state_hash_before']=self.before;c['state_hash_after']=statehash(self.wrapper.model)
        c['passed']=True;c['formal_batch']=batch
        write(self.out/'REPRO_AND_CORRECTNESS.json',c)
        # Remaining cache footprint and costs from all exact geometries before full execution.
        items=self.all_items();totalpatch=0
        for i in items:
            w,h=Image.open(i['path']).size;totalpatch+=(448//14)*(int(448*max(w,h)/min(w,h))//14)
        rate=c['batch1_seconds_per_image'] if batch==1 else max(x['seconds_per_image'] for x in c['batch_features'])
        estimate=(now()-self.start)+len(items)*rate*3
        estimate+=sum(len(self.queries[p['asset']]['normal']+self.queries[p['asset']]['defect'])*5*p['gpu_search_per_image_seconds']*3 for p in probes)
        disk=totalpatch*384*4+sum(p['bank_bytes']*5 for p in probes)+200*1024**2
        cost=dict(probes=probes,batch_probes=c['batch_features'],selected_batch=batch,estimated_total_seconds=estimate,
            gpu_budget_seconds=7200,allowed=estimate<7200,feature_cache_payload_bytes=totalpatch*384*4,
            estimated_disk_bytes=disk,free_disk_bytes=shutil.disk_usage(self.out).free,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_host_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            estimate_note='conservative3x measured feature/search times; GPU reservation estimate, CPUFAISS separately measured; no complete inference speed claim')
        assert disk<shutil.disk_usage(self.out).free, 'disk estimate exceeds free space'
        write(self.out/'COST_PROBE.json',cost)
        print('[gates] all passed; budget estimate',estimate,'seconds',flush=True)
        return c

    def all_items(self):
        items={}
        for asset in ASSETS:
            for ref in range(3001,3006):
                for item in self.support_items(asset,ref):items[item['key']]=item
            for side in ['defect','normal']:
                for p in self.queries[asset][side]:
                    item=self.item(p);items[item['key']]=item
        return list(items.values())

    def full(self):
        c=read(self.out/'REPRO_AND_CORRECTNESS.json');assert c['passed']
        cost=read(self.out/'COST_PROBE.json');assert cost['allowed'],'estimated budget exceeded; no full'
        items=self.all_items();self.extract(items,c['formal_batch'])
        bankman={};metrics=[]
        for asset,fault in ASSETS.items():
            for ref in range(3001,3006):
                name=f'{asset}__{fault}_ref{ref}';dest=self.out/'raw_scores'/f'{name}.json'
                if dest.exists():
                    data=read(dest);assert data['identity']==self.before and data['ref']==ref
                    assert all(set(data[s])==set(self.queries[asset][s]) for s in ['normal','defect'])
                    bankman[name]=read(self.out/'bank_meta'/f'{name}.json');metrics.append(data['metrics']);continue
                gt=now();supp=self.support_items(asset,ref);bank,index=self.bank(supp)
                bp=self.cache/'banks'/f'{name}.npy';bp.parent.mkdir(exist_ok=True);np.save(bp,bank)
                meta=dict(support_paths=self.source['supports'][asset][str(ref)],original_shots=10,augmented_images=80,
                          angles=PARAMS['angles'],patches=len(bank),candidates=index.ntotal,dimension=384,
                          bank_bytes=bank.nbytes,sha256=arrsha(bank),file_sha256=sha(bp),features=[i['key'] for i in supp],
                          query_used_for_bank=False,device='faiss GPU exact FP32',masking=False)
                assert index.ntotal==len(bank)
                data=dict(identity=self.before,asset=asset,ref=ref,defect={},normal={})
                checkpoints=self.cache/'score_progress'/f'{name}.json'
                if checkpoints.exists():
                    data=read(checkpoints);assert data['identity']==self.before
                for side in ['defect','normal']:
                    for k,p in enumerate(self.queries[asset][side]):
                        if p in data[side]:continue
                        item=self.item(p);score,dist=self.score(self.features(item),index)
                        grid=self.records[item['key']]['grid'];mp=self.cache/'distance_maps'/name/f'{item["key"]}.npy'
                        mp.parent.mkdir(parents=True,exist_ok=True);np.save(mp,dist.reshape(grid))
                        data[side][p]=score
                        if k%50==0:write(checkpoints,data)
                    write(checkpoints,data)
                assert arrsha(bank)==meta['sha256']
                row=dict(asset=asset,ref=ref,AnomalyDINO=auc(list(data['defect'].values()),list(data['normal'].values())))
                for arm,field,label in [('Y-RAW','top8','YOLO_top8'),('W-STD','official_pc','WRN_PC')]:
                    old=read(self.mech/'raw_scores/phaseA'/f'{asset}__{fault}_{arm}_ref{ref}.json')
                    assert all(set(data[s])==set(old[s]) for s in ['normal','defect'])
                    row[label]=auc([v[field] for v in old['defect'].values()],[v[field] for v in old['normal'].values()])
                row['AD_minus_YOLO']=row['AnomalyDINO']-row['YOLO_top8'];row['AD_minus_WRN']=row['AnomalyDINO']-row['WRN_PC']
                data['metrics']=row;write(dest,data);write(self.out/'bank_meta'/f'{name}.json',meta)
                metrics.append(row);bankman[name]=meta
                print('[config]',name,row,'seconds',now()-gt,flush=True)
                del index,bank
        assert len(metrics)==10 and statehash(self.wrapper.model)==self.before
        # Independent reread all output paths, labels, finite scores and AUROC.
        for row in metrics:
            name=f'{row["asset"]}__{ASSETS[row["asset"]]}_ref{row["ref"]}'
            d=read(self.out/'raw_scores'/f'{name}.json')
            assert all(set(d[s])==set(self.queries[row['asset']][s]) and len(d[s])==len(self.queries[row['asset']][s]) for s in ['defect','normal'])
            assert all(np.isfinite(v) for s in ['defect','normal'] for v in d[s].values())
            assert auc(list(d['defect'].values()),list(d['normal'].values()))==row['AnomalyDINO']
        summary={}
        for asset in ASSETS:
            rs=[r for r in metrics if r['asset']==asset];summary[asset]={}
            for field in ['AnomalyDINO','YOLO_top8','WRN_PC','AD_minus_YOLO','AD_minus_WRN']:
                v=[r[field] for r in rs]
                summary[asset][field]=dict(mean=float(np.mean(v)),std=float(np.std(v,ddof=0)),min=min(v),max=max(v),
                                         positive_groups=sum(x>0 for x in v),negative_groups=sum(x<0 for x in v))
        write(self.out/'BANK_MANIFEST.json',bankman)
        write(self.out/'CACHE_MANIFEST.json',dict(features=self.records,format='per path+angle raw final patch features, FP32',
             resume='verify tag and SHA256 before reuse; per configuration every50query checkpoint',items=len(self.records)))
        c['final_model_state_hash']=statehash(self.wrapper.model);c['independent_auroc_configs']=10;c['all_paths_labels_finite']=True
        write(self.out/'REPRO_AND_CORRECTNESS.json',c)
        write(self.out/'metrics.json',dict(configs_completed=10,correctness_passed=1,per_config=metrics,summary=summary))
        with (self.out/'paired_comparisons.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(metrics[0]),lineterminator='\n');w.writeheader();w.writerows(metrics)
        write(self.out/'RESOURCE_COST.json',dict(stages=dict(self.cost),total_session_seconds=now()-self.start,
              peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
              peak_host_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
              cache_bytes=sum(p.stat().st_size for p in self.cache.rglob('*') if p.is_file() and not p.is_symlink()),
              cache_hits=self.hits,new_feature_files=self.new,formal_batch=c['formal_batch'],faiss_device='GPU',
              cpu_faiss_threads=4,gpu_hours_limit=2,notes='synchronized wall times, not CUDA event compute time; CPUFAISS separate in COST_PROBE; no training',
              model_state_unchanged=True))
        self.report()

    def report(self):
        r=read(self.out/'metrics.json');cost=read(self.out/'RESOURCE_COST.json');f=read(self.out/'FROZEN_INPUTS.json')
        lines=['# M3BASELINE-AnomalyDINO-v1','',
          '完成2资产×5参照组=10个新增AnomalyDINO配置。YOLO与WRN复用M3MECH已验证phaseA分数；所有路径、标签与源文件哈希绑定。', '',
          '|资产|YOLO+top8|WRN+官方PC|AnomalyDINO|AD−YOLO|AD−WRN|','|---|---:|---:|---:|---:|---:|']
        for a,s in r['summary'].items():lines.append('|'+a+'|'+'|'.join(f"{s[k]['mean']:.6f}" for k in ['YOLO_top8','WRN_PC','AnomalyDINO','AD_minus_YOLO','AD_minus_WRN'])+'|')
        lines+=['','|资产|方法/配对差|均值|std(ddof=0)|范围|正/负组数|','|---|---|---:|---:|---|---|']
        for a,s in r['summary'].items():
            for k,v in s.items():lines.append(f"|{a}|{k}|{v['mean']:.6f}|{v['std']:.6f}|[{v['min']:.6f},{v['max']:.6f}]|{v['positive_groups']}/{v['negative_groups']}|")
        lines+=['','逐组配对差见paired_comparisons.csv；全部原始逐图分数见raw_scores。AUROC分别用平均秩与sklearn独立核算，非独立显著性检验。','',
          '1. 模型身份：本地HF格式DINOv2-small，ViT-S/14，384维、12层、6头、无register。严格逆转换加载官方模型，所有参数完整，独立split-QKV前向核验最终patch特征；详见MODEL_IDENTITY与REPRO_AND_CORRECTNESS。没有下载新权重。','',
          '2–3. 两资产实际表现和同10N跨参照组差距见主表与逐组CSV；五组不是五个独立数据集，10N只是裁剪文件预算，旋转后80张不是80shot。','',
          f"4. 本次执行墙钟{cost['total_session_seconds']:.1f}s；峰值allocated显存{cost['peak_allocated_bytes']/2**30:.3f}GiB，峰值主机RSS{cost['peak_host_rss_bytes']/2**30:.3f}GiB；缓存{cost['cache_bytes']/2**30:.3f}GiB。各库候选数、字节及SHA见BANK_MANIFEST；特征与检索分段时间见RESOURCE_COST，CPU/GPU FAISS探针分开记录。这不是完整部署吞吐测试。",'']
        for a,s in r['summary'].items():
            d=s['AD_minus_YOLO']
            rec='差距支持下一轮优先研究特征适配，但本轮不执行训练' if d['mean']>0 else 'YOLO均值接近或领先，可考虑下一轮扩展冻结方法的资产范围，但本轮不扩大范围'
            lines+= [f"5–6. {a}：AD−YOLO={d['mean']:+.6f}，{d['positive_groups']}/5组为正、{d['negative_groups']}/5组为负；{rec}。",'']
        lines+= ['7. 无法单独归因：领域预训练、输入短边448与YOLO320几何差异、8角度支持增强、局部特征维数、无容量裁剪的记忆库、mean_top1p与top8/官方PC评分均不同。这是同支持预算的系统比较，不是主干因果实验。','',
          '官方自定义数据集回退为agnostic_no_mask（两资产固定rotation=True、masking=False，支持mask_ref_images=False），在查询分数前冻结，不按资产表现选择。未启用PCA，详见METHOD_SPEC。未报告无GT像素定位准确率、测试集最优F1、阈值或统计显著性。','',
          f"官方AnomalyDINO commit `{f['commits']['AnomalyDINO']}`；官方DINOv2 commit `{f['commits']['DINOv2']}`。官方组件和严格加载/批量/双建库/独立AUROC门均实际执行。未训练新模型、未运行SubspaceAD、未下载Giant。",'',
          '验收以正确完成为准，不要求YOLO获胜。代码、报告和小型分数按仓库流程登记同步；原图、权重、特征/距离缓存不入Git。']
        (self.out/'BASELINE_REPORT.md').write_text('\n'.join(lines)+'\n')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',default='/root/auto-exp');ap.add_argument('--mech',default='/root/autodl-tmp/version10-12')
    ap.add_argument('--out',default='/root/autodl-tmp/version10-12/M3BASELINE-AnomalyDINO-v1')
    ap.add_argument('--model',default='/root/autodl-tmp/model/DINOv2-small')
    ap.add_argument('--official',default='/root/auto-exp/baseline/anomalydino/AnomalyDINO')
    ap.add_argument('--dino-repo',default='/root/autodl-tmp/version10-12/baseline_assets/dinov2')
    ap.add_argument('--stage',choices=['prepare','full','all'],default='all');args=ap.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    failures=read(out/'FAILURES.json') if (out/'FAILURES.json').exists() else []
    assert len(failures)<2,'two formal failures; stop for diagnosis'
    log=(out/'run.log').open('a',buffering=1)
    class Tee:
        def __init__(self,stream):self.stream=stream
        def write(self,s):self.stream.write(s);log.write(s)
        def flush(self):self.stream.flush();log.flush()
    sys.stdout=Tee(sys.stdout);sys.stderr=Tee(sys.stderr)
    try:
        e=Experiment(args);e.freeze()
        if args.stage in ['prepare','all']:e.gates()
        if args.stage in ['full','all']:e.full()
    except Exception:
        failures.append(dict(time=time.time(),stage=args.stage,traceback=traceback.format_exc()))
        write(out/'FAILURES.json',failures);raise


if __name__=='__main__':main()
