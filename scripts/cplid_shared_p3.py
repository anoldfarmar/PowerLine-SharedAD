#!/usr/bin/env python3
"""Freeze then user-run R0/R1/R2 on CPLID. Never train or tune on test labels."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
import torch
import yaml

from insplad_freeze_utils import INIT, INIT_SHA, file_sha, json_write
from cplid_shared_p3_core import Pipeline, affine, iou, match, nearest, score, state_hash, to_input

ROOT=Path(__file__).resolve().parents[1]
BASE=Path('/root/autodl-tmp/version10-12')
TASK='CPLID-sharedP3-R0R1R2-v1'
CODE=['cplid_shared_p3.py','cplid_shared_p3_core.py','cplid_shared_p3_launch.sh',
      'insplad_freeze_utils.py','m3score_pcverify.py']


def read(path):
    return json.loads(Path(path).read_text())


def digest(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def xml_boxes(path):
    return [[float(obj.findtext('bndbox/'+k)) for k in ['xmin','ymin','xmax','ymax']]
            for obj in ET.parse(path).getroot().findall('object')]


def immutable(path,obj):
    if path.exists():
        assert read(path)==obj,f'Frozen identity changed: {path}; do not overwrite/reuse old cache'
    else:json_write(path,obj)


def prepare(args):
    d=args.detector_dir
    a=read(d/'DETECTOR_ACCEPTANCE.json');assert a['passed']
    frozen=read(d/'FROZEN_INPUTS.json');det=d/'runs/frozen_p3/weights/best.pt'
    assert file_sha(det)==a['weights']['best']['sha256'] and file_sha(INIT)==INIT_SHA
    for p,h in a['source_hashes'].items():assert file_sha(p)==h
    prior=args.prior
    supports=read(prior/'SUPPORT_LISTS.json')['lists']
    train={Path(r['path']).name:r for r in frozen['items']['train']}
    support={}
    for ref,entry in sorted(supports.items()):
        rows=[]
        for r in entry['entries']:
            im=train[Path(r['image_path']).name]
            assert r['image_sha256']==im['sha256'] and r['source_group']==im['source_group']
            b=[float(v) for v in r['box']]
            assert b in im['boxes']
            rows.append(dict(path=im['path'],sha256=im['sha256'],source_group=im['source_group'],
                             box=b,GT_index=im['boxes'].index(b)))
        assert len(rows)==10 and len({r['path'] for r in rows})==10 and len({r['source_group'] for r in rows})==10
        support[ref]=rows
    assert list(support)==[f'ref{r}' for r in range(3001,3006)]
    queries=[]
    for r in frozen['items']['test']:
        queries.append(dict(id='normal/'+Path(r['path']).stem,path=r['path'],sha256=r['sha256'],
                            size=[r['width'],r['height']],boxes=r['boxes'],labels=[0]*len(r['boxes']),
                            source_group=r['source_group'],source_independence='unknown beyond exact duplicates'))
    historical=[json.loads(l) for l in (prior/'DATA_MANIFEST.jsonl').read_text().splitlines() if l.strip()]
    fault_xml={}
    for r in sorted((r for r in historical if r.get('side')=='defective'),key=lambda r:r['path']):
        p=args.data/'Defective_Insulators/images'/Path(r['path']).name
        ins=args.data/'Defective_Insulators/labels/insulator'/(p.stem+'.xml')
        defect=args.data/'Defective_Insulators/labels/defect'/(p.stem+'.xml')
        assert file_sha(p)==r['sha256']
        bs=xml_boxes(ins);ds=xml_boxes(defect)
        assert bs==r['ins_boxes'] and ds==r['def_boxes'] and bs and ds
        with Image.open(p) as image:size=list(image.size)
        assert size==[r['width'],r['height']]
        labels=[]
        for b in bs:
            labels.append(int(any(b[0]<=x[0]<x[2]<=b[2] and b[1]<=x[1]<x[3]<=b[3] for x in ds)))
        assert all(any(b[0]<=x[0]<x[2]<=b[2] and b[1]<=x[1]<x[3]<=b[3] for b in bs) for x in ds)
        fault_xml[str(ins)]=file_sha(ins);fault_xml[str(defect)]=file_sha(defect)
        queries.append(dict(id='defective/'+p.stem,path=str(p),sha256=r['sha256'],size=size,
                            boxes=bs,defect_boxes=ds,labels=labels,source_group=r['group_id'],
                            source_independence='synthetic; foreground/background provenance unresolved'))
    # Validate all roles by content; synthetic ancestry remains unknown, never infer independence from different SHA.
    for rows in frozen['items'].values():
        for r in rows:
            assert file_sha(r['path'])==r['sha256'] and file_sha(r['xml'])==r['xml_sha256']
    training={r['sha256'] for r in frozen['items']['train']+frozen['items']['val']}
    assert not training.intersection(r['sha256'] for r in queries)
    assert len({r['sha256'] for rows in support.values() for r in rows}.intersection(r['sha256'] for r in queries))==0
    protocol=dict(task=TASK,detector=str(det),detector_sha256=file_sha(det),coco=str(INIT),coco_sha256=INIT_SHA,
        detector_input=dict(size=[1280,1280],geometry='RGB LetterBox auto=False, pad114, bilinear OpenCV resize',
                            conf=.25,nms_iou=.7,max_det=300,classes=[0],frame_batch=1),
        features=dict(layer=4,channels=64,stride=8,channel_L2=True,crop_size=[320,320],
                      crop_integer='floor left/top, ceil right/bottom, clipped to image',
                      crop_resize='PIL bilinear RGB',roi_size=[40,40],aligned=True,sampling_ratio=1),
        bank=dict(shots='10 fixed normal training files/groups/GT objects per reference; one preselected object per file',
                  support_geometry='GT-anchored for ALL arms: R0/R1 GT crop; R2 GT ROI from full-image forward',
                  coreset='official anomalib KCenterGreedy ratio0.1 seed7001; independently seeded per path/ref',
                  R0_R1='identical support feature path; verified shared bank, no old bank reused'),
        query=dict(R0='GT native crop -> Resize320 -> original COCO prefix0..4',
                   R1='new detector -> native predicted crop -> Resize320 -> extra COCO prefix0..4',
                   R2='same new detector and same predictions -> same-forward P3 -> RoIAlign'),
        scoring=dict(distance='exact FP32 Euclidean on channel-normalized patches, official PatchCore distance',
                     reduction='mean largest8 patch 1NN distances',query_chunk=2048,bank_chunk=512),
        matching=dict(iou=.5,rule='confidence-descending greedy one-to-one whole-insulator GT matching',
                      unmatched='unassigned prediction; scored and counted as FP, never assigned fault label',
                      no_box='unlocalized; no fallback, no anomaly score; missed GT stay in coverage denominator'),
        evaluation=dict(conditional='R1/R2 AUROC over identical matched GT objects; R0 also reported on same subset',
                        whole_image_AUROC=False,threshold_tuning=False,alarm_recall='not defined without independent calibration',
                        interpretation='coverage is localization recall, not thresholded fault-diagnosis recall'),
        timing=dict(reference='ref3001 banks only, fixed representative; all queries and boxes',
                    rounds=3,warmup=5,order='alternate R1/R2 by frame+round',frame_batch=1,crop_batch=32,
                    boundary='decoded PIL RGB -> preprocessing/transfer/detection/NMS/crop-or-ROI/NN -> boxes,scores,status',
                    decode='separate',throughput='sequential frame_batch1 throughput only; no larger-frame-batch claim',
                    profiling='one separate all-frame stage-profile pass, not added to independent end-to-end timing'),
        numeric=dict(FP32=True,TF32=False,seed=0,atol=1e-4,rtol=1e-4),
        no_training=True,no_new_layers=True,no_parameter_search=True,
        limits='CPLID defects synthetic; unknown shared source ancestry; R0 is a reference, not guaranteed upper bound')
    sources={str(d/f):file_sha(d/f) for f in ['FROZEN_INPUTS.json','DETECTOR_ACCEPTANCE.json','FREEZE_CORRECTNESS.json']}
    sources.update({str(prior/f):file_sha(prior/f) for f in ['SUPPORT_LISTS.json','DATA_MANIFEST.jsonl','SPLIT_MANIFEST.json']})
    sources.update(fault_xml)
    all_inputs=dict(protocol=protocol,supports=support,queries=queries,
                    training_probes=frozen['probe_crops'],source_sha256=sources,
                    code_sha256={n:file_sha(ROOT/'scripts'/n) for n in CODE},
                    repository_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    # Repository commit is provenance, not a cache dependency; adding this task commit does not invalidate inputs.
    stable={k:v for k,v in all_inputs.items() if k!='repository_commit'}
    all_inputs['identity']=digest(stable)
    args.out.mkdir(parents=True,exist_ok=True)
    path=args.out/'FROZEN_INPUTS.json'
    if path.exists():
        assert read(path)['identity']==all_inputs['identity'],'Existing frozen protocol differs; use a new version'
    else:json_write(path,all_inputs)
    json_write(args.out/'PROTOCOL.json',protocol)
    lines=['# Frozen CPLID R0/R1/R2 protocol','',
        '| Arm | Query | Normal support bank |','|---|---|---|',
        '| R0 | GT native crop, Resize320, COCO P3 | same10 GT crops |',
        '| R1 | predicted native crop, Resize320, extra COCO prefix | same10 GT crops |',
        '| R2 | same prediction, same-forward full-frame P3, ROI40 | same10 GT ROIs from full-frame P3 |','',
        'GT-anchored support construction is deliberate and frozen: no supports dropped/reselected because detector misses.',
        'R1 support geometry is GT, query geometry is predicted; the comparison isolates query localization and full-frame representation, not a fully automatic support selection system.',
        'Five original ref3001–3005 lists; 15 arm/reference evaluations, NOT15 detector trainings or independent datasets.',
        'Full images use1280 square LetterBox, native crops320; layer4 C64 stride8 only. Exact settings: PROTOCOL.json.',
        'No-box: unlocalized, score null, no fallback. Missed faulty GT remain in localization coverage denominator.',
        'Unmatched predictions count as FP and have unknown anomaly label. Do not label every box on a faulty frame as faulty.',
        'AUROC is per GT object, conditional on matching for R1/R2; R0 all-GT and same-matched-subset both reported.',
        'No test-derived alarm threshold or whole-image AUROC. Fault localization coverage is NOT fault diagnosis recall.',
        'Timing: ref3001 only, all queries, three alternating rounds plus separate profiling; batch1 sequential throughput.',
        'R0 is not a complete-system speed baseline. R1/R2 run actual detector and retrieval, no feature cache during timing.',
        'GT crops use boxes (floor/ceil clipping); regression gate checks different boxes on same image give different features.',
        'Source groups only establish exact-duplicate isolation, not proven capture independence; faulty images are synthetic.',
        'No training, adapters, P2, score changes or tuning. GPU execution only on explicit user launch.', '']
    (args.out/'METHOD_SPEC.md').write_text('\n'.join(lines))
    contract=dict(task=TASK,title='Frozen CPLID R0/R1/R2 user-run evaluation',gate_required=None,
                  authorized_by='User requested detector acceptance then frozen next-stage launch instructions',
                  scope=dict(training=False,evaluation='same detector R0/R1/R2, fixed5supportlists and whole-frame timing'),
                  acceptance=[dict(type='metric_min',path=str(args.out/'REPRO_AND_CORRECTNESS.json'),key='passed',min=1),
                              dict(type='metric_min',path=str(args.out/'metrics.json'),key='completed',min=1),
                              dict(type='csv_min_rows',path=str(args.out/'per_reference_metrics.csv'),min_rows=15),
                              dict(type='file_exists',path=str(args.out/'TIMING_RAW.jsonl')),
                              dict(type='file_exists',path=str(args.out/'REPORT.md'))])
    (args.out/'TASK.yaml').write_text(yaml.safe_dump(contract,sort_keys=False))
    print(json.dumps(dict(status='PROTOCOL_FROZEN_NOT_RUN',identity=all_inputs['identity'],
                          query_frames=len(queries),GT_objects=sum(len(r['boxes']) for r in queries),
                          normal_frames=sum(r['id'].startswith('normal/') for r in queries),
                          references=list(support),output=str(args.out)),indent=2))


def validate(f):
    assert digest({k:v for k,v in f.items() if k not in ['identity','repository_commit']})==f['identity'], 'Frozen manifest content changed'
    for name,h in f['code_sha256'].items():assert file_sha(ROOT/'scripts'/name)==h,f'Frozen code changed: {name}'
    for p,h in f['source_sha256'].items():assert file_sha(p)==h,f'Frozen source changed: {p}'
    p=f['protocol']
    assert file_sha(p['detector'])==p['detector_sha256'] and file_sha(p['coco'])==p['coco_sha256']
    for r in f['queries']+[r for rows in f['supports'].values() for r in rows]:assert file_sha(r['path'])==r['sha256']


class Cache:
    def __init__(self,out,identity):
        self.out=out;self.identity=identity;self.path=out/'CACHE_MANIFEST.json'
        self.manifest=read(self.path) if self.path.exists() else dict(identity=identity,files={})
        assert self.manifest['identity']==identity

    def get(self,key,fn):
        p=self.out/'cache'/(key+'.pt');k=str(p.relative_to(self.out))
        if k in self.manifest['files']:
            assert file_sha(p)==self.manifest['files'][k]
            r=torch.load(p,weights_only=False,map_location='cpu');assert r['identity']==self.identity
            return r['payload']
        assert not p.exists(),f'Uncommitted partial cache {p}; inspect before moving it aside'
        value=fn();p.parent.mkdir(parents=True,exist_ok=True)
        temp=p.with_suffix('.tmp');torch.save(dict(identity=self.identity,payload=value),temp);temp.replace(p)
        self.manifest['files'][k]=file_sha(p);json_write(self.path,self.manifest)
        return value


def seed(n):
    random.seed(n);np.random.seed(n);torch.manual_seed(n);torch.cuda.manual_seed_all(n)


def coreset(embedding):
    from m3score_pcverify import KCenterGreedy
    seed(7001)
    idx=KCenterGreedy(embedding=embedding,sampling_ratio=.1).select_coreset_idxs()
    return embedding[idx],list(map(int,idx))


def close(a,b):
    assert a.shape==b.shape
    assert torch.allclose(a,b,atol=1e-4,rtol=1e-4),f'Correctness mismatch {float((a-b).abs().max())}'
    return float((a-b).abs().max()) if a.numel() else 0.


@torch.inference_mode()
def gates(pipe,f,banks):
    from torchvision.ops import roi_align
    probes=f['training_probes']
    images=[Image.open(r['path']).convert('RGB') for r in probes]
    evidence={}
    # Independent affine and sampling test using a linear coordinate ramp, including non-square native geometry.
    assert affine((1152,864))==(1280/1152,(0,160))
    b=to_input([[0,0,1152,864]],(1152,864));close(b,torch.tensor([[0.,160.,1280.,1120.]]))
    yy,xx=torch.meshgrid(torch.arange(160.,device=pipe.device),torch.arange(160.,device=pipe.device),indexing='ij')
    ramp=(xx+10*yy)[None,None]
    roi=torch.tensor([[0.,80.,240.,400.,560.]],device=pipe.device)
    actual=roi_align(ramp,roi,(40,40),spatial_scale=.125,aligned=True,sampling_ratio=1)[0,0]
    expected=(torch.arange(40.,device=pipe.device)+10)[None,:]+10*(torch.arange(40.,device=pipe.device)+30)[:,None]
    evidence['coordinate_ramp_maxdiff']=close(actual,expected)
    # Prefix batch1 vs formal crop32 (and actual NN scores), no test labels/scores used.
    r=probes[0];im=images[0];w,h=im.size
    boxes=[r['xyxy'],[0,0,w,h],[0,0,w/3,h/3],[w/3,h/3,w,h]]
    single=pipe.crops(im,boxes,batch=1);batched=pipe.crops(im,boxes,batch=32)
    evidence['crop_batch_feature_maxdiff']=close(single,batched)
    assert not torch.allclose(batched[0],batched[1],atol=1e-4,rtol=1e-4),'BBox ignored regression'
    bank=banks['ref3001']['R1']
    evidence['crop_batch_score_maxdiff']=close(torch.tensor(score(single,bank)),torch.tensor(score(batched,bank)))
    q=batched[:1,:37]
    direct=torch.cdist(q.reshape(-1,64).double(),bank.double()).min(1).values
    # nearest() assumes1600 patches, so compare actual complete distance vectors.
    d=nearest(batched[:1],bank)
    evidence['distance_float64_maxdiff']=close(d[0,:37].double(),direct)
    evidence['distance_chunk_maxdiff']=close(d,nearest(batched[:1],bank,query_chunk=701,bank_chunk=127))
    # No-box/multi-box/boundary/portrait paths exercise extraction, not a replacement detection output.
    _,p3=pipe.detect([im])
    multi=pipe.rois(p3,im,boxes)
    individual=torch.cat([pipe.rois(p3,im,[b]) for b in boxes])
    evidence['roi_batch_feature_maxdiff']=close(multi,individual)
    evidence['roi_batch_score_maxdiff']=close(torch.tensor(score(multi,banks['ref3001']['R2'])),
                                           torch.tensor(score(individual,banks['ref3001']['R2'])))
    evidence['empty_crop_shape']=list(pipe.crops(im,[]).shape)
    evidence['empty_roi_shape']=list(pipe.rois(p3,im,[]).shape)
    assert evidence['empty_roi_shape']==[0,1600,64]==evidence['empty_crop_shape']
    for test in [im,im.transpose(Image.Transpose.ROTATE_90)]:
        _,p=pipe.detect([test]);tw,th=test.size
        e=pipe.rois(p,test,[[0,0,tw,th],[0,0,tw/2,th/2],[tw/2,th/2,tw,th]])
        assert e.shape==(3,1600,64) and torch.isfinite(e).all()
    for method in ['R1','R2']:
        a=pipe.run(im,method,banks['ref3001'][method]);b=pipe.run(im,method,banks['ref3001'][method])
        assert a['boxes']==b['boxes']
        evidence[method+'_repeat_feature_maxdiff']=close(a['features'],b['features'])
        evidence[method+'_repeat_score_maxdiff']=close(torch.tensor(a['scores']),torch.tensor(b['scores']))
        evidence[method+'_actual_forward_counts']=a['counters']
    assert pipe.run(im,'R1')['boxes']==pipe.run(im,'R2')['boxes']
    evidence['matching_synthetic']=match([dict(xyxy=[0,0,10,10]),dict(xyxy=[0,0,10,10]),dict(xyxy=[20,20,30,30])],[[0,0,10,10],[20,20,30,30]])
    assert evidence['matching_synthetic']=={0:0,2:1}
    return evidence


def roc(rows):
    from sklearn.metrics import roc_auc_score
    rows=[r for r in rows if r['label'] is not None]
    y=[r['label'] for r in rows];s=[r['score'] for r in rows]
    assert np.isfinite(s).all()
    if len(set(y))<2:return None
    # Independent rank-based AUROC with tie handling, separate from sklearn.
    from scipy.stats import rankdata
    rank=rankdata(s);n1=sum(y);n0=len(y)-n1
    independent=float((sum(rank[i] for i,v in enumerate(y) if v==1)-n1*(n1+1)/2)/(n0*n1))
    result=float(roc_auc_score(y,s));assert abs(result-independent)<1e-12
    return result


def run(args):
    f=read(args.out/'FROZEN_INPUTS.json');validate(f)
    active=read(ROOT/'registry/active_task.json');assert active['task']==TASK
    if (args.out/'metrics.json').exists() and read(args.out/'metrics.json').get('completed'):
        print('Already complete; no rerun. See REPORT.md');return
    assert torch.cuda.is_available(),'Use the configured CUDA environment'
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;seed(0)
    def gpu_competition():
        import os
        output=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
        others=[int(x.strip()) for x in output.splitlines() if x.strip().isdigit() and int(x.strip())!=os.getpid()]
        assert not others,f'Other CUDA compute processes active: {others}; stop this run and use an idle GPU'
    gpu_competition()
    started=time.perf_counter();p=f['protocol'];cache=Cache(args.out,f['identity'])
    pipe=Pipeline(p['detector'],p['coco']);pipe.sync()
    initial=[state_hash(pipe.model),state_hash(pipe.prefix)]
    startup=time.perf_counter()-started
    def budget():
        assert time.perf_counter()-started<args.max_hours*3600,'User run wall-time cap reached; verified cache is resumable'
    banks={};manifest={};build_start=time.perf_counter()
    for ref,rows in f['supports'].items():
        support_features=[]
        for r in rows:
            budget()
            def build(r=r):
                im=Image.open(r['path']).convert('RGB');_,p3=pipe.detect([im])
                return dict(crop=pipe.crops(im,[r['box']]).cpu(),roi=pipe.rois(p3,im,[r['box']]).cpu())
            support_features.append(cache.get('support/'+digest(r),build))
        banks[ref]={};manifest[ref]={}
        for name,key in [('R0','crop'),('R2','roi')]:
            embedding=torch.cat([r[key] for r in support_features]).reshape(-1,64).to(pipe.device)
            def build_bank():
                b,idx=coreset(embedding);return dict(bank=b.cpu(),indices=idx)
            value=cache.get(f'banks/{ref}_{name}',build_bank)
            bank=value['bank'].to(pipe.device);assert bank.shape==(1600,64) and torch.isfinite(bank).all()
            banks[ref][name]=bank
            if ref=='ref3001':
                again,idx=coreset(embedding)
                assert idx==value['indices'];close(again,bank)
            manifest[ref][name]=dict(supports=rows,input_patches=len(embedding),prototypes=len(bank),
                                     seed=7001,ratio=.1,indices=value['indices'],bytes=bank.numel()*4)
        banks[ref]['R1']=banks[ref]['R0']
        manifest[ref]['R1']=dict(verified_identical_to='R0',support_geometry='GT crop',prototypes=1600)
    pipe.sync();bank_seconds=time.perf_counter()-build_start
    json_write(args.out/'BANK_MANIFEST.json',manifest)
    gate_evidence=gates(pipe,f,banks)
    assert initial==[state_hash(pipe.model),state_hash(pipe.prefix)]
    gate_evidence.update(passed=True,state_unchanged_before_queries=True,bank_repeat='ref3001 R0/R2 exact indices/features',
                         atol=1e-4,rtol=1e-4,formal_frame_batch=1,formal_crop_batch=32)
    json_write(args.out/'REPRO_AND_CORRECTNESS.json',gate_evidence)
    all_rows={ref:[] for ref in banks};coverage=[]
    for index,r in enumerate(f['queries']):
        budget()
        def extract(r=r):
            im=Image.open(r['path']).convert('RGB')
            bs,p3=pipe.detect([im]);pred=[b['xyxy'] for b in bs[0]]
            return dict(boxes=bs[0],R0=pipe.crops(im,r['boxes']).cpu(),R1=pipe.crops(im,pred).cpu(),
                        R2=pipe.rois(p3,im,pred).cpu())
        value=cache.get('query/'+r['id'],extract)
        matches=match(value['boxes'],r['boxes'])
        coverage.append(dict(id=r['id'],GT=len(r['boxes']),fault_GT=sum(r['labels']),
                             predictions=len(value['boxes']),matched=len(matches),
                             matched_fault=sum(r['labels'][j] for j in matches.values()),
                             FP=len(value['boxes'])-len(matches),matched_GT_indices=sorted(matches.values()),
                             GT_input_wh=[[(b[2]-b[0])*affine(tuple(r['size']))[0],(b[3]-b[1])*affine(tuple(r['size']))[0]] for b in r['boxes']],
                             GT_p3_wh=[[(b[2]-b[0])*affine(tuple(r['size']))[0]/8,(b[3]-b[1])*affine(tuple(r['size']))[0]/8] for b in r['boxes']]))
        for ref in banks:
            for arm in ['R0','R1','R2']:
                scores=score(value[arm].to(pipe.device),banks[ref][arm])
                for i,s in enumerate(scores):
                    j=i if arm=='R0' else matches.get(i)
                    all_rows[ref].append(dict(reference=ref,arm=arm,image_id=r['id'],path=r['path'],
                        object_index=i,GT_index=j,label=r['labels'][j] if j is not None else None,
                        score=s,matched=j is not None,xyxy=r['boxes'][i] if arm=='R0' else value['boxes'][i]['xyxy']))
        if index%25==0:print(f'[accuracy] {index+1}/{len(f["queries"])} frames',flush=True)
    json_write(args.out/'COVERAGE.json',dict(frames=coverage,summary={k:sum(r[k] for r in coverage)
                           for k in ['GT','fault_GT','predictions','matched','matched_fault','FP']},
                           no_box_frames=sum(r['predictions']==0 for r in coverage)))
    metrics=[];coverage_by_id={r['id']:r for r in coverage}
    for ref,rows in all_rows.items():
        json_write(args.out/'raw_scores'/(ref+'.json'),rows)
        for arm in ['R0','R1','R2']:
            rs=[r for r in rows if r['arm']==arm]
            paired=[r for r in rs if r['GT_index'] in coverage_by_id[r['image_id']]['matched_GT_indices']]
            metrics.append(dict(reference=ref,arm=arm,AUROC=roc(rs),paired_AUROC=roc(paired),
                                labeled_objects=sum(r['label'] is not None for r in rs),paired_objects=len(paired)))
    with (args.out/'per_reference_metrics.csv').open('w',newline='') as fp:
        writer=csv.DictWriter(fp,fieldnames=list(metrics[0]));writer.writeheader();writer.writerows(metrics)
    timing(args,pipe,f,banks,coverage,budget,gpu_competition)
    assert initial==[state_hash(pipe.model),state_hash(pipe.prefix)],'Inference mutated model/BN'
    gate_evidence['state_unchanged_after_all_inference']=True
    json_write(args.out/'REPRO_AND_CORRECTNESS.json',gate_evidence)
    json_write(args.out/'RESOURCE_COST.json',dict(startup_seconds=startup,bank_build_or_verified_load_seconds=bank_seconds,
        wall_seconds=time.perf_counter()-started,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),GPU=torch.cuda.get_device_name(),
        environment=dict(python=sys.executable,torch=torch.__version__,numpy=np.__version__),
        feature_cache_bytes=sum(x.stat().st_size for x in (args.out/'cache').rglob('*.pt')),
        initial_and_final_state_hashes=initial,bank_geometry='10GT objects,16000patches,1600prototypes per arm/ref'))
    json_write(args.out/'metrics.json',dict(completed=True,configurations=15,metrics=metrics))
    report(args.out)


def timing(args,pipe,f,banks,coverage,budget,gpu_competition):
    # A partial timing run is never appended/resumed as though it were one uninterrupted benchmark.
    path=args.out/'TIMING_RAW.jsonl'
    if path.exists():
        assert (args.out/'SPEED_METRICS.json').exists(),'Partial timing exists: inspect/move timing aside before explicit rerun'
        return
    rows=[];decode=[];stages=[]
    for r in f['queries'][:5]:
        im=Image.open(r['path']).convert('RGB')
        for arm in ['R1','R2']:pipe.run(im,arm,banks['ref3001'][arm])
    for repeat in range(3):
        gpu_competition()
        for i,r in enumerate(f['queries']):
            budget();t=time.perf_counter();im=Image.open(r['path']).convert('RGB');im.load()
            decode.append(dict(id=r['id'],round=repeat,ms=(time.perf_counter()-t)*1000))
            pair=[]
            for arm in (['R1','R2'] if (i+repeat)%2==0 else ['R2','R1']):
                pipe.sync();t=time.perf_counter();v=pipe.run(im,arm,banks['ref3001'][arm]);pipe.sync()
                ms=(time.perf_counter()-t)*1000
                assert len(v['boxes'])==coverage[i]['predictions']
                row=dict(method=arm,image_id=r['id'],boxes=len(v['boxes']),batch=1,crop_batch=32,
                         round=repeat,ms=ms,counters=v['counters'],status=v['status'],reference='ref3001',
                         protocol_identity=f['identity'])
                with path.open('a') as fp:fp.write(json.dumps(row)+'\n')
                rows.append(row);pair.append(v)
            assert pair[0]['boxes']==pair[1]['boxes'],'Arms have different detection outputs'
        print(f'[timing] round {repeat+1}/3 completed',flush=True)
    for r in f['queries']:
        budget();im=Image.open(r['path']).convert('RGB')
        for arm in ['R1','R2']:
            v=pipe.run(im,arm,banks['ref3001'][arm],profile=True)
            stages.append(dict(method=arm,image_id=r['id'],boxes=len(v['boxes']),**v['stages']))
    json_write(args.out/'DECODE_TIMING.json',decode);json_write(args.out/'STAGE_PROFILE.json',stages)
    def summarize(rs):
        if not rs:return None
        ms=np.array([r['ms'] for r in rs]);objects=sum(r['boxes'] for r in rs)
        return dict(mean_ms=float(ms.mean()),median_ms=float(np.median(ms)),P95_ms=float(np.quantile(ms,.95)),
                    sequential_batch1_fps=1000*len(ms)/float(ms.sum()),frame_observations=len(rs),
                    amortized_ms_per_object=float(ms.sum())/objects if objects else None)
    groups={}
    for name,predicate in [('all',lambda n:True),('no_box',lambda n:n==0),('1',lambda n:n==1),
                           ('2-4',lambda n:2<=n<=4),('5+',lambda n:n>=5),('located',lambda n:n>0)]:
        arms={a:summarize([r for r in rows if r['method']==a and predicate(r['boxes'])]) for a in ['R1','R2']}
        arms['speedup']=arms['R1']['mean_ms']/arms['R2']['mean_ms'] if arms['R1'] else None
        groups[name]=arms
    diagnostic={a:float(np.mean([r['diagnostic_ms'] for r in stages if r['method']==a])) for a in ['R1','R2']}
    json_write(args.out/'SPEED_METRICS.json',dict(groups=groups,diagnostic_mean_ms=diagnostic,
        diagnostic_speedup=diagnostic['R1']/diagnostic['R2'],frame_box_distribution=dict(Counter(r['predictions'] for r in coverage)),
        precision='FP32 noTF32',rounds=3,frame_batch=1,bank_reference='ref3001',
        note='Sequential batch1 throughput, not batched-frame throughput. IO/decode excluded and separately saved.'))


def report(out):
    m=read(out/'metrics.json')['metrics'];cov=read(out/'COVERAGE.json');s=read(out/'SPEED_METRICS.json')
    lines=['# CPLID R0/R1/R2 results','', '| Arm | mean AUROC | std(ddof=0) | min–max | paired mean |',
           '|---|---:|---:|---:|---:|']
    for arm in ['R0','R1','R2']:
        rs=[r for r in m if r['arm']==arm];v=np.array([r['AUROC'] for r in rs])
        lines.append(f"| {arm} | {v.mean():.6f} | {v.std():.6f} | {v.min():.6f}–{v.max():.6f} | {np.mean([r['paired_AUROC'] for r in rs]):.6f} |")
    delta=[next(r['AUROC'] for r in m if r['reference']==ref and r['arm']=='R2')-
           next(r['AUROC'] for r in m if r['reference']==ref and r['arm']=='R1') for ref in sorted({r['reference'] for r in m})]
    lines += ['',f'R2−R1 paired reference differences: {delta}; mean {np.mean(delta):.6f}; positive/negative/zero {sum(x>0 for x in delta)}/{sum(x<0 for x in delta)}/{sum(x==0 for x in delta)}.',
              f'Localization coverage counts (not alarm recall): {cov["summary"]}; no-box frames {cov["no_box_frames"]}.',
              '', '| Timing scope | R1 mean ms | R2 mean ms | full speedup |','|---|---:|---:|---:|']
    for name,arms in s['groups'].items():
        if arms['R1']:lines.append(f"| {name} | {arms['R1']['mean_ms']:.3f} | {arms['R2']['mean_ms']:.3f} | {arms['speedup']:.3f}x |")
        else:lines.append(f'| {name} | empty | empty | — |')
    lines += ['',f'Diagnostic-only speedup {s["diagnostic_speedup"]:.3f}x; this is NOT full-system speedup.',
              'P95, medians and sequential batch1 throughput: SPEED_METRICS.json. No larger-frame-batch throughput measured.',
              'R1/R2 actual forward counters in TIMING_RAW.jsonl; R2 has no extra crop prefix. R0 is not system timing baseline.',
              'Normal support objects GT-anchored for all arms. Conditional AUROC excludes unmatched GT but coverage retains them.',
              'No-box is unlocalized, never normal; no alarm threshold selected; no whole-image AUROC or pixel localization claim.',
              'Synthetic defects and unresolved source ancestry limit generalization/causal claims; five references are not independent datasets.',
              'Interpret next steps from accuracy loss and object P3 sizes in COVERAGE.json; do not automatically add P2/adapters/training.', '']
    (out/'REPORT.md').write_text('\n'.join(lines))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--mode',choices=['prepare','run'],required=True)
    ap.add_argument('--out',type=Path,default=BASE/TASK)
    ap.add_argument('--detector-dir',type=Path,default=BASE/'CPLID-frozenP3-det-v1')
    ap.add_argument('--data',type=Path,default=Path('/root/autodl-tmp/data/CPLID'))
    ap.add_argument('--prior',type=Path,default=BASE/'CPLID-T0T1T2-detector-top8-v1')
    ap.add_argument('--max-hours',type=float,default=2)
    args=ap.parse_args()
    if args.mode=='prepare':prepare(args)
    else:
        with torch.inference_mode():run(args)


if __name__=='__main__':
    main()
