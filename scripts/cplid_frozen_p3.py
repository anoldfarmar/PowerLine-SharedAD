#!/usr/bin/env python3
"""Normal-only CPLID localization, strict COCO P3 freezing, user-run training."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
import torch
import yaml
from ultralytics import YOLO

from insplad_freeze_utils import INIT, INIT_SHA, file_sha, json_write, prefix_state, tensor_hash
from insplad_frozen_p3_trainer import FrozenP3Trainer, audit_checkpoint, probe_inputs

ROOT=Path(__file__).resolve().parents[1]
BASE=Path('/root/autodl-tmp/version10-12')
TASK='CPLID-frozenP3-det-v1'
CODE=['cplid_frozen_p3.py','insplad_frozen_p3_trainer.py','insplad_freeze_utils.py']


def read(path):
    return json.loads(Path(path).read_text())


def plain(value):
    if isinstance(value,dict):return {str(k):plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [plain(v) for v in value]
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    return value


def voc_boxes(path,width,height):
    """Same continuous VOC xyxy convention as previous CPLID work, no -1 offset."""
    xml=ET.parse(path).getroot()
    assert int(float(xml.findtext('size/width')))==width
    assert int(float(xml.findtext('size/height')))==height
    raw=[];boxes=[];clipped=0
    for obj in xml.findall('object'):
        assert (obj.findtext('name') or '').strip()=='insulator',f'Unexpected normal XML class: {path}'
        box=[float(obj.findtext('bndbox/'+k)) for k in ['xmin','ymin','xmax','ymax']]
        assert all(math.isfinite(v) for v in box)
        x1,y1,x2,y2=box
        assert x1>=0 and y1>=0 and x2>x1 and y2>y1 and x2<=width+2 and y2<=height+2
        bounded=[max(0.,min(x1,width-1)),max(0.,min(y1,height-1)),min(x2,width),min(y2,height)]
        assert bounded[2]>bounded[0] and bounded[3]>bounded[1]
        clipped+=int(bounded!=box);raw.append(box);boxes.append(bounded)
    assert boxes, f'No insulator GT: {path}'
    return raw,boxes,clipped


def yolo_labels(boxes,width,height):
    return ''.join('0 %.8f %.8f %.8f %.8f\n'%((x1+x2)/(2*width),(y1+y2)/(2*height),(x2-x1)/width,(y2-y1)/height)
                   for x1,y1,x2,y2 in boxes)


def write_identical(path,text):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        assert path.read_text()==text, f'Existing derived file changed: {path}'
    else:path.write_text(text)


def prepare(args):
    import ultralytics
    assert args.out.resolve()!=args.prior.resolve()
    assert file_sha(INIT)==INIT_SHA
    prior_split=args.prior/'SPLIT_MANIFEST.json';prior_data=args.prior/'DATA_MANIFEST.jsonl'
    audit=read(args.prior/'AUDIT_SUMMARY.json')
    assert audit['audit_ok']
    split=read(prior_split)
    records=[json.loads(line) for line in prior_data.read_text().splitlines() if line.strip()]
    normal={r['path']:r for r in records if r.get('side')=='normal'}
    assert len(normal)==600
    previous=read(args.recipe_source/'FROZEN_INPUTS.json')
    for name in ['insplad_frozen_p3_trainer.py','insplad_freeze_utils.py']:
        assert file_sha(ROOT/'scripts'/name)==previous['code_sha256'][name]
    recipe=dict(previous['recipe']);recipe.update(epochs=args.epochs,imgsz=args.imgsz,batch=args.batch)
    assert recipe['freeze']==5 and not recipe['amp'] and not recipe['channels_last']
    names={'train':'train_images','val':'val_images','test':'test_images'}
    groups={s:set(split[s+'_groups']) for s in names}
    assert groups['train'].isdisjoint(groups['val']|groups['test']) and groups['val'].isdisjoint(groups['test'])
    all_paths=[p for key in names.values() for p in split[key]]
    assert len(all_paths)==len(set(all_paths))==600 and set(all_paths)==set(normal)
    items={};counts={};image_hash_sets={};probes=[]
    for side,key in names.items():
        side_items=[];boxes_count=0;clips=0
        for old_path in split[key]:
            rec=normal[old_path]
            assert rec['group_id'] in groups[side]
            image=(args.data/'Normal_Insulators/images'/Path(old_path).name).resolve()
            xml=(args.data/'Normal_Insulators/labels'/Path(rec['xml']).name).resolve()
            assert image.is_file() and xml.is_file()
            assert file_sha(image)==rec['sha256'], f'Changed image while relocating old split: {image}'
            with Image.open(image) as im:width,height=im.size
            assert [width,height]==[rec['width'],rec['height']]
            raw,boxes,nclip=voc_boxes(xml,width,height)
            assert sorted([[round(v,2) for v in b] for b in raw])==sorted(rec['ins_boxes']),f'XML boxes differ from audited source: {xml}'
            side_items.append(dict(path=str(image),xml=str(xml),sha256=rec['sha256'],xml_sha256=file_sha(xml),
                                   width=width,height=height,boxes=boxes,source_group=rec['group_id'],old_path=old_path))
            boxes_count+=len(boxes);clips+=nclip
        items[side]=side_items
        counts[side]=dict(images=len(side_items),groups=len(groups[side]),insulator_boxes=boxes_count,clipped_boxes=clips)
        image_hash_sets[side]={r['sha256'] for r in side_items}
    assert image_hash_sets['train'].isdisjoint(image_hash_sets['val']|image_hash_sets['test'])
    assert image_hash_sets['val'].isdisjoint(image_hash_sets['test'])
    # Probes from training only, including small and large GT boxes; no fault/test selection.
    train_boxes=[dict(path=r['path'],xyxy=box) for r in items['train'] for box in r['boxes']]
    ranked=sorted(train_boxes,key=lambda r:((r['xyxy'][2]-r['xyxy'][0])*(r['xyxy'][3]-r['xyxy'][1]),r['path'],r['xyxy']))
    probes=[ranked[0],ranked[len(ranked)//3],ranked[2*len(ranked)//3],ranked[-1]]
    sources={str(p):file_sha(p) for p in [prior_split,prior_data,args.prior/'AUDIT_SUMMARY.json',args.recipe_source/'FROZEN_INPUTS.json']}
    frozen=dict(task=args.out.name,data_root=str(args.data.resolve()),classes={0:'insulator'},
        recipe=recipe,init=str(INIT),init_sha256=INIT_SHA,counts=counts,items=items,probe_crops=probes,
        prior_split_path=str(prior_split),prior_split_seed=split['seed'],source_sha256=sources,
        code_sha256={n:file_sha(ROOT/'scripts'/n) for n in CODE},
        environment=dict(python=sys.executable,torch=torch.__version__,numpy=np.__version__,ultralytics=ultralytics.__version__),
        policy=dict(training='Normal_Insulators only; full images and all insulator GT boxes',
          checkpoint_selection='best by normal validation detector fitness; last also retained',
          faulty_images_used_in_training=False,faulty_images_used_in_validation=False,
          split='Reuse existing identified-source-group split without resampling; path relocation verified by SHA256',
          head='one-class insulator, not binary normal/defect detector',
          initialization='original local COCO YOLOv8n; NOT prior T1/T2 or InsPLAD fine-tuned checkpoints',
          frozen='modules0-4 parameters and BN, including P3 output; train modules5 onward',
          precision='FP32, no TF32',voc='legacy continuous xyxy, no minus-one offset, clip derived labels only',
          anomaly_evaluation='NOT run in this task',
          historical_reference='Old CPLID T1/T2 used different recipe and full fine-tuning; not a controlled matched comparison'))
    frozen['frozen_id']=hashlib.sha256(json.dumps(frozen,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    args.out.mkdir(parents=True,exist_ok=True)
    manifest=args.out/'FROZEN_INPUTS.json'
    if manifest.exists():assert read(manifest)==plain(frozen),'Frozen data/code/recipe changed; use a new output directory'
    json_write(manifest,frozen)
    for side,rows in items.items():
        for r in rows:
            src=Path(r['path']);dst=args.out/'dataset/images'/side/src.name
            dst.parent.mkdir(parents=True,exist_ok=True)
            if dst.is_symlink():assert dst.resolve()==src
            else:
                assert not dst.exists(),f'Existing non-symlink at {dst}'
                dst.symlink_to(src)
            write_identical(args.out/'dataset/labels'/side/(src.stem+'.txt'),yolo_labels(r['boxes'],r['width'],r['height']))
    data=dict(path=str(args.out/'dataset'),train='images/train',val='images/val',test='images/test',names={0:'insulator'})
    write_identical(args.out/'dataset/data.yaml',yaml.safe_dump(data,sort_keys=False))
    task=dict(task=args.out.name,title='CPLID strict frozen-P3 single-class detector',gate_required=None,
        authorized_by='User requested CPLID detector training like InsPLAD; training is user launched',
        scope=dict(training='Normal-only fixed split; freeze0-4 and BN, train5 onward; no anomaly scoring'),
        acceptance=[dict(type='metric_min',path=str(args.out/'FREEZE_CORRECTNESS.json'),key='passed',min=1),
                    dict(type='metric_min',path=str(args.out/'DETECTION_RESULTS.json'),key='completed',min=1),
                    dict(type='file_exists',path=str(args.out/'DETECTION_REPORT.md'))])
    write_identical(args.out/'TASK.yaml',yaml.safe_dump(task,sort_keys=False))
    json_write(args.out/'DATA_AUDIT.json',dict(passed=True,counts=counts,groups_disjoint=True,exact_image_hashes_disjoint=True,
        matched_legacy_image_hashes=600,xml_boxes_match_legacy_audit=True,defective_images_used=0,
        identity_limit='CPLID defects are synthetic; unknown original-source links are not claimed independent'))
    json_write(args.out/'PREPARE.json',dict(ready=True,counts=counts,gpu_training_started=False,
        train_dataset=str(args.out/'dataset/data.yaml'),code_and_inputs_frozen=True))
    print(json.dumps(dict(ready=True,counts=counts,result_dir=str(args.out)),indent=2),flush=True)
    return frozen


def evaluate(args,frozen):
    run=args.out/'runs/frozen_p3';audit=read(run/'freeze_audit/TRAIN_END.json')
    assert audit['passed']
    audits={name:audit_checkpoint(run/'weights'/f'{name}.pt',frozen['probe_crops'],'cuda:0') for name in ['best','last']}
    json_write(args.out/'FREEZE_CORRECTNESS.json',dict(passed=True,train=audit,checkpoints=audits))
    results={}
    for side in ['val','test']:
        model=YOLO(str(run/'weights/best.pt'))
        metric=model.val(data=str(args.out/'dataset/data.yaml'),split=side,imgsz=frozen['recipe']['imgsz'],
            batch=frozen['recipe']['batch'],device=0,half=False,workers=4,plots=False,save_json=True,
            project=str(args.out/'evaluation'),name=side+'_best',exist_ok=True,verbose=False)
        results[side]=dict(images=frozen['counts'][side]['images'],GT_boxes=frozen['counts'][side]['insulator_boxes'],
            mAP50=float(metric.box.map50),mAP50_95=float(metric.box.map),per_class=plain(metric.summary(decimals=8)),
            predictions=str(args.out/'evaluation'/(side+'_best')/'predictions.json'),
            metric='Native Ultralytics detector AP; do not mix with historical COCO maxDets100 AP')
        json_write(args.out/'DETECTION_RESULTS.json',dict(completed=False,splits=results))
    json_write(args.out/'DETECTION_RESULTS.json',dict(completed=True,splits=results,weights=audits['best']))
    text=['# CPLID strict frozen-P3 detector','',
          'One class: insulator. Normal-only detection training; no defective image was used in training or checkpoint selection.',
          'Original group split reused with SHA256-verified path relocation. Image counts differ from source-group counts.','',
          '| Split | Images | GT insulator boxes | best mAP50 | best mAP50:95 |','|---|---:|---:|---:|---:|']
    for side,r in results.items():text.append(f"| {side} | {r['images']} | {r['GT_boxes']} | {r['mAP50']:.4f} | {r['mAP50_95']:.4f} |")
    text += ['', 'Training recipe: '+json.dumps(frozen['recipe']),
       'Both serialized best.pt and last.pt preserve original COCO modules0–4 including BN exactly; fixed-input P3 checks passed.',
       'P4 entry updates were verified, so only preserving the prefix is not confused with no training.',
       'Validation selects best checkpoint; normal test is evaluated only afterwards, without selecting thresholds or checkpoint.',
       'This is localization training, not a defect-class detector, joint anomaly training, shared-ROI evaluation or AUROC experiment.',
       frozen['policy']['historical_reference'],
       'CPLID faulty images are synthetic and source identity may be unknown; no unverified source-independence claim.','']
    (args.out/'DETECTION_REPORT.md').write_text('\n'.join(text))


def publish(args,frozen):
    def git(*commands):return subprocess.check_output(['git',*commands],cwd=ROOT).decode().strip()
    assert git('branch','--show-current')=='main'
    assert git('remote','get-url','origin')=='https://gitcode.com/gcw_95nPugfk/bounded-proto-det.git'
    assert not git('diff','--cached','--name-only'),'Unrelated staged files; stop sync'
    for name,sha in frozen['code_sha256'].items():
        content=subprocess.check_output(['git','show',f'HEAD:scripts/{name}'],cwd=ROOT)
        assert hashlib.sha256(content).hexdigest()==sha,'Prepared training code must be committed unchanged'
    subprocess.run(['git','fetch','origin','main'],cwd=ROOT,check=True)
    assert int(git('rev-list','--left-right','--count','HEAD...origin/main').split()[1])==0,'Remote ahead; stop without merge/force'
    names=['FROZEN_INPUTS.json','DATA_AUDIT.json','PREPARE.json','FREEZE_CORRECTNESS.json','DETECTION_RESULTS.json','DETECTION_REPORT.md','RESOURCE_COST.json']
    dst=ROOT/'outputs/reports'/frozen['task'];dst.mkdir(parents=True,exist_ok=True)
    for name in names:shutil.copyfile(args.out/name,dst/name)
    paths=[str((dst/name).relative_to(ROOT)) for name in names]
    subprocess.run(['git','add','--',*paths],cwd=ROOT,check=True)
    base=git('show','HEAD:registry/runs.jsonl')
    known={(r['task'],r.get('ts')) for r in map(json.loads,base.splitlines())}
    new=[line for line in (ROOT/'registry/runs.jsonl').read_text().splitlines() if line.strip() and
         json.loads(line)['task']==frozen['task'] and (json.loads(line)['task'],json.loads(line).get('ts')) not in known]
    assert new and all(json.loads(line)['status']=='PASS' for line in new)
    blob=subprocess.check_output(['git','hash-object','-w','--stdin'],cwd=ROOT,input=(base+'\n'+'\n'.join(new)+'\n').encode()).decode().strip()
    subprocess.run(['git','update-index','--cacheinfo',f'100644,{blob},registry/runs.jsonl'],cwd=ROOT,check=True)
    assert all(p in paths+['registry/runs.jsonl'] for p in git('diff','--cached','--name-only').splitlines())
    for command in [['diff','--cached','--check'],['diff','--cached','--stat'],['commit','-m','Publish CPLID strict frozen-P3 detector results'],['push','origin','main']]:
        subprocess.run(['git',*command],cwd=ROOT,check=True)
    assert git('rev-parse','HEAD')==git('rev-parse','origin/main')
    print('[published]',git('rev-parse','HEAD'),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['prepare','cpu-check','train','evaluate'],default='prepare')
    parser.add_argument('--data',type=Path,default=Path('/root/autodl-tmp/data/CPLID'))
    parser.add_argument('--out',type=Path,default=BASE/TASK)
    parser.add_argument('--prior',type=Path,default=BASE/'CPLID-T0T1T2-detector-top8-v1')
    parser.add_argument('--recipe-source',type=Path,default=BASE/'InsPLAD-frozenP3-14class-v1')
    parser.add_argument('--epochs',type=int,default=60);parser.add_argument('--imgsz',type=int,default=1280)
    parser.add_argument('--batch',type=int,default=8);parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();assert args.epochs>0 and args.batch>0 and args.imgsz>0 and args.imgsz%32==0
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    frozen=prepare(args)
    if args.mode=='prepare':return
    if args.mode=='cpu-check':
        model=YOLO(str(INIT)).model.cpu().float().eval();before=tensor_hash(prefix_state(model))
        inputs=probe_inputs(frozen['probe_crops'])
        with torch.inference_mode():
            from insplad_freeze_utils import prefix_features
            one=prefix_features(model,inputs);two=prefix_features(model,inputs)
        assert torch.equal(one,two) and before==tensor_hash(prefix_state(model))
        json_write(args.out/'CPU_PREFLIGHT.json',dict(passed=True,shape=list(one.shape),repeat_max_diff=float((one-two).abs().max()),
            prefix_hash=before,model_state_unchanged=True,gpu_used=False,training_started=False))
        print('[cpu-check] passed; no optimizer step, no GPU training',flush=True);return
    active=read(ROOT/'registry/active_task.json')['task']
    assert active in [frozen['task'],TASK+'-prepare','M3SHARE-speed-v1'],'Unrelated active task; resolve first'
    subprocess.run([sys.executable,str(ROOT/'scripts/start_task.py'),str(args.out/'TASK.yaml')],cwd=ROOT,check=True)
    assert torch.cuda.is_available()
    tick=time.perf_counter();run=args.out/'runs/frozen_p3';last=run/'weights/last.pt'
    if args.mode=='train':
        assert args.resume or not last.exists(),'Existing checkpoint; use --resume or --mode evaluate, never overwrite'
        overrides=dict(**frozen['recipe'],data=str(args.out/'dataset/data.yaml'),model=str(INIT),
                       project=str(args.out/'runs'),name='frozen_p3',exist_ok=True)
        if args.resume:
            assert last.is_file()
            if (run/'freeze_audit').exists():
                stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
                shutil.copytree(run/'freeze_audit',args.out/'resume_audits'/stamp)
            overrides['resume']=str(last)
        trainer=FrozenP3Trainer(overrides=overrides);trainer.train()
    evaluate(args,frozen)
    json_write(args.out/'RESOURCE_COST.json',dict(invocation_seconds=time.perf_counter()-tick,
        peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),epochs=frozen['recipe']['epochs'],
        training_started_by='user',resumed=args.resume,note='Includes final evaluation; resumed runs report only this invocation'))
    subprocess.run([sys.executable,str(ROOT/'scripts/accept_check.py'),str(args.out/'TASK.yaml')],cwd=ROOT,check=True)
    subprocess.run([sys.executable,str(ROOT/'scripts/log_run.py'),'--task',frozen['task'],'--status','PASS',
                    '--result',str(args.out/'DETECTION_REPORT.md'),'--note','CPLID normal-only strict frozen0-4/BN; best/last P3 preservation and normal val/test detection evaluated'],cwd=ROOT,check=True)
    publish(args,frozen)
    print('[done]',args.out/'DETECTION_REPORT.md',flush=True)


if __name__=='__main__':main()
