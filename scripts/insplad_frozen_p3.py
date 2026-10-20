#!/usr/bin/env python3
"""User-run InsPLAD detector: frozen original COCO modules 0..4, 14 selected classes.

Default mode prepares/audits data on CPU. train is only launched by the user.
No CPLID, anomaly test fitting, adapters, or ROI feature experiment is included.
"""
import argparse
import collections
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import torch
import yaml
from ultralytics import YOLO
import ultralytics

from insplad_freeze_utils import (INIT,INIT_SHA,SELECTED_IDS,file_sha,fixed_recall,json_write)
from insplad_frozen_p3_trainer import FrozenP3Trainer,audit_checkpoint

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_OUT=Path('/root/autodl-tmp/version10-12/InsPLAD-frozenP3-14class-v1')
HISTORICAL=Path('/root/autodl-tmp/runs/yolov8n_det/weights/best.pt')
HISTORICAL_SHA='ea1526c29d4afcfb83cbc2a8d3eb48f498c380082d2a4dfbcc17d0b9e9c425fe'


def read(p):
    return json.loads(Path(p).read_text())


def immutable_text(path,text):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        assert path.read_text()==text,f'Prepared file changed: {path}'
    else:
        path.write_text(text)


def link_image(path,target):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() or path.is_symlink():
        assert path.is_symlink() and path.resolve()==target.resolve(),path
    else:
        path.symlink_to(target.resolve())


def label_line(annotation,image,category):
    x,y,w,h=annotation['bbox'];iw=image['width'];ih=image['height']
    x1=max(0,min(iw,x));y1=max(0,min(ih,y));x2=max(0,min(iw,x+w));y2=max(0,min(ih,y+h))
    assert x2>x1 and y2>y1,annotation
    return f'{category} {(x1+x2)/2/iw:.9f} {(y1+y2)/2/ih:.9f} {(x2-x1)/iw:.9f} {(y2-y1)/ih:.9f}\n'


def prepare(args):
    assert args.ids==SELECTED_IDS,'Current authorized class set is C01–C12, C17, C18'
    assert file_sha(INIT)==INIT_SHA and file_sha(HISTORICAL)==HISTORICAL_SHA
    raw={s:read(args.data/'annotations'/f'instances_{s}.json') for s in ['train','val']}
    cats=sorted(raw['train']['categories'],key=lambda c:c['id'])
    assert [(c['id'],c['name']) for c in cats]==[(c['id'],c['name']) for c in sorted(raw['val']['categories'],key=lambda c:c['id'])]
    names={c['id']:c['name'] for c in cats};mapping={cid:i for i,cid in enumerate(args.ids)}
    assert set(mapping)<=set(names)
    assert len(args.ids)==14
    out=args.out;out.mkdir(parents=True,exist_ok=True)
    records={};counts={};probes=[];paths={};anomaly_inventory={};ambiguities={}
    for split,d in raw.items():
        bystem=collections.defaultdict(list)
        for im in d['images']:
            bystem[Path(im['file_name']).stem].append(im)
        ambiguous={stem:ims for stem,ims in bystem.items() if len(ims)>1}
        ambiguities[split]=ambiguous
        images={im['id']:im for im in d['images'] if Path(im['file_name']).stem not in ambiguous}
        byimage={i:[] for i in images}
        for ann in d['annotations']:
            if ann['image_id'] in byimage and ann['category_id'] in mapping and not ann.get('iscrowd',0):
                byimage[ann['image_id']].append(ann)
        split_records=[];box_counts={str(cid):0 for cid in args.ids}
        for iid,image in sorted(images.items()):
            path=args.data/split/image['file_name'];assert path.is_file(),path
            selected=sorted(byimage[iid],key=lambda a:a['id'])
            for ann in selected:
                box_counts[str(ann['category_id'])]+=1
            split_records.append(dict(image_id=iid,path=str(path),file_name=image['file_name'],
                width=image['width'],height=image['height'],selected_annotations=selected))
            paths[str(path)]=split
            if split=='train' and len(probes)<4 and selected:
                ann=selected[0];x,y,w,h=ann['bbox']
                probes.append(dict(path=str(path),xyxy=[int(round(x)),int(round(y)),int(round(x+w)),int(round(y+h))],
                                   category_id=ann['category_id'],annotation_id=ann['id']))
        records[split]=split_records
        counts[split]=dict(images=len(images),positive_images=sum(bool(r['selected_annotations']) for r in split_records),
                           background_images=sum(not r['selected_annotations'] for r in split_records),boxes=box_counts,
                           original_image_records=len(d['images']),ambiguous_filename_groups_excluded=len(ambiguous),
                           excluded_annotation_records=sum(a['image_id'] not in images for a in d['annotations']))
        if split=='train':
            assert all(n>0 for n in box_counts.values()),'Selected class absent from training'
    print('[prepare] hashing original InsPLAD detection train/val images (CPU only)',flush=True)
    hashes={p:file_sha(p) for p in sorted(paths)}
    train_hashes={h for p,h in hashes.items() if paths[p]=='train'}
    val_hashes={h for p,h in hashes.items() if paths[p]=='val'}
    duplicate_hashes=sorted(train_hashes & val_hashes)
    review_path=Path('/root/autodl-tmp/version10-12/InsPLAD-class-review-v1/CLASS_INVENTORY.json')
    if review_path.exists():
        anomaly_inventory={r['code']:dict(unsupervised=r['unsupervised'],defect_supervised=r['defect_supervised'])
                           for r in read(review_path) if r['category_id'] in mapping}
    recipe=dict(epochs=args.epochs,imgsz=args.imgsz,batch=args.batch,optimizer='AdamW',lr0=0.001,lrf=0.01,
                weight_decay=0.0005,warmup_epochs=3,seed=0,patience=0,amp=False,freeze=5,
                channels_last=False,compile=False,workers=8,mosaic=1.0,close_mosaic=10,mixup=0.0,
                copy_paste=0.0,hsv_h=0.015,hsv_s=0.7,hsv_v=0.4,degrees=0.0,translate=0.1,scale=0.5,
                fliplr=0.5,flipud=0.0,device=0,cache=False,deterministic=True,val=True,plots=True,
                save=True,save_period=10,verbose=True)
    sources={str(args.data/'annotations'/f'instances_{s}.json'):file_sha(args.data/'annotations'/f'instances_{s}.json') for s in raw}
    trainer_source=Path(ultralytics.__file__).parent/'engine/trainer.py'
    frozen=dict(task='InsPLAD-frozenP3-14class-v1',dataset='InsPLAD detection annotations only',
        selected_codes=[f'C{i:02d}' for i in args.ids],selected_category_ids=args.ids,
        class_mapping={str(cid):dict(new_id=mapping[cid],name=names[cid]) for cid in args.ids},
        all_original_names=[names[c['id']] for c in cats],recipe=recipe,counts=counts,records=records,
        original_image_sha256=hashes,annotation_sha256=sources,probe_crops=probes,
        init=dict(path=str(INIT),sha256=INIT_SHA),historical=dict(path=str(HISTORICAL),sha256=HISTORICAL_SHA),
        code_sha256={n:file_sha(ROOT/'scripts'/n) for n in ['insplad_frozen_p3.py','insplad_frozen_p3_trainer.py','insplad_freeze_utils.py']},
        environment=dict(torch=torch.__version__,ultralytics=ultralytics.__version__,trainer_path=str(trainer_source),
                         trainer_sha256=file_sha(trainer_source)),anomaly_inventory=anomaly_inventory,
        split_audit=dict(exact_duplicate_image_hashes=duplicate_hashes,ambiguous_image_records_excluded=ambiguities,
                         source_frame_isolation='not established; original train/val membership retained for unambiguous files'),
        reference_note='Historical 18-class fine-tune evaluated on same 14 target categories; not a matched same-recipe 14-class training ablation')
    target=out/'FROZEN_INPUTS.json'
    if target.exists():
        assert read(target)==frozen,'Inputs, recipe or code changed; use a new output directory'
    json_write(target,frozen)
    for split,rs in records.items():
        for dataset in ['dataset','historical_dataset']:
            (out/dataset/'images'/split).mkdir(parents=True,exist_ok=True)
        for r in rs:
            for dataset in ['dataset','historical_dataset']:
                link_image(out/dataset/'images'/split/r['file_name'],Path(r['path']))
            stem=Path(r['file_name']).stem
            content=''.join(label_line(a,r,mapping[a['category_id']]) for a in r['selected_annotations'])
            original=''.join(label_line(a,r,a['category_id']-1) for a in r['selected_annotations'])
            immutable_text(out/'dataset/labels'/split/(stem+'.txt'),content)
            immutable_text(out/'historical_dataset/labels'/split/(stem+'.txt'),original)
    for dataset,newnames in [('dataset',[names[cid] for cid in args.ids]),('historical_dataset',[c['name'] for c in cats])]:
        doc=dict(path=str((out/dataset).resolve()),train='images/train',val='images/val',names=newnames)
        immutable_text(out/dataset/'data.yaml',yaml.safe_dump(doc,sort_keys=False))
    checkpoint=YOLO(str(INIT)).model
    assert len(checkpoint.model)>5
    assert all(torch.equal(v,v.half().float()) for k,v in checkpoint.model[:5].state_dict().items() if v.is_floating_point()),'Original prefix not FP16 representable; checkpoint policy needs review'
    historical=YOLO(str(HISTORICAL)).model
    assert list(historical.names.values())==[c['name'] for c in cats]
    del checkpoint,historical
    json_write(out/'DATA_AUDIT.json',dict(prepared=True,counts=counts,selected_codes=frozen['selected_codes'],
        original_annotations_unchanged=True,image_storage='symlinks, not copies',train_val_exact_duplicates=len(duplicate_hashes),
        ambiguous_filename_groups=ambiguities,
        anomaly_training_or_query_changes=False))
    contract=dict(task=frozen['task'],gate_required=None,authorized_by='User selected C01-C12,C17,C18; user launches training',
        results_dir=str(out),scope=dict(classes=frozen['selected_codes'],freeze=list(range(5)),data='InsPLAD only',
        recipe=recipe,adapter=False,anomaly_evaluation=False),acceptance=[
        dict(type='metric_min',path=str(out/'FREEZE_CORRECTNESS.json'),key='passed',min=1),
        dict(type='metric_min',path=str(out/'DETECTION_RESULTS.json'),key='completed',min=1),
        dict(type='file_exists',path=str(out/'DETECTION_REPORT.md'))])
    (out/'TASK.yaml').write_text(yaml.safe_dump(contract,sort_keys=False))
    print(json.dumps(dict(status='PREPARED_CPU_ONLY',classes=frozen['selected_codes'],counts=counts,
                         duplicate_hashes=len(duplicate_hashes),output=str(out)),indent=2),flush=True)
    return frozen


def evaluate(args,frozen):
    from insplad_frozen_p3_trainer import audit_checkpoint
    best=args.out/'runs/frozen_p3/weights/best.pt';last=best.parent/'last.pt'
    assert best.is_file() and last.is_file()
    train_audit=read(args.out/'runs/frozen_p3/freeze_audit/TRAIN_END.json')
    assert train_audit['passed']
    checks={n:audit_checkpoint(p,frozen['probe_crops'],device='cuda:0') for n,p in [('best',best),('last',last)]}
    json_write(args.out/'FREEZE_CORRECTNESS.json',dict(passed=True,train=train_audit,checkpoints=checks))
    results={};original=read(args.data/'annotations/instances_val.json')
    images={i['id']:i for i in original['images']}
    for method,path,data,classes,mapping in [
        ('frozen_p3',best,args.out/'dataset/data.yaml',None,{i:cid for i,cid in enumerate(args.ids)}),
        ('historical_full',HISTORICAL,args.out/'historical_dataset/data.yaml',[cid-1 for cid in args.ids],{cid-1:cid for cid in args.ids})]:
        model=YOLO(str(path))
        metric=model.val(data=str(data),imgsz=args.imgsz,batch=args.batch,device=0,workers=8,
              half=False,conf=0.001,iou=0.7,max_det=300,save_json=True,plots=True,
              classes=classes,project=str(args.out/'evaluation'),name=method,exist_ok=True,verbose=True)
        prediction_path=args.out/'evaluation'/method/'predictions.json'
        assert prediction_path.is_file(),prediction_path
        predictions=read(prediction_path)
        recall=fixed_recall(original['annotations'],images,predictions,args.ids,mapping,args.imgsz,0.25)
        results[method]=dict(weights=str(path),weights_sha256=file_sha(path),mAP50=float(metric.box.map50),
             mAP50_95=float(metric.box.map),per_class=metric.summary(decimals=8),fixed_threshold_recall=recall,
             speed=metric.speed,predictions=str(prediction_path),class_id_to_original=mapping)
        json_write(args.out/'DETECTION_RESULTS.json',dict(completed=False,methods=results))
    evaluated_ids=[cid for cid in args.ids if frozen['counts']['val']['boxes'][str(cid)]>0]
    json_write(args.out/'DETECTION_RESULTS.json',dict(completed=True,methods=results,
               reference_note=frozen['reference_note'],map_evaluated_category_ids=evaluated_ids,
               map_evaluated_class_count=len(evaluated_ids)))
    byname={c['name']:c['id'] for c in original['categories']}
    per={method:{r['Class']:r for r in result['per_class']} for method,result in results.items()}
    names=[frozen['class_mapping'][str(cid)]['name'] for cid in args.ids]
    lines=['# InsPLAD strict frozen-P3 detector','',
           'Classes: '+', '.join(frozen['selected_codes']),
           f'Trains {len(args.ids)} classes; mAP covers {len(evaluated_ids)} classes with validation GT. C18 sphere has zero val GT and AP/recall is N/A.',
           'Modules 0–4 and all their parameters/BN buffers match original COCO weights exactly. Fixed-input P3 reload checks passed.','',
           '| Method | mAP50 | mAP50:95 |','|---|---:|---:|']
    for m,r in results.items():
        lines.append(f"| {m} | {r['mAP50']:.4f} | {r['mAP50_95']:.4f} |")
    lines+=['','| Class | Frozen AP50:95 | Historical AP50:95 | Frozen small recall | Historical small recall | Small GT |',
            '|---|---:|---:|---:|---:|---:|']
    for name in names:
        cid=byname[name]
        f=results['frozen_p3']['fixed_threshold_recall']['per_class'][str(cid)]['small']
        h=results['historical_full']['fixed_threshold_recall']['per_class'][str(cid)]['small']
        show=lambda v:'N/A' if v is None else f'{v:.4f}'
        fa=per['frozen_p3'].get(name,{}).get('mAP50-95')
        ha=per['historical_full'].get(name,{}).get('mAP50-95')
        lines.append(f"| {name} | {show(fa)} | {show(ha)} | {show(f['recall'])} | {show(h['recall'])} | {f['gt']} |")
    lines+=['','Recall uses fixed confidence .25 / IoU .5. Small means GT area <32² after uniform resize to the declared detector input size; this is not COCO AP-small.',
            frozen['reference_note'],
            'This experiment verifies localization and same-input feature preservation. It does not evaluate shared full-image ROI anomaly scores or promise crop-level AUROC retention.',
            'Original split membership retained; 46 ambiguous training filenames excluded from the derived dataset. Source isolation from anomaly benchmarks has not been established.','']
    (args.out/'DETECTION_REPORT.md').write_text('\n'.join(lines))


def activate(args,frozen):
    active=ROOT/'registry/active_task.json'
    if active.exists():
        current=read(active)['task']
        if current!=frozen['task']:
            assert current=='InsPLAD-COCO-top8-full-v1','Unrelated active task; resolve first'
            prior=Path('/root/autodl-tmp/version10-12/InsPLAD-COCO-top8-full-v1')
            assert read(prior/'metrics.json')['all_complete'] and read(prior/'REPRO_AND_CORRECTNESS.json')['passed']
    subprocess.run([sys.executable,str(ROOT/'scripts/start_task.py'),str(args.out/'TASK.yaml')],cwd=ROOT,check=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode',choices=['prepare','train','evaluate'],default='prepare')
    ap.add_argument('--out',type=Path,default=DEFAULT_OUT)
    ap.add_argument('--data',type=Path,default=Path('/root/autodl-tmp/data/insplad'))
    ap.add_argument('--epochs',type=int,default=60)
    ap.add_argument('--imgsz',type=int,default=1280)
    ap.add_argument('--batch',type=int,default=8)
    ap.add_argument('--resume',action='store_true')
    args=ap.parse_args();args.ids=SELECTED_IDS
    assert args.epochs>0 and args.batch>0 and args.imgsz%32==0
    frozen=prepare(args)
    if args.mode=='prepare':
        return
    assert torch.cuda.is_available()
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    activate(args,frozen)
    if args.mode=='train':
        weights=args.out/'runs/frozen_p3/weights/last.pt'
        assert args.resume or not weights.exists(),'Checkpoint exists; use --resume or a new output directory'
        if args.resume:
            assert weights.is_file()
        overrides=dict(**frozen['recipe'],data=str(args.out/'dataset/data.yaml'),model=str(INIT),
                       project=str(args.out/'runs'),name='frozen_p3',exist_ok=True)
        if args.resume:
            overrides['resume']=str(weights)
        trainer=FrozenP3Trainer(overrides=overrides)
        trainer.train()
    evaluate(args,frozen)
    subprocess.run([sys.executable,str(ROOT/'scripts/accept_check.py'),str(args.out/'TASK.yaml')],cwd=ROOT,check=True)
    subprocess.run([sys.executable,str(ROOT/'scripts/log_run.py'),'--task',frozen['task'],'--status','PASS',
                    '--result',str(args.out/'DETECTION_REPORT.md'),
                    '--note','14-class strict frozen-P3 detector; best/last prefix and fixed-input P3 checks passed'],cwd=ROOT,check=True)
    print('[done]',args.out/'DETECTION_REPORT.md',flush=True)


if __name__=='__main__':
    main()
