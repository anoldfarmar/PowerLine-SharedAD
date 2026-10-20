"""Strict prefix freezing and detection-recall utilities (no dataset mutation)."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

SELECTED_IDS=list(range(1,13))+[17,18]
PREFIXES=tuple(f'model.{i}.' for i in range(5))
INIT=Path('/root/autodl-tmp/third_party/yolov8n.pt')
INIT_SHA='f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36'


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def json_write(path,obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(p)


def prefix_state(model):
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items() if k.startswith(PREFIXES)}


def tensor_hash(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):
        h.update(k.encode());h.update(v.contiguous().numpy().tobytes())
    return h.hexdigest()


def assert_prefix_equal(model,reference):
    current=prefix_state(model)
    assert current.keys()==reference.keys()
    bad=[k for k in current if not torch.equal(current[k],reference[k])]
    assert not bad, f'Frozen prefix changed: {bad[:8]}'
    return tensor_hash(current)


def lock_prefix(model):
    for layer in model.model[:5]:
        layer.eval()
        for p in layer.parameters():
            p.requires_grad_(False)


def assert_locked(model):
    assert all(not p.requires_grad for layer in model.model[:5] for p in layer.parameters())
    assert all(not m.training for layer in model.model[:5] for m in layer.modules() if isinstance(m,torch.nn.BatchNorm2d))


@torch.inference_mode()
def prefix_features(model,inputs):
    x=inputs
    for layer in model.model[:5]:
        x=layer(x)
    assert x.shape[1]==64
    return x.detach().cpu()


def iou_xywh(a,b):
    ax,ay,aw,ah=a;bx,by,bw,bh=b
    inter=max(0,min(ax+aw,bx+bw)-max(ax,bx))*max(0,min(ay+ah,by+bh)-max(ay,by))
    return inter/max(aw*ah+bw*bh-inter,1e-12)


def fixed_recall(annotations,images,predictions,selected_ids,category_mapping,imgsz=1280,conf=0.25):
    """Greedy class-aware IoU .5 matching, at a predeclared confidence threshold.

    Size bins use GT area after uniform resize to detector imgsz (padding has
    no area effect). These are input-size bins, not official COCO AP-small.
    """
    selected=set(selected_ids);gt={};sizes={};counts={};hits={}
    for cid in selected:
        counts[cid]={s:0 for s in ['all','small','medium','large']}
        hits[cid]={s:0 for s in ['all','small','medium','large']}
    for a in annotations:
        cid=a['category_id']
        if cid not in selected or a.get('iscrowd',0):
            continue
        im=images[a['image_id']];stem=Path(im['file_name']).stem
        stem=str(int(stem)) if stem.isnumeric() else stem
        factor=imgsz/max(im['width'],im['height']);area=a['bbox'][2]*a['bbox'][3]*factor**2
        size='small' if area<32**2 else ('medium' if area<96**2 else 'large')
        gt.setdefault((stem,cid),[]).append((a['bbox'],size))
        counts[cid]['all']+=1;counts[cid][size]+=1
    pred={}
    for p in predictions:
        cid=category_mapping.get(int(p['category_id']))
        if cid in selected and p['score']>=conf:
            pred.setdefault((str(p['image_id']),cid),[]).append(p)
    total_predictions=0;matched=0
    for key,ps in pred.items():
        candidates=gt.get(key,[]);used=set()
        for p in sorted(ps,key=lambda p:-p['score']):
            total_predictions+=1
            scores=[(iou_xywh(p['bbox'],box),i) for i,(box,_) in enumerate(candidates) if i not in used]
            overlap,i=max(scores,default=(-1,-1))
            if overlap>=0.5:
                used.add(i);size=candidates[i][1];cid=key[1]
                hits[cid]['all']+=1;hits[cid][size]+=1;matched+=1
    result={}
    for cid in selected:
        result[str(cid)]={s:dict(gt=counts[cid][s],matched=hits[cid][s],
                              recall=hits[cid][s]/counts[cid][s] if counts[cid][s] else None)
                          for s in counts[cid]}
    return dict(per_class=result,conf=conf,iou=0.5,input_imgsz=imgsz,
                predictions_above_threshold=total_predictions,true_positives=matched,
                precision=matched/total_predictions if total_predictions else 0,
                note='Input-resized GT area bins, not official COCO AP-small; missed GTs included')
