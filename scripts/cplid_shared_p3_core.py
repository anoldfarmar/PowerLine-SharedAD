#!/usr/bin/env python3
"""CPLID full-frame LetterBox, shared layer4 hook and real native-image crops."""
import math
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.ops import roi_align
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.utils.nms import non_max_suppression


def normalize(x):
    return F.normalize(x.flatten(2).transpose(1,2), dim=-1)


def crop_box(b, size):
    w,h = size
    return [max(0,min(w-1,math.floor(b[0]))),max(0,min(h-1,math.floor(b[1]))),
            max(1,min(w,math.ceil(b[2]))),max(1,min(h,math.ceil(b[3])))]


def affine(size, imgsz=1280):
    w,h = size; gain = min(imgsz/w,imgsz/h)
    nw,nh = round(w*gain),round(h*gain)
    return gain, (round((imgsz-nw)/2-.1),round((imgsz-nh)/2-.1))


def to_input(boxes, size, imgsz=1280):
    gain,(left,top) = affine(size,imgsz)
    b = torch.as_tensor(boxes,dtype=torch.float32).reshape(-1,4).clone()
    b[:,[0,2]] = b[:,[0,2]]*gain+left
    b[:,[1,3]] = b[:,[1,3]]*gain+top
    return b


def iou(a,b):
    area = max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    return area/max((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-area,1e-12)


def match(boxes, gt):
    """Frozen confidence-order matching. Return pred index -> GT index, unmatched ignored by AUROC only."""
    used=set(); out={}
    for i,p in enumerate(boxes):
        overlap,j=max(((iou(p['xyxy'],b),j) for j,b in enumerate(gt) if j not in used),default=(-1,-1))
        if overlap>=.5:
            used.add(j);out[i]=j
    return out


def nearest(features, bank, query_chunk=2048, bank_chunk=512):
    from m3score_pcverify import _cls
    flat = features.reshape(-1,64)
    pieces=[]
    for q in flat.split(query_chunk):
        best=None
        for b in bank.split(bank_chunk):
            d=_cls().euclidean_dist(q,b).min(1).values
            best=d if best is None else torch.minimum(best,d)
        pieces.append(best)
    return torch.cat(pieces).reshape(len(features),1600)


def score(features,bank):
    if not len(features):
        return []
    return nearest(features,bank).topk(8,dim=1).values.mean(1).cpu().tolist()


def state_hash(model):
    from insplad_freeze_utils import tensor_hash
    return tensor_hash({k:v.detach().cpu() for k,v in model.state_dict().items()})


class Pipeline:
    def __init__(self, detector, coco, device='cuda:0', crop_batch=32):
        self.device=device;self.crop_batch=crop_batch
        self.model=YOLO(str(detector)).model.to(device).float().eval().requires_grad_(False)
        original=YOLO(str(coco)).model.cpu().float().eval()
        self.prefix=torch.nn.Sequential(*list(original.model[:5])).to(device).float().eval().requires_grad_(False)
        self.counts=dict(full=0,full_images=0,crop=0,crop_objects=0)
        self.p3=None
        self.model.model[4].register_forward_hook(self.hook)
        self.prefix[4].register_forward_hook(self.crop_hook)

    def hook(self,module,args,out):
        self.p3=out;self.counts['full']+=1;self.counts['full_images']+=len(out)

    def crop_hook(self,module,args,out):
        self.counts['crop']+=1;self.counts['crop_objects']+=len(out)

    def sync(self):
        if str(self.device).startswith('cuda'):
            torch.cuda.synchronize()

    @torch.inference_mode()
    def detect(self,images):
        arrays=[LetterBox((1280,1280),auto=False)(image=np.asarray(im)) for im in images]
        x=torch.from_numpy(np.stack(arrays)).permute(0,3,1,2).contiguous().to(self.device).float()/255
        self.p3=None
        raw=self.model(x)
        p3=self.p3;self.p3=None
        assert p3.shape[1:]==(64,160,160)
        rows=non_max_suppression(raw,conf_thres=.25,iou_thres=.7,nc=1,max_det=300,max_time_img=60.)
        boxes=[]
        for im,r in zip(images,rows):
            gain,(left,top)=affine(im.size)
            result=[]
            for x1,y1,x2,y2,conf,cls in r.detach().cpu().tolist():
                assert int(cls)==0
                b=[max(0.,min(im.width,(x1-left)/gain)),max(0.,min(im.height,(y1-top)/gain)),
                   max(0.,min(im.width,(x2-left)/gain)),max(0.,min(im.height,(y2-top)/gain))]
                if b[2]>b[0] and b[3]>b[1]:
                    result.append(dict(xyxy=b,confidence=conf,class_id=0))
            boxes.append(sorted(result,key=lambda p:(-p['confidence'],*p['xyxy'])))
        return boxes,p3

    @torch.inference_mode()
    def crops(self,image,boxes,batch=None):
        if not boxes:
            return torch.empty((0,1600,64),device=self.device)
        images=[np.asarray(image.crop(crop_box(b,image.size)).resize((320,320),Image.Resampling.BILINEAR)).copy() for b in boxes]
        x=torch.from_numpy(np.stack(images)).permute(0,3,1,2).contiguous().to(self.device).float()/255
        return torch.cat([normalize(self.prefix(p)) for p in x.split(batch or self.crop_batch)])

    def rois(self,p3,image,boxes):
        if not boxes:
            return torch.empty((0,1600,64),device=self.device)
        b=to_input(boxes,image.size).to(self.device)
        rois=torch.cat([torch.zeros((len(b),1),device=self.device),b],dim=1)
        return normalize(roi_align(p3,rois,(40,40),spatial_scale=.125,aligned=True,sampling_ratio=1))

    @torch.inference_mode()
    def run(self,image,method,bank=None,profile=False):
        assert method in ['R1','R2']
        before=dict(self.counts);times={}
        def stage(name,fn):
            if profile:self.sync()
            t=time.perf_counter();r=fn()
            if profile:self.sync();times[name]=(time.perf_counter()-t)*1000
            return r
        boxes,p3=stage('letterbox_transfer_detector_nms_coordinates_ms',lambda:self.detect([image]))
        bs=[p['xyxy'] for p in boxes[0]]
        if profile:self.sync()
        diagnostic=time.perf_counter()
        f=stage('crop_resize_extra_prefix_ms' if method=='R1' else 'roi_align_ms',
                lambda:self.crops(image,bs) if method=='R1' else self.rois(p3,image,bs))
        values=stage('nearest_top8_ms',lambda:score(f,bank) if bank is not None else [])
        if profile:self.sync();times['diagnostic_ms']=(time.perf_counter()-diagnostic)*1000
        counts={k:self.counts[k]-before[k] for k in before}
        assert counts['full']==1 and counts['full_images']==1
        assert counts['crop']==(math.ceil(len(bs)/self.crop_batch) if method=='R1' else 0)
        assert counts['crop_objects']==(len(bs) if method=='R1' else 0)
        return dict(boxes=boxes[0],features=f,scores=values,status='located' if bs else 'unlocalized',
                    counters=counts,stages=times)
