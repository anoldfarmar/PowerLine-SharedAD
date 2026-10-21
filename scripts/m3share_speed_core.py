#!/usr/bin/env python3
"""Actual, untuned shared/crop inference. No cached feature path exists here."""
import copy
import math
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops import roi_align
from ultralytics import YOLO
from ultralytics.utils.nms import non_max_suppression

import m3_common as common
import m3score_pcverify as pv
import insplad_shared_roi as old

METHODS = ['T-CROP-small', 'T-SHARE-small']


def normalized(f):
    return F.normalize(f.flatten(2).transpose(1,2),dim=2)


def native_to_input(boxes, native_wh, network_wh=(320,320), gain_xy=None, pad_xy=(0,0)):
    """Stretch or explicit LetterBox affine. Runtime uses stretch (no padding)."""
    gains = gain_xy or (network_wh[0]/native_wh[0], network_wh[1]/native_wh[1])
    t = torch.as_tensor(boxes,dtype=torch.float32).reshape(-1,4).clone()
    t[:,[0,2]] = t[:,[0,2]]*gains[0]+pad_xy[0]
    t[:,[1,3]] = t[:,[1,3]]*gains[1]+pad_xy[1]
    return t


def candidate_boxes(rows, expected, wh):
    result = []
    for x1,y1,x2,y2,conf,cls in rows:
        if int(cls) != expected:
            continue
        b = [max(0.,min(320.,float(v))) for v in [x1,y1,x2,y2]]
        if b[2] > b[0] and b[3] > b[1]:
            result.append(dict(input_xyxy=b,native_xyxy=[b[0]*wh[0]/320,b[1]*wh[1]/320,
                                                       b[2]*wh[0]/320,b[3]*wh[1]/320],
                               confidence=float(conf),class_id=int(cls)))
    return sorted(result,key=lambda r:(-r['confidence'],*r['input_xyxy']))


def crop_box(box, wh):
    x1,y1,x2,y2 = box
    return [max(0,min(wh[0]-1,math.floor(x1))),max(0,min(wh[1]-1,math.floor(y1))),
            max(1,min(wh[0],math.ceil(x2))),max(1,min(wh[1],math.ceil(y2)))]


def nearest(features, bank, query_chunk=2048, bank_chunk=512):
    flat = features.reshape(-1,64)
    pieces = []
    for start in range(0,len(flat),query_chunk):
        q = flat[start:start+query_chunk]
        running = None
        for i in range(0,len(bank),bank_chunk):
            d = pv._cls().euclidean_dist(q,bank[i:i+bank_chunk]).min(1).values
            running = d if running is None else torch.minimum(running,d)
        pieces.append(running)
    return torch.cat(pieces).reshape(len(features),1600)


class Stages:
    def __init__(self, enabled):
        self.enabled = enabled
        self.times = {}

    def call(self, name, fn):
        if not self.enabled:
            return fn()
        torch.cuda.synchronize()
        tick = time.perf_counter()
        result = fn()
        torch.cuda.synchronize()
        self.times[name] = 1000*(time.perf_counter()-tick)
        return result


class Pipeline:
    def __init__(self, weights, crop_batch=32):
        tick = time.perf_counter()
        self.model = YOLO(str(weights)).model.cuda().float().eval()
        self.model.requires_grad_(False)
        self.prefix = torch.nn.Sequential(*copy.deepcopy(list(self.model.model[:5]))).cuda().float().eval()
        self.prefix.requires_grad_(False)
        self.crop_batch = crop_batch
        self.tf = common.make_transform(320)
        self.p3 = None
        self.det_calls = self.det_images = self.crop_calls = self.crop_images = 0
        self.model.model[4].register_forward_hook(self.det_hook)
        self.prefix[4].register_forward_hook(self.crop_hook)
        torch.cuda.synchronize()
        self.load_seconds = time.perf_counter()-tick

    def det_hook(self, module, args, result):
        self.det_calls += 1
        self.det_images += len(result)
        self.p3 = result

    def crop_hook(self, module, args, result):
        self.crop_calls += 1
        self.crop_images += len(result)

    def counters(self):
        return dict(full_backbone_calls=self.det_calls,full_images=self.det_images,
                    crop_backbone_calls=self.crop_calls,crop_images=self.crop_images)

    @torch.inference_mode()
    def run(self, images, assets, method, banks=None, profile=False, overrides=None, keep_features=False):
        assert method in METHODS and len(images)==len(assets)
        start_counts = self.counters()
        stage = Stages(profile)
        inputs = stage.call('preprocess_transfer_ms',lambda:torch.stack([self.tf(im) for im in images]).cuda())
        def detect():
            self.p3 = None
            prediction = self.model(inputs)
            assert self.p3 is not None
            rows = non_max_suppression(prediction,conf_thres=.25,iou_thres=.7,nc=14,
                                       max_det=300,max_time_img=60.)
            return [candidate_boxes(r.detach().cpu().tolist(),old.ASSET_CLASS[a],im.size)
                    for r,a,im in zip(rows,assets,images)]
        boxes = stage.call('full_detection_nms_coordinates_ms',detect)
        if overrides is not None:
            boxes = overrides
        p3 = self.p3
        self.p3 = None
        regions = []
        effective = []
        for index,(image,asset,frame_boxes) in enumerate(zip(images,assets,boxes)):
            actual = frame_boxes or [dict(native_xyxy=[0.,0.,float(image.width),float(image.height)],
                                         confidence=None,class_id=old.ASSET_CLASS[asset])]
            effective.append(len(actual))
            for row in actual:
                regions.append(dict(frame_index=index,asset=asset,native_xyxy=row['native_xyxy'],
                                    crop_integer_xyxy=crop_box(row['native_xyxy'],image.size),
                                    fallback=not bool(frame_boxes)))
        if profile:
            torch.cuda.synchronize()
            diagnostic_start = time.perf_counter()
        if method == METHODS[0]:
            del p3  # Crop arm does not retain unused detection features.
            def crop_inputs():
                tensors = [self.tf(images[r['frame_index']].crop(r['crop_integer_xyxy'])) for r in regions]
                return torch.stack(tensors).cuda()
            crop = stage.call('crop_resize_transfer_or_roi_ms',crop_inputs)
            def extract():
                return torch.cat([normalized(self.prefix(crop[i:i+self.crop_batch]))
                                  for i in range(0,len(crop),self.crop_batch)])
            features = stage.call('extra_crop_prefix_ms',extract)
        else:
            def extract_roi():
                rois = []
                for r in regions:
                    image = images[r['frame_index']]
                    box = r['native_xyxy']
                    b = [box[0]*320/image.width,box[1]*320/image.height,
                         box[2]*320/image.width,box[3]*320/image.height]
                    rois.append([r['frame_index'],*b])
                tensor = torch.tensor(rois,device='cuda',dtype=torch.float32)
                return normalized(roi_align(p3,tensor,(40,40),spatial_scale=.125,aligned=True,sampling_ratio=1))
            features = stage.call('crop_resize_transfer_or_roi_ms',extract_roi)
            if profile:
                stage.times['extra_crop_prefix_ms'] = 0.
        def score():
            if banks is None:
                return [None]*len(regions)
            positions = defaultdict(list)
            for i,r in enumerate(regions):
                positions[r['asset']].append(i)
            result = [None]*len(regions)
            for asset,indices in positions.items():
                q = features[indices]
                values = nearest(q,banks[asset]).topk(8,dim=1).values.mean(1).cpu().tolist()
                for index,value in zip(indices,values):
                    result[index] = value
            return result
        values = stage.call('nearest_neighbor_score_ms',score)
        outputs = []
        cursor = 0
        for i,frame_boxes in enumerate(boxes):
            n = effective[i]
            outputs.append(dict(boxes=frame_boxes,target_box_count=len(frame_boxes),
                                fallback=not bool(frame_boxes),scores=values[cursor:cursor+n],
                                crop_boxes_integer=[r['crop_integer_xyxy'] for r in regions[cursor:cursor+n]]))
            cursor += n
        if profile:
            torch.cuda.synchronize()
            stage.times['diagnostic_stage_ms'] = 1000*(time.perf_counter()-diagnostic_start)
        counts = {k:v-start_counts[k] for k,v in self.counters().items()}
        assert counts['full_backbone_calls']==1 and counts['full_images']==len(images)
        expected_calls = math.ceil(len(regions)/self.crop_batch) if method==METHODS[0] else 0
        assert counts['crop_backbone_calls']==expected_calls
        assert counts['crop_images']==(len(regions) if method==METHODS[0] else 0)
        result = dict(outputs=outputs,counters=counts,stages=stage.times,effective_regions=len(regions))
        if keep_features:
            result['features'] = features.detach().cpu()
        return result


def compare_results(first, second, features=False, scores=True):
    assert len(first['outputs'])==len(second['outputs'])
    evidence = dict(box_maxdiff=0.,score_maxdiff=0.)
    for a,b in zip(first['outputs'],second['outputs']):
        assert a['target_box_count']==b['target_box_count'] and a['fallback']==b['fallback']
        if a['boxes']:
            assert [r['class_id'] for r in a['boxes']]==[r['class_id'] for r in b['boxes']]
            evidence['box_maxdiff'] = max(evidence['box_maxdiff'],old.close(
                torch.tensor([r['native_xyxy'] for r in a['boxes']])/320,
                torch.tensor([r['native_xyxy'] for r in b['boxes']])/320))
        if scores:
            evidence['score_maxdiff'] = max(evidence['score_maxdiff'],old.close(torch.tensor(a['scores']),torch.tensor(b['scores'])))
    if features:
        evidence['feature_maxdiff'] = old.close(first['features'],second['features'])
    return evidence
