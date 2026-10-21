#!/usr/bin/env python3
"""Supplement the completed detector audit; CPU only, no fault-image inference."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox

from insplad_freeze_utils import (INIT, INIT_SHA, assert_prefix_equal, file_sha,
                                 iou_xywh, json_write, prefix_features, prefix_state)


def read(p):
    return json.loads(Path(p).read_text())


def xywh(b):
    return [b[0], b[1], b[2]-b[0], b[3]-b[1]]


def recall(rows, predictions):
    known = {Path(r['path']).name: r for r in rows}
    assert all(p['category_id'] == 1 and p['file_name'] in known for p in predictions)
    tp = count = 0
    per_image = []
    for name, r in known.items():
        ps = sorted([p for p in predictions if p['file_name'] == name and p['score'] >= .25],
                    key=lambda p: -p['score'])
        used = set()
        for p in ps:
            overlap, i = max(((iou_xywh(p['bbox'], xywh(b)), i) for i, b in enumerate(r['boxes'])
                              if i not in used), default=(-1, -1))
            if overlap >= .5:
                used.add(i)
        tp += len(used)
        count += len(ps)
        per_image.append(dict(path=r['path'], gt=len(r['boxes']), predictions=len(ps), matched=len(used)))
    n = sum(len(r['boxes']) for r in rows)
    return dict(conf=.25, iou=.5, GT=n, predictions=count, TP=tp, FN=n-tp, FP=count-tp,
                recall=tp/n, precision=tp/count if count else 0, per_image=per_image,
                matching='confidence-descending greedy one-to-one IoU; all GT in denominator')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--detector-dir', type=Path, default=Path('/root/autodl-tmp/version10-12/CPLID-frozenP3-det-v1'))
    args = ap.parse_args()
    d = args.detector_dir
    frozen = read(d/'FROZEN_INPUTS.json')
    audit = read(d/'FREEZE_CORRECTNESS.json')
    epochs = read(d/'runs/frozen_p3/freeze_audit/EPOCH_CHECKS.json')['epochs']
    assert audit['passed'] and audit['train']['p4_entry_updated']
    assert len(epochs) == 60 and [x['epoch'] for x in epochs] == list(range(1,61))
    assert all(x['ema_prefix_exact'] and x['p3_max_abs_diff'] <= 1e-4 for x in epochs)
    assert file_sha(INIT) == INIT_SHA
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    original = YOLO(str(INIT)).model.cpu().float().eval()
    state = prefix_state(original)
    probes = []
    # Training images only; separately test genuine whole-image LetterBox and GT crop.
    for r in frozen['probe_crops']:
        with Image.open(r['path']) as im:
            im = im.convert('RGB')
            full = LetterBox((1280,1280), auto=False)(image=np.asarray(im))
            crop = np.asarray(im.crop(tuple(round(v) for v in r['xyxy'])).resize((320,320), Image.Resampling.BILINEAR))
        for kind, array in [('whole_letterbox',full), ('GT_crop',crop)]:
            x = torch.from_numpy(array.copy()).permute(2,0,1).unsqueeze(0).float()/255
            probes.append((kind, r['path'], x, prefix_features(original,x)))
    weights = {}
    for name in ['best','last']:
        p = d/f'runs/frozen_p3/weights/{name}.pt'
        assert file_sha(p) == audit['checkpoints'][name]['sha256']
        model = YOLO(str(p)).model.cpu().float().eval()
        h = assert_prefix_equal(model,state)
        checks = []
        for kind,path,x,reference in probes:
            actual = prefix_features(model,x)
            diff = float((actual-reference).abs().max())
            assert torch.allclose(actual,reference,atol=1e-4,rtol=1e-4)
            checks.append(dict(kind=kind,path=path,shape=list(actual.shape),max_abs_diff=diff))
        weights[name] = dict(sha256=file_sha(p),prefix_hash=h,probes=checks)
        del model
    result = read(d/'DETECTION_RESULTS.json')
    fixed = {}
    for split in ['val','test']:
        p = Path(result['splits'][split]['predictions'])
        fixed[split] = recall(frozen['items'][split],read(p))
        fixed[split]['prediction_file_sha256'] = file_sha(p)
    vis = d/'acceptance/vis'
    vis.mkdir(parents=True, exist_ok=True)
    predictions = read(result['splits']['val']['predictions'])
    for r in sorted(frozen['items']['val'],key=lambda r:r['path'])[:6]:
        im = Image.open(r['path']).convert('RGB'); draw = ImageDraw.Draw(im)
        for b in r['boxes']:
            draw.rectangle(b, outline='lime', width=3)
        for p in predictions:
            if p['file_name'] == Path(r['path']).name and p['score'] >= .25:
                x,y,w,h = p['bbox']; draw.rectangle([x,y,x+w,y+h],outline='red',width=3)
                draw.text((x,y),f"pred {p['score']:.3f}",fill='red')
        im.save(vis/(Path(r['path']).stem+'.jpg'))
    out = dict(passed=True,device='cpu',epochs_verified=len(epochs),optimizer_steps=audit['train']['optimizer_step_checks'],
               output_tolerance=dict(atol=1e-4,rtol=1e-4),weights=weights,fixed_threshold=fixed,
               source_hashes={str(d/f):file_sha(d/f) for f in ['FROZEN_INPUTS.json','FREEZE_CORRECTNESS.json','DETECTION_RESULTS.json']},
               visualizations=str(vis),scope='normal val/test only; no fault inference; prefix exact and output checks rerun')
    json_write(d/'DETECTOR_ACCEPTANCE.json',out)
    lines = ['# CPLID detector supplemental acceptance','',
             'PASS: actual best/last SHA, exact prefix including BN, 60 epoch live/EMA evidence, CPU whole-image/crop P3 probes.',
             '', '| Split | mAP50 | mAP50:95 | fixed P@.25 | fixed R@.25 | TP / GT |',
             '|---|---:|---:|---:|---:|---:|']
    for s in ['val','test']:
        a=result['splits'][s]; b=fixed[s]
        lines.append(f"| normal {s} | {a['mAP50']:.6f} | {a['mAP50_95']:.6f} | {b['precision']:.6f} | {b['recall']:.6f} | {b['TP']} / {b['GT']} |")
    lines += ['', 'Native Ultralytics AP; fixed P/R use cached predictions and confidence-descending one-to-one IoU>=.5 matching.',
              'Native summary P/R use its own operating point and must not be called P/R@.25.',
              'All missed GT remain in the recall denominator. No anomaly performance claim follows from detector acceptance.',
              f'GT green / prediction red: {vis}', '']
    (d/'DETECTOR_ACCEPTANCE.md').write_text('\n'.join(lines))
    print(json.dumps(dict(passed=True,fixed_threshold={s:{k:v for k,v in r.items() if k!='per_image'} for s,r in fixed.items()}),indent=2))


if __name__ == '__main__':
    main()
