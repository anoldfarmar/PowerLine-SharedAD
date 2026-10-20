"""Finish the frozen-P3 report from saved predictions, without GPU inference.

Use the same COCO evaluator on both prediction files. Preserve native YOLO
metrics separately where already saved: COCO AP and YOLO AP are not identical.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from faster_coco_eval import COCO,COCOeval_faster
import insplad_frozen_p3 as original
from insplad_freeze_utils import file_sha
from insplad_frozen_p3_recover import corrected_recall,json_write


def coco_metrics(ground_truth,predictions):
    gt=COCO(ground_truth)
    dt=gt.loadRes(predictions)
    evaluation=COCOeval_faster(gt,dt,iouType='bbox')
    evaluation.params.imgIds=sorted(gt.getImgIds())
    evaluation.params.catIds=sorted(gt.getCatIds())
    evaluation.params.maxDets=[1,10,100]
    evaluation.evaluate();evaluation.accumulate();evaluation.summarize()
    values=evaluation.stats_as_dict
    mean=lambda x:float(x[x>=0].mean()) if np.any(x>=0) else None
    per={}
    for i,cid in enumerate(evaluation.params.catIds):
        precision=evaluation.eval['precision'][:,:,i,0,-1]
        per[cid]=dict(AP50_95=mean(precision),AP50=mean(precision[0]),
                      AP75=mean(precision[5]))
    return dict(mAP50=float(values['AP_50']),mAP50_95=float(values['AP_all']),
                metric_definition='COCO AP, IoU .50:.95 step .05, 101 recall points, maxDets=100',per_class=per)


def finish(out,frozen):
    out=Path(out);selected=frozen['selected_category_ids']
    names={cid:frozen['class_mapping'][str(cid)]['name'] for cid in selected}
    records=frozen['records']['val'];lookup={r['file_name']:r for r in records}
    assert len(lookup)==len(records)
    categories=[dict(id=cid,name=names[cid]) for cid in selected]
    ground_truth=dict(info=dict(description='Frozen 14-target InsPLAD validation labels'),licenses=[],
        images=[dict(id=r['image_id'],file_name=r['file_name'],width=r['width'],height=r['height']) for r in records],
        categories=categories,annotations=[])
    for path,h in frozen['annotation_sha256'].items():
        assert file_sha(path)==h,'Original annotations changed'
    for r in records:
        path=out/'dataset/labels/val'/f"{Path(r['file_name']).stem}.txt"
        original_to_new={cid:i for i,cid in enumerate(selected)}
        expected=''.join(original.label_line(a,r,original_to_new[a['category_id']]) for a in r['selected_annotations'])
        assert path.read_text()==expected,'Prepared validation label changed'
        other=out/'historical_dataset/labels/val'/path.name
        assert other.read_text()==''.join(original.label_line(a,r,a['category_id']-1) for a in r['selected_annotations'])
        for row in path.read_text().splitlines():
            c,x,y,w,h=map(float,row.split());cid=selected[int(c)]
            box=[(x-w/2)*r['width'],(y-h/2)*r['height'],w*r['width'],h*r['height']]
            ground_truth['annotations'].append(dict(id=len(ground_truth['annotations'])+1,
                image_id=r['image_id'],category_id=cid,bbox=box,area=box[2]*box[3],iscrowd=0))
    assert len(ground_truth['annotations'])==sum(frozen['counts']['val']['boxes'].values())
    raw=original.read(Path(next(iter(frozen['annotation_sha256']))).with_name('instances_val.json'))
    images={r['id']:r for r in raw['images']}
    partial=out/'DETECTION_RESULTS.json'
    old=original.read(partial) if partial.exists() else {}
    if old and not (out/'DETECTION_RESULTS.before_cached_recovery.json').exists():
        json_write(out/'DETECTION_RESULTS.before_cached_recovery.json',old)
    all_results={};sources={}
    for method,weights,model_to_original in [
        ('frozen_p3',out/'runs/frozen_p3/weights/best.pt',{i:cid for i,cid in enumerate(selected)}),
        ('historical_full',original.HISTORICAL,{cid-1:cid for cid in selected})]:
        path=out/'evaluation'/method/'predictions.json'
        predictions=original.read(path);sources[str(path)]=file_sha(path)
        mapping={k+1:v for k,v in model_to_original.items()}
        recall=corrected_recall(raw['annotations'],images,predictions,selected,model_to_original,frozen['recipe']['imgsz'],.25)
        converted=[]
        for p in predictions:
            assert p['file_name'] in lookup,'Prediction image not in frozen validation set'
            r=lookup[p['file_name']];stem=Path(r['file_name']).stem
            expected_image_id=int(stem) if stem.isnumeric() else stem
            assert p['image_id']==expected_image_id,'Prediction image binding mismatch'
            assert np.isfinite(p['bbox']).all() and np.isfinite(p['score'])
            if p['category_id'] not in mapping:
                continue
            converted.append(dict(image_id=r['image_id'],category_id=mapping[p['category_id']],
                                  bbox=p['bbox'],score=p['score']))
        metrics=coco_metrics(ground_truth,converted)
        stats=[]
        for cid in selected:
            ap=metrics['per_class'][cid]
            stats.append(dict(Class=names[cid],category_id=cid,**ap,
                validation_instances=frozen['counts']['val']['boxes'][str(cid)]))
        native=old.get('methods',{}).get(method,{})
        all_results[method]=dict(weights=str(weights),weights_sha256=file_sha(weights),
            mAP50=metrics['mAP50'],mAP50_95=metrics['mAP50_95'],metric_definition=metrics['metric_definition'],
            per_class=stats,fixed_threshold_recall=recall,predictions=str(path),
            native_yolo_metrics_saved_before_recovery={k:native[k] for k in ['mAP50','mAP50_95'] if k in native})
        print(f"[cached COCO] {method} AP50={metrics['mAP50']:.6f} AP50:95={metrics['mAP50_95']:.6f}; excluded non-target predictions={recall['excluded_non_target_prediction_count']}",flush=True)
    proof=original.read(out/'FREEZE_CORRECTNESS.json');assert proof['passed']
    evaluated=[cid for cid in selected if frozen['counts']['val']['boxes'][str(cid)]>0]
    lines=['# InsPLAD strict frozen-P3 detector — completed from cached predictions','',
           'Both methods use the same COCO AP protocol (maxDets=100), recomputed on CPU from saved prediction JSON. No new training or model inference.',
           'Prediction JSON has rounded boxes/scores; these COCO metrics are distinct from native YOLO AP. Available native metrics are preserved separately in JSON.',
           f'Training targets: {len(selected)}; validation AP covers {len(evaluated)} classes with GT. C18 sphere has no validation GT.','',
           '| Method | COCO AP50 | COCO AP50:95 |','|---|---:|---:|']
    for method,result in all_results.items():
        lines.append(f"| {method} | {result['mAP50']:.4f} | {result['mAP50_95']:.4f} |")
    a=all_results['frozen_p3'];b=all_results['historical_full']
    lines+=['',f"Frozen−historical AP50:95 = {100*(a['mAP50_95']-b['mAP50_95']):+.2f} percentage points.",'',
            '| Class | Frozen AP50:95 | Historical AP50:95 | Frozen recall@.25 | Historical recall@.25 | Small GT | Frozen small recall | Historical small recall |',
            '|---|---:|---:|---:|---:|---:|---:|---:|']
    show=lambda v:'N/A' if v is None else f'{v:.4f}'
    for i,cid in enumerate(selected):
        ar=a['fixed_threshold_recall']['per_class'][str(cid)];br=b['fixed_threshold_recall']['per_class'][str(cid)]
        lines.append('| '+' | '.join([names[cid],show(a['per_class'][i]['AP50_95']),show(b['per_class'][i]['AP50_95']),
             show(ar['all']['recall']),show(br['all']['recall']),str(ar['small']['gt']),
             show(ar['small']['recall']),show(br['small']['recall'])])+' |')
    lines+=['','Recall uses fixed confidence .25 and IoU .5. Small bins use area <32² after uniform resize to the declared detector input, not COCO AP-small.',
            'Historical model outputs for excluded C13-C16 are ignored in this selected-target evaluation. Unknown/out-of-range class IDs remain errors.',
            frozen['reference_note'],
            'Both best and last checkpoints passed prefix parameter/BN identity and same-input P3 tests. This does not establish shared-ROI anomaly AUROC.','']
    (out/'DETECTION_REPORT.md').write_text('\n'.join(lines))
    json_write(out/'DETECTION_RESULTS.json',dict(completed=True,methods=all_results,
       metric_definition='COCO AP from cached prediction JSON, maxDets=100',
       reference_note=frozen['reference_note'],map_evaluated_category_ids=evaluated,map_evaluated_class_count=len(evaluated)))
    json_write(out/'CACHED_REPORT_RECOVERY.json',dict(recovery_code_sha256=file_sha(Path(__file__)),
       wrapper_code_sha256=file_sha(Path(__file__).with_name('insplad_frozen_p3_recover.py')),
       prediction_sha256=sources,original_frozen_inputs_sha256=file_sha(out/'FROZEN_INPUTS.json'),
       training_restarted=False,gpu_inference_repeated=False,
       correction='Filter valid non-target classes from old 18-class head; preserve both class-ID spaces',
       reporting_metric='Common COCO AP, explicitly separate from original YOLO AP'))
    subprocess.run([sys.executable,str(original.ROOT/'scripts/accept_check.py'),str(out/'TASK.yaml')],cwd=original.ROOT,check=True)
    subprocess.run([sys.executable,str(original.ROOT/'scripts/log_run.py'),'--task',frozen['task'],'--status','PASS',
                    '--result',str(out/'DETECTION_REPORT.md'),'--note','Completed CPU report from saved predictions; target class mapping fixed; COCO AP metric explicit'],cwd=original.ROOT,check=True)
    print('[done]',out/'DETECTION_REPORT.md',flush=True)
