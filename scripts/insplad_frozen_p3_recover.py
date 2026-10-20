#!/usr/bin/env python3
"""Evaluation-only recovery: strict JSON scalars and one-based prediction IDs.

Keep all three original frozen training files byte-for-byte unchanged. This
entry point cannot invoke training and records the reporting corrections.
"""
import argparse
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import torch

import insplad_frozen_p3 as original
from insplad_freeze_utils import file_sha, fixed_recall as original_recall
from ultralytics.models.yolo.detect.val import DetectionValidator


def json_scalar(value):
    if isinstance(value,np.generic):
        return value.item()
    if isinstance(value,np.ndarray):
        return value.tolist()
    if isinstance(value,torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f'Unsupported JSON value: {type(value).__name__}')


def json_write(path,obj):
    path=Path(path)
    # Serialize before touching any existing result; preserve NaN rejection.
    text=json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False,default=json_scalar)+'\n'
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(text);temp.replace(path)


def corrected_recall(annotations,images,predictions,selected_ids,category_mapping,imgsz=1280,conf=0.25):
    # Original evaluate() supplies model class IDs (zero-based). This installed
    # validator exports JSON category IDs as model_id+1 for non-COCO datasets.
    json_mapping={int(k)+1:v for k,v in category_mapping.items()}
    exported_ids={int(p['category_id']) for p in predictions}
    # Old 18-class model may export non-target C13-C16 even when validation
    # requested a subset. They are valid model classes, not corrupt class IDs.
    invalid=exported_ids-set(range(1,max(json_mapping)+1))
    assert not invalid,f'Invalid prediction JSON categories: {sorted(invalid)}'
    retained=[p for p in predictions if int(p['category_id']) in json_mapping]
    result=original_recall(annotations,images,retained,selected_ids,json_mapping,imgsz,conf)
    result['prediction_json_category_id_base']=1
    result['prediction_json_category_to_original']=json_mapping
    result['excluded_non_target_category_ids']=sorted(exported_ids-set(json_mapping))
    result['excluded_non_target_prediction_count']=len(predictions)-len(retained)
    return result


def verify_existing_run(out):
    frozen=json.loads((out/'FROZEN_INPUTS.json').read_text())
    for name,expected in frozen['code_sha256'].items():
        assert file_sha(original.ROOT/'scripts'/name)==expected,f'Original training file changed: {name}'
    source=inspect.getsource(DetectionValidator.init_metrics)
    assert 'list(range(1, len(model.names) + 1))' in source,'Validator JSON category convention changed; inspect before recovery'
    proof=json.loads((out/'FREEZE_CORRECTNESS.json').read_text())
    assert proof['passed'] and proof['train']['epochs_audited']==frozen['recipe']['epochs']
    for name in ['best','last']:
        checkpoint=out/f'runs/frozen_p3/weights/{name}.pt'
        assert file_sha(checkpoint)==proof['checkpoints'][name]['sha256']
        assert proof['checkpoints'][name]['passed']
    return frozen


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=original.DEFAULT_OUT)
    parser.add_argument('--check',action='store_true',help='Verify recovery prerequisites on CPU; do not evaluate')
    parser.add_argument('--from-predictions',action='store_true',help='CPU-only: finish from both saved prediction JSON files')
    args=parser.parse_args()
    frozen=verify_existing_run(args.out)
    print('[recovery] completed training and unchanged best/last weights verified',flush=True)
    if args.check:
        print('[check PASS] no training or GPU evaluation started',flush=True)
        return
    if args.from_predictions:
        from insplad_frozen_p3_cached_report import finish
        finish(args.out,frozen)
        return
    json_write(args.out/'RECOVERY_MANIFEST.json',dict(
        original_frozen_inputs_sha256=file_sha(args.out/'FROZEN_INPUTS.json'),
        recovery_code_sha256=file_sha(Path(__file__)),
        validator_source=inspect.getfile(DetectionValidator),
        validator_source_sha256=file_sha(Path(inspect.getfile(DetectionValidator))),
        corrections=['NumPy metric scalars converted losslessly to JSON-native numbers',
                     'Prediction JSON class IDs mapped from one-based IDs to original InsPLAD IDs for fixed-threshold recall'],
        training_restarted=False,training_code_unchanged=True,
        prediction_category_convention='JSON category_id=model_class_id+1',
        frozen_best_sha256=file_sha(args.out/'runs/frozen_p3/weights/best.pt')))
    original.json_write=json_write
    original.fixed_recall=corrected_recall
    recipe=frozen['recipe']
    # Hard-wire evaluate: no route to train(), resume or optimizer updates.
    data=Path(next(iter(frozen['annotation_sha256']))).parent.parent
    sys.argv=[str(Path(original.__file__)),'--mode','evaluate','--out',str(args.out),'--data',str(data),
              '--epochs',str(recipe['epochs']),'--imgsz',str(recipe['imgsz']),'--batch',str(recipe['batch'])]
    original.main()


if __name__=='__main__':
    main()
