#!/usr/bin/env python3
"""CPU readiness only. Explicitly NOT formal GPU numerical acceptance."""
import json
from pathlib import Path
import subprocess
import sys

from PIL import Image
import torch
from torchvision.ops import roi_align

from cplid_shared_p3 import close, read, validate
from cplid_shared_p3_core import Pipeline, state_hash
from insplad_freeze_utils import json_write


def main():
    root=Path(__file__).resolve().parents[1]
    out=Path('/root/autodl-tmp/version10-12/CPLID-sharedP3-R0R1R2-v1')
    subprocess.run([sys.executable,'-m','unittest','tests/test_cplid_shared_p3.py','-v'],cwd=root,check=True)
    subprocess.run(['bash','-n',str(root/'scripts/cplid_shared_p3_launch.sh')],check=True)
    f=read(out/'FROZEN_INPUTS.json');validate(f)
    import m3score_pcverify as pv
    dependencies=pv.source_meta()
    torch.set_num_threads(4)
    p=f['protocol'];pipe=Pipeline(p['detector'],p['coco'],device='cpu')
    initial=[state_hash(pipe.model),state_hash(pipe.prefix)]
    r=f['training_probes'][0];im=Image.open(r['path']).convert('RGB')
    with torch.inference_mode():
        a=pipe.run(im,'R1');b=pipe.run(im,'R2')
        assert a['boxes']==b['boxes'] and len(a['boxes'])>0
        assert a['features'].shape==b['features'].shape
        assert torch.isfinite(a['features']).all() and torch.isfinite(b['features']).all()
        boxes=[r['xyxy'],[0,0,*im.size]]
        batch_diff=close(pipe.crops(im,boxes,batch=1),pipe.crops(im,boxes,batch=32))
        yy,xx=torch.meshgrid(torch.arange(160.),torch.arange(160.),indexing='ij')
        actual=roi_align((xx+10*yy)[None,None],torch.tensor([[0.,80.,240.,400.,560.]]),
                         (40,40),spatial_scale=.125,aligned=True,sampling_ratio=1)[0,0]
        expected=(torch.arange(40.)+10)[None,:]+10*(torch.arange(40.)+30)[:,None]
        geometry_diff=close(actual,expected)
        assert initial==[state_hash(pipe.model),state_hash(pipe.prefix)]
    result=dict(passed=True,scope='CPU preparation only, not formal GPU gates or anomaly evaluation',
        device='cpu',frozen_identity=f['identity'],unit_tests_passed=4,shell_syntax=True,
        training_probe_only=True,boxes_equal=True,boxes=len(a['boxes']),
        R1_counts=a['counters'],R2_counts=b['counters'],feature_shape=list(a['features'].shape),
        crop_batch_maxdiff=batch_diff,coordinate_ramp_maxdiff=geometry_diff,state_unchanged=True,
        dependencies=dependencies,GPU_evaluation_started=False,formal_GPU_numerical_gates='PENDING_USER_RUN')
    json_write(out/'PREPARATION_CHECKS.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
