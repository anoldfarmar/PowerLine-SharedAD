#!/usr/bin/env python3
"""CPU numerical/geometry/statistics tests; fake CPU pipeline is NOT GPU gating."""
import argparse
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

import insplad_shared_roi as old
import m3share_speed_core as core
from m3share_speed import group
from m3share_speed_report import accuracy_summary, compute_metrics


class Features(torch.nn.Module):
    def forward(self,x):
        return (F.adaptive_avg_pool2d(x,(40,40)).mean(1,keepdim=True)+1).repeat(1,64,1,1)


class Detector(torch.nn.Module):
    def __init__(self):
        super().__init__();self.features=Features()

    def forward(self,x):
        return self.features(x)


class Tests(unittest.TestCase):
    def test_stretch_and_letterbox_coordinates(self):
        boxes=[[0,0,640,360],[160,90,480,270]]
        torch.testing.assert_close(core.native_to_input(boxes,(640,360)),torch.tensor([[0.,0,320,320],[80,80,240,240]]))
        torch.testing.assert_close(core.native_to_input(boxes[:1],(640,360),gain_xy=(.5,.5),pad_xy=(0,70)),
                                   torch.tensor([[0.,70,320,250]]))

    def test_border_crop_rounding(self):
        self.assertEqual(core.crop_box([.1,.1,638.8,356.9],(639,357)),[0,0,639,357])
        self.assertEqual(core.crop_box([637.8,355.5,639,357],(639,357)),[637,355,639,357])

    def test_candidate_filter_sort_and_native_mapping(self):
        rows=[[0,0,80,80,.99,2],[20,20,100,100,.8,1],[0,0,120,120,.8,1],[340,0,350,8,.9,1]]
        result=core.candidate_boxes(rows,1,(640,360))
        self.assertEqual(len(result),2)
        self.assertEqual(result[0]['native_xyxy'],[0.,0.,240.,135.])
        self.assertEqual(result[1]['native_xyxy'],[40.,22.5,200.,112.5])

    def test_actual_chunk_function_vs_direct_cpu(self):
        torch.manual_seed(23)
        query=F.normalize(torch.randn(2,1600,64),dim=-1)
        bank=F.normalize(torch.randn(37,64),dim=-1)
        blocked=core.nearest(query,bank,query_chunk=137,bank_chunk=7)
        direct=torch.cdist(query.double().reshape(-1,64),bank.double()).min(1).values.reshape(2,1600)
        torch.testing.assert_close(blocked.double(),direct,atol=1e-4,rtol=1e-4)

    def test_zero_and_multi_pipeline_counters_on_cpu_fake(self):
        # Test control flow only. No YOLO model is loaded, .cuda is mocked.
        pipeline=core.Pipeline.__new__(core.Pipeline)
        pipeline.model=Detector();pipeline.prefix=Features();pipeline.crop_batch=32
        pipeline.tf=__import__('m3_common').make_transform(320)
        pipeline.p3=None
        pipeline.det_calls=pipeline.det_images=pipeline.crop_calls=pipeline.crop_images=0
        pipeline.model.features.register_forward_hook(pipeline.det_hook)
        pipeline.prefix.register_forward_hook(pipeline.crop_hook)
        rows=[torch.zeros((0,6)),torch.tensor([[0.,0.,320.,320.,.9-i*.001,1] for i in range(35)])]
        tensor_constructor=torch.tensor
        def cpu_tensor(*args,**kwargs):
            if kwargs.get('device')=='cuda':kwargs['device']='cpu'
            return tensor_constructor(*args,**kwargs)
        images=[Image.new('RGB',(639,357),(150,90,40)),Image.new('RGB',(317,598),(30,90,120))]
        bank=F.normalize(torch.randn(19,64),dim=-1)
        with patch.object(torch.Tensor,'cuda',lambda self,*a,**kw:self), \
             patch.object(torch,'tensor',cpu_tensor), \
             patch.object(core,'non_max_suppression',return_value=rows):
            results={m:pipeline.run(images,['yoke-suspension']*2,m,{'yoke-suspension':bank},keep_features=True) for m in core.METHODS}
        for result in results.values():
            self.assertEqual([r['target_box_count'] for r in result['outputs']],[0,35])
            self.assertTrue(result['outputs'][0]['fallback'])
            self.assertEqual(len(result['outputs'][0]['scores']),1)
            self.assertEqual(list(result['features'].shape),[36,1600,64])
        self.assertEqual(results[core.METHODS[0]]['counters']['crop_backbone_calls'],2)
        self.assertEqual(results[core.METHODS[0]]['counters']['crop_images'],36)
        self.assertEqual(results[core.METHODS[1]]['counters']['crop_backbone_calls'],0)
        core.compare_results(results[core.METHODS[0]],results[core.METHODS[1]],scores=False)

    def test_accuracy_never_invents_multi_box_aggregation(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for arm in ['A','C']:
                old.write(root/'raw_scores'/f'asset__rust_{arm}.json',dict(frozen_id='old',normal={'n':.1},defect={'d':.8,'multi':.9}))
            frozen=dict(upstream_frozen_id='old',source_queries={'asset':dict(normal=['n'],defects={'rust':['d','multi']})},
                        frames={'n':{'target_box_count':1},'d':{'target_box_count':1},'multi':{'target_box_count':2}},accuracy_query_ids=['n','d','multi'])
            records={m:{'n':{'scores':[.1]},'d':{'scores':[.9]},'multi':{'scores':[.2,.8]}} for m in core.METHODS}
            result=accuracy_summary(root,frozen,records)
            self.assertIsNone(result['new_all_query_system_auroc'])
            self.assertTrue(result['all_queries_retained'])
            self.assertEqual(result['new_local_arms']['asset/rust']['exactly_one_box'][core.METHODS[0]]['auroc'],1.)
            self.assertIsNone(result['new_local_arms']['asset/rust']['no_box_full_frame_fallback'][core.METHODS[0]]['auroc'])

    def test_metrics_keep_all_rounds_and_empty_groups(self):
        ids=['zero','two'];frozen=dict(speed_ids=ids,frozen_id='frozen',scope='local',frames={p:dict(group=group(n),target_box_count=n) for p,n in zip(ids,[0,2])})
        rows=[]
        for m,ms in zip(core.METHODS,[10.,5.]):
            for phase in ['single_frame','throughput_batch','stage_profile']:
                for p,n in zip(ids,[0,2]):
                    rows.append(dict(method=m,phase=phase,repeat=0,frame_ids=[p],target_boxes=[n],batch=1,
                        elapsed_ms=ms,frozen_id='frozen',effective_regions=max(1,n),
                        counters=dict(full_backbone_calls=1,full_images=1,crop_backbone_calls=1 if m==core.METHODS[0] else 0,crop_images=max(1,n) if m==core.METHODS[0] else 0),
                        stages={'diagnostic_stage_ms':ms/2}))
        protocol=dict(latency_repeats=1,throughput_repeats=1,chosen_batch=1)
        metrics=compute_metrics(frozen,protocol,rows)
        self.assertEqual(metrics['ratios']['local_complete_mean_speedup'],2.)
        self.assertEqual(metrics['no_box_rate'],.5)
        self.assertIsNone(metrics['box_count_latency_ms'][core.METHODS[0]]['>=5']['mean'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path)
    args=parser.parse_args();torch.set_num_threads(4)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    if args.out:
        old.write(args.out,dict(passed=result.wasSuccessful(),tests_run=result.testsRun,
                               failures=len(result.failures),errors=len(result.errors),gpu_used=False,
                               code_sha256={n:old.sha(old.ROOT/'scripts'/n) for n in __import__('m3share_speed').CODE},
                               real_model_forward_executed=False,note='CPU numerical tests and mocked-pipeline control flow; not formal GPU correctness gates'))
    raise SystemExit(0 if result.wasSuccessful() else 1)
