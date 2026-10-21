import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'scripts'))

#!/usr/bin/env python3
"""CPU tests only; synthetic geometry, fallbacks, ordering, metrics and cache."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import insplad_shared_roi as subject


class Tests(unittest.TestCase):
    def test_full_roi_identity(self):
        grid=torch.arange(2*64*40*40,dtype=torch.float32).reshape(2,64,40,40)
        boxes=torch.tensor([[0.,0.,320.,320.]]*2)
        self.assertTrue(torch.equal(grid,subject.roi(grid,boxes)))

    def test_roi_coordinates(self):
        # Channel0 is x coordinate, channel1 y. First ROI sample of [80,80,240,240]
        # lands at feature coordinate 9.75 after aligned half-pixel correction.
        yy,xx=torch.meshgrid(torch.arange(40.),torch.arange(40.),indexing='ij')
        f=torch.zeros(1,64,40,40);f[0,0]=xx;f[0,1]=yy
        got=subject.roi(f,torch.tensor([[80.,80.,240.,240.]]))
        self.assertAlmostEqual(float(got[0,0,0,0]),9.75)
        self.assertAlmostEqual(float(got[0,1,0,0]),9.75)

    def test_selection_and_fallback(self):
        rows=torch.tensor([[0,0,100,100,.99,2],[30,30,90,90,.8,1],[-5,-5,80,80,.7,1]])
        result=subject.select_box(rows,1)
        self.assertEqual(result['box_input_xyxy'],[30.,30.,90.,90.])
        self.assertFalse(result['fallback'])
        fallback=subject.select_box(rows,7)
        self.assertTrue(fallback['fallback'])
        self.assertEqual(fallback['box_input_xyxy'],[0.,0.,320.,320.])
        invalid=subject.select_box(torch.tensor([[330,0,350,10,.9,1]]),1)
        self.assertTrue(invalid['fallback'])

    def test_auc_label_binding_and_ties(self):
        self.assertEqual(subject.auc(dict(normal={'n':1.},defect={'d':1.}),['n'],['d']),.5)
        with self.assertRaises(AssertionError):
            subject.auc(dict(normal={'n':0.},defect={'d':1.}),['d'],['n'])

    def test_report_from_bound_scores(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            args=SimpleNamespace(out=root)
            frozen=dict(frozen_id='test',data={'asset':dict(normal=['n'],defects={'fault':['d']})})
            for arm in 'ABC':
                subject.write(root/'raw_scores'/f'asset__fault_{arm}.json',
                              dict(frozen_id='test',arm=arm,normal={'n':0.},defect={'d':1.},auroc=1.))
            subject.report(args,frozen)
            self.assertEqual(subject.read(root/'metrics.json')['configurations'],3)
            self.assertIn('1.0000',(root/'PERFORMANCE_TABLE.md').read_text())

    def test_cache_order_alias_resume_and_tamper(self):
        class Dummy:
            calls=0
            def extract(self,paths,asset):
                self.calls+=1
                f=torch.stack([torch.full((1600,64),float(int(p))) for p in paths])
                return dict(A=f,C=f+10,detections=[{'fallback':False} for p in paths],full_roi_maxdiff=0.)
        with tempfile.TemporaryDirectory() as td:
            args=SimpleNamespace(out=Path(td),batch=2)
            frozen=dict(frozen_id='test',protocol={'roi':'test'},data={'asset':dict(support=['1','2'],query=['3'])})
            dummy=Dummy();cache=subject.FeatureCache(args,frozen,dummy,'asset')
            self.assertEqual(dummy.calls,2)
            self.assertTrue(torch.equal(cache.get(['3','1'],'B')[:,0,0],torch.tensor([3.,1.])))
            self.assertTrue(torch.equal(cache.get(['2'],'C')[:,0,0],torch.tensor([12.])))
            subject.FeatureCache(args,frozen,dummy,'asset')
            self.assertEqual(dummy.calls,2)
            first=Path(cache.entries[0]['file'])
            with first.open('ab') as f:
                f.write(b'changed')
            with self.assertRaises(AssertionError):
                subject.FeatureCache(args,frozen,dummy,'asset')


if __name__=='__main__':
    torch.set_num_threads(4)
    unittest.main()
