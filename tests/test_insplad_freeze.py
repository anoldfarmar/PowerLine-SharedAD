import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'scripts'))

#!/usr/bin/env python3
"""CPU checks: actual YOLO prefix freezing, gradient routing, and recall matching."""
import unittest
import json
import tempfile
from pathlib import Path
import torch
from PIL import Image
from ultralytics import YOLO
from insplad_freeze_utils import (INIT,assert_locked,assert_prefix_equal,fixed_recall,
                                  lock_prefix,prefix_features,prefix_state)


class FreezeTest(unittest.TestCase):
    def test_actual_yolo_prefix_and_later_gradients(self):
        torch.set_num_threads(2)
        m=YOLO(str(INIT)).model.float()
        # Checkpoint loading disables grads; the real trainer re-enables the
        # unfrozen parameters during _setup_train(). Mirror that step here.
        for p in m.parameters():
            p.requires_grad_(True)
        reference=prefix_state(m)
        m.train();lock_prefix(m);assert_locked(m)
        x=torch.rand(2,3,64,64)
        a=prefix_features(m,x)
        # A downstream backward pass without an optimizer step: no model training.
        y=x
        for layer in m.model[:6]:
            y=layer(y)
        y.square().mean().backward()
        self.assertTrue(all(p.grad is None for layer in m.model[:5] for p in layer.parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in m.model[5].parameters()))
        assert_prefix_equal(m,reference)
        torch.testing.assert_close(a,prefix_features(m,x),atol=0,rtol=0)
        key=next(k for k in reference if k.endswith('running_mean'))
        m.state_dict()[key].add_(0.1)
        with self.assertRaisesRegex(AssertionError,'Frozen prefix changed'):
            assert_prefix_equal(m,reference)

    def test_recall_keeps_missed_gt_and_handles_class_ids(self):
        images={1:dict(file_name='001.jpg',width=1280,height=720)}
        anns=[dict(category_id=2,image_id=1,bbox=[10,10,10,10]),
              dict(category_id=2,image_id=1,bbox=[100,100,10,10])]
        predictions=[dict(category_id=0,image_id=1,bbox=[10,10,10,10],score=.9),
                     dict(category_id=0,image_id=1,bbox=[10,10,10,10],score=.8)]
        r=fixed_recall(anns,images,predictions,[2],{0:2})
        self.assertEqual(r['per_class']['2']['small']['gt'],2)
        self.assertEqual(r['per_class']['2']['small']['recall'],.5)
        self.assertEqual(r['precision'],.5)
        # The wrong class is not credited even with perfect box overlap.
        r=fixed_recall(anns,images,predictions,[2],{0:1})
        self.assertEqual(r['per_class']['2']['small']['recall'],0)

    def test_real_trainer_setup_without_training(self):
        from insplad_frozen_p3_trainer import FrozenP3Trainer
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);ds=root/'dataset'
            for split in ['train','val']:
                (ds/'images'/split).mkdir(parents=True)
                (ds/'labels'/split).mkdir(parents=True)
                for i in range(2):
                    Image.new('RGB',(64,64),(60+40*i,90,120)).save(ds/'images'/split/f'{i}.jpg')
                    (ds/'labels'/split/f'{i}.txt').write_text('0 0.5 0.5 0.5 0.5\n')
            import yaml
            data=ds/'data.yaml'
            data.write_text(yaml.safe_dump(dict(path=str(ds),train='images/train',val='images/val',
                           names=[f'class{i}' for i in range(14)])))
            (root/'FROZEN_INPUTS.json').write_text(json.dumps(dict(probe_crops=[
                dict(path=str(ds/'images/train/0.jpg'),xyxy=[0,0,64,64])])) )
            trainer=FrozenP3Trainer(overrides=dict(model=str(INIT),data=str(data),epochs=1,imgsz=64,
                batch=2,device='cpu',workers=0,freeze=5,amp=False,channels_last=False,compile=False,
                plots=False,cache=False,project=str(root/'runs'),name='setup',exist_ok=True,verbose=False))
            trainer._setup_train()
            trainer._model_train()
            assert_locked(trainer.model)
            assert_prefix_equal(trainer.model,trainer.reference)
            self.assertEqual(trainer.step_checks,0)
            self.assertTrue(any(p.requires_grad for p in trainer.model.model[5].parameters()))


if __name__=='__main__':
    unittest.main()
