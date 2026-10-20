"""Ultralytics trainer that locks original COCO stem/P3 parameters and BN."""
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils.torch_utils import unwrap_model

from insplad_freeze_utils import (INIT, INIT_SHA, assert_locked, assert_prefix_equal,
    file_sha, json_write, lock_prefix, prefix_features, prefix_state, tensor_hash)


def probe_inputs(probes):
    tf=transforms.Compose([transforms.Resize((320,320)),transforms.ToTensor()])
    inputs=[]
    for p in probes:
        with Image.open(p['path']) as im:
            inputs.append(tf(im.convert('RGB').crop(tuple(p['xyxy']))))
    return torch.stack(inputs)


class FrozenP3Trainer(DetectionTrainer):
    def get_model(self,cfg=None,weights=None,verbose=True):
        model=super().get_model(cfg,weights,verbose)
        lock_prefix(model)
        return model

    def _model_train(self):
        super()._model_train()
        lock_prefix(unwrap_model(self.model))
        assert_locked(unwrap_model(self.model))

    def _setup_train(self):
        assert self.args.freeze==5 and not self.args.amp and not self.args.channels_last
        assert file_sha(INIT)==INIT_SHA
        super()._setup_train()
        self.audit_dir=Path(self.save_dir)/'freeze_audit'
        self.audit_dir.mkdir(exist_ok=True)
        manifest=json.loads((Path(self.args.data).parents[1]/'FROZEN_INPUTS.json').read_text())
        reference=YOLO(str(INIT)).model.float().eval().to(self.device)
        self.reference=prefix_state(reference)
        self.inputs=probe_inputs(manifest['probe_crops']).to(self.device)
        self.reference_features=prefix_features(reference,self.inputs)
        model=unwrap_model(self.model);lock_prefix(model)
        assert_locked(model);assert_prefix_equal(model,self.reference)
        torch.testing.assert_close(prefix_features(model,self.inputs),self.reference_features,atol=1e-4,rtol=1e-4)
        trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert trainable>0
        frozen_ids={id(p) for layer in model.model[:5] for p in layer.parameters()}
        # Optimizers may contain frozen parameters, but none may acquire gradients.
        self.prefix_parameters=[p for p in model.parameters() if id(p) in frozen_ids]
        self.tail_start={k:v.detach().cpu().clone() for k,v in model.named_parameters()
                         if k.startswith('model.5.') and v.requires_grad}
        self.step_checks=0;self.epoch_checks=[]
        json_write(self.audit_dir/'START.json',dict(prefix_hash=tensor_hash(self.reference),
            trainable_parameters=trainable,probe_shape=list(self.reference_features.shape),
            frozen_layers=list(range(5)),bn_eval=True,checkpoint_source=str(INIT)))
        self.add_callback('on_train_epoch_end',self.audit_epoch)
        del reference

    def preprocess_batch(self,batch):
        model=unwrap_model(self.model)
        lock_prefix(model);assert_locked(model)
        return super().preprocess_batch(batch)

    def optimizer_step(self):
        assert all(p.grad is None for p in self.prefix_parameters), 'Frozen prefix received a gradient'
        super().optimizer_step()
        self.step_checks+=1
        if self.step_checks==1:
            assert_prefix_equal(unwrap_model(self.model),self.reference)

    def audit_epoch(self,trainer):
        model=unwrap_model(self.model)
        assert_locked(model)
        live_hash=assert_prefix_equal(model,self.reference)
        assert_prefix_equal(self.ema.ema,self.reference)
        features=prefix_features(model,self.inputs)
        torch.testing.assert_close(features,self.reference_features,atol=1e-4,rtol=1e-4)
        self.epoch_checks.append(dict(epoch=self.epoch+1,prefix_hash=live_hash,
           p3_max_abs_diff=float((features-self.reference_features).abs().max()),ema_prefix_exact=True))
        json_write(self.audit_dir/'EPOCH_CHECKS.json',dict(epochs=self.epoch_checks,optimizer_step_checks=self.step_checks))

    def save_model(self):
        assert_prefix_equal(unwrap_model(self.model),self.reference)
        assert_prefix_equal(self.ema.ema,self.reference)
        return super().save_model()

    def final_eval(self):
        model=unwrap_model(self.model);lock_prefix(model)
        assert_prefix_equal(model,self.reference)
        current=model.state_dict()
        changed=any(not torch.equal(current[k].detach().cpu(),v) for k,v in self.tail_start.items())
        assert changed, 'P4 entry layer did not update'
        json_write(self.audit_dir/'TRAIN_END.json',dict(passed=True,prefix_hash=tensor_hash(self.reference),
            p4_entry_updated=changed,optimizer_step_checks=self.step_checks,epochs_audited=len(self.epoch_checks)))
        super().final_eval()


def audit_checkpoint(weights,probes,device='cpu'):
    """Check the actual serialized EMA checkpoint, including BN counters."""
    reference=YOLO(str(INIT)).model.float().eval().to(device)
    candidate=YOLO(str(weights)).model.float().eval().to(device)
    expected=prefix_state(reference)
    h=assert_prefix_equal(candidate,expected)
    inputs=probe_inputs(probes).to(device)
    a=prefix_features(reference,inputs);b=prefix_features(candidate,inputs)
    torch.testing.assert_close(a,b,atol=1e-4,rtol=1e-4)
    return dict(passed=True,weights=str(weights),sha256=file_sha(weights),prefix_hash=h,
                p3_max_abs_diff=float((a-b).abs().max()),shape=list(a.shape),device=str(device))
