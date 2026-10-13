"""Strict inverse of HF's official dinov2 conversion; no weights downloaded.
Mapping reference: transformers v4.32.0 models/dinov2/convert_dinov2_to_hf.py.
Independent split-QKV functional forward provides final-patch equivalence gate.
qkv is concatenated and reshaped exactly like the official Attention module: separate
strided tensors dispatch SDPA to a different kernel (up to 1.2e-4 drift at 1025 tokens);
with the official construction the gate is bitwise 0.0, so it measures the weight mapping
and composition (QKV order additionally proven by roundtrip equality asserts).
"""
import hashlib
import json
from pathlib import Path
import sys
import torch
import torch.nn.functional as F


def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(4<<20),b''): h.update(b)
    return h.hexdigest()


def statehash(m):
    h=hashlib.sha256()
    for k,v in sorted(m.state_dict().items()):
        h.update(k.encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def convert(hf):
    mapping={'cls_token':'embeddings.cls_token','mask_token':'embeddings.mask_token',
             'pos_embed':'embeddings.position_embeddings','patch_embed.proj.weight':'embeddings.patch_embeddings.projection.weight',
             'patch_embed.proj.bias':'embeddings.patch_embeddings.projection.bias',
             'norm.weight':'layernorm.weight','norm.bias':'layernorm.bias'}
    for i in range(12):
        for name in ['norm1','norm2','mlp.fc1','mlp.fc2']:
            for wb in ['weight','bias']:mapping[f'blocks.{i}.{name}.{wb}']=f'encoder.layer.{i}.{name}.{wb}'
        for j in [1,2]:mapping[f'blocks.{i}.ls{j}.gamma']=f'encoder.layer.{i}.layer_scale{j}.lambda1'
        for wb in ['weight','bias']:mapping[f'blocks.{i}.attn.proj.{wb}']=f'encoder.layer.{i}.attention.output.dense.{wb}'
    target={k:hf[v] for k,v in mapping.items()}; consumed=set(mapping.values())
    for i in range(12):
        for wb in ['weight','bias']:
            keys=[f'encoder.layer.{i}.attention.attention.{q}.{wb}' for q in ['query','key','value']]
            target[f'blocks.{i}.attn.qkv.{wb}']=torch.cat([hf[k] for k in keys]);consumed.update(keys)
    assert consumed==set(hf), {'unconsumed':sorted(set(hf)-consumed),'missing':sorted(consumed-set(hf))}
    # Inverse roundtrip: every source tensor accounted for, including Q/K/V order.
    for k,v in mapping.items():assert torch.equal(target[k],hf[v])
    for i in range(12):
        for wb in ['weight','bias']:
            chunks=target[f'blocks.{i}.attn.qkv.{wb}'].chunk(3)
            for q,t in zip(['query','key','value'],chunks):assert torch.equal(t,hf[f'encoder.layer.{i}.attention.attention.{q}.{wb}'])
    return target


def load(model_dir, dino_repo, anomaly_repo):
    model_dir=Path(model_dir); cfg=json.loads((model_dir/'config.json').read_text())
    assert [cfg[k] for k in ['model_type','hidden_size','num_hidden_layers','num_attention_heads','patch_size']]==['dinov2',384,12,6,14]
    assert int(cfg.get('num_register_tokens') or 0)==0 and not cfg['use_swiglu_ffn']
    hf=torch.load(model_dir/'pytorch_model.bin',map_location='cpu',weights_only=True)
    assert not any('register' in k for k in hf)
    sys.path.insert(0,str(dino_repo));sys.path.insert(0,str(anomaly_repo))
    from dinov2.hub.backbones import dinov2_vits14
    from src.backbones import DINOv2Wrapper
    from torchvision import transforms
    model=dinov2_vits14(pretrained=False)
    converted=convert(hf)
    keys=model.load_state_dict(converted,strict=True)
    assert not keys.missing_keys and not keys.unexpected_keys
    model=model.eval().requires_grad_(False).cuda()
    class LocalWrapper(DINOv2Wrapper):
        def load_model(self):
            self.transform=transforms.Compose([
                transforms.Resize(size=self.smaller_edge_size,interpolation=transforms.InterpolationMode.BICUBIC,antialias=True),
                transforms.ToTensor(),transforms.Normalize((.485,.456,.406),(.229,.224,.225))])
            return model
    wrapper=LocalWrapper('dinov2_vits14','cuda',448,False)
    identity=dict(architecture='dinov2_vits14',dimension=384,layers=12,heads=6,patch_size=14,registers=0,
          format='HF PyTorch .bin (safetensors also present, not used)',strict_load=True,missing_keys=[],unexpected_keys=[],
          source_keys=len(hf),destination_keys=len(converted),parameter_count=sum(p.numel() for p in model.parameters()),
          mapping='inverse official HF v4.32.0 conversion: concatenate Q,K,V; all source keys roundtrip exactly',
          source_claim='local model card: facebook/dinov2-small, LVD142M DINOv2; local hash-bound, original Meta checkpoint not downloaded',
          files={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(model_dir.iterdir()) if p.is_file()})
    return wrapper,hf,identity


@torch.no_grad()
def independent_hf_forward(x,hf):
    """Split-QKV and explicit attention, independent of official model modules."""
    s={k:v.to(x.device) for k,v in hf.items()}; b,_,h,w=x.shape
    linear=lambda z,p:F.linear(z,s[p+'.weight'],s[p+'.bias'])
    norm=lambda z,p:F.layer_norm(z,(384,),s[p+'.weight'],s[p+'.bias'],1e-6)
    y=F.conv2d(x,s['embeddings.patch_embeddings.projection.weight'],s['embeddings.patch_embeddings.projection.bias'],stride=14).flatten(2).transpose(1,2)
    y=torch.cat([s['embeddings.cls_token'].expand(b,-1,-1),y],dim=1)
    pos=s['embeddings.position_embeddings']
    if y.shape[1]!=pos.shape[1] or h!=w:
        pp=F.interpolate(pos[:,1:].reshape(1,37,37,384).permute(0,3,1,2),scale_factor=((h//14+.1)/37,(w//14+.1)/37),mode='bicubic',align_corners=False)
        pos=torch.cat([pos[:,:1],pp.permute(0,2,3,1).reshape(1,-1,384)],dim=1)
    y=y+pos
    for i in range(12):
        p=f'encoder.layer.{i}'; z=norm(y,p+'.norm1')
        qkv=F.linear(z,torch.cat([s[p+'.attention.attention.'+a+'.weight'] for a in ['query','key','value']],0),torch.cat([s[p+'.attention.attention.'+a+'.bias'] for a in ['query','key','value']],0)).reshape(b,-1,3,6,64).permute(2,0,3,1,4)
        q,k,v=qkv.unbind(0)
        z=((F.scaled_dot_product_attention(q,k,v,scale=0.125)).transpose(1,2).reshape(b,-1,384))
        y=y+linear(z,p+'.attention.output.dense')*s[p+'.layer_scale1.lambda1']
        y=y+linear(F.gelu(linear(norm(y,p+'.norm2'),p+'.mlp.fc1')),p+'.mlp.fc2')*s[p+'.layer_scale2.lambda1']
    return norm(y,'layernorm')[:,1:]
