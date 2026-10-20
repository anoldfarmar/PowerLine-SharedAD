#!/usr/bin/env python3
"""User-run InsPLAD small-crop A/B/C diagnostic, fixed 10N/ref3001.

No training, no original-image mapping, no annotation/split changes. prepare
and --cpu-check never use CUDA; run executes gates before formal scoring.
"""
import argparse
import csv
import hashlib
import json
import os
import resource
import shutil
import time
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score
from torchvision.ops import roi_align

ROOT = Path(__file__).resolve().parents[1]
BASE = Path('/root/autodl-tmp/version10-12')
TASK = 'InsPLAD-sharedROI-ABC-ref3001-v1'
ASSET_CLASS = {'glass-insulator': 7, 'lightning-rod-suspension': 5,
               'polymer-insulator-upper-shackle': 11, 'vari-grip': 9,
               'yoke-suspension': 1}
SOURCE_NAMES = ['insplad_shared_roi.py', 'm3_common.py', 'm3score_pcverify.py']
COCO_SHA = 'f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36'
DET_SHA = '7a7118fd4bcedc16dead2e50bdb53af5bcb5215a1785c9ab5583fafa8d318d00'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def state_hash(model):
    h = hashlib.sha256()
    for key, tensor in sorted(model.state_dict().items()):
        h.update(key.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def deterministic():
    torch.set_num_threads(4)
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def close(a, b):
    torch.testing.assert_close(a, b, atol=1e-4, rtol=1e-4)
    return float((a - b).abs().max())


def vectors(feature):
    assert feature.shape[1:] == (64, 40, 40)
    result = F.normalize(feature.flatten(2).transpose(1, 2), dim=2)
    assert torch.isfinite(result).all()
    close(result.norm(dim=2), torch.ones_like(result[:, :, 0]))
    return result


def roi(feature, boxes):
    # Boxes are in the 320x320 network input, not original crop coordinates.
    ids = torch.arange(len(feature), device=feature.device, dtype=torch.float32)[:, None]
    return roi_align(feature, torch.cat([ids, boxes], 1), (40, 40),
                     spatial_scale=1 / 8, sampling_ratio=1, aligned=True)


def select_box(detections, expected):
    """No anomaly label/score is available here. Stable explicit tie breaking."""
    assert torch.isfinite(detections).all(), 'Non-finite detector output'
    rows = detections.detach().cpu().tolist()
    valid = []
    for x1, y1, x2, y2, conf, cls in rows:
        if int(cls) != expected:
            continue
        box = [max(0., min(320., v)) for v in [x1, y1, x2, y2]]
        if box[2] > box[0] and box[3] > box[1]:
            valid.append((conf, box))
    valid.sort(key=lambda item: (-item[0], *item[1]))
    conf, box = valid[0] if valid else (None, [0., 0., 320., 320.])
    return dict(box_input_xyxy=box, confidence=conf, fallback=not bool(valid),
                expected_class=expected, expected_class_candidates=len(valid),
                all_class_detections=rows)


class Shared:
    def __init__(self, path, device):
        from ultralytics import YOLO
        self.device = device
        self.model = YOLO(str(path)).model.to(device).float().eval()
        self.model.requires_grad_(False)
        assert self.model.model[-1].nc == 14
        self.calls = 0
        self.feature = None
        self.handle = self.model.model[4].register_forward_hook(self.capture)

    def capture(self, module, inputs, output):
        self.calls += 1
        self.feature = output

    @torch.inference_mode()
    def extract(self, paths, asset):
        import m3_common as common
        from ultralytics.utils.nms import non_max_suppression
        transform = common.make_transform(320)
        images = []
        native = []
        for path in paths:
            with Image.open(path) as im:
                native.append(list(im.size))
                images.append(transform(im.convert('RGB')))
        inputs = torch.stack(images).to(self.device)
        self.feature = None
        before = self.calls
        prediction = self.model(inputs)
        assert self.calls == before + 1, 'P3 must execute exactly once per full detection forward'
        feature = self.feature
        assert feature is not None
        original = vectors(feature)
        full_boxes = torch.tensor([[0., 0., 320., 320.]], device=self.device).repeat(len(paths), 1)
        full = vectors(roi(feature, full_boxes))
        # With this aligned, single-centre configuration a full-image ROI is
        # an identity. Check EVERY batch, including cache-producing batches.
        identity_diff = close(original, full)
        assert torch.equal(original, full), 'Full-image ROI must be an exact identity for B cache aliasing'
        det = non_max_suppression(prediction, conf_thres=.25, iou_thres=.7,
                                  max_det=300, nc=14, max_time_img=60.)
        assert len(det) == len(paths)
        records = [select_box(rows, ASSET_CLASS[asset]) for rows in det]
        boxes = torch.tensor([r['box_input_xyxy'] for r in records], device=self.device)
        selected = vectors(roi(feature, boxes))
        for rec, size in zip(records, native):
            b = rec['box_input_xyxy']
            rec['native_size_wh'] = size
            rec['box_native_xyxy'] = [b[0]*size[0]/320, b[1]*size[1]/320,
                                      b[2]*size[0]/320, b[3]*size[1]/320]
            rec['box_area_fraction'] = (b[2]-b[0])*(b[3]-b[1])/(320*320)
        self.feature = None
        return dict(A=original.cpu(), B=full.cpu(), C=selected.cpu(),
                    detections=records, full_roi_maxdiff=identity_diff)


def prepare(args):
    import ultralytics
    import torchvision
    import m3score_pcverify as pv
    original = read(args.baseline / 'FROZEN_INPUTS.json')
    assert original['frozen_id'] == digest({k:v for k,v in original.items() if k != 'frozen_id'})
    assert read(args.baseline / 'REPRO_AND_CORRECTNESS.json')['passed']
    assert read(args.detector / 'FREEZE_CORRECTNESS.json')['passed']
    det_inputs = read(args.detector / 'FROZEN_INPUTS.json')
    checkpoint = args.detector / 'runs/frozen_p3/weights/best.pt'
    coco = Path(original['weights']['COCO'])
    assert sha(checkpoint) == DET_SHA and sha(coco) == COCO_SHA
    audit = read(args.detector / 'FREEZE_CORRECTNESS.json')['checkpoints']['best']
    assert audit['sha256'] == DET_SHA and audit['p3_max_abs_diff'] == 0
    for name in ['m3_common.py', 'm3score_pcverify.py']:
        assert sha(ROOT / 'scripts' / name) == original['code_sha256'][name]
    data = {}
    hashes = {}
    sources = {}
    query_file = ROOT/'data/splits/insplad-fd/query_k5.json'
    support_file = ROOT/'outputs/M3full-frozen/SUPPORT_LISTS.json'
    for path in [query_file,support_file]:
        assert sha(path) == original['source_sha256'][str(path)]
    queries, supports = read(query_file), read(support_file)
    for asset, d in original['data'].items():
        assert asset in ASSET_CLASS
        cls = ASSET_CLASS[asset]
        assert det_inputs['class_mapping'][str(cls+1)]['new_id'] == cls
        assert det_inputs['class_mapping'][str(cls+1)]['name'].replace(' ', '-') == asset
        support = d['supports']['10N']['3001']
        assert support == supports[f'{asset}_ref3001']['N10']
        assert d['normal'] == list(dict.fromkeys(queries['normal_query'][f'{asset}/good']))
        for fault, paths in d['defects'].items():
            assert paths == list(dict.fromkeys(queries['defect_query'][f'{asset}/{fault}']))
        assert len(support) == len(set(support)) == 10
        assert set(support).isdisjoint(d['query'])
        assert set(d['query']) == set(d['normal']) | {p for ps in d['defects'].values() for p in ps}
        assert set(d['normal']).isdisjoint(p for ps in d['defects'].values() for p in ps)
        data[asset] = dict(support=support, normal=d['normal'], defects=d['defects'], query=d['query'])
        for path in support + d['query']:
            actual = sha(path)
            assert actual == original['image_sha256'][path], f'Image changed: {path}'
            hashes[path] = actual
        for fault in d['defects']:
            path = args.baseline / 'raw_scores' / f'{asset}__{fault}_COCO-T8_10N_ref3001.json'
            record = read(path)
            auc(record, d['normal'], d['defects'][fault])
            assert record['frozen_id'] == original['frozen_id']
            sources[str(path)] = sha(path)
    args.out.mkdir(parents=True, exist_ok=True)
    deps = [query_file, support_file, args.baseline/'FROZEN_INPUTS.json', args.baseline/'REPRO_AND_CORRECTNESS.json',
            args.detector/'FROZEN_INPUTS.json', args.detector/'FREEZE_CORRECTNESS.json']
    spec = dict(task=TASK, data=data, baseline=str(args.baseline), detector=str(args.detector),
                weights=dict(coco=str(coco), detector=str(checkpoint)),
                weight_sha256=dict(coco=COCO_SHA, detector=DET_SHA), image_sha256=hashes,
                source_sha256={**sources, **{str(p): sha(p) for p in deps}},
                code_sha256={n: sha(ROOT/'scripts'/n) for n in SOURCE_NAMES},
                environment=dict(python=__import__('sys').executable, torch=torch.__version__,
                                 torchvision=torchvision.__version__, ultralytics=ultralytics.__version__,
                                 patchcore=pv.source_meta()),
                protocol=dict(ref=3001, shots=10, batch=args.batch, input='PIL RGB Resize((320,320)) ToTensor; no LetterBox',
                              layer=4, feature_shape=[1600,64], normalized='channel L2 AFTER ROI extraction',
                              roi=dict(output_size=[40,40], spatial_scale=.125, aligned=True, sampling_ratio=1),
                              detection=dict(conf=.25, iou=.7, max_det=300, class_aware_nms=True,
                                             expected_classes=ASSET_CLASS, selection='highest confidence in known asset class; lexicographic box tie break',
                                             fallback='full-image ROI; never drop support/query'),
                              bank='official KCenterGreedy ratio .1 seed7001; support-only, per arm',
                              score='official exact Euclidean 1NN; mean top8 of 1600 patch distances',
                              precision='FP32; TF32 false', atol=1e-4, rtol=1e-4))
    spec['frozen_id'] = digest(spec)
    path = args.out/'FROZEN_INPUTS.json'
    if path.exists():
        assert read(path) == spec, 'Frozen inputs/code/environment differ; use fresh --out'
    write(path, spec)
    estimated = sum(len(d['support'])+len(d['query']) for d in data.values())*1600*64*4*2
    remaining = max(0, estimated-sum(p.stat().st_size for p in (args.out/'cache').rglob('*.npz')))
    free = shutil.disk_usage(args.out).free
    assert free > remaining + (1 << 30), f'Insufficient disk: need {remaining+(1<<30)}, have {free}'
    status = dict(ready=True, gpu_run_started=False, images=len(hashes), combinations=7,
                  shots=10, ref=3001, expected_score_files=21, feature_cache_bytes_estimate=estimated,
                  free_disk_bytes=free, batch=args.batch)
    write(args.out/'PREPARE.json', status)
    print(json.dumps(status, indent=2), flush=True)
    return spec


def auc(record, normal, defect):
    assert set(record['normal']) == set(normal) and set(record['defect']) == set(defect)
    assert set(normal).isdisjoint(defect) and normal and defect
    n = np.array([record['normal'][p] for p in normal], dtype=np.float64)
    d = np.array([record['defect'][p] for p in defect], dtype=np.float64)
    assert np.isfinite(n).all() and np.isfinite(d).all()
    sklearn = float(roc_auc_score([0]*len(n)+[1]*len(d), np.r_[n,d]))
    independent = float(((d[:,None] > n).sum()+.5*(d[:,None] == n).sum())/(len(d)*len(n)))
    assert abs(sklearn-independent) < 1e-12
    if 'auroc' in record:
        assert abs(record['auroc']-independent) < 1e-12
    return sklearn


class FeatureCache:
    def __init__(self, args, frozen, runner, asset):
        self.folder = args.out/'cache/features'/asset
        self.folder.mkdir(parents=True, exist_ok=True)
        d = frozen['data'][asset]
        self.paths = d['support'] + d['query']
        assert len(self.paths) == len(set(self.paths))
        self.lookup = {}
        self.entries = []
        self.loaded = None
        self.loaded_path = None
        for i in range(0, len(self.paths), args.batch):
            paths = self.paths[i:i+args.batch]
            file = self.folder/f'{i:06d}.npz'
            meta = file.with_suffix('.json')
            binding = dict(frozen_id=frozen['frozen_id'], paths=paths, batch=args.batch,
                           geometry=frozen['protocol']['roi'])
            if file.exists() and meta.exists():
                entry = read(meta)
                assert entry['binding'] == binding and entry['sha256'] == sha(file), f'Stale cache: {file}'
            else:
                result = runner.extract(paths, asset)
                tmp = file.with_suffix('.tmp.npz')
                np.savez(tmp, A=result['A'].numpy(), C=result['C'].numpy())
                tmp.replace(file)
                entry = dict(binding=binding, sha256=sha(file), file=str(file),
                             detections=result['detections'], full_roi_maxdiff=result['full_roi_maxdiff'],
                             B='exact alias of A, actual full ROI equality verified before writing')
                write(meta, entry)
            self.entries.append(entry)
            for j, path in enumerate(paths):
                self.lookup[path] = (file, j)
            if i % (args.batch*20) == 0:
                print(f'[features] {asset} {min(i+args.batch,len(self.paths))}/{len(self.paths)}', flush=True)
        self.records = {p:r for e in self.entries for p,r in zip(e['binding']['paths'], e['detections'])}
        assert set(self.records) == set(self.paths)

    def get(self, paths, arm):
        key = 'A' if arm == 'B' else arm
        rows = []
        for path in paths:
            file, index = self.lookup[path]
            if self.loaded_path != file:
                with np.load(file, allow_pickle=False) as stored:
                    self.loaded = {k: stored[k] for k in ['A','C']}
                self.loaded_path = file
            rows.append(self.loaded[key][index])
        return torch.from_numpy(np.stack(rows).copy())


def scores(pm, feature):
    device = pm.memory_bank.device
    with torch.inference_mode():
        dist, _ = pm.nearest_neighbors(feature.to(device).reshape(-1,64), 1)
        result = dist.reshape(len(feature),1600).topk(8, dim=1).values.mean(1).cpu()
    assert torch.isfinite(result).all()
    return result


def prefix_gate(frozen, runner, paths):
    from ultralytics import YOLO
    import m3_common as common
    coco = YOLO(frozen['weights']['coco']).model.to(runner.device).float().eval()
    prefixes = tuple(f'model.{i}.' for i in range(5))
    current = {k:v for k,v in runner.model.state_dict().items() if k.startswith(prefixes)}
    reference = {k:v for k,v in coco.state_dict().items() if k.startswith(prefixes)}
    assert current.keys() == reference.keys()
    assert all(torch.equal(current[k],reference[k]) for k in current)
    transform = common.make_transform(320)
    images = []
    for path in paths:
        with Image.open(path) as im:
            images.append(transform(im.convert('RGB')))
    x = torch.stack(images).to(runner.device)
    with torch.inference_mode():
        a, b = x, x
        for layer in coco.model[:5]:
            a = layer(a)
        for layer in runner.model.model[:5]:
            b = layer(b)
    result = dict(prefix_parameters_and_bn_exact=True, normalized_p3_maxdiff=close(vectors(a),vectors(b)))
    del coco
    return result


def smoke(frozen, runner):
    evidence = {}
    representative = frozen['data']['yoke-suspension']['support'][:2]
    evidence['coco_prefix'] = prefix_gate(frozen, runner, representative)
    for asset, d in frozen['data'].items():
        paths = d['support'][:1]+d['normal'][:2]+next(iter(d['defects'].values()))[:2]
        many = runner.extract(paths, asset)
        singles = [runner.extract([p], asset) for p in paths]
        g = {}
        for arm in 'ABC':
            g[f'{arm}_batch_feature_maxdiff'] = close(many[arm], torch.cat([s[arm] for s in singles]))
        for a,b in zip(many['detections'], [s['detections'][0] for s in singles]):
            assert a['fallback'] == b['fallback']
            g['box_batch_maxdiff'] = max(g.get('box_batch_maxdiff',0),
                close(torch.tensor(a['box_input_xyxy'])/320, torch.tensor(b['box_input_xyxy'])/320)*320)
        repeated = runner.extract(paths, asset)
        g['repeat_C_feature_maxdiff'] = close(many['C'],repeated['C'])
        assert many['detections'] == repeated['detections']
        g['same_forward_p3_and_full_roi_verified'] = True
        evidence[asset] = g
    return evidence


def bank_for(args, frozen, cache, asset, arm):
    import m3score_pcverify as pv
    file = args.out/'cache/banks'/f'{asset}_{arm}.npy'
    meta = file.with_suffix('.json')
    support = frozen['data'][asset]['support']
    binding = dict(frozen_id=frozen['frozen_id'], asset=asset, arm=arm, support=support, seed=7001)
    if file.exists() and meta.exists():
        entry = read(meta)
        assert entry['binding'] == binding and entry['sha256'] == sha(file)
        pm = pv._cls()(layers=['layer2'], backbone='wide_resnet50_2', pre_trained=False, num_neighbors=9).cuda().eval()
        pm.memory_bank = torch.from_numpy(np.load(file, allow_pickle=False)).cuda()
        assert list(pm.memory_bank.shape) == [1600,64]
    else:
        pm, info = pv.make_pm(lambda ps: cache.get(ps,arm), support, 7001)
        assert info['n_support_patches'] == 16000 and info['n_bank'] == 1600
        file.parent.mkdir(parents=True, exist_ok=True)
        tmp = file.with_suffix('.tmp.npy')
        np.save(tmp, pm.memory_bank.cpu().numpy(), allow_pickle=False)
        tmp.replace(file)
        entry = dict(binding=binding, sha256=sha(file), file=str(file), bank=info,
                     support_boxes=[cache.records[p] for p in support] if arm=='C' else 'full image')
        write(meta, entry)
    return pm, entry


def fallback_summary(records, paths):
    count = sum(records[p]['fallback'] for p in paths)
    area = [records[p]['box_area_fraction'] for p in paths]
    return dict(total=len(paths), fallback=count, fallback_rate=count/len(paths),
                predicted_box_count=len(paths)-count, mean_roi_area_fraction=float(np.mean(area)))


def report(args, frozen, complete=True):
    rows = []
    for asset,d in frozen['data'].items():
        for fault, defect in d['defects'].items():
            vals = {}
            for arm in 'ABC':
                path = args.out/'raw_scores'/f'{asset}__{fault}_{arm}.json'
                record = read(path)
                assert record['frozen_id'] == frozen['frozen_id'] and record['arm'] == arm
                vals[arm] = auc(record,d['normal'],defect)
            rows.append(dict(combination=f'{asset}/{fault}', **vals,
                             C_minus_A_pp=100*(vals['C']-vals['A']), B_minus_A_pp=100*(vals['B']-vals['A'])))
    with (args.out/'performance_comparison.csv').open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    macro = {arm:float(np.mean([r[arm] for r in rows])) for arm in 'ABC'}
    write(args.out/'metrics.json',dict(completed=complete, configurations=len(rows)*3, combinations=len(rows),
                                     ref=3001,shots=10,rows=rows,macro=macro))
    lines = ['# InsPLAD small-image shared P3 A/B/C','',
             '10N / ref3001 only. AUROC; one support bank per asset and arm, no training.', '',
             '| Asset / fault | A: original COCO top8 | B: full-image ROI | C: predicted ROI | C−A (pp) |',
             '|---|---:|---:|---:|---:|']
    for row in rows+[dict(combination='MACRO (7 pairs)',**macro,C_minus_A_pp=100*(macro['C']-macro['A']))]:
        lines.append(f"| {row['combination']} | {row['A']:.4f} | {row['B']:.4f} | {row['C']:.4f} | {row['C_minus_A_pp']:+.2f} |")
    lines += ['', 'A is re-scored from the identical frozen COCO prefix and checked against all historical per-image scores.',
              'B is an identity/control under aligned=True, sampling_ratio=1, full 320 ROI and output40.',
              'C uses the highest-confidence prediction of the known asset class; class comes from the benchmark asset, not the fault label.',
              'This is class-conditional component diagnosis, not autonomous unknown-class end-to-end recognition.',
              'Missing/invalid expected-class detections fall back to the full image for BOTH supports and queries; none are dropped.',
              'Inspect DETECTION_SUMMARY.json and detections/ to distinguish real ROI use from fallback.',
              'No bbox GT: no detection mAP, localization accuracy or detection recall claim on these crops.',
              '320-square crop input differs from detector training1280/LetterBox; detection domain/scale shift is a limitation.',
              'Small pre-crops do not establish performance on original1920x1080 scenes. One ref is not a robustness/significance study.',
              'All arms have channel-L2 P3, 1600-patch output, normal-only KCenter .1 seed7001 and exact Euclidean top8.', '']
    (args.out/'PERFORMANCE_TABLE.md').write_text('\n'.join(lines))


def run(args, frozen):
    import m3score_pcverify as pv
    assert torch.cuda.is_available(), 'Use configured CUDA environment; no silent CPU formal run'
    start = time.monotonic()
    runner = Shared(frozen['weights']['detector'],'cuda')
    before = state_hash(runner.model)
    gates = dict(passed=False, smoke=smoke(frozen,runner), banks={})
    write(args.out/'REPRO_AND_CORRECTNESS.json',gates)
    costs = {}
    banks = {}
    caches = {}
    detections = {}
    for asset,d in frozen['data'].items():
        tick = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        cache = FeatureCache(args,frozen,runner,asset)
        torch.cuda.synchronize()
        feature_seconds = time.monotonic()-tick
        caches[asset] = cache.entries
        write(args.out/'CACHE_MANIFEST.json',caches)
        write(args.out/'detections'/f'{asset}.json',cache.records)
        detections[asset] = dict(support=fallback_summary(cache.records,d['support']),
                                 normal=fallback_summary(cache.records,d['normal']),
                                 defects={f:fallback_summary(cache.records,ps) for f,ps in d['defects'].items()})
        write(args.out/'DETECTION_SUMMARY.json',detections)
        # Cache/live binding and batch feature correctness, even after resume.
        live_paths = d['support'][:1]+d['normal'][:1]+next(iter(d['defects'].values()))[:1]
        live = runner.extract(live_paths,asset)
        for arm in 'ABC':
            close(live[arm],cache.get(live_paths,arm))
        arm_scores = {}
        bank_time = time.monotonic()
        bank_a = None
        for arm in 'ABC':
            print(f'[bank] {asset} {arm}: 10 normal supports -> 1600 patches',flush=True)
            pm, entry = bank_for(args,frozen,cache,asset,arm)
            if arm == 'A':
                bank_a = pm.memory_bank.detach().clone()
            if arm == 'B':
                assert torch.equal(pm.memory_bank,bank_a), 'Full-ROI B bank must equal A bank'
            evidence = {}
            if asset == 'yoke-suspension' and arm == 'C':
                other, _ = pv.make_pm(lambda ps: cache.get(ps,arm),d['support'],7001)
                evidence['repeat_bank_maxdiff'] = close(pm.memory_bank,other.memory_bank)
                assert torch.equal(pm.memory_bank,other.memory_bank)
                del other
            probe = cache.get(live_paths,arm)
            evidence['score_batch_maxdiff'] = close(scores(pm,probe),torch.cat([scores(pm,p[None]) for p in probe]))
            with torch.inference_mode():
                # Use held-out normal query patches, not support patches that
                # may be selected into the bank (exact self matches suffer
                # cancellation in the official float32 Euclidean kernel).
                q = probe[1,:17].cuda()
                actual, _ = pm.nearest_neighbors(q,1)
                independent = torch.cdist(q.double(),pm.memory_bank.double()).min(1).values
            evidence['nn_float64_maxdiff'] = close(actual.double(),independent)
            vals = []
            for i in range(0,len(d['query']),args.batch):
                vals.extend(scores(pm,cache.get(d['query'][i:i+args.batch],arm)).tolist())
            arm_scores[arm] = dict(zip(d['query'],vals))
            if arm == 'B':
                evidence['A_B_score_maxdiff'] = close(torch.tensor(list(arm_scores['A'].values())),torch.tensor(vals))
            for fault,defect in d['defects'].items():
                record = dict(frozen_id=frozen['frozen_id'],asset=asset,fault=fault,arm=arm,
                              ref=3001,shots=10,bank_sha256=entry['sha256'],
                              normal={p:arm_scores[arm][p] for p in d['normal']},
                              defect={p:arm_scores[arm][p] for p in defect})
                record['auroc'] = auc(record,d['normal'],defect)
                if arm == 'A':
                    previous = read(args.baseline/'raw_scores'/f'{asset}__{fault}_COCO-T8_10N_ref3001.json')
                    paths = d['normal']+defect
                    old = [previous['normal'][p] if p in previous['normal'] else previous['defect'][p] for p in paths]
                    evidence[f'historical_{fault}_score_maxdiff'] = close(
                        torch.tensor([arm_scores[arm][p] for p in paths],dtype=torch.float64),torch.tensor(old,dtype=torch.float64))
                write(args.out/'raw_scores'/f'{asset}__{fault}_{arm}.json',record)
                print(f"[score] {asset}/{fault} {arm} AUROC={record['auroc']:.6f}",flush=True)
            banks[f'{asset}_{arm}'] = entry
            gates['banks'][f'{asset}_{arm}'] = evidence
            write(args.out/'BANK_MANIFEST.json',banks)
            write(args.out/'REPRO_AND_CORRECTNESS.json',gates)
            del pm
        torch.cuda.synchronize()
        costs[asset] = dict(feature_stage_wall_seconds=feature_seconds,
                           bank_and_scoring_wall_seconds=time.monotonic()-bank_time,
                           peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                           images=len(cache.paths), full_forward_batches=len(cache.entries),
                           C_support_fallback_rate=detections[asset]['support']['fallback_rate'])
        del cache,bank_a
    assert state_hash(runner.model) == before, 'Model parameters/BN mutated'
    gates.update(passed=True,model_state_unchanged=True,all_queries_preserved=True)
    write(args.out/'REPRO_AND_CORRECTNESS.json',gates)
    report(args,frozen)
    write(args.out/'RESOURCE_COST.json',dict(invocation_wall_seconds=time.monotonic()-start,stages=costs,
          gpu=torch.cuda.get_device_name(0), host_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
          cache_bytes=sum(p.stat().st_size for p in (args.out/'cache').rglob('*') if p.is_file()),
          note='Stage wall times include CPU/I/O and cache hits; not isolated GPU kernel times; correctness smoke excluded from per-asset times.'))
    write(args.out/'COMPLETED.json',dict(completed=True,score_files=21,ref=3001,shots=10))
    print(f'[done] {args.out}/PERFORMANCE_TABLE.md',flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['prepare','run','report'],default='prepare')
    parser.add_argument('--out',type=Path,default=BASE/TASK)
    parser.add_argument('--baseline',type=Path,default=BASE/'InsPLAD-COCO-top8-full-v1')
    parser.add_argument('--detector',type=Path,default=BASE/'InsPLAD-frozenP3-14class-v1')
    parser.add_argument('--batch',type=int,default=16)
    parser.add_argument('--cpu-check',action='store_true')
    args = parser.parse_args()
    assert 1 <= args.batch <= 32
    deterministic()
    if args.mode == 'report':
        frozen = read(args.out/'FROZEN_INPUTS.json')
        assert read(args.out/'REPRO_AND_CORRECTNESS.json')['passed']
        report(args,frozen)
        return
    frozen = prepare(args)
    if args.cpu_check:
        runner = Shared(frozen['weights']['detector'],'cpu')
        before = state_hash(runner.model)
        asset = 'yoke-suspension'
        paths = frozen['data'][asset]['support'][:2]
        check = dict(smoke_all_assets=smoke(frozen,runner))
        result = runner.extract(paths,asset)
        check['full_roi_maxdiff'] = result['full_roi_maxdiff']
        check['p3_calls'] = runner.calls
        check['shapes'] = {a:list(result[a].shape) for a in 'ABC'}
        assert state_hash(runner.model) == before
        check.update(passed=True,model_state_unchanged=True,gpu_used=False)
        write(args.out/'CPU_PREFLIGHT.json',check)
        print(json.dumps(check,indent=2),flush=True)
    if args.mode == 'run':
        run(args,frozen)


if __name__ == '__main__':
    main()
