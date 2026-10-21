#!/usr/bin/env python3
"""M3SHARE-speed-v1: bounded local analysis, NOT raw-scene end-to-end proof."""
import argparse
from collections import Counter
from datetime import datetime, timezone, timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import random
import resource
import statistics
import subprocess
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
from PIL import Image
import torch

import insplad_shared_roi as old
import m3score_pcverify as pv
from m3share_speed_core import (METHODS, Pipeline, candidate_boxes, compare_results,
                               native_to_input, nearest)

TASK = 'M3SHARE-speed-v1'
SOURCE = old.BASE/old.TASK
CODE = ['m3share_speed.py','m3share_speed_core.py','m3share_speed_report.py','m3share_speed_publish.py']


def group(n):
    return '0' if n==0 else ('1' if n==1 else ('2-4' if n<5 else '>=5'))


def decode(path):
    tick = time.perf_counter()
    with Image.open(path) as im:
        result = im.convert('RGB')
        result.load()
    return result,1000*(time.perf_counter()-tick)


def idle_gpu():
    text = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory',
                                    '--format=csv,noheader,nounits']).decode().strip()
    rows = [r for r in text.splitlines() if r.strip()]
    other = [r for r in rows if r.split(',')[0].strip().isdigit() and int(r.split(',')[0])!=os.getpid()]
    assert not other, f'External GPU compute process present: {other}; do not benchmark competition'
    return text


def freeze(args):
    import torchvision, ultralytics
    # Read device identity without creating a CUDA model/context in prepare.
    gpu_fields=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,driver_version',
                                         '--format=csv,noheader,nounits']).decode().strip().splitlines()[0].split(',')
    gpu_fields=[value.strip() for value in gpu_fields]
    upstream = old.read(args.source/'FROZEN_INPUTS.json')
    assert old.read(args.source/'COMPLETED.json')['completed']
    assert old.read(args.source/'REPRO_AND_CORRECTNESS.json')['passed']
    for name,sha in upstream['code_sha256'].items():
        assert old.sha(old.ROOT/'scripts'/name)==sha
    assert old.sha(upstream['weights']['detector'])==old.DET_SHA
    sources = {str(args.source/n):old.sha(args.source/n) for n in ['FROZEN_INPUTS.json','REPRO_AND_CORRECTNESS.json','BANK_MANIFEST.json','CACHE_MANIFEST.json']}
    frames, supports, speed_ids = {}, {}, []
    for asset,data in upstream['data'].items():
        assert len(data['support'])==len(set(data['support']))==10
        assert set(data['support']).isdisjoint(data['query'])
        saved = old.read(args.source/'detections'/f'{asset}.json')
        sources[str(args.source/'detections'/f'{asset}.json')] = old.sha(args.source/'detections'/f'{asset}.json')
        for fault in data['defects']:
            for arm in ['A','C']:
                path=args.source/'raw_scores'/f'{asset}__{fault}_{arm}.json'
                sources[str(path)]=old.sha(path)
        supports[asset] = data['support']
        for path in data['support']+data['query']:
            assert old.sha(path)==upstream['image_sha256'][path]
            with Image.open(path) as im:
                size = list(im.size)
            assert size==saved[path]['native_size_wh']
            boxes = candidate_boxes(saved[path]['all_class_detections'],old.ASSET_CLASS[asset],size)
            assert len(boxes)==saved[path]['expected_class_candidates']
            labels = [fault for fault,ps in data['defects'].items() if path in ps]
            label = labels[0] if labels else 'good'
            assert len(labels)<=1
            frames[path] = dict(asset=asset,native_size_wh=size,image_label=label,
                                role='support' if path in data['support'] else 'query',
                                sha256=upstream['image_sha256'][path],boxes=boxes,
                                target_box_count=len(boxes),group=group(len(boxes)),
                                box_area_fractions=[(r['native_xyxy'][2]-r['native_xyxy'][0])*(r['native_xyxy'][3]-r['native_xyxy'][1])/(size[0]*size[1]) for r in boxes],
                                raw_scene_identity='UNKNOWN: pre-cropped defect_supervised image, no verified source-frame mapping',
                                label_availability='image-level asset/fault only; no component bbox or pixel anomaly GT')
        ordered = sorted(data['query'],key=lambda p:hashlib.sha256((TASK+'\n'+p).encode()).hexdigest())
        speed_ids.extend(ordered[:64])
    assert len(speed_ids)==320
    raw_annotation = Path('/root/autodl-tmp/data/insplad/annotations/instances_val.json')
    ann = old.read(raw_annotation)
    crop_names = {Path(p).name for p,r in frames.items() if r['role']=='query'}
    overlap = crop_names & {Path(r['file_name']).name for r in ann['images']}
    assert not overlap, 'Investigate exact raw/crop name overlap before declaring unavailable'
    audit_path = old.ROOT/'outputs/M3diag5-testaudit/DATA_AUDIT.md'
    audit = dict(reliable_raw_scene_anomaly_mapping_available=False,
                 exact_filename_overlap_with_detector_val=len(overlap),
                 detector_val_annotation=str(raw_annotation),annotation_sha256=old.sha(raw_annotation),
                 detector_image_metadata_fields=sorted({k for im in ann['images'] for k in im}),
                 prior_audit=str(audit_path),prior_audit_sha256=old.sha(audit_path),
                 caution='Prior bare-DJI collision counts are NOT source identities or reliable leakage proof; only metadata bridge absence is used here.')
    result = dict(task=TASK,scope='LOCAL_PRE_CROPPED_INPUT_ONLY; not original1920x1080 scene end-to-end speed',
                  upstream_frozen_id=upstream['frozen_id'],source_dir=str(args.source),source_sha256=sources,
                  weights=upstream['weights'],weight_sha256=upstream['weight_sha256'],
                  supports=supports,frames=frames,speed_ids=speed_ids,accuracy_query_ids=[p for a,d in upstream['data'].items() for p in d['query']],
                  source_queries=upstream['data'],data_audit=audit,
                  protocol=dict(detector_input='PIL RGB Resize((320,320)) ToTensor, stretch, NO LetterBox',
                     confidence=.25,nms_iou=.7,max_det=300,feature_layer=4,feature_channels=64,
                     roi_size=[40,40],aligned=True,sampling_ratio=1,spatial_scale=.125,
                     crop_input=320,crop_rounding='floor left/top, ceil right/bottom, clip to native image',
                     all_expected_class_boxes=True,box_sort='descending confidence then lexicographic input xyxy',
                     fallback='full native image: CROP performs crop prefix, SHARE takes full-frame ROI; both diagnose, not return-normal',
                     crop_batch=32,coreset_ratio=.1,bank_seed=7001,precision='FP32, no TF32',
                     scoring='channel L2 after extraction; official exact Euclidean NN1, top8 mean',
                     fixed_gate_atol=1e-4,fixed_gate_rtol=1e-4,
                     query_chunk=2048,bank_chunk=512,
                     probe_rules=dict(latency_probe_frames_per_asset=4,warmup_min=20,warmup_max=200,warmup_target_ms=1000,
                        repeats_min=3,repeats_max=10,target_ms_per_arm=5000,throughput_candidates=[1,8,16],
                        batch_choice='max geometric mean paired throughput among batch-gate-passing candidates below80percent VRAM',
                        stage_profile='first8 speed frames from each nonempty box-count bin; separately synchronized')),
                  environment=dict(python=__import__('sys').executable,torch=torch.__version__,torchvision=torchvision.__version__,
                     ultralytics=ultralytics.__version__,numpy=np.__version__,cuda=torch.version.cuda,
                     cudnn=torch.backends.cudnn.version(),gpu=gpu_fields[0],driver=gpu_fields[2],
                     gpu_total_bytes=int(gpu_fields[1])*1024*1024,
                     cpu_threads=4,patchcore=pv.source_meta()),
                  code_sha256={n:old.sha(old.ROOT/'scripts'/n) for n in CODE})
    result['frozen_id'] = old.digest(result)
    args.out.mkdir(parents=True,exist_ok=True)
    target=args.out/'FROZEN_INPUTS.json'
    if target.exists():
        assert old.read(target)==result,'Frozen inputs/code/environment changed; use a new output directory'
    old.write(target,result)
    old.write(args.out/'DATA_AUDIT.json',audit)
    from m3share_speed_report import method_spec
    (args.out/'METHOD_SPEC.md').write_text(method_spec(result))
    print(f'[frozen] 320 timing frames; {len(result["accuracy_query_ids"])} accuracy queries; local input only',flush=True)
    return result


def bank_build(args,frozen,pipe):
    banks={m:{} for m in METHODS}; meta={}; costs={}
    for asset,paths in frozen['supports'].items():
        images=[decode(p)[0] for p in paths]
        pairs={}
        for method in METHODS:
            tick=time.perf_counter()
            result=pipe.run(images,[asset]*len(images),method,keep_features=True)
            assert all(not r['fallback'] for r in result['outputs']), 'Unexpected support fallback, record/handle explicitly'
            for path,row in zip(paths,result['outputs']):
                ref=frozen['frames'][path]
                assert row['target_box_count']==ref['target_box_count']
                old.close(torch.tensor([b['native_xyxy'] for b in row['boxes']])/320,
                          torch.tensor([b['native_xyxy'] for b in ref['boxes']])/320)
            pairs[method]=result
            features=result['features']; n=len(features)
            random.seed(7001);np.random.seed(7001);torch.manual_seed(7001);torch.cuda.manual_seed_all(7001)
            pm=pv._cls()(layers=['layer2'],backbone='wide_resnet50_2',pre_trained=False,num_neighbors=9).cuda()
            embedding=features.cuda().reshape(-1,64)
            kg=pv.KCenterGreedy(embedding=embedding,sampling_ratio=.1)
            indices=kg.select_coreset_idxs()
            bank=kg.embedding[torch.as_tensor(indices,device='cuda')].clone()
            assert len(bank)==int(n*1600*.1)
            banks[method][asset]=bank
            path=args.out/'cache/banks'/f'{asset}_{method}.npy';path.parent.mkdir(parents=True,exist_ok=True)
            np.save(path,bank.cpu().numpy(),allow_pickle=False)
            torch.cuda.synchronize()
            key=f'{asset}_{method}'
            meta[key]=dict(method=method,asset=asset,support_paths=paths,support_image_count=len(paths),
                support_boxes=[r['boxes'] for r in result['outputs']],normal_region_count=n,
                candidate_patches=n*1600,bank_size=len(bank),dimension=64,bytes=bank.numel()*4,
                seed=7001,ratio=.1,coreset_indices=[int(i) for i in indices],file=str(path),sha256=old.sha(path),
                feature_path='native predicted crop -> Resize320 -> prefix0:5' if method==METHODS[0] else 'same detection P3 -> all native-mapped ROIs',
                seconds=time.perf_counter()-tick)
            costs[key]=meta[key]['seconds']
            del kg,pm,embedding
        compare_results(pairs[METHODS[0]],pairs[METHODS[1]],scores=False)
        print(f'[banks] {asset}: {len(paths)} files; {len(pairs[METHODS[0]]["features"])} matched normal ROIs',flush=True)
    old.write(args.out/'BANK_MANIFEST.json',meta)
    return banks,meta,costs


def merge_results(items):
    return dict(outputs=[r for item in items for r in item['outputs']],
                features=torch.cat([item['features'] for item in items]))


def correctness(args,frozen,pipe,banks):
    gates=dict(passed=False,atol=1e-4,rtol=1e-4,geometry={},batch={},counters={})
    for index,layer in enumerate(pipe.prefix):
        ref=pipe.model.model[index].state_dict()
        assert layer.state_dict().keys()==ref.keys()
        assert all(torch.equal(value,ref[key]) for key,value in layer.state_dict().items())
    gates['crop_prefix_matches_detector_parameters_and_bn']=True
    points=native_to_input([[0,0,640,360],[160,90,480,270]],(640,360))
    old.close(points,torch.tensor([[0,0,320,320],[80,80,240,240]],dtype=torch.float32))
    letter=native_to_input([[0,0,640,360]],(640,360),gain_xy=(.5,.5),pad_xy=(0,70))
    old.close(letter,torch.tensor([[0,70,320,250]],dtype=torch.float32))
    yy,xx=torch.meshgrid(torch.arange(40.,device='cuda'),torch.arange(40.,device='cuda'),indexing='ij')
    feature=torch.stack([xx,yy])[None]
    from torchvision.ops import roi_align
    sampled=roi_align(feature,torch.tensor([[0.,80,80,240,240]],device='cuda'),(40,40),spatial_scale=.125,aligned=True,sampling_ratio=1)
    old.close(sampled[0,:,0,0].cpu(),torch.tensor([9.75,9.75]))
    gates['geometry']=dict(stretch_roundtrip=True,explicit_letterbox_affine=True,roi_ramp_origin=sampled[0,:,0,0].cpu().tolist())
    # Forced-box fixtures test the branches without pretending they are
    # detector results; actual full detection still executes and is counted.
    fixture=Image.new('RGB',(639,357),(150,90,40))
    region=lambda b:dict(native_xyxy=b,confidence=.9,class_id=1)
    synthetic=[[],[region([0,0,639,357])],[region([0,0,1,1]),region([40.3,30.2,600.5,330.7])],
               [region([0,0,639,357]) for _ in range(35)]]
    accepted={m:[1] for m in METHODS}
    for method in METHODS:
        r=pipe.run([fixture]*4,['yoke-suspension']*4,method,banks[method],overrides=synthetic,keep_features=True)
        assert [x['target_box_count'] for x in r['outputs']]==[0,1,2,35]
        assert r['outputs'][0]['fallback'] and len(r['outputs'][0]['scores'])==1
        assert np.isfinite([s for frame in r['outputs'] for s in frame['scores']]).all()
        gates['counters'][method]=r['counters']
    # Fair crop batching: compare32 to1 on the actual50 support files/53ROIs,
    # not merely identical synthetic pixels.
    support_ids=[p for paths in frozen['supports'].values() for p in paths]
    support_images=[decode(p)[0] for p in support_ids]
    support_assets=[frozen['frames'][p]['asset'] for p in support_ids]
    pipe.crop_batch=1
    a=pipe.run(support_images,support_assets,METHODS[0],banks[METHODS[0]],keep_features=True)
    pipe.crop_batch=32
    b=pipe.run(support_images,support_assets,METHODS[0],banks[METHODS[0]],keep_features=True)
    gates['crop_batch_1_vs_32']=compare_results(a,b,features=True)
    ids=[]
    for asset in frozen['supports']:
        ids.extend([p for p in frozen['speed_ids'] if frozen['frames'][p]['asset']==asset][:4])
    images=[decode(p)[0] for p in ids];assets=[frozen['frames'][p]['asset'] for p in ids]
    for method in METHODS:
        singles=[pipe.run([im],[a],method,banks[method],keep_features=True) for im,a in zip(images,assets)]
        single=merge_results(singles)
        gates['batch'][method]={}
        for batch in [8,16]:
            try:
                many=merge_results([pipe.run(images[i:i+batch],assets[i:i+batch],method,banks[method],keep_features=True)
                                    for i in range(0,len(images),batch)])
                gates['batch'][method][str(batch)]=dict(passed=True,**compare_results(single,many,features=True))
                accepted[method].append(batch)
            except AssertionError as error:
                gates['batch'][method][str(batch)]=dict(passed=False,error=str(error),action='exclude this batch, do not relax tolerances')
        if method==METHODS[0]:
            first=single
        else:
            compare_results(first,single,scores=False)
        query=single['features'][:1].cuda()
        flat=query.reshape(-1,64);bank=banks[method][assets[0]]
        direct=pv._cls().euclidean_dist(flat,bank).min(1).values
        blocked=nearest(query,bank,query_chunk=137,bank_chunk=113).flatten()
        gates[f'{method}_NN_chunk_maxdiff']=old.close(direct,blocked)
    # Match live boxes on all frozen timing inputs BEFORE formal timing.
    allowed=sorted(set(accepted[METHODS[0]]) & set(accepted[METHODS[1]]))
    identity_batch=max(allowed)
    gates['accepted_frame_batches']=allowed
    for i in range(0,len(frozen['speed_ids']),identity_batch):
        ids=frozen['speed_ids'][i:i+identity_batch]
        r=pipe.run([decode(p)[0] for p in ids],[frozen['frames'][p]['asset'] for p in ids],METHODS[1],banks[METHODS[1]])
        for p,row in zip(ids,r['outputs']):
            expected=frozen['frames'][p]
            assert row['target_box_count']==expected['target_box_count']
            if row['boxes']:
                old.close(torch.tensor([b['native_xyxy'] for b in row['boxes']])/320,
                          torch.tensor([b['native_xyxy'] for b in expected['boxes']])/320)
    gates.update(passed=True,actual_full_forward_counts_verified=True,actual_crop_batch_counts_verified=True,
                 zero_multi_boundary_nonsquare_executed=True,all_speed_inputs_prediction_identity_checked=True)
    old.write(args.out/'REPRO_AND_CORRECTNESS.json',gates)
    return gates


def timed(pipe,images,assets,method,banks,profile=False):
    torch.cuda.synchronize();tick=time.perf_counter()
    result=pipe.run(images,assets,method,banks[method],profile=profile)
    torch.cuda.synchronize()
    return 1000*(time.perf_counter()-tick),result


def probe(args,frozen,pipe,banks,decoded,allowed_batches):
    ids=[]
    for asset in frozen['supports']:
        ids.extend([p for p in frozen['speed_ids'] if frozen['frames'][p]['asset']==asset][:4])
    for _ in range(10):
        for method in METHODS:
            timed(pipe,[decoded[ids[0]]],[frozen['frames'][ids[0]]['asset']],method,banks)
    latency={m:[] for m in METHODS};batches={}
    for i,p in enumerate(ids):
        for method in (METHODS if i%2==0 else METHODS[::-1]):
            ms,_=timed(pipe,[decoded[p]],[frozen['frames'][p]['asset']],method,banks)
            latency[method].append(ms)
    for batch in allowed_batches:
        items=ids[:batch];images=[decoded[p] for p in items];assets=[frozen['frames'][p]['asset'] for p in items]
        values={m:[] for m in METHODS};peak=0
        for repeat in range(3):
            for method in (METHODS if repeat%2==0 else METHODS[::-1]):
                torch.cuda.reset_peak_memory_stats()
                ms,_=timed(pipe,images,assets,method,banks)
                values[method].append(ms);peak=max(peak,torch.cuda.max_memory_allocated())
        fps={m:batch*1000/statistics.median(values[m]) for m in METHODS}
        batches[str(batch)]=dict(milliseconds=values,fps=fps,peak_allocated_bytes=peak,
                                selection_score=math.sqrt(fps[METHODS[0]]*fps[METHODS[1]]))
    eligible=[b for b in allowed_batches if batches[str(b)]['peak_allocated_bytes']<.8*frozen['environment']['gpu_total_bytes']]
    assert eligible
    chosen=max(eligible,key=lambda b:batches[str(b)]['selection_score'])
    typical=min(statistics.median(v) for v in latency.values())
    n=len(frozen['speed_ids'])
    warmup=max(20,min(200,math.ceil(1000/typical)))
    repeats=max(3,min(10,math.ceil(5000/(n*typical))))
    throughput_repeats=max(3,min(10,math.ceil(5000/(n/min(batches[str(chosen)]['fps'].values())*1000))))
    protocol=dict(chosen_batch=chosen,crop_batch=32,warmup_calls_per_method=warmup,
                  latency_repeats=repeats,throughput_repeats=throughput_repeats,
                  throughput_warmup_batches_per_method=10,stage_profile_repeats=1,
                  stage_profile_per_box_group=8,no_best_round_selection=True,
                  sample_count=n,batch_candidates=allowed_batches,main_timing='sync immediately before/after complete call only',
                  profiling='separate runs; sync at stages; not added to main latency',
                  frozen_id=frozen['frozen_id'])
    old.write(args.out/'PROBE.json',dict(latency_ms=latency,throughput=batches,protocol=protocol))
    target=args.out/'TIMING_PROTOCOL.json'
    assert not target.exists(),'Formal timing protocol already frozen; do not choose new repeats after results'
    old.write(target,protocol)
    print('[probe/frozen]',json.dumps(protocol),flush=True)
    return protocol


def formal(args,frozen,pipe,banks,decoded,protocol):
    recordfile=args.out/'TIMING_RAW.jsonl'
    assert not recordfile.exists(),'Never append a fresh benchmark to old formal repeats'
    peak={m:0 for m in METHODS};baseline={m:0 for m in METHODS};count=0
    ids=frozen['speed_ids']
    def run_items(items,method,phase,repeat,order,profile=False):
        nonlocal count
        torch.cuda.reset_peak_memory_stats();baseline[method]=max(baseline[method],torch.cuda.memory_allocated())
        ms,r=timed(pipe,[decoded[p] for p in items],[frozen['frames'][p]['asset'] for p in items],method,banks,profile)
        peak[method]=max(peak[method],torch.cuda.max_memory_allocated())
        rec=dict(method=method,phase=phase,repeat=repeat,pair_order=order,batch=len(items),
                 frame_ids=items,target_boxes=[o['target_box_count'] for o in r['outputs']],
                 fallbacks=[o['fallback'] for o in r['outputs']],elapsed_ms=ms,
                 counters=r['counters'],effective_regions=r['effective_regions'],
                 frozen_id=frozen['frozen_id'],environment_id=old.digest(frozen['environment']),
                 timestamp=datetime.now(timezone(timedelta(hours=8))).isoformat(),stages=r['stages'])
        stream.write(json.dumps(rec,ensure_ascii=False,allow_nan=False)+'\n');stream.flush();count+=1
        return r
    # Warm different content/ROI shapes, outside all formal timing records.
    for i in range(protocol['warmup_calls_per_method']):
        p=ids[i%len(ids)]
        for method in METHODS:
            timed(pipe,[decoded[p]],[frozen['frames'][p]['asset']],method,banks)
    with recordfile.open('x') as stream:
        for repeat in range(protocol['latency_repeats']):
            idle_gpu()
            for i,p in enumerate(ids):
                order=METHODS if (i+repeat)%2==0 else METHODS[::-1]
                a=run_items([p],order[0],'single_frame',repeat,0)
                b=run_items([p],order[1],'single_frame',repeat,1)
                compare_results(a,b,scores=False)
            print(f'[latency] round{repeat+1}/{protocol["latency_repeats"]}',flush=True)
        batch=protocol['chosen_batch']
        for i in range(protocol['throughput_warmup_batches_per_method']):
            items=ids[(i*batch)%len(ids):][:batch]
            for method in METHODS:
                timed(pipe,[decoded[p] for p in items],[frozen['frames'][p]['asset'] for p in items],method,banks)
        for repeat in range(protocol['throughput_repeats']):
            idle_gpu()
            for i in range(0,len(ids),batch):
                order=METHODS if (i//batch+repeat)%2==0 else METHODS[::-1]
                a=run_items(ids[i:i+batch],order[0],'throughput_batch',repeat,0)
                b=run_items(ids[i:i+batch],order[1],'throughput_batch',repeat,1)
                compare_results(a,b,scores=False)
            print(f'[throughput] round{repeat+1}/{protocol["throughput_repeats"]}, batch{batch}',flush=True)
        profile_ids=[]
        for bin_name in ['0','1','2-4','>=5']:
            profile_ids.extend([p for p in ids if frozen['frames'][p]['group']==bin_name][:8])
        for i,p in enumerate(profile_ids):
            for index,method in enumerate(METHODS if i%2==0 else METHODS[::-1]):
                run_items([p],method,'stage_profile',0,index,True)
    return dict(timing_records=count,peak_gpu_allocated_bytes=peak,resident_gpu_baseline_bytes=baseline,
                note='Both arms banks and shared model resident for fair alternation; peak allocated is not isolated deployment footprint')


def accuracy(args,frozen,pipe,banks,batch):
    records={m:{} for m in METHODS}; ids=frozen['accuracy_query_ids']
    rawpath=args.out/'accuracy/raw_scores.jsonl';rawpath.parent.mkdir(exist_ok=True)
    with rawpath.open('w') as stream:
        for i in range(0,len(ids),batch):
            items=ids[i:i+batch];images=[decode(p)[0] for p in items];assets=[frozen['frames'][p]['asset'] for p in items]
            for method in METHODS:
                result=pipe.run(images,assets,method,banks[method])
                for p,row in zip(items,result['outputs']):
                    assert row['target_box_count']==frozen['frames'][p]['target_box_count']
                    records[method][p]=row
                    stream.write(json.dumps(dict(method=method,path=p,image_label=frozen['frames'][p]['image_label'],**row),ensure_ascii=False)+'\n')
            for p in items:
                compare_results(dict(outputs=[records[METHODS[0]][p]]),dict(outputs=[records[METHODS[1]][p]]),scores=False)
            if i%(batch*50)==0:
                print(f'[accuracy] {min(i+batch,len(ids))}/{len(ids)}',flush=True)
    from m3share_speed_report import accuracy_summary
    summary=accuracy_summary(args.source,frozen,records)
    summary['raw_scores']=dict(path=str(rawpath),sha256=old.sha(rawpath))
    old.write(args.out/'ACCURACY_SUMMARY.json',summary)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=SOURCE)
    parser.add_argument('--out',type=Path,default=old.BASE/TASK)
    parser.add_argument('--mode',choices=['prepare','run','report'],default='prepare')
    args=parser.parse_args()
    old.deterministic()
    if args.mode=='report':
        from m3share_speed_report import report
        report(args.out)
        return
    frozen=freeze(args)
    if args.mode=='prepare':
        cpu=old.read(args.out/'CPU_TESTS.json')
        assert cpu['passed'] and cpu['tests_run']>=7
        for name,sha in frozen['code_sha256'].items():
            assert cpu['code_sha256'][name]==sha,'CPU tests must match prepared code'
        old.write(args.out/'PREPARE.json',dict(ready=True,cpu_tests_passed=cpu['passed'],
            cpu_tests=cpu['tests_run'],source_images_verified=len(frozen['frames']),speed_frames=320,
            accuracy_queries=len(frozen['accuracy_query_ids']),frozen_id=frozen['frozen_id'],
            formal_gpu_started=(args.out/'REPRO_AND_CORRECTNESS.json').exists(),
            real_raw_scene_mapping_available=False,scope=frozen['scope'],
            formal_gpu_gates_and_probe='USER MUST RUN; no timing or performance claims yet'))
        return
    assert torch.cuda.is_available()
    assert not (args.out/'TIMING_RAW.jsonl').exists() and not (args.out/'TIMING_PROTOCOL.json').exists(), 'Existing frozen timing run: preserve it and diagnose, do not silently rerun/select rounds'
    start=time.perf_counter();idle_gpu()
    pipe=Pipeline(frozen['weights']['detector'])
    before=old.state_hash(pipe.model);prefix_before=old.state_hash(pipe.prefix)
    banks,bank_meta,bank_cost=bank_build(args,frozen,pipe)
    gates=correctness(args,frozen,pipe,banks)
    assert old.state_hash(pipe.model)==before and old.state_hash(pipe.prefix)==prefix_before
    decoded={};io=[]
    for p in frozen['speed_ids']:
        decoded[p],ms=decode(p);io.append(dict(path=p,decode_ms=ms))
    old.write(args.out/'DISK_IO.json',dict(records=io,note='PIL open/decode/RGB conversion, separate warm-cache filesystem pass; not a cold-disk claim'))
    protocol=probe(args,frozen,pipe,banks,decoded,gates['accepted_frame_batches'])
    resources=formal(args,frozen,pipe,banks,decoded,protocol)
    accuracy(args,frozen,pipe,banks,protocol['chosen_batch'])
    assert old.state_hash(pipe.model)==before and old.state_hash(pipe.prefix)==prefix_before
    gates.update(model_state_unchanged=True,prefix_state_unchanged=True,
                 state_sha256=before,prefix_state_sha256=prefix_before,
                 formal_actual_forward_execution=True,all_accuracy_queries_retained=True)
    old.write(args.out/'REPRO_AND_CORRECTNESS.json',gates)
    for path,sha in frozen['source_sha256'].items():
        assert old.sha(path)==sha,'Source experiment modified during this task'
    resources.update(invocation_wall_seconds=time.perf_counter()-start,model_load_seconds=pipe.load_seconds,
                     bank_build_seconds=bank_cost,bank_bytes={k:v['bytes'] for k,v in bank_meta.items()},
                     host_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                     timing_frames=len(frozen['speed_ids']),accuracy_queries=len(frozen['accuracy_query_ids']),
                     gpu_competition_check='no other compute PID before probe and every formal round',
                     scope=frozen['scope'])
    old.write(args.out/'RESOURCE_COST.json',resources)
    from m3share_speed_report import report
    report(args.out)
    old.write(args.out/'COMPLETED.json',dict(complete=True,raw_scene_end_to_end_measured=False,local_native_crop_pipeline_measured=True,
                                          timing_records=resources['timing_records'],accuracy_queries=len(frozen['accuracy_query_ids'])))
    print('[done]',args.out/'SPEED_REPORT.md',flush=True)


if __name__=='__main__':
    main()
