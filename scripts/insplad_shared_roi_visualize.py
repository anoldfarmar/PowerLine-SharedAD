#!/usr/bin/env python3
"""Render 20 audited C examples from saved features/banks, CPU only."""
import argparse
import csv
import hashlib
import json
import textwrap
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image
import torch

import insplad_shared_roi as source
import m3score_pcverify as pv

TASK = 'InsPLAD-sharedROI-C-vis20-v1'
SEED = 'C-vis20-fixed-v1'


def choose_examples(frozen, detections):
    """Stratify by image label/detection status, never by anomaly score."""
    examples = []
    rank = lambda p: hashlib.sha256((SEED+'\n'+p).encode()).hexdigest()
    for asset, data in frozen['data'].items():
        records = detections[asset]
        used = set()
        def pick(paths, fallback, role, fault):
            candidates = [p for p in paths if p not in used and records[p]['fallback'] == fallback]
            assert candidates, f'No candidate for {asset}/{role}'
            path = min(candidates, key=rank)
            used.add(path)
            examples.append(dict(id=f'{len(examples)+1:02d}', asset=asset, path=path,
                                 image_label=fault, selection_role=role,
                                 fallback=fallback, detection=records[path]))
        pick(data['normal'], False, 'normal_with_box', 'good')
        faults = list(data['defects'])
        for i in range(2):
            fault = faults[i % len(faults)]
            pick(data['defects'][fault], False, 'fault_with_box', fault)
        pool = [p for paths in data['defects'].values() for p in paths]
        candidates = [p for p in pool if records[p]['fallback'] and p not in used]
        assert candidates, f'No fault fallback available for {asset}'
        path = min(candidates, key=rank)
        fault = next(f for f, paths in data['defects'].items() if path in paths)
        pick([path], True, 'fault_full_image_fallback', fault)
    assert len(examples) == 20 and len({e['path'] for e in examples}) == 20
    return examples


def top_positions(distance, box):
    # Stable index tie breaking is display-only; the top8 mean is unaffected.
    order = np.argsort(-distance.ravel(), kind='stable')[:8]
    x1,y1,x2,y2 = box
    return [dict(rank=i+1, row=int(k//40), col=int(k%40), distance=float(distance.ravel()[k]),
                 x=x1+(int(k%40)+.5)*(x2-x1)/40,
                 y=y1+(int(k//40)+.5)*(y2-y1)/40) for i,k in enumerate(order)]


def retrieve_maps(folder, frozen, caches, banks, asset, paths):
    entry = banks[f'{asset}_C']
    assert entry['binding']['frozen_id'] == frozen['frozen_id']
    assert entry['binding']['support'] == frozen['data'][asset]['support']
    assert source.sha(entry['file']) == entry['sha256']
    bank = torch.from_numpy(np.load(entry['file'], allow_pickle=False))
    assert list(bank.shape) == [1600,64] and bank.dtype == torch.float32
    needed = set(paths)
    features, provenance = {}, {}
    for chunk in caches[asset]:
        take = needed.intersection(chunk['binding']['paths'])
        if not take:
            continue
        assert chunk['binding']['frozen_id'] == frozen['frozen_id']
        assert source.sha(chunk['file']) == chunk['sha256']
        with np.load(chunk['file'], allow_pickle=False) as stored:
            array = stored['C']
            for path in take:
                index = chunk['binding']['paths'].index(path)
                features[path] = torch.from_numpy(array[index].copy())
                provenance[path] = dict(chunk=chunk['file'], sha256=chunk['sha256'], index=index,
                                        detection=chunk['detections'][index])
    assert set(features) == needed
    maps = {}
    with torch.inference_mode():
        for path in paths:
            # The actual official FP32 Euclidean kernel, reused on CPU.
            values = pv._cls().euclidean_dist(features[path],bank).min(1).values
            assert torch.isfinite(values).all()
            maps[path] = values.numpy().reshape(40,40)
    return maps, provenance


def paint(ax, image, record, distance=None, norm=None, marks=False, all_boxes=False):
    width,height = image.size
    box = record['box_native_xyxy']
    x1,y1,x2,y2 = box
    ax.imshow(image, extent=(0,width,height,0))
    if distance is not None:
        # Project ONLY into the actual selected ROI, never stretch it over the
        # whole native image. Outside the ROI remains the unmodified original.
        ax.imshow(distance, cmap='viridis', norm=norm, interpolation='bilinear',
                  extent=(x1,x2,y2,y1), alpha=.48)
    if all_boxes:
        for xa,ya,xb,yb,conf,cls in record['all_class_detections']:
            xa,xb = xa*width/320, xb*width/320
            ya,yb = ya*height/320, yb*height/320
            ax.add_patch(Rectangle((xa,ya),xb-xa,yb-ya,fill=False,edgecolor='gray',lw=.7,alpha=.6))
    color = '#ff9500' if record['fallback'] else '#00ef85'
    ax.add_patch(Rectangle((x1,y1),x2-x1,y2-y1,fill=False,edgecolor=color,
                           lw=2,linestyle='--' if record['fallback'] else '-'))
    if marks and distance is not None:
        for item in top_positions(distance,box):
            ax.plot(item['x'],item['y'],'s',ms=6,mfc='none',mec='red' if item['rank']==1 else '#ff9500',mew=1.3)
            ax.annotate(str(item['rank']),(item['x'],item['y']),xytext=(3,3),textcoords='offset points',
                        fontsize=8,color='white',bbox=dict(facecolor='black',alpha=.5,pad=.1))
    ax.set_xlim(0,width);ax.set_ylim(height,0)
    ax.set_xticks([]);ax.set_yticks([])


def render(example, image, distance, vmax):
    record = example['detection']
    norm = Normalize(0,vmax,clip=True)
    fig, axes = plt.subplots(2,2,figsize=(14,10))
    paint(axes[0,0],image,record,all_boxes=True)
    if record['fallback']:
        status = 'FALLBACK: no valid expected-class box; use full image'
    else:
        status = f"Predicted target box | confidence={record['confidence']:.3f}"
    axes[0,0].set_title(status,fontsize=10,color='#b34e00' if record['fallback'] else 'black')
    paint(axes[0,1],image,record,distance,norm,marks=True)
    axes[0,1].set_title('C: ROI distance overlay in original coordinates + top8',fontsize=10)
    paint(axes[1,0],image,record,distance,norm,marks=True)
    x1,y1,x2,y2 = record['box_native_xyxy']
    axes[1,0].set_xlim(x1,x2);axes[1,0].set_ylim(y2,y1)
    axes[1,0].set_title('Zoom of the actual ROI (same cached shared P3 features)',fontsize=10)
    im = axes[1,1].imshow(distance,cmap='viridis',norm=norm,interpolation='nearest')
    for item in top_positions(distance,[0,0,40,40]):
        axes[1,1].plot(item['col'],item['row'],'s',ms=9,mfc='none',
                       mec='red' if item['rank']==1 else '#ff9500',mew=1.5)
        axes[1,1].annotate(str(item['rank']),(item['col'],item['row']),color='white',fontsize=9)
    axes[1,1].set_xticks([]);axes[1,1].set_yticks([])
    axes[1,1].set_title(f'Raw 40 x 40 NN distance | fixed asset vmax={vmax:.4f}',fontsize=10)
    fig.colorbar(im,ax=axes[1,1],fraction=.046,pad=.03,label='Euclidean distance to C normal bank')
    fig.suptitle(f"#{example['id']}  {example['asset']} / {example['image_label']}\n"
                 f"Image-level GT label only | C top8 score={example['saved_score']:.5f} | {example['selection_role']}",fontsize=13)
    caption = '\n'.join(textwrap.wrap(Path(example['path']).name,110))
    fig.text(.03,.025,caption+'\nGreen: chosen detector box; orange dashed: full-frame fallback; gray: detector outputs. '
             'Red: top1; orange: top2-8.\nViridis yellow = farther from normal bank; not pixel GT, attention, or a calibrated defect probability.',fontsize=8)
    fig.subplots_adjust(left=.04,right=.96,bottom=.14,top=.88,hspace=.2,wspace=.18)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=source.BASE/source.TASK)
    parser.add_argument('--out',type=Path)
    args = parser.parse_args()
    folder = args.source
    out = args.out or folder/'vis_C_20'
    assert out.resolve() != folder.resolve()
    out.mkdir(parents=True,exist_ok=True)
    (out/'images').mkdir(exist_ok=True)
    source.deterministic()
    assert source.read(folder/'COMPLETED.json')['completed']
    assert source.read(folder/'REPRO_AND_CORRECTNESS.json')['passed']
    frozen = source.read(folder/'FROZEN_INPUTS.json')
    for name, sha in frozen['code_sha256'].items():
        assert source.sha(source.ROOT/'scripts'/name) == sha
    assert pv.source_meta() == frozen['environment']['patchcore']
    inputs = ['FROZEN_INPUTS.json','BANK_MANIFEST.json','CACHE_MANIFEST.json','REPRO_AND_CORRECTNESS.json']
    source_hashes = {name:source.sha(folder/name) for name in inputs}
    detections = {a:source.read(folder/'detections'/f'{a}.json') for a in frozen['data']}
    examples = choose_examples(frozen,detections)
    selection = dict(frozen_id=frozen['frozen_id'],seed=SEED,examples=examples,
                     rule='Per asset: 1 good with box, 2 faults with box (cover fault types), 1 fault fallback. SHA256(seed+path) ordering. No score selection.',
                     note='Purposely stratified inspection set; not representative prevalence or a new evaluation split.')
    selection_file = out/'SELECTION.json'
    if selection_file.exists():
        assert source.read(selection_file) == selection
    source.write(selection_file,selection)
    maps, display, evidence, cache_sources = {}, {}, {}, {}
    raw = {}
    for asset, data in frozen['data'].items():
        chosen = [e for e in examples if e['asset']==asset]
        calibration = data['normal'][:12]
        paths = list(dict.fromkeys([e['path'] for e in chosen]+calibration))
        for path in paths:
            assert source.sha(path) == frozen['image_sha256'][path]
        distances, provenance = retrieve_maps(folder,frozen,source.read(folder/'CACHE_MANIFEST.json'),
                                              source.read(folder/'BANK_MANIFEST.json'),asset,paths)
        vmax = float(np.percentile(np.concatenate([distances[p].ravel() for p in calibration]),99.5))
        assert np.isfinite(vmax) and vmax > 0
        display[asset] = dict(vmin=0,vmax=vmax,calibration=calibration,
                             rule='First12 frozen normal query crops; 99.5 percentile of C patch distances; shared within asset, never per-image autoscale')
        for i,path in enumerate(calibration):
            raw[f'{asset}_normal_calibration_{i:02d}'] = distances[path]
        for example in chosen:
            path = example['path']
            assert provenance[path]['detection'] == example['detection']
            fault = example['image_label'] if example['image_label']!='good' else next(iter(data['defects']))
            scores_file = folder/'raw_scores'/f'{asset}__{fault}_C.json'
            scores = source.read(scores_file)
            assert scores['frozen_id'] == frozen['frozen_id']
            source.auc(scores,data['normal'],data['defects'][fault])
            saved = scores['normal' if example['image_label']=='good' else 'defect'][path]
            recomputed = float(np.sort(distances[path].ravel())[-8:].mean())
            np.testing.assert_allclose(recomputed,saved,atol=1e-4,rtol=1e-4)
            example.update(saved_score=saved,rendered_score=recomputed,vmax=vmax,
                           image_sha256=frozen['image_sha256'][path],top8=top_positions(distances[path],example['detection']['box_native_xyxy']))
            maps[example['id']] = distances[path]
            raw[f"sample_{example['id']}"] = distances[path]
            evidence[example['id']] = dict(score_abs_diff=abs(saved-recomputed),source_scores=str(scores_file),
                                           source_scores_sha256=source.sha(scores_file))
            cache_sources[example['id']] = provenance[path]
        print(f'[maps] {asset}: 4 examples + 12 fixed normal calibration; CPU cached-only',flush=True)
    outputs = {}
    with PdfPages(out/'C_20_examples.pdf') as pdf:
        for example in examples:
            with Image.open(example['path']) as opened:
                image = opened.convert('RGB')
            fig = render(example,image,maps[example['id']],example['vmax'])
            filename = f"{example['id']}_{example['asset']}_{example['image_label']}.png"
            fig.savefig(out/'images'/filename,dpi=140)
            pdf.savefig(fig)
            plt.close(fig)
            example['png'] = f'images/{filename}'
            outputs[example['png']] = source.sha(out/example['png'])
    fig,axes = plt.subplots(5,4,figsize=(16,19))
    for ax,example in zip(axes.ravel(),examples):
        with Image.open(example['path']) as opened:
            image = opened.convert('RGB')
        paint(ax,image,example['detection'],maps[example['id']],Normalize(0,example['vmax'],clip=True))
        status = 'FALLBACK' if example['fallback'] else f"conf={example['detection']['confidence']:.2f}"
        ax.set_title(f"#{example['id']} {example['asset']}\n{example['image_label']} | {status}",fontsize=8)
    fig.suptitle('C shared-P3: 20 fixed inspection examples | Viridis ROI overlays',fontsize=15)
    fig.subplots_adjust(top=.94,bottom=.02,left=.02,right=.98,wspace=.08,hspace=.22)
    fig.savefig(out/'OVERVIEW_20.png',dpi=140)
    plt.close(fig)
    np.savez_compressed(out/'distance_maps.npz',**raw)
    fields = ['id','asset','image_label','selection_role','fallback','saved_score','rendered_score','vmax','path','png']
    with (out/'samples.csv').open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore')
        writer.writeheader();writer.writerows(examples)
    source.write(out/'DISPLAY_RULES.json',display)
    source.write(out/'MANIFEST.json',dict(task=TASK,frozen_id=frozen['frozen_id'],examples=examples,
                  source_sha256=source_hashes,cache_sources=cache_sources,output_png_sha256=outputs,
                  script_sha256=source.sha(Path(__file__)),distance_computation='Official exact Euclidean kernel on cached C features/bank, CPU FP32; NO backbone forward or retraining'))
    check = dict(passed=True,examples=20,png_count=len(outputs),fault_examples=sum(e['image_label']!='good' for e in examples),
                 normal_examples=sum(e['image_label']=='good' for e in examples),fallback_examples=sum(e['fallback'] for e in examples),
                 all_seven_fault_combinations_covered=len({(e['asset'],e['image_label']) for e in examples if e['image_label']!='good'})==7,
                 gpu_used=False,new_backbone_forwards=0,max_abs_score_diff=max(e['score_abs_diff'] for e in evidence.values()),scores=evidence)
    assert check['all_seven_fault_combinations_covered'] and check['fallback_examples']==5
    for name,hash_value in source_hashes.items():
        assert source.sha(folder/name) == hash_value, 'Original experiment file changed'
    source.write(out/'VIS_CORRECTNESS.json',check)
    text = '''# C 检测框与异常距离可视化：20张

优先打开 C_20_examples.pdf（每页一张，20页），或 OVERVIEW_20.png 总览。
images/ 是20张独立的四面板高清图；C_VIS20.zip 可整体下载。

每页：左上原图+检测框，右上原图坐标下的 ROI 距离热力图与 top8，左下 ROI 放大，右下原始40×40距离图。
绿色实线=实际采用的预测框；橙色虚线=没有有效的指定类别检测框，回退整张图；灰框=其他检测输出。
红色小方框=top1，橙色小方框=top2–8。黄色更远离正常库，紫色更接近。
热力图只投影到实际 ROI 内，不把 ROI 距离图错误拉伸到全图。
图中 good/rust 等是图像级数据标签，不是模型判定，不存在像素或框 GT，也未选告警阈值。

每资产4张：正常有框1张、异常有框2张、异常回退1张。共5正常、15异常，其中5张回退，覆盖全部7个故障组合。
分层后以固定 SHA256(seed+path) 选样，不看异常分数。这是刻意包含失败案例的检查集，不能用它估计总体回退率。
显示范围在同资产内固定：原冻结正常查询前12张的 C patch 距离99.5分位。跨资产颜色强度不可直接比较。

只复用已完成实验的 C 框、ROI特征与正常库，在 CPU 上执行原官方欧氏最近邻距离核。
20张的 top8 分数都已核对原始逐图分数，容差 atol=rtol=1e-4。
distance_maps.npz 保留20张原始40×40矩阵和色标校准矩阵；MANIFEST.json/samples.csv 可追溯路径、框和分数。
没有重跑骨干、修改实验评分、训练模型或声明定位精度。高响应区域是异常距离响应，不是注意力或缺陷概率。
'''
    (out/'README.md').write_text(text)
    with zipfile.ZipFile(out/'C_VIS20.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(out.rglob('*')):
            if path.is_file() and path.name!='C_VIS20.zip':
                archive.write(path,path.relative_to(out))
    print(json.dumps(check,indent=2),flush=True)
    print(f'[done] {out}/C_20_examples.pdf',flush=True)


if __name__=='__main__':
    main()
