#!/usr/bin/env python3
"""Read-only scoring audit, fixed visualizations and honest M3NVE report generation."""
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from scipy.stats import rankdata

import m3nve_run as E


def main():
    start=time.perf_counter()
    out=E.OUT
    result=E.read(out/'nve_eval.json'); mechanisms=E.read(out/'MECHANISMS.json')
    frozen=E.verify_frozen(); lists=E.read(E.HIST/'SUPPORT_LISTS.json')
    selection=E.read(out/'VISUAL_SELECTION.json')
    reports=E.read(out/'repro_and_correctness.json')
    assert reports['passed'] and len(reports['repro'])==60
    worst=0.; image_rows=0; audit=[]
    configlookup={E.cfg_name(c,s):c for c in E.configurations(lists) for s in range(3)}
    for r in result['per_config']:
        name=r['config']; c=configlookup[name]
        score=E.read(out/'raw_scores'/f'{name}.json')
        pc=E.read(E.HIST/'raw_scores_patchcore'/f"{c['key']}.json")
        hist=E.read(E.HIST/'raw_scores_ours'/f'{name}.json')
        for side in ['normal','defect']:
            assert set(score[side])==set(c['query'][side])==set(pc[side])==set(hist[side])
            assert len(score[side])==len(c['query'][side])
            assert np.isfinite(np.array(list(score[side].values()))).all()
            diff=max(abs(v[0]-hist[side][p]) for p,v in score[side].items()); worst=max(worst,diff)
            assert diff<=1e-4
            image_rows+=len(score[side])
        sd=np.array(list(score['defect'].values())); sn=np.array(list(score['normal'].values()))
        for i,arm in enumerate(['A','B','C']):
            rr=rankdata(np.r_[sd[:,i],sn[:,i]],method='average')
            nd=len(sd); nn=len(sn)
            independent=float((rr[:nd].sum()-nd*(nd+1)/2)/(nd*nn))
            assert abs(independent-r[arm])<1e-12
        for lhs,rhs in [('C','B'),('B','A'),('C','A'),('A','PatchCore'),('B','PatchCore'),('C','PatchCore')]:
            assert abs(r[lhs+'-'+rhs]-(r[lhs]-r[rhs]))<1e-12
        audit.append(dict(config=name,rows=len(sd)+len(sn),paths_finite_auroc_pass=True))
    assert len(audit)==60 and len(list((out/'raw_scores').glob('*.json')))==60
    # Independently recompute every per-reference summary and ddof=0 spread.
    for cell, summary in result['summary'].items():
        rs=[r for r in result['per_config'] if r['combo']+'_'+r['budget']==cell]
        assert len(rs)==15
        for field, values in summary.items():
            groups=[np.mean([r[field] for r in rs if r['ref']==j]) for j in range(3001,3006)]
            assert abs(np.mean(groups)-values['mean'])<1e-12
            assert abs(np.std(groups,ddof=0)-values['std'])<1e-12
            assert sum(r[field]<0 for r in rs)==values['negative_configs']
    E.save('INDEPENDENT_AUDIT.json',dict(passed=True,configs=60,image_rows=image_rows,arm_image_scores=3*image_rows,
           max_A_historical_diff=worst,auroc_implementation='scipy.stats.rankdata(method=average)',details=audit))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    vis=out/'vis'; vis.mkdir(exist_ok=True)
    assert len(selection)==16
    for item in selection:
        maps=np.load(out/'cache'/f'vis_{item["id"]}.npy')
        fig,axs=plt.subplots(1,6,figsize=(19,3.8))
        axs[0].imshow(Image.open(item['path']).convert('RGB').resize((320,320)))
        axs[0].set_title('Input ('+item['side']+' image)')
        for i,title in enumerate(['A','B: residual','C: NVE','Raw residual norm','Clean residual norm']):
            im=axs[i+1].imshow(maps[i],cmap='coolwarm' if i<3 else 'viridis',vmin=-1 if i<3 else 0,vmax=1 if i<3 else 2)
            axs[i+1].set_title(title)
            fig.colorbar(im,ax=axs[i+1],fraction=.046,pad=.04)
        for ax in axs: ax.axis('off')
        fig.suptitle(item['config']+' | '+item['side']+' | A/B/C='+','.join(f'{v:.4f}' for v in item['scores']))
        fig.tight_layout(); fig.savefig(vis/f'{item["id"]}.png',dpi=120); plt.close(fig)
    # Norm distributions retain all patches; fixed histogram domain, no sample picking.
    fig,axs=plt.subplots(2,4,figsize=(18,7))
    for col,(cell,_) in enumerate(result['summary'].items()):
        combo,g=cell.rsplit('_',1); cc=combo.replace('/','__')
        for row,side in enumerate(['normal','defect']):
            h0=np.zeros(60); h1=np.zeros(60); bins=np.linspace(0,2,61)
            for ref in range(3001,3006):
                key=f'{cc}_{g}_ref{ref}'; z=np.load(out/'cache'/f'norms_{key}.npz')
                nd=len(lists['queries'][f'{cc}_{g}']['defect'])
                sl=slice(nd,None) if side=='normal' else slice(0,nd)
                h0+=np.histogram(z['raw'][sl],bins=bins)[0]
                h1+=np.histogram(z['clean'][sl],bins=bins)[0]
            ax=axs[row,col]
            ax.plot((bins[:-1]+bins[1:])/2,h0/h0.sum(),label='Raw')
            ax.plot((bins[:-1]+bins[1:])/2,h1/h1.sum(),label='NVE')
            ax.set_title(f'{cc} {g}\npatches from {side} images');ax.legend();ax.set_xlabel('Residual norm')
    fig.tight_layout();fig.savefig(vis/'norm_distributions.png',dpi=140);plt.close(fig)
    # Environment artifacts generated by the authorized report script, no shell redirection.
    (out/'ENVIRONMENT.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    cost=E.read(out/'RESOURCE_COST.json')
    cost['report_seconds']=time.perf_counter()-start
    cost['elapsed_since_freeze_seconds']=time.time()-frozen['timestamp']
    cost['current_output_bytes']=sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
    cost['new_representation_note']='residual banks: raw+clean 2*60*256*128*4 bytes; query residuals/projections ephemeral, reused across3seeds; norm caches include all20normalconfigs'
    E.save('RESOURCE_COST.json',cost)
    lines=['# M3NVE-residual-v1：有界残差对照', '',
      '本轮为 **NVE-inspired 有界残差匹配**，不是完整 IDEAL 复现。无 IDE 训练、无异常参考 mask；NVE 来自原论文。', '',
      f'完成60个冻结数据配置×A/B/C=180臂配置；PatchCore仅复用历史20配置原始分数。独立审计覆盖{image_rows}逐图行、{3*image_rows}三臂分数，路径、有限性、平均秩AUROC和配对汇总全部通过。',
      f'A对历史全60配置最坏逐路径差={worst:.9g}，固定容差1e-4；人工向量、批量/单图、mapper及骨干参数/BN buffers不变检查通过。', '',
      '## 主结果（AUROC，0–1）', '',
      '|资产 / 支持预算|A|B|C|PatchCore|B−A|C−B|C−A|',
      '|---|---:|---:|---:|---:|---:|---:|---:|']
    for cell,summary in result['summary'].items():
        lines.append('|'+cell+'|'+'|'.join(f"{summary[f]['mean']:.6f}" for f in ['A','B','C','PatchCore','B-A','C-B','C-A'])+'|')
    lines+=['','每项先对同参照组3支持种子取均值，再对5参照组取均值。PatchCore不依赖故障种子，重复对应仅用于配对。', '',
            '## 稳定性与逐配置配对', '',
            '|单元|比较|五组均值|组间std(ddof=0)|组均值范围|正收益组/5|负收益原配置/15|',
            '|---|---|---:|---:|---|---:|---:|']
    for cell,summary in result['summary'].items():
        for field in ['C-B','B-A','C-A','A-PatchCore','B-PatchCore','C-PatchCore']:
            s=summary[field]
            lines.append(f"|{cell}|{field}|{s['mean']:.6f}|{s['std']:.6f}|[{s['range'][0]:.6f}, {s['range'][1]:.6f}]|{s['positive_groups']}/5|{s['negative_configs']}/15|")
    lines+=['','逐种子×参照组全部成绩和配对差见 [paired_configs.csv](paired_configs.csv)；全部组均值见 [nve_eval.json](nve_eval.json)。以下保留每个原始配置，避免将组均值正收益解释为所有种子均正。','',
            '|配置|A|B|C|PC|C−B|B−A|C−A|','|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in result['per_config']:
        lines.append('|'+r['config']+'|'+'|'.join(f'{r[f]:.6f}' for f in ['A','B','C','PatchCore','C-B','B-A','C-A'])+'|')
    lines+=['','## 六项裁决','']
    for cell,s in result['summary'].items():
        cb=s['C-B'];ba=s['B-A'];ca=s['C-A']
        if cb['mean']>0 and ca['mean']>0: verdict='C>B且C>A，均值上值得继续，但仍须结合参照组和原配置负收益判断稳定性'
        elif cb['mean']>0: verdict='C>B但整体仍不优于A：降噪有局部价值，整体改造尚不成立'
        elif ba['mean']>0: verdict='B>A而C≤B：残差方向有价值，本配置下NVE未获增量支持'
        else: verdict='B/C均值未提供总体改善，保留负结果；这不能否定论文在其他特征/监督条件下的结果'
        lines.append(f"{cell}：{verdict}。C−B={cb['mean']:+.6f}（{cb['positive_groups']}/5正组，{cb['negative_configs']}/15负配置），B−A={ba['mean']:+.6f}，C−A={ca['mean']:+.6f}。")
        lines.append('')
    stable=all(s['C-B']['positive_groups']==5 and s['C-B']['negative_configs']==0 for s in result['summary'].values())
    lines += [f'1. C是否稳定优于B：{"四个单元均满足5/5正收益组且没有负收益配置" if stable else "不满足跨资产、跨预算、全部参照组与种子的稳定增量要求"}。以上逐单元结果给出边界；不作独立显著性声明。','',
      '2. 改善来源以同配置恒等分解 C−A=(B−A)+(C−B)判断。B−A是“残差匹配评分改造”的整体作用，包含与dN组合方式的变化，不应解释为单独向量相减的因果效应。','',
      '3. 资产与预算一致性：分别看上表方向和原始配置负收益；1N1A与5N5A查询集合不同，不能将差异解释为纯shot效应。各臂参照组范围如下。','',
      '|单元|A组均值范围|B组均值范围|C组均值范围|','|---|---|---|---|']
    for cell,s in result['summary'].items():
        lines.append('|'+cell+'|'+'|'.join(f"[{s[a]['range'][0]:.4f},{s[a]['range'][1]:.4f}]" for a in ['A','B','C'])+'|')
    lines+=['','4. 与PatchCore的差距（正差表示PatchCore更高；缩小量负数表示扩大）：','',
            '|单元|PC−A|PC−B|PC−C|C相对A缩小量|','|---|---:|---:|---:|---:|']
    for cell,s in result['summary'].items():
        lines.append(f"|{cell}|{-s['A-PatchCore']['mean']:.6f}|{-s['B-PatchCore']['mean']:.6f}|{-s['C-PatchCore']['mean']:.6f}|{s['C-A']['mean']:.6f}|")
    lines+=['','1N1A比较原1N PatchCore；5N5A比较原10N PatchCore，监督组成不同。未重建PatchCore库、未新增PatchCore实验。','',
       '5. 机制检查：下表按图像标签统计所属全部patch。两类图的范数都降低只能说明两类图内表示都被压缩；余弦匹配对缩放不敏感，且没有像素mask，不能据此声称故障像素证据被删或测得故障像素保留率。结合图像分数C−B与AUROC变化评估可能的共同削弱。','',
       '|单元|图像标签|raw范数均值|clean范数均值|clean/raw均值(非零)|零残差/总patch|零秩/总patch|','|---|---|---:|---:|---:|---|---|']
    for cell in result['summary']:
        combo,g=cell.rsplit('_',1);cc=combo.replace('/','__')
        for side in ['normal','defect']:
            ms=[mechanisms[f'{cc}_{g}_ref{ref}'][side] for ref in range(3001,3006)]
            count=sum(m['raw_norm']['n'] for m in ms)
            values=[sum(m[field]['mean']*m[field]['n'] for m in ms)/sum(m[field]['n'] for m in ms) for field in ['raw_norm','clean_norm','ratio_nonzero']]
            lines.append(f"|{cell}|{side}|{values[0]:.6f}|{values[1]:.6f}|{values[2]:.6f}|{sum(m['zero_residual_count'] for m in ms)}/{count}|{sum(m['zero_rank_count'] for m in ms)}/{count}|")
    lines+=['','每配置的范数分位数、std、r_eff分布、故障库统计、正常/故障图像分数变化见 [MECHANISMS.json](MECHANISMS.json)。查询PCA统计跨种子去重（20个正常库配置）；跨参照和预算有重复图像，不当作独立patch样本。', '',
      '6. 资源成本：', '',
      f"准备/复现/探针 {cost['prepare_seconds']:.2f}s，全量缓存评分 {cost['full_seconds']:.2f}s，合计 {cost['measured_compute_workflow_seconds']:.2f}s；报告另用 {cost['report_seconds']:.2f}s。这是本轮执行阶段墙钟，不是CUDA kernel计时或完整推理速度。", '',
      f"峰值allocated显存 {cost['peak_allocated_bytes']/2**30:.3f} GiB；映射特征payload {cost['mapped_feature_payload_bytes']:,} bytes；N/F库payload {cost['normal_fault_bank_payload_bytes']:,} bytes；新增raw+clean残差库 {cost['added_residual_bank_payload_bytes']:,} bytes；全部本轮缓存 {cost['cache_total_bytes']:,} bytes。临时查询残差/投影按batch保留，跨3种子共享；另存范数诊断缓存。", '',
      '图像batch=32，局部SVD块=2048。查询和故障支持复用历史P3缓存；正常参照39张缺失缓存，各按历史单图提取一次。未进行部署测速，不引用此前加速比。', '',
      '## 冻结、局限与复现', '',
      f"本轮torch={frozen['env']['torch']}，CUDA={frozen['env']['cuda']}；历史torch={frozen['historical_environment']['torch']}。版本差异已披露，未放宽复现容差。FP32薄SVD gesvdj；k12/r4/alpha.8/eps1e-8/rank_tol1e-6，未搜索参数。", '',
      '详见 [METHOD_SPEC.md](METHOD_SPEC.md)、[FROZEN_INPUTS.json](FROZEN_INPUTS.json)、[BANK_MANIFEST.json](BANK_MANIFEST.json)、[CACHE_MANIFEST.json](CACHE_MANIFEST.json)、[INDEPENDENT_AUDIT.json](INDEPENDENT_AUDIT.json)。冻结输入包括全部源图、权重、名单、代码、历史分数、划分SHA256。骨干与mapper参数/BN状态前后全SHA256一致。', '',
      'k是历史裁剪支持预算，不代表独立源帧数。源身份未知不记零重叠。没有查询标签驱动阈值或调参；只做探索性图像级AUROC，未验证校准召回、误报率或完整H1。不同参照组重叠、查询共享，不独立。', '',
      '## 固定样本可视化', '',
      '按冻结路径顺序每资产预算前2正常和前2故障图，seed0/ref3001；未按提升挑选，无另选失败案例。A/B/C固定[-1,1]，范数固定[0,2]；无逐图min-max。', '',
      '![范数分布](vis/norm_distributions.png)', '']
    for item in selection:
        lines.extend([f"{item['id']}：`{item['path']}`",'',f"![{item['id']}](vis/{item['id']}.png)",''])
    lines+=['## 执行命令','', '从工作目录 `/root/auto-exp` 执行；现有结果禁止覆盖，复现需先由人类安排独立输出/归档。','',
      '```bash',f'python3 scripts/start_task.py plan/tasks/{E.TASK}.yaml',
      '/root/miniconda3/bin/python scripts/m3nve_run.py prepare',
      '# 确认cost_estimate.json full_authorized=true且复现/正确性通过后：',
      '/root/miniconda3/bin/python scripts/m3nve_run.py full',
      '/root/miniconda3/bin/python scripts/m3nve_report.py',
      f'python3 scripts/accept_check.py plan/tasks/{E.TASK}.yaml','```','',
      '日志prepare.log/full.log在服务器保留；环境ENVIRONMENT.txt；Git提交及同步状态另见HANDOFF.md。验收不以击败PatchCore为条件。不自动启动IDEAL训练、参数搜索或下一轮任务。','']
    (out/'NVE_REPORT.md').write_text('\n'.join(lines))
    E.save('REPORT_COMPLETE.json',dict(independent_audit=True,visualizations=len(selection)+1,report_seconds=time.perf_counter()-start))
    print(json.dumps(dict(independent_audit=True,rows=image_rows,max_A_diff=worst,summary=result['summary']),ensure_ascii=False,indent=2))


if __name__=='__main__': main()
