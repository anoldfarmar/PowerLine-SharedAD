#!/usr/bin/env python3
"""Read-only statistical reconstruction; no invented multi-box aggregation."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import numpy as np
import insplad_shared_roi as old
from m3share_speed_core import METHODS


def method_spec(frozen):
    return f'''# M3SHARE-speed-v1 方法与计时边界（测速前冻结）

## 数据身份：只允许局部结论

本机可可靠使用的是旧 InsPLAD `defect_supervised` 原始文件本身，亦即先前已裁剪好的小图；不是已确认对应的1920×1080原始航拍帧。
没有通过裸DJI号猜原图。检测验证大图与异常小图缺少可核验的跨包桥接，数据身份审计见 DATA_AUDIT.json。
因此本任务实际完成的是 **已解码的小图文件→检测→诊断输出** 的局部流水线成本分析，不是用户最终关心的原始大场景端到端速度。
速度集为每资产固定路径哈希前64张，共320张；按原有五资产顺序运行，不因框数、标签、耗时或分数筛选；包括全部无框样本。
辅助精度使用全部5215张冻结查询。支持恰为原10N/ref3001的50张文件；增强或多个预测框不计作新增独立shot。

## 保留旧 A/C 的真实定义

- 原 A：已有小图→RGB Resize320→COCO原生P3→匹配正常库→top8，没有从整幅航拍图开始，也没有部件检测/再次裁剪。
  历史MC.Backbone的forward_layers实际上执行0–9层但使用第4层特征；该额外P4/P5工作不作为本轮两阶段参照负担。
- 原 C：同一已有小图→冻结14类检测模型完整前向，hook取得同一次前向第4层P3→已知资产最高置信度框RoIAlign→C路径正常库→top8。
  无指定类别框时取全图ROI，仍执行异常诊断。原C只保留一个框，不能用它原样研究多部件摊销。
- 原ABC脚本一次完整检测前向同时派生A/B/C特征，并共享了缓存；其97.74秒是整轮实验开销，**不是三个独立系统的推理延迟**。

原 A/C 名称、库和历史精度保持不变；本轮另加下列参照，不把新参照改称旧A/C。

## 新参照臂

| 臂 | 实际路径 | 主干次数 |
|---|---|---|
| T-CROP-small | 小图原始分辨率文件→检测→全部指定资产预测框从原文件裁剪→Resize320→批量前缀0–4→本臂正常库→top8 | 每原图batch一次完整检测；每最多32个裁剪一次额外前缀 |
| T-SHARE-small | 相同小图与检测→同次完整前向P3→全部相同框批量RoIAlign→本臂正常库→top8 | 每原图batch一次完整检测；额外裁剪前缀0次 |

检测检查点：`{frozen['weights']['detector']}`；SHA256 `{frozen['weight_sha256']['detector']}`。
两个新臂使用同一检测模型实例；Crop额外前缀是其前五层的独立深拷贝，权重相同但确实另做前向。共享权重不等于共享前向。
检测模型原初始化COCO检查点与前缀冻结身份沿用已通过的上游审计；本轮以状态哈希和实际hook计数再次检查推理不变性。

共同配置：检测输入PIL RGB `Resize((320,320))→ToTensor`（拉伸，**没有LetterBox**），conf=.25、NMS IoU=.7、max_det=300。
按已知资产类别过滤，保留全部有效框，置信度降序后按xyxy字典序；不使用故障标签选框或异常分数选框。
框先还原到原文件坐标；Crop按左上floor、右下ceil并裁边界，Share按真实缩放系数投影回320输入坐标。
RoIAlign第4层P3、stride8、64通道、40×40、aligned=True、sampling_ratio=1，之后通道L2；Crop同样得到1600×64通道L2特征。
合成门另检验显式LetterBox仿射，但计时路径本身没有LetterBox，不能将该测试说成使用了LetterBox。

## 正常库和回退

两臂正常支持文件、所用检测框对象相同。每臂分别按自己的特征流程提取全部正常ROI，官方KCenterGreedy ratio=.1、seed7001；不把Crop库当作Share库。
原vari-grip的10张支持有3张各有2个目标候选，故新臂有13个正常ROI（20800候选patch，2080库patch）；其他资产各10ROI（1600库patch）。
这是新all-box参照；不把它与原C每文件top1库的差异隐去。没有用查询建库。
无框时沿用原C明确的“全图回退并诊断”：Crop重新对整张原文件运行一次前缀，Share对现有P3取全图ROI。
两臂都输出回退标记和分数，不把无框记为正常，也没有跳过无框诊断。
精度为FP32、TF32关闭；精确欧氏1NN，patch top8均值；分块仅控制内存，不是近似检索。

## 正确性、探针与正式计时

先执行：同框/同支持对象、合成坐标与零框/多框/边界/非方形路径、裁剪batch1对32、原图batch1对8/16、实际分块对直接距离、状态哈希和hook次数。
固定atol=rtol=1e-4；超限的8/16候选被排除，必要时回到原图batch1，绝不放宽容差；Crop内部32对1不通过则停机诊断。
几何门的强制框只是合成夹具，不是正式检测结果或速度样本。
正式计时必须每次执行检测和必要的裁剪前缀，没有从特征缓存读取的入口；正常库加载/构建和模型加载单列，不计稳态。

主边界：已解码RGB原文件→预处理与H2D→检测/NMS/坐标→原文件裁剪Resize及H2D或RoIAlign→必要额外前缀→精确检索→CPU框/分数输出。
完整调用前后CUDA同步，中间不为分阶段记录同步。分阶段profile是另外独立执行的一组调用，在阶段边界同步，不机械累加冒充主计时。
读盘解码单独写DISK_IO.json；准备阶段已读过文件哈希，因此不宣称冷盘IO。
单帧原图batch1，图内裁剪批量；吞吐采用精度门通过的共同batch，两臂相同。probe在正式计时前按已冻结规则选择共同batch、预热和重复次数，写TIMING_PROTOCOL.json后不再改。
两臂按帧和轮次交替；所有轮次保留，不挑最优。每轮前检查其他CUDA进程；检查不等于排除了所有温度、桌面或系统噪声。
峰值显存报告CUDA已分配值；为交替公平，两臂库及同模型常驻，不将其冒充孤立部署显存。

## 精度口径

旧A/C全量及有框条件精度单独重算并保留名字，不混同新all-box库。
新臂只在“恰好一个目标框”的共同图像子集和“无框全图回退”子集报告图像标签AUROC；这不是逐部件GT精度。
多框图只保存每个框的分数，不临时定义max/mean整图聚合，不报告臆造的全量新系统AUROC。
不选告警阈值、不报告F1或像素定位准确率；有框不代表定位正确，无框不代表正常。
'''


def auc_subset(normal,defect,normal_scores,defect_scores):
    result=dict(normal_count=len(normal),defect_count=len(defect),auroc=None)
    if normal and defect:
        record=dict(normal={p:normal_scores[p] for p in normal},defect={p:defect_scores[p] for p in defect})
        result['auroc']=old.auc(record,normal,defect)
    else:
        result['reason']='AUROC undefined: missing a label class in this fixed subset'
    return result


def accuracy_summary(source,frozen,records):
    legacy={};local={};zero_counts={}
    for asset,data in frozen['source_queries'].items():
        zero_counts[asset]=dict(normal=sum(frozen['frames'][p]['target_box_count']==0 for p in data['normal']),
            defects={fault:sum(frozen['frames'][p]['target_box_count']==0 for p in ps) for fault,ps in data['defects'].items()},
            note='No expected-class output, not GT detector recall')
        for fault,defect in data['defects'].items():
            key=f'{asset}/{fault}';legacy[key]={};local[key]={}
            for arm in ['A','C']:
                r=old.read(source/'raw_scores'/f'{asset}__{fault}_{arm}.json')
                assert r['frozen_id']==frozen['upstream_frozen_id']
                old.auc(r,data['normal'],defect)
                legacy[key][arm]={}
                for name,predicate in [('all_queries',lambda n:True),('at_least_one_box',lambda n:n>0),('no_box',lambda n:n==0)]:
                    ns=[p for p in data['normal'] if predicate(frozen['frames'][p]['target_box_count'])]
                    ds=[p for p in defect if predicate(frozen['frames'][p]['target_box_count'])]
                    legacy[key][arm][name]=auc_subset(ns,ds,r['normal'],r['defect'])
            for name,nboxes in [('exactly_one_box',1),('no_box_full_frame_fallback',0)]:
                ns=[p for p in data['normal'] if frozen['frames'][p]['target_box_count']==nboxes]
                ds=[p for p in defect if frozen['frames'][p]['target_box_count']==nboxes]
                local[key][name]={}
                for method in METHODS:
                    for p in ns+ds:
                        assert len(records[method][p]['scores'])==1
                    normal_scores={p:records[method][p]['scores'][0] for p in ns}
                    defect_scores={p:records[method][p]['scores'][0] for p in ds}
                    local[key][name][method]=auc_subset(ns,ds,normal_scores,defect_scores)
    return dict(legacy_A_C=legacy,new_local_arms=local,no_box_label_counts=zero_counts,
         all_queries_retained=all(set(records[m])==set(frozen['accuracy_query_ids']) for m in METHODS),
         new_all_query_system_auroc=None,reason='No frozen multi-box image aggregation or component-level anomaly GT; do not manufacture all-query AUROC',
         labels='image-level only, not per predicted object',threshold_selected=False,
         source_identity_limit='No reliable raw-full-scene mapping; local native-crop inputs only')


def stats(values):
    x=np.asarray(values,dtype=np.float64)
    if len(x)==0:
        return dict(n=0,mean=None,median=None,p95=None)
    return dict(n=len(x),mean=float(x.mean()),median=float(np.median(x)),p95=float(np.percentile(x,95)))


def compute_metrics(frozen,protocol,rows):
    methods={};groups={};profiles={};rounds={};counts=Counter(frozen['frames'][p]['group'] for p in frozen['speed_ids'])
    for method in METHODS:
        single=[r for r in rows if r['method']==method and r['phase']=='single_frame']
        batches=[r for r in rows if r['method']==method and r['phase']=='throughput_batch']
        stages=[r for r in rows if r['method']==method and r['phase']=='stage_profile']
        assert len(single)==len(frozen['speed_ids'])*protocol['latency_repeats']
        for repeat in range(protocol['latency_repeats']):
            assert [r['frame_ids'][0] for r in single if r['repeat']==repeat]==frozen['speed_ids']
        for repeat in range(protocol['throughput_repeats']):
            actual=[p for r in batches if r['repeat']==repeat for p in r['frame_ids']]
            assert actual==frozen['speed_ids']
        has_box=[r for r in single if r['target_boxes'][0]>0]
        milliseconds=sum(r['elapsed_ms'] for r in batches)
        frames=sum(r['batch'] for r in batches)
        methods[method]=dict(single_frame_ms=stats([r['elapsed_ms'] for r in single]),
            same_has_box_subset_ms=stats([r['elapsed_ms'] for r in has_box]),
            has_box_amortized_ms_per_predicted_component=sum(r['elapsed_ms'] for r in has_box)/sum(r['target_boxes'][0] for r in has_box),
            throughput_batch=protocol['chosen_batch'],throughput_frames=frames,throughput_total_ms=milliseconds,
            decoded_input_frames_per_second=1000*frames/milliseconds,
            single_frame_counter_totals={key:sum(r['counters'][key] for r in single) for key in single[0]['counters']})
        groups[method]={name:stats([r['elapsed_ms'] for r in single if frozen['frames'][r['frame_ids'][0]]['group']==name])
                        for name in ['0','1','2-4','>=5']}
        profiles[method]={name:stats([r['stages'][name] for r in stages]) for name in stages[0]['stages']}
        rounds[method]={str(i):stats([r['elapsed_ms'] for r in single if r['repeat']==i]) for i in range(protocol['latency_repeats'])}
    for r in rows:
        assert r['frozen_id']==frozen['frozen_id'] and r['batch']==len(r['frame_ids'])
        assert r['counters']['full_backbone_calls']==1 and r['counters']['full_images']==r['batch']
        if r['method']==METHODS[1]:
            assert r['counters']['crop_backbone_calls']==0 and r['counters']['crop_images']==0
        else:
            assert r['counters']['crop_images']==r['effective_regions']
        assert r['target_boxes']==[frozen['frames'][p]['target_box_count'] for p in r['frame_ids']]
    crop,share=METHODS
    ratios=dict(local_complete_mean_speedup=methods[crop]['single_frame_ms']['mean']/methods[share]['single_frame_ms']['mean'],
        local_complete_median_speedup=methods[crop]['single_frame_ms']['median']/methods[share]['single_frame_ms']['median'],
        same_has_box_mean_speedup=methods[crop]['same_has_box_subset_ms']['mean']/methods[share]['same_has_box_subset_ms']['mean'],
        diagnostic_stage_mean_speedup=profiles[crop]['diagnostic_stage_ms']['mean']/profiles[share]['diagnostic_stage_ms']['mean'],
        throughput_speedup=methods[share]['decoded_input_frames_per_second']/methods[crop]['decoded_input_frames_per_second'])
    for name in ['0','1','2-4','>=5']:
        ratios[f'group_{name}_speedup']=groups[crop][name]['mean']/groups[share][name]['mean'] if groups[crop][name]['n'] else None
    return dict(scope=frozen['scope'],methods=methods,box_count_latency_ms=groups,profile_ms=profiles,
                ratios=ratios,per_round_ms=rounds,speed_frame_box_groups=dict(counts),
                no_box_rate=counts['0']/len(frozen['speed_ids']),
                actual_target_box_distribution=dict(Counter(str(frozen['frames'][p]['target_box_count']) for p in frozen['speed_ids'])),
                quantile='numpy percentile95 linear over all predeclared single-frame repetitions; no best-round selection',
                raw_scene_end_to_end_speedup=None)


def report(out):
    out=Path(out);frozen=old.read(out/'FROZEN_INPUTS.json');protocol=old.read(out/'TIMING_PROTOCOL.json')
    rows=[json.loads(line) for line in (out/'TIMING_RAW.jsonl').read_text().splitlines() if line.strip()]
    assert old.read(out/'REPRO_AND_CORRECTNESS.json')['passed']
    metrics=compute_metrics(frozen,protocol,rows)
    old.write(out/'SPEED_METRICS.json',metrics)
    acc=old.read(out/'ACCURACY_SUMMARY.json');cost=old.read(out/'RESOURCE_COST.json')
    crop,share=METHODS;ratios=metrics['ratios']
    lines=['# M3SHARE-speed-v1：小图局部流水线实际成本','',
      '**结论范围：没有完成原始航拍大场景端到端速度验证。** 可靠原图—异常小图对应缺失；下列数字来自既有小图文件的实际检测＋诊断调用，不从特征缓存假装推理。','',
      f"320张固定计时样本；全部5215查询辅助精度分析。单帧重复{protocol['latency_repeats']}轮，吞吐batch={protocol['chosen_batch']}、重复{protocol['throughput_repeats']}轮。",'',
      '| 臂 | 单帧mean ms | median ms | P95 ms | 吞吐：输入文件/s | 有框每预测部件摊销ms |','|---|---:|---:|---:|---:|---:|']
    for method in METHODS:
        m=metrics['methods'][method];s=m['single_frame_ms']
        lines.append(f"| {method} | {s['mean']:.3f} | {s['median']:.3f} | {s['p95']:.3f} | {m['decoded_input_frames_per_second']:.2f} | {m['has_box_amortized_ms_per_predicted_component']:.3f} |")
    lines += ['',f"局部完整调用mean加速比 **{ratios['local_complete_mean_speedup']:.3f}×**；仅异常诊断阶段mean加速比 **{ratios['diagnostic_stage_mean_speedup']:.3f}×**（独立分阶段profile，不与完整系统混用）。",
      f"相同有框子集完整调用加速比 {ratios['same_has_box_mean_speedup']:.3f}×；吞吐改善 {ratios['throughput_speedup']:.3f}×。",'',
      '## 框数、无框与收益','',f"无有效指定类别框比例 {100*metrics['no_box_rate']:.2f}%；保留并执行全图回退诊断，不按正常处理。",'',
      '| 每输入文件目标框数 | 独立样本数 | Crop mean ms | Share mean ms | 加速比 |','|---|---:|---:|---:|---:|']
    for name in ['0','1','2-4','>=5']:
        n=metrics['speed_frame_box_groups'].get(name,0)
        a=metrics['box_count_latency_ms'][crop][name]['mean'];b=metrics['box_count_latency_ms'][share][name]['mean']
        lines.append(f'| {name} | {n} | '+(f'{a:.3f} | {b:.3f} | {a/b:.3f}× |' if n else 'N/A | N/A | N/A |'))
    one=ratios['group_1_speedup'];multi=ratios['group_2-4_speedup']
    if one is not None and multi is not None:
        lines += ['',f'本批1框组加速比{one:.3f}×，2–4框组{multi:.3f}×；'+
                  ('多框组收益更大，但只是这些既有样本的条件比较，不是控制目标数的因果实验。' if multi>one else '本批没有呈现多框组收益更大的现象，不能预设共享收益随框数单调增加。')]
    lines += ['', '空组不合成数据。框数是同检测规则下的有效目标候选，不保证定位正确，也不是人工部件总数。', '',
      '## 分阶段profile（独立调用，各阶段同步）','',
      '| 阶段 | Crop mean ms | Share mean ms |','|---|---:|---:|']
    for key in ['preprocess_transfer_ms','full_detection_nms_coordinates_ms','crop_resize_transfer_or_roi_ms','extra_crop_prefix_ms','nearest_neighbor_score_ms','diagnostic_stage_ms']:
        lines.append(f"| {key} | {metrics['profile_ms'][crop][key]['mean']:.3f} | {metrics['profile_ms'][share][key]['mean']:.3f} |")
    lines += ['', '分阶段包含必要传输/CPU操作；不机械求和等同独立完整延迟。读盘解码见DISK_IO.json，属于另外的暖文件系统缓存读取，不含于主延迟。', '',
      '## 精度：只在有定义的评分对象上比较','',
      '旧A/C定义不变，旧全量结果另存ACCURACY_SUMMARY.json；新臂采用all-box正常库，不能把原C的top1正常库直接移用。',
      '新多框路径没有预先定义整图聚合、也没有逐预测部件异常GT，故不报告伪造的全量新系统或部件AUROC。下表只用同一恰好1框的图像标签子集。','',
      '| 组合 | 正常/异常数 | Crop AUROC | Share AUROC | Share−Crop pp |','|---|---:|---:|---:|---:|']
    for combo,value in acc['new_local_arms'].items():
        a=value['exactly_one_box'][crop];b=value['exactly_one_box'][share]
        cell=f"{a['auroc']:.4f} | {b['auroc']:.4f} | {100*(b['auroc']-a['auroc']):+.2f}" if a['auroc'] is not None and b['auroc'] is not None else 'N/A | N/A | N/A'
        lines.append(f"| {combo} | {a['normal_count']}/{a['defect_count']} | {cell} |")
    lines += ['', '无框全图回退子集的正常/异常构成和AUROC（若两类齐全）也保存在ACCURACY_SUMMARY.json。无框率不是基于框GT的漏检率。', '',
      '## 成本、实现与最终回答','',
      f"- 实际模型加载 {cost['model_load_seconds']:.3f}s；建库时间、每库候选数量/库大小见BANK_MANIFEST.json和RESOURCE_COST.json，均未混入稳态。",
      '- 两臂实际计数器见每条TIMING_RAW：Share完整检测1次/输入batch、额外裁剪前缀0次；Crop额外前缀按最多32裁剪分批，不逐部件故意低效前向。',
      '- 确实共享的是同一次检测前向的第4层空间P3，而不是仅共用权重。旧A/C整轮缓存耗时未拿来作推理延迟。',
      '- 完整系统的原始大场景加速仍为未知；上面的“完整调用”只在预裁剪小图边界成立。正常库匹配各自路径，vari-grip含13正常ROI（仍是10文件预算）。',
      '- 精度和收益是否随部件数变化，只按实有样本表述；非方形/边界/35框合成门是正确性测试，不补充正式≥5框速度样本。',
      '- 无P2新融合、无重新出框机制、无训练/蒸馏/Adapter、无阈值调优、无像素GT或定位准确率声明。',
      '- 下一步首先人工核查本报告计时边界、框与回退；没有可靠原图身份前，不建议把本地结果包装成三路原始分辨率对照。若实现/计数/精度门失败，先修正并保留失败记录，而不是放宽门限或追加训练。',
      '- 本轮不自动启动下一方向。只有补齐可靠原始场景及支持对象对应后，才适合另立真实原图分辨率对照。', '',
      '峰值显存、主机RSS、全部轮次、环境及启动/建库成本详见机读文件；交替运行时两臂库与同模型常驻，显存不是孤立部署配置。','']
    (out/'SPEED_REPORT.md').write_text('\n'.join(lines))
    return metrics
