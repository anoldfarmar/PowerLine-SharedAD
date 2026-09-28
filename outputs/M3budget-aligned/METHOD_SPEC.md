# METHOD_SPEC — M3BUDGET-aligned-v1 (冻结于首次新增查询评分前)

## 1. 数据与身份
- 支持: SUPPORT_LISTS.json N5/N10 (严格复用; N5⊂N10 断言); 查询: query_k5 (各预算/方法同一集合)。
- 支持与查询不相交断言; glass 历史仅 g2。
- 权重: YOLO best.pt (sha256 冻结); WRN timm wide_resnet50_2.racm_in1k (加载态 feature_extractor 哈希须等于
  M3TRANSFER-wrnfix audit_evidence.json 记录值, 不符即停)。

## 2. 特征路径 (沿用最新核验实现, 单一共享函数)
- YOLO: best.pt -> feats[4] P3 (B,64,40,40 断言) -> NCHW dim=1 逐位置 L2 -> fp32; 确定性非 TF32。
  特征缓存 (uniq = N5∪N10∪查询, 去重) 重建于数据盘 /root/autodl-tmp/m3budget-{asset}-cache;
  确定性卷积保证与 M3SCORE/M3TRANSFER 历史缓存逐位一致 (锚点核验)。
- WRN: feature_extractor -> feature_pooler(逐层, AvgPool2d(3,1,1)) -> generate_embedding -> reshape_embedding
  (m3wrnfix 的 std_extract); 无 L2; 320 输入 + to_tensor + resize + normalize。
  官方一致烟测: train 分支返回 embedding vs std_extract (小样本, atol/rtol=1e-4), 状态还原+哈希断言。

## 3. 建库 (PV.make_pm, RNG 隔离)
- 每配置 pc_seed=7000+(ref-3000); make_pm 内部重置 py/np/torch/cuda; 探针/验证在前不改变正式序列。
- YOLO 10N: 10x1600=16000 patch -> ~1600 原型; WRN 5N: 5x1600=8000 patch -> ~800 原型 (以实际 shape 记录)。
- 双建库 torch.equal: 每新增分支 (YOLO-10N, WRN-5N) yoke+glass ref3001。
- 复用分支库哈希核对: YOLO-5N vs M3SCORE/M3TRANSFER 清单, WRN-10N vs wrnfix BANK_MANIFEST。

## 4. 评分 (同库共享)
- PC: 官方 nearest_neighbors(1) 距离/位置 + compute_anomaly_score(nn=9)。
- T8: 同一距离矩阵沿每图 patch 轴 top8 均值; 无重加权; top-k 固定 8; WRN 用一般欧氏距离。

## 5. batch 规则与批量精度门 (正式评分前 assert)
- 探针: YOLO 缓存行 chunk 32/64; WRN 提取 16/32。吞吐+显存 (预留 20%) 记录; 不按 AUROC 选择。
- 门: 特征 bs=1 vs 正式批 (YOLO: 活图提取 vs 缓存行, 三种代表图: 正常支持/正常查询/故障查询;
  WRN: 同上), atol=rtol=1e-4; 固定库 PC+T8 两种 batch 一致; 直接 vs 分块距离 (>=2 块+尾块,
  2500 行) vs float64; 两模型 state_dict 哈希前后一致。
- 失败: 检查 TF32/autocast/预处理/实现 -> 减半 batch 重验; 不放宽容差; 不带病评分。

## 6. 复用规则 (REUSE_MANIFEST)
- 复用 40 配置逐图分数原样复制为新命名 + 记录 source_task/source_commit(查询该文件最后提交)/文件 sha256。
- 独立重算 80 配置平均秩 AUROC (<1e-12)。
- 锚点: YOLO-5N yoke+glass 各一组从当前代码重评分 (库哈希核对+逐图 max|diff|<=1e-4);
  WRN-10N yoke 重评分 (库哈希+逐图), glass 库哈希核对。

## 7. 分析 (预注册)
主表: 资产 x 预算 x {Y-PC,Y-T8,W-PC,W-T8} 5 组均值/范围/组间 std。
比较: 同预算 Y-T8-W-T8, Y-T8-W-PC; YOLO 10N-5N (PC/T8); WRN 10N-5N (PC/T8); 各组合内 T8-PC。
每项: 每组配对差/均值/范围/ddof0 std/正负组数。无显著性声明; 系统比较非因果。

## 8. 产物与预算
见合同 acceptance; 大缓存放数据盘; GPU 1h 上限; 评分吞吐非端到端延迟。
