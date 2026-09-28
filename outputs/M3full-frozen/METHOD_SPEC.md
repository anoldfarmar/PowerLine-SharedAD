# METHOD_SPEC — M3FULL-frozen-benchmark-v1 (冻结于首次新增查询评分前)

## 1. 数据 (preflight 已定, 全部 7 组合可行)
- 查询: 冻结 query_k5 全部 7 故障组合 + 5 资产正常查询 (glass 705 / lightning 315 / polymer 725+22 / vari 358+295 / yoke 295; NQ 各 500)。
- 支持: yoke/glass 直接复用 M3BENCH SUPPORT_LISTS N5/N10 (M3BUDGET 已核验);
  新增 3 资产从合格池 (defect_supervised {train,val}/{good,normal}, 路径级排除全部查询路径)
  排序后 numpy Generator(PCG64(ref_id)) 置乱取前 10 (N10)、前 5 (N5); 组间重叠矩阵记录。
- 源组身份: 按 M3DIAG5 审计口径 — 裸 DJI 重号严重, 支持与查询的源组交集只记录不解读;
  不宣称独立未见/骨干未见; 支持以裁剪计数。

## 2. 方法 (沿用 M3BUDGET 冻结实现, 单一共享函数)
- Y-PC/Y-T8: best.pt → P3 (B,64,40,40) → 逐位置 L2 → 官方 coreset → 官方 PC / 同距离 top8。
- W-PC/W-T8: timm wide_resnet50_2.racm_in1k (加载态哈希须=wrnfix 审计值) → layer2+layer3 →
  官方 feature_pooler → generate_embedding → reshape_embedding → 官方 coreset → PC / top8。无额外 L2。
- 320 输入, 各自预处理沿用 M3BUDGET; ratio=0.1, nn=9, top-k=8, pc_seed=7000+(ref−3000)。
- 同库两评分共享同一次最近邻; 同资产多故障共用正常库与 NQ 评分。

## 3. 正确性门 (正式评分前 assert, 门限沿用 M3BUDGET)
数据集合/不相交; 权重哈希; WRN 官方路径烟测; 官方 pred_score 对照; batch(支持/NQ/故障Q) bs1 vs 正式批
atol=rtol=1e-4; 分块距离跨块+尾块 vs fp64; top8 patch 轴合成验证; 状态哈希不变;
双建库 torch.equal (每新增资产×主干≥1组); 库 manifest 全存档。

## 4. 复用与锚点
- M3BUDGET 80 配置逐文件复制 + 溯源 (task/commit/sha256) + AUROC 独立重算 (≤1e-12)。
- 锚点: YOLO-5N yoke ref3001、WRN-10N yoke ref3001 从当前代码重评分, 逐图 ≤1e-4; 库哈希核对。

## 5. 计算 (GPU 2h 上限)
FP32 确定性非 TF32; YOLO bs32 缓存 (uniq=N10∪查询, 按路径去重), 评分 chunk 64 图; WRN bs32 流式;
主干各加载一次; 每查询 batch 服务多库; 大缓存数据盘; 预检磁盘空间。

## 6. 统计 (预注册)
每组合×预算×方法: 5 组 AUROC 均值/范围/std/配对差/正负组数。
宏平均: 每参照组跨组合平均→5 组汇总; 资产等权: 资产内故障平均→跨 5 资产。
比较: Y-T8−W-PC (主), Y-T8−W-T8 (对照), T8−PC (各主干×预算), 10N−5N (各方法)。
无显著性声明; 退化组合单独列出。

## 7. 结论边界
可回答: 冻结方案跨组合准确率竞争力。不可证明: 源帧独立/骨干未见/新确认集泛化/召回误报达标/部署达标/双库创新。
