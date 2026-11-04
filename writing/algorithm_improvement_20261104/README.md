> **公开副本说明：**仅包含经筛选的文本、代码和汇总结果；下述历史本地包说明中提及的逐片数据、权重、私有文件及部分附件未公开。实际范围见 [PUBLIC_EXPORT.md](../PUBLIC_EXPORT.md)。

# 算法改进实验：2026-11-04

已完成现有 UNI2-h 特征上的算法改进实验：10种配置的30次初筛训练、内部锁定候选后的4次新增种子训练、2个确定性线性对照，以及重复bag与集成评价。GPU 3主流程实测10.35分钟，峰值分配显存4.70GiB。所有结果均通过独立CPU核验。

**结论：尚无稳定性能提升。** 内部初筛选定Attention＋L2，内部三种子平均AUROC为0.9365（Hybrid对照0.9353）；全部五种子的外部平均AUROC为0.8061（对照0.8056）。模型参数减少56.75%，但新增种子及集成未提供性能优越的证据。原论文v7及历史checkpoint保留。

## 阅读结果

- [完整中文实测报告与下一轮优先方向](RESULTS_REPORT_zh.md)
- [文献调研与适配依据](LITERATURE_REVIEW_zh.md)
- [图A1：全部配置及逐种子表现](figures/Figure_A1_all_configurations.pdf)
- [图A2：新增种子与集成效果](figures/Figure_A2_stability_and_aggregation.pdf)
- [图A3：锁定集成的ROC及混淆矩阵](figures/Figure_A3_locked_ensemble_ROC_confusion.pdf)
- 全部统计汇总（完整文件保留于本地档案）与独立核验报告（完整文件保留于本地档案）

图件另提供SVG及450dpi PNG。跨种子AUROC均值与平均预测得到的ensemble AUROC分别记录，不可互换。

## 入口

- `LITERATURE_REVIEW_zh.md`：八篇论文原文/官方代码及本地适配边界。
- `CODE_AUDIT_zh.md`：现有结构、训练行为和可改进点。
- `PROTOCOL.md`：训练前固定的正式方案；优先于早期建议 `EVALUATION_DESIGN_zh.md`。
- `configurations.json`：十个神经网络配置、三个初筛种子及两个新增种子。
- `models.py` / `runner.py`：模型与完整执行流程。
- `test_models.py` / `qa/`：CPU模型检查、训练前独立审计、完成后的结果核验。
- `run_console.log` / `outputs/`：完整训练日志、所有权重、预测、采样索引、选择锁和运行环境。
- `analyze_results.py`：训练完成后的固定统计和图表生成。
- `write_report.py`：从统计结果生成中文实测报告。

研究流程为30次初筛训练 → 只按内部AUROC锁定一个候选 → 候选与Hybrid control各补两个种子 → 所有已冻结模型统一外部评价。两个确定性线性对照另列，不参与神经网络选型。全部完成与失败记录均保留；外部结果不触发本轮重新选型。

## 计算和运行

使用 `python`，软件版本见 `requirements_used.txt`。仅授权物理 GPU 3，单进程、Torch CPU线程2、显存上限15%。GPU须由宿主执行环境提供；受限沙箱可能未映射NVIDIA设备。

```bash
CUDA_VISIBLE_DEVICES=3 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MPLCONFIGDIR=/tmp/trident-algorithm-mpl python -u runner.py
```

程序拒绝覆盖已有运行目录。重跑需要新建独立目录并保留本次原始记录，不能删去不利结果后重复同一路径。训练脚本使用项目原特征文件与 v7 的编号映射，具有本地依赖，不是无需数据的独立公开基准。

## 解释范围

原数据划分和历史外部队列按作者既有核查结果沿用。外部队列已参与前期开发观察，因此本轮改进结果为算法开发与历史外部评价证据，不能称为全新的确认性外部验证。内部选择、额外种子稳定性、固定阈值性能和重复采样/集成效果分别报告。

医院已有资料继续留空。真实患者标识、特征路径及编号对应仅保存在私有输入表，未公开发布。所有 checkpoint 留在本地。
