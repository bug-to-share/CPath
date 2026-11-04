# 基于现有 UNI2-h 特征的算法改进调研

检索日期：2026-11-04（北京时间）。范围：小样本计算病理 MIL、冻结病理基础模型特征、随机性与跨来源泛化。本文是实验设计依据，不宣称下述方法已经在本项目取得改善；具体执行配置和实测结果以本目录实验方案、代码与结果表为准。

## 对当前项目的判断

已阅读 v7 README 与受控训练方案：已有固定 144/49 内部训练/评估划分、168 张外部切片、1536 维 UNI2-h 特征；此前完成 matched attention、fixed scaling、learned gate 三种配置各 3 个种子的受控训练。历史主模型与新训练 checkpoint 分开保存。医院已有但尚未导入的临床资料继续保留原有占位。

本轮最值得验证的假设是：**在固定高质量特征和小训练集条件下，减少聚合器自由度、保持原始特征空间、扩大每次更新的样本覆盖，并约束同片重复采样的一致性，可能比继续叠加分支更有效。**这是对项目条件和文献的推断，需要通过本地实验验证。

不能把高 AUROC 自动解释成固定阈值下的高敏感度。模型排序能力、运行间稳定性和阈值工作点应分别评价；选种子、选阈值和算法改进也须分别记录。

## 八篇优先参考的原始研究

### 1. nnMIL：最直接对应本项目的新证据

- **题目**：*nnMIL: a generalizable multiple instance learning framework for computational pathology*。
- **作者/年份/地位**：Xiangde Luo、Jinxi Xiang、Yuanfeng Ji、Ruijiang Li；Nature Biomedical Engineering，2026-08-25 正式发表，DOI 10.1038/s41551-026-01767-8。注意另一 DOI `01766-9` 是 Research Briefing，不是原始研究。
- **核心**：在随机特征子空间计算 attention，但在原始全维特征上聚合；patch 采样配合大 batch；推断遍历特征子空间。
- **可核对设置**：子空间 256 维、dropout 0.25、默认 batch 32、AdamW 学习率 3e-4、weight decay 1e-4、5 epoch warmup 后 cosine；推断 stride=64。原文最多 100 epoch，不能把我们统一的 30 epoch 比较叫完整复现。
- **本项目落地**：1536 维特征对应 21 个 256 维滑窗；分类取平均 logit。先单测原空间聚合，再测随机子空间，最后测 batch32，避免打包解释贡献。若保留既有采样和训练方案，命名为 **nnMIL-inspired adaptation**。
- 来源：[期刊原文](https://www.nature.com/articles/s41551-026-01767-8)、[可读全文 v2](https://arxiv.org/html/2511.14907v2)、[官方代码](https://github.com/Luoxd1996/nnMIL)。

### 2. ACMIL：防止 attention 集中过度

- **题目**：*Attention-Challenging Multiple Instance Learning for Whole Slide Image Classification*。
- **作者/年份/地位**：Yunlong Zhang 等；ECCV 2024 正式论文；预印本最早为 2023 年，不能把预印本年份当会议年份。
- **核心**：多 attention 分支、分支多样性约束，以及训练时随机屏蔽 top-K 高 attention 实例（STKIM）。
- **可核对设置**：原文分析 K=10，多分支常用 M=5；mask 概率 0.6 或 0.8 在其不同数据集表现较好；推断关闭 mask。
- **本项目落地**：第一步只在 matched attention 上加 STKIM：K=min(10,N-1)，p=0.6，保留至少一个实例。这是 **ACMIL 的 STKIM 单组件改编**，不是完整 ACMIL。记录 top10 attention 累积质量、熵和重复 bag 预测波动，判断机制是否成立。普通提高 softmax temperature 不等于 ACMIL。
- 来源：[ECCV 原文 PDF](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06954.pdf)、[官方代码](https://github.com/dazhangyu123/ACMIL)。

### 3. PseMix：有限切片上的 bag 级增强

- **题目**：*Pseudo-Bag Mixup Augmentation for Multiple Instance Learning-Based Whole Slide Image Classification*。
- **作者/年份/地位**：Pei Liu、Luping Ji、Xinyu Zhang、Feng Ye；IEEE Transactions on Medical Imaging 43(5):1841–1852，2024，DOI 10.1109/TMI.2024.3351213。
- **核心**：先按表型划分实例并进行分层 pseudo-bag 采样，再混合 pseudo-bag 与标签；不依赖当前模型预测来生成增强。
- **可核对设置**：官方代码默认 pseudo-bag 数 30、表型数 8、聚类微调 8 次；mixup 使用 Beta 分布，混合后标签权重由实际 pseudo-bag 比例决定。
- **本项目落地**：完整 PseMix 可复用现成 embedding。若第一轮仅做同类切片的 256+256 patch 拼接，应称 **same-class bag mixing**，不能说复现 PseMix；其优点是保留 hard label，代价是改变同一切片内的组织组成。只从训练切片采样。
- 来源：[作者预印本](https://arxiv.org/abs/2306.16180)、[官方实现与设置](https://github.com/liupei101/psemix)。

### 4. C²Aug：同类跨 bag 增强的近期方向

- **题目**：*Contrastive Cross-Bag Augmentation for Multiple Instance Learning-based Whole Slide Image Classification*。
- **作者/年份/地位**：Bo Zhang 等；CVPR 2026，21089–21098；预印本最早 2025 年。正式会议页面的作者名单应优先于旧版预印本。
- **核心**：从相同 bag 标签的多个训练切片采样，结合 bag-level 和 group-level contrastive learning。原文同时指出，混合可能增加关键实例比例，导致对小病灶切片不利。
- **本项目落地**：可先采用同类两片各 256 patch、以 0.5 概率增强作为低成本 **inspired adaptation**；这些数值是本项目建议，不是原文默认参数。完整复现还需组别构造与两个 contrastive 目标，不能省略后沿用 C²Aug 名称。
- 来源：[CVPR 正式页面](https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_Contrastive_Cross-Bag_Augmentation_for_Multiple_Instance_Learning-based_Whole_Slide_Image_CVPR_2026_paper.html)、[预印本全文](https://arxiv.org/html/2508.03081v1)、[官方代码](https://github.com/weiaicunzai/mixup)。

### 5. DGR-MIL：为现有潜在原型增加明确训练目标

- **题目**：*DGR-MIL: Exploring Diverse Global Representation in Multiple Instance Learning for Whole Slide Image Classification*。
- **作者/年份/地位**：Wenhui Zhu 等；ECCV 2024，333–351，DOI 10.1007/978-3-031-72920-1_19。PMC 于 2025 年归档，不改变论文年份。
- **核心**：少量全局向量通过 cross-attention 汇总实例；向正类 bag 中心对齐、远离负类中心，并约束向量多样性。
- **可核对设置**：中心 momentum 0.4；triplet 与 diversity loss 权重各 0.1；原文不同数据集全局向量数约 3–5；200 epoch 训练中前 20 epoch 仅分类损失。
- **本项目落地**：第二阶段才考虑 class-conditioned prototype regularization，须与现有潜在原型分开消融。即使加了类标签约束，也只能说类相关潜在表征；不能把它自动命名为经病理验证的 MALT/SS 形态原型。
- 来源：[原文作者全文](https://arxiv.org/html/2407.03575v1)、[正式发表记录](https://pmc.ncbi.nlm.nih.gov/articles/PMC12425359/)、[官方代码](https://github.com/ChongQingNoSubway/DGR-MIL)。

### 6. HistAug：有吸引力，但当前特征兼容性不足

- **题目**：*Controllable Latent Space Augmentation for Digital Pathology*。
- **作者/年份/地位**：Sofiène Boutaj 等；ICCV 2025；方法名 **HistAug**。
- **核心**：训练条件生成器模拟图像变换后的 embedding，而非随意添加噪声；可对整张 WSI 使用一致的增强参数。
- **现有支持**：原文训练 UNI/CONCH 生成器；截至本次检索，官方仓库还列出 CONCH v1.5、Virchow2、H-optimus-1，未列 UNI2-h。
- **本项目落地**：不能将其他编码器的增强器直接视为 UNI2-h 兼容。先做普通 feature noise/dropout 只能称通用扰动基线。真正迁移 HistAug 需要原始 patch 的变换前后 UNI2-h 特征对及增强器训练，因此不列入本轮最高优先级。
- 来源：[原文全文](https://arxiv.org/html/2508.14588v1)、[官方代码及支持模型](https://github.com/MICS-Lab/HistAug)。

### 7. R-Drop：双视图一致性的通用依据

- **题目**：*R-Drop: Regularized Dropout for Neural Networks*。
- **作者/年份/地位**：Xiaobo Liang、Lijun Wu 等；NeurIPS 2021 正式论文。
- **核心**：同一样本经过不同 dropout 两次前向，最小化两个预测分布之间的双向 KL 散度。
- **本项目落地**：从同一训练切片抽两个 512-patch bag，同时计算各自分类损失与对称 Bernoulli KL；建议 λ=0.1 作为预定起点。由于同时改变了 patch 采样，这属于 **R-Drop-inspired bag consistency**，不是原论文原样复现。需与仅双视图平均分类损失的版本比较，才能区分“多看一袋实例”与一致性项的作用。
- 来源：[NeurIPS 正式论文页](https://proceedings.neurips.cc/paper_files/paper/2021/hash/5a66b9200f29ac3fa0ae244cc2a51b39-Abstract.html)、[官方代码](https://github.com/dropreg/R-Drop)。

### 8. Deep ensembles：先用已有 checkpoint 测试低成本稳定化

- **题目**：*Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles*。
- **作者/年份/地位**：Balaji Lakshminarayanan、Alexander Pritzel、Charles Blundell；NIPS 2017 正式论文。
- **核心**：组合独立训练模型的预测；原文包含更完整的训练与不确定性评估框架。
- **本项目落地**：对固定的 seeds 17、29、43 使用等权平均预测概率，无需重训练；所有种子均纳入、权重不据外部结果调整。称 **equal-weight seed ensemble** 即可，不声称完整复现。若不同种子对同类切片系统性犯同样错误，ensemble 也可能没有收益。
- 来源：[正式论文页](https://papers.nips.cc/paper_files/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html)、[作者预印本](https://arxiv.org/abs/1612.01474)。

## 建议先执行的五组实验

以下为本地实验建议；执行者可以因资源与现有代码调整，但应在查看新增外部结果前写入方案。

| 优先级 | 实验 | 建议固定设置 | 需要新增特征提取 | 关键比较 |
|---|---|---|---|---|
| P0 | 原始 embedding 简化对照 | 每片 mean pooling→训练集拟合 standardization→L2 logistic；C=0.1、1、10 仅内部选择；另报全片与512-patch版本区别 | 否 | 判断判别信息是否已在线性空间；不依赖复杂 gate |
| P0 | 原空间 attention + 随机子空间 | 1536维 value、256维 attention 子空间、21窗平均 logit；共同30epoch方案下先batch8，另测batch32 | 否 | 先区分保留原空间和随机子空间，再区分batch效果 |
| P0 | 同片双 bag 一致性 | 每视图512patch；两视图 BCE均值+0.1对称KL；需有两视图无KL对照 | 否 | 与matched attention比较稳定性；与双视图无KL比较正则贡献 |
| P1 | Attention top-K masking | 单分支ABMIL，K=10，p=.6，仅训练启用 | 否 | 相同架构/采样/训练预算下是否减少集中程度与过拟合 |
| P1 | 等权种子集成 | 固定全部3seed概率均值；不调外部权重 | 否 | 相对单模型平均及逐种子结果，避免仅与最差种子比较 |

可选下一轮：同类 bag mixing、完整 PseMix、class-conditioned prototype regularization。只有前五组显示出可重复方向时，再扩展完整方法或组合最有效的单项。

## 本轮设计时应固定的记录

1. 保持现有作者确认的数据任务与划分；本轮只研究算法。外部数据不参与梯度、标准化/PCA拟合、训练 bag 增强来源或新超参数选择。
2. 用共同 seeds 17、29、43；训练 bag、顺序和公共参数初始化尽量配对；有额外视图时记录具体采样种子。所有预定配置保留完整结果。
3. 固定内部选择规则，并在新增外部评估前冻结配置。因该外部队列此前已多次查看，新一轮表现描述为算法开发中的复测，不另称全新的盲法确认。
4. 主指标同时给内部/外部 AUROC、每种子结果和平均±SD；阈值0.5下给敏感度、特异度、balanced accuracy及混淆矩阵。阈值优化只可从内部数据产生，并与固定0.5分开。
5. 一致性和稳定性须同时报告重复bag指标与跨训练seed指标。高attention熵不是自动更好的定位；高不确定性也不自动等于校准良好的临床概率。
6. 统一记录参数数、优化步数、训练和推断耗时、显存峰值。30epoch batch32的优化步数少于batch8，须如实说明；需要进一步排除优化步数影响时再加等步数对照。

## 计算代价估计

已有日志中9次30epoch受控训练约185秒，仅作为本机缓存命中条件下的参照，不能视为所有新模型的保证。线性池化与已有三种子集成通常主要消耗CPU和HDF5读取；轻量MIL第一轮预计为分钟至几十分钟级，取决于训练运行数、重复评估和全片池化读取。双bag前向通常增加训练计算；21窗子空间集成主要增加推断时间。完整HistAug或重新提取其他编码器特征不在该估计内。最终以实测日志为准。

## 检索范围和限制

使用 `nature-academic-search` 的 multi-source-search 工作流。当前会话未挂载专用 PubMed/CrossRef/arXiv MCP，因此通过网页工具检索并核对期刊、会议原文、作者 arXiv 全文及官方仓库；未使用二手解读作为结论依据。本次为面向实施的定向检索，非系统综述，也未进行文献引用量排名。

部分 Nature 页面出现登录跳转，CVF PDF出现403，PMC偶发浏览器校验；相应方法细节改由作者 arXiv全文或官方实现核实。年份优先取正式期刊/会议记录，仓库功能和可用预训练模型按2026-11-04检索时状态记录。本文列出的起始超参数中，凡标注“本项目建议/改编”者均不代表原论文默认设置。
