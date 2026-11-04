"""Generate the Chinese study report from verified numerical summaries."""
from pathlib import Path
import json
import pandas as pd

R=Path(__file__).resolve().parent
S=json.loads((R/'analysis/summary.json').read_text())
run=pd.read_csv(R/'outputs/run_metrics.csv')
screen=pd.read_csv(R/'analysis/screen_summary.csv')
confirm=pd.read_csv(R/'analysis/confirmation_summary.csv')
ens=pd.read_csv(R/'analysis/ensemble_metrics.csv')
linear=pd.read_csv(R/'analysis/linear_controls.csv')
pairs=pd.read_csv(R/'analysis/paired_auc_comparisons.csv')
winner=S['selected_variant'];assert winner=='attention_l2'

def link(name,path):return f'[{name}]({R/path})'
def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+
                     ['| '+' | '.join(map(str,row))+' |' for row in rows])
def one(df,**filters):
    for k,v in filters.items():df=df[df[k].eq(v)]
    assert len(df)==1,(filters,len(df))
    return df.iloc[0]
def ms(row):return f'{row.auroc_mean:.4f} ± {row.auroc_sd:.4f}'

names={
'hybrid_control':'原 Hybrid 受控重训练', 'attention_control':'匹配的 attention 对照',
'hybrid_l2':'Hybrid＋输入 L2', 'attention_l2':'Attention＋输入 L2（内部选定）',
'attention_mask':'Attention＋随机遮挡高权重实例', 'direct_attention':'原始特征空间 attention',
'subspace_attention':'随机特征子空间 attention', 'subspace_largebatch':'子空间＋batch32/lr3e-4',
'hybrid_twobag':'Hybrid＋双 bag 分类损失', 'hybrid_consistency':'Hybrid＋双 bag 一致性'}
screen_rows=[]
for v,name in names.items():
    g=run[(run.variant==v)&(run.stage=='screen')&(run.scope=='internal')]
    screen_rows.append([name,ms(one(screen,variant=v,scope='internal')),ms(one(screen,variant=v,scope='external')),
                        f'{int(g.parameters.iloc[0]):,}',f'{g.wall_seconds_training_including_eval.mean():.1f} s'])
check_rows=[]
for tag,label in [('screen_three','初筛3种子：17/29/43'),('additional_two','新增2种子：59/71'),('all_five','全部5种子')]:
    for scope,zh in [('internal','内部'),('external','外部')]:
        a=one(confirm,variant=winner,scope=scope,seed_group=tag);b=one(confirm,variant='hybrid_control',scope=scope,seed_group=tag)
        check_rows.append([label,zh,ms(a),ms(b),f'{a.auroc_mean-b.auroc_mean:+.4f}'])
ensemble_rows=[]
for mode,label in [('three_seed_reference','3种子概率平均，固定参考bag'),('five_seed_reference','5种子概率平均，固定参考bag'),
                   ('five_seed_five_bag','5种子×5个新bag概率平均')]:
    for scope,zh in [('internal','内部'),('external','外部')]:
        a=one(ens,variant=winner,scope=scope,mode=mode);b=one(ens,variant='hybrid_control',scope=scope,mode=mode)
        ensemble_rows.append([label,zh,f'{a.auroc:.4f}',f'{b.auroc:.4f}'])
op_rows=[]
for v,name in [(winner,'内部选定的 Attention＋L2'),('hybrid_control','原 Hybrid 受控重训练')]:
    e=one(ens,variant=v,scope='external',mode='five_seed_five_bag')
    op_rows.append([name,f'{e.auroc:.4f}',f'{e.sensitivity*100:.2f}%',f'{e.specificity*100:.2f}%',
                   f'{e.balanced_accuracy*100:.2f}%',f'{e.tp} / {e.fn} / {e.tn} / {e.fp}'])
lin_rows=[]
for v,label in [('mean_logistic','原特征均值＋L2正则 logistic'),('l2mean_logistic','逐tile归一化均值＋L2正则 logistic')]:
    for inference,tag in [('reference_bag','固定512-row参考bag'),('full_feature_mean','全部可用特征均值')]:
        a=one(linear,variant=v,scope='internal',inference=inference);b=one(linear,variant=v,scope='external',inference=inference)
        lin_rows.append([label,tag,f'{a.auroc:.4f}',f'{b.auroc:.4f}',f'{100*b.sensitivity:.2f}%',f'{100*b.specificity:.2f}%'])
paired=one(pairs,scope='external',candidate_mode='five_seed_five_bag',comparator='hybrid_control')
hist=one(pairs,scope='external',candidate_mode='five_seed_five_bag',comparator='historical_primary_hybrid')
ratio=1-run[run.variant==winner].parameters.iloc[0]/run[run.variant=='hybrid_control'].parameters.iloc[0]
runtime=S['runtime']
text=f'''# 算法改进调研与实测报告

日期：2026-11-04（北京时间）。任务：在现有 MALT–SS 项目、UNI2-h 特征和数据范围内，调研、设计并执行算法改进。

## 本轮结论

**完成了34次神经网络训练、2个确定性线性对照及全部预定评价；尚未找到可以稳定替换现有主模型的性能改进。**

内部三种子初筛选出的方案是 **Attention＋输入L2归一化**。其内部平均 AUROC 为 **0.9365**，原 Hybrid 对照为 **0.9353**，只增加 **0.0011**；预先设定的工程目标是增加至少0.01且至少2/3种子不降低。虽然初筛3个配对种子均未降低，增幅仍不达目标。新增两个种子的差异方向不一致，全部5种子外部平均 AUROC 为 **0.8061 vs 0.8056**，两者均值接近。

这个候选的明确价值是模型简化：参数从 **1,521,672** 降到 **658,178**，减少 **{ratio*100:.2f}%**。这涉及采用更简单的聚合结构及输入归一化，不能全部归因于L2操作。现有证据支持保留原论文主模型，并把本轮结果作为算法开发和简化模型的依据。

## 调研依据与实际测试范围

调研了8篇原始研究及官方代码，完整检索说明见 {link('文献调研','LITERATURE_REVIEW_zh.md')}。本轮优先实现可复用现有embedding的方向：

- **特征子空间与原空间聚合**：来自 [nnMIL，Nature Biomedical Engineering 2026](https://www.nature.com/articles/s41551-026-01767-8) 的方法思路；本地版本保留统一30轮训练，未复现原论文完整训练配置。
- **训练时遮挡少数高attention实例**：参考 [ACMIL，ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06954.pdf)，本轮只测试STKIM式操作，没有完整多分支ACMIL。
- **双份输入的一致性损失**：参考 [R-Drop，NeurIPS 2021](https://proceedings.neurips.cc/paper_files/paper/2021/hash/5a66b9200f29ac3fa0ae244cc2a51b39-Abstract.html)，本轮同时改变patch采样，因此是bag一致性的改编，并配有“双bag、无一致性项”的对照。
- **模型简化和确定性线性探针**：检查已有固定特征能否由简单分类器利用。
- **固定种子与多bag平均**：完整保留所有种子，区分训练波动、采样波动和预测集成。

未在本轮执行完整PseMix/C²Aug、带类别约束的原型辅助损失、HistAug迁移或编码器微调。它们在文献调研中列为后续方向，不能计入本轮已完成实验。

## 实验怎样控制

1. 沿用作者核查过的144张训练、49张内部评价、168张历史外部评价切片和现有1536维UNI2-h特征。
2. 10种配置各跑17/29/43三个种子，统一30轮，保存逐epoch指标、最佳和末轮权重。除明确标出的变体，使用batch8、AdamW 5e-5、weight decay 1e-4及正类BCE权重77/67。
3. 原Hybrid和attention控制组逐一复现历史训练bag、顺序、初始化和所选epoch，确认比较起点一致。
4. 只按内部3种子平均AUROC选择8个改进候选之一，写入选择锁；随后候选与Hybrid各补59/71两个种子。选择锁没有依据新增种子或外部结果更换。
5. 所有模型冻结后统一计算外部输出。外部队列此前已被观察，因此此处是开发阶段的历史外部评价，不能称为全新确认性验证。
6. 所有固定阈值指标均用0.5；没有事后调阈值、概率校准或剔除不利运行。原始论文的历史单checkpoint结果另列。

正式方案与精确公式见 {link('PROTOCOL.md','PROTOCOL.md')}；模型和执行代码分别为 {link('models.py','models.py')} 与 {link('runner.py','runner.py')}。

## 全部神经网络初筛结果

下表为**三个独立训练模型的AUROC均值±样本标准差**，不是将预测平均后的ensemble AUROC，也不是临床置信区间。

{table(['配置','内部AUROC','外部AUROC','参数量','单run平均耗时'],screen_rows)}

耗时含训练期内部评价，来自本次共享GPU环境，不是独占硬件的严格速度基准。子空间模型每次评价包含21个特征窗口的logit平均；跨种子/跨bag则使用概率平均。这两层汇总方式有明确区分。

{link('全部逐种子指标','analysis/screen_per_seed.csv')} · {link('汇总表','analysis/screen_summary.csv')} · {link('全部配置对比图','figures/Figure_A1_all_configurations.pdf')}

## 新增种子是否支持初筛优势

{table(['种子组','队列','Attention＋L2','Hybrid对照','平均AUROC差值'],check_rows)}

新增种子59在内部评价中候选低于Hybrid，种子71则高于Hybrid；外部差异方向同样不一致。全部5种子的平均表现没有形成可靠的性能优势。这里的新增种子检验训练稳定性，复用相同临床样本，不能增加临床样本量。

## 集成与固定阈值表现

{table(['推断方式','队列','Attention＋L2的AUROC','Hybrid对照的AUROC'],ensemble_rows)}

5种子×5新bag的外部工作点如下，全部来自各自对应的完整概率集成：

{table(['模型','外部AUROC','敏感度','特异度','平衡准确率','TP / FN / TN / FP'],op_rows)}

候选减Hybrid的外部AUROC差值为 **{paired.difference:+.4f}**，条件配对95%区间为 **[{paired.conditional_paired_ci_low:+.4f}, {paired.conditional_paired_ci_high:+.4f}]**。区间包含零，不支持候选优于Hybrid。

原论文历史主模型的内部/外部AUROC是 **0.9599 / 0.8721**，它是单个历史checkpoint，与本轮重训练均值和ensemble属于不同结果。候选5种子×5bag集成相对该历史模型的外部差值为 **{hist.difference:+.4f}**，条件配对95%区间 **[{hist.conditional_paired_ci_low:+.4f}, {hist.conditional_paired_ci_high:+.4f}]**。本轮没有获得替换历史主结果的依据。

上述区间使用10,000次类别分层配对WSI bootstrap，只描述固定预测的条件性不确定性，不校正内部方法/epoch选择或此前对外部队列的观察。敏感度、特异度的Wilson区间保存在完整指标CSV中。

{link('训练稳定性与集成效果图','figures/Figure_A2_stability_and_aggregation.pdf')} · {link('固定ensemble的ROC与混淆矩阵','figures/Figure_A3_locked_ensemble_ROC_confusion.pdf')} · {link('完整配对区间','analysis/paired_auc_comparisons.csv')}

## 简单线性对照提供的线索

{table(['线性对照','推断特征','内部AUROC','外部AUROC','外部敏感度','外部特异度'],lin_rows)}

两个模型都仅在144张训练切片的全部特征均值上拟合StandardScaler与L2正则logistic（C=1），没有调参；固定512-row推断与全特征均值推断单列。线性模型只有1,537个可训练分类系数，外部AUROC约0.83–0.85，说明简单聚合是有竞争力的开发基线，但其固定阈值敏感度仍较低。这不能解释成临床诊断能力已达标。

## 对不同改法的判断

- **输入归一化不是统一有效的改进**：L2使attention初筛外部均值从0.7270变为0.7726，但使Hybrid从0.7633变为0.7166；需要按结构单独判断。当前L2将每tile范数固定为sqrt(1536)，同时改变了幅度，未保持抽查输入约16–17的范数。
- **双bag一致性应与双bag对照比较**：外部均值0.7307高于仅双bag的0.6886，但二者都低于原Hybrid的0.7633；不能只挑有利的局部比较称为整体提升。
- **子空间改编降低了本轮外部种子离散程度，但损失了内部辨别力**：内部AUROC约0.87–0.88，外部约0.77–0.79。这个结果只适用于本地实现和30轮配方，不能据此否定原nnMIL。
- **原空间attention直接替换效果较差**：内部均值0.9041，外部0.6051；本轮不适合直接替换原聚合器。
- **集成没有普遍提高性能**：必须看具体组合及固定阈值的敏感度/特异度，不能把平均单模型AUROC当作ensemble AUROC。

## 下一轮最值得做的改进（尚未执行）

| 优先级 | 具体实验 | 本轮结果给出的理由 | 数据与算力 |
|---|---|---|---|
| 1 | 保留Hybrid主干，增加零初始化的原始embedding均值残差分支；与原Hybrid单因素配对 | 简单均值线性模型提供可用信号，但直接替换原结构损失内部性能，值得检验互补性 | 现有特征；无需新病历或提取特征；按本次吞吐，3种子训练约1–3分钟，开发/核验另计 |
| 2 | 为两组潜在原型增加bag级类别辅助损失，权重预先固定，分别比较有/无该损失 | 原型两库目前无直接类别监督；可测试训练信号是否更明确 | 现有slide标签即可；不把bag标签强行作为所有patch的标签；预计数分钟级训练 |
| 3 | 在训练集内部选择正则强度或缩小投影维度，并独立对比EMA/权重平均 | 当前简单模型有竞争力，外部波动不只来自gate | 现有特征；少量CPU/GPU；保持内部选择与外部探索性解释 |
| 4 | 完整nnMIL训练配方或PseMix/C²Aug增强 | 当前仅为适配版本，完整训练日程/伪bag构造可能改变结果 | 不需新临床数据，但需更多实现核验；计算约数十分钟内的量级估计，取决于组合数 |

以上是下一轮假设，不是已经证实有效的改法。比较前仍应固定内部选择规则和全部候选，并保留所有结果。若目标是投稿中可信的泛化结论，还需要不参与开发选择的数据评价；本轮34次训练不能替代该证据。

## 计算、文件与核验

- 已执行：34次神经网络训练（共1,020个epoch）＋2个确定性线性模型；8,246条主评价预测；10,850条额外bag预测；全部最佳/末轮权重与采样索引保留。
- 物理GPU：3；主流程耗时 **{runtime['wall_seconds']:.1f}秒（{runtime['wall_seconds']/60:.2f}分钟）**，包括读取、训练及推断；另有14.5秒测速。峰值PyTorch分配显存 **{runtime['peak_gpu_allocated_mib']/1024:.2f}GiB**。文献、实现及CPU审计耗时另计。
- 逐epoch指标、最佳和末轮checkpoint在 `outputs/<variant>_seed<seed>/`；候选权重路径为 `outputs/attention_l2_seed<seed>/best_model.pth`，其中 `<seed>` 分别为17、29、43、59、71。
- 对外部集未执行额外筛选；医院字段未填写；原v7正文、图表、预测和checkpoint保留。
- {link('CPU模型核验','qa/model_cpu_checks.json')}、{link('训练前独立审计','qa/preflight_audit.md')}、{link('后处理复算','analysis/postprocessing_validation.json')}及{link('完成后的独立审计','qa/results_independent_audit.md')}全部通过。独立检查覆盖34个checkpoint、204个CPU复现预测、全部主预测与集成、完整训练bag索引及bootstrap区间；CPU/GPU预测最大差1.79×10⁻⁷。
- {link('全部统计JSON','analysis/summary.json')} 可追溯每项结果；图件均提供PDF、SVG和450dpi PNG。
'''
(R/'RESULTS_REPORT_zh.md').write_text(text)
print('Wrote RESULTS_REPORT_zh.md from saved results.')
