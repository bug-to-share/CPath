# v8新增引用与使用范围

核对日期：2026-11-04。根据用户指定审阅报告补充，与论文中的具体问题对应；不以文献的性能数字替代本研究证据。

| BibTeX键 | 已核对的一手来源 | 文稿中的用途与边界 |
|---|---|---|
| focus2025 | [PubMed 40262321](https://pubmed.ncbi.nlm.nih.gov/40262321/)；J Autoimmun 153:103418，DOI 10.1016/j.jaut.2025.103418 | SS小唾液腺focus score研究，终点与MALT/SS分类不同 |
| lymphoma2025 | [PubMed 40365164](https://pubmed.ncbi.nlm.nih.gov/40365164/)；PNAS Nexus 4(5):pgaf137，DOI 10.1093/pnasnexus/pgaf137 | MALT/DLBCL分类近邻研究；病理选择的图像不能当作与本项目同设定的全切片基线 |
| nnmil2026 | [期刊条目](https://www.nature.com/articles/s41551-026-01767-8)、[作者全文](https://arxiv.org/html/2511.14907v2)；DOI 10.1038/s41551-026-01767-8 | 支持讨论训练/特征抽样；本地subspace实验是受启发的改编，未称完整复现。期刊条目元数据与作者稿交叉核对，期刊全文访问受限 |
| uni2card | [MahmoodLab/UNI2-h官方模型卡](https://huggingface.co/MahmoodLab/UNI2-h) | 具体编码器来源；与原始UNI家族论文分别引用，实际使用权重revision仍待T01确认 |
| acmil2024 | [ECCV原论文](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06954.pdf) | attention masking的来源；只实现组件不等于完整ACMIL |
| rrt2024 | [作者论文](https://arxiv.org/abs/2402.17228)；CVPR 2024 | 解释历史本地grouped-region transformer与正式R²T重嵌入架构的区别 |
| rdrop2021 | [NeurIPS论文](https://proceedings.neurips.cc/paper_files/paper/2021/hash/5a66b9200f29ac3fa0ae244cc2a51b39-Abstract.html) | 两路一致性约束的来源；本地实验同时改变patch与dropout，不写成原样R-Drop |

原v7的22项BibTeX条目保留，新添7项。正文按实际引用排版，正文与补充的引用序号分别生成，未引用的继承条目不强行放入参考文献列表。未加入“首个同任务研究”“优于所有最新方法”等需要更强检索及实验依据的说法。
