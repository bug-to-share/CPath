> Historical local-package README. File availability claims below apply to the original local archive, not this filtered public copy. See [README.md](README.md).

# MALT–SS manuscript v7.5

本版从 v7 单独修订，保留原表数值与历史主模型结果；Figure 1b 已按作者要求换为新的模型结构示意图。正文聚焦按内部验证 AUROC 选出的最佳 checkpoint 及其内外部表现；性能表以加粗和下划线标出第一、第二名。

- 正文 PDF（完整文件保留于本地档案）
- 带行号正文（完整文件保留于本地档案）
- 补充材料 PDF（完整文件保留于本地档案）
- LaTeX 源码与数据包（完整文件保留于本地档案）
- [中文修改说明](REVISION_NOTES_zh.md)
- [医院/作者待补资料](MISSING_INFORMATION.md)

正文及带行号版均为 18 页，补充材料为 30 页。正文保留 6 张图、4 张表，全部嵌入 Results；恢复原 ROC/PR 图与主表 2–4。主表 2、4 及可比的补充性能表按每个指标的最佳/次佳显示值加粗/加下划线，并列采用相同标记；置信区间、人数、阈值及不同人群的单模型结果不作排名。排名仅描述数值次序。

摘要、正文结果和讨论以主要模型及集成表现为主，不再强调 seed，也不设置训练稳定性主图。完整重复训练、候选筛选和线性对照仍在补充材料与 Source Data 中。主模型仍是内部验证 AUROC 选定的 epoch 15，内外部结果保持同一 checkpoint 对应关系。此次没有训练、推理或改选模型。

Figure 1b 保留已确认的 AI 辅助示意图；原矢量 a 面板与组织图均保留。30 个原始图文件、原始数据和已完成实验输出未更改。上一轮新增的五张定量图保留为可选源资产，本次不编入主文或补充 PDF；上一轮布局与说明属于历史记录。当前版以 `RANKED_TABLES_NOTES_zh.md` 和 `qa/ranked_tables_check.json` 为准。

编译（解压后在本目录执行）：

```bash
bash build.sh
```

需要 pdfLaTeX、BibTeX、latexmk 及常用 TeX Live 宏包；不需访问训练数据、GPU 或网络。三个入口是 `main.tex`、`main_review.tex`、`supplement.tex`。

表格再生成与核对（生成后先编译再检查）：

```bash
python build_development_tables.py
python format_performance_tables.py
bash build.sh
python validate_v7_5.py
python validate_ranked_tables.py
python render_qa.py
python package_v7_5.py
python check_clean_build.py
```

数值核对需要 NumPy/pandas/scikit-learn，PDF 检查需要 PyMuPDF/Pillow；依赖分别列于 `requirements_render.txt` 与 `requirements_qa.txt`。本机数值核对使用 `python`，PDF 检查使用 PATH 中的 `python`；可通过 `PDF_QA_PYTHON` 指定后者。在所有依赖安装到同一环境时，可直接使用上述命令。

当前布局与数值检查见 `qa/ranked_tables_check.json` 和 `qa/validation_v7_5.json`。`inline_layout_check.json`、`figure1b_replacement_check.json` 和 `table_visualization_check.json` 分别记录先前修订阶段；对应的历史验证脚本不用于当前布局。

`development_analyses/` 纳入已经完成的 34 次神经训练与 2 个线性对照的协议、代码、逐例预测、所有候选/种子结果和次级分析。此次修稿没有启动新训练。`Source_Data_draft.zip` 和 `Source_Code_draft.zip` 分别提供数值和实现档案。论文源码包可独立编译；完整训练仍依赖受控的特征文件、权重、输入映射和原工程，不等同于已验证的便携训练发行版。

临床表单仅保存在本地 `confirmation_forms_PRIVATE/`，不进入源码包。原始 WSI、患者映射、伦理原件、模型权重和特征矩阵均不打包。资料缺项保留为棕色标记，公开存储库与共享安排尚未完成。
