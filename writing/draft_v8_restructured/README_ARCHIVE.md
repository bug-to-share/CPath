> Historical local-package README. File availability claims below apply to the original local archive, not this filtered public copy. See [README.md](README.md).

# MALT–SS英文论文 v8

2026-11-04。依据《研究现状与论文审阅报告》重新组织并撰写，面向病理学和口腔医学读者。当前为完整英文工作稿，医院尚未导入的事实和作者信息仍明确留空。

## 阅读入口

- [正文PDF](MALT_SS_manuscript_v8.pdf)：15页，4张主图、3张主表。
- [带行号的审阅PDF](MALT_SS_manuscript_v8_with_line_numbers.pdf)：用于逐段批注。
- [补充材料PDF](Supplementary_information_v8.pdf)：14页，完整数学定义、9张补充表、5张补充图。
- [LaTeX与图表源码包](MALT_SS_LaTeX_sources_v8.zip)。
- [相较v7的详细修改说明](REVISION_NOTES_zh.md)。
- [医院及作者待补资料](COMPLETION_CHECKLIST_zh.md)。
- [写作方案与术语表](WRITING_PLAN_zh.md)。
- [新增引用核验记录](LITERATURE_UPDATES_zh.md)。

## 研究主线

固定病理基础模型特征支持当前唾液腺MALT/SS材料中的切片分类；简单基线、受控重训练和重复抽样用于刻画这种分类信号。主模型保留既有checkpoint，新增34次训练和2个线性对照的全部相关结果进入正文及补充。没有以外部最高seed替换主结果，也没有把内部筛选候选写成已证实的改进。

## 编译与重绘

在本目录运行：

```bash
bash build.sh
```

需要TeX Live中的`latexmk`、`pdflatex`、BibTeX及`preamble.tex`所列常用宏包。脚本依次构建正文、补充和带行号正文，并复制为上述PDF。全部LaTeX输入均在源码包内；编译不需要原始WSI、GPU或模型权重。

定量图和表可从随包的冻结CSV/JSON重新生成：

```bash
python render_materials.py
python validate_materials.py
bash build.sh
```

重绘所用环境：Python 3.10、NumPy 1.26.4、pandas 2.3.2、scikit-learn 1.7.2、Matplotlib 3.10.6。本机对应解释器为`python`。`inspect_pdfs.py`另需PyMuPDF；本机默认`python`可执行。数值验证仅复算现有预测，不训练或推断模型。

`prepare_sources.py`只用于本次初始材料汇集，依赖原项目目录，**不要在已整理的v8目录再次运行**；它未放入独立源码包。原实验代码和checkpoint留在`../algorithm_improvement_20261104/`等项目档案中；本包是论文编译和图表复现包，不是完整训练发布包。

## 文件结构与验证

`main.tex`、`supplement.tex`、`review.tex`为入口；`sections/`为正文；`tables/`为生成表格；`figures/`含PDF、SVG及450dpi PNG；`source_data/`保留数值依据；`protocols/`保留既有实验协议。

`qa/numerical_validation.json`记录预测复算、汇总一致性和导入文件校验；`qa/pdf_validation.json`记录PDF、字体及编译检查；`qa/portable_build.json`记录独立解压编译；`qa/source_manifest.json`记录来源与SHA-256。原v7文件逐一校验保持不变。供视觉检查使用的页面渲染图仅保留在本地`qa/`，不进入源码包。

源码包未包含原始WSI、患者映射、伦理批件截图或模型权重。所附逐片预测为项目的publication-coded研究输出；正式共享范围仍由作者和数据提供方确定，本地生成不代表已经公开发布。
