# v8 — filtered public manuscript copy

本目录是经过筛选的公开副本，不是完整投稿稿。原始本地文件保持不变。

- [正文 PDF](MALT_SS_manuscript_v8.pdf)
- [带行号正文 PDF](MALT_SS_manuscript_v8_with_line_numbers.pdf)
- [补充材料 PDF](Supplementary_information_v8.pdf)
- [LaTeX 正文入口](main.tex) · [补充材料入口](supplement.tex)
- [本地版本的原始说明（历史记录）](README_ARCHIVE.md)

组织图像、逐片观测图和逐例表格已以明确占位说明替代；对应原图、逐片预测、患者映射、权重、私有确认表及压缩包均未上传。图表编号和原图注保留，汇总性能数值保持不变。正文中的 Source Data 指完整本地研究档案，公开数据范围以 [公开说明](../PUBLIC_EXPORT.md) 为准。

## 编译

安装 TeX Live、latexmk 和 BibTeX 后，在本目录运行 `bash build.sh`。编译只依赖本目录公开文件，不需要原始数据。重训练、重绘、逐片复算和原本地归档验证脚本仍需获授权的本地数据；它们不构成可公开复现实验数据包。
