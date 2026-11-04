"""Render the local review synthesis as an editable Word document.

Requires python-docx. PDF export is performed separately with LibreOffice.
"""
from pathlib import Path
import re
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor
from docx.opc.constants import RELATIONSHIP_TYPE as RT

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / '审稿意见综合总结与改进计划.md'
OUTPUT = ROOT / '审稿意见综合总结与改进计划.docx'
FONT = 'Noto Sans CJK SC'
doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21), Cm(29.7)
sec.top_margin, sec.bottom_margin = Cm(1.8), Cm(1.8)
sec.left_margin, sec.right_margin = Cm(1.9), Cm(1.9)
sec.header_distance, sec.footer_distance = Cm(.7), Cm(.7)


def font(style, size, color=None):
    style.font.name = FONT
    style.font.size = Pt(size)
    style.element.get_or_add_rPr().get_or_add_rFonts().set(qn('w:eastAsia'), FONT)
    if color:
        style.font.color.rgb = RGBColor.from_string(color)


font(doc.styles['Normal'], 10.5)
normal = doc.styles['Normal'].paragraph_format
normal.line_spacing = 1.18
normal.space_after = Pt(6)
normal.widow_control = True
for name, size in [('Title', 20), ('Heading 1', 15), ('Heading 2', 12)]:
    font(doc.styles[name], size, '173E57')
    doc.styles[name].paragraph_format.keep_with_next = True
    doc.styles[name].paragraph_format.space_before = Pt(10)
    doc.styles[name].paragraph_format.space_after = Pt(6)
for name in ['List Bullet', 'List Number']:
    font(doc.styles[name], 10.5)
    doc.styles[name].paragraph_format.space_after = Pt(5)

header = sec.header.paragraphs[0]
header.add_run('MALT SS · v7.5 审稿意见综合与改进计划')
header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
for run in header.runs:
    run.font.name, run.font.size = FONT, Pt(8)
    run.font.color.rgb = RGBColor.from_string('65717A')
footer = sec.footer.paragraphs[0]
footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
footer.add_run('第 ')
field = OxmlElement('w:fldSimple')
field.set(qn('w:instr'), 'PAGE')
footer._p.append(field)
footer.add_run(' 页')


def link(p, label, url):
    h = OxmlElement('w:hyperlink')
    h.set(qn('r:id'), p.part.relate_to(url, RT.HYPERLINK, is_external=True))
    r = OxmlElement('w:r')
    props = OxmlElement('w:rPr')
    color = OxmlElement('w:color'); color.set(qn('w:val'), '245B85')
    props.append(color); r.append(props)
    t = OxmlElement('w:t'); t.text = label; r.append(t)
    h.append(r); p._p.append(h)


def inline(p, text):
    pattern = r'(\*\*.*?\*\*|`[^`]+`|\[[^\]]+\]\([^)]+\))'
    for token in re.split(pattern, text):
        if not token:
            continue
        if token.startswith('[') and re.fullmatch(r'\[[^\]]+\]\([^)]+\)', token):
            m = re.match(r'\[([^\]]+)\]\(([^)]+)\)', token)
            link(p, m[1], m[2])
        else:
            bold = token.startswith('**') and token.endswith('**')
            code = token.startswith('`') and token.endswith('`')
            run = p.add_run(token[2:-2] if bold else token[1:-1] if code else token)
            run.bold = bold
            if code:
                run.font.size = Pt(8.5)


def table(lines):
    rows = [[c.strip() for c in line.strip().strip('|').split('|')] for line in lines]
    rows = [r for r in rows if not all(re.fullmatch(r':?-+:?', c) for c in r)]
    t = doc.add_table(rows=0, cols=len(rows[0]))
    t.style = 'Table Grid'
    t.autofit = False
    widths = [4.2, 5.5, 7.5]
    for col, width in zip(t.columns, widths):
        col.width = Cm(width)
    for n, cells in enumerate(rows):
        row = t.add_row()
        trpr = row._tr.get_or_add_trPr()
        trpr.append(OxmlElement('w:cantSplit'))
        if n == 0:
            trpr.append(OxmlElement('w:tblHeader'))
        for cell, content, width in zip(row.cells, cells, widths):
            cell.width = Cm(width)
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(4)
            p.paragraph_format.space_before = Pt(4)
            p.paragraph_format.line_spacing = 1.1
            inline(p, content)
            for run in p.runs:
                run.font.size = Pt(9)
                if n == 0:
                    run.bold = True
            if n == 0:
                shd = OxmlElement('w:shd'); shd.set(qn('w:fill'), 'E8EFF4')
                cell._tc.get_or_add_tcPr().append(shd)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


lines = SOURCE.read_text().splitlines()
i = 0
while i < len(lines):
    line = lines[i].strip()
    if not line:
        i += 1
        continue
    if line.startswith('|'):
        block = []
        while i < len(lines) and lines[i].strip().startswith('|'):
            block.append(lines[i]); i += 1
        table(block)
        continue
    if line.startswith('# '):
        inline(doc.add_paragraph(style='Title'), line[2:])
    elif line.startswith('## '):
        p = doc.add_paragraph(style='Heading 1')
        if line.startswith('## 二 '):
            p.paragraph_format.page_break_before = True
        inline(p, line[3:])
    elif line.startswith('### '):
        inline(doc.add_paragraph(style='Heading 2'), line[4:])
    elif line.startswith('- '):
        inline(doc.add_paragraph(style='List Bullet'), line[2:])
    elif re.match(r'^\d+\. ', line):
        # Preserve source numbering for each independent list.
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Cm(.3)
        p.paragraph_format.first_line_indent = Cm(-.3)
        inline(p, line)
    else:
        inline(doc.add_paragraph(), line)
    i += 1

doc.core_properties.title = 'MALT SS 论文 v7.5 审稿意见综合总结与改进计划'
doc.core_properties.subject = '两份审稿意见的综合判断、现有证据与下一步行动'
doc.core_properties.author = '研究团队工作文档'
doc.save(OUTPUT)
print(OUTPUT)
