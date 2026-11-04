"""Render the Chinese review plan to editable DOCX; no model or data inference."""
from pathlib import Path
import csv
import json
import re
from docx import Document
from docx.shared import Cm, Pt, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT = Path(__file__).resolve().parent
TEXT = ROOT / '问题汇总与优化方案.md'
FONT = 'Noto Sans CJK SC'


def font(run, size=None):
    run.font.name = FONT
    run._element.get_or_add_rPr().get_or_add_rFonts().set(qn('w:eastAsia'), FONT)
    if size:
        run.font.size = Pt(size)


def inline(p, text, size=None):
    for part in re.split(r'(\*\*.*?\*\*|`[^`]+`)', text):
        if not part:
            continue
        bold = part.startswith('**') and part.endswith('**')
        code = part.startswith('`') and part.endswith('`')
        value = part[2:-2] if bold else part[1:-1] if code else part
        run = p.add_run(value)
        font(run, size)
        run.bold = bold


def setup(doc):
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21), Cm(29.7)
    sec.top_margin, sec.bottom_margin = Cm(1.7), Cm(1.7)
    sec.left_margin = sec.right_margin = Cm(1.8)
    sec.header_distance = sec.footer_distance = Cm(.65)
    styles = [('Normal', 10), ('Title', 20), ('Heading 1', 14),
              ('Heading 2', 11.5), ('Heading 3', 10.5)]
    for name, size in styles:
        style = doc.styles[name]
        style.font.name = FONT
        style._element.get_or_add_rPr().get_or_add_rFonts().set(qn('w:eastAsia'), FONT)
        style.font.size = Pt(size)
        style.paragraph_format.line_spacing = 1.2
        style.paragraph_format.space_after = Pt(5)
        if name != 'Normal':
            style.font.color.rgb = RGBColor.from_string('243D50')
            style.paragraph_format.keep_with_next = True
            style.paragraph_format.space_before = Pt(10)
    header = sec.header.paragraphs[0]
    inline(header, 'MALT–SS论文｜审稿汇总与优化方案', 8)
    header.paragraph_format.space_after = Pt(0)
    footer = sec.footer.paragraphs[0]
    footer.alignment = 2
    inline(footer, '工作文件 · ', 8)
    field = OxmlElement('w:fldSimple')
    field.set(qn('w:instr'), 'PAGE')
    footer._p.append(field)
    doc.core_properties.title = 'MALT–SS论文：审稿问题汇总与优化方案'
    doc.core_properties.subject = 'v5审稿意见整合与修订计划'
    doc.core_properties.author = ''


def add_table(doc, rows):
    n = len(rows[0])
    table = doc.add_table(rows=0, cols=n)
    table.style = 'Table Grid'
    table.autofit = False
    if n == 3:
        if rows[0][0] == '外部意见':
            widths = [4.2, 10.2, 3.0]
        else:
            widths = [3.6, 7.5, 6.3]
    elif n == 4:
        widths = [3.1, 6.3, 3.1, 4.9]
    else:
        widths = [17.4 / n] * n
    for col, width in zip(table.columns, widths):
        col.width = Cm(width)
    for ri, row in enumerate(rows):
        cells = table.add_row().cells
        for cell, txt, width in zip(cells, row, widths):
            cell.width = Cm(width)
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(3)
            p.paragraph_format.space_after = Pt(3)
            p.paragraph_format.line_spacing = 1.1
            inline(p, txt, 8.5)
            if ri == 0:
                for run in p.runs:
                    run.bold = True
                shade = OxmlElement('w:shd')
                shade.set(qn('w:fill'), 'E8EFF4')
                cell._tc.get_or_add_tcPr().append(shade)
        pr = table.rows[-1]._tr.get_or_add_trPr()
        pr.append(OxmlElement('w:cantSplit'))
        if ri == 0:
            pr.append(OxmlElement('w:tblHeader'))
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def main():
    text = TEXT.read_text()
    doc = Document()
    setup(doc)
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line.startswith('|'):
            rows = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                cells = [x.strip() for x in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r':?-+:?', x) for x in cells):
                    rows.append(cells)
                i += 1
            assert all(len(r) == len(rows[0]) for r in rows)
            add_table(doc, rows)
            continue
        if line.startswith('# '):
            p = doc.add_paragraph(style='Title')
            inline(p, line[2:])
        elif line.startswith('## '):
            p = doc.add_paragraph(style='Heading 1')
            inline(p, line[3:])
        elif line.startswith('### '):
            p = doc.add_paragraph(style='Heading 2')
            inline(p, line[4:])
        elif line.startswith('- '):
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Cm(.3)
            p.paragraph_format.first_line_indent = Cm(-.3)
            inline(p, '• ' + line[2:])
        else:
            p = doc.add_paragraph()
            inline(p, line)
        i += 1
    output = ROOT / '问题汇总与优化方案.docx'
    doc.save(output)
    issues = []
    for match in re.finditer(r'^### (U\d{2})｜(.+?)【(.+?)】\n(.*?)(?=^### |^## |\Z)', text, re.M | re.S):
        uid, title, priority, body = match.groups()
        parts = {}
        for name in ['问题与影响', '优化', '资源与完成标准']:
            found = re.search(r'\*\*' + name + r'：\*\*\s*(.*?)(?=\n\n|\Z)', body, re.S)
            assert found, (uid, name)
            parts[name] = found.group(1).strip()
        issues.append({'编号': uid, '问题': title, '优先级与条件': priority,
                       **parts, '状态': '优化方案待执行；已有修正及医院待录入状态见正文'})
    assert len(issues) == 20
    assert [x['编号'] for x in issues] == [f'U{i:02d}' for i in range(1, 21)]
    with (ROOT / '问题台账.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(issues[0]))
        writer.writeheader()
        writer.writerows(issues)
    (ROOT / 'document_build.json').write_text(json.dumps({
        'source': TEXT.name, 'docx': output.name, 'issues': 20,
        'tables': len(doc.tables), 'paragraphs': len(doc.paragraphs),
        'source_characters': len(text), 'font': FONT,
        'new_training': 0, 'new_model_inference': 0,
    }, ensure_ascii=False, indent=2) + '\n')
    print(output)
    print(f'20 issues; {len(doc.tables)} tables; {len(text)} source characters')


if __name__ == '__main__':
    main()
