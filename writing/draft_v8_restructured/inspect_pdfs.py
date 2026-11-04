"""Check compiled PDF text/layout and build diagnostics; render contact sheets."""
from pathlib import Path
import hashlib
import json
import re
import fitz
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
QA = ROOT / 'qa'
QA.mkdir(exist_ok=True)
report = {'status': 'PASS', 'documents': {}, 'main_figures': {}, 'logs': {}}
documents = {'main': 'MALT_SS_manuscript_v8.pdf',
             'supplement': 'Supplementary_information_v8.pdf',
             'review': 'MALT_SS_manuscript_v8_with_line_numbers.pdf'}
for key, filename in documents.items():
    doc = fitz.open(ROOT / filename)
    text = '\n'.join(p.get_text() for p in doc)
    bad = []
    for i, page in enumerate(doc):
        for block in page.get_text('dict')['blocks']:
            for line in block.get('lines', []):
                for span in line.get('spans', []):
                    if not span['text'].strip():
                        continue
                    x0,y0,x1,y1 = span['bbox']
                    if x0 < -1 or y0 < -1 or x1 > page.rect.width+1 or y1 > page.rect.height+1:
                        bad.append({'page': i+1, 'text': span['text'], 'bbox': span['bbox']})
    assert not bad, bad
    assert '\ufffd' not in text, f'replacement character in {key}'
    assert not re.search(r'\bSP\b|non[-–]MALT', text), f'obsolete class in {key}'
    compiled = ROOT / 'build' / f'{key}.pdf'
    identical = hashlib.sha256((ROOT/filename).read_bytes()).digest() == hashlib.sha256(compiled.read_bytes()).digest()
    assert identical, f'outdated delivery PDF {key}'
    report['documents'][key] = {'pages': len(doc), 'text_characters': len(text),
                                'off_page_text_spans': bad, 'matches_build': identical}
    if key != 'review':
        thumbs = []
        for i, page in enumerate(doc):
            pix = page.get_pixmap(matrix=fitz.Matrix(.54,.54), alpha=False)
            img = Image.frombytes('RGB', [pix.width,pix.height], pix.samples)
            card = Image.new('RGB', (img.width+20,img.height+34), '#e8e8e8')
            card.paste(img,(10,24));ImageDraw.Draw(card).text((10,6),f'{key} page {i+1}',fill='black')
            thumbs.append(card)
        cols=4; rows=(len(thumbs)+cols-1)//cols
        sheet=Image.new('RGB',(cols*thumbs[0].width,rows*thumbs[0].height),'#d8d8d8')
        for i,img in enumerate(thumbs):sheet.paste(img,((i%cols)*img.width,(i//cols)*img.height))
        sheet.save(QA/f'{key}_contact_sheet.png')
        for i in ([0,3,6,9] if key=='main' else [6,7,8]):
            if i<len(doc):doc[i].get_pixmap(matrix=fitz.Matrix(1.4,1.4),alpha=False).save(QA/f'{key}_page_{i+1}.png')

for pdf in sorted((ROOT/'figures').glob('Figure_[1-4]_*.pdf')):
    doc=fitz.open(pdf); sizes=[];bounds=[]
    for page in doc:
        for block in page.get_text('dict')['blocks']:
            for line in block.get('lines',[]):
                for span in line.get('spans',[]):
                    if span['text'].strip():
                        sizes.append(span['size'])
                        r=fitz.Rect(span['bbox'])
                        assert page.rect.contains(r), (pdf.name,span['text'])
        assert all('Type3' not in str(font) for font in page.get_fonts(full=True)), pdf.name
    assert min(sizes)>=8.0,(pdf.name,min(sizes))
    report['main_figures'][pdf.name]={'min_text_size_pt':min(sizes),'type3_fonts':False,'text_within_page':True}

for key in documents:
    log=(ROOT/'build'/f'{key}.log').read_text(errors='replace')
    issues=[line for line in log.splitlines() if re.search(r'Overfull|Underfull|undefined|LaTeX Warning|Package .+ Warning|^!',line)]
    assert not issues,(key,issues)
    report['logs'][key]={'warnings_or_errors':issues}
(QA/'pdf_validation.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
