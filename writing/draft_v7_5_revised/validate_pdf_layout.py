"""Check compiled PDF text, counters and page bounds with PyMuPDF."""
from pathlib import Path
import fitz,json,re
R=Path(__file__).resolve().parent
Q=R/'qa'
pdfs={}
for name in ['main','supplement','main_review']:
    log=(R/'build'/f'{name}.log').read_text()
    bad=re.findall(r'^.*(?:Warning|Overfull|Underfull|undefined|^!).*$',log,re.M)
    assert not bad,(name,bad)
    doc=fitz.open(R/'build'/f'{name}.pdf');out=[]
    for n,page in enumerate(doc,1):
        for word in page.get_text('words'):
            x0,y0,x1,y1,*_=word
            if x0 < -1 or y0 < -1 or x1>page.rect.width+1 or y1>page.rect.height+1:out.append(n)
    assert not out,(name,out)
    text='\n\n'.join(p.get_text() for p in doc)
    (Q/f'{name}_text.txt').write_text(text)
    if name=='supplement':
        assert all(re.search(r'Table\s+S'+str(i)+r'\b',text) for i in range(1,20))
        assert all(re.search(r'Figure\s+S'+str(i)+r'\b',text) for i in range(1,5))
        assert 'SM10' in text
    pdfs[name]={'pages':len(doc),'no_build_warnings':True,'no_text_outside_page':True}
    if name in {'main','main_review'}:
        discussion=text.index('\nDiscussion\n')
        placements={}
        for kind,count in [('Figure',6),('Table',4)]:
            assert len(re.findall(rf'(?m)^{kind}\s+\d+:',text))==count,(name,kind,'unexpected caption count')
            locations=[]
            for i in range(1,count+1):
                pattern=rf'(?m)^{kind}\s+{i}:'
                matches=list(re.finditer(pattern,text))
                assert len(matches)==1,(name,kind,i,len(matches))
                assert matches[0].start()<discussion,(name,kind,i,'outside Results')
                locations.append(matches[0].start())
                placements[f'{kind} {i}']=next(n for n,p in enumerate(doc,1) if re.search(pattern,p.get_text()))
            assert locations==sorted(locations),(name,kind,'number order')
        pdfs[name]['inline_float_pages']=placements
        pdfs[name]['all_main_floats_within_results']=True
print(json.dumps(pdfs))
