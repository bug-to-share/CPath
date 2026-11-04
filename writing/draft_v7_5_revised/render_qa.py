"""Render final manuscript pages and compact contact sheets; preserve original figures."""
from pathlib import Path
import fitz
from PIL import Image,ImageOps,ImageDraw
R=Path(__file__).resolve().parent
OUT=R/'qa/rendered';OUT.mkdir(exist_ok=True)
for name in ['main','supplement','main_review']:
    doc=fitz.open(R/'build'/f'{name}.pdf');thumbs=[]
    for old in OUT.glob(f'{name}_*.png'):
        tail=old.stem.removeprefix(name+'_')
        if tail.isdigit() and int(tail)>len(doc):old.unlink()
    for i,page in enumerate(doc,1):
        pix=page.get_pixmap(matrix=fitz.Matrix(1,1),alpha=False)
        im=Image.frombytes('RGB',[pix.width,pix.height],pix.samples)
        if name!='main_review' or i==1:im.save(OUT/f'{name}_{i:02d}.png')
        im.thumbnail((310,438))
        im=ImageOps.expand(im,border=(3,22,3,3),fill='white')
        ImageDraw.Draw(im).text((8,4),f'{name} {i}',fill='black');thumbs.append(im)
    for start in range(0,len(thumbs),12):
        group=thumbs[start:start+12];w=max(x.width for x in group);h=max(x.height for x in group)
        canvas=Image.new('RGB',(w*4,h*3),'#cccccc')
        for j,im in enumerate(group):canvas.paste(im,((j%4)*w,(j//4)*h))
        canvas.save(OUT/f'{name}_contact_{start+1:02d}.png')
    print(f'{name}: {len(doc)} pages rendered')
