"""Compile all three entry points from a clean unpacked source archive."""
from pathlib import Path
import hashlib,json,re,subprocess,tempfile,zipfile
import fitz
R=Path(__file__).resolve().parent
archive=R/'MALT_SS_LaTeX_sources_v7_5.zip'
report={'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest()}
with tempfile.TemporaryDirectory(prefix='trident_v75_compile_') as tmp:
    root=Path(tmp)
    with zipfile.ZipFile(archive) as z:
        names=z.namelist()
        assert not any(Path(n).is_absolute() or '..' in Path(n).parts for n in names)
        assert not any('PRIVATE' in n.upper() or Path(n).suffix in {'.pth','.pt','.bin','.h5','.ndpi','.svs','.joblib','.pkl','.npy'} for n in names)
        assert not any(n.startswith('build/') for n in names)
        z.extractall(root)
    with (R/'qa/clean_build_console.log').open('w') as out:
        proc=subprocess.run(['bash','build.sh'],cwd=root,stdout=out,stderr=subprocess.STDOUT,timeout=120)
    assert proc.returncode==0,proc.returncode
    results={}
    for name in ['main','main_review','supplement']:
        log=(root/'build'/f'{name}.log').read_text()
        bad=re.findall(r'^.*(?:Warning|Overfull|Underfull|undefined|^!).*$',log,re.M)
        assert not bad,(name,bad)
        a,b=fitz.open(R/'build'/f'{name}.pdf'),fitz.open(root/'build'/f'{name}.pdf')
        assert len(a)==len(b)
        assert [p.get_text() for p in a]==[p.get_text() for p in b],name
        results[name]={'pages':len(b),'extracted_text_matches':True,'warnings':0}
        if name in {'main','main_review'}:
            panel=fitz.Pixmap(str(R/'figures/Figure_1b_model_ai.png'))
            figure_page=next(p for p in b if 'Figure 1:' in p.get_text())
            assert sum(i['digest']==panel.digest for i in figure_page.get_image_info(hashes=True))==1
            results[name]['figure_1b_pixels_match_selected_candidate']=True
    report.update(status='PASS',archive_members=len(names),entry_points=results)
(R/'qa/clean_build_check.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
