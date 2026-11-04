"""Package self-contained manuscript/figure sources; verify an extracted build."""
from pathlib import Path
import hashlib
import json
import re
import subprocess
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parent
ZIP = ROOT / 'MALT_SS_LaTeX_sources_v8.zip'
QA = ROOT / 'qa'
files = []
for pattern in ('*.tex', '*.bib', '*.md', 'build.sh', 'render_materials.py',
                'validate_materials.py', 'inspect_pdfs.py', 'package_sources.py'):
    files.extend(ROOT.glob(pattern))
for directory in ('sections', 'tables', 'figures', 'source_data', 'protocols'):
    files.extend(p for p in (ROOT/directory).rglob('*') if p.is_file())
for name in ('source_manifest.json', 'numerical_validation.json', 'pdf_validation.json',
             'figure_contract.json', 'table_generation.json'):
    files.append(QA/name)
files = sorted(set(files))
manifest = [{'path':str(p.relative_to(ROOT)), 'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
             'bytes':p.stat().st_size} for p in files]
(QA/'package_manifest.json').write_text(json.dumps(manifest,indent=2))
with zipfile.ZipFile(ZIP,'w',zipfile.ZIP_DEFLATED) as z:
    for p in files:z.write(p,ROOT.name+'/'+str(p.relative_to(ROOT)))
    z.write(QA/'package_manifest.json',ROOT.name+'/qa/package_manifest.json')

temp = Path(tempfile.mkdtemp(prefix='trident_v8_portable_'))
with zipfile.ZipFile(ZIP) as z:
    assert z.testzip() is None
    z.extractall(temp)
build_root = temp/ROOT.name
log_path = QA/'portable_build.log'
with log_path.open('w') as log:
    result = subprocess.run(['bash','build.sh'],cwd=build_root,stdout=log,stderr=subprocess.STDOUT)
assert result.returncode == 0, log_path
warnings = {}
for name in ('main','supplement','review'):
    log=(build_root/'build'/f'{name}.log').read_text(errors='replace')
    warnings[name]=[line for line in log.splitlines() if re.search(r'Overfull|Underfull|undefined|LaTeX Warning|Package .+ Warning|^!',line)]
    assert not warnings[name], warnings[name]
report={'status':'PASS','zip':ZIP.name,'source_files':len(files)+1,
        'zip_bytes':ZIP.stat().st_size,'sha256':hashlib.sha256(ZIP.read_bytes()).hexdigest(),
        'extracted_directory':str(build_root),'independent_compile_returncode':result.returncode,
        'documents':['main','supplement','review'],'compile_issues':warnings,
        'scope':'Manuscript and quantitative figure sources; excludes original WSIs, patient mappings, weights, and training archive.'}
(QA/'portable_build.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
