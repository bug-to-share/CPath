"""Package the local writing revision; no publication, private records or model weights."""
from pathlib import Path
import hashlib,json,zipfile
R=Path(__file__).resolve().parent
FORBIDDEN={'.pth','.pt','.bin','.h5','.ndpi','.svs','.joblib','.pkl','.npy'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def allowed(p):
    rel=p.relative_to(R)
    return p.is_file() and 'PRIVATE' not in str(rel).upper() and '__pycache__' not in str(rel) and p.suffix not in FORBIDDEN

def collect(folder,suffixes=None):
    return [p for p in (R/folder).rglob('*') if allowed(p) and (suffixes is None or p.suffix in suffixes)]
def pack(name,files):
    files=sorted(set(files))
    with zipfile.ZipFile(R/name,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files:
            assert allowed(p),p
            z.write(p,str(p.relative_to(R)))
    return dict(file=name,n_files=len(files),bytes=(R/name).stat().st_size,sha256=sha(R/name))

def main():
    data=collect('supporting_data')+collect('source_data')
    data+=collect('analysis',{'.csv','.json','.md'})
    data+=collect('experiments/outputs',{'.csv','.json','.npz'})
    data+=collect('experiments/controlled_training',{'.csv','.json','.npz'})
    data+=collect('development_analyses/analysis',{'.csv','.json'})
    data+=collect('development_analyses/outputs',{'.csv','.json','.npz'})
    code=collect('supporting_code',{'.py','.md','.json','.txt'})
    code+=[p for p in (R/'experiments').glob('*') if p.suffix in {'.py','.md'}]
    code+=[p for p in (R/'development_analyses').glob('*') if p.suffix in {'.py','.md','.json','.txt'}]
    code+=[p for p in R.glob('*.py')]+[R/'analysis/analyze_existing_evidence.py']
    code += [R/'README.md',R/'requirements_qa.txt',R/'requirements_render.txt']
    docs=[p for p in R.glob('*') if p.is_file() and p.suffix in {'.tex','.bib','.sh','.md','.txt','.csv'}]
    docs+=[R/'latexmkrc']+collect('sections')+collect('figures')+collect('assets')
    docs += collect('figure1b_candidates',{'.md','.txt','.json'})
    qa=[R/'qa'/name for name in ['validation_v7_5.json','v7_source_hashes.json','independently_recomputed_metrics.csv','inline_layout_check.json','figure1b_replacement_check.json','table_visualization_check.json','table_visualization_before.json','ranked_tables_check.json','ranked_tables_before.json','ranked_table_marks.json','ranked_table_baseline_cells.json']]
    outputs=[pack('Source_Data_draft.zip',data),pack('Source_Code_draft.zip',code),
             pack('MALT_SS_LaTeX_sources_v7_5.zip',docs+data+code+qa)]
    (R/'qa/package_manifest.json').write_text(json.dumps(outputs,indent=2)+'\n')
    print(json.dumps(outputs,indent=2))
if __name__=='__main__':main()
