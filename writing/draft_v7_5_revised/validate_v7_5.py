"""CPU-only manuscript QA: independently recompute retained scores and check assets."""
from pathlib import Path
import csv,hashlib,json,re,subprocess,sys,os
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score,brier_score_loss,confusion_matrix,roc_auc_score
R=Path(__file__).resolve().parent
Q=R/'qa';Q.mkdir(exist_ok=True)
A=R/'development_analyses'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def met(f,col='score'):
    y=f.label.to_numpy(int);s=f[col].to_numpy(float)
    tn,fp,fn,tp=map(int,confusion_matrix(y,s>=.5,labels=[0,1]).ravel())
    return dict(auroc=roc_auc_score(y,s),average_precision=average_precision_score(y,s),
                sensitivity=tp/(tp+fn),specificity=tn/(tn+fp),balanced_accuracy=(tp/(tp+fn)+tn/(tn+fp))/2,
                accuracy=(tp+tn)/len(y),brier_score=brier_score_loss(y,s),tn=tn,fp=fp,fn=fn,tp=tp)
report={}
# Original files and the original chart/data assets are immutable in this revision.
m=json.loads((Q/'v7_source_hashes.json').read_text());v7=R.parent/'draft_v7_focused_narrative'
if v7.exists():
    diffs=[p for p,h in m.items() if not (v7/p).exists() or sha(v7/p)!=h]
    assert not diffs,diffs
    report['v7_source_unchanged_files']=len(m)
for folder in ['figures','assets','supporting_data','source_data']:
    selected={p:h for p,h in m.items() if p.startswith(folder+'/')}
    assert all(sha(R/p)==h for p,h in selected.items()),folder
    report[folder+'_byte_identical_files']=len(selected)
if v7.exists():
    def numeric_cells(path):
        s=path.read_text();body='\n'.join(re.findall(r'\\midrule(.*?)\\bottomrule',s,re.S))
        return re.findall(r'(?<![A-Za-z])\d+(?:\.\d+)?',body)
    retained=numeric_cells(R/'sections/tables.tex')
    assert numeric_cells(v7/'sections/tables.tex')==retained
    report['original_main_table_numbers_retained_in_main']=True
# Run-level recomputation from all 8,246 prediction rows, without calling the author analysis script.
f=pd.read_csv(A/'outputs/all_predictions.csv');keys=['variant','seed','stage','scope','inference']
calc=pd.DataFrame([dict(zip(keys,k),**met(g)) for k,g in f.groupby(keys)])
ref=pd.read_csv(A/'analysis/run_metrics_verified.csv')
j=calc.merge(ref,on=keys,suffixes=('_calc','_ref'),validate='one_to_one')
assert len(j)==len(ref)==76
maxerr={k:float(abs(j[k+'_calc']-j[k+'_ref']).max()) for k in met(f)}
assert max(maxerr.values())<1e-12,maxerr
report['neural_and_linear_recomputation']={'prediction_rows':len(f),'metric_groups':len(j),'max_errors':maxerr}
calc.to_csv(Q/'independently_recomputed_metrics.csv',index=False)
# Screen and all seed groups: recompute means and sample SDs rather than trust summary files.
for file in ['screen_summary.csv','confirmation_summary.csv']:
    for row in pd.read_csv(A/'analysis'/file).to_dict('records'):
        seeds=[int(x) for x in row['seeds'].split(',')]
        g=calc[(calc.variant==row['variant'])&(calc.scope==row['scope'])&(calc.inference=='reference_bag')&calc.seed.isin(seeds)]
        assert len(g)==len(seeds)
        for k in ['auroc','sensitivity','specificity','balanced_accuracy','accuracy','brier_score','average_precision']:
            assert abs(g[k].mean()-row[k+'_mean'])<1e-12,(file,k)
            assert abs(g[k].std(ddof=1)-row[k+'_sd'])<1e-12,(file,k)
report['screen_and_seed_summary_checks']='PASS'
# Independent ensemble construction and metric checks, including five-bag means.
epred=pd.read_csv(A/'analysis/ensemble_predictions.csv')
eref=pd.read_csv(A/'analysis/ensemble_metrics.csv')
bag=pd.read_csv(A/'outputs/five_bag_mean_predictions.csv')
for (v,scope,mode),g in epred.groupby(['variant','scope','mode']):
    base=f[f.inference.eq('reference_bag')] if mode.endswith('reference') else bag
    seeds=([17,29,43] if mode.startswith('three') else [59,71] if mode.startswith('additional') else [17,29,43,59,71])
    raw=base[(base.variant==v)&(base.scope==scope)&base.seed.isin(seeds)]
    assert raw.seed.nunique()==len(seeds)
    made=raw.groupby(['study_slide_id','label'],as_index=False).score.mean()
    merged=g.merge(made,on=['study_slide_id','label'],suffixes=('_saved','_new'),validate='one_to_one')
    assert len(merged)==len(g) and max(abs(merged.score_saved-merged.score_new))<1e-12
    row=eref[(eref.variant==v)&(eref.scope==scope)&(eref['mode']==mode)].iloc[0]
    for k,value in met(g).items():assert abs(value-row[k])<1e-12,(v,mode,k)
report['aggregation_metrics_checked']=len(eref)
# Recompute post hoc tag subset and Brier references.
ex=pd.read_csv(R/'supporting_data/external_final_predictions.csv')
audit=json.loads((R/'analysis/review_numeric_audit.json').read_text())
for frame,key in [(ex,'external_main'),(ex[~ex.stain_group.str.contains('IHC',case=False,na=False)],'archive_tag_sensitivity_analysis_not_confirmed_HE_validation')]:
    for k,value in met(frame,'malt_probability').items():
        ak='brier' if k=='brier_score' else k
        assert abs(value-audit[key][ak])<1e-12,(key,k)
report['tag_subset_metrics_checked']=True
br=audit['descriptive_brier_references']
assert abs(np.mean((ex.label-ex.label.mean())**2)-br['external_prevalence_constant'])<1e-12
assert abs(np.mean((ex.label-67/144)**2)-br['training_prevalence_constant'])<1e-12
# Check generated tables are reproducible from the data and are current.
before=sha(R/'sections/development_tables.tex')
subprocess.run([sys.executable,str(R/'build_development_tables.py')],check=True,capture_output=True)
assert sha(R/'sections/development_tables.tex')==before
report['supplemental_tables_regeneration']='PASS'
# All source markers are retained; update the location register after the rewrite.
markers=[]
for p in sorted((R/'sections').glob('*.tex')):
    if p.name=='controlled_training_details.tex':continue # archived, no longer included
    for i,line in enumerate(p.read_text().splitlines(),1):
        for marker in re.findall(r'\[(?:MISSING|AUTHOR CHECK)[^\]]+\]',line):
            markers.append(dict(file=str(p.relative_to(R)),line=i,marker=marker))
with (R/'missing_information_register.csv').open('w',newline='',encoding='utf-8-sig') as out:
    w=csv.DictWriter(out,fieldnames=['file','line','marker']);w.writeheader();w.writerows(markers)
ids=set(re.findall(r'\b[ACMRP]\d\d\b','\n'.join(x['marker'] for x in markers)))
assert ids=={'A01','A02','A03','A04','A05','A06','A07','A08','C02','C03','C04','C05','C06','M01','M02','P01','P03','R01','R02'},ids
report['pending_marker_occurrences']=len(markers);report['pending_information_ids']=sorted(ids)
# Build diagnostics, table/figure counts and PDF page geometry.
pdfs=json.loads(subprocess.run([os.environ.get('PDF_QA_PYTHON','python'), str(R/'validate_pdf_layout.py')],check=True,capture_output=True,text=True).stdout)
report['pdf_checks']=pdfs
report['status']='PASS'
(Q/'validation_v7_5.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
