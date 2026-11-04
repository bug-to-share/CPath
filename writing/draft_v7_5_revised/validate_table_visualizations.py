# Historical validator for revisions/before_ranked_tables; not the current layout.
# Current validator: validate_ranked_tables.py.
"""Check plotted source records and table migration independently of plotting code."""
from pathlib import Path
import hashlib,json,re,subprocess,os
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

R=Path(__file__).resolve().parent
S=R/'analysis/table_visualization'
A=R/'development_analyses/analysis'
Q=R/'qa'
snapshot=json.loads((Q/'table_visualization_before.json').read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
changed=[p for p,h in snapshot['original_data_hashes'].items() if sha(R/p)!=h]
assert not changed,changed

def read(p):return pd.read_csv(p)
def identical(actual,expected,keys):
    a=actual.sort_values(keys).reset_index(drop=True)
    b=expected.sort_values(keys).reset_index(drop=True)
    assert_frame_equal(a[b.columns],b,check_exact=False,rtol=1e-13,atol=1e-13)

internal=read(R/'supporting_data/internal_model_metrics.csv')
identical(read(S/'figure2_internal_metrics.csv'),internal,['aggregator'])
external=json.loads((R/'supporting_data/external_final_metrics.json').read_text())
hybrid=internal[internal.aggregator.eq('hybrid_evidence_mil')].iloc[0]
intervals=read(S/'figure3_cohort_intervals.csv')
assert len(intervals)==16
for row in intervals.to_dict('records'):
    source=hybrid if row['cohort']=='Internal' else external
    for column,suffix in [('value',''),('ci_low','_ci_low'),('ci_high','_ci_high')]:
        assert abs(row[column]-source[row['metric']+suffix])<1e-13

pred=read(R/'supporting_data/external_final_predictions.csv')
group_specs=[('MALT: H&E-like','MALT_BJ','H&E-like'),('MALT: IHC/special','MALT_BJ','IHC / special stain'),('MALT: HE archive','MALT_HE',None),('SS: stain unknown','SS_BJ',None)]
groups=read(S/'figure3_archive_groups.csv').set_index('group')
for label,cohort,tag in group_specs:
    d=pred[pred.cohort.eq(cohort)]
    if tag is not None:d=d[d.stain_group.eq(tag)]
    correct=int((d.label==(d.malt_probability>=.5).astype(int)).sum())
    assert groups.loc[label].to_dict()==dict(n=len(d),correct=correct,incorrect=len(d)-correct)

audit=json.loads((R/'analysis/review_numeric_audit.json').read_text())
posthoc=read(S/'figure3_tag_exclusion.csv')
assert len(posthoc)==8
for row in posthoc.to_dict('records'):
    key='external_main' if row['cohort']=='Full external' else 'archive_tag_sensitivity_analysis_not_confirmed_HE_validation'
    assert abs(row['value']-audit[key][row['metric']])<1e-13

neural=read(A/'selected_control_all_five_per_seed.csv')
linear=read(A/'linear_controls.csv');linear=linear[linear.inference.eq('reference_bag')]
expected=pd.concat([neural,linear],ignore_index=True)
identical(read(S/'figure4_seed_and_linear_metrics.csv'),expected,['variant','scope','seed'])
summary=read(S/'figure4_summary.csv');assert len(summary)==16
for row in summary.to_dict('records'):
    z=expected[expected.variant.eq(row['variant'])&expected.scope.eq(row['scope'])][row['metric']]
    assert len(z)==row['n_runs'] and abs(z.mean()-row['mean'])<1e-13
    assert (len(z)==1 and pd.isna(row['sd'])) or abs(z.std(ddof=1)-row['sd'])<1e-13

screen=read(A/'run_metrics_verified.csv');screen=screen[screen.stage.eq('screen')&screen.inference.eq('reference_bag')]
assert len(screen)==60
identical(read(S/'figureS7_all_screen_seeds.csv'),screen,['variant','scope','seed'])
agg=read(A/'ensemble_metrics.csv');agg=agg[~agg['mode'].eq('additional_two_seed_reference')]
assert len(agg)==32
identical(read(S/'figureS8_aggregation_metrics.csv'),agg,['variant','scope','mode'])

old=snapshot['section_texts']['sections/tables.tex']
new=(R/'sections/tables.tex').read_text()+(R/'sections/relocated_numeric_tables.tex').read_text()
numeric=lambda s:re.findall(r'(?<![A-Za-z])\d+(?:\.\d+)?','\n'.join(re.findall(r'\\midrule(.*?)\\bottomrule',s,re.S)))
assert numeric(old)==numeric(new)
oldrows=re.findall(r'\\midrule(.*?)\\bottomrule',old,re.S)
newrows=re.findall(r'\\midrule(.*?)\\bottomrule',new,re.S)
assert oldrows==newrows
assert len(re.findall(r'\\begin\{table\}',(R/'sections/tables.tex').read_text()))==1
assert len(re.findall(r'\\begin\{table\}',(R/'sections/relocated_numeric_tables.tex').read_text()))==3

oldfigure=snapshot['section_texts']['sections/figures.tex']
newfigure=(R/'sections/figures.tex').read_text()
for macro in ['MainFigureOne','MainFigureFour','MainFigureFive','MainFigureSix']:
    def block(s):
        start=s.index('\\newcommand{\\'+macro+'}')
        end=s.find('\\newcommand',start+12)
        return s[start:end if end>=0 else len(s)].strip()
    assert block(oldfigure)==block(newfigure),macro

layout=json.loads(subprocess.run([os.environ.get('PDF_QA_PYTHON','python'),str(R/'validate_pdf_layout.py')],capture_output=True,text=True,check=True).stdout)
report={'status':'PASS','original_data_and_asset_files_unchanged':len(snapshot['original_data_hashes']),
 'old_main_table_rows_verbatim_retained':sum(len(re.findall(r'\\\\',x)) for x in oldrows),
 'main_tables_before':4,'main_tables_after':1,'main_figures_before':6,'main_figures_after':7,
 'relocated_tables':['S20','S21','S22'],'original_ROC_PR_figures_retained_as':['S5','S6'],
 'new_quantitative_figures':5,'source_record_checks':{'internal_models':12,'cohort_intervals':16,'archive_groups':4,'tag_subset_metrics':8,'seed_and_linear_records':len(expected),'seed_summary_records':16,'screen_records':60,'aggregation_records':32},
 'retained_main_figure_macros_unchanged':['Figure 1 including accepted b panel','reader comparison (now Figure 5)','case evidence (now Figure 6)','sampling and errors (now Figure 7)'],
 'new_training_or_inference':False,'pdf_checks':layout}
(Q/'table_visualization_check.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
