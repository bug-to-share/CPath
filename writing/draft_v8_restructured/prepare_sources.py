"""Copy numeric evidence into the manuscript bundle without altering earlier work."""
from pathlib import Path
import hashlib, json, shutil

R=Path(__file__).resolve().parent
W=R.parent; V=W/'draft_v7_focused_narrative'; A=W/'algorithm_improvement_20261104'
for d in ['sections','tables','figures','source_data/historical','source_data/controlled','source_data/algorithm','source_data/review','qa','build','protocols']:
    (R/d).mkdir(parents=True,exist_ok=True)
manifest=[]
def copy(src,dest):
    target=R/dest; target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(src,target)
    manifest.append({'source':str(src.relative_to(W)),'destination':str(target.relative_to(R)),
                     'sha256':hashlib.sha256(src.read_bytes()).hexdigest()})
for p in (V/'supporting_data').iterdir():
    if p.suffix in ['.csv','.json'] and not p.name.startswith(('reader_','manifest')):
        copy(p,'source_data/historical/'+p.name)
for name in ['reference_predictions.csv','reference_model_metrics.csv','sampling_per_draw_metrics.csv',
             'sampling_summary.csv','sampling_per_slide.csv','external_paired_comparisons.csv']:
    copy(V/'experiments/outputs'/name,'source_data/historical/'+name)
for name in ['brier_descriptive.csv','roi_mask_coverage.csv','model_checkpoint_inventory.csv','paired_error_exchange.csv']:
    copy(V/'analysis'/name,'source_data/historical/'+name)
for name in ['configuration_summary.csv','run_metrics.csv','all_predictions.csv','paired_seed_differences.csv']:
    copy(V/'experiments/controlled_training'/name,'source_data/controlled/'+name)
for p in (A/'analysis').glob('*'):
    if p.suffix in ['.json','.csv']:
        copy(p,'source_data/algorithm/'+p.name)
for name in ['run_metrics.csv','all_predictions.csv','selection_lock.json','runtime.json']:
    copy(A/'outputs'/name,'source_data/algorithm/original_'+name)
copy(W/'research_review_20261104/numeric_audit.json','source_data/review/numeric_audit.json')
copy(V/'references.bib','references.bib')
copy(V/'sections/methods_equations.tex','sections/supplement_equations.tex')
copy(A/'PROTOCOL.md','protocols/algorithm_study.md')
copy(A/'configurations.json','protocols/algorithm_configurations.json')
copy(V/'experiments/CONTROLLED_TRAINING_PROTOCOL.md','protocols/controlled_training.md')
copy(V/'experiments/PROTOCOL.md','protocols/fixed_checkpoint_sampling.md')
copy(V/'source_data/technical_provenance.json','qa/inherited_technical_provenance.json')
for n in ['Figure_5_case_evidence','Supplementary_Figure_S2_histological_context']:
    copy(V/'figures'/f'{n}.pdf','figures/archive_'+n+'.pdf')
for n in ['Figure_A1_all_configurations','Figure_A2_stability_and_aggregation','Figure_A3_locked_ensemble_ROC_confusion']:
    copy(A/'figures'/f'{n}.pdf','figures/'+n+'.pdf')
(R/'qa/source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
old={str(p.relative_to(V)):hashlib.sha256(p.read_bytes()).hexdigest() for p in V.rglob('*')
     if p.is_file() and '__pycache__' not in p.parts}
(R/'qa/v7_before.json').write_text(json.dumps(old,indent=2)+'\n')
print(f'Copied {len(manifest)} evidence files; archived {len(old)} v7 hashes.')
