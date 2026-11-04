"""Read-only numerical audit of archived predictions; writes only this review folder."""
from pathlib import Path
import csv
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, roc_auc_score

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent.parent
V7 = ROOT / 'writing/draft_v7_focused_narrative'
ALG = ROOT / 'writing/algorithm_improvement_20261104'


def metrics(frame, score):
    y = frame['label'].to_numpy(dtype=int)
    p = frame[score].to_numpy(dtype=float)
    tn, fp, fn, tp = confusion_matrix(y, p >= .5, labels=[0, 1]).ravel()
    return dict(n=len(y), n_malt=int(sum(y)), n_ss=int(sum(y == 0)),
                auroc=float(roc_auc_score(y, p)), average_precision=float(average_precision_score(y, p)),
                sensitivity=float(tp / (tp + fn)), specificity=float(tn / (tn + fp)),
                balanced_accuracy=float((tp / (tp + fn) + tn / (tn + fp)) / 2),
                accuracy=float((tp + tn) / len(y)), npv=float(tn / (tn + fn)),
                brier=float(brier_score_loss(y, p)), tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))


external_path = V7 / 'supporting_data/external_final_predictions.csv'
external = pd.read_csv(external_path)
result = {'date': '2026-11-04', 'timezone': 'Asia/Shanghai',
          'scope': 'Existing fixed predictions only; no model training or new clinical validation.',
          'external_main': metrics(external, 'malt_probability')}
archived = json.loads((V7 / 'supporting_data/external_final_metrics.json').read_text())
assert abs(result['external_main']['auroc'] - archived['auroc']) < 1e-12
assert result['external_main']['tp'] == archived['tp']

# These selections use provisional archive tags. SS stain and patient linkage are unconfirmed.
filtered = external[~external.stain_group.str.contains('IHC', case=False, na=False)]
result['archive_tag_sensitivity_analysis_not_confirmed_HE_validation'] = metrics(filtered, 'malt_probability')
result['archive_tag_caution'] = ('Excluding IHC/special-stain archive tags is a post hoc descriptive calculation. '
                               'It retains all 16 SS slides with unspecified stain; it does not establish H&E-only '
                               'or patient-independent clinical performance.')
result['archive_tag_counts'] = []
for (cohort, tag, label), frame in external.groupby(['cohort', 'stain_group', 'label']):
    result['archive_tag_counts'].append(dict(cohort=cohort, tag=tag, label=int(label), n=len(frame),
                                           positive_at_0_5=int(frame.malt_probability.ge(.5).sum())))
prevalence = float(external.label.mean())
train_prior = 67 / 144
result['descriptive_brier_references'] = {
    'external_prevalence': prevalence,
    'external_prevalence_constant': prevalence * (1-prevalence),
    'training_prevalence_constant': float(np.mean((external.label.to_numpy()-train_prior)**2)),
    'caution': 'The external-prevalence constant uses evaluation labels and is descriptive, not a deployable comparator.'}

run = pd.read_csv(ALG / 'outputs/all_predictions.csv')
result['algorithm_prediction_columns'] = list(run.columns)
verified = pd.read_csv(ALG / 'analysis/run_metrics_verified.csv')
recomputed = []
for keys, frame in run.groupby(['variant', 'seed', 'stage', 'scope', 'inference']):
    row = dict(zip(['variant', 'seed', 'stage', 'scope', 'inference'], keys))
    row.update(metrics(frame, 'score'))
    recomputed.append(row)
recomputed = pd.DataFrame(recomputed)
merge = recomputed.merge(verified, on=['variant', 'seed', 'stage', 'scope', 'inference'], suffixes=('_new', '_old'), validate='one_to_one')
assert len(merge) == len(verified)
errors = {}
for key in ['auroc', 'average_precision', 'sensitivity', 'specificity', 'balanced_accuracy', 'accuracy', 'tn', 'fp', 'fn', 'tp']:
    errors[key] = float(np.max(np.abs(merge[key+'_new'] - merge[key+'_old'])))
    assert errors[key] < 1e-10, (key, errors[key])
result['independent_recomputation'] = {'prediction_rows': len(run), 'metric_rows': len(merge), 'max_errors': errors}
recomputed.to_csv(OUT / 'recomputed_algorithm_metrics.csv', index=False)
confirmation = pd.read_csv(ALG / 'analysis/confirmation_summary.csv')
result['five_seed_summary'] = confirmation[confirmation.seed_group.eq('all_five')].to_dict('records')
for row in result['five_seed_summary']:
    selected = recomputed[(recomputed.variant == row['variant']) & (recomputed.scope == row['scope']) & (recomputed.inference == 'reference_bag')]
    assert len(selected) == 5
    assert abs(selected.auroc.mean() - row['auroc_mean']) < 1e-12
    assert abs(selected.auroc.std(ddof=1) - row['auroc_sd']) < 1e-12
summary = json.loads((ALG / 'analysis/summary.json').read_text())
result['engineering_success'] = summary['engineering_success']
linear = pd.read_csv(ALG / 'analysis/linear_controls.csv')
result['linear_reference_bag_controls'] = linear[linear.inference.eq('reference_bag')].to_dict('records')
runtime = json.loads((ALG / 'outputs/runtime.json').read_text())
result['completed_neural_runs'] = runtime['n_neural_runs']
result['new_neural_run_archives'] = len(list((ALG / 'outputs').glob('*/history.csv')))
assert result['completed_neural_runs'] == result['new_neural_run_archives'] == 34

files = [external_path, V7 / 'sections/methods.tex', V7 / 'sections/results.tex',
         V7 / 'MALT_SS_manuscript_v7.pdf', V7 / 'Supplementary_information_v7.pdf',
         ALG / 'outputs/all_predictions.csv', ALG / 'analysis/run_metrics_verified.csv',
         ALG / 'analysis/confirmation_summary.csv', ALG / 'analysis/linear_controls.csv',
         ALG / 'analysis/summary.json', ROOT / 'scripts/ss_experiment_suite.py']
result['input_sha256'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
path = OUT / 'numeric_audit.json'
path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({k: result[k] for k in ['external_main', 'archive_tag_sensitivity_analysis_not_confirmed_HE_validation',
                                     'descriptive_brier_references', 'completed_neural_runs']}, indent=2))
