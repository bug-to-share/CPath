"""Check v8 evidence against saved predictions; no model training or inference."""
from pathlib import Path
import hashlib
import json
import re
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

ROOT = Path(__file__).resolve().parent
H = ROOT / 'source_data/historical'
A = ROOT / 'source_data/algorithm'
checks = []


def check(name, condition):
    checks.append({'check': name, 'passed': bool(condition)})
    if not condition:
        raise AssertionError(name)


def same(name, actual, expected, atol=1e-8):
    check(name, np.allclose(actual, expected, atol=atol, rtol=0))


def metrics(frame, score='score'):
    y = frame.label.to_numpy()
    p = frame[score].to_numpy()
    tn, fp, fn, tp = confusion_matrix(y, p >= 0.5, labels=[0, 1]).ravel()
    return dict(n=len(y), n_malt=int(y.sum()), n_ss=int((1-y).sum()),
                auroc=roc_auc_score(y, p), average_precision=average_precision_score(y, p),
                sensitivity=tp/(tp+fn), specificity=tn/(tn+fp),
                balanced_accuracy=(tp/(tp+fn)+tn/(tn+fp))/2,
                accuracy=(tp+tn)/len(y), brier=np.mean((p-y)**2),
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))


primary = {}
internal = pd.read_csv(H / 'internal_predictions_hybrid_evidence_mil.csv')
external = pd.read_csv(H / 'external_final_predictions.csv')
im = pd.read_csv(H / 'internal_model_metrics.csv').set_index('aggregator').loc['hybrid_evidence_mil']
em = json.loads((H / 'external_final_metrics.json').read_text())
for scope, frame, score, expected in [('internal', internal, 'probability', im),
                                     ('external', external, 'malt_probability', em)]:
    result = metrics(frame, score)
    for key in ('auroc', 'sensitivity', 'specificity', 'accuracy', 'tn', 'fp', 'fn', 'tp'):
        same(f'primary {scope}: {key}', result[key], expected[key])
    same(f'primary {scope}: AP', result['average_precision'], expected['auprc'])
    primary[scope] = result

review = json.loads((ROOT / 'source_data/review/numeric_audit.json').read_text())
subset = external[external.stain_group != 'IHC / special stain']
subset_metrics = metrics(subset, 'malt_probability')
expected = review['archive_tag_sensitivity_analysis_not_confirmed_HE_validation']
for key in subset_metrics:
    same(f'archive-tag descriptive subset: {key}', subset_metrics[key], expected[key])
for row in review['archive_tag_counts']:
    g = external[(external.cohort == row['cohort']) & (external.stain_group == row['tag'])]
    same(f'archive count {row["tag"]}', len(g), row['n'])
    same(f'archive positives {row["tag"]}', (g.malt_probability >= .5).sum(), row['positive_at_0_5'])
for key, prob in [('external_prevalence_constant', 152/168), ('training_prevalence_constant', 67/144)]:
    same(key, np.mean((prob-external.label)**2), review['descriptive_brier_references'][key])

predictions = pd.read_csv(A / 'original_all_predictions.csv')
archived_metrics = pd.read_csv(A / 'original_run_metrics.csv')
keys = ['variant', 'seed', 'stage', 'scope', 'inference']
check('algorithm prediction rows: 8246', len(predictions) == 8246)
check('algorithm metric rows: 76', len(archived_metrics) == 76)
check('no duplicate algorithm slide prediction', not predictions.duplicated(keys+['study_slide_id']).any())
for group_key, g in predictions.groupby(keys, dropna=False):
    selected = archived_metrics
    for column, value in zip(keys, group_key):
        selected = selected[selected[column] == value]
    check(f'unique metric row {group_key}', len(selected) == 1)
    for key, value in metrics(g).items():
        same(f'algorithm {group_key}: {key}', value, selected.iloc[0][key])
neural = archived_metrics[archived_metrics.stage != 'deterministic_control']
check('34 neural training runs', len(neural[['variant', 'seed']].drop_duplicates()) == 34)

five = pd.read_csv(A / 'selected_control_all_five_per_seed.csv')
summary = pd.read_csv(A / 'confirmation_summary.csv')
for (variant, scope), g in five.groupby(['variant', 'scope']):
    check(f'all five seeds {variant} {scope}', set(g.seed) == {17,29,43,59,71})
    r = summary[(summary.variant == variant) & (summary.scope == scope) & (summary.seed_group == 'all_five')].iloc[0]
    for key in ['auroc', 'sensitivity', 'specificity']:
        same(f'five-seed mean {variant} {scope} {key}', g[key].mean(), r[key+'_mean'])
    same(f'five-seed SD {variant} {scope}', g.auroc.std(ddof=1), r.auroc_sd)
same('parameter reduction percent', 100*(1-658178/1521672), 56.75, atol=.005)

ens = pd.read_csv(A / 'ensemble_predictions.csv')
ens_metrics = pd.read_csv(A / 'ensemble_metrics.csv')
for (variant, scope, mode), g in ens.groupby(['variant', 'scope', 'mode']):
    r = ens_metrics[(ens_metrics.variant == variant) & (ens_metrics.scope == scope) & (ens_metrics['mode'] == mode)].iloc[0]
    for key, value in metrics(g).items():
        same(f'ensemble {variant} {scope} {mode}: {key}', value, r['brier_score' if key == 'brier' else key])

draws = pd.read_csv(H / 'sampling_per_draw_metrics.csv')
slides = pd.read_csv(H / 'sampling_per_slide.csv')
sampling = pd.read_csv(H / 'sampling_summary.csv')
for scope in ['internal', 'external']:
    g = slides[slides.scope == scope]
    d = draws[draws.scope == scope]
    r = sampling[sampling.scope == scope].iloc[0]
    check(f'20 sampling draws {scope}', len(d) == 20)
    same(f'threshold crossings {scope}', g.crosses_threshold_0_5.sum(), r.threshold_crossing_slides)
    same(f'median score SD {scope}', g.score_sd.median(), r.score_sd_median)
    same(f'draw AUROC extrema {scope}', [d.auroc.min(), d.auroc.max()], [r.draw_auroc_min, r.draw_auroc_max])
    same(f'mean-bag AUROC {scope}', metrics(g, 'score_mean')['auroc'], r.mean_bag_auroc)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

manifest = json.loads((ROOT / 'qa/source_manifest.json').read_text())
for entry in manifest:
    # references.bib intentionally incorporates seven new entries; all other imports are frozen.
    if entry['destination'] != 'references.bib':
        check('copied evidence hash '+entry['destination'], sha(ROOT / entry['destination']) == entry['sha256'])
old = ROOT.parent / 'draft_v7_focused_narrative'
old_status = 'not present in portable bundle; original-workspace check recorded separately'
if old.is_dir():
    old_hashes = json.loads((ROOT / 'qa/v7_before.json').read_text())
    for filename, expected_hash in old_hashes.items():
        check('v7 preserved '+filename, sha(old / filename) == expected_hash)
    old_status = f'{len(old_hashes)} pre-existing files unchanged'

sources = '\n'.join(p.read_text() for p in (ROOT / 'sections').glob('*.tex'))
check('SS terminology; no SP/non-MALT class', re.search(r'\bSP\b|non[-–]MALT', sources) is None)
defined = set(re.findall(r'@\w+\s*\{\s*([^,]+)', (ROOT / 'references.bib').read_text()))
used = {key.strip() for group in re.findall(r'\\cite\w*\{([^}]+)\}', sources) for key in group.split(',')}
check('all cited bibliography keys defined', used <= defined)
check('all 12 region cases use publication case names', all(name in (ROOT/'tables/S9_regions.tex').read_text() for name in pd.read_csv(H/'interpretation_probability_comparison.csv').publication_case))

result = {'status': 'PASS', 'checks_passed': len(checks), 'primary_recomputed': primary,
          'archive_subset_recomputed': subset_metrics, 'algorithm_prediction_rows': len(predictions),
          'algorithm_metric_rows': len(archived_metrics), 'ensemble_metric_rows': len(ens_metrics),
          'v7_preservation': old_status, 'checks': checks}
(ROOT / 'qa/numerical_validation.json').write_text(json.dumps(result, indent=2))
print(json.dumps({k:v for k,v in result.items() if k != 'checks'}, indent=2))
