#!/usr/bin/env python
"""CPU-only reporting for the locked, bounded algorithm study.

Figure contract: quantitative grids with all prespecified screen seeds, the
internally selected candidate, and the original Hybrid control. Selection is
never recomputed from external scores. Error bars in the screen are sample SD
across training seeds, not clinical confidence intervals. ROC/CM panels use a
specified probability ensemble, not the best single seed. Export editable
PDF/SVG and 450 dpi PNG; all plot data are saved as CSV before rendering.
"""
from pathlib import Path
import hashlib
import json
import os

os.environ.setdefault('MPLCONFIGDIR', '/tmp/trident_algorithm_mpl')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('OMP_NUM_THREADS', '2')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                             confusion_matrix, roc_curve)

HERE = Path(__file__).resolve().parent
OUT = HERE / 'outputs'
ANA = HERE / 'analysis'
FIG = HERE / 'figures'
OLD = HERE.parent / 'draft_v7_focused_narrative'
SCREEN = [17, 29, 43]
EXTRA = [59, 71]
ALL = SCREEN + EXTRA
BASELINE = 'hybrid_control'
BOOTSTRAPS = 10000
BOOTSEED = 20261104
VARIANTS = ['hybrid_control', 'attention_control', 'hybrid_l2', 'attention_l2',
            'attention_mask', 'direct_attention', 'subspace_attention',
            'subspace_largebatch', 'hybrid_twobag', 'hybrid_consistency']
LABELS = {'hybrid_control': 'Hybrid control', 'attention_control': 'Attention control',
          'hybrid_l2': 'Hybrid + tile L2', 'attention_l2': 'Attention + tile L2',
          'attention_mask': 'Attention + top-K mask', 'direct_attention': 'Raw-space attention',
          'subspace_attention': 'Subspace attention', 'subspace_largebatch': 'Subspace + batch/LR recipe',
          'hybrid_twobag': 'Hybrid + two bags', 'hybrid_consistency': 'Hybrid + bag consistency',
          'mean_logistic': 'Mean-feature logistic', 'l2mean_logistic': 'L2 mean-feature logistic'}
MEASURES = ['auroc', 'average_precision', 'sensitivity', 'specificity', 'balanced_accuracy',
            'accuracy', 'brier_score']
BLUE, ORANGE, GREY = '#416B8E', '#C47A45', '#8C939B'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def writejson(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def clean_records(df):
    return json.loads(df.to_json(orient='records'))


def wilson(k, n):
    if n == 0:
        return np.nan, np.nan
    z = 1.959963984540054
    p, d = k / n, 1 + z*z/n
    center = (p + z*z/(2*n)) / d
    half = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return center-half, center+half


def metric(g):
    y, p = g.label.to_numpy(dtype=int), g.score.to_numpy(dtype=float)
    assert len(y) and set(y) == {0, 1} and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    tn, fp, fn, tp = confusion_matrix(y, p >= .5, labels=[0, 1]).ravel()
    m = {'n': len(y), 'n_malt': int(y.sum()), 'n_ss': int((y == 0).sum()),
         'auroc': float(roc_auc_score(y, p)), 'average_precision': float(average_precision_score(y, p)),
         'sensitivity': float(tp/(tp+fn)), 'specificity': float(tn/(tn+fp)),
         'balanced_accuracy': float(.5*(tp/(tp+fn)+tn/(tn+fp))),
         'accuracy': float((tp+tn)/len(y)), 'brier_score': float(brier_score_loss(y, p)),
         'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp), 'threshold': .5}
    for name, k, n in [('sensitivity', tp, tp+fn), ('specificity', tn, tn+fp),
                        ('accuracy', tp+tn, len(y))]:
        m[name+'_wilson_low'], m[name+'_wilson_high'] = map(float, wilson(k, n))
    return m


class PairedBootstrap:
    """Exact AUC ranking kernels with shared stratified bootstrap counts."""
    def __init__(self, labels):
        labels = labels.sort_values('study_slide_id').reset_index(drop=True)
        self.ids = labels.study_slide_id.tolist()
        self.y = labels.label.to_numpy(dtype=int)
        self.pos, self.neg = np.flatnonzero(self.y == 1), np.flatnonzero(self.y == 0)
        self.npos, self.nneg = len(self.pos), len(self.neg)
        rng = np.random.default_rng(BOOTSEED)
        self.wp = rng.multinomial(self.npos, np.full(self.npos, 1/self.npos), size=BOOTSTRAPS).astype(float)
        self.wn = rng.multinomial(self.nneg, np.full(self.nneg, 1/self.nneg), size=BOOTSTRAPS).astype(float)

    def evaluate(self, g):
        g = g.set_index('study_slide_id').loc[self.ids]
        assert np.array_equal(self.y, g.label.to_numpy(dtype=int))
        p = g.score.to_numpy(dtype=float)
        delta = p[self.pos, None] - p[None, self.neg]
        kernel = (delta > 0).astype(float) + .5*(delta == 0)
        point = float(kernel.mean())
        assert abs(point-roc_auc_score(self.y, p)) < 1e-12
        values = np.sum((self.wp @ kernel) * self.wn, axis=1)/(self.npos*self.nneg)
        return point, values


def summarize(df, groups):
    records = []
    for keys, g in df.groupby(groups, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        r = dict(zip(groups, keys))
        r.update(n_seeds=len(g), seeds=','.join(str(int(s)) for s in sorted(g.seed)),
                 n_slides=int(g.n.iloc[0]))
        for name in MEASURES:
            r[name+'_mean'] = float(g[name].mean())
            r[name+'_sd'] = float(g[name].std(ddof=1)) if len(g) > 1 else 0.
        records.append(r)
    return pd.DataFrame(records)


def aggregate(pred, variants, seeds, mode):
    rows = pred[pred.variant.isin(variants) & pred.seed.isin(seeds)].copy()
    count = rows.groupby(['variant', 'scope', 'study_slide_id']).seed.nunique()
    assert count.eq(len(seeds)).all(), (mode, count.value_counts().to_dict())
    assert not rows.duplicated(['variant', 'seed', 'scope', 'study_slide_id']).any()
    result = rows.groupby(['variant', 'scope', 'study_slide_id', 'label'], as_index=False).score.mean()
    result['mode'] = mode
    result['n_training_seeds'] = len(seeds)
    return result


def group_metrics(pred, keys):
    rec = []
    for key, g in pred.groupby(keys, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        rec.append({**dict(zip(keys, key)), **metric(g)})
    return pd.DataFrame(rec)


def style():
    matplotlib.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
        'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 8, 'xtick.labelsize': 7,
        'ytick.labelsize': 7.5, 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.linewidth': .65, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
        'legend.frameon': False, 'legend.fontsize': 7, 'savefig.facecolor': 'white'})


def savefig(fig, stem):
    for suffix in ['pdf', 'svg', 'png']:
        fig.savefig(FIG / f'{stem}.{suffix}', dpi=450, bbox_inches='tight', pad_inches=.07)
    plt.close(fig)


def fig_screen(screen, winner):
    fig, axes = plt.subplots(1, 2, figsize=(7.7, 4.7), sharey=True)
    min_bar = screen.groupby(['variant', 'scope']).auroc.agg(['mean', 'std'])
    lower = max(0., min(.49, screen.auroc.min()-.02, (min_bar['mean']-min_bar['std']).min()-.02))
    for ax, scope, letter in zip(axes, ['internal', 'external'], ['a', 'b']):
        for i, v in enumerate(VARIANTS):
            values = screen[(screen.scope == scope) & (screen.variant == v)].sort_values('seed').auroc.to_numpy()
            color = ORANGE if v == winner else BLUE if v == BASELINE else GREY
            ax.axhline(i, color='#EFF1F3', linewidth=.5, zorder=0)
            ax.errorbar(values.mean(), i, xerr=values.std(ddof=1), fmt='D', color=color,
                        markersize=4, capsize=2.5, linewidth=1.2, zorder=3)
            ax.scatter(values, i + np.array([-.15, 0, .15]), s=14, color=color,
                       edgecolor='white', linewidth=.3, alpha=.9, zorder=4)
        ax.axvline(.5, color='#BBBBBB', linestyle='--', linewidth=.8)
        ax.set_xlim(lower, 1.015)
        ax.set_xticks([.5, .6, .7, .8, .9, 1.0])
        ax.set_xlabel('AUROC')
        n = 49 if scope == 'internal' else 168
        ax.set_title(f'{letter}   {scope.capitalize()} evaluation (n = {n})', loc='left', pad=12)
        ax.set_yticks(range(len(VARIANTS)))
        ax.set_yticklabels([LABELS[v] + (' *' if v == winner else '') for v in VARIANTS])
        ax.invert_yaxis()
    # Shared y-axis gets inverted only once.
    axes[0].set_ylim(len(VARIANTS)-.5, -.5)
    fig.subplots_adjust(left=.3, right=.98, bottom=.16, top=.91, wspace=.14)
    fig.text(.3, .025, 'Dots: seeds 17, 29, 43. Diamonds / bars: mean ± sample SD.\n'
             '* Candidate locked using internal AUROC; identical reference patch bags.', fontsize=7)
    savefig(fig, 'Figure_A1_all_configurations')


def fig_stability(run, five, ensembles, winner):
    fig, axes = plt.subplots(2, 2, figsize=(7.7, 6.4))
    low = max(0., min(.45, run.auroc.min()-.04, five.auroc.min()-.04, ensembles.auroc.min()-.04))
    plotdata = []
    modes = [('three_seed_mean_reference', 'Mean across 3 seeds | 1 bag'),
             ('five_seed_mean_reference', 'Mean across 5 seeds | 1 bag'),
             ('five_seed_mean_five_bag', 'Mean across 5 seeds | 5 bags'),
             ('three_seed_reference', '3-seed ensemble | 1 bag'),
             ('five_seed_reference', '5-seed ensemble | 1 bag'),
             ('five_seed_five_bag', '5-seed ensemble | 5 bags')]
    for col, scope in enumerate(['internal', 'external']):
        ax = axes[0, col]
        for v, c in [(BASELINE, BLUE), (winner, ORANGE)]:
            g = run[(run.variant == v) & (run.scope == scope)].set_index('seed').loc[ALL]
            ax.plot(range(5), g.auroc, '-o', color=c, markersize=4, linewidth=1,
                    label=LABELS[v])
        ax.axvspan(2.5, 4.5, color='#E9EEF3', alpha=.65, zorder=0)
        ax.axvline(2.5, color='#AAAAAA', linestyle='--', linewidth=.75)
        ax.set_xticks(range(5), ALL)
        ax.set_xlim(-.3, 4.4)
        ax.set_ylim(low, 1.015)
        ax.set_ylabel('Reference-bag AUROC')
        ax.set_xlabel('Training seed')
        ax.set_title(f'{"ab"[col]}   {scope.capitalize()}: added seeds 59 and 71', loc='left')
        ax.legend(loc='lower left', fontsize=6.5)
        ax2 = axes[1, col]
        for v, color, offset in [(BASELINE, BLUE, -.10), (winner, ORANGE, .10)]:
            for i, (mode, label) in enumerate(modes):
                if mode.startswith('three_seed_mean'):
                    g = run[(run.variant == v) & (run.scope == scope) & run.seed.isin(SCREEN)]
                    value, sd, category = g.auroc.mean(), g.auroc.std(ddof=1), 'mean_AUROC_across_models'
                elif mode.startswith('five_seed_mean_reference'):
                    g = run[(run.variant == v) & (run.scope == scope) & run.seed.isin(ALL)]
                    value, sd, category = g.auroc.mean(), g.auroc.std(ddof=1), 'mean_AUROC_across_models'
                elif mode.startswith('five_seed_mean_five'):
                    g = five[(five.variant == v) & (five.scope == scope)]
                    value, sd, category = g.auroc.mean(), g.auroc.std(ddof=1), 'mean_AUROC_across_models'
                else:
                    g = ensembles[(ensembles.variant == v) & (ensembles.scope == scope) & (ensembles['mode'] == mode)]
                    assert len(g) == 1
                    value, sd, category = float(g.auroc.iloc[0]), 0., 'AUROC_of_probability_ensemble'
                marker = 'o' if i < 3 else 'D'
                ax2.errorbar(value, i+offset, xerr=sd if i < 3 else None, fmt=marker,
                             color=color, capsize=2, markersize=4, linewidth=.9)
                plotdata.append({'variant': v, 'scope': scope, 'display_mode': mode,
                                 'auroc': value, 'seed_sd': sd, 'estimand': category})
        ax2.set_yticks(range(len(modes)), [x[1] for x in modes])
        ax2.invert_yaxis()
        ax2.axhline(2.5, color='#BBBBBB', linestyle='--', linewidth=.7)
        ax2.set_xlim(low, 1.015)
        ax2.set_xlabel('AUROC')
        ax2.set_title(f'{"cd"[col]}   {scope.capitalize()}: aggregation effects', loc='left', pad=9)
        ax2.grid(axis='x', color='#EEEEEE', linewidth=.5)
    # Avoid repeating the long mode labels at the center of the layout.
    axes[1, 1].set_yticklabels([])
    fig.subplots_adjust(left=.29, right=.985, bottom=.105, top=.94, hspace=.43, wspace=.30)
    fig.text(.29, .018, 'Circles / bars: mean AUROC ± SD across models. Diamonds: AUROC after averaging predictions.\n'
             'Five-bag means average five new patch samples; the reference bag is separate.', fontsize=6.7)
    pd.DataFrame(plotdata).to_csv(ANA/'figure_A2_source_data.csv', index=False)
    savefig(fig, 'Figure_A2_stability_and_aggregation')


def fig_roc(ensemble_predictions, ensemble_metrics, winner):
    chosen = ensemble_predictions[ensemble_predictions['mode'].eq('five_seed_five_bag')]
    fig = plt.figure(figsize=(7.7, 5.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.35, 1.0], hspace=.60, wspace=.45)
    rocs = []
    for col, scope in enumerate(['internal', 'external']):
        ax = fig.add_subplot(gs[0, col*2:col*2+2])
        for v, color in [(BASELINE, BLUE), (winner, ORANGE)]:
            g = chosen[(chosen.variant == v) & (chosen.scope == scope)]
            fpr, tpr, thresholds = roc_curve(g.label, g.score)
            auc = roc_auc_score(g.label, g.score)
            ax.plot(fpr, tpr, color=color, linewidth=1.6, label=f'{LABELS[v]}: {auc:.3f}')
            rocs.extend({'variant': v, 'scope': scope, 'fpr': x, 'tpr': y,
                         'threshold': None if not np.isfinite(t) else t}
                        for x, y, t in zip(fpr, tpr, thresholds))
        ax.plot([0, 1], [0, 1], '--', color='#BBBBBB', linewidth=.8)
        ax.set(xlim=(-.015, 1.015), ylim=(-.015, 1.035), xlabel='False-positive rate', ylabel='True-positive rate')
        ax.set_title(f'{"ab"[col]}   {scope.capitalize()} evaluation', loc='left')
        ax.legend(loc='lower right', fontsize=6.5)
    for col, (scope, v) in enumerate([(s, v) for s in ['internal', 'external'] for v in [BASELINE, winner]]):
        ax = fig.add_subplot(gs[1, col])
        g = chosen[(chosen.variant == v) & (chosen.scope == scope)]
        cm = confusion_matrix(g.label, g.score >= .5, labels=[0, 1])
        scaled = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(scaled, cmap='Blues' if v == BASELINE else 'Oranges', vmin=0, vmax=1)
        for r in range(2):
            for c in range(2):
                ax.text(c, r, f'{cm[r,c]}\n({scaled[r,c]*100:.1f}%)', ha='center', va='center',
                        fontsize=7.3, color='white' if scaled[r,c] > .65 else '#252525')
        ax.set_xticks([0, 1], ['SS', 'MALT'])
        ax.set_yticks([0, 1], ['SS', 'MALT'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Reference' if col in [0, 2] else '')
        ax.set_title(f'{"cdef"[col]}   {scope.capitalize()}\n'+('Hybrid control' if v == BASELINE else 'Locked candidate'),
                     loc='left', fontsize=8, pad=7)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.subplots_adjust(left=.09, right=.985, bottom=.105, top=.93)
    fig.text(.09, .015, 'Mode: equal probability mean over all 5 seeds × 5 new patch bags. Threshold = 0.5.\n'
             'Confusion cells show counts and within-reference-class percentages. Candidate selected internally.', fontsize=6.8)
    pd.DataFrame(rocs).to_csv(ANA/'figure_A3_ROC_source_data.csv', index=False)
    savefig(fig, 'Figure_A3_locked_ensemble_ROC_confusion')


def main():
    runtime = json.loads((OUT/'runtime.json').read_text())
    assert runtime.get('complete') is True and runtime.get('pilot') is False, 'Full training/evaluation must finish first.'
    lock = json.loads((OUT/'selection_lock.json').read_text())
    lock_hash = sha(OUT/'selection_lock.json')
    source_names = ['all_predictions.csv', 'run_metrics.csv', 'selection_lock.json',
                    'five_bag_mean_predictions.csv', 'repeated_bag_predictions.csv', 'runtime.json']
    initial_source_hashes = {name: sha(OUT/name) for name in source_names}
    winner = lock['winner']
    assert winner in VARIANTS and winner not in [BASELINE, 'attention_control']
    ANA.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    raw = pd.read_csv(OUT/'all_predictions.csv')
    stored = pd.read_csv(OUT/'run_metrics.csv').rename(columns={'brier': 'brier_score'})
    repeated = pd.read_csv(OUT/'repeated_bag_predictions.csv')
    five = pd.read_csv(OUT/'five_bag_mean_predictions.csv')
    assert len(stored) == 76
    assert len(raw) == (34+4)*217
    assert len(repeated) == 2*5*5*217
    assert len(five) == 2*5*217
    assert not raw.duplicated(['variant', 'seed', 'scope', 'inference', 'study_slide_id']).any()
    run = group_metrics(raw, ['variant', 'seed', 'stage', 'scope', 'inference'])
    merged = run.merge(stored, on=['variant', 'seed', 'stage', 'scope', 'inference'], suffixes=('_check', '_saved'), validate='one_to_one')
    checks = {}
    for name in MEASURES + ['tn', 'fp', 'fn', 'tp']:
        checks[name] = float(np.max(np.abs(merged[name+'_check']-merged[name+'_saved'])))
        assert checks[name] < 1e-10, (name, checks[name])
    run.to_csv(ANA/'run_metrics_verified.csv', index=False)
    screen = run[run.stage.eq('screen')].copy()
    assert len(screen) == 60
    screen.to_csv(ANA/'screen_per_seed.csv', index=False)
    screen_summary = summarize(screen, ['variant', 'scope'])
    screen_summary.to_csv(ANA/'screen_summary.csv', index=False)
    confirm = run[run.variant.isin([BASELINE, winner]) & run.seed.isin(ALL)].copy()
    confirm.to_csv(ANA/'selected_control_all_five_per_seed.csv', index=False)
    confirm[confirm.seed.isin(EXTRA)].to_csv(ANA/'confirmation_additional_seeds.csv', index=False)
    summaries = []
    for tag, seeds in [('screen_three', SCREEN), ('additional_two', EXTRA), ('all_five', ALL)]:
        x = summarize(confirm[confirm.seed.isin(seeds)], ['variant', 'scope'])
        x['seed_group'] = tag
        summaries.append(x)
    confirmation_summary = pd.concat(summaries, ignore_index=True)
    confirmation_summary.to_csv(ANA/'confirmation_summary.csv', index=False)
    linear = run[run.stage.eq('deterministic_control')].copy()
    linear.to_csv(ANA/'linear_controls.csv', index=False)
    neural_pred = raw[raw.stage.isin(['screen', 'confirmation'])].copy()
    assert neural_pred.inference.eq('reference_bag').all()
    ens = [aggregate(neural_pred, VARIANTS, SCREEN, 'three_seed_reference'),
           aggregate(neural_pred, [BASELINE, winner], EXTRA, 'additional_two_seed_reference'),
           aggregate(neural_pred, [BASELINE, winner], ALL, 'five_seed_reference')]
    five_check = repeated.groupby(['variant', 'seed', 'scope', 'study_slide_id', 'label'], as_index=False).score.mean()
    joined = five.merge(five_check, on=['variant', 'seed', 'scope', 'study_slide_id', 'label'], suffixes=('_saved', '_check'), validate='one_to_one')
    assert np.max(np.abs(joined.score_saved-joined.score_check)) < 1e-12
    five_metrics = group_metrics(five, ['variant', 'seed', 'scope'])
    five_metrics.to_csv(ANA/'five_bag_per_seed_metrics.csv', index=False)
    five_metrics_summaries = []
    for tag, seeds in [('screen_three', SCREEN), ('additional_two', EXTRA), ('all_five', ALL)]:
        g = summarize(five_metrics[five_metrics.seed.isin(seeds)], ['variant', 'scope'])
        g['seed_group'] = tag
        five_metrics_summaries.append(g)
    pd.concat(five_metrics_summaries).to_csv(ANA/'five_bag_seed_summary.csv', index=False)
    ens += [aggregate(five, [BASELINE, winner], SCREEN, 'three_seed_five_bag'),
            aggregate(five, [BASELINE, winner], ALL, 'five_seed_five_bag')]
    ensembles = pd.concat(ens, ignore_index=True)
    ensembles.to_csv(ANA/'ensemble_predictions.csv', index=False)
    e_metrics = group_metrics(ensembles, ['variant', 'scope', 'mode'])
    # One class-stratified resampling plan per cohort, shared for every model.
    boot = {}
    distributions = {}
    for scope in ['internal', 'external']:
        lab = raw[raw.scope.eq(scope)][['study_slide_id', 'label']].drop_duplicates()
        assert not lab.study_slide_id.duplicated().any()
        assert len(lab) == (49 if scope == 'internal' else 168)
        boot[scope] = PairedBootstrap(lab)
    for ix, r in e_metrics.iterrows():
        g = ensembles[(ensembles.variant == r.variant) & (ensembles.scope == r.scope) & (ensembles['mode'] == r['mode'])]
        point, values = boot[r.scope].evaluate(g)
        distributions[(r.variant, r.scope, r['mode'])] = (point, values)
        e_metrics.loc[ix, 'auroc_conditional_bootstrap_low'], e_metrics.loc[ix, 'auroc_conditional_bootstrap_high'] = np.quantile(values, [.025, .975])
    e_metrics.to_csv(ANA/'ensemble_metrics.csv', index=False)
    history = []
    for scope, filename, scorecol in [('internal', 'internal_predictions_hybrid_evidence_mil.csv', 'probability'),
                                       ('external', 'external_final_predictions.csv', 'malt_probability')]:
        h = pd.read_csv(OLD/'supporting_data'/filename)
        h = h.rename(columns={scorecol: 'score'})[['study_slide_id', 'label', 'score']]
        h['scope'] = scope
        h['variant'] = 'historical_primary_hybrid'
        h['mode'] = 'historical_saved_reference'
        history.append(h)
        point, values = boot[scope].evaluate(h)
        distributions[('historical_primary_hybrid', scope, 'historical_saved_reference')] = (point, values)
        assert abs(point-({'internal': .9598662207357859, 'external': .8721217105263158}[scope])) < 1e-10
    history = pd.concat(history, ignore_index=True)
    history.to_csv(ANA/'historical_primary_predictions.csv', index=False)
    h_metrics = group_metrics(history, ['variant', 'scope', 'mode'])
    for ix, r in h_metrics.iterrows():
        _, values = distributions[(r.variant, r.scope, r['mode'])]
        h_metrics.loc[ix, 'auroc_conditional_bootstrap_low'], h_metrics.loc[ix, 'auroc_conditional_bootstrap_high'] = np.quantile(values, [.025, .975])
    h_metrics.to_csv(ANA/'historical_primary_metrics.csv', index=False)
    comparisons = []
    modes = ['three_seed_reference', 'additional_two_seed_reference', 'five_seed_reference', 'three_seed_five_bag', 'five_seed_five_bag']
    for scope in ['internal', 'external']:
        for mode in modes:
            for comparator, comparemode in [(BASELINE, mode), ('historical_primary_hybrid', 'historical_saved_reference')]:
                pa, a = distributions[(winner, scope, mode)]
                pb, b = distributions[(comparator, scope, comparemode)]
                lo, hi = np.quantile(a-b, [.025, .975])
                comparisons.append({'scope': scope, 'candidate': winner, 'candidate_mode': mode,
                                    'comparator': comparator, 'comparator_mode': comparemode,
                                    'candidate_auroc': pa, 'comparator_auroc': pb, 'difference': pa-pb,
                                    'conditional_paired_ci_low': lo, 'conditional_paired_ci_high': hi,
                                    'n_bootstrap': BOOTSTRAPS, 'bootstrap_seed': BOOTSEED,
                                    'selection_adjusted': False})
    pairs = pd.DataFrame(comparisons)
    pairs.to_csv(ANA/'paired_auc_comparisons.csv', index=False)
    # Pair seed-level differences without treating seeds as extra clinical data.
    c = confirm[confirm.variant.eq(winner)]
    b = confirm[confirm.variant.eq(BASELINE)]
    paired_seed = c.merge(b, on=['seed', 'scope'], suffixes=('_candidate', '_control'), validate='one_to_one')
    for m in MEASURES:
        paired_seed[m+'_difference'] = paired_seed[m+'_candidate']-paired_seed[m+'_control']
    paired_seed.to_csv(ANA/'selected_control_paired_seed_differences.csv', index=False)
    stability = repeated.groupby(['variant', 'seed', 'scope', 'study_slide_id', 'label']).score.agg(['mean', 'std', 'min', 'max']).reset_index()
    stability['crosses_0_5'] = (stability['min'] < .5) & (stability['max'] >= .5)
    stability.to_csv(ANA/'five_bag_per_slide_stability.csv', index=False)
    stability_summ = stability.groupby(['variant', 'seed', 'scope']).agg(median_probability_sd=('std', 'median'),
                    mean_probability_sd=('std', 'mean'), crossing_slides=('crosses_0_5', 'sum'), n_slides=('study_slide_id', 'size')).reset_index()
    stability_summ.to_csv(ANA/'five_bag_stability_per_seed.csv', index=False)
    internal = paired_seed[paired_seed.scope.eq('internal') & paired_seed.seed.isin(SCREEN)]
    gain = float(internal.auroc_difference.mean())
    nondecrease = int((internal.auroc_difference >= -1e-12).sum())
    bac_delta = float(internal.balanced_accuracy_difference.mean())
    success = {'scope': 'internal', 'seed_group': '17,29,43', 'mean_auroc_difference': gain,
               'minimum_mean_auroc_improvement': .01, 'nondecreasing_paired_seeds': nondecrease,
               'required_nondecreasing_paired_seeds': 2,
               'engineering_target_met': bool(gain >= .01-1e-12 and nondecrease >= 2),
               'mean_balanced_accuracy_difference': bac_delta,
               'balanced_accuracy_drop_flag': bool(bac_delta < -.02),
               'balanced_accuracy_drop_flag_threshold': -.02}
    summaries = {}
    for scope in ['internal', 'external']:
        summaries[scope] = {}
        for tag, seeds in [('screen_three', SCREEN), ('additional_two', EXTRA), ('all_five', ALL)]:
            g = paired_seed[paired_seed.scope.eq(scope) & paired_seed.seed.isin(seeds)]
            summaries[scope][tag] = {'mean_auroc_difference': float(g.auroc_difference.mean()),
                                    'nondecreasing_paired_seeds': int((g.auroc_difference >= -1e-12).sum()),
                                    'n_seeds': len(g), 'mean_balanced_accuracy_difference': float(g.balanced_accuracy_difference.mean()),
                                    'balanced_accuracy_drop_flag': bool(g.balanced_accuracy_difference.mean() < -.02)}
    style()
    fig_screen(screen, winner)
    fig_stability(confirm, five_metrics, e_metrics, winner)
    fig_roc(ensembles, e_metrics, winner)
    contract = {'backend': 'Python / matplotlib', 'archetype': 'quantitative grid',
                'exports': ['editable PDF', 'editable SVG', '450dpi PNG'],
                'A1': 'All ten prespecified configurations: individual seeds and mean ± sample SD; candidate selection internal only.',
                'A2': 'Added seeds 59/71 and separate model-mean versus ensemble-AUROC estimands; probability bag/seed averages.',
                'A3': 'Locked candidate and Hybrid control, 5 seeds × 5 new bags probability ensemble; ROC and threshold0.5 confusion.',
                'review_risks': ['No external reselection', 'No best-seed reporting', 'No seed SD as clinical CI',
                                 'No model-mean AUROC confused with AUROC after prediction averaging',
                                 'Subspace variants already average 21 window logits within each prediction'],
                'bootstrap_interpretation': 'Conditional on fitted predictions; does not correct internal method/epoch selection or prior external cohort use.'}
    writejson(FIG/'figure_contract.json', contract)
    summary = {'status': 'PASS', 'selected_variant': winner, 'selected_display_name': LABELS[winner],
        'selection_lock_sha256': lock_hash, 'selection_was_external': False,
        'selection': {'scope': lock['selection_scope'], 'frozen_utc': lock['frozen_utc'],
                      'ranking': lock['ranking'], 'screen_seeds': SCREEN, 'additional_seeds': EXTRA},
        'engineering_success': success, 'paired_seed_summary': summaries,
        'screen_summary': clean_records(screen_summary), 'confirmation_summary': clean_records(confirmation_summary),
        'ensemble_metrics': clean_records(e_metrics), 'paired_auc_comparisons': clean_records(pairs),
        'linear_controls': clean_records(linear), 'historical_primary_metrics': clean_records(h_metrics),
        'bootstrap': {'n_resamples': BOOTSTRAPS, 'seed': BOOTSEED, 'method': 'paired class-stratified count-weighted AUC ranking kernel',
                      'conditional': True, 'selection_adjusted': False},
        'checks': {'metric_recalculation_maximum_errors': checks, 'neural_run_count': 34, 'metric_rows': len(stored),
                   'all_prediction_rows': len(raw), 'repeated_bag_rows': len(repeated), 'five_bag_mean_rows': len(five),
                   'source_files_unmodified': True},
        'source_sha256': initial_source_hashes,
        'analysis_script_sha256': sha(__file__), 'runtime': runtime}
    assert sha(OUT/'selection_lock.json') == lock_hash
    assert {name: sha(OUT/name) for name in source_names} == initial_source_hashes
    writejson(ANA/'summary.json', summary)
    print(json.dumps({'status': 'PASS', 'winner': winner, 'engineering_success': success,
                      'n_metric_rows': len(run), 'n_ensemble_metrics': len(e_metrics),
                      'analysis_directory': str(ANA)}, indent=2))


if __name__ == '__main__':
    main()
