"""CPU-only, reproducible checks of archived predictions, model weights and ROI masks.

No inference, training, calibration fitting, case replacement or clinical annotation.
Published outputs use study codes. Private source resolution is read locally only.
Run: MPLCONFIGDIR=/tmp/trident_v6_mpl CUDA_VISIBLE_DEVICES='' <python> this_file.py
"""
from pathlib import Path
import hashlib
import json
import os
import platform
import sys

os.environ.setdefault('MPLCONFIGDIR', '/tmp/trident_v6_mpl')
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
import pandas as pd
import scipy
from scipy.stats import binomtest
import shapely
from shapely.geometry import box, shape
from shapely.ops import unary_union
import openslide
from PIL import Image
import torch

OUT = Path(__file__).resolve().parent
PROJECT = OUT.parents[2]
V5 = PROJECT / 'writing/draft_v5_review_revision'
DATA = V5 / 'supporting_data'
EXPERIMENTS = PROJECT / 'scripts/result/ss_experiments/malt_vs_non_malt'
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 20261104
inputs = []


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def csv(path, label=None):
    inputs.append({'input': label or str(path.relative_to(V5)), 'sha256': sha(path)})
    return pd.read_csv(path)


def save_csv(frame, name):
    frame.to_csv(OUT / name, index=False)


def save_json(obj, name):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def bootstrap_brier(y, score, seed):
    """Conditional-on-class-count, slide-level percentile bootstrap of squared loss."""
    losses = (score - y) ** 2
    rng = np.random.default_rng(seed)
    classes = [np.flatnonzero(y == c) for c in [0, 1]]
    sampled = np.concatenate([rng.choice(ix, size=(BOOTSTRAP_DRAWS, len(ix)), replace=True)
                              for ix in classes], axis=1)
    means = losses[sampled].mean(axis=1)
    return float(losses.mean()), *[float(v) for v in np.quantile(means, [.025, .975])]


def predictions():
    names = ['hybrid_evidence_mil', 'abmil', 'clam_sb', 'rrtmil', 'hybrid_clam_rrt_ensemble']
    scores = None
    for name in names:
        df = csv(DATA / f'internal_predictions_{name}.csv')
        assert len(df) == 49 and df.study_slide_id.is_unique
        assert np.array_equal((df.probability >= .5).astype(int), df.prediction_at_0_5)
        vals = df[['study_slide_id', 'label', 'class_name', 'probability']].rename(columns={'probability': name})
        if scores is None:
            scores = vals
        else:
            scores = scores.merge(vals, on=['study_slide_id', 'label', 'class_name'], validate='one_to_one')
    assert len(scores) == 49
    scores = scores.sort_values('study_slide_id').reset_index(drop=True)
    for name in names:
        scores[f'{name}_correct'] = (scores[name] >= .5).astype(int).eq(scores.label)
    members = ['hybrid_evidence_mil', 'clam_sb', 'rrtmil']
    assert np.allclose(scores[members].mean(axis=1), scores.hybrid_clam_rrt_ensemble, atol=1e-7)
    scores['all_three_members_wrong'] = ~scores[[f'{x}_correct' for x in members]].any(axis=1)
    save_csv(scores, 'internal_scores_and_errors.csv')

    paired, summaries = [], []
    reference = 'hybrid_evidence_mil'
    for candidate in ['abmil', 'hybrid_clam_rrt_ensemble']:
        for row in scores.to_dict(orient='records'):
            rc, cc = row[f'{reference}_correct'], row[f'{candidate}_correct']
            transition = ('both_correct' if rc and cc else 'both_wrong' if not rc and not cc
                          else 'corrected_by_candidate' if cc else 'new_error_in_candidate')
            paired.append({'reference_model': reference, 'candidate_model': candidate,
                           'study_slide_id': row['study_slide_id'], 'label': row['label'],
                           'class_name': row['class_name'], 'reference_score': row[reference],
                           'candidate_score': row[candidate], 'reference_correct': rc,
                           'candidate_correct': cc, 'transition': transition})
        subset = pd.DataFrame(paired).query('candidate_model == @candidate')
        counts = subset.transition.value_counts()
        corrected = int(counts.get('corrected_by_candidate', 0))
        new = int(counts.get('new_error_in_candidate', 0))
        summaries.append({'reference_model': reference, 'candidate_model': candidate,
                          'threshold': .5, 'n': len(subset),
                          **{k: int(counts.get(k, 0)) for k in ['both_correct', 'both_wrong', 'corrected_by_candidate', 'new_error_in_candidate']},
                          'net_corrected': corrected - new,
                          'exact_mcnemar_two_sided_p': float(binomtest(corrected, corrected + new, .5).pvalue) if corrected + new else 1.0,
                          'interpretation': 'descriptive paired threshold comparison; no superiority claim'})
    save_csv(pd.DataFrame(paired), 'paired_error_exchange.csv')
    save_csv(pd.DataFrame(summaries), 'paired_error_summary.csv')

    brier = []
    for j, name in enumerate(names):
        point, low, high = bootstrap_brier(scores.label.to_numpy(), scores[name].to_numpy(), BOOTSTRAP_SEED + j)
        brier.append({'cohort': 'internal', 'model': name, 'n': 49, 'n_ss': 26, 'n_malt': 23,
                      'brier_score': point, 'ci_low': low, 'ci_high': high,
                      'bootstrap_seed': BOOTSTRAP_SEED + j, 'bootstrap_draws': BOOTSTRAP_DRAWS})
    ext = csv(DATA / 'external_final_predictions.csv')
    assert len(ext) == 168 and ext.study_slide_id.is_unique
    point, low, high = bootstrap_brier(ext.label.to_numpy(), ext.malt_probability.to_numpy(), BOOTSTRAP_SEED + len(names))
    brier.append({'cohort': 'external', 'model': 'hybrid_evidence_mil', 'n': 168, 'n_ss': 16, 'n_malt': 152,
                  'brier_score': point, 'ci_low': low, 'ci_high': high,
                  'bootstrap_seed': BOOTSTRAP_SEED + len(names), 'bootstrap_draws': BOOTSTRAP_DRAWS})
    save_csv(pd.DataFrame(brier), 'brier_descriptive.csv')
    return {'paired_error_summary': summaries, 'brier': brier,
            'ensemble_all_three_members_wrong': scores.loc[scores.all_three_members_wrong, 'study_slide_id'].tolist()}


def roi_qc():
    locations_path = V5 / 'source_data/roi_locations.json'
    inputs.append({'input': 'source_data/roi_locations.json', 'sha256': sha(locations_path)})
    locations = json.loads(locations_path.read_text())
    mapping = pd.read_csv(V5 / 'qa/slide_code_mapping_PRIVATE.csv')
    cohort = pd.read_csv(EXPERIMENTS / 'uni_v2_hybrid_evidence_mil/cohort_split.csv')
    rows, sources = [], []
    for loc in locations:
        code = loc['study_slide_id']
        sid = mapping.loc[mapping.study_slide_id.eq(code), 'original_slide_id'].iloc[0]
        row = cohort.loc[cohort.slide_id.eq(sid)].iloc[0]
        mask_paths = [p for p in (PROJECT / 'scripts/result').glob('*/contours_geojson/*.geojson') if p.stem == sid]
        assert len(mask_paths) == 1
        mask_path = mask_paths[0]
        document = json.loads(mask_path.read_text())
        polygons = [shape(f['geometry']) for f in document['features']]
        assert all(g.is_valid for g in polygons), 'Do not silently repair invalid source geometry'
        tissue = unary_union(polygons)
        config_path = mask_path.parents[1] / '_config_segmentation.json'
        cfg = json.loads(config_path.read_text())
        sources.append({'study_slide_id': code, 'mask_sha256': sha(mask_path),
                        'segmentation_config_sha256': sha(config_path),
                        'segmentation_magnification': cfg['seg_mag'],
                        'holes_are_tissue': cfg['holes_are_tissue'],
                        'polygon_count': len(polygons), 'all_polygons_valid': True})
        slide = openslide.OpenSlide(row.slide_path)
        historical_thumbnail_path = mask_path.parents[1] / 'thumbnails' / (sid + '.jpg')
        historical_thumbnail_size = Image.open(historical_thumbnail_path).size
        sources[-1].update({'stored_feature_level0_width': loc['level0_width'],
                            'stored_feature_level0_height': loc['level0_height'],
                            'current_openslide_level0_width': slide.dimensions[0],
                            'current_openslide_level0_height': slide.dimensions[1],
                            'historical_thumbnail_width': historical_thumbnail_size[0],
                            'historical_thumbnail_height': historical_thumbnail_size[1],
                            'extent_matches_current_reader': slide.dimensions == (loc['level0_width'], loc['level0_height']),
                            'historical_thumbnail_sha256': sha(historical_thumbnail_path)})
        case_num = int(loc['publication_case'].split()[-1])
        for direction, neutral in [('positive', 'highest_margin'), ('negative', 'lowest_margin')]:
            x, y = int(loc[direction + '_x']), int(loc[direction + '_y'])
            width = loc['patch_width_level0']
            assert 0 <= x < x + width <= slide.dimensions[0] and 0 <= y < y + width <= slide.dimensions[1]
            window = box(x, y, x + width, y + width)
            coverage = tissue.intersection(window).area / window.area
            im = slide.read_region((x, y), 0, (width, width)).convert('RGB')
            rgb = np.asarray(im)
            original = V5 / 'assets' / f'case_{case_num:02d}_{direction}.png'
            resized = np.asarray(im.resize((220, 220), Image.Resampling.LANCZOS))
            exact = np.array_equal(resized, np.asarray(Image.open(original).convert('RGB')))
            assert exact, 'Source ROI pixels differ from existing figure thumbnail'
            rows.append({'publication_case': loc['publication_case'], 'study_slide_id': code,
                         'margin_rank': neutral, 'roi_x': x, 'roi_y': y,
                         'patch_width_level0': width, 'roi_area_level0_pixels': width ** 2,
                         'mask_coverage_fraction': coverage,
                         'all_rgb_ge_230_fraction': float((rgb.min(axis=2) >= 230).mean()),
                         'all_rgb_ge_240_fraction': float((rgb.min(axis=2) >= 240).mean()),
                         'all_rgb_ge_250_fraction': float((rgb.min(axis=2) >= 250).mean()),
                         'source_mask_holes_are_tissue': cfg['holes_are_tissue'],
                         'stored_feature_level0_height': loc['level0_height'],
                         'current_openslide_level0_height': slide.dimensions[1],
                         'original_thumbnail_pixels_verified': exact,
                         'roi_rgb_sha256': hashlib.sha256(rgb.tobytes()).hexdigest(),
                         'thumbnail_sha256': sha(original),
                         'pathologist_tissue_or_artifact_confirmation': 'PENDING_HOSPITAL_RECORDS'})
        slide.close()
    save_csv(pd.DataFrame(rows), 'roi_mask_coverage.csv')
    save_json({'sources': sources,
               'coordinate_system': 'level-0 slide pixels; no geospatial reprojection',
               'coordinate_basis': 'WSI.segment_tissue rescales mask contours by 1/mpp_reduction_factor before GeoJSON export; EPSG:3857 is a code placeholder, not a real geospatial CRS',
               'mask_fraction_definition': 'area(union of original automatic segmentation polygons intersect ROI square) / ROI square area',
               'rgb_proxy_definition': 'fraction of original full-resolution ROI pixels whose red, green and blue values are all at least 230, 240 or 250 on the 0-255 scale; uniform descriptive thresholds, no tuning or pathology labels',
               'limits': ['Mask coverage is not a manual tissue fraction; holes_are_tissue=true may count internal blank spaces.',
                          'High RGB intensity is a near-white pixel descriptor, not a validated background, tissue, artifact or quality classifier.',
                          'Neither metric establishes why a slide was misclassified.',
                          'All eight existing ranked ROIs were retained; no crop substitution or pixel adjustment.']}, 'roi_mask_provenance.json')
    return rows


def checkpoints():
    sys.path.insert(0, str(PROJECT / 'scripts'))
    from ss_experiment_suite import SlideClassifier
    rows, shapes = [], {}
    for name in ['hybrid_evidence_mil', 'abmil', 'clam_sb', 'rrtmil']:
        folder = EXPERIMENTS / f'uni_v2_{name}'
        path = folder / 'best_model.pth'
        state = torch.load(path, map_location='cpu', weights_only=True)
        model = SlideClassifier(name, 1536, 256, .3).cpu()
        model.load_state_dict(state, strict=True)
        config = json.loads((folder / 'config.json').read_text())
        summary = json.loads((folder / 'summary.json').read_text())
        rows.append({'model': name, 'parameter_count': sum(p.numel() for p in model.parameters()),
                     'buffer_elements': sum(p.numel() for p in model.buffers()),
                     'checkpoint_bytes': path.stat().st_size, 'checkpoint_mib': path.stat().st_size / 2**20,
                     'checkpoint_sha256': sha(path), 'checkpoint_strict_load': True,
                     'input_feature_dim': 1536, 'classifier_input_dim': model.classifier[0].in_features,
                     'classifier_hidden_dim': model.classifier[0].out_features,
                     'attention_head_dim': 512 if name == 'abmil' else 256,
                     'epochs_budget': config['epochs'], 'batch_size': config['batch_size'],
                     'training_seed': config['seed'], 'num_patches': config['num_patches'],
                     'learning_rate': config['lr'], 'weight_decay': config['weight_decay'],
                     'best_epoch': summary.get('best_epoch')})
        shapes[name] = {key: list(value.shape) for key, value in state.items()}
    save_csv(pd.DataFrame(rows), 'model_checkpoint_inventory.csv')
    save_json(shapes, 'checkpoint_tensor_shapes.json')
    return rows


if __name__ == '__main__':
    OUT.mkdir(exist_ok=True, parents=True)
    summary = predictions()
    summary['roi_qc'] = roi_qc()
    summary['checkpoints'] = checkpoints()
    save_json(summary, 'existing_evidence_summary.json')
    save_json({'inputs': inputs, 'script_sha256': sha(__file__),
               'runtime': {'python': sys.version, 'numpy': np.__version__, 'pandas': pd.__version__,
                           'scipy': scipy.__version__, 'shapely': shapely.__version__,
                           'torch': torch.__version__, 'platform': platform.platform()},
               'new_inference_runs': 0, 'new_training_runs': 0,
               'gpu_used': False, 'clinical_annotations_added': False,
               'bootstrap': '10,000 stratified slide-level draws, percentile 95% intervals; class counts fixed; archived scores fixed; does not quantify bag or training variability'},
              'analysis_provenance.json')
    print(json.dumps({'paired': summary['paired_error_summary'], 'brier': summary['brier'],
                      'roi_coverage': [{k: r[k] for k in ['publication_case', 'margin_rank', 'mask_coverage_fraction', 'all_rgb_ge_240_fraction']} for r in summary['roi_qc']],
                      'parameters': {r['model']: r['parameter_count'] for r in summary['checkpoints']}}, indent=2))
