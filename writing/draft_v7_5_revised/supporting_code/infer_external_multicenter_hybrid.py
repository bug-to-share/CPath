#!/usr/bin/env python
"""Run the trained Hybrid-Evidence MIL model on external TRIDENT features."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

ROOT = Path('/path/to/private/workspace')
sys.path.insert(0, str(ROOT / 'scripts'))
import ss_experiment_suite as suite  # noqa: E402


def stain_group(path: str, cohort: str) -> str:
    if cohort == 'MALT_HE':
        return 'H&E'
    if cohort == 'SS_BJ':
        return 'SS (slide series)'
    name = Path(path).name.lower()
    if any(x in name for x in (' he', '-he', 'qss', 'ssn', 'mxl', 'gst')):
        return 'H&E-like'
    return 'IHC / special stain'


def case_id(path: str, cohort: str) -> str:
    if cohort == 'MALT_HE':
        return Path(path).parent.name
    matches = re.findall(r'(?i)(h\d{6}|\d{6,8})', Path(path).name)
    if matches:
        return matches[0].upper()
    return Path(path).parent.name


def sample_h5(path: Path, seed: int, index: int, n_patches: int) -> tuple[torch.Tensor, torch.Tensor, int]:
    with h5py.File(path, 'r') as f:
        features = f['features'][:]
        coords = f['coords'][:]
    n = len(features)
    rng = np.random.default_rng(seed + index)
    keep = rng.choice(n, size=n_patches, replace=n < n_patches)
    return torch.from_numpy(features[keep]).float().unsqueeze(0), torch.from_numpy(coords[keep]).long().unsqueeze(0), n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--job-dir', default=str(ROOT / 'scripts/result/external_multicenter_hybrid_all'))
    ap.add_argument('--gpu', type=int, default=2)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--num-patches', type=int, default=512)
    args = ap.parse_args()

    job = Path(args.job_dir)
    manifest = pd.read_csv(job / 'external_manifest.csv')
    feat_dir = job / '20x_512px_0px_overlap/features_uni_v2'
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    model = suite.SlideClassifier('hybrid_evidence_mil', 1536, 256, 0.3).to(device)
    checkpoint = ROOT / 'scripts/result/ss_experiments/malt_vs_non_malt/uni_v2_hybrid_evidence_mil/best_model.pth'
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    records = []
    for index, row in manifest.iterrows():
        fp = feat_dir / f"{row.slide_id}.h5"
        record = row.to_dict()
        record['feature_path'] = str(fp)
        record['stain_group'] = stain_group(row.source_path, row.cohort)
        record['case_id'] = case_id(row.source_path, row.cohort)
        if not fp.exists():
            record.update({'status': 'missing_feature', 'n_available_patches': np.nan, 'malt_probability': np.nan})
            records.append(record)
            continue
        try:
            features, coords, n_available = sample_h5(fp, args.seed, index, args.num_patches)
            with torch.no_grad():
                logits = model({'features': features, 'coords': coords}).squeeze()
                probability = float(torch.sigmoid(logits).cpu())
            record.update({
                'status': 'ok', 'n_available_patches': n_available, 'malt_probability': probability,
                'prediction_threshold_0_5': int(probability >= 0.5),
                'prediction_threshold_0_448': int(probability >= 0.448),
            })
        except Exception as exc:  # preserve failed slides in the audit trail
            record.update({
                'status': f'error: {type(exc).__name__}',
                'error_message': str(exc),
                'n_available_patches': np.nan,
                'malt_probability': np.nan,
            })
        records.append(record)
        if (index + 1) % 25 == 0:
            print(f'Processed {index + 1}/{len(manifest)}')

    predictions = pd.DataFrame(records)
    predictions.to_csv(job / 'hybrid_external_slide_predictions.csv', index=False)
    valid = predictions[predictions.status.eq('ok')].copy()
    if valid.empty:
        pd.DataFrame().to_csv(job / 'hybrid_external_summary_by_cohort_stain.csv', index=False)
        pd.DataFrame().to_csv(job / 'hybrid_external_case_summary.csv', index=False)
        print(f'No successful predictions; audit trail written to {job}')
        return
    valid['correct_0_5'] = valid['prediction_threshold_0_5'].eq(valid.label)
    valid['correct_0_448'] = valid['prediction_threshold_0_448'].eq(valid.label)
    summary = valid.groupby(['cohort', 'stain_group', 'label']).agg(
        n_slides=('slide_id', 'size'),
        n_cases=('case_id', 'nunique'),
        probability_mean=('malt_probability', 'mean'),
        probability_median=('malt_probability', 'median'),
        probability_min=('malt_probability', 'min'),
        probability_max=('malt_probability', 'max'),
        accuracy_0_5=('correct_0_5', 'mean'),
        accuracy_0_448=('correct_0_448', 'mean'),
    ).reset_index()
    summary.to_csv(job / 'hybrid_external_summary_by_cohort_stain.csv', index=False)

    case_summary = valid.groupby(['cohort', 'stain_group', 'label', 'case_id']).agg(
        n_slides=('slide_id', 'size'),
        malt_probability_mean=('malt_probability', 'mean'),
        malt_probability_median=('malt_probability', 'median'),
        malt_probability_max=('malt_probability', 'max'),
        positive_fraction_0_5=('prediction_threshold_0_5', 'mean'),
        positive_fraction_0_448=('prediction_threshold_0_448', 'mean'),
    ).reset_index()
    case_summary.to_csv(job / 'hybrid_external_case_summary.csv', index=False)
    print(f'Wrote {len(valid)}/{len(predictions)} successful predictions to {job}')


if __name__ == '__main__':
    main()
