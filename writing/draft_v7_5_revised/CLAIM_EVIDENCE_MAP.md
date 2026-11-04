# v7.5 claim–evidence map

| Manuscript statement | Evidence | Scope |
|---|---|---|
| Primary discrimination | supporting_data/internal_model_metrics.csv; external_final_predictions.csv; main Tables 2–3 | Fixed historical checkpoint, WSI level; validation reuse |
| Model differences | analysis/, experiments/checkpoint_analysis/ (as archived); S4/S11/S14 | Complete implementations; paired intervals conditional on fixed predictions |
| No consistent gate benefit | experiments/controlled_training/run_metrics.csv; S15/Fig S4 | Three seeds, fixed split, unequal parameter counts |
| No substantial improvement in bounded screen | development_analyses/analysis/screen_summary.csv, confirmation_summary.csv; S16/S17 | 34 runs, selected internally, five seeds for two models; external previously examined |
| Simple controls discriminate but fixed-threshold sensitivity is limited | development_analyses/analysis/linear_controls.csv; S17 | Train-only fitting; reference-bag and full-feature inference separated |
| Mixed-material external spectrum matters | supporting_data/external_final_predictions.csv; analysis/review_numeric_audit.json; S18 | Post hoc tag exclusion, no H&E-only claim |
| Sampling can change assigned class | Original repeated-bag files; S10/Fig 6 | Technical repeats, fixed model |
| Regions are traceable | ROI audit, S6/S12, Fig 5 | Image provenance only; pathology annotation pending |
| Reader observations are provisional | supporting_data/reader_aggregate_PROVISIONAL.csv; Table 4/Fig 4 | No paired or assisted-reader inference |

No missing hospital-held information is filled by inference. No training, clinical validation, public repository deposition or weight release is claimed by this manuscript revision.
