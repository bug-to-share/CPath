# Archived-evidence analysis for manuscript revision v6

This directory contains CPU-only checks of archived scores, saved checkpoints and the eight existing Figure 5 ROI windows. No training, inference, evaluation-set calibration fitting, pathology annotation or crop replacement was performed. Clinical and reader records remain pending incorporation from the hospital. The analysis concerns the accepted MALT-versus-SS grouping and existing cohorts.

Reproduce from the TRIDENT project root:

```bash
MPLCONFIGDIR=/tmp/trident_v6_mpl CUDA_VISIBLE_DEVICES='' python writing/draft_v6_completed_analyses/analysis/analyze_existing_evidence.py
```

The script needs the existing v5 files, original checkpoints, source WSIs and GeoJSON masks, and the local private slide-code mapping. It reads these inputs without changing them. Outputs use publication study codes and contain no original slide identifiers or patient names. The runtime versions are current verification versions, not recovered historical training versions.

## Paired error exchange

`internal_scores_and_errors.csv` joins all 49 internal cases for Hybrid, ABMIL, CLAM-SB-style, RRTMIL-style and the archived three-model ensemble. The archived ensemble equals the arithmetic mean of the three member scores within 1e-7. All archived binary predictions agree with score >= 0.5.

`paired_error_exchange.csv` gives each case in the two pairwise comparisons. The reference is always Hybrid; therefore "corrected by candidate" means a Hybrid error corrected by ABMIL or the ensemble, while "new error in candidate" means a Hybrid-correct slide incorrectly classified by that candidate. `paired_error_summary.csv` reports:

| Candidate compared with Hybrid | Both correct | Both wrong | Corrected by candidate | New error in candidate |
|---|---:|---:|---:|---:|
| ABMIL | 39 | 3 | 3 | 4 |
| Hybrid + CLAM + RRT ensemble | 42 | 5 | 1 | 1 |

ABMIL corrects [withheld slide code], [withheld slide code] and [withheld slide code], while introducing errors in [withheld slide code], [withheld slide code], [withheld slide code] and [withheld slide code]. The ensemble corrects the SS case [withheld slide code] (Hybrid 0.5038888; ensemble 0.305785219) and introduces a MALT error in [withheld slide code] (0.512466 versus 0.27916791). [withheld slide code] and [withheld slide code] are incorrectly classified by all three ensemble members. The two-sided exact McNemar P value is 1.0 for each comparison; these small descriptive exchanges do not establish superiority or equivalence.

## Score quality

`brier_descriptive.csv` gives the squared-error Brier score and 95% percentile intervals from 10,000 slide-level bootstrap resamples stratified by class, retaining the observed class counts. Seeds are explicit per row. The archived model scores remain fixed, so these intervals do not represent bag-sampling or retraining variability. No calibration model was fitted.

Hybrid Brier scores are 0.08947 (95% interval 0.04288–0.14633) internally and 0.09192 (0.07022–0.11562) externally. The internal ensemble Brier score is 0.08288 (0.04122–0.13299). Brier scores combine calibration and discrimination characteristics and depend on case mix; they should not be read as isolated calibration errors or evidence that sigmoid outputs are clinical risk probabilities. The intervals for separate models are descriptive and are not paired superiority tests.

## Original ROI mask audit

`roi_mask_coverage.csv` contains all eight originally displayed 1024 × 1024 level-0 windows. For each window, the numerator is the area of its geometric intersection with the union of the original automatic tissue-segmentation polygons; the denominator is 1024². `roi_mask_provenance.json` records source hashes, coordinate conventions and segmentation settings. GeoJSON coordinates are level-0 image pixels; the EPSG label written by the historical code is a placeholder and no geospatial reprojection is applied.

Seven windows have 100% archived-contour coverage. The Case 10 lowest-margin window has 48.1108% coverage. All eight full-resolution RGB windows, resized by the original 1024-to-220 Lanczos operation, reproduce the archived thumbnails with exact pixel equality. The existing near-blank-looking ROI is retained.

The original segmentation ran at 1× and had `holes_are_tissue=true`; internal blank spaces can therefore be included within a contour. **Contour overlap is not manually confirmed tissue content.** Uniform RGB-threshold descriptors (all channels >= 230, 240 or 250) are retained only as transparent auxiliary pixel statistics; they are not validated tissue/background fractions and are not required in the manuscript. Neither contour overlap nor image brightness identifies the cause of the false-negative prediction. Pathology confirmation remains blank pending the hospital record.

One metadata discrepancy is explicitly retained: Case 10 has stored feature extent 102400 × 62208, whereas the current OpenSlide reader reports 102400 × 70400. The historical thumbnail and overlay are both 999 × 607, consistent in aspect with the stored feature extent. The original full-resolution ROI crops were nevertheless all reproduced exactly at their recorded coordinates. Keep the stored geometry when marking the historical overlay; substituting the current whole-slide height alone would misregister marks relative to that overlay. This audit verifies ROI pixels, not agreement of all historical and current whole-slide extent metadata. The discrepancy should be reconciled before asserting fully verified whole-slide registration.

## Model provenance and configuration

`model_checkpoint_inventory.csv` records SHA-256 checksums, file sizes, parameter counts and training budgets for four archived models. Each checkpoint was strictly loaded on CPU into the original implementation. All four loaded with no missing or unexpected keys and contain no nonparameter buffer elements. `checkpoint_tensor_shapes.json` retains every tensor shape.

| Model | Parameters | Checkpoint MiB | Classifier input dimension | Epoch budget | Batch size | Selected epoch |
|---|---:|---:|---:|---:|---:|---:|
| Hybrid | 1,521,672 | 5.8180 | 256 | 30 | 8 | 15 |
| ABMIL | 6,690,306 | 25.5287 | 1536 | 50 | 16 | 37 |
| CLAM-SB-style | 591,874 | 2.2636 | 256 | 20 | 8 | 16 |
| RRTMIL-style | 1,711,874 | 6.5456 | 256 | 20 | 4 | 11 |

All use 1536-dimensional UNI features, 512 sampled feature rows, seed 42, learning rate 5e-5 and weight decay 1e-4. The original ABMIL implementation retains 1536-dimensional representations and a 512-dimensional attention head; Hybrid projects to 256 dimensions. Shared input features therefore do not isolate an aggregation mechanism. Parameter and checkpoint-size differences do not measure latency or training cost. These are full archived implementations, not a matched-capacity ablation.

`existing_evidence_summary.json` collects the numerical findings. `analysis_provenance.json` records hashed public input tables, code hash, runtime versions, and the bootstrap definition. No hospital diagnosis, stain identity, blinding or clinical metadata has been filled by inference.
