# Bounded algorithm improvement study

Frozen before new training on 4 November 2026 (Asia/Shanghai). This file and `configurations.json` govern execution; `EVALUATION_DESIGN_zh.md` records independent design suggestions, superseded where they differ.

## Objective and evidence status

Improve the stability and discrimination of MALT-versus-SS classification using the existing UNI2-h embeddings. Reuse the author-reviewed 144 training / 49 internal evaluation split and the existing 168-slide external analysis set. All original files, checkpoints and manuscript results remain unchanged. Hospital-held clinical fields stay blank.

The external collection has already been examined during previous development. It is a historical external evaluation cohort, not a fresh confirmatory test. No new external predictions are computed until the internal-only selection is written to `selection_lock.json`. External outcomes cannot trigger another configuration, seed, threshold or checkpoint choice in this study.

## Fixed screen

Ten neural configurations, each with seeds 17, 29, 43 (30 runs):

| ID | Intervention | Optimization |
|---|---|---|
| hybrid_control | Original Hybrid, identical initialization/bags to prior controlled runs | 30 epochs, batch 8, AdamW 5e-5 |
| attention_control | Original matched-attention control | same |
| hybrid_l2 | Per-tile L2 normalization times sqrt(1536), eps=1e-12, before Hybrid | same |
| attention_l2 | Identical input transform before matched attention | same |
| attention_mask | Training-only stochastic masking of 6 of the top 10 attention logits per slide | same |
| direct_attention | Gated attention on raw 1536-D features; weighted pooling in original feature space; linear classifier | same |
| subspace_attention | Attention computed on a random sorted set of 256 feature coordinates per training batch; pooling on original 1536-D features | same |
| subspace_largebatch | Same subspace model, batch 32 and lr 3e-4 | 30 epochs, 150 optimizer steps versus 540 for batch 8 |
| hybrid_twobag | Mean supervised BCE over two independently sampled bags per slide | common settings, two forwards |
| hybrid_consistency | Same two bags and BCE as preceding control, plus 0.1 × symmetric Bernoulli KL | common settings, two forwards |

These are source-informed local adaptations, not claimed reproductions of full ACMIL, nnMIL or R-Drop. Direct/subspace attention use tanh and sigmoid attention paths with dropout 0.25, hidden dimension 256, no value projection, and a shared initial linear classification head. Their original-space pooling is a structural comparison, not a pure normalization ablation. Subspace inference averages logits from 21 contiguous 256-D windows with stride 64; sigmoid is applied after this mean. Thus its inference includes a predeclared feature-subspace ensemble, even under the “reference bag” patch-sampling condition. Training uses one shared sorted coordinate subset per minibatch. The large-batch learning-rate recipe changes two factors and update count; any benefit cannot be assigned to batch size alone.

All other models retain original dropout 0.3. Common weight decay is 1e-4 and positive BCE weight is 77/67. No scheduler, early stopping, checkpoint averaging, threshold optimization or calibration fitting. The first strictly highest internal reference-bag AUROC selects each checkpoint; final weights and all 30 epoch histories are also saved. Three seeds quantify training variation, not new clinical observations.

## Pairing and reproducibility

The first training view, row orders and dropout RNG reuse `controlled-v6|seed|epoch|study_slide_id`, `controlled-v6|seed|epoch|order`, and `controlled-v6|seed|epoch|batch_index|dropout` SHA256-derived 32-bit seeds. Exact indices/orders are saved, including for seeds 59/71. Second view: `algorithm-v8-view2|seed|epoch|study_slide_id`; second-forward RNG: `controlled-v6|seed|epoch|batch_index|dropout-view2`. Both two-view variants use identical inputs and RNG. Symmetric KL is `0.5 * mean((sigmoid(z1)-sigmoid(z2)) * (z1-z2))`, differentiable through both logits. This is an adaptation of consistency regularization, not the same stochastic perturbation as original R-Drop.

Training bags contain 512 rows, without replacement when possible, otherwise with replacement. Internal/reference evaluation bags reproduce the original recorded sampling convention. Original-space feature dimensions and feature hashes are verified. Shared original-model tensors start identically within each seed; different computation graphs need not consume identical dropout masks after diverging operations. Direct/subspace models share their classification-head initialization, but attention dimensions differ.

`hybrid_control` and `attention_control` are checked against all three prior controlled runs (scores tolerance 3e-5, same chosen epoch and decisions). A failure stops execution before candidate selection rather than silently redefining the baseline.

## Selection and confirmation of training stability

Choose one neural candidate from the eight interventions, using only mean internal AUROC over seeds 17/29/43. Rank rounded-to-12-decimal mean AUROC descending; exact ties use smaller sample SD, then parameter count, then ID. Neither control is a candidate. Record the entire ranking, all candidate run hashes and chosen model in `selection_lock.json` before external inference. Selecting a candidate does not assert that it beats a control.

Then run the chosen candidate and Hybrid control with two further predefined seeds, 59 and 71 (four runs; total 34). Keep the initial choice regardless of these outcomes. Report the additional seeds separately and all five together; the extra seeds test training stability on the same cohort, not clinical generalization.

The engineering success target, fixed here, is internal mean AUROC improvement of at least 0.01 over Hybrid control, with at least two of three paired initial seeds non-decreasing. A balanced-accuracy decrease exceeding 0.02 is flagged as an operating-point tradeoff. These are practical criteria, not established clinical minimum important differences. Report additional-seed consistency and historical external paired differences separately; do not retroactively redefine success around favorable metrics.

## Deterministic simple controls

Two additional train-only StandardScaler + L2 logistic regression controls use raw mean features or the mean of per-tile L2-normalized features (scaled by sqrt(1536)). Fit on all feature rows of each of the 144 training slides, with C=1, balanced class weights, LBFGS, max_iter=3000, tolerance=1e-8. No hyperparameter search. Save training representations, scaler and estimator. Their inference uses the same 512-row reference bags as the neural primary comparisons, and also reports full-available-feature means separately. They are deterministic structural controls and do not enter the neural seed-ranking procedure.

## Locked evaluation and reporting

After both stages are frozen, evaluate every trained neural checkpoint on the historical external reference bags. Report all internal and external AUROCs, average precision, sensitivity, specificity, balanced accuracy, accuracy, confusion counts and Brier score at threshold 0.5. Do not combine metrics from different checkpoints.

Report equal-probability three-seed ensembles for every neural configuration, and five-seed ensembles for the chosen candidate and Hybrid control. No seed is removed. For those two models only, also evaluate five additional deterministic patch bags per slide: seeds derived from `algorithm-v8-eval|scope|study_slide_id|rep`, reps 1–5. Save all individual scores/indices, their arithmetic probability mean, and the five-seed × five-bag probability mean. Keep this patch-sampling effect separate from both seed ensembling and the subspace model's feature-window inference.

Conditional paired 95% AUROC intervals use 10,000 class-stratified bootstrap resamples, shared between models, seed 20261104. They condition on fitted predictions and do not adjust for internal method selection, epoch selection or prior external-cohort use. No formal claims of superiority from three/five training seeds. All runs, failures and negative findings are retained.

## Resources and delivery

Only physical GPU 3, a single training process, at most 15% of total GPU memory reserved for this process, two CPU Torch threads. Existing GPU jobs are not stopped or altered. A pilot measures throughput before the bounded run; it neither selects configurations nor optimizes outcomes. No new feature extraction, new hospital data, public release or manuscript-result replacement. Write code, source-grounded literature notes, exact logs/checkpoints/scores, independent audits, comparison figures and a Chinese findings report to this directory.
