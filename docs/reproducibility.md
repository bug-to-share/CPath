# Workflow and reproducibility notes

## Analysis workflow

1. Prepare slide-level labels in the local `dataset/{MALT,SS,SP}` directories.
2. Run TRIDENT preprocessing and extract UNI2-h patch features into the expected local workspace layout.
3. Run `scripts/ss_experiment_suite.py` for the selected binary task and MIL aggregators.
4. Review the generated local summaries, predictions, and figures; none of these row-level outputs should be committed.

The committed CSV files are aggregate-only extracts from the local exploratory runs. Figures show model- or experiment-level summaries and intentionally omit case-level heatmaps and representative patient slides.

## Limitations of the archived metrics

- The internal experiment used a single slide-level split. The archived training loop selected checkpoints using performance on that held-out split, so those metrics are not from an untouched test set.
- Patient-level independence was not established for the archived split. Repeated slides from one patient could undermine slide-level separation.
- The external result is a retrospectively reviewed analysis cohort, not a prospectively specified locked external test set.
- External class counts are small and imbalanced; the specificity estimate is particularly uncertain.
- Some baseline implementations are lightweight “-style” implementations rather than exact reproductions of every original training recipe.

For confirmatory work, define patient-level train/validation/test partitions before model development, use validation data for all model and threshold choices, lock the test set, and report uncertainty at the patient level. Re-run the experiments before making confirmatory performance claims.
