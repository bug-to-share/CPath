# CPath

Research code and aggregate experimental summaries for weakly supervised whole-slide classification of salivary-gland MALT lymphoma.

## Repository contents

- `scripts/ss_experiment_suite.py`: exploratory UNI2-h MIL benchmark, including the Hybrid-Evidence aggregator and baseline implementations.
- `scripts/evaluate_mil_ensembles.py`: probability-level ensemble analysis from local prediction files.
- `results/`: aggregate metrics and figures only; no slide-level prediction rows or image data.
- `docs/`: workflow, result interpretation, and reproducibility notes.
- [`writing/`](writing/README.md): filtered public manuscript copies (v7.5/v8), aggregate figures, literature reviews and revision plans. See [export scope](writing/PUBLIC_EXPORT.md) for withheld material.

No whole-slide images, tissue patches, embeddings, model checkpoints, patient-level manifests, or logs are distributed here. UNI2-h/CONCH weights and TRIDENT dependencies must be obtained and used under their own access and license terms.

## Data layout expected by the scripts

Set `CPATH_ROOT` to a local workspace that is **not** this public code checkout unless you intentionally keep data outside the repository. The workspace should contain:

```text
$CPATH_ROOT/
├── dataset/
│   ├── MALT/*.ndpi
│   ├── SS/*.ndpi
│   └── SP/*.ndpi
└── scripts/result/
    ├── MALT/20x_512px_0px_overlap/features_uni_v2/*.h5
    ├── SS/20x_512px_0px_overlap/features_uni_v2/*.h5
    └── SP/20x_512px_0px_overlap/features_uni_v2/*.h5
```

The scripts use slide filenames to match feature files and will create row-level cohort, prediction, and checkpoint outputs under the workspace. Keep those outputs local and do not commit them.

## Environment and usage

Use Python 3.10+ with PyTorch, NumPy, pandas, h5py, scikit-learn, and Matplotlib. Install a compatible TRIDENT checkout separately and make its `trident` package importable. See `requirements-experiment.txt` for the Python packages used by these scripts.

```bash
export CPATH_ROOT=/path/to/private/local/workspace
python scripts/ss_experiment_suite.py --tasks malt_vs_non_malt --seed 42
```

The available task and aggregator choices are printed by:

```bash
python scripts/ss_experiment_suite.py --help
```

## Important interpretation note

The committed metrics are exploratory slide-level results. In the archived experiment implementation, the reported holdout split was also used for checkpoint selection; it is therefore not an untouched test set and should not be interpreted as an unbiased confirmatory estimate. The external aggregate reflects a retrospectively reviewed analysis cohort and is also exploratory. See `docs/reproducibility.md` before reusing these values.
