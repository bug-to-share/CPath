# v7.5 terminology ledger

| Term | Meaning / constraint |
|---|---|
| MALT / SS | Author-confirmed diagnostic groups; SS history in a patient with MALT is a separate attribute. |
| Internal validation set | The 49 WSIs used for epoch selection and reported evaluation; never an independent test. |
| External evaluation | Historical reviewed collection; later development reuses a previously examined cohort. |
| Historical primary checkpoint | Original seed-42 Hybrid checkpoint; retain its results without replacing them by a favorable later seed. |
| Controlled training runs | Fixed partition and bags across three/five seeds; seeds are not patients or independent clinical samples. |
| Hybrid-Evidence MIL / Hybrid | Full custom architecture; no claim of component superiority or verified disease-specific prototype identity. |
| Matched attention | 256-dimensional attention control; distinct from historical 1,536-dimensional ABMIL. |
| Normalized attention | Matched attention with per-tile L2 normalization times sqrt(1536). |
| Mean-feature logistic regression | Deterministic linear control; distinct from historical neural mean pooling. |
| AUPRC | Average precision, retained for consistency with original figures. |
| Model score | Uncalibrated sigmoid output; not an individual clinical risk. |
| H&E-like / IHC/special / HE-directory | Archive descriptors only until hospital verification. |
| Prototype margin / gate coefficient | Internal similarity/scale; not morphology labels or causal feature importance. |
| -style comparator | Local compact model, not an official full-method reproduction. |
