# Notebook index

Notebooks are controllers and inspectable experiment records. Reusable logic is
implemented in `src/`; notebooks should import it instead of duplicating large
functions.

## Canonical workflow

| Stage | Notebook |
| --- | --- |
| Collection cleanup | `youtube_data_cleaning.ipynb` |
| M0 identity audit | `m0_identity_baseline.ipynb` |
| M1 Beqi-inspired rules | `m1_beqi_rules_baseline.ipynb` |
| M2 Eflomal code-switching | `m2_eflomal.ipynb` |
| M3 CMDR-style code-switching | `m3_cmdr.ipynb` |
| M4 Oolel external data | `m4_oolel_external_baseline.ipynb` |
| M5 exploratory mining | `m5_youtube_unsupervised_baseline.ipynb` |
| M6 Gold-train linguistic engine | `m6_gold_linguistic_engine.ipynb` |
| Final TNT benchmark | `m7_tnt_edit_gold_dev_benchmark.ipynb` |

## Superseded experiments

The large mT5, compact autoregressive, copy-aware, relative-copy,
synthetic-validation, and hierarchical-preview notebooks are not part of the
public tree. Their outcomes and reasons for rejection are preserved in
`docs/experiment_failures.md`. Local copies may be kept under the ignored
`local_archive/experimental_notebooks/` directory.

Before committing a notebook, restart it, remove accidental secrets or absolute
machine-specific paths, and keep outputs only when they provide necessary
evidence that is not already exported under `results/`.
