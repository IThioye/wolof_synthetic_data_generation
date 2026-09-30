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

## Current encoder-pretraining experiment

`wolof_encoder_pretraining.ipynb` pretrains the exact TNT V3 character encoder
on the leakage-audited formal-news and informal-comment corpus. It includes a
local speed test, visible batch progress, epoch-level resume, development
checkpoint selection, plots, and a separately disabled test-evaluation cell.

`tnt_hybrid_v3_language_pretrained_m1_gate.ipynb` transfers that encoder into
a fresh TNT V3 model, retains the control's random normalization heads, and
runs the unchanged M1 synthetic then Gold-train protocol in separate resumable
cells. It compares on Gold-dev only; Gold-test remains disabled.

## Post-benchmark architecture experiments

| Experiment | Notebook |
| --- | --- |
| TNT V2 boundary/word-head ablation on an inner Gold-train split | `tnt_hybrid_gold_train_ablation.ipynb` |
| TNT V2 full-hybrid M1 synthetic-pretraining gate | `tnt_hybrid_m1_pretrain_gate.ipynb` |
| TNT V3 diagnostic-driven M1 pretraining → Gold fine-tuning gate | `tnt_hybrid_v3_m1_pretrain_gate.ipynb` |
| TNT V3 inference ablation and short Gold adaptation | `tnt_hybrid_v3_optimization.ipynb` |

The V2 ablation notebook is isolated from the frozen benchmark. The V3 M1 gate
uses Gold-dev for checkpoint selection so it can be compared under the same
protocol as the completed V1/V2 M1 runs; it never opens Gold-test.

## Superseded experiments

The large mT5, compact autoregressive, copy-aware, relative-copy,
synthetic-validation, and hierarchical-preview notebooks are not part of the
public tree. Their outcomes and reasons for rejection are preserved in
`docs/experiment_failures.md`. Local copies may be kept under the ignored
`local_archive/experimental_notebooks/` directory.

Before committing a notebook, restart it, remove accidental secrets or absolute
machine-specific paths, and keep outputs only when they provide necessary
evidence that is not already exported under `results/`.
