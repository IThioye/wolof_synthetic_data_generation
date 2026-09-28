# Reproducible project workflow

This file describes the final retained workflow. Earlier mT5, autoregressive,
copy-aware, and synthetic-validation experiments remain in the repository as
development history but are not the canonical benchmark.

## 1. Prepare the authentic corpus

1. Collect public YouTube comments with `scripts/scraping/`.
2. Clean and deduplicate them with `notebooks/youtube_data_cleaning.ipynb`.
3. Prioritize likely informal/code-switched candidates.
4. Normalize selected comments in the Flask sentence-annotation interface.
5. Enrich Gold-train token occurrences in the linguistic annotation interface.
6. Freeze the corrected video-disjoint split under
   `data/annotations/splits/`: 142 train, 30 development, and 29 test pairs.

The test video and its targets are excluded from generator learning, model
selection, thresholds, and manual method development.

## 2. Build the synthetic methods

Run or inspect the corresponding notebook for each method:

- M1: `m1_beqi_rules_baseline.ipynb`;
- M2: `m2_eflomal.ipynb`;
- M3: `m3_cmdr.ipynb`;
- M4: `m4_oolel_external_baseline.ipynb`;
- M6: `m6_gold_linguistic_engine.ipynb`.

M5 is retained as an exploratory artifact but is not a final trained condition.
The retained methods are matched on 3,330 unique formal targets before training.

## 3. Run the controlled evaluator

Use `notebooks/m7_tnt_edit_gold_dev_benchmark.ipynb`.

The fixed protocol is:

- seed 2026;
- one shared character vocabulary and random initialization policy;
- TNT-inspired encoder edit model with 2,148,676 trainable parameters;
- 20 synthetic epochs for M1, M2, M3, M4, and M6;
- 20 Gold epochs for every trained condition;
- Gold-development generation every five epochs and CER checkpoint selection;
- batch size 64 for synthetic training, 32 for Gold training, and 64 for
  evaluation;
- learning rates `3e-4` and `1e-4` for synthetic and Gold stages respectively.

Gold-only omits synthetic pretraining. M0 is scored directly as unchanged input
and is not trained.

## 4. Freeze and evaluate

After development selection, the chosen run manifests are recorded in
`results/m7_seq2seq/tnt_edit_gold_dev_v4/frozen_test_manifest.json`. Its
selection fingerprint is:

```text
b3fca292941a02bd95f2eacc1258658b3a089e9ae80c6323163196d19069b564
```

The frozen test is then evaluated once. Predictions and metrics are stored under
`results/m7_seq2seq/tnt_edit_gold_dev_v4/frozen_test/`.

## 5. Rebuild the analysis

Without retraining, run:

```bash
python scripts/reporting/build_final_benchmark_analysis.py
```

Canonical report tables and figures are written to
`results/tnt_edit_report_analysis/`. Do not mix them with earlier result
families.

## 6. Verify and package

```bash
python -m pytest
python scripts/audit_utf8.py
```

The clean report and defence packages live under `report/` and `presentation/`.
See `docs/repository_audit.md` before publishing datasets or rewriting history.
