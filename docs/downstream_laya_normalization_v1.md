# Downstream normalization V1: Laya × InjongoIntent Wolof

## Experimental question

> Does the existing Wolof normalizer recover intent-classification performance
> lost under synthetic noisy text?

One Laya classifier is fine-tuned on **REAL official train only** and then
frozen. The same checkpoint is evaluated on paired `REAL`, `NOISY`, and
`NORMALIZED` versions of the official test examples. A negative or null effect
of normalization is a valid result.

This branch does not change TNT Hybrid V1/V2/V3 implementations or historical
results.

## Locked protocol

| Role | Official file | Rows | Use |
|---|---:|---:|---|
| Optimization | `InjongoIntent/wol/train.jsonl` | 2,240 | REAL training only |
| Selection and calibration | `InjongoIntent/wol/dev.jsonl` | 320 | Epoch selection and choice temperature |
| Final evaluation | `InjongoIntent/wol/test.jsonl` | 638 | Frozen benchmark only |

No random split, additional holdout, or redistribution is performed. All three
official splits contain the same 40 intents. Train is balanced at 56 examples
per intent.

Injongo's `example_id` is not globally unique: an ID such as
`train-00000000` is reused across intent blocks. The loader preserves this
official value and adds `row_id = split:intent:example_id` for collision-free
paired joins. It never changes the split assignment or text.

Pinned upstream sources:

- Masakhane NLU revision: `1f8be590da6699aee3dc23de6f63e801e2352eff`
- Laya multilingual revision: `e4e9ddf21a7b1903b7acffd8814ad4307bf63a67`
- Laya official notebook repository revision inspected:
  `fa9a2a7070b1789912a49ae24603bbfb1a78b001`
- Python API used by the adaptation: `laya==0.3.21`

## Implementation map

- `src/downstream/injongo.py`: pinned download, split validation, typed-choice
  preparation through Laya's official `build_sequence`.
- `src/downstream/laya_model.py`: pinned model snapshot, dry-run and batched
  inference.
- `src/downstream/laya_finetune.py`: adaptation of the official RLCD +
  cross-entropy DDP training loop.
- `src/downstream/noise.py`: deterministic reuse of the existing M1 generator.
- `src/downstream/tnt_normalizer.py`: read-only TNT Hybrid V3 checkpoint adapter
  with hash validation.
- `src/downstream/benchmark.py`: frozen three-condition evaluation and exports.
- `src/downstream/metrics.py`: classification, delta and paired-effect metrics.

The Laya question is one typed `choice` question containing all 40 intent
labels. The preprocessing guard checks that all 40 mask markers survive and
that all rendered options remain token-distinct. The local dry-run produced a
205-token sequence with 40/40 distinct options.

## Selected existing normalizer

The benchmark uses the locked selection in
`results/m7_seq2seq/tnt_hybrid_v3_optimization_v1/selected_candidate.json`:

- candidate: `m1_v3_gold_adapt_lr_5e5_word_keep_t050`;
- M1 pretraining: 20 epochs;
- Gold adaptation: selected epoch 3, learning rate `5e-5`;
- word KEEP threshold: `0.50`;
- selection: Gold-dev WER;
- expected local path:
  `checkpoints/m7_seq2seq/tnt_hybrid_v3_optimization_v1/m1_v3/seed_2026/gold_adapt_lr_5e5_word_keep_t050/attempt_001/best_model`.

The checkpoint is a local large artifact ignored by Git. The benchmark verifies
its model and config SHA-256 values against the locked selection manifest. It
does not retrain or modify TNT.

## What to run

From the repository root:

```powershell
git checkout experiment/laya-downstream-normalization-v1
python -m pip install -r requirements.txt
python -m pip install -r requirements-downstream.txt
python -m pytest tests/test_downstream_normalization.py -q
jupyter lab notebooks/downstream/laya_injongo_wolof_finetuning.ipynb
```

In the fine-tuning notebook:

1. Run cells sequentially.
2. Keep `RUN_CPU_DRY_RUN=True` for the initial validation.
3. Confirm the converted examples show 40 distinct options.
4. Set `RUN_FULL_TRAINING=True` only in the configuration cell.
5. Keep `NPROC_PER_NODE=1` locally. This uses a direct Python process on
   Windows; `torchrun` is reserved for `NPROC_PER_NODE > 1` on a multi-GPU
   runtime.
6. Run the training cell. It streams epoch/batch progress.
7. Confirm `training_manifest.json` states `"test_used": false`.

The default official hyperparameters retained are four epochs, micro-batch 8,
gradient accumulation 4, encoder LR `2.5e-5`, head LR `1e-4`, RLCD group size
4, exploration sigma `0.4 → 0.1`, gradient checkpointing and FP16. Dev
macro-F1 selects the checkpoint. Dev then fits the choice temperature; this
does not alter argmax predictions. Because no extra split is authorized, dev
serves both selection and calibration, which must be stated when interpreting
calibration quality.

After the trained checkpoint has been reviewed:

```powershell
jupyter lab notebooks/downstream/laya_normalization_benchmark.ipynb
```

The benchmark notebook keeps `RUN_FROZEN_TEST=False` by default. Review the
checkpoint and locked TNT manifest, then switch it once to `True`. The cell
loads official test and writes to `results/downstream/laya_normalization_v1/`:

- aggregate accuracy and macro-F1;
- per-class precision, recall, F1 and support;
- one confusion matrix per condition;
- predictions per condition;
- `noise_drop`, `normalization_gain`, and conditional `recovery_rate`;
- paired `HELPED`, `UNCHANGED_CORRECT`, `UNRESOLVED`, `HARMED` analysis;
- reproducibility manifests and checkpoint hashes.

`NOISY` is always described as **synthetic noisy Wolof**. V1 uses the existing
deterministic M1 Beqi-inspired rules and records each operation. The seed 2026
is retained in the schema/protocol even though this generator currently has no
random branch.

## Important limitations of V1

- Laya's public guidance recommends keeping choice sets around 20 options,
  whereas Injongo requires all 40. The strict option checks protect correctness,
  but the larger decision set remains an experimental limitation.
- Dev is reused for checkpoint selection and calibration because creating a new
  split is forbidden by the protocol.
- Synthetic M1 noise is not equivalent to naturally occurring informal Wolof.
- The result measures robustness to this controlled corruption distribution,
  not general NLU quality on every informal Wolof variety.
