# Normalization router audit V1

This experiment asks whether the frozen TNT V3 output should be accepted or
rejected before the frozen Laya intent classifier result is retained. It does
not retrain or modify TNT or Laya.

## Inputs

- Frozen paired outputs from downstream V2: REAL, M1 light/medium/heavy, M5 and
  M6.
- Frozen TNT V3 checkpoint and frozen Laya checkpoint, validated by the V2
  hash guards.
- Existing `data/vocabulary.txt` for reproducible surface/OOV indicators.

Injongo reuses the raw `example_id` between intent partitions. The audit keeps
it as metadata but uses the globally unique `row_id` as `routing_group_id`.
All six variants of one utterance are therefore assigned to the same fold.

## Leakage boundary

`feature_families.json` is authoritative:

- `deployable_surface`: available from the input alone;
- `deployable_tnt`: available after one frozen TNT forward/generation pass;
- `oracle_analysis_only`: requires REAL or the synthetic corruption process
  and must not be used to claim deployability.

Gold intent confidences are stored for diagnosis but excluded from the router
features. The character, word and boundary softmax scores are not calibrated.

## Run

```powershell
python -m src.downstream.routing_audit --device cuda --batch-size 128
```

After the frozen head scores have been cached, reproduce the statistical and
routing analyses without running TNT again:

```powershell
python -m src.downstream.routing_audit --reuse-tnt-signals
```

The grouped logistic regression and depth-three decision tree are diagnostic
baselines, not new normalization architectures. Five outer group folds are
used. Classifier probability thresholds are chosen in four inner group folds
by final downstream correctness, not by raw routing-label accuracy.

## Outputs

Results are written only to
`results/downstream/normalization_router_audit_v1/`. The decision-oriented
summary is `routing_audit_report.md`; machine-readable results are in
`routing_audit_summary.json` and the CSV files alongside it.
