# Downstream Wolof analysis V2

V2 reuses the exact frozen Laya and TNT checkpoints from V1. No model is
trained. V1 remains under `results/downstream/laya_normalization_v1/`; every V2
artifact is written under `results/downstream/laya_normalization_v2/`.

## Frozen references

- Laya SHA-256: `9f3aa8e8040e83db2dcb564e0645e73fd19025a0376b0aa09d8a838bac1396ce`
- TNT SHA-256: `78b1cbc9777ac8fbb07b189d277cdeb387bea845e5e340f709ae0eaa2d967780`
- Injongo test SHA-256: `97956d275da88024942085b56ae665340b3a5b7045c343be94ae24267b04b01e`
- Seed: `2026`
- REAL guard: accuracy `0.8087774294670846`, macro-F1
  `0.8117716357020083`, tolerance `1e-9`.

The runner stops if any hash, ID/intent pairing, REAL score, exact M1-heavy
text, or V1 paired-category count changes.

## Controlled M1 severity

M1-heavy delegates to the original V1 function and is text-identical to V1.
For light and medium, every eligible occurrence of the same ordered M1 rules is
assigned a deterministic SHA-256-derived number. An occurrence is applied when
that number is below:

- light: `2 / 9.31191222570533 = 0.2147786568`;
- medium: `5 / 9.31191222570533 = 0.5369466420`;
- heavy: `1.0`.

This keeps the original rule inventory/order and sentence-to-sentence
variation. It does not force a quota per sentence. Realized means are reported
and tested for strict ordering.

## Cross-method applicability

- M1: applicable to arbitrary Wolof text.
- M2: not applicable because its Eflomal candidates must be constrained by the
  paired French translation of each input; Injongo has no French translation.
- M3: not applicable for the same paired-French requirement in CMDR scoring.
- M4: not applicable because Oolel is an external pair corpus, not a function
  that corrupts a supplied Injongo sentence.
- M5: applicable through the 114 accepted token mappings, but low coverage must
  be reported separately.
- M6: applicable through frozen lexical mappings and POS/context templates;
  its manifest remains provisional and real-word/context errors are possible.

M0 is represented by REAL and is not a noise method.

## Error-analysis heuristics

All text distances are measured per row against REAL with the same character,
token and chrF implementations as the normalization benchmark. Intent cues are
transparent token log-odds learned from official Injongo train only. No
dev/test label is used to learn them.

UNRESOLVED primary labels are assigned by an explicit hierarchy:

1. textual degradation after normalization;
2. normalized text close to REAL while Laya remains wrong;
3. very high initial corruption;
4. train-derived discriminative cue lost and not restored;
5. partial but insufficient normalization;
6. other/unclear.

The component flags are retained independently in the CSV, so the primary
taxonomy does not hide overlapping explanations. These labels are heuristic,
not human ground truth. Review sheets sorted by degradation and confidence are
provided.

## Execute

```powershell
python -m pytest tests/test_downstream_analysis_v2.py -q
python -u -m src.downstream.v2_runner --repository-root . --seed 2026
```

Or use `notebooks/downstream/laya_downstream_analysis_v2.ipynb`. The notebook
has no training cell and keeps rerunning disabled by default.
