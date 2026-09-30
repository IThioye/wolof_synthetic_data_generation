# Future idea: hybrid TNT and word-aware Wolof normalizer

## Status

V2 is implemented in `src/modeling/tnt_hybrid_transformer.py`. Its completed M1
gate did not improve the frozen TNT V1 strongly enough: diagnostics showed that
the segment head still learned PAD as a frequent terminator, the boundary head
never recovered rare insertion/deletion actions, and the word gate barely
changed character decisions.

The diagnostic-driven V3 experiment is implemented in
`src/modeling/tnt_hybrid_v3_transformer.py` and controlled from
`notebooks/tnt_hybrid_v3_m1_pretrain_gate.ipynb`. It predicts replacement length
separately, supervises replacement content without PAD targets, applies focal
loss to boundary actions, and adds reviewed French/entity/protection features.
V3 remains an experimental post-PFE model and is not part of the frozen report
benchmark.

This V2 experiment was **not** used in the frozen benchmark and must not be
presented as a completed result until its ablations have been run and analysed.
The prepared protocol also records bounded inverse-square-root class weights
from the inner-training labels. V2 applies them to the character-operation head
as well as the boundary and word heads. This is particularly important for
`DELETE`, `SUBSTITUTE`, `EXPAND`, and the rare space-insertion/deletion actions;
no validation or test labels enter that calculation. V2 also supervises only
one end-of-segment PAD rather than allowing unused expansion slots to dominate
the character loss.

## Motivation

The TNT-inspired evaluator improved character error rate because it predicts local character edits in parallel. Its main weakness is that a locally plausible character sequence can still leave the whole word wrong, miss a word boundary, or damage a protected entity. A future model could retain TNT's efficient character repair while adding explicit word- and boundary-level decisions.

## Proposed architecture

Use a shared character encoder with four complementary heads:

1. **Character-operation head:** predict `KEEP`, `DELETE`, `SUBSTITUTE`, or `EXPAND` for every source character, as in the current TNT-inspired model.
2. **Character-segment head:** predict the replacement or expansion characters anchored to each edited position.
3. **Boundary head:** predict whether a space should be kept, inserted, deleted, or left unchanged between adjacent characters. This gives spacing splits and merges a direct objective instead of forcing them through local character segments.
4. **Word-level correction head:** pool the encoded characters of each source token and predict one of three actions: keep the token, apply a candidate lexical correction, or request a generated replacement. Candidate corrections can come from the formal Wolof lexicon, Gold-train mappings, or a constrained character generator.

The word head should not blindly override the character heads. A small learned or calibrated arbitration layer can accept a word-level candidate only when it is consistent with the boundary prediction and when its confidence exceeds the character-only alternative.

## Language and entity protection

Language, named-entity, and protection features can be supplied as auxiliary embeddings or masks. They should be obtained by the same method for every synthetic condition. Protected French spans, names, acronyms, and numbers should default to `KEEP`, while still allowing explicitly supervised corrections when the Gold annotation says the surface form itself is wrong.

## Training objective

A possible joint loss is:

`L = L_operation + L_segment + lambda_b * L_boundary + lambda_w * L_word + lambda_c * L_consistency`

where the consistency term penalizes contradictions between character, boundary, and word decisions. Class weighting or focal loss can counter the dominant `KEEP` class, but weights must be tuned only on an inner split of Gold train—not on Gold development or test.

## Inference

1. Encode the full sentence once.
2. Predict character operations and boundary edits in parallel.
3. Produce word-level candidates for tokens marked as uncertain or editable.
4. Select the highest-confidence non-overlapping edits.
5. Preserve the original input whenever the combined evidence is below a calibrated threshold.

This remains mostly non-autoregressive, so it should retain the speed advantage of the TNT evaluator while adding lexical and spacing awareness.

## Fair evaluation protocol

- Keep the current 29-sentence test set frozen.
- Develop the architecture and thresholds only with Gold train, preferably through an inner train/validation split.
- Retrain every synthetic-data condition with the identical hybrid architecture, initialization, seed policy, and budget.
- Compare against the existing TNT evaluator as a second evaluator, rather than replacing its frozen results.
- Report CER, WER, chrF, exact match, exact edit precision/recall/F1, and overcorrection.
- Add a test stratum containing already-standard sentences to measure preservation safety.
- Use multiple seeds and, if possible, a second human reference for ambiguous normalizations.

## Minimal implementation sequence

1. [x] Add an explicit boundary-alignment target and verify it on spacing split/merge examples.
2. [x] Add a simple token-level `KEEP` versus `EDIT` auxiliary head.
3. [x] Add method-independent French/entity/protection features and an optional hard protection rule.
4. [x] Diagnose the completed V2 M1 heads on Gold-dev.
5. Run the isolated V3 M1 synthetic-pretraining gate and compare it with the matched V1 and V2 runs.
6. If V3 improves correction F1 and overcorrection without degrading CER/WER, run feature/loss ablations before repeating the complete synthetic-method benchmark.

## Main hypothesis

The character head should preserve TNT's CER gains, while boundary and word-level supervision should reduce the gap on WER and exact edit recovery. This remains a hypothesis until evaluated; a more complex model can also overfit the small Gold corpus or increase overcorrection.
