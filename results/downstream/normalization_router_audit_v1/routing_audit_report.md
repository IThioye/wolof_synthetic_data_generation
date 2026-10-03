# Normalization router feasibility audit V1

## Scope

This audit reuses the frozen V1/V2 Laya and TNT predictions. It trains neither
model. Grouped routing experiments keep every condition of one utterance in
the same fold. Injongo reuses raw `example_id` values between intents, so the
globally unique `row_id` is recorded as `routing_group_id`. Oracle features are
analysis-only and are never described as deployable.

## Is there an oracle opportunity?

- Always skip: accuracy **71.08%**, macro-F1 **71.83%**.
- Always normalize: accuracy **71.79%**, macro-F1 **72.85%**.
- Perfect help-only oracle: accuracy **76.80%**, macro-F1 **77.26%**.
- Oracle gain over skip: **+5.72%**; over always normalize: **+5.02%**.

There is therefore real headroom for selection. This is an upper bound, not a
deployable result.

## Severity pattern

| condition   | severity   |   rows |   helped |   p_helped |   harmed |   p_harmed |   neutral_correct |   p_neutral_correct |   neutral_wrong |   p_neutral_wrong |   net_benefit |   expected_accuracy_gain |
|:------------|:-----------|-------:|---------:|-----------:|---------:|-----------:|------------------:|--------------------:|----------------:|------------------:|--------------:|-------------------------:|
| REAL        | clean      |    638 |        5 | 0.00783699 |       24 |  0.0376176 |               492 |            0.77116  |             117 |          0.183386 |    -0.0297806 |               -0.0297806 |
| M1_LIGHT    | light      |    638 |       24 | 0.0376176  |       37 |  0.0579937 |               437 |            0.684953 |             140 |          0.219436 |    -0.0203762 |               -0.0203762 |
| M1_MEDIUM   | medium     |    638 |       58 | 0.0909091  |       40 |  0.0626959 |               368 |            0.576803 |             172 |          0.269592 |     0.0282132 |                0.0282132 |
| M1_HEAVY    | heavy      |    638 |      110 | 0.172414   |       35 |  0.0548589 |               275 |            0.431034 |             218 |          0.341693 |     0.117555  |                0.117555  |

The sign changes between light and medium M1 noise in this controlled setup.
It is not a universal operation-count threshold: M5/M6 use different
transformations, and operation count is unavailable on naturally noisy text.

## Method pattern

| condition   | noise_method              | severity        |   rows |   helped |   p_helped |   harmed |   p_harmed |   neutral_correct |   p_neutral_correct |   neutral_wrong |   p_neutral_wrong |   net_benefit |   expected_accuracy_gain |
|:------------|:--------------------------|:----------------|-------:|---------:|-----------:|---------:|-----------:|------------------:|--------------------:|----------------:|------------------:|--------------:|-------------------------:|
| REAL        | none                      | clean           |    638 |        5 | 0.00783699 |       24 |  0.0376176 |               492 |            0.77116  |             117 |          0.183386 |    -0.0297806 |               -0.0297806 |
| M1_LIGHT    | m1_light                  | light           |    638 |       24 | 0.0376176  |       37 |  0.0579937 |               437 |            0.684953 |             140 |          0.219436 |    -0.0203762 |               -0.0203762 |
| M1_MEDIUM   | m1_medium                 | medium          |    638 |       58 | 0.0909091  |       40 |  0.0626959 |               368 |            0.576803 |             172 |          0.269592 |     0.0282132 |                0.0282132 |
| M1_HEAVY    | m1_heavy                  | heavy           |    638 |      110 | 0.172414   |       35 |  0.0548589 |               275 |            0.431034 |             218 |          0.341693 |     0.117555  |                0.117555  |
| M5          | m5_youtube_mined_variants | method_specific |    638 |        4 | 0.00626959 |       25 |  0.039185  |               492 |            0.77116  |             117 |          0.183386 |    -0.0329154 |               -0.0329154 |
| M6          | m6_gold_linguistic_engine | method_specific |    638 |       18 | 0.0282132  |       31 |  0.0485893 |               465 |            0.72884  |             124 |          0.194357 |    -0.0203762 |               -0.0203762 |

## Are HELPED and HARMED separable?

There are 411 clear-outcome rows. The strongest descriptive
deployable signals are:

| feature                               |   helped_median |   harmed_median |   auc_best_orientation |   rank_biserial_helped_vs_harmed |   p_value_bh |
|:--------------------------------------|----------------:|----------------:|-----------------------:|---------------------------------:|-------------:|
| tnt_output_chrf                       |       65.5843   |      79.7449    |               0.695324 |                        -0.390649 |  4.00575e-11 |
| tnt_word_max_change_probability       |        0.914004 |       0.82658   |               0.691472 |                         0.382943 |  9.15273e-11 |
| tnt_operation_max_change_probability  |        0.989005 |       0.974038  |               0.686596 |                         0.373193 |  2.26403e-10 |
| tnt_output_char_edit_rate             |        0.13253  |       0.0800051 |               0.686394 |                         0.372788 |  2.26403e-10 |
| tnt_operation_mean_keep_probability   |        0.566115 |       0.61257   |               0.686216 |                        -0.372432 |  2.26403e-10 |
| tnt_operation_mean_change_probability |        0.433885 |       0.38743   |               0.686216 |                         0.372432 |  2.26403e-10 |
| tnt_output_char_edits                 |        7        |       4         |               0.683933 |                         0.367865 |  3.08795e-10 |
| tnt_word_mean_change_probability      |        0.37203  |       0.28697   |               0.680389 |                         0.360778 |  7.10907e-10 |

These univariate figures are descriptive. Multiple comparisons use a
Benjamini-Hochberg adjustment, and held-out routing performance below is the
decisive test.

## Leakage-controlled routing baselines

| strategy                         | feature_set          |   accuracy |   macro_f1 |   normalization_rate |   helped_routed_to_tnt |   harmed_routed_to_tnt |
|:---------------------------------|:---------------------|-----------:|-----------:|---------------------:|-----------------------:|-----------------------:|
| always_skip                      | none                 |   0.710815 |   0.718345 |             0        |                      0 |                      0 |
| always_normalize                 | none                 |   0.717868 |   0.728539 |             1        |                    219 |                    192 |
| oracle_help_only                 | oracle_outcome       |   0.768025 |   0.772572 |             0.05721  |                    219 |                      0 |
| oracle_choose_best               | oracle_outcome       |   0.768025 |   0.772572 |             0.05721  |                    219 |                      0 |
| oracle_severity_threshold        | oracle_analysis_only |   0.737461 |   0.745904 |             0.388454 |                    187 |                     85 |
| oracle_feature_threshold         | oracle_analysis_only |   0.734326 |   0.74263  |             0.386625 |                    182 |                     92 |
| grouped_threshold_deployable_tnt | deployable_tnt       |   0.725183 |   0.734078 |             0.405695 |                    160 |                    105 |
| grouped_threshold_deployable_all | deployable_all       |   0.725183 |   0.734078 |             0.405695 |                    160 |                    105 |
| grouped_logistic_regression      | deployable_all       |   0.730408 |   0.739014 |             0.265413 |                    145 |                     70 |
| grouped_decision_tree            | deployable_all       |   0.729101 |   0.737414 |             0.276646 |                    141 |                     71 |

Best simple deployable strategy: **grouped_logistic_regression**, accuracy **73.04%**
(+1.96% versus always skip), macro-F1
**73.90%**, normalization rate **26.54%**.

### Results by condition

| condition   |   always_normalize |   always_skip |   grouped_logistic_regression |   oracle_help_only |
|:------------|-------------------:|--------------:|------------------------------:|-------------------:|
| M1_HEAVY    |           0.603448 |      0.485893 |                      0.586207 |           0.658307 |
| M1_LIGHT    |           0.722571 |      0.742947 |                      0.750784 |           0.780564 |
| M1_MEDIUM   |           0.667712 |      0.639498 |                      0.661442 |           0.730408 |
| M5          |           0.777429 |      0.810345 |                      0.805643 |           0.816614 |
| M6          |           0.757053 |      0.777429 |                      0.774295 |           0.805643 |
| REAL        |           0.778997 |      0.808777 |                      0.804075 |           0.816614 |

## TNT KEEP signals

- Character head: `operation_projection`; KEEP is action 0 among
  `keep/delete/substitute/expand`.
- Word head: `word_projection`; KEEP is action 0 among `keep/edit`, computed
  from contextual encoder states mean-pooled over each whitespace-delimited
  word and enriched by the existing lexical flags.
- Boundary head: `boundary_projection`; no-change is `keep_space` on an input
  space and `none` elsewhere. Changes are principally `delete_space` and
  `insert_space_before`.
- The logits exist during inference but the normalizer wrapper previously
  returned only generated text. This audit reads the frozen heads directly;
  no architecture or checkpoint was changed.
- These softmax scores are **not calibrated probabilities**. Training used
  label smoothing, class weighting and focal loss for boundaries, with no
  post-hoc calibration. `1 - P(KEEP)` is therefore a useful score to audit,
  not yet a probability of normalization need.

Descriptive KEEP separability (HELPED versus HARMED):

| feature                                 |   helped_median |   harmed_median |   auc_best_orientation |   rank_biserial_helped_vs_harmed |   p_value_bh |
|:----------------------------------------|----------------:|----------------:|-----------------------:|---------------------------------:|-------------:|
| tnt_operation_mean_keep_probability     |        0.566115 |        0.61257  |               0.686216 |                       -0.372432  |  2.26403e-10 |
| tnt_word_mean_keep_probability          |        0.62797  |        0.71303  |               0.680389 |                       -0.360778  |  7.10907e-10 |
| tnt_boundary_mean_no_change_probability |        0.869589 |        0.871751 |               0.547612 |                       -0.0952245 |  0.112308    |

## Feasibility decision (A–F)

- **A — Oracle headroom:** yes. The oracle is +5.72%
  above always skip, so routing is not intrinsically pointless.
- **B — Deployable separability:** moderate, not strong. The best univariate
  AUCs are around 0.70, and HELPED/HARMED distributions still overlap.
- **C — Simple rule/classifier:** yes on the pooled controlled benchmark.
  `grouped_logistic_regression` beats always skip by +1.96%
  and always normalize by +1.25% under
  grouped out-of-fold evaluation.
- **D — KEEP/logit signal:** useful but uncalibrated. Character and word KEEP
  scores have moderate association; boundary confidence is substantially
  weaker. They can be normalization-need scores only after independent
  calibration.
- **E — Beyond M1-heavy:** not convincingly. The pooled gain mainly comes from
  M1 medium/heavy. On M5 and M6 the best pooled router is more conservative
  than always-TNT but does not consistently beat always-skip.
- **F — Recommended next step:** do **not** build a neural router yet. Preserve
  this small logistic/threshold baseline, collect or reserve human-noisy
  routing data, and calibrate a conservative abstention threshold there. A
  complex router is justified only if that independent evaluation confirms a
  gain beyond the M1 distribution.
