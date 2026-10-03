# Downstream Wolof analysis V2 — results

## Protocol integrity

- REAL was reproduced exactly: accuracy `80.88%`, macro-F1 `81.18%`.
- Frozen Laya SHA-256: `9f3aa8e8040e83db2dcb564e0645e73fd19025a0376b0aa09d8a838bac1396ce`.
- Frozen TNT SHA-256: `78b1cbc9777ac8fbb07b189d277cdeb387bea845e5e340f709ae0eaa2d967780`.
- Final analysis code commit: `8503ea8344e5f6ddfa2c3b768d21f92de89a2d64`.
- No Laya or TNT training occurred.

## 1. Clean preservation

| Condition | Accuracy | Macro-F1 |
|---|---:|---:|
| REAL | 80.88% | 81.18% |
| TNT_REAL | 77.90% | 78.67% |
| Delta | -2.98 pp | -2.51 pp |

TNT changed 338/638 clean sentences (`52.98%`). Sentence identity was `47.02%`
and token identity `94.47%`. It introduced CER `2.06%`, WER `5.58%`, and an
average of `1.10` character edits per sentence.

Paired downstream outcomes:

- 492 remained correct;
- 117 remained wrong;
- 24 were harmed;
- 5 were helped;
- net clean gain: `-19` correct examples.

Therefore the current TNT model does not reliably abstain on already-formal
Injongo text. The largest clean harm concentrations were `update_playlist` (5),
`shopping_list_update` (4), then `make_call`, `plug_type`, `text`, and
`share_location` (2 each).

## 2. M1 severity curve

| Severity | Mean ops | NOISY acc. | NORMALIZED acc. | Acc. gain | Acc. recovery | NOISY F1 | NORMALIZED F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Light | 2.03 | 74.29% | 72.26% | -2.04 pp | -30.95% | 74.44% | 72.91% |
| Medium | 4.99 | 63.95% | 66.77% | +2.82 pp | 16.67% | 64.20% | 67.71% |
| Heavy | 9.31 | 48.59% | 60.34% | +11.76 pp | 36.41% | 45.97% | 60.87% |

TNT is unnecessary or harmful under light M1 noise, begins to help at medium
severity, and provides its largest benefit under the matched heavy condition.
This is consistent with a correction/overcorrection trade-off rather than a
uniform normalization benefit.

At heavy severity the strongest gains remained `time`, `pay_bill`,
`cancel_reservation`, `cook_time`, `calendar_update`, and `min_payment`.
`timezone`, `plug_type`, `food_last`, and `restaurant_suggestion` regressed.

## 3. Cross-method generalization

| Method | Changed rows | Mean ops | NOISY acc. | NORMALIZED acc. | Gain |
|---|---:|---:|---:|---:|---:|
| M1 heavy | 638 | 9.31 | 48.59% | 60.34% | +11.76 pp |
| M5 | 46 | 0.08 | 81.03% | 77.74% | -3.29 pp |
| M6 | 576 | 1.74 | 77.74% | 75.71% | -2.04 pp |

Because M5 changes only 46 rows, its full-set score is dominated by clean
sentences. On the 46 actually modified rows, accuracy fell from `73.91%` to
`67.39%`: no helped case and 3 harmed cases. On the 576 M6-modified rows,
accuracy fell from `77.26%` to `75.35%`: 18 helped versus 29 harmed.

The evidence therefore does **not** show generalization beyond M1. TNT appears
specialized to its M1 pretraining distribution. M2/M3 were not forced onto
Injongo because both require a paired French translation for their candidate
selection. M4 is an external pair dataset, not a conditional corruption
function. These exclusions avoid adding a translation model or inventing a new
generator under an existing method name.

## 4. Detailed V1 error analysis

The frozen V1 counts were reproduced exactly: 110 HELPED, 35 HARMED, 275
UNCHANGED_CORRECT, and 218 UNRESOLVED.

The heuristic primary taxonomy for UNRESOLVED contains:

| Heuristic category | Rows |
|---|---:|
| Discriminative cue not restored | 70 |
| Close to REAL but Laya still wrong | 59 |
| Normalization insufficient | 43 |
| Highly distorted/ambiguous input | 39 |
| Wrong or textually harmful normalization | 7 |

These are measurable review heuristics, not human-validated causal labels. All
component flags remain available independently in the detailed CSV.

Textual restoration was positive on average in every downstream category,
including HARMED. Mean CER improvement was `0.138` for HELPED and `0.109` for
HARMED. The distinction is clearer in gold-intent confidence: `+0.643` for
HELPED versus `-0.537` for HARMED. Thus lower CER alone does not guarantee
semantic usefulness to the classifier.

The association between textual improvement and gold-intent confidence was
positive but weak: Spearman rho was `0.133` for delta-CER, `0.103` for
delta-WER, and `0.129` for delta-chrF. These are associations, not causal
estimates.

Only 3/110 HELPED examples were exact reconstructions. The remaining 107 show
that partial restoration can be enough for downstream recovery.

### Timezone cue

Eight `timezone` examples changed from a correct NOISY `timezone` prediction to
`weather` after TNT. All eight REAL sentences contained `oreer`; all eight
normalized outputs failed to retain/restore that exact cue. This quantifies the
previous observation around `fiso oreer` across eight examples rather than one
anecdote.

## Overall conclusion

TNT provides a real downstream benefit under sufficiently severe, matched M1
noise, but the benefit is conditional. It over-normalizes clean and lightly
corrupted text and does not improve M5 or M6 corruption. A future deployment
would need reliable normalization-need detection, better KEEP calibration, or
training across multiple realistic noise distributions before routing every
input through TNT.
