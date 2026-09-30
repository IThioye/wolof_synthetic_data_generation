# Gold Annotation Guidelines

Status: normalization policy confirmed and frozen on 8 August 2026.

## Annotation target

For each real YouTube comment, create one meaning-preserving normalized target.
The target must correct informal Wolof spelling consistently without adding
information that is absent from the comment.

## Manual policy decision

The following policy applies to all retained annotations:

- [x] **Normalization policy:** normalize
  Wolof orthography, preserve intentional French code-switched spans, and fix
  only obvious spelling/punctuation within those French spans.
- [ ] **Monolingual formalization policy:** translate every French lexical span
  into formal Wolof wherever a reliable equivalent exists.

The first option produces a benchmark for normalization of code-switched text.
The second produces a benchmark for translation/formalization and requires more
linguistic judgment. The two policies must not be mixed in one target dataset.

Decision, date, and reviewer:

```text
Chosen policy: Normalization policy; intentional French/code-switched spans are preserved.
Decision date: 8 August 2026
Confirmed by: Primary annotator and project author
Reason: This preserves the communicative code-switching found in YouTube comments and keeps the benchmark focused on normalization rather than translation.
```

## Status definitions

- `keep`: useful Wolof is present, the meaning is sufficiently clear, and a
  reliable normalized target can be written.
- `discard_false_positive`: the classifier was wrong; there is no useful Wolof.
- `discard_uninteresting`: Wolof is present but the row is unusable, duplicated,
  contextless, or otherwise outside the benchmark scope.
- `skip_uncertain`: the intended meaning or correct normalization is uncertain.

## Sentence-level rules

1. Preserve the original meaning and communicative intent.
2. Normalize Wolof spelling and diacritics consistently.
3. Do not invent missing words solely to make a sentence more elegant.
4. Preserve names, acronyms, numbers, and named entities unless clearly mistyped.
5. Follow the chosen French-span policy consistently.
6. Record uncertainty with `skip_uncertain`; do not guess.
7. Review the final sentence after token corrections because token-by-token
   choices may produce an unnatural sentence.

## Quality review

After every 50 kept examples:

- review 10 randomly selected kept annotations;
- check policy consistency and meaning preservation;
- record recurring disagreements or new rules here;
- revisit earlier annotations if a rule changes.

Before locking the benchmark, ask a second qualified Wolof speaker to review a
stratified sample across videos and transformation types if one is available.

## Optimized sentence-level workflow

The Flask interface treats the reviewed sentence as the authoritative label.
Token suggestions are optional assistance and are collapsed by default. At
save time, the application aligns the reviewed target back to the source words
and stores both the derived token corrections and the exact sentence target.
This avoids the earlier risk of the sentence and `token_corrections_json`
diverging.

Recommended order for each row:

1. Read the source and edit it into the formal target. The editor starts from
   the observed sentence to avoid anchoring the label to a noisy suggestion.
   A proposed sentence remains available through **Use suggestion**.
2. Edit the sentence directly when the required correction is clear.
3. Use **Keep unchanged & next** only when the source genuinely needs no
   normalization. Identity pairs are useful training evidence, not filler.
4. Open **Token-level assistance** only for difficult or ambiguous cases.
5. Use `skip_uncertain` rather than forcing a target whose meaning is unclear.

Keyboard shortcuts:

- `Ctrl+Enter`: save and advance;
- `Alt+O`: keep the original sentence and save;
- `Alt+A`: restore the automatic suggestion;
- `Alt+T`: open or close token assistance;
- `Alt+1` to `Alt+4`: select the four review decisions;
- `Alt+S`: skip without saving.

The interface logs annotation duration and whether the final sentence retained
the source, accepted the suggestion, or was edited. These fields support a
later speed/anchoring audit; they are not model labels.

## Starting a separate extension campaign

The canonical 201-pair benchmark history remains the default. The convenient
launcher writes new annotations to separate append-only files while still
skipping canonical rows and reusing their learned corrections:

```powershell
python scripts/run_sentence_annotation.py --campaign gold_extension_v1 --target 1000
```

### Lookup-based word suggestions

The token assistant searches single-word entries in
`data/lookup_table_wolof.csv`. It ranks them with a normalized character
distance and low-cost Wolof orthographic rewrites such as `gn -> ñ`, `kh -> x`,
and `ou -> u`. It does **not** reward candidates merely for having the same
length: distance is normalized by the observed token, and no separate length
penalty is added. Therefore a four-character informal spelling may safely
suggest a three-character standard spelling.

A lookup candidate is applied to the optional automatic sentence only when:

- its normalized distance is at most `0.10`;
- it is separated from the second Wolof candidate by at least `0.08`; and
- a French lookup match is not equally plausible.

The defaults were checked only against Gold-train token pairs. The stricter
gate accepted relatively few edits but avoided the much lower precision of the
old permissive threshold. Even accepted lookup changes remain visibly flagged:
character distance cannot determine contextual choices such as `xol` versus
`xool`. Four-to-three-character alternatives are still listed even when their
score is too high for automatic insertion.

Exact French entries, exact French/Wolof overlaps, close French spellings,
protected entities, and low-confidence Wolof matches remain unchanged and are
flagged for review. These are annotation-efficiency gates, not linguistic
ground truth. The thresholds can be changed for a new session without changing
the code:

The POS abbreviations and definitions shown beside Wolof candidates come from
the French-language dictionary columns. They are explanatory metadata only;
their language does not affect the distance score or automatically assign a
POS label to the annotation.

```powershell
python scripts/run_sentence_annotation.py `
  --campaign gold_extension_v1 `
  --distance-threshold 0.10 `
  --candidate-margin 0.08 `
  --french-margin 0.04
```

The equivalent manual configuration is:

```powershell
$campaign = "data/annotations/campaigns/gold_extension_v1"
$env:PFE_GOLD_ANNOTATIONS_PATH = "$campaign/gold_annotations.csv"
$env:PFE_TOKEN_CORRECTIONS_PATH = "$campaign/token_corrections.csv"
$env:PFE_SENTENCE_ANNOTATION_EVENTS_PATH = "$campaign/annotation_events.csv"
$env:PFE_ANNOTATION_TARGET_KEPT = "1000"
python -m src.annotation.flask_annotation_app
```

The target changes only the progress display. It does not rebuild or mutate
the frozen train/dev/test files. A new held-out split must be chosen by video
before the extension data is used for final evaluation.
