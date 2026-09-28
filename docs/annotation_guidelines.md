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
