# Gold-Train Linguistic Annotation Guidelines

This annotation is used to build the isolated M6 probabilistic generation
engine. It operates exclusively on the corrected `gold_train.csv`; development
and test token corrections are not loaded.

## Run the interface

From the project directory:

```powershell
python -m src.annotation.flask_linguistic_annotation_app
```

Then open `http://127.0.0.1:5001`.

The application can run without a dictionary. It stores occurrence-level
reviews in `data/annotations/token_linguistic_annotations.csv` and reusable
formal-token entries in `data/annotations/formal_token_lookup.csv`.

## Annotation dimensions

- `lemma`: normalized dictionary form. For an uninflected token this can equal
  the formal token.
- `pos`: grammatical part of speech using the displayed Universal POS-style
  labels. Use `UNKNOWN` rather than guessing.
- `source_language`: language of the observed informal form.
- `target_language`: language of the normalized formal form. A misspelled
  French word corrected to French is `fr` for both source and target.
- `entity_type`: named-entity status; this is separate from language and POS.
- `protected`: whether the generation engine should avoid corrupting this
  lexical item. Names, acronyms and some intentional French forms will often be
  protected. Ordinary correct Wolof is not automatically protected.
- `error types`: one or more labels for the transformation observed in this
  occurrence. Simple labels are preselected automatically, but must be
  confirmed. Add higher-level interpretations such as `phonetic_spelling` or
  `french_spelling` when appropriate.
- `review_status`: select `uncertain` whenever a qualified decision cannot be
  made.

Keep **Save these lexical fields to the curated lookup** selected when the
lemma, POS, target language, entity type and protection decision can safely be
reused for the same formal form. Clear it for context-dependent or ambiguous
classifications.

## Optional seed lookup

The sources `data/senegalese_surnames.txt`,
`data/lookup_table_wolof.csv`, and `data/Lexique400/Lexique4.tsv` are converted
without being modified:

```powershell
python -m src.pipelines.prepare_multilingual_lookup_seed --overwrite
```

This writes `data/annotations/formal_token_lookup_seed.csv`. Its canonical
columns begin with:

```text
formal_token,lemma,pos,target_language,entity_type,protected,review_status
```

The reviewed Senegalese name list has the most specific seed-level precedence:
a match receives POS `PROPN`, entity `PERSON`, protection `yes`, and unknown
language unless a reviewer has made a contextual decision. This prevents a
surface match in the Wolof or French dictionaries from turning names such as
`Oumy` or `Awa` into common nouns. A contextual French decision is retained for
ambiguous forms such as `Cheikh`.

The seed file is read-only. Manually reviewed application entries are appended
to the separate curated lookup. A single curated choice does not override a
seed entry known to have multiple POS values or Wolof/French language
ambiguity, because that choice may be contextual. Dictionary homographs with
more than one mapped POS are deliberately prefilled as `UNKNOWN`; `UNKNOWN`
remains visibly selected until the reviewer makes a decision. Their POS
candidates and French definitions are shown in the interface. Do not add
information obtained from gold development or test targets.

The source dictionary uses French for its grammatical labels and definitions;
that does **not** classify its headwords as French. The adapter maps labels such
as `nom`, `verbe`, and `adverbe` to Universal POS values while assigning the
headword language `wo`. When several Wolof definitions all map to the same POS,
the interface displays every definition but locks the shared POS: it is not
selecting the first definition. When the definitions imply different POS
values, POS remains `UNKNOWN` and must be selected for the occurrence. A
Wolof-only unambiguous seed hit starts with both languages `wo`, entity
`UNKNOWN`, protection `no`, and review `reviewed`.

Lexique is evaluated for every corrected gold-training form except reviewed
names. A French-only match receives its mapped POS, target and source language
`fr`, entity `UNKNOWN`, protection `yes`, and review `reviewed`. Changed French
occurrences also receive a proposed `french_spelling` label. A form present in
both Wolof and French resources now starts with language `unknown`, protection
`uncertain`, and only the language-specific POS candidates. Selecting `wo` or
`fr` filters the POS choices accordingly. This decision is occurrence-level
and is not written as a global token rule after one review.

## Important distinctions

- POS, language and entity type are independent. A French name can have POS
  `PROPN`, language `fr`, and entity type `PERSON`.
- Selecting entity type `PERSON` automatically assigns POS `PROPN` and
  protection `yes`. The server enforces this relationship even if a request
  submits inconsistent values.
- `identity` means that no error occurred in this particular observation. It
  does not by itself mean the word must be protected in every generated
  sentence.
- The old automatic token category is displayed only as a hint because its
  French/Wolof decisions were not manually validated.
- The character edit sequence is calculated automatically in the
  formal-to-informal generation direction and stored as JSON. Do not manually
  transcribe insertions, deletions or replacements. Confirm the preselected
  multi-label error types, and only add the linguistic interpretation the
  character comparison cannot determine.
- More than one error label may apply. For example, `bu ne -> buneh` contains a
  spacing merge and an insertion. `identity` is exclusive and cannot be
  combined with another label.
- Hover over an error-label chip to see its definition. Mechanical labels are
  preselected; occurrence-specific source language, error labels, review status,
  and notes remain editable even when dictionary lexical fields are locked.
- For a dictionary homograph, the POS selector contains only the POS candidates
  supplied by the adapted lookup plus `UNKNOWN`; it no longer silently displays
  the first candidate as the current choice. An unmapped or missing form still
  exposes the full POS inventory because no reliable candidate set exists.
- An unchanged Wolof lookup occurrence has fixed Wolof source/target language,
  entity `UNKNOWN`, protection `no`, review `reviewed`, and error `identity`.
  A French lookup fixes both languages to `fr` and protection to `yes`; a name
  lookup fixes POS to `PROPN`, entity to `PERSON`, protection to `yes`, and
  leaves language `unknown`. If the lookup POS is unique, the service saves the
  identity automatically with an `AUTO_IDENTITY_*` note. If POS is ambiguous or
  unknown, or the surface form is both Wolof and French, the occurrence remains
  in the queue.
- The CSV is occurrence-level. Seeing `PROPN` repeatedly for the same formal
  token means that the name occurred at several positions; the `occurrence_id`
  identifies the individual observation.
- When a reviewer saves reusable lexical fields, every other unchanged
  occurrence of that formal token is immediately saved as `identity` with its
  own occurrence ID, source index, token position, and timestamp. This also
  applies when a reviewer deliberately leaves POS as `UNKNOWN`. A seed-level
  ambiguous or unmapped POS is not treated as reviewed merely because its
  placeholder value is `UNKNOWN`.
- A changed informal form is not skipped solely because its normalized formal
  token was seen before. The lexical fields are reused, but source language,
  transformation/error labels, and contextual interpretation remain
  occurrence-specific and therefore still require review.
- `spacing_merge` is an approved exception to that manual-review rule. If the
  normalized formal unit contains a boundary removed by the informal form, the
  occurrence is saved automatically with source and target language `wo`, POS
  and entity `UNKNOWN`, protection `no`, and status `reviewed`. The complete
  mechanically inferred error-label set and character edit sequence are kept,
  the row receives note `AUTO_SPACING_MERGE`, and the interface advances to the
  next unannotated token. These rows are occurrence evidence and are not added
  to the reusable lexical lookup.
- Once two distinct human-reviewed occurrences of the exact same
  `informal token -> formal token` mapping agree on lemma, POS, both language
  fields, entity type, protection, and the complete error-label set, that
  signature becomes the mapping default. Remaining occurrences of that exact
  pair are saved automatically with their own occurrence metadata and the note
  `AUTO_CONFIRMED_MAPPING_DEFAULT`. Automatic rows never count as one of the
  two confirmations. If the most-supported human signatures are tied, no
  default is propagated. Different informal spellings of the same formal token
  are also kept separate.
