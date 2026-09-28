# Work Progress Report

> **Historical development record.** This file explains how the project evolved.
> For the retained benchmark and current reproduction path, use the root
> `README.md`, `docs/Work_process.md`, and `results/README.md`.

This report summarizes the current project work based on the ordered workflow in `Work_process.md`. The project focuses on collecting informal Wolof/French online text, filtering useful code-switched content, building lexical resources, and creating pipelines for synthetic code-switching and recovery to formal Wolof.

## 1. Data Collection

The first stage collected YouTube comments as a source of informal public text. The scraping workflow uses `playlist_links_scraping.py` to extract video URLs from selected YouTube playlists with `yt-dlp`, then uses `comment_scraping.py` to download comments for each video and append them as JSON lines in `data/comments_data.json`.

Current output:

- `data/youtube/comments_data.json`: 396,559 scraped comment records.
- Each record stores the comment text and the source video URL.
- The scraper keeps track of already processed videos to avoid duplicate scraping.

## 2. YouTube Data Cleaning

The cleaning stage is handled in `youtube_data_cleaning.ipynb`. It removes URLs, usernames, timestamps, emojis, extra whitespace, and very short comments. It also estimates a French-word ratio and removes comments that are mostly French or too noisy.

Current output:

- `data/youtube/clean_data.csv`: 242,174 cleaned comments.
- Main columns: `text`, `video_url`, `clean_comment`, and `french_ratio`.

This step produced the main raw corpus used for later informal Wolof and code-switching filtering.

## 3. Dictionary and Translation Data Preparation

The dictionary preparation stage is in `dictionnary_data.ipynb` (listed as `dictionary_data.ipynb` in `Work_process.md`). It combines dictionary entries and parallel translation data, keeps the Wolof side, and applies handwritten Wolof informalization rules.

The rules simulate common informal spelling changes, such as simplifying doubled vowels, converting Wolof-specific characters into more common spellings, and changing some formal graphemes into informal variants.

Current resources:

- `data/translation data/train-00000-of-00001.parquet`: 17,777 French-Wolof sentence pairs.
- `data/translation data/wolof.csv`: 7,971 entries.
- `data/translation data/dictionnary_translations.json`: 9,626 dictionary translation entries.
- `data/translation data/translation_informal_wolof.csv`: generated informal Wolof variants.

## 4. Synthetic Code-Switching Experiments

Two approaches were explored for generating synthetic informal code-switched text.

In `cmdr.ipynb`, the M3 CMDR-style approach trains seeded n-gram embeddings on paired French and Wolof data. It combines embedding similarity, sentence-level lexical alignment, and positional compatibility to select French replacements. Training uses one worker for reproducibility. The final export records every accepted replacement and its component scores. Named-entity protection is available, but it is disabled in the current controlled exports so that M1, M2, and M3 use the same setting.

In `eflomal.ipynb`, the M2 alignment approach reuses the saved Eflomal Wolof-to-French lexicon. Candidate replacements are constrained to French words appearing in the paired sentence, then selected deterministically by alignment probability. The remaining Wolof text is processed by the same M1 spelling engine, so the comparison isolates the contribution of alignment-based code-switching as far as possible. Acronyms and numbers remain protected; optional named-entity protection is controlled by the same setting described above.

Current output:

- `artifacts/lexicon.pkl`: saved Wolof-to-French lexical alignment artifact.
- `artifacts/cmdr_word2vec.model`: deterministic M3 n-gram embedding model.
- `artifacts/cmdr_alignment_table.pkl`: M3 lexical co-occurrence table.
- `data/synthetic/m2_eflomal_rules_pairs.csv`: 17,777 M2 training pairs.
- `data/synthetic/m2_eflomal_manifest.json`: M2 hashes, parameters, and counts.
- `data/synthetic/m2_eflomal_review_sample.csv`: 100 M2 examples awaiting manual review.
- `data/synthetic/m3_cmdr_rules_pairs.csv`: 17,777 M3 training pairs.
- `data/synthetic/m3_cmdr_manifest.json`: M3 hashes, parameters, and counts.
- `data/synthetic/m3_cmdr_review_sample.csv`: 100 M3 examples awaiting manual review.
- `data/Lexique400/Lexique4.tsv`: canonical French wordlist used by the reverse pipeline and linguistic annotation support.

The generated M2 data contain 13,661 rows with at least one alignment substitution (49,063 substitutions total). M3 contains 6,176 such rows (9,548 substitutions total). These counts measure activity, not quality. The manual review files must be completed before either condition is accepted for final interpretation; preliminary inspection shows that some CMDR neighbors are distributionally related but not valid replacements.

A Windows-native alternative is available in `ibm1_alignment.ipynb`. It uses
NLTK IBM Model 1 on the same parallel corpus and saves an Eflomal-compatible
lexicon as `artifacts/ibm1_lexicon.pkl` without overwriting the Eflomal baseline.
The two lexicons should be compared through coverage, manually reviewed top
translations, and downstream synthetic examples before selecting one.

## 4.1 M7 downstream normalization benchmark

`notebooks/m7_seq2seq_benchmark.ipynb` and `src/modeling/` implement the shared
downstream experiment. All retained synthetic methods are loaded through the
same validated schema and can be deterministically size-matched. The initial
architecture is now fixed as `google/mt5-small`, pinned to revision
`73fb5dbe4756edadc8fbe8c769b0a109493acf7a`. The detected NVIDIA GeForce RTX
5060 Laptop GPU has approximately 8 GB VRAM. The matched default uses 3,433
synthetic pairs per method, one initial seed (2026), maximum source/target
lengths of 192/96, bfloat16 and TF32, physical batch size 32, accumulation 1,
and fixed greedy decoding. M0 is no longer trained: it is an identity artifact,
not a synthetic-error generator, and the equivalent no-normalization control
can be scored directly without spending a full synthetic-training run.
The main benchmark now uses independent rank-8 LoRA adapters on a frozen mT5
base: 344,064 of 300,520,832 parameters are trainable (0.1145%). Gradient
checkpointing is disabled. The LoRA learning rates are fixed at `1e-3` for the
initial stage and `5e-4` for gold continuation. The optimizer was changed uniformly from AdamW to
Adafactor after the first visible full-fine-tuning gold diagnostic showed severe
memory pressure and GPU underutilization; those aborted attempts are not
benchmark results.
Strict deterministic CUDA kernels were subsequently disabled after a LoRA
diagnostic still showed only 1–6% GPU utilization and approximately 217 seconds
per optimizer step. Fixed seeds remain in place, and the physical/effective
batch adjustment is applied identically to every condition.
Three seeds remain preferable if the whole design can
be repeated before the test set is opened.

The reported conditions are gold-only and a two-stage synthetic-then-gold
curriculum. For every method, the runner automatically trains one epoch on its
synthetic corpus, retains that selected checkpoint as an auditable prerequisite,
and fine-tunes it on the identical locked gold training split for five epochs.
The intermediate synthetic checkpoint is not reported as a benchmark result
and does not run autoregressive development prediction. Development loss
selects checkpoints at both stages. After the lowest-loss post-gold checkpoint
is restored, development predictions and CER, WER, chrF, and correction metrics
are generated once. This materially reduces run time while applying the same
selection rule to every condition. The test cell remains disabled until the
method list and checkpoint paths are explicitly frozen.

The shared evaluator records corpus CER, WER, chrF, exact match, correction
precision/recall/F1, false-edit overcorrection, and unchanged-sentence
overcorrection. The new runner versions every condition under a numbered
`attempt` directory, resumes interrupted Trainer checkpoints, skips an unchanged
completed run, and creates a new attempt when its data/configuration fingerprint
changes or when an explicit rerun is requested. It writes configuration and run
manifests, the selected model, development predictions, hashes, metrics,
training-history CSV files, per-run curves, aggregate result CSV/Markdown files,
and comparison plots. This makes it possible to modify and rerun one method
without retraining or overwriting the others. Exact held-out input or pair
overlap causes a hard failure. The video-disjoint gold
split is now locked at 142 training, 30 development, and 29 test pairs. Before
training, 59 legacy Streamlit training sentences whose full-sentence field had
remained equal to the automatic suggestion were reconstructed from their saved,
approved token corrections. The original split was archived, the corrected
split was regenerated with the same source IDs and video assignments, and its
new hashes were recorded.

M6 has now been exported with the uncapped empirical edit-budget configuration:
17,777 pairs, including 14,215 changed rows. M5 still has three unresolved
manifest gates and M6 remains provisional because their manual reviews are not
complete. At the project owner's request, the current M7 execution bypasses
these review gates with `allow_provisional=true`; every affected run records
`quality_gate_override=true`. These results must therefore be labelled as
review-bypassed unless the reviews are later completed without changing the
datasets. Model scores remain pending until execution completes.

## 4.2 M5 final run and isolated M6 implementation

M5 was regenerated after the locked gold split was created. The mining corpus
contains 239,795 eligible comments after excluding every comment from the one
development and one test video. The final run produced 19,828 candidates, 114
accepted mappings, and 17,777 common-source pairs, of which 1,821 changed. Its
manifest is non-provisional, but M5 remains blocked from M7 until the 100-row
mapping and 100-row generation reviews are completed and finalized.

`notebooks/m6_gold_linguistic_engine.ipynb` and
`src/generation/gold_linguistic_engine.py` implement M6 as a standalone
gold-train linguistic generator. It consumes the 1,315 finalized occurrence
annotations and rejects any source identifier outside the 142 locked training
sentences. Under the current thresholds, 865 reviewed, unprotected Wolof
occurrences provide 588 observed changes, 171 reusable lexical or phrase
mappings, and 105 POS-conditioned character transformations. The engine also
learns POS-specific error frequencies and an empirical sentence edit-budget
distribution from gold train. The maximum can be capped for the final benchmark
or set to `None` for an uncapped diagnostic that preserves observed train counts
up to 15 rather than applying every possible edit.

M6 does not load M5 mappings and does not apply M1 rules. French, named
entities, ambiguous lookup entries, acronyms, numbers, and sentence-medial
title-case tokens are protected. Each selected edit records its position,
error category, learned source, confidence, evidence count, POS, and seed. Full
export is independent of M5 and produces a separate 100-row M6 quality-review
sheet; its manifest remains provisional until that review is complete. A later
M5+M6 or M1+M6 combination, if tested, must be named and reported separately as
an ablation rather than used as the primary M6 condition.

Spacing-merge annotations are represented as exact multiword phrase mappings,
not generalized POS templates. The current profile retains 80 such mappings
from 93 observed events. Their `UNKNOWN` POS is intentional and does not prevent
phrase matching.

## 5. Informal Code-Switched Comment Classification

The script `classify_wolof.py` uses a local Ollama model, `gemma3:12b`, to classify cleaned comments as useful informal Wolof/code-switched content or mostly French. It batches comments, validates that model responses contain all expected IDs, retries failed batches, falls back to one-by-one classification when needed, and saves checkpoint files for resumability.

Current output:

- `checkpoints/`: batch classification checkpoint files.
- `data/checkpoint_data.csv`: 7,250 classified rows merged from checkpoints.

This step is important because it turns the large cleaned comment corpus into a smaller dataset focused on useful informal Wolof/French material.

## 6. Dataset Filtering

The checkpoint outputs are merged by `merge_checkpoint_data.py`, then filtered by `get_filtered_dataset.py`. The filtering keeps only comments labeled as informally code-switched.

Current output:

- `data/filtered_data.csv`: 5,498 retained comments after the index-join fix.
- This corresponds to 75.8 percent of the currently classified checkpoint subset.

## 7. Vocabulary Building

The script `build_vocabulary.py` builds a formal Wolof vocabulary file from the available translation and dictionary resources. It reads CSV, JSON, and Parquet inputs, extracts words from columns named `wolof`, removes numeric tokens, deduplicates words, sorts them, and writes them to a text file.

Current output:

- `data/vocabulary.txt`: 32,178 unique Wolof vocabulary entries.

This vocabulary is later used to recover formal Wolof spellings from informal text.

## 8. Reverse Code-Switching Pipeline

The final current stage is `reverse_code_switched.py`, which attempts to recover formal Wolof from informal code-switched Wolof/French input. It uses the saved lexicon, the formal Wolof vocabulary, a French corpus vocabulary, a general French wordlist, and NER-based entity protection.

The reverse pipeline:

- Masks named entities, acronyms, and numbers.
- Classifies tokens as formal Wolof, substituted French, native French, ambiguous, or informal Wolof.
- Converts substituted French words back to likely Wolof forms using the reverse lexicon.
- Applies reverse normalization rules and weighted edit distance to recover formal Wolof spellings.
- Restores protected entities and saves the recovered output.

Current output:

- `data/test_exported_data.csv`: 111 test rows.
- `data/test_exported_data_recovered.csv`: 111 rows with an added `recovered_formal_wolof` column.

## Current Project Status

So far, the project has built an end-to-end experimental pipeline: YouTube comment collection, cleaning, code-switched content filtering, vocabulary creation, synthetic code-switch generation, and reverse recovery from informal code-switched input to more formal Wolof.

The strongest completed parts are the data collection/cleaning pipeline, the checkpoint-based classifier workflow, the filtered informal comment dataset, the Wolof vocabulary, and the first working version of the reverse recovery pipeline.

Recommended next steps:

- Continue diversity-ordered classification beyond the current 7,250 checkpointed rows to cover more videos in `data/youtube/clean_data.csv`.
- Manually evaluate samples from `data/filtered_data.csv` to estimate classification quality.
- Evaluate `data/test_exported_data_recovered.csv` against human corrections or a small gold dataset.
- Fix encoding issues visible in some notebook/script outputs so Wolof characters display correctly.
- Consolidate notebook experiments into reusable scripts once the final method is selected.
