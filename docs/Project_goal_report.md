# Project Goal and Experimental Plan

> **Historical planning record.** Some model choices and future-tense sections
> below document earlier stages of the PFE. The final protocol is defined in the
> root `README.md`, `docs/Work_process.md`, and the frozen result manifests.

**Project:** Synthetic Data for Robust Informal Wolof Processing

**Last revised:** 24 August 2026

## 1. Project summary

The original motivation of this project is to make Wolof language technology usable when people write Wolof informally. Most available Wolof resources and models are trained predominantly on formal written Wolof. As a result, spelling variation, social-media noise, and code-switching can prevent semantically equivalent informal text from being retrieved, understood by an LLM, or processed correctly by a machine-translation system.

The long-term goal is therefore to bridge the representation gap between formal Wolof resources and the informal Wolof used by real users. Because sufficiently large paired formal-informal corpora are not available, the PFE focuses on **synthetic data generation as the data-augmentation strategy for bridging this gap**.

The primary controlled benchmark is text normalization: informal and code-switched Wolof is mapped to a consistent written form while preserving meaning. The target policy was confirmed on 8 August 2026: normalize Wolof orthography while preserving intentional French/code-switched spans. Normalization is both a useful downstream task and a measurable proxy for broader robustness; it is not the project's only underlying motivation.

The central experiment is not simply to generate plausible informal sentences. It is to determine whether different synthetic formal-to-informal generation methods produce data that helps the same downstream normalization model correct real YouTube comments. If the synthetic data successfully teaches the model to bridge formal and informal forms, the same principle may later support semantic retrieval, LLM querying, and machine translation.

The project therefore has two connected directions:

1. **Synthetic generation:** transform normalized text into realistic informal or code-switched variants.
2. **Downstream normalization:** train a model in the reverse direction, from informal text to its normalized target, and evaluate it on held-out human-annotated data.

The proposed contribution is a **gold-train linguistic generation engine** that learns realistic error frequencies, protected categories, reusable lexical variants, and POS-conditioned transformations from the training portion of the human annotations. It is deliberately isolated from the unsupervised YouTube miner and the hand-written rules so their contributions can be compared cleanly. Optional combinations may be studied later as explicitly named ablations.

## 2. Motivation and problem statement

Wolof is widely used online, but its social-media writing is highly variable. YouTube comments may contain phonetic spellings, omitted characters, abbreviations, repeated letters, punctuation noise, French-Wolof code-switching, names, and multiple acceptable spellings. Meanwhile, the corpora used to train Wolof models are much more formal. A system can therefore fail to connect an informal query with a relevant formal document even when both express the same meaning. Similar failures can affect LLM prompts and machine-translation inputs.

Large paired corpora containing an informal sentence and its normalized or semantically equivalent formal version are not readily available. This prevents direct large-scale supervised adaptation.

This creates two problems:

- retrieval, language-model, translation, and normalization systems are poorly adapted to informal Wolof;
- supervised adaptation has too little real paired formal-informal data;
- synthetic examples can be produced at scale, but their usefulness and realism are uncertain.

The project addresses these problems through a controlled comparison. The project-controlled generators will create synthetic informal inputs from a common normalized corpus. The published Oolel dataset is added as an external baseline and reported separately unless Oolel is rerun on that common source pool. Identical model architectures will then be trained on each retained synthetic dataset and tested on the same untouched gold benchmark.

Synthetic quality will not be judged from automatic surface similarity alone. A synthetic sentence may look different from a human sentence and still teach useful corrections, or it may look realistic while changing the meaning. The evaluation must therefore combine intrinsic data analysis, human inspection, and downstream normalization performance.

## 3. Main objective

The long-term objective is:

> To reduce the gap between formal Wolof resources and informal real-world Wolof so that semantic systems can process both forms more reliably.

The operational objective evaluated in this PFE is:

> To design and evaluate a YouTube-informed synthetic data generation method for Wolof normalization, and to determine whether it provides a measurable advantage over generic rule-based and alignment-based synthetic data.

This narrower benchmark is necessary because it provides paired targets and direct error measures. The intended result is an evidence-based comparison. The experiment may show that the proposed method is better, equivalent, or worse than a simpler baseline. Any of these outcomes is valid if the benchmark is leakage-free and the analysis explains the result.

## 4. Research questions

The final report should answer the following questions:

1. **RQ1 — Generic synthetic data:** Does training on rule-generated informal Wolof improve normalization of real YouTube comments compared with a model trained without synthetic augmentation?
2. **RQ2 — LLM-generated data:** How does the external Oolel-generated dataset compare with explicit rule-based generation, and how much of any difference may be due to its different formal source corpus?
3. **RQ3 — Unsupervised YouTube adaptation:** Do variants mined from unpaired YouTube comments improve on generic linguistic rules?
4. **RQ4 — Gold-informed generation:** Does a standalone engine learned from gold-training linguistic annotations outperform the unsupervised YouTube-aware method or generic rules under the same downstream conditions?
5. **RQ5 — Training regime:** Are the differences still visible after every synthetic-data model is fine-tuned on the same gold training set?
6. **RQ6 — Error coverage:** Which error categories are helped or harmed by each generation method?

A secondary application question may be investigated if time and evaluation data permit:

7. **RQ7 — Retrieval transfer:** Does normalization or synthetic adaptation improve retrieval when an informal Wolof query must retrieve a semantically matching formal Wolof passage?

## 5. Concrete deliverables

The project should produce:

- a documented corpus of collected and cleaned YouTube comments;
- a reproducible LLM-assisted filtering pipeline;
- a manually corrected gold dataset with provenance and annotation decisions;
- a video-disjoint train, development, and test split with a locked manifest;
- synthetic datasets generated by the selected baselines and the proposed method;
- an external LLM-generated Wolof baseline based on the published Oolel synthetic dataset;
- an unsupervised lexical-variant mining component for unpaired YouTube comments;
- a standalone gold-informed linguistic generator that records the origin and type of every transformation;
- one controlled downstream normalization benchmark using the same model configuration across methods;
- optionally, a small retrieval robustness case study using manually verified query-document relevance pairs;
- intrinsic, automatic, human, and error-category analyses;
- reproducible scripts, configurations, artifacts, and documentation;
- a final report of approximately 50 pages.

## 6. Data resources and present state

The project currently contains the following resources:

| Resource | Current state | Intended use |
|---|---:|---|
| Raw YouTube comments | 396,559 | Original unpaired source data |
| Cleaned comments | 242,174 from 193 videos | Candidate pool and corpus analysis |
| LLM-classified comments | 7,250 from 5 videos | Annotation candidate selection |
| Fixed filtered candidate set | 5,498 positive rows from 5 videos | Current annotation input |
| Annotation decisions | 359 | Annotation stopped at the selected operational size |
| Kept annotated pairs | 201 across 5 videos | Locked as 142 train, 30 development, and 29 test pairs |
| Token corrections | 918 corrections; 614 distinct informal tokens | Error analysis and train-only mappings |
| Gold-train linguistic annotations | 1,315/1,315 token occurrences | M6 POS/language/protection/error learning |
| Rule-generated synthetic data | 17,777 and 27,403-row datasets | Existing baseline material |
| Oolel-generated external synthetic data | 3,438 raw / 3,433 accepted pairs | Pinned and validated published LLM baseline; manual review pending |
| Eflomal lexicon | 1,989 entries | Existing Linux/WSL alignment artifact |
| IBM Model 1 trial lexicon | 1,672 entries from 2,000 pairs | Windows-native alignment alternative |

These counts reflect the final annotation-size decision for the main benchmark. On 8 August 2026, the 201 kept pairs were locked into 142 training, 30 development, and 29 test pairs, with three, one, and one videos respectively. An automated cross-split audit found no exact input duplicates, exact pair duplicates, or input pairs with at least 95 character similarity. The manifest records the source and split hashes. Gold-test targets are now closed to method selection and tuning.

## 7. Experimental generation methods

### 7.1 Non-synthetic controls

The benchmark needs controls that show whether synthetic data adds value:

- **Copy or no-training baseline:** measures how much of the test set is already unchanged.
- **Gold-only model:** trains only on the gold training split.
- **Optional normalized-identity augmentation:** trains on normalized-to-normalized examples to measure whether gains come merely from additional target-language exposure.
- **Oolel-Corrector external system baseline:** evaluates the published corrector directly on the locked gold test inputs. Because its architecture and original training data differ from the project's fixed seq2seq models, it is reported separately and is not used to rank synthetic generation methods.

### 7.2 Method 1: Generic rule-based generator

This baseline applies documented Wolof-oriented spelling and noise transformations without using YouTube comments or gold annotations. It represents the existing rules-only synthetic approach.

Possible transformations include character substitutions, diacritic removal or replacement, vowel or consonant deletion, spacing changes, repeated characters, abbreviations, and controlled punctuation noise. Each output must keep its clean source and transformation metadata.

### 7.3 Method 2: Alignment or CMDR baseline

This benchmark family now contains two exported conditions over the same 17,777 formal-source rows. M2 uses the saved Eflomal bilingual lexicon and constrains replacement candidates to French tokens found in the paired sentence. M3 uses seeded CMDR-style n-gram embeddings together with lexical-alignment and position scores. Both then apply the shared M1 spelling engine and record row-level transformation provenance. The existing Eflomal artifact can therefore be used without rebuilding it on Windows. IBM Model 1 remains a portability experiment whose mappings require validation before it can replace Eflomal.

M2 and M3 have been exported under the common core schema, with method-specific alignment metadata, manifests, and deterministic 100-row review samples. They remain provisional until manual review measures semantic preservation and code-switch plausibility. Alignment is used for lexical correspondences; a translation model is not required solely to identify word pairs.

### 7.4 Method 3: External Oolel LLM-generated baseline

The public `soynade-research/Wolof-Non-Standard-Orthography` dataset provides 3,438 synthetic pairs. Its `wo` field contains standard Wolof, its non-standard field contains an LLM-generated informal variant, and an `en` field contains the English source or translation. According to the dataset card, the non-standard side was produced by prompting the Wolof-specialized Oolel model using social-media patterns and linguistic instructions. The dataset is released under CC BY-SA 4.0.

This baseline represents **LLM-based synthetic generation**. It should not be confused with `Oolel-Corrector`: that corrector was subsequently fine-tuned on the synthetic dataset for the reverse informal-to-standard task and must not be described as the model that originally generated its own training pairs.

`Oolel-Corrector` may nevertheless be evaluated as an external end-to-end system. This answers how an existing approximately 2B-parameter Wolof corrector performs on the project's real gold benchmark. It does not isolate synthetic-data quality because both the architecture and training data differ from the controlled seq2seq experiments.

Two evaluation options must be distinguished:

1. **Published-dataset baseline:** use the 3,438 published pairs directly. This is reproducible at the dataset level, but its formal source corpus differs from the source pool used by the project's generators, so performance cannot be attributed solely to the generation method.
2. **Common-source Oolel generation:** prompt a specified Oolel checkpoint on the same normalized sentences used by all other methods. This is the fairer method-level comparison, but it requires documenting the exact checkpoint revision, prompt, decoding parameters, seed, output parsing, and rejection rules. The dataset card does not provide all original generation details, so a reproduction would be a new Oolel-based baseline rather than an exact reconstruction.

At minimum, the published dataset should be normalized to the common schema, stripped of wrapper tags, checked for duplicates and unchanged examples, licensed and revision-pinned, and manually reviewed. The main result table must label it as an external dataset baseline if the common-source generation option is not implemented.

### 7.5 Method 4: Unsupervised YouTube-aware generator

This is a separate benchmark method. It uses unpaired YouTube comments but no manual correction pairs and is not an input to the standalone M6 condition.

The method should:

1. construct a trusted normalized vocabulary from the training resources;
2. identify frequent out-of-vocabulary comment tokens;
3. propose formal candidates using weighted character similarity and known Wolof transformations;
4. use frequency, context, and language or entity protection to score candidates;
5. reject ambiguous, French, named-entity, rare, or low-confidence mappings;
6. invert accepted informal-to-formal mappings when generating synthetic informal text;
7. save confidence, frequency, evidence, and provenance for every mapping.

This method tests whether the unpaired target-domain corpus contains useful information even without gold supervision.

The final leakage-safe M5 mining run excludes all comments from the locked
development and test videos. It produced 114 accepted mappings and 17,777
common-source pairs, including 1,821 changed rows. Its two seeded 100-row manual
reviews remain mandatory before the method is accepted for M7.

### 7.6 Method 5: Proposed standalone gold-informed linguistic engine

M6 isolates what can be learned from the manually reviewed gold-training
evidence. Its transformation procedure is:

1. verify exact occurrence coverage against the 142 locked gold-training pairs;
2. retain only reviewed, unprotected Wolof-to-Wolof evidence;
3. estimate POS-conditioned error frequencies and an empirical number of changed tokens per sentence, with an optional configurable cap;
4. learn reusable formal-to-informal lexical and spacing mappings;
5. learn conservative insertion, deletion, replacement, diacritic, and repetition templates that recur for the same POS and character context;
6. protect French, names/entities, ambiguous lookup entries, acronyms, numbers, and sentence-medial title-case tokens;
7. apply a deterministic, seed-controlled sample of eligible learned edits to the common formal source corpus.

Every generated example records its clean source, noisy output, method, seed,
transformation count, error categories, transformed positions, mapping source,
confidence, evidence count, and POS. The implementation uses neither M5
mappings nor M1 rules. The current profile derives 171 lexical or phrase
mappings and 105 repeated POS-conditioned transformation templates from 865
eligible occurrences, including 588 observed changes. Full export remains
provisional only until the separate 100-row M6 quality review is completed.

Spacing merges are learned as exact multiword phrase mappings rather than POS
templates. Consequently, sentence-like corrected units labelled `UNKNOWN` can
still teach mappings such as separate formal words becoming one informal span.
The current profile retains 80 spacing-merge mappings from 93 observed events.

An M5+M6 or M1+M6 condition may be evaluated later, but it must be explicitly
named as an optional combination and cannot replace standalone M6 in the clean
comparison.

Using the gold **training split** to design this generator is not cheating. It is supervised method development, analogous to learning an error model from training data. It becomes leakage only if development or test annotations influence mappings, thresholds, distributions, or rule selection.

## 8. Gold target policy

The annotation target was frozen on 8 August 2026 as follows:

> Normalize Wolof spelling and social-media noise while preserving intentional French words or spans, rather than translating every French span into Wolof.

This makes the task text normalization rather than full translation and preserves the communicative code-switching found in the source comments. Ambiguous or unreliable cases are excluded with `skip_uncertain`; names, URLs, numbers, emojis, and intentional French spans are preserved unless they contain an obvious surface error.

If the project instead requires monolingual formal Wolof targets, that choice changes the task substantially and must be documented before annotation continues.

## 9. Leakage prevention and benchmark construction

The gold dataset must be split by `video_id`, not randomly by comment, into train, development, and test sets. The split script should record row identifiers and hashes in a locked manifest.

The following restrictions apply:

- gold-test pairs must never be used to create rules, dictionaries, error distributions, thresholds, prompts, or synthetic data;
- gold-development data may select thresholds and configurations but may not train final mappings;
- gold-training data may be used by the proposed supervised M6 engine;
- unsupervised YouTube mining should exclude comments from development and test videos whenever possible;
- duplicate or near-duplicate comments must not cross splits;
- all project-controlled generators must receive the same normalized source pool;
- the published Oolel dataset must be labeled as an external-data comparison unless Oolel is rerun on the common source pool.

The final operational benchmark target is at least 200 kept pairs across at least five videos. This target has been reached with 201 kept pairs. It is a pragmatic stopping rule based on project time and annotation cost, not a guarantee of statistical power. The report must therefore acknowledge the small development/test sets, avoid overinterpreting marginal differences, and use multiple training seeds or bootstrap uncertainty where feasible.

## 10. Controlled downstream training

The downstream direction is:

`informal or code-switched input -> normalized target`

The controlled benchmark uses `google/mt5-small` at pinned revision `73fb5dbe4756edadc8fbe8c769b0a109493acf7a`. Its pretrained SentencePiece tokenization is retained; a claim that fully character-level tokenization is essential should not be made without a separate experiment. To fit the local 8-GB GPU/16-GB RAM environment, the base is frozen and every condition trains an independent, identically configured rank-8 LoRA adapter (344,064 trainable parameters, or 0.1145%).

Each synthetic method is evaluated through the same two-stage curriculum: one
synthetic-pretraining epoch followed by five epochs on the identical gold
training split. The synthetic checkpoint is saved as an auditable prerequisite,
but only the post-gold model is assigned generated development metrics and
reported in the main comparison. A separately trained gold-only model measures
whether synthetic initialization adds value.

The shared M7 implementation is now available in
`notebooks/m7_seq2seq_benchmark.ipynb` and `src/modeling/`. Its initial
fixed architecture is the pinned `google/mt5-small` revision and fixed LoRA
adapter configuration above. It validates
the common synthetic schema, blocks exact held-out input or pair leakage,
selects checkpoints using teacher-forced development loss, skips costly
generated metrics for the internal synthetic prerequisite, generates the
development normalization metrics once after gold fine-tuning, and
keeps locked-test evaluation behind an explicit frozen-checkpoint confirmation. Versioned attempts support
resume, unchanged-run skipping, and isolated method reruns, while preserving
models, predictions, metrics, histories, tables, and plots. Model training
remains pending until the retained synthetic methods pass manual review and
their final leakage-safe exports are frozen.

Dataset size, normalized source sampling, architecture, tokenizer, hyperparameter budget, stopping rule, seeds, and gold fine-tuning data must remain constant across the method-controlled comparison. Multiple seeds should be used when compute allows. The 3,433 accepted Oolel pairs set the matched 3,433-example budget, but the Oolel result must still be identified as source-corpus-confounded. A future common-source Oolel generation run would be a separate, more controlled condition.

The external `Oolel-Corrector` evaluation follows a separate protocol: pin the exact model revision and license; verify that gold-test inputs or targets do not overlap its published training dataset; apply the same input cleaning and comparable output cleanup; do not fine-tune on test data; calculate the same test metrics; and report parameter count, hardware, latency, and memory requirements. Results should be presented in two tables: a main fixed-seq2seq synthetic-data comparison and a separate external-system comparison.

## 11. Evaluation plan

### 11.1 Intrinsic synthetic-data analysis

For every generator, report:

- dataset size and rejection rate;
- number of edits per sentence;
- distribution of error categories;
- changed-token and changed-character rates;
- vocabulary and mapping coverage;
- French-Wolof switching rate where relevant;
- duplication and unchanged-output rates;
- semantic or target corruption found during inspection;
- overlap with real gold-training error distributions.

These statistics diagnose the data but do not by themselves prove that a method is useful.

### 11.2 Human evaluation

A blinded sample from every method should be reviewed for:

- meaning preservation;
- plausibility as informal Wolof or code-switched text;
- grammatical or orthographic acceptability for the intended style;
- severity of corruption;
- obvious artifacts or impossible substitutions.

At least 100 examples per selected final method is desirable, subject to available annotation time. This part requires manual work and should use a short, consistent rubric.

### 11.3 Downstream automatic metrics

The normalization benchmark should report:

- character error rate (CER);
- word error rate (WER);
- chrF;
- exact-match accuracy;
- token-level normalization precision, recall, and F1;
- overcorrection rate on tokens that should remain unchanged.
- inference time and computational footprint for the external system comparison.

Results should include uncertainty estimates or variation across seeds where possible. The main comparison is performance on the locked gold test set, not similarity between synthetic methods.

### 11.4 Error-category analysis and ablations

Test performance should be broken down by categories such as character substitution, deletion, insertion or repetition, spacing, abbreviation, punctuation, code-switching, lexical replacement, and other or ambiguous cases.

The standalone M6 method may be ablated into lexical-mapping-only and
generalized-transformation-only conditions. If time permits, separately named
M5+M6 or M1+M6 combinations can test complementarity after the isolated results
have been frozen. This avoids confusing component mixtures with the clean
method comparison.

## 12. Expected contributions

The project aims to contribute:

1. a documented real-world Wolof YouTube normalization dataset;
2. a leakage-aware benchmark for low-resource normalization;
3. an unsupervised method for mining target-domain lexical variants;
4. an auditable train-only linguistic generator whose contribution is isolated from rules and unpaired-comment mining;
5. an empirical comparison showing when synthetic data does or does not improve real normalization;
6. evidence and a reproducible methodology for bridging formal and informal Wolof representations;
7. a detailed error analysis that can guide future retrieval, LLM, translation, and normalization work.

No claim of superiority should be made before the experiments. If differences are marginal, that is an important result: it may indicate insufficient gold-test power, similar coverage among generators, low-quality mappings, or a ceiling imposed by the downstream model.

## 13. Scope limits

The following are not required for the core PFE unless the main benchmark is completed early:

- CycleGAN or adversarial training;
- full unsupervised machine translation or iterative back-translation;
- a separate large language model generation baseline;
- a new tokenizer architecture;
- full RAG evaluation;
- full Wolof-to-French translation evaluation.

These ideas can be discussed as related work or future work. Semantic retrieval remains the original motivating application, but a full RAG pipeline is not a suitable primary benchmark because it mixes retrieval quality with knowledge-base coverage and answer-generation quality. If the core normalization benchmark is complete, a focused retrieval case study can compare formal and informal queries against a fixed corpus with manually verified relevance judgments. This tests retrieval robustness without introducing the additional confounders of answer generation.

## 14. Final experimental sequence

The intended sequence is:

1. freeze the annotation policy;
2. expand annotations across enough videos;
3. create and lock the video-disjoint gold split;
4. implement and validate unsupervised variant mining;
5. implement and validate the standalone gold-informed linguistic generator;
6. export equally sized, traceable synthetic datasets;
7. perform intrinsic and manual quality checks;
8. train identical downstream models under both training regimes;
9. evaluate on the untouched gold test set;
10. analyze errors, ablations, limitations, and statistical uncertainty;
11. write the final report using the recorded evidence and artifacts.

The detailed operational status of these steps is maintained in `docs/PFE_completion_checklist.md`.
