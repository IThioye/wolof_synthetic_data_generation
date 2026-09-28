# PFE Completion Checklist

**Last revised:** 11 September 2026

**Purpose:** single status tracker for completed work and remaining PFE tasks

## How to use this file

Markdown is a good format for this checklist because it is readable, version-controlled, and supported by VS Code and GitHub.

- `- [x]` means completed.
- `- [ ]` means not completed.
- In VS Code, use **Ctrl+Shift+V** to open the rendered Markdown preview.
- To update a task reliably, edit `[ ]` to `[x]` in the source file. Clicking a rendered box is not supported by every Markdown viewer.
- Tasks marked **[MANUAL]** require a decision, annotation, inspection, or writing that should not be delegated entirely to code.
- A box should be checked only when its evidence or output file exists.

## Current snapshot

| Item | Current status |
|---|---:|
| Raw YouTube comments | 396,559 |
| Cleaned comments | 242,174 across 193 videos |
| LLM-classified comments | 7,250 across 5 videos |
| Positive annotation candidates | 5,498 across 5 videos |
| Annotation history rows | 477 |
| Kept gold pairs | 201 |
| Token corrections | 918; 614 distinct informal tokens |
| Gold benchmark split | 142 train / 30 development / 29 test |
| Gold-train linguistic annotation | 1,315 / 1,315 token occurrences complete |
| External Oolel synthetic dataset | 3,433 accepted pairs imported |
| Current gold video coverage | 5 videos |
| Minimum split target | 200 kept pairs across at least 5 videos — reached |
| Downstream model results | Frozen 29-sentence test complete for M0, gold-only, M1--M4, and M6 |

# Part I — Completed work

## A. Project framing and methodology

- [x] Defined the main task as informal/code-switched Wolof to normalized text.
- [x] Defined synthetic generation as the reverse training-data operation: normalized text to informal text.
- [x] Established that downstream performance on real held-out annotations is the primary test of synthetic-data usefulness.
- [x] Replaced the overextended CycleGAN/IBT blueprint with a feasible controlled benchmark.
- [x] Documented unsupervised YouTube lexical mining as an exploratory method and excluded it from the final matched benchmark.
- [x] Added the published Oolel LLM-generated Wolof dataset as a distinct external benchmark baseline.
- [x] Defined any future `Oolel-Corrector` run as a separate external-system comparison rather than part of the generator ranking.
- [x] Defined the main proposed method as a standalone gold-train linguistic generation engine.
- [x] Established that gold-training information may be used by the proposed method without leakage.
- [x] Established that development and test annotations must not influence learned mappings or error distributions.
- [x] Positioned semantic retrieval as the original motivating application and normalization as the primary controlled benchmark.
- [x] Limited full RAG and full translation evaluation to future work while retaining an optional focused retrieval case study.
- [x] Documented the revised objective, research questions, methods, evaluation, and scope in `docs/Project_goal_report.md`.

## B. Data collection and preparation

- [x] Collected 396,559 raw YouTube comments.
- [x] Cleaned 242,174 comments from 193 videos.
- [x] Built a vocabulary pipeline in `src/pipelines/build_vocabulary.py`.
- [x] Implemented YouTube scraping scripts under `scripts/scraping/`.
- [x] Corrected project-root and data-path handling in `src/config.py`.
- [x] Corrected scraping-script path handling.
- [x] Documented data lineage in `docs/Data_lineage.md`.
- [x] Documented the running environment in `docs/Running_environment.md`.

## C. Classification and annotation infrastructure

- [x] Implemented the Wolof comment classification pipeline.
- [x] Classified 7,250 comments from 5 videos.
- [x] Produced a fixed filtered candidate dataset containing 5,498 positive rows.
- [x] Fixed the filtered-checkpoint join so `source_index` and `video_url` are preserved.
- [x] Added stable append-only checkpoint identifiers.
- [x] Added bounded classifier runs through `--max-rows`.
- [x] Defined unclassified rows as rows with missing classifications rather than relying on row order.
- [x] Added round-robin video ordering to reduce single-video concentration during classification and annotation.
- [x] Implemented the Flask annotation application.
- [x] Implemented keep, discard, uncertain, and false-positive annotation decisions.
- [x] Saved annotation decisions to `data/annotations/gold_annotations.csv`.
- [x] Saved token corrections to `data/annotations/token_corrections.csv`.
- [x] Recorded 359 annotation decisions.
- [x] Kept 201 corrected pairs.
- [x] Recorded 918 token corrections covering 614 distinct informal tokens.
- [x] Created annotation guidance in `docs/annotation_guidelines.md`.
- [x] Implemented the gold-train linguistic annotation application and reusable Wolof/French/name lookup support.
- [x] Completed POS, language, entity, protection, and error-type annotation for all 1,315 current gold-train token occurrences.
- [x] Reconciled the linguistic annotations after the final train-sentence corrections.
- [x] Verified zero missing IDs, orphan IDs, duplicate IDs, and token-pair mismatches in the finalized linguistic annotations.

## D. Existing synthetic generation and alignment work

- [x] Produced a 17,777-row rules-based synthetic dataset.
- [x] Produced a 27,403-row combined synthetic dataset.
- [x] Built an Eflomal alignment lexicon with 1,989 entries.
- [x] Preserved the existing Eflomal artifact so Linux/WSL is not required for routine use.
- [x] Prototyped CMDR generation in a notebook.
- [x] Finalized M2 Eflomal generation with deterministic selection, row-level substitution provenance, a manifest, and a 100-row review export.
- [x] Finalized M3 CMDR generation with single-worker seeded training, saved resources, row-level score provenance, a manifest, and a 100-row review export.
- [x] Generated 17,777 M2 pairs and 17,777 M3 pairs from the canonical parallel source pool.
- [x] Implemented a Windows-native IBM Model 1 alternative in `src/alignment/ibm1_lexicon.py`.
- [x] Created `notebooks/ibm1_alignment.ipynb` for testing IBM Model 1.
- [x] Generated a 2,000-pair IBM Model 1 trial artifact with 1,672 entries and 2,775 links.
- [x] Added NLTK to the project requirements.
- [x] Determined that IBM Model 1 should not replace Eflomal automatically because some trial mappings are noisy.

## E. Benchmark and reproducibility infrastructure

- [x] Implemented video-disjoint train/development/test preparation in `src/pipelines/prepare_gold_splits.py`.
- [x] Added a minimum-data guard so the split is not silently created from inadequate annotations.
- [x] Added split-manifest hashing for reproducibility.
- [x] Added 69 workflow tests covering M2/M3 generation, M5 finalization, gold-correction reconciliation, isolated M6 train-only/language/POS/protection/determinism/uncapped-budget/spacing-phrase guards, and M7 metric/schema/leakage/rerun/aggregation/LoRA/checkpoint-policy/method-selection/two-stage/edit-model checks.
- [x] Verified that all 69 tests pass after excluding M0 from trained M7 conditions.
- [x] Added a UTF-8 audit script at `scripts/audit_utf8.py`.
- [x] Added `.editorconfig` encoding controls.
- [x] Corrected known text-encoding issues in project files.
- [x] Verified that project Python files compile.

# Part II — Remaining work

## Phase 1. Freeze the gold-data protocol

- [x] **[MANUAL] Decide the target policy:** normalize Wolof while preserving intentional French/code-switched spans.
- [x] Record the chosen target policy and decision date explicitly in `docs/annotation_guidelines.md`.
- [ ] Define how annotators should handle ambiguous French/Wolof tokens.
- [ ] Define how to handle names, URLs, emojis, punctuation, and numbers.
- [ ] Define how to handle multiple acceptable normalizations.
- [ ] Define when a comment is too uncertain or semantically unclear to keep.
- [ ] Add a compact error-category taxonomy to the annotation guidelines.
- [ ] Add error categories to future gold annotations or derive them in a reviewed post-processing pass.
- [ ] **[MANUAL] Recheck previously kept annotations against the final policy.**

**Phase 1 exit condition:** the target and exclusion policies are unambiguous, and existing kept pairs conform to them.

## Phase 2. Expand and lock the gold benchmark

- [x] Stop annotation at the revised operational target rather than expanding the classification pool further.
- [x] Confirm that annotation covered five distinct videos.
- [x] **[MANUAL] Annotate comments from at least 5 videos.**
- [x] **[MANUAL] Reach at least 200 kept pairs.**
- [x] Slightly exceed the minimum with 201 kept pairs.
- [x] Verify through a dry run that the current data pass the 200-pair/5-video split guard.
- [x] **[MANUAL] Explicitly confirm that annotation is frozen at 201 kept pairs across 5 videos.**
- [x] Review exact and near-duplicate comments before splitting; the proposed split has no cross-split exact pairs, exact inputs, or input pairs at or above 95 similarity.
- [x] Run the video-disjoint train/development/test split: 142 train, 30 development, and 29 test pairs.
- [x] Confirm that no `video_id` occurs in more than one locked split.
- [x] Confirm that no exact or near-duplicate pair crosses the locked splits.
- [x] Save and freeze the split manifest and SHA-256 hashes.
- [x] Record final split counts by rows and videos.
- [ ] Derive and record final split counts by error category.
- [x] Freeze the protocol: do not inspect or tune against gold-test targets after locking.

**Phase 2 exit condition:** a versioned, video-disjoint gold benchmark exists and passes all integrity checks.

## Phase 3. Validate the existing baselines

- [x] Define the initial common synthetic-data schema for deterministic benchmark generators.
- [x] Require fields for formal text, synthetic text, method, seed, source identifier/corpus, edit count, error types, and applied-rule provenance.
- [x] Export the identity control and existing rules-only data into the common schema.
- [x] Select the 17,777-row parallel corpus as the canonical common source pool; keep the 27,403-row mixed resource outside the controlled comparison.
- [x] Centralize and document every active rules-only transformation; M1 applies the ordered rules deterministically rather than sampling them.
- [x] Export M2 Eflomal and M3 CMDR into the common core schema.
- [x] Save M2/M3 method-specific alignment metadata without changing their common training columns.
- [x] Create the reproducible M4 Oolel download, validation, overlap, export, manifest, and review notebook.
- [x] Select the published 3,438-pair Oolel dataset as an external-data condition; do not describe it as common-source controlled.
- [x] Pin Oolel revision `ce17c80cff6626a91dc4e38b43de12e005bc240a` and record its CC BY-SA 4.0 license.
- [x] Import `soynade-research/Wolof-Non-Standard-Orthography` without mixing it into the gold annotations.
- [x] Resolve and document the dataset-card/schema difference by accepting both `non_standard` and the actual `non_standardized` field.
- [x] Strip and validate the Oolel wrapper/schema; accept 3,433 pairs and reject 5 duplicate pairs.
- [x] Check Oolel automatically for empty, unchanged, duplicate, and malformed outputs; semantic validation remains manual.
- [x] Check exact Oolel overlap against the locked gold development/test resources; no exact input, target, or pair overlap was found.
- [x] Record that regeneration is not the current M4 condition and that the published card does not report the original prompt, decoding parameters, or seed.
- [x] Label direct use of the published data as an external-dataset baseline because it does not share a guaranteed common formal source pool.
- [ ] **[MANUAL] Inspect at least 100 Oolel-generated examples using the common quality rubric.**
- [ ] **[MANUAL] Inspect at least 100 IBM Model 1 mappings.**
- [ ] Calculate mapping precision for the inspected IBM Model 1 sample.
- [ ] Decide whether to retain Eflomal, adopt IBM Model 1, or report IBM Model 1 only as a portability experiment.
- [ ] **[MANUAL] Inspect at least 100 generated examples from each retained baseline.**
- [ ] **[MANUAL] Complete the 100-row M2 Eflomal review file.**
- [ ] **[MANUAL] Complete the 100-row M3 CMDR review file; pay particular attention to semantically related but incorrect substitutions.**
- [ ] Remove or repair transformations that change meaning or corrupt protected spans.

**Phase 3 exit condition:** each retained baseline produces traceable, reviewable examples under one schema.

## Phase 4. Implement unsupervised YouTube-aware lexical mining

- [x] Build and load a trusted normalized vocabulary using formal resources only.
- [x] Implement extraction of frequent out-of-vocabulary tokens from eligible unpaired YouTube comments.
- [x] Exclude all comments from the locked development and test videos before final M5 mining.
- [x] Implement candidate retrieval using character n-grams and weighted character similarity.
- [x] Add candidate evidence from known Wolof spelling transformations through canonical signatures.
- [x] Add frequency and local-context evidence.
- [x] Add French-vocabulary, capitalization/entity, length, repetition, URL-cleaning, and numeric guards; manually audit residual lowercase names.
- [x] Implement scoring and ranking of candidate lexical mappings.
- [x] Implement rejection for low frequency, ambiguity, low similarity, and low confidence.
- [x] Save 19,828 candidate mappings with scores/reasons and 114 accepted mappings.
- [ ] **[MANUAL] Review a stratified sample of at least 100 proposed mappings.**
- [ ] Measure mapping precision and coverage at several confidence thresholds.
- [ ] Choose the threshold using training analysis and the gold development split only.
- [x] Implement deterministic generation by safely reversing accepted mappings while retaining punctuation.
- [x] Export the final leakage-safe M5 dataset: 17,777 rows, including 1,821 changed rows.
- [x] Rerun M5 after locking the gold split, automatically excluding the development and test videos.
- [x] Verify that the replacement M5 manifest records `provisional: false` and `excluded_video_count: 2`.
- [x] Add a guarded M5 review-finalization step that records mapping precision and generation-quality statistics.
- [x] Write unit tests for rule-compatible mining, entity filtering, deterministic generation, and punctuation preservation.

**Phase 4 exit condition:** the unsupervised method has acceptable reviewed precision and a reproducible synthetic export.

## Phase 5. Implement the standalone gold-informed linguistic engine

- [x] Complete and integrity-check occurrence-level linguistic annotation for all 1,315 gold-training tokens.
- [x] Implement token-mapping learning restricted to the 142 locked gold-training source IDs.
- [x] Estimate train-only POS-conditioned error-category frequencies.
- [x] Estimate and deterministically sample the train-only edit-count distribution with a configurable cap or `None` for an uncapped empirical diagnostic.
- [ ] Estimate train-only severity distributions.
- [x] Implement protected-token handling for train-observed French/ambiguous forms, likely names, acronyms, and numbers.
- [x] Apply reviewed gold-training lexical and spacing mappings.
- [x] Learn conservative POS-conditioned character transformations from repeated gold-training evidence.
- [x] Preserve spacing-merge annotations as exact multiword phrase mappings even when their POS is `UNKNOWN`.
- [x] Remove every M5 mapping dependency from the standalone M6 condition.
- [x] Remove the M1 rule fallback from the standalone M6 condition.
- [x] Reserve any M5+M6, M1+M6, or alignment-enhanced combination for a separately named optional ablation.
- [ ] Reject unchanged examples when a changed example is required.
- [ ] Reject malformed or semantically unsafe transformations beyond the configurable edit budget and manual review gate.
- [x] Record transformation positions, categories, sources, confidence, evidence count, and seed.
- [x] Add deterministic per-source seed handling.
- [x] Add unit tests for train-only learning, Wolof/review filtering, POS conditioning, protection precedence, zero-edit budgets, and absence of M5/M1 inputs.
- [x] Create `notebooks/m6_gold_linguistic_engine.ipynb` with integrity audit, preview, guarded export, and review finalization.
- [x] Run the M6 notebook preview and inspect obvious systematic artifacts.
- [x] Export the standalone M6 synthetic dataset.
- [ ] **[MANUAL] Inspect all 100 M6 review examples using the same rubric as the baselines.**

**Phase 5 exit condition:** the isolated proposed method is implemented, tested, traceable, exported, and manually validated without consuming M5 or M1.

## Phase 6. Create the controlled synthetic benchmark

- [x] Select one common normalized target pool for all generators.
- [x] Ensure that test sentences and test-video comments cannot enter generation resources.
- [x] Generate the same number of examples for every main method: 3,330 matched targets each.
- [x] Use documented random seed 2026.
- [x] Remove exact duplicates and audit prohibited held-out leakage.
- [x] Calculate unchanged-output and rejection rates.
- [x] Calculate edit counts and changed-character/token rates.
- [x] Calculate available error-category distributions.
- [x] Calculate lexical/mapping coverage for methods exposing it.
- [x] Calculate code-switching rates for M2 and M3.
- [x] Compare synthetic source-to-target distributions with gold-training distributions.
- [ ] **[MANUAL] Complete blinded quality ratings for retained methods.**
- [x] Freeze the selected training inputs through run fingerprints and record artifact hashes.

**Phase 6 exit condition:** all methods have comparable, frozen datasets and intrinsic quality reports.

## Phase 7. Train the downstream normalization models

- [x] Document the mT5/LoRA feasibility experiments and exclude them from the final controlled table because of runtime and unstable generation.
- [x] Select one fixed TNT-inspired non-autoregressive character edit Transformer for the final benchmark (2,148,676 trainable parameters).
- [x] Use one shared vocabulary and random initialization for gold-only and M1--M4/M6.
- [x] Fix the final protocol at 20 synthetic epochs, 20 gold epochs, Gold-development CER selection every 5 epochs, and deterministic edit reconstruction.
- [x] Implement a shared M7 training and inference pipeline with visible progress and selective resumption.
- [x] Record the RTX 5060 Laptop environment, training parameters, runtime, metrics, and checkpoint provenance.
- [x] Define and record dataset, training, and evaluation seeds.
- [x] Exclude M0 from trained M7 conditions and use it as the direct unchanged-input reference in the frozen evaluation.
- [x] Implement gold-only and automatic two-stage synthetic-then-gold orchestration; retain the synthetic checkpoint as an internal prerequisite without reporting it as a separate result.
- [x] Add exact held-out input/pair leakage guards before training.
- [x] Add protected one-time test evaluation requiring an explicit frozen-checkpoint confirmation.
- [x] Add pre-GPU readiness checks so unavailable or non-final selected methods block an all-method final run.
- [x] Add immutable numbered attempts, input/configuration fingerprints, unchanged-run skipping, interrupted-run resume, and single-method reruns.
- [x] Save per-attempt training histories, training curves, development predictions, selected models, metrics, hashes, and manifests.
- [x] Save accessible run indexes, aggregate CSV tables, Markdown tables, and development/test comparison plots under `results/m7_seq2seq/`.
- [x] Train the gold-only baseline.
- [x] Run one synthetic-pretraining stage for M1, M2, M3, M4, and M6.
- [x] Fine-tune every synthetic-pretrained model on the identical gold training split and report only this post-gold result.
- [x] Use the gold development split for checkpoint selection and tuning only.
- [x] Score the direct unchanged-input baseline during the one-time locked-test evaluation without training M0.
- [ ] Run multiple seeds if compute permits.
- [x] Implement configuration, manifest, checkpoint, prediction, metric, summary, aggregation, and plot export.
- [x] Generate the final M7 run artifacts by executing the long-running training cells.
- [x] Freeze selection fingerprint `b3fca292...69b564` before the test run and do not select methods using gold-test performance.
- [ ] Pin the exact `soynade-research/Oolel-Corrector` model revision and record its AGPL-3.0 license.
- [ ] Check gold-test inputs and targets for exact or near overlap with the published Oolel training dataset.
- [ ] Define identical input preprocessing and comparable output cleanup for `Oolel-Corrector` and the project models.
- [ ] Run `Oolel-Corrector` on the locked gold test inputs without test fine-tuning.
- [ ] Record Oolel parameter count, hardware, peak memory, inference time, and decoding configuration.

**Phase 7 exit condition:** met for the controlled project conditions; the optional external Oolel-Corrector run remains separate.

## Phase 8. Evaluate and analyze results

- [x] Implement and test corpus CER evaluation.
- [x] Implement and test corpus WER evaluation.
- [x] Implement and test corpus chrF evaluation using character n-grams up to order six and beta two.
- [x] Implement exact-match evaluation.
- [x] Implement edit-based normalization precision, recall, and F1.
- [x] Implement false-edit and unchanged-sentence overcorrection measurements.
- [x] Evaluate all frozen project models once on the locked gold test set.
- [ ] Calculate the same CER, WER, chrF, exact-match, token F1, and overcorrection metrics for `Oolel-Corrector`.
- [x] Present the TNT-inspired edit-Transformer synthetic-data results in the main controlled table.
- [ ] Present `Oolel-Corrector` in a separate external-system table or clearly separated panel.
- [x] Keep the not-yet-run `Oolel-Corrector` outside claims about the Oolel synthetic-generation method.
- [ ] Report mean and variation across seeds where available.
- [ ] Break down results by error category.
- [x] Compare every post-gold synthetic condition with gold-only and M0 performance.
- [ ] If time permits, ablate M6 lexical mappings versus generalized linguistic transformations.
- [ ] If time permits, evaluate a separately named M5+M6 combination after the standalone comparison is frozen.
- [ ] If time permits, evaluate a separately named M1+M6 combination.
- [x] Inspect and report partial character-level successes, the single exact M1 sentence, inaccurate edits, and overcorrection behavior.
- [x] Explain the mixed result: lower CER but worse WER and weak exact correction scores.
- [x] Separate observed evidence from causal hypotheses in Chapter 5.
- [x] Run a 5,000-resample paired CER bootstrap and state that it excludes training-seed variance.
- [x] Analyze gold error labels, POS/language/protection patterns, and synthetic transformation exposure.

**Phase 8 exit condition:** the research questions can be answered from a complete result table and reviewed error analysis.

## Optional Phase 8B. Application-level retrieval case study

Start this phase only after the primary normalization benchmark is complete.

- [ ] Define a fixed corpus of formal Wolof passages.
- [ ] **[MANUAL] Create and verify a small set of informal queries with relevant passage judgments.**
- [ ] Evaluate retrieval using the original informal queries.
- [ ] Evaluate retrieval after normalization by each selected model.
- [ ] Compare recall at k, mean reciprocal rank, or another justified retrieval metric.
- [ ] Check whether normalization improves matching without changing query meaning.
- [ ] Report this as secondary evidence rather than as the primary synthetic-data benchmark.

**Optional Phase 8B exit condition:** the project has a small, controlled demonstration of whether the formal-informal bridge transfers to semantic retrieval.

## Phase 9. Prepare the guideline-compliant final report

- [ ] Finalize the report outline and assign a target page range to each chapter.
- [x] Write the introduction and problem statement.
- [x] Write Wolof, normalization, code-switching, and low-resource NLP background.
- [ ] Write the related-work review and verify every citation against the original paper.
- [x] Write the corpus collection, cleaning, classification, and annotation methodology.
- [x] Document the annotation target, error taxonomy, and quality controls.
- [x] Describe each baseline and the proposed method precisely enough to reproduce it.
- [x] Document the leakage policy and video-disjoint split.
- [x] Document the implemented controlled training setup; update it with measured runtime and final run count after execution.
- [x] Present intrinsic synthetic-data results.
- [ ] Present human-review results.
- [x] Present downstream automatic results.
- [ ] Present error-category analysis and ablations.
- [x] Discuss limitations, threats to validity, and ethical considerations.
- [x] Write conclusions and realistic future work.
- [x] Add dataset-flow and experimental-design figures.
- [x] Add result tables with captions, units, and sample sizes.
- [x] Ensure Chapter 5 numbers are traceable to frozen artifacts or the report-analysis script.
- [x] Remove unsupported claims such as assuming the proposed method is superior.
- [ ] **[MANUAL] Proofread the complete report for clarity and consistency.**
- [ ] **[MANUAL] Obtain supervisor feedback and apply the required revisions.**
- [x] Compile the LaTeX report through Biber and repeated pdfLaTeX passes; verify that citations and cross-references resolve.
- [x] Bring the main body within the required 60--90-page range: the compiled body is 82 pages (121 pages including front matter, bibliography, and appendices).
- [x] Generate PDF/A-2b output, verify A4 format and embedded fonts, and create the guideline-compliant submission filename.
- [x] Keep both summaries within 250--300 words and include at least five keywords in each.
- [x] Add the dedicated generative-AI-use declaration and distinguish tool assistance from student decisions and verification.
- [x] Verify that numbered tables, figures, and equations are referenced and that the final LaTeX log has no unresolved citations, references, or overfull boxes.
- [x] Perform sampled manual rendering inspection of the cover, summaries, result figures, and AI declaration.
- [ ] **[MANUAL] Run a complete spelling, grammar, and page-by-page rendering inspection.**
- [ ] **[MANUAL] Replace the remaining acknowledgement name placeholders.**
- [ ] **[MANUAL/SUPERVISOR] Decide whether to condense the introduction (7 pages), state of the art (21), project context (13), and conclusion (8) toward the guideline's recommended per-chapter ranges; the total 81-page body already satisfies the mandatory overall range.**
- [ ] **[MANUAL] Create the final source ZIP after the last content revision.**

**Phase 9 exit condition:** the submitted report is complete, evidence-backed, reproducible, and close to the required length.

## Phase 10. Final reproducibility package

- [ ] Update `docs/Work_report.md` with the final implemented state.
- [ ] Update `docs/Work_process.md` with the exact end-to-end workflow.
- [ ] Update `docs/Running_environment.md` with final package and hardware details.
- [ ] Remove stale references to Streamlit or obsolete file paths.
- [ ] Create one command sequence for rebuilding each non-private artifact.
- [ ] Run the complete test suite.
- [ ] Run the UTF-8 audit.
- [ ] Run Python compilation checks.
- [ ] Verify that notebooks execute in documented order.
- [ ] Record known non-reproducible external inputs and API dependencies.
- [ ] Confirm that private credentials and restricted raw data are excluded from submission.
- [ ] Tag or archive the final code, configuration, manifests, and report version.

**Phase 10 exit condition:** another person can understand and rerun the permitted parts of the project from the documentation.

# Immediate next actions

These are the tasks to do next, in order:

- [x] **[MANUAL] Choose and document the French-span normalization policy.**
- [x] Complete and reconcile all 1,315 gold-train linguistic occurrence annotations.
- [x] Run the standalone M6 preview and select the uncapped empirical edit-budget configuration.
- [x] Export 17,777 standalone M6 pairs, including 14,215 changed rows.
- [ ] Complete the M6 100-row manual quality review and finalize its manifest.
- [x] Record the user-authorized provisional M7 review-gate bypass in run manifests.
- [ ] Complete the outstanding manual quality reviews for every retained method.
- [x] Freeze the comparable 3,330-row method selections and complete downstream training.
- [x] Implement and test the M7 status/preflight command, versioned runner, selective rerun path, result aggregation, and plotting.
- [x] Complete development selection and the one-time frozen 29-sentence test evaluation.
- [x] Write Chapters 4 and 5 with final metrics, prediction diagnostics, gold transformation analysis, and limitations.
- [x] Write the final conclusion and future-work chapter from the now-complete evidence.
- [ ] **[MANUAL] Review the mixed-result interpretation and representative predictions for wording accuracy.**
- [x] Confirm the guideline length calculation: 81 main-body pages, within the required 60--90 pages; front matter, bibliography, and appendices are excluded.

# Useful commands

Run these commands from the project root using the project environment described in `docs/Running_environment.md`.

```powershell
# Start the annotation interface
python -m src.annotation.flask_annotation_app

# Run workflow tests
python -m pytest

# Audit text files for UTF-8 problems
python scripts/audit_utf8.py

# Inspect split-tool options before creating the locked benchmark
python -m src.pipelines.prepare_gold_splits --help

# Inspect classifier options for the next bounded batch
python -m src.pipelines.classify_wolof --help
```

The revised 200-pair/5-video minimum was reached, annotation was frozen, and the corrected split was locked before final training. Do not recreate or tune against the frozen test split.
