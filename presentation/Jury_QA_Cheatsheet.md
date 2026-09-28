# Jury Q&A Cheatsheet

## Synthetic data for informal Wolof normalization

**Defence rule:** start with the one-sentence answer, give two or three exact facts, then state the limitation yourself. Do not defend a stronger claim than the experiment supports.

## Numbers to remember

- **Formal source pool:** 17,777 French–Wolof pairs from `galsenai/french-wolof-translation`.
- **Source labels inside that pool:** Google SMOL 7,405; mafand 6,366; FLORES 2,009; Microsoft NTREX 1,997.
- **YouTube pipeline:** 396,559 raw comments → 242,174 mechanically cleaned → 7,250 locally classified from five videos → 5,498 annotation candidates → 201 kept Gold pairs.
- **Gold split:** 142 train, 30 development, 29 frozen test; videos are disjoint.
- **Gold-train enrichment:** 1,315 token occurrences; 1,297 reviewed and 18 uncertain.
- **Controlled synthetic budget:** 3,330 common formal targets for each of M1, M2, M3, M4, and M6.
- **Evaluator:** TNT-inspired character edit Transformer, 2.15 million parameters, trained from scratch.
- **Final test:** identity and Gold-only CER 0.220; M4 0.191; M1 0.195; M2 0.200; M6 0.206; M3 0.210.
- **Main conclusion:** synthetic pretraining transferred useful character correspondences, but no system became a reliable word-level normalizer.

---

## 1. Where does the formal corpus come from, and how do you know it follows standardized Wolof?

**Short answer:** The common source pool is the 17,777-row `galsenai/french-wolof-translation` dataset on Hugging Face. I use its Wolof side as the operational formal reference; I do not claim that I independently certified every sentence against an orthographic authority.

- The dataset aggregates four recorded sources: Google SMOL, mafand, FLORES, and Microsoft NTREX.
- The French side supports M2/M3 bilingual evidence; the Wolof side is kept unchanged as the target of local generation.
- Unicode normalization, empty-value checks, canonicalization, duplicate controls, common-target matching, and manifest hashes protect data integrity.
- Additional dictionaries expand lexical evidence, but absence from a dictionary is never treated as proof that a Wolof form is wrong.
- The authentic Gold targets were manually reconstructed independently from YouTube comments and are the actual development/test references.

**Limitation to volunteer:** The 17,777 targets were not all re-annotated by a Wolof linguist. “Formal” is an operational dataset role based on curated translation resources, not a guarantee that every sentence is error-free. Possible source-corpus noise is part of the validity limitations.

## 2. How did you filter comments to keep usable rows?

**Short answer:** Filtering had three layers: mechanical cleaning, heuristic language filtering, and a local-LLM candidate queue; only manual annotation could turn a candidate into Gold data.

1. From 396,559 public YouTube comments, remove URLs, mentions, timestamp-like strings, broad emoji ranges, repeated punctuation, redundant whitespace, empty rows, and comments of two tokens or fewer.
2. Reject rows where more than half of tokens contain no usable alphabetic material, and rows with an estimated French-token ratio of at least 0.5. This produced 242,174 comments from 193 videos.
3. Use local Gemma 3 12B at temperature zero to prioritize plausible informal Wolof or Wolof–French comments. In the final five-video campaign, 5,498 of 7,250 classified rows entered the annotation queue.
4. Manually choose `keep`, `discard_false_positive`, `discard_uninteresting`, or `skip_uncertain`.

**Important nuance:** `langdetect` and Gemma were triage tools, not Gold labelers. Their mistakes affect efficiency and coverage, while the final target remains manual.

## 3. How did you annotate the data?

**Short answer:** I manually reconstructed the complete standardized sentence while preserving intentional French and protected content, then recorded occurrence-level corrections and linguistic properties on Gold train.

- The sentence interface shows candidates in deterministic round-robin order across videos.
- For a kept comment, the full normalized sentence and token correspondences are saved with source index and video provenance.
- The policy corrects non-standard Wolof but does not translate legitimate French spans into Wolof. Names, entities, numbers, and meaningful French are preserved unless their own spelling needs correction.
- The annotation history is append-only; the latest decision per source index wins. It contains 477 history rows but 201 unique kept comments.
- Only the 142 Gold-train sentences receive detailed linguistic enrichment: informal and normalized units, lemma, POS, source/target language, entity class, protection, multi-valued error labels, edit script, provenance, and review status.
- Splitting is by whole video, not random sentence: 142/30/29 across three/one/one videos.

**Limitation to volunteer:** This is mainly single-annotator Gold. Ambiguous spellings and segmentations can admit several valid references; independent double annotation and agreement remain future work.

## 4. How were KEEP, DELETE, SUBSTITUTE, and EXPAND measured automatically?

**Short answer:** A deterministic character-level Levenshtein alignment converts every informal–formal pair into one operation per source position.

- **KEEP:** emit the same aligned character.
- **DELETE:** emit nothing for that source character.
- **SUBSTITUTE:** emit one different character.
- **EXPAND:** emit a multi-character segment anchored to one source position; this can represent insertions and some boundary repairs.
- A terminal position can anchor characters inserted after the final source character.

This automatic operation distribution is different from the manually reviewed linguistic labels such as phonetic spelling, diacritic change, or spacing merge. The model operations say **how strings differ**; the linguistic labels say **what phenomenon the annotator believes occurred**.

For correction precision/recall, a separate word-level sequence matcher extracts source-span/replacement signatures. A predicted edit counts as correct only when both the span and replacement exactly equal a Gold edit.

## 5. What is the difference between M2 and M3, and how was code-switching implemented?

**Short answer:** Both insert French material from the sentence's paired French translation and then apply M1 spelling rules. M2 trusts statistical word alignments; M3 uses a stricter distributional ranking.

**M2 — Eflomal alignment:**

- Train Eflomal on all 17,777 French–Wolof pairs.
- Retain a Wolof→French link only with at least three observations and conditional probability at least 0.10.
- For each row, French candidates must occur in that row's paired French sentence.
- Replace about 20% of eligible Wolof positions, with at least one substitution when possible.
- Mask inserted French before M1 rules so French spelling is not corrupted.

**M3 — CMDR-style distributional selection:**

- Train a shared 150-dimensional Word2Vec space over Wolof/French n-grams.
- Score candidates using cosine similarity, bilingual co-occurrence evidence, relative sentence position, and an n-gram penalty.
- Require similarity ≥ 0.70 and alignment evidence ≥ 0.01; allow at most two non-overlapping substitutions.
- Protect the inserted French, then apply the same M1 rules.

In the matched data, M2 inserts French in 94.5% of rows, 2.62 substitutions per row; M3 does so in 32.9%, 0.46 per row. Gold train contains manually labelled French in 62.0% of sentences, but only 13.5% of occurrences per sentence on average. These are related exposure statistics, not identical probabilities.

## 6. Why is there no M5 in the final benchmark?

**Short answer:** M5 was implemented, but its conservative unsupervised mining produced too little non-identity supervision for a balanced comparison.

- M5 mines frequent out-of-vocabulary YouTube forms and ranks formal candidates by character similarity, rule compatibility, context overlap, and frequency.
- It excludes development/test videos before mining.
- Of 19,828 proposed candidates, only 114 passed the frequency, score, similarity, and margin thresholds.
- Its 17,777-row export changed only 1,821 rows and left 15,956 unchanged.

Training it beside generators that change almost every matched row would mostly test identity pretraining, not the quality of an equally active generator. The negative result is documented rather than hidden.

## 7. How does M6 find lexical mappings and contextual rules, and how is sampling done?

**Short answer:** M6 learns only from reviewed, unprotected Wolof occurrences in Gold train, separates exact mappings from reusable character templates, and applies a deterministic empirical edit budget.

- Eligibility requires reviewed status, Wolof source and target, `protected=no`, and non-empty forms: 865 occurrences remain, 588 changed.
- A lexical mapping groups observations by formal unit and keeps the dominant informal realization when support is at least one and dominance at least 0.50. Multiword spacing-merge mappings are allowed.
- A contextual template comes from insert/delete/replace edits and records POS, error type, source/destination strings, neighboring characters, and prefix/internal/suffix position.
- Only selected generalizable categories are used; a template needs at least two examples and dominance at least 0.60.
- The learned profile contains 171 lexical mappings and 105 contextual templates.
- A source-ID-derived seed samples the number of edits from the empirical Gold-train distribution. Candidates are weighted by confidence, evidence, and POS/error profiles; overlapping edits are rejected.
- French spans, entities, acronyms, numbers, and suspicious title-case tokens are protected.

**Fairness answer:** This is train-only supervised feature extraction, not test leakage. M6 has information unavailable to M1–M4, so it tests whether limited authentic training knowledge improves generation; it is not a data-independent baseline.

## 8. How was TNT adapted, and what is the normalization process?

**Short answer:** I kept TNT's idea of predicting edits and recovery labels together, but adapted it into a small character-level Wolof sentence normalizer trained directly on informal→formal pairs.

- Bidirectional encoder-only Transformer, three layers, hidden size 192, four heads, feed-forward size 768, dropout 0.1.
- Shared 120-symbol character vocabulary; maximum sequence length 256.
- One head predicts `KEEP/DELETE/SUBSTITUTE/EXPAND`; a second predicts up to 32 output characters anchored to each source position.
- Levenshtein alignment constructs targets automatically.
- At inference, all source positions are predicted in one pass and reconstructed in order: no beam search, teacher forcing, or autoregressive feedback loop.
- Total trainable parameters: 2,148,676.

The adaptation is not an exact TNT reproduction. The paper uses normalization as pretraining for content moderation; this project uses paired Wolof data and evaluates the reconstructed normalized sentence itself.

Every synthetic condition starts from the same initialization, trains for 20 epochs on 3,330 synthetic pairs, resets the optimizer, and fine-tunes for 20 epochs on the same 142 Gold pairs. Development CER selects among scheduled checkpoints; the 29-row test remains frozen.

## 9. Did you try anything else? Aren't there better approaches?

**Short answer:** Yes. The final architecture is a controlled and computationally feasible evaluator, not a claim that no stronger normalizer exists.

- mT5-small full fine-tuning was too slow on the available laptop; rank-8 LoRA reduced trainable parameters but not enough of the end-to-end cost and early outputs were unstable.
- mBART was considered as a multilingual seq2seq alternative.
- NLLB is primarily optimized for translation, not same-language orthographic normalization.
- A compact autoregressive character Transformer learned sanity samples but accumulated decoding errors and repetition on sentences.
- A pointer/copy-aware variant increasingly copied the input instead of learning selective edits.
- M5 tested unsupervised lexical mining but had insufficient accepted coverage.

A stronger future study should use multiple evaluators: keep TNT as a fast interpretable probe, add a pretrained mT5/mBART model, and test a character–word hybrid. The same architecture must still be used across all generators to preserve causal attribution.

## 10. How are the metrics calculated, and why not ordinary accuracy or classification precision?

**Short answer:** Normalization produces a variable-length string, so edit-distance and character-overlap metrics are more informative than ordinary class accuracy. Precision, recall, and F1 are still reported—at the correction-edit level.

- **CER:** total character substitutions + deletions + insertions divided by the number of reference characters. Lower is better.
- **WER:** the analogous calculation on case-folded word-and-punctuation tokens. Lower is better.
- **chrF:** F-score over character n-grams of orders 1–6, with beta 2, reported on 0–100. Higher is better.
- **Exact match:** percentage of whole predictions identical to the reference after Unicode/whitespace normalization.
- **Correction P/R/F1:** compare exact source-span/replacement edit signatures with human edits.
- **Overcorrection:** false predicted edits divided by all predicted edits, `FP / (TP + FP)`.

Plain character or operation accuracy would be dominated by the majority `KEEP` class—81.1% in Gold train—and could reward a model that copies everything. Token accuracy is also ambiguous after splits and merges. CER and WER expose different failure scales; edit P/R/F1 measures whether the actual correction is right.

## 11. Why does Gold-only equal the identity baseline on every test metric? Was it trained?

**Short answer:** Yes, Gold-only was trained. Its selected checkpoint returned all 29 test inputs unchanged, so its predictions—and therefore every metric—are exactly identical to identity.

- It trained for 20 epochs on 142 Gold-train pairs from the same shared random initialization.
- Development was evaluated at epochs 5, 10, 15, and 20; epoch 5 was selected by CER.
- Gold train contains 81.1% `KEEP` character operations, and 142 sentences are sparse supervision for a model trained from scratch.
- The simplest generalization was therefore to copy unseen text.

This is a meaningful negative control: it shows that the character gains of the synthetic conditions came from synthetic pretraining, not merely from architecture or Gold fine-tuning. It does **not** prove that authentic data are unhelpful; it shows that this amount and setup were insufficient.

## 12. What does overcorrection tell us, and why is it important?

**Short answer:** It tells us how often the model's proposed edits do not exactly match a human-required edit. This matters because valid Wolof, French spans, and names should not be “corrected” into something worse.

- Overcorrection is `FP / (TP + FP)` over exact token-span edit signatures.
- Test values are very high: about 96.5% to 100% for the synthetic systems.
- The model may lower CER by changing a few characters yet still produce the wrong full token or miss a space; that edit remains a false positive under the strict definition.
- High overcorrection explains why character improvement does not imply a deployable corrector.

**Nuance:** The measure is deliberately strict. A partial improvement can be counted false even if it reduces CER, and alternative valid normalizations are not credited. Also, every test sentence needs correction, so this split cannot measure sentence-level safety on already-standard inputs.

## 13. If you continued the project, what would you do?

**Short answer:** First strengthen authentic evaluation, then improve generators and evaluators without touching the frozen test, and finally test the original downstream goal.

1. Collect more Gold pairs across channels, topics, time periods, and platforms; double-annotate a subset and allow multiple references.
2. Build a balanced test with unchanged, light, and heavy normalization plus explicit code-switch, entity, spacing, and length strata.
3. Calibrate M1–M3 edit frequency using Gold-train only; add grammatical and semantic filters to French substitutions.
4. Improve M6 hierarchy and spacing rules while keeping development/test annotations inaccessible.
5. Evaluate with multiple seeds and a second pretrained multilingual model.
6. Test the saved hybrid TNT + boundary + word-correction idea to preserve CER gains while improving WER.
7. Run downstream retrieval, translation, and LLM-query experiments with human relevance or translation judgments.

The decisive next step is not another decimal on 29 sentences; it is demonstrating that the observed character-level gains survive broader authentic data, stronger evaluators, and semantic tasks.

---

# High-risk follow-up questions

## Why generate formal→informal data if the trained task is informal→formal?

Because standardized sentences are available and informal parallel sentences are scarce. Each generator perturbs a formal target to create a synthetic informal source; training reverses that pair so the evaluator learns informal→formal normalization. Generation is the data-construction direction, not the deployment direction.

## Is M6 cheating by learning errors from Gold train?

No held-out target is used, so it is not leakage. However, it is a different supervision class and must be labelled Gold-train-informed. Its purpose is precisely to test whether a small authentic training sample can guide generation. The comparison is fair only with that qualification.

## Where is the evidence that systems fail on informal Wolof?

The translation screenshots and observed behaviour motivated the project, but they are demonstrations rather than a systematic claim about every commercial system. The controlled experiment tests a narrower proposition: whether synthetic pretraining helps reconstruct authentic informal comments. Broader claims about retrieval, translation, or LLMs remain future downstream evaluations.

## Does M4's formal corpus differ from the common pool?

The published Oolel data originally contain 3,430 unique formal targets, and all 3,430 occur in the larger project formal corpus. The controlled benchmark uses the same sampled 3,330 targets for every method, so formal target content and quantity are matched. M4 is still external because its original prompt, decoding, and row-level decisions are unavailable.

## Can Gold French prevalence be compared directly with M2/M3 switching rates?

Only descriptively. Gold says whether a sentence contains at least one manually identified French occurrence and gives a within-sentence occurrence ratio. M2/M3 traces say whether the generator inserted French. These statistics reveal exposure mismatch, but they are not estimates of the same probability.
