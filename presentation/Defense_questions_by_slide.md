# Defence questions by main slide

The full suggested answers also appear in the PowerPoint speaker notes. These are short revision prompts for the 20-minute presentation.

## 1 — Title and scope

- What is the final deliverable? — A controlled comparison of synthetic-data generation methods and an auditable pipeline; the normalizer is the evaluator, not the final product.
- Is this a production spell-checker? — No. It is a pilot benchmark built to estimate synthetic-data utility.

## 2 — Motivation

- Where does the claim that systems struggle with informal Wolof come from? — From observed translation failures, the formal-resource bias found in the review, and the orthographic divergence measured in the gold corpus. It is treated as a hypothesis, not a universal fact.
- Why not benchmark Google and DeepL directly? — Their systems are uncontrolled and change over time; normalization provides reproducible paired references.

## 3 — Research question and contributions

- Why use normalization instead of retrieval? — It isolates whether synthetic pairs teach recoverable mappings. Retrieval remains the downstream motivation and future validation task.
- What is genuinely new? — The authentic benchmark, matched comparison, linguistic analysis and M6 engine, rather than a new Transformer architecture.

## 4 — State of the art

- Why keep only Beqi and Oolel? — They are the two precedents directly instantiated in the final benchmark. Other explored papers did not enter the final experiment.
- Is Oolel directly comparable with M1? — Its saved data utility is comparable under the same evaluator, but its original generation process is less reproducible.

## 5 — End-to-end pipeline

- Why are generation and training directions opposite? — The methods create informal inputs from formal targets; the evaluator then learns the inverse normalization mapping from each pair.
- Is M6 data leakage? — No held-out labels are used. M6 learns only from gold train and is reported as supervised augmentation; gold-only controls the same data without synthetic pretraining.
- Why 3,330 synthetic pairs? — That is the common matched formal-target set available to every retained method.

## 6 — Data used

- Does Oolel overlap cause leakage? — No. Its formal targets overlap the synthetic-training pool, not the authentic video-held-out test set.
- What does Oolel add? — Alternative informal realizations; all 3,430 unique formal targets already occur in the 17,777-pair pool.
- Why retain French–Wolof pairs? — M2 and M3 require the paired French sentence to propose code-switched substitutions.

## 7 — Gold construction

- Why only 201 gold pairs? — Complete manual reconstruction and linguistic review are costly; the study fixed a feasible pilot scope.
- Why split by video? — To reduce contextual leakage between train and evaluation, although one-video dev/test splits remain topic-sensitive.
- Was intentional French translated? — No. It was preserved because the task is normalization, not translation.

## 8 — Annotation analysis

- What is POS? — Part of speech: the grammatical class of a token, such as verb, noun, pronoun or proper noun.
- Can a token have multiple error labels? — Yes. One form may combine phonetic spelling, deletion, repetition or spacing errors.
- Why protect French words and entities? — They are frequently valid unchanged spans; indiscriminate Wolof rules would create false edits.

## 9 — Benchmark family map

- Why are M2 and M3 extensions of M1? — They first insert French substitutions, protect them, and then apply M1’s orthographic rules to the remaining Wolof.
- Why include gold-only? — It measures what 142 authentic training pairs achieve without synthetic pretraining.
- Why include identity? — It checks whether training improves over returning the input unchanged.

## 10 — M1 and M2

- Why use rules? — They are fast, transparent and encode known Wolof spelling variation.
- Why can M2 fail despite high alignment probability? — Alignment indicates sentence-level translation correspondence, not grammatical substitutability in a local context.
- How are entities treated? — M1 masks detected entities; M2 also protects inserted French before applying Wolof rules.

## 11 — M3 and M4

- Is CMDR a translation model? — No. It ranks local phrase substitutions drawn from the row’s paired French sentence.
- Why can CMDR exchange names? — Distributional similarity reflects shared contexts, not semantic identity or safe substitution.
- Can Oolel be reproduced exactly? — The saved dataset is inspectable, but its original prompt, random seed and decoding settings are unavailable.

## 12 — M6

- Is learning from gold train cheating? — No development or test annotations are used. It is explicitly a supervised augmentation condition, not a data-independent method.
- Why not train a seq2seq generator on 142 pairs? — The sample is too small for reliable free generation; structured mappings expose evidence and constrain errors.
- Why are spacing merges underrepresented? — Few merge rules passed the mapping gates, so they occur in only about 0.4% of matched M6 rows.

## 13 — Evaluator architecture

- Why not mT5 or mBART? — Pilot runs were too slow for an equal multi-method comparison on the available hardware. The compact model made the benchmark feasible.
- Why character-level modeling? — Informal spellings create unseen word forms; character edits directly represent the main variation.
- Does it understand semantics? — Only weakly. It measures orthographic transfer and cannot establish semantic correctness.
- Is synthetic “pretraining” the same as language-model pretraining? — No. It is the first supervised training stage before authentic fine-tuning.

## 14 — Training protocol

- Why select checkpoints on gold development? — Selection should reflect the authentic target domain, not each generator’s own synthetic distribution.
- Why equal synthetic sample sizes? — To prevent corpus size from confounding the method comparison.
- Why only one training seed? — Time and hardware constraints; bootstrap intervals cover test-sentence uncertainty, not training variance.

## 15 — Metrics

- Why not BLEU? — CER, WER and chrF better match short, character-heavy normalization changes.
- Why measure overcorrection? — A normalizer must also preserve names, legitimate French and correct context.
- Is every false exact edit linguistically wrong? — No. It may be partially useful, but it does not match the annotated source span and replacement.

## 16 — CER results

- Why does gold-only equal identity? — Its selected checkpoint copies every frozen-test input; this reflects the small-data model behavior, not the uselessness of authentic data.
- Is M4 significantly better than M1? — No definitive significance claim is made because their bootstrap intervals overlap.
- What does lower CER prove? — Useful character-level transfer, not recovered words, meaning or exact edits.

## 17 — WER and chrF

- Why can CER improve while WER worsens? — A helpful character edit lowers CER, but WER still counts the entire token as wrong until it exactly matches.
- Is M4’s chrF gain meaningful? — It is a small point-estimate gain of 0.29, not decisive evidence.
- Which method is best overall? — None dominates: M4/M1 lead CER; M6 is best among synthetic conditions on WER and correction F1.

## 18 — Overcorrection and failure analysis

- How can 96.5% be the best overcorrection rate? — It is only relatively best; every method remains poor under strict exact-edit scoring.
- Why omit identity and gold-only? — They propose no edits, so the rate has a zero denominator; displaying zero would falsely imply safety.
- Can a false edit still help? — Yes. It may reduce character distance or be linguistically plausible while missing the exact reference.
- Can the study measure safety on already-correct sentences? — No. Every frozen-test input requires at least one correction.

## 19 — Limitations and perspectives

- What should be improved first? — Expand and independently review the authentic corpus across more channels and topics.
- Why not immediately use a much larger model? — A larger model cannot compensate for weak evaluation coverage and would add pretrained-knowledge confounds.
- Is the benchmark still useful? — Yes as a controlled pilot and diagnostic framework, but not as a population-level estimate.

## 20 — Conclusion

- Which method would you recommend? — M4 for the best CER point estimate, M1 for transparency and reproducibility, and M6 as a promising selective-noise direction; none is ready for unsupervised deployment.
- What is the main scientific lesson? — Synthetic utility depends on matching authentic edit frequency, context and protection—not simply maximizing noise.
- Did the project solve the original retrieval problem? — It built and evaluated the normalization bridge; direct semantic-retrieval evaluation is the next step.
