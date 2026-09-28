# PFE Final Defence Guide

## Files

- Presentation: `PFE_Final_Defence_Ibrahima_THIOYE.pptx`
- PDF preview: `PFE_Final_Defence_Ibrahima_THIOYE.pdf`
- Main presentation: slides 1--16
- Backup material for questions: slides 17--28
- Speaker notes are embedded in the PowerPoint file.

## Before the defence

- [ ] Replace the placeholder on slide 10 with the normalization/manipulation-recovery figure from the TNT paper, PDF page 4737.
- [ ] Add the TNT citation immediately below the inserted image.
- [ ] Confirm whether the jury expects the presentation in English or French.
- [ ] Rehearse once with a timer and aim to reach slide 12 by minute 14.
- [ ] Keep the complete report and frozen result CSVs available locally in case the jury asks for exact values.
- [ ] Test the presentation computer, fonts, Wolof characters, charts, Presenter View and slide notes.
- [ ] Export a final PDF after inserting the TNT image.
- [ ] Bring a second copy on a USB drive and an online copy if permitted.

## Twenty-minute pacing

| Time | Slides | Purpose |
|---:|---|---|
| 0:00--0:30 | 1 | State the practical problem and the project objective. |
| 0:30--2:30 | 2--3 | Explain the formal/informal gap, research question and positioning. |
| 2:30--4:50 | 4--5 | Explain collection, filtering, gold annotation and the split. |
| 4:50--6:00 | 6 | Present the main gold-data findings. |
| 6:00--8:40 | 7--8 | Introduce the five synthetic methods and one common example. |
| 8:40--10:00 | 9 | Compare Gold, M1 and M4 transformation supervision. |
| 10:00--12:50 | 10--11 | Justify the architecture, training protocol and metrics. |
| 12:50--16:00 | 12--13 | Present the main result and immediately qualify it using WER/F1. |
| 16:00--17:10 | 14 | Give one interpretable prediction example. |
| 17:10--18:40 | 15 | State limitations and future work. |
| 18:40--19:30 | 16 | Give the three conclusions and invite questions. |
| 19:30--20:00 | buffer | Pause, breathe and absorb a small delay without rushing. |

## Core narrative in five sentences

1. Informal Wolof differs orthographically from the formal Wolof found in most available resources, creating a plausible domain mismatch for NLP applications.
2. Because authentic paired data are scarce, I constructed a 201-sentence gold benchmark and compared five ways of generating synthetic informal inputs from formal targets.
3. Every method used the same 3,330 targets, edit Transformer, initialization, gold fine-tuning data, development selection rule and frozen test.
4. Synthetic pretraining improved character-level reconstruction, with M4 Oolel and M1 rules strongest, but word-level recovery and exact corrections remained weak.
5. The contribution is therefore a reproducible comparison framework and annotated evidence base, not a claim that the resulting normalizer is ready for deployment.

## Exact result values worth memorizing

| Condition | CER ↓ | WER ↓ | chrF ↑ | Correction F1 ↑ |
|---|---:|---:|---:|---:|
| Identity | 0.220 | **0.570** | 58.50 | 0.000 |
| Gold only | 0.220 | **0.570** | 58.50 | 0.000 |
| M1 rules | 0.195 | 0.607 | 56.73 | 0.028 |
| M2 Eflomal | 0.200 | 0.614 | 57.90 | 0.000 |
| M3 CMDR | 0.210 | 0.629 | 55.09 | 0.014 |
| M4 Oolel | **0.191** | 0.598 | **58.79** | 0.015 |
| M6 Gold-informed | 0.206 | 0.592 | 57.98 | **0.030** |

Interpretation:

- M4 has the best observed character-level result.
- Identity has the best WER.
- M6 has the best correction F1 among trained synthetic conditions.
- M1 produces the only exact normalized test sentence.
- The metric disagreement prevents a universal ranking.

## Architectural decisions: short answers

### Why character tokenization?

Informal variation is primarily orthographic and often creates unseen word forms. Characters preserve diacritics, repeated letters, spelling substitutions and spacing evidence without requiring a reliable Wolof word vocabulary.

### Why an edit model?

The target normally copies most of the input and changes only selected spans. KEEP, DELETE, SUBSTITUTE and EXPAND make that structure explicit. The non-autoregressive reconstruction also avoids cumulative decoding errors observed in earlier compact sequence-to-sequence experiments.

### Why not mT5?

mT5 offers stronger multilingual priors, but complete repeated fine-tuning and generation across all benchmark conditions was too slow on the available laptop. LoRA reduced trainable weights but not all backbone and decoding costs. The final 2.15M-parameter evaluator made a controlled comparison feasible.

### Why not mBART or NLLB?

mBART is a valid future evaluator. It was not added after the protocol was fixed because that would require a new complete matched run. NLLB is primarily a translation system, whereas this task preserves meaning and edits orthography within the same language.

### Why train from scratch?

It limits semantic knowledge, which is acknowledged. It also gives every condition exactly the same initialization and removes uncertain prior exposure as a confound. The conclusion is explicitly restricted to this evaluator.

### Why 3,330 synthetic rows?

M4 was the smallest retained source after validation. The formal-target intersection across all five methods contained 3,430 unique targets. A fixed-seed sample of 3,330 left an equal and comfortably valid set for each condition.

### Why twenty epochs?

The same fixed budget was assigned to every synthetic stage and every gold stage. Generated development CER was checked only at epochs 5, 10, 15 and 20. All epochs ran; selection did not stop training early.

## Metric decisions: short answers

### Why CER as the primary metric?

The task is dominated by character-level spelling variation, and CER gives credit for partial movement toward the reference. The model also predicts character edits directly, making CER aligned with the evaluator's intended sensitivity.

### Why report WER as well?

CER can improve while the output token remains wrong. WER tests whether complete words and boundaries were recovered. Its disagreement with CER is one of the project's main findings.

### Why chrF?

chrF measures character n-gram overlap and is less brittle than exact match. It complements CER but still does not prove semantic equivalence.

### What is correction F1?

A predicted edit is correct only if both its source span and replacement tokens match the human edit. It is intentionally strict and closer to the question of whether the system proposed the right correction.

### What does overcorrection mean here?

The reported rate is `false proposed edits / all proposed edits`. It is edit-level. Since every test sentence needs at least one correction, the benchmark cannot measure sentence-level preservation on already-standard text.

## Anticipated jury questions

### 1. Why informal-to-formal instead of formal-to-informal?

Formal-to-informal is used to manufacture training pairs because formal targets are available. Informal-to-formal is the downstream direction because the aim is to connect user text to resources and models built around formal Wolof. Normalization also offers an observable target and interpretable errors. A deployed system should preserve the original message and show normalization as an optional representation.

Use backup slide 22.

### 2. Why did you not directly evaluate semantic retrieval?

The original goal was retrieval robustness, but a valid retrieval benchmark requires informal queries, relevant-document judgments and enough independent topics. Those labels were not available. Text normalization became a controlled proxy and a way to compare data-generation methods first. Direct retrieval evaluation is the highest-value downstream extension.

Use backup slide 23.

### 3. Where does the assumption that models do not understand informal Wolof come from?

Do not defend that wording as a universal result. Say that it was an initial practical observation and a domain-mismatch hypothesis. The project demonstrates recurrent orthographic differences and measurable normalization difficulty, but it does not directly test semantic understanding, retrieval or LLM answers.

Use backup slide 24.

### 4. Is M6 cheating because it learns gold errors?

M6 uses only the gold training partition. Development and test annotations are excluded, video groups are disjoint, and exact held-out overlap is zero. This is not held-out leakage. However, M6 is a supervised train-informed method and is not described as a data-independent baseline.

Use backup slide 25.

### 5. Why does Gold-only simply copy the input?

It sees only 142 sentences, while 81.1% of aligned source positions in Gold train are KEEP. With a small randomly initialized model, copying is the easiest dominant behavior. Synthetic pretraining provides many more changed positions and recurring correspondences, which makes non-identity edits learnable.

### 6. Does M4 win?

M4 wins the main observed character metrics: CER and chrF. It does not win WER or correction F1. The test contains 29 sentences, one seed and one evaluator. Therefore M4 is the strongest observed character-level condition, not a universally superior generator.

Use backup slide 27.

### 7. Why is M1 almost as good as M4 despite being simple?

M1 supplies broad, consistent character correspondences, and its KEEP/DELETE/SUBSTITUTE distribution is relatively close to Gold train at aggregate level. The evaluator is character-level, so transparent character rules match its inductive bias. This does not mean every M1 sentence is natural.

### 8. Why do M2 and M3 not outperform M1?

Their code-switching components add substitutions that may be lexically related without being contextually natural. M2 changes almost every row through alignment and produces the strongest perturbation distribution. M3 is more selective but still inherits M1 noise. Extra complexity does not help when it introduces distribution mismatch.

### 9. Why is M6 not the best method if it learns from real errors?

M6 is too conservative. It assigns 95.6% of source positions to KEEP, compared with 81.1% in Gold train. It learns useful mappings and protection behavior, but sparse training evidence does not provide enough rule coverage, especially for spacing merges and sentence-level edit density.

### 10. Why did you exclude M5?

After removing development and test videos, only 114 mappings passed its conservative filters. It changed 1,821 of 17,777 sentences and left 15,956 unchanged. Training it beside the other methods would mostly compare identity exposure rather than a substantive synthetic generator. It remains documented as a negative result.

### 11. Is the 29-sentence test statistically meaningful?

It supports a pilot result, not population generalization. Paired bootstrap intervals quantify sensitivity to resampling these 29 sentences; they do not cover new videos, optimization seeds or other platforms. This is why small decimal differences are not treated as a definitive ordering.

Use backup slide 21.

### 12. Why is the split video-disjoint?

Comments from the same video share topic, names and community style. Keeping complete videos in one partition reduces direct contextual leakage. The trade-off is that development and test each contain only one video, so topic and split are confounded.

### 13. Did you protect entities and French spans?

Yes. The gold policy preserves legitimate French material and names. M1 masks entities, acronyms and numbers. M6 explicitly uses language, named-entity and protection labels. M2 and M3 protect inserted French spans from Wolof spelling rules, but M2's final export did not apply the optional named-entity model, which remains a limitation.

### 14. Can more synthetic data solve the problem?

Not automatically. More realizations can improve coverage, but repeating systematic or unnatural transformations amplifies mismatch. The next experiment should vary quality and diversity while holding total updates constant, rather than simply increasing row count.

### 15. What is the strongest contribution?

The strongest contribution is the complete evaluation chain: authentic gold annotation, occurrence-level error analysis, traceable independent generators, matched targets, a shared evaluator, frozen checkpoint selection and one-time test evidence. It turns an under-specified idea—“generate informal Wolof”—into a reproducible comparison.

## Phrases to use carefully

Prefer:

- “plausible domain mismatch” instead of “models do not understand informal Wolof”;
- “strongest observed character-level result” instead of “best method”;
- “train-informed supervised method” instead of “unsupervised M6”;
- “improves CER” instead of “solves normalization”;
- “normalization proxy” instead of “retrieval improvement”;
- “no held-out leakage” instead of “completely fair in every sense.”

Avoid:

- claiming statistical significance beyond the observed test sample;
- calling the model a production corrector;
- equating source-to-target distance with naturalness;
- describing M4's undocumented transformations as known linguistic rules;
- presenting Gold-dev as training data—it selected checkpoints but did not contribute gradients.

## If time starts running out

- Keep slides 1--7 intact because they establish the question and fairness.
- Present slide 8 with only M1, M4 and M6 examples.
- On slide 9, state only that M1 and M4 have the closest shared edit balance to Gold.
- On slides 10--11, give one sentence per architectural decision and metric family.
- Never skip slides 12--13: the result and its qualification belong together.
- Summarize slide 14 in one sentence, then move to limits and conclusion.

## Final thirty-second answer template

For an unexpected question:

1. State what the project actually measured.
2. Give one relevant number or protocol safeguard.
3. State the limit of the evidence.
4. Explain the next experiment that would answer the broader version.

Example: “In this project I measured normalization utility with a fixed edit Transformer, not retrieval quality. M4 reduced CER from 0.220 to 0.191 on the frozen test, but WER remained worse than identity. With only 29 test sentences and one evaluator, I interpret that as character-level transfer. A direct retrieval benchmark with informal queries and relevance judgments would test the broader claim.”
