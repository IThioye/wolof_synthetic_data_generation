# Synthetic Data for Robust Informal Wolof Processing

> **Drafting status.** Sections 1 to 3 are a first complete report draft based on the project artifacts available on 26 July 2026. Text marked `[TO COMPLETE]` requires personal or organizational information that is not present in the repository. Quantities marked as a current snapshot must be updated before submission.

## 1. Introduction

### 1.1 General context and issues

Natural language processing has progressed rapidly through large pretrained models, multilingual representations, and sequence-to-sequence architectures. However, this progress is unevenly distributed across languages and domains. Models perform best when their training data resemble the inputs observed during use. When the training and inference distributions differ, even a technically strong model can fail for reasons that are not primarily semantic. This problem is particularly visible for low-resource languages, for which available corpora are limited and concentrated in a small number of formal sources.

Wolof is a representative case. It is widely spoken in Senegal and is also used in neighboring countries and diaspora communities, yet the digital resources reviewed during this project are predominantly dictionaries, translations, news-like text, or other relatively formal material. In contrast, users communicating on social-media platforms write Wolof in diverse ways. Their messages may contain phonetic spellings, omitted diacritics, character substitutions, abbreviations, repeated letters, unconventional word boundaries, expressive punctuation, and French-Wolof code-switching. These forms are not necessarily random mistakes. Some reflect typing convenience, the influence of speech, community conventions, multilingual identity, or the limited availability of convenient Wolof keyboard layouts. Al Sharou et al. warn that non-standard content should not automatically be treated as meaningless noise because it may contain useful linguistic and social information [1].

This difference creates a domain gap between the language represented during training and the language encountered in real use:

```text
Training distribution                Real-use distribution
-------------------------------      --------------------------------
formal and standardized Wolof   ->   informal and code-switched Wolof
dictionary/news/translation text     social-media and user-written text
consistent spelling                  variable, phonetic spelling
mostly controlled language           French insertions and online conventions
```

The consequences extend beyond spelling correction. In semantic retrieval, an informal query may not retrieve a relevant document written in formal Wolof even when the two express the same intent. An LLM may misunderstand an informal prompt because its spelling and code-switch patterns are weakly represented in pretraining. A machine-translation system may translate a formal sentence correctly but behave unpredictably when the same content is written phonetically or mixed with French. Tokenization and language identification also become harder in multilingual social-media text; Das and Gambäck showed that code-mixing, borrowing, and phonetic typing challenge language boundaries that appear straightforward in cleaner domains [2]. For Wolof specifically, Dione demonstrated that language-specific tokenization and normalization rules are important even for processing formal natural text, notably because of clitics, multiword expressions, and word-boundary phenomena [3]. Informal text amplifies these difficulties.

The initial motivation of this project was therefore to facilitate semantic retrieval over new Wolof corpora by reducing the mismatch between formal documents and informal user queries. The same bridge could support LLM querying and machine translation. A direct solution would be to adapt each downstream model using a large parallel corpus in which every informal sentence is paired with a meaning-equivalent formal sentence. Such a resource is not currently available at the required scale. Manually constructing it would require fluent annotators, a stable normalization policy, extensive quality control, and considerable time.

The project consequently shifted from a single retrieval application to the underlying data problem: **how can useful formal-informal Wolof training pairs be created when naturally paired data are scarce?** Synthetic data generation offers a possible answer. Starting from available formal Wolof resources, a generator can introduce realistic spelling variation, social-media conventions, and controlled code-switching. The resulting pairs can train a model in the reverse direction, from informal input to normalized text. Mbaye and Diallo follow a related principle in Beqi, modeling Wolof spelling correction as noisy-to-correct sequence transduction and using synthetic data to address the lack of paired errors [4]. Their work supports the feasibility of synthetic augmentation for Wolof, but it does not resolve how to reproduce the informal and code-switched patterns found in a specific real-world corpus.

This PFE investigates that remaining question. It compares several ways of generating synthetic informal Wolof:

- generic hand-written spelling and noise rules;
- alignment-based or lexical code-switching methods;
- an external LLM-generated baseline created with the Wolof-specialized Oolel model;
- unsupervised lexical variants mined from unpaired YouTube comments;
- a proposed standalone linguistic engine using patterns learned only from the training portion of manually normalized data.

The project deliberately separates **synthetic plausibility** from **synthetic usefulness**. A generated sentence can look informal but change the source meaning; conversely, a sentence that does not perfectly imitate a human comment may still teach a normalizer an important correspondence. The principal evaluation is therefore downstream: models trained under controlled conditions on each synthetic dataset will be tested on the same untouched set of manually normalized YouTube comments. Intrinsic data statistics and human inspection complement this evaluation but do not replace it.

The work is conducted in the context of `[TO COMPLETE — host organization]`, an organization operating in `[TO COMPLETE — sector and principal activities]`. Within this context, the project contributes to `[TO COMPLETE — the organization's data/AI objective or research program]`. The organizational presentation is developed in Section 3.1; it must be completed with validated information about the host structure, team, and supervision.

### 1.2 Problem statement and objectives

#### 1.2.1 Problem statement

The operational problem can be summarized as follows:

> Wolof NLP resources and models are predominantly exposed to formal text, whereas real users frequently produce informal and French-Wolof code-switched text. Large paired formal-informal corpora are unavailable, making supervised adaptation difficult. It is therefore necessary to determine whether synthetic formal-to-informal data can bridge this distribution gap and improve processing of authentic informal Wolof.

The central research question is:

> **Can a synthetic data generator informed by real YouTube comments produce training data that improves normalization of authentic informal and code-switched Wolof more than generic rule-based, alignment-based, or external LLM-generated data?**

This question is divided into six primary research questions:

1. **RQ1 — Generic augmentation:** Does training on rule-generated informal Wolof improve normalization of real YouTube comments compared with a model that receives no synthetic augmentation?
2. **RQ2 — LLM-generated data:** How does the external Oolel-generated dataset compare with explicit rule-based generation, and how much of any difference may be due to its different formal source corpus?
3. **RQ3 — Unsupervised domain adaptation:** Do lexical variants mined from unpaired YouTube comments improve on generic linguistic rules without using manual correction pairs?
4. **RQ4 — Gold-informed generation:** Does a standalone generator learned from gold-training linguistic annotations improve on the unsupervised YouTube-aware method or generic rules?
5. **RQ5 — Synthetic initialization:** After identical gold fine-tuning, does initialization on each synthetic method improve normalization compared with gold-only training?
6. **RQ6 — Error coverage:** Which spelling, spacing, lexical, and code-switching categories are helped or harmed by each method?

A secondary question may be studied only if the primary normalization benchmark is completed:

7. **RQ7 — Retrieval transfer:** Does normalizing an informal Wolof query improve retrieval of a semantically relevant formal Wolof passage?

This formulation makes the hierarchy of the project explicit. Robust semantic access is the long-term goal; synthetic data generation is the proposed strategy; normalization is the main controlled task used to measure whether the strategy works. Full retrieval-augmented generation is not used as the primary evaluation because it would combine three separate sources of error: query representation, document retrieval, and answer generation. A focused retrieval experiment can later isolate the first two factors using fixed relevance judgments.

#### 1.2.2 Hypotheses

The experimental design is based on the following hypotheses, which must be tested rather than assumed:

- **H1:** exposing a normalizer to synthetic informal variants will improve its robustness compared with gold-only training or a copy baseline;
- **H2:** Oolel-generated data will be more contextually varied than local rules, but greater apparent naturalness will not necessarily produce better downstream normalization;
- **H3:** transformations extracted from the real target domain will cover phenomena that generic rules and the external LLM dataset miss;
- **H4:** a small amount of gold-training supervision will improve the precision and distribution of generated transformations, provided that test information is strictly excluded;
- **H5:** gains may be error-specific rather than uniform, meaning that aggregate metrics must be supported by category-level analysis;
- **H6:** a method that appears more natural during human inspection is not necessarily the method that produces the best downstream model.

The report will not claim in advance that the proposed method is superior. A marginal, neutral, or negative result is scientifically useful if it is obtained under a fair comparison and accompanied by an analysis of data quality, benchmark size, and error coverage.

#### 1.2.3 SMART objectives

The objectives are formulated to be specific, measurable, achievable within the PFE, relevant to the initial problem, and time-bound by the final submission date.

| ID | SMART objective | Measurement and acceptance criterion | Current status |
|---|---|---|---|
| O1 | Build a documented corpus of authentic informal Wolof candidates from public YouTube comments. | Preserve provenance; report raw and cleaned counts; make collection, cleaning, filtering, and encoding steps reproducible. | Substantially completed: 396,559 raw and 242,174 cleaned comments. |
| O2 | Construct a manually normalized gold benchmark with a consistent target policy. | Reach at least 200 kept pairs across at least 5 videos; create video-disjoint train, development, and test splits with a locked manifest. | Completed: 201 kept pairs from 5 videos, split into 142 train, 30 development, and 29 test pairs. |
| O3 | Implement comparable synthetic generation methods. | Export rules-only, Oolel LLM-generated, unsupervised YouTube-aware, and standalone gold-informed linguistic data under one traceable schema; retain an alignment/CMDR baseline if its quality is sufficient. | All generator implementations now exist; M6 export and the remaining manual quality gates are pending. |
| O4 | Control data leakage and experimental confounders. | Exclude test annotations and test-video comments from method development; use the same normalized source pool, dataset sizes, architecture, tuning budget, and seeds across project-controlled methods; label the published Oolel data as an external-source comparison. | Split and manifest tooling implemented; final split not yet possible. |
| O5 | Train and compare downstream normalization models. | Compare gold-only with identically configured synthetic-pretraining-plus-gold conditions using the same sequence-to-sequence architecture. | Versioned automatic two-stage runner and fixed configuration implemented; long-running model execution pending. |
| O6 | Evaluate synthetic data from complementary perspectives. | Report CER, WER, chrF, exact match, token-normalization F1, overcorrection, intrinsic distributions, and blinded human review. | Automatic scorers, result aggregation, Markdown tables, and plots implemented; final scores and human ratings pending. |
| O7 | Produce a reproducible technical and scientific report. | Maintain code, configurations, data lineage, checks, result artifacts, and an approximately 50-page final report by `[TO COMPLETE — submission date]`. | Documentation and report drafting in progress. |
| O8 | Optionally validate transfer to semantic retrieval. | Build a small manually verified informal-query/formal-document relevance set and compare retrieval before and after normalization. | Optional; begins only after O1–O7 are secured. |

#### 1.2.4 Expected contributions

The expected contributions are both operational and scientific:

1. a documented corpus of authentic informal and code-switched Wolof comments;
2. an annotation workflow and leakage-aware gold benchmark for Wolof normalization;
3. a controlled comparison of synthetic generation strategies;
4. an unsupervised lexical-variant miner using unpaired target-domain comments;
5. an auditable gold-train linguistic generator whose transformations retain provenance and error categories;
6. empirical evidence about whether and when synthetic data improves normalization of real Wolof text;
7. a methodology that can later be transferred to retrieval, LLM-input robustness, and machine translation.

#### 1.2.5 Scope boundaries

The PFE does not attempt to solve every aspect of informal Wolof processing. Its core scope excludes a full production RAG system, a new foundation model, a complete Wolof-to-French translation system, and large adversarial or cycle-consistent training unless the controlled baselines are completed early. These approaches are relevant to the state of the art, but they would substantially increase implementation risk and make it harder to isolate the effect of synthetic data quality. The primary deliverable is a rigorous synthetic-data benchmark, not a claim that one complex architecture solves all downstream applications.

### 1.3 Report outline

The remainder of this report is organized as follows.

**Chapter 2 — State of the art** introduces the theoretical foundations needed to understand the project: domain shift, non-standard text, lexical normalization, code-switching, synthetic data generation, sequence-to-sequence learning, multilingual Transformers, word alignment, and evaluation. It critically compares Wolof-specific work with normalization, code-mixing, and unsupervised style-transfer approaches in other languages. It then positions the proposed method relative to these approaches.

**Chapter 3 — Mission context and contributions** presents the host organization and project environment, defines the scope of the mission, distinguishes completed contributions from planned experiments, and describes the data, organizational, ethical, technical, and scientific constraints.

**Chapter 4 — Methodology and technical approach** gives the reproducible project pipeline. It details comment collection, cleaning, LLM-assisted filtering, annotation, formal-data preparation, rules-only generation, lexical alignment, the external Oolel LLM-generated baseline, unsupervised YouTube mining, the standalone gold-train linguistic generator, gold splitting, downstream training, and evaluation. It also specifies software dependencies, hardware, seeds, schemas, and hyperparameters.

**Chapter 5 — Results and analysis** reports the final corpus statistics, mapping quality, human evaluations, and normalization results. All methods are compared in matched tables. Error-category analysis and ablations identify which transformations account for any improvement and which errors remain unresolved.

**Chapter 6 — Conclusion and outlook** answers the research questions, compares the achievements with the initial objectives, discusses limitations and threats to validity, and proposes future extensions to semantic retrieval, LLM robustness, machine translation, and additional Wolof data collection.

The appendices contain non-essential implementation details, configuration tables, representative code extracts, annotation material, additional results, and environment instructions.

## 2. State of the art

### 2.1 Theoretical foundations

#### 2.1.1 Low-resource NLP and domain mismatch

A language is described as low-resource when the data, annotations, tools, and pretrained representations available for it are insufficient for a target task. This notion is relative: a language may have enough text for simple vocabulary construction but lack sentence-aligned data, annotated social-media text, evaluation benchmarks, or pretrained coverage. Wolof has formal lexical and translation resources, yet the paired data required to model informal writing remain scarce. The relevant shortage in this PFE is therefore not merely the absolute number of Wolof sentences; it is the absence of representative **formal-informal pairs** and trustworthy held-out annotations.

The problem is compounded by domain mismatch. Let a model be trained on samples from a source distribution \(P_s(x,y)\), where \(x\) is primarily formal text, while deployment inputs follow a target distribution \(P_t(x,y)\), where \(x\) is informal and code-switched. When \(P_s(x) \neq P_t(x)\), patterns learned on the source may not generalize to the target. For semantic systems, this can manifest as different tokenization, weak lexical overlap, unreliable embeddings, or incorrect generation even when the intended meaning is unchanged.

There are three broad responses to this mismatch:

1. **normalize the input**, converting target-domain text toward the formal distribution;
2. **adapt the model**, exposing it to target-domain variants during training;
3. **learn invariant representations**, so formal and informal variants map to similar semantic representations.

This project primarily combines the first two. Synthetic informal text adapts a normalizer, and the normalizer acts as a bridge from real user language to formal resources. A future retrieval experiment can test whether this bridge also improves semantic matching.

#### 2.1.2 Non-standard text and the meaning of “noise”

Social-media processing often groups spelling variation, abbreviations, emojis, code-switching, and irregular punctuation under the term *noise*. This terminology is useful when a phenomenon damages task performance, but it can conceal important distinctions. Al Sharou et al. distinguish harmful corruption from non-standard content that carries meaning or style [1]. For example, a duplicated character may be accidental or may express emphasis; a French word may be a meaningful code-switch rather than an error; and the absence of a Wolof diacritic may reflect keyboard constraints rather than lack of linguistic competence.

This distinction directly affects the annotation target. A normalizer should correct the phenomena defined as normalization errors while preserving communicative content. Over-cleaning may remove identity, emphasis, named entities, or intentional French spans. Under-cleaning may leave variants that prevent a downstream system from connecting the input to formal resources. The project therefore requires an explicit policy rather than a generic “clean all noise” instruction.

The recommended policy is to normalize informal Wolof orthography and social-media artifacts while preserving intentional French spans. This defines the task as code-switched normalization rather than translation to monolingual Wolof. The alternative—translating every French segment—would require stronger bilingual semantic judgments and would conflate normalization with machine translation.

#### 2.1.3 Text normalization, spelling correction, and style transfer

Lexical normalization maps a non-standard token or sequence to a standard form. Traditional spell checking often relies on edit distance, dictionaries, language models, or noisy-channel reasoning. These methods can be effective for typographical errors, but social-media variants are not always small accidental edits. Abbreviations and phonetic forms may differ substantially from their formal equivalents, and context may be required to resolve ambiguity.

Normalization can also be modeled as sequence transduction:

\[
\hat{y} = \arg\max_y P(y \mid x),
\]

where \(x\) is an informal sentence and \(y\) is its normalized target. This formulation permits character, token, and sentence-level corrections within one model. Beqi applies this perspective to Wolof spelling correction using synthetic noisy-correct pairs and sequence-to-sequence models [4]. The present project extends the scope from isolated spelling corruption toward the mixed phenomena found in YouTube comments, including code-switching and real-domain lexical patterns.

Formal-to-informal generation can alternatively be viewed as text style transfer. Style-transfer research attempts to change an attribute while preserving attribute-independent content [15]. In this project, the source attribute is formal/standardized writing and the target attribute is informal/code-switched writing. The analogy is useful because it emphasizes meaning preservation. It is imperfect, however: many “informal” changes are orthographic mappings rather than free stylistic generation, and French insertion can introduce translation and grammatical constraints. Consequently, this work uses style transfer as a conceptual perspective, not as proof that a complex unsupervised style-transfer architecture is necessary.

#### 2.1.4 Code-switching and word-level language ambiguity

Code-switching is the use of elements from more than one language within a discourse, sentence, or constituent. In digital multilingual communities it can occur alongside phonetic spelling, transliteration, borrowing, and ambiguous tokens. Das and Gambäck show why social-media language identification is harder than document-level language identification: the unit of analysis may need to be the token, while short tokens can belong to several languages [2].

For French-Wolof text, an automatic generator must decide which spans may be substituted without changing the intended meaning or producing an implausible sentence. A purely random dictionary replacement ignores context and grammar. A strict language identifier may misclassify shared or borrowed words. Named entities, numbers, URLs, acronyms, and already intentional French spans should normally be protected. These considerations motivate the use of confidence thresholds and explicit rejection rather than forcing every formal sentence to contain a switch.

Linguistic theories such as the Equivalence Constraint and Matrix Language Frame approaches attempt to describe permissible switching positions. They can inform a generator, but operationalizing them requires reliable parsing and language-specific grammatical resources. For Wolof-French, those resources are limited. This makes lightweight lexical or alignment-based baselines more feasible for the PFE, provided their limitations are acknowledged.

#### 2.1.5 Synthetic data generation as data augmentation

Synthetic data generation replaces an unavailable observed training pair with an automatically constructed pair. In the present direction, a formal target \(y\) is transformed by a generator \(g\) into a synthetic informal source \(\tilde{x}=g(y;\theta,z)\), where \(\theta\) represents rules or learned mappings and \(z\) controls random choices. The pair \((\tilde{x},y)\) then trains an informal-to-formal model.

Synthetic generation has several advantages:

- formal monolingual or parallel text can be reused at scale;
- transformations are controllable and reproducible;
- rare error categories can be deliberately represented;
- every generated source has a known clean target.

Its weaknesses are equally important:

- generated errors may be unrealistic or too regular;
- the generator may corrupt meaning or protected tokens;
- synthetic and real error distributions may differ;
- a downstream model may learn generator artifacts rather than general normalization;
- methods can appear better simply because they use more data or easier examples.

Dekker and van der Goot demonstrate that custom error generation can approach training on manually annotated normalization data for English, while an unsupervised embedding-based approach also performs competitively [8]. Their findings justify both supervised error modeling and annotation-free mining as baselines. They also show why the comparison must include downstream performance rather than only generated-text appearance.

#### 2.1.6 Unsupervised lexical-variant mining

Unsupervised normalization methods exploit large unlabeled corpora to identify relations between non-standard and standard forms. Gouws et al. mine domain-specific lexical variants from noisy text without manually paired data and report a reduction in word error rate relative to their baseline [5]. Li and Liu combine surface similarity with semantic evidence from continuous word representations and use discriminative reranking to combine candidate systems [6]. Roy et al. propose a language-independent unsupervised normalization approach and evaluate its effect on information retrieval and stance detection [7]. The last point is particularly relevant to the original motivation of this PFE: normalization should be assessed by whether it improves a downstream task, not solely by token-level accuracy.

These approaches suggest a YouTube-aware method that begins with frequent out-of-vocabulary comment tokens, proposes formal candidates through weighted character similarity and Wolof-oriented transformations, and ranks them with frequency and context. However, their transfer to Wolof is not automatic. Distributional representations may be unreliable for rare tokens, formal vocabularies may be incomplete, and French words or names may appear out of vocabulary even though they should not be normalized. High-confidence filtering and manual precision estimation are therefore necessary.

#### 2.1.7 Sequence-to-sequence models and Transformers

Encoder-decoder models learn a conditional transformation from an input sequence to an output sequence. Recurrent LSTM-based models were widely used for this purpose, including spelling correction. Transformers replaced recurrence with self-attention, allowing each position to model relationships with other positions and enabling more parallel training [9]. This architecture is well suited to normalization because a prediction may depend on sentence context rather than an isolated token.

Pretrained text-to-text models reduce the amount of task-specific data required. T5 represents different NLP problems through a unified text-to-text formulation [10]. mT5 extends this framework to 101 languages using multilingual pretraining [11], while mBART pretrains a multilingual sequence-to-sequence denoising autoencoder and reports strong transfer in low-resource translation settings [12]. Pretraining does not guarantee strong Wolof performance: coverage, tokenizer segmentation, corpus quality, and the amount of Wolof observed during pretraining still matter. Nevertheless, these models offer a more plausible starting point than training a large Transformer from scratch on a few hundred gold pairs.

Jawahar et al. provide a relevant code-mixing example. They use mT5 and mBART for English-to-Hinglish generation and report benefits from a curriculum that first fine-tunes on synthetic code-mixed data and then on gold data [13]. The project adopts this training structure as a hypothesis to test: every retained synthetic method initializes an otherwise identical model before identical gold fine-tuning, and its final result is compared with gold-only training.

Tokenization requires careful treatment. Informal forms create rare or unseen surface tokens, so subword segmentation can help share information across variants. Beqi reports that subwording choices materially affect Wolof spelling-correction performance [4]. This motivates using the selected pretrained model's subword tokenizer consistently and reporting its behavior. It does not justify claiming that a fully character-level tokenizer is essential without a separate ablation.

#### 2.1.8 Lexical alignment for controlled code-switching

Parallel French-Wolof data can provide lexical correspondences. Statistical word alignment estimates which source and target tokens are likely translations. IBM-style models provide a classical probabilistic foundation for estimating translation and alignment parameters [20], while Eflomal uses efficient Bayesian alignment with Markov chain Monte Carlo [14]. Alignment is appropriate when the immediate task is to extract candidate word or phrase correspondences; a full translation model is unnecessary solely for building a lexicon.

The value of an alignment-derived lexicon depends on corpus size, preprocessing, direction, and ambiguity. Frequent function words may receive diffuse mappings, and one-to-many translations can be reduced incorrectly to one substitute. Alignment probabilities therefore do not guarantee that a replacement is grammatical in a new sentence. In this project, Eflomal and IBM Model 1 are treated as resources for a controlled baseline, not as ground truth. Mappings and generated examples require validation.

#### 2.1.9 Evaluation of normalization and synthetic quality

No single metric captures the project objective. Character error rate is sensitive to spelling corrections, while word error rate captures token-level edits and spacing changes. Exact match is strict but interpretable. chrF gives a character n-gram similarity that is often more tolerant of morphology and small orthographic differences. Token-level normalization precision, recall, and F1 distinguish successful corrections from missed errors, and overcorrection measures damage to tokens that should remain unchanged.

These metrics also have limitations. A single reference may penalize another valid normalization. A high similarity score may hide a semantic substitution, while exact match may over-penalize punctuation. Consequently, the benchmark should combine automatic metrics, error-category analysis, and manual review.

Intrinsic synthetic-data evaluation should report edit counts, changed-token rates, rejection and duplication, mapping coverage, code-switch frequency, and similarity to gold-training error distributions. Human evaluation should assess meaning preservation, plausibility, corruption severity, and artifacts. Most importantly, the same downstream architecture and data budget must be used for all methods. Otherwise, a performance difference cannot be attributed to generation quality.

### 2.2 Related work

#### 2.2.1 Wolof normalization and linguistic preprocessing

Wolof-specific normalization research establishes both the relevance of the task and the limitations of generic tools. Dione's finite-state tokenizer integrates language-specific rules for clitics, multiword expressions, and normalization within a Wolof grammar pipeline [3]. This work shows that token boundaries and normalization cannot always be handled by language-independent preprocessing. Its finite-state approach is interpretable and linguistically grounded, but its goal is processing natural text for a formal grammar rather than learning the broad variation of contemporary social media.

Beqi is the closest direct precedent to this PFE [4]. Mbaye and Diallo address the absence of a Wolof noisy-correct parallel corpus by generating synthetic errors and training LSTM and Transformer spelling-correction models. The study demonstrates that synthetic data can support Wolof correction and that segmentation choices matter. The present project differs in four respects. First, its target domain is authentic YouTube writing. Second, it includes French-Wolof code-switching and protected spans. Third, it compares several generators using one locked real benchmark. Fourth, it asks whether patterns observed in unpaired and gold-training YouTube data improve on generic rules.

These differences are not claims that the present approach is automatically stronger. Beqi benefits from a focused spelling-correction definition, while the broader project target introduces ambiguity and additional evaluation difficulty. The comparison motivates a rules-only baseline closely related to synthetic spelling corruption before adding more complex components.

#### 2.2.2 Rule-based and custom error generation

Rule-based generation offers transparency. Character substitutions, deletion of diacritics, vowel or consonant changes, spacing noise, repetitions, and punctuation modifications can be applied with explicit probabilities. This makes it possible to trace an output and reproduce it from a random seed. It also allows transformations to be constrained to Wolof tokens while protecting entities or French spans.

The central weakness is distribution design. Rules created from linguistic intuition may overproduce transformations that are easy to describe and underproduce lexicalized or community-specific forms. Uniform random application can create sentences with too many errors or combinations rarely used by people. Dekker and van der Goot's custom error generation results show that synthetic corruption can be highly useful, especially when informed by observed annotations [8]. Their work also raises the methodological issue central to this PFE: a gold-informed generator is valid supervised development when it learns only from training data, but it must be compared separately with a fully annotation-free method.

#### 2.2.3 Unsupervised normalization and lexical mining

Gouws et al., Li and Liu, and Roy et al. represent complementary forms of unsupervised normalization [5]–[7]. Gouws et al. focus on mining variants from a large noisy corpus. Li and Liu combine surface and distributional similarity and improve candidate selection through reranking. Roy et al. emphasize a language-independent algorithm and demonstrate downstream gains in retrieval and stance detection.

Together, these studies support using the collected YouTube corpus as more than a source of examples for manual annotation. Its token frequencies and contexts can reveal target-domain variants without paired labels. However, the project must avoid assuming that an out-of-vocabulary form is erroneous. A candidate may be French, a name, a valid Wolof word missing from the formal vocabulary, or a context-specific expression. The proposed unsupervised method therefore prioritizes precision over coverage and records rejected mappings as well as accepted ones.

#### 2.2.4 Synthetic code-mixing

Code-mixing generation has been approached through dictionaries, word alignment, embeddings, linguistic constraints, machine translation, and neural generation. Das and Gambäck's analysis highlights the difficulty of even detecting token-level language boundaries in noisy social text [2]. This is a warning against treating language identification as an infallible guardrail.

Jawahar et al. generate code-mixed data using bilingual distributed representations and compare their method with back-translation and Equivalence Constraint-based alternatives [13]. Their curriculum-learning result provides a direct precedent for synthetic pretraining followed by gold fine-tuning. The method concerns English-Hinglish translation, so its numerical results cannot be transferred to Wolof-French; its experimental structure is more transferable than its language-specific assumptions.

More complex work treats code-switching as style transfer. Li and Vu apply cycle-consistent adversarial learning to generate code-switched text and report improvements in language modeling and speech recognition [18]. Chandu and Black use a two-stage adversarial framework with monolingual corpora and a limited set of natural code-switched sentences [19]. These studies show that neural generators can learn distributional properties beyond fixed replacement rules. They also require more training complexity, stabilization, data, and evaluation than the current PFE can safely support. Adversarial naturalness is not equivalent to semantic preservation, and reproducing such approaches for Wolof would introduce another major research problem before the core baselines are established.

#### 2.2.5 LLM-generated Wolof synthetic data

LLM prompting provides another way to generate informal variants. The `soynade-research/Wolof-Non-Standard-Orthography` dataset contains 3,438 standard/non-standard pairs and an English field [22]. Its dataset card states that standard Wolof sentences were transformed by prompting Oolel, a Wolof-specialized language model, with instructions derived from social-media patterns and linguistic transformations. The generated outputs include phonetic spelling, missing diacritics, character substitutions, word merging, and French code-switching.

This resource is especially relevant because it is language-specific and already targets the same normalization direction as the PFE. It also represents a fundamentally different generator: instead of composing explicit local rules, an LLM produces the entire output conditionally. This can create more varied and context-sensitive transformations, but it reduces transparency. Prompting may change content, produce formatting artifacts, apply too many transformations, or reproduce regularities of the model's fine-tuning data. The dataset card itself notes that synthetic text may not cover all real-world variation [22].

The published dataset and a newly prompted model baseline are not methodologically identical. Directly using the published pairs is reproducible at the dataset level, but their formal source sentences come from `galsenai/english-wolof-smol-translation`, not necessarily the common source pool used by the project's other generators. Any downstream difference could therefore reflect both source-corpus and generation-method effects. Running a pinned Oolel checkpoint on the common source pool would provide a fairer generator comparison, but the prompt, decoding configuration, model revision, parsing, and rejection rules must be documented. Because the original dataset card does not expose every generation detail, such a run should be labeled an Oolel-based reproduction rather than an exact recreation.

The `Oolel-Corrector` model should also be distinguished from the generator. It is a reverse informal-to-standard corrector fine-tuned on the published synthetic dataset; it did not generate the training dataset itself [23]. Using that corrector as though it were one of the fixed seq2seq data conditions would change both training data and architecture, making it unsuitable for the controlled generator ranking. It can, however, be run unchanged on the locked gold test set as an **external end-to-end system baseline**. This separate experiment answers how an existing Wolof corrector performs on the project's real benchmark.

#### 2.2.6 Unsupervised text style transfer

Text style transfer is relevant because the formal and informal forms should differ in surface realization while preserving content. Jin et al. organize neural style-transfer methods according to data availability and discuss the difficulty of evaluating content preservation, style strength, and fluency simultaneously [15]. Zhang et al. formulate style transfer as unsupervised machine translation, using pseudo-parallel data, iterative back-translation, and a style classifier [17]. He et al. provide a probabilistic formulation in which latent parallel sequences connect non-parallel domains and relate this view to back-translation and adversarial objectives [16].

These methods are attractive when large non-parallel corpora exist in both styles, but their assumptions are demanding in the present setting. Defining a reliable “informal Wolof” style classifier is itself difficult; the informal corpus includes mixed languages and diverse phenomena; and cycle consistency does not guarantee that facts or lexical meaning are preserved. For this reason, unsupervised learning is retained in a narrower and more auditable form—lexical mining—rather than making full unsupervised style transfer the main contribution.

#### 2.2.7 Comparative synthesis

Table 2.1 summarizes the approaches most relevant to the project.

| Approach | Required resources | Strengths | Main limitations for this PFE | Role in project |
|---|---|---|---|---|
| Wolof finite-state normalization [3] | Linguistic rules and lexicons | Interpretable; language-specific | Limited coverage of open-ended social variation | Theoretical and preprocessing reference |
| Beqi synthetic correction [4] | Formal Wolof and synthetic error rules | Direct Wolof evidence; seq2seq correction | Focused spelling setting; limited real code-switch modeling | Closest task baseline and methodological precedent |
| Custom error generation [8] | Formal text; optionally annotated errors | Controlled and effective; traceable | Can copy annotation biases or overfit rule distributions | Rules baseline and standalone gold-informed M6 |
| Unsupervised variant mining [5]–[7] | Large noisy corpus and formal candidates | Uses target domain without paired labels | Ambiguity, OOV names/French, unreliable rare contexts | Separate M5 benchmark |
| Word-alignment substitution [14] | Parallel French-Wolof text | Simple lexical correspondences | Context and grammar not guaranteed | Code-switch baseline |
| Embedding-based code-mixing [13] | Bilingual representations and training text | Dependency-light; supports curriculum learning | Evidence from another language pair | CMDR-style baseline and training inspiration |
| Oolel LLM-generated data [22] | Wolof-specialized LLM or published 3,438-pair dataset | Language-specific; contextual and varied generation | Prompt opacity, possible semantic drift, and different source corpus | External LLM baseline; common-source reproduction if feasible |
| Adversarial code-switch transfer [18], [19] | Monolingual and code-switched corpora; complex training | Can model distributional naturalness | High implementation risk; semantic control difficult | Related work/future extension |
| Unsupervised style transfer [16], [17] | Non-parallel style corpora and auxiliary objectives | Does not require direct pairs | Hard evaluation and unstable assumptions | Related work, not core method |
| Multilingual pretrained seq2seq [11], [12] | Pretrained model and task pairs | Transfer learning; contextual correction | Wolof coverage and compute remain uncertain | Candidate downstream normalizer |

The synthesis reveals that no reviewed method simultaneously provides Wolof specificity, real social-media adaptation, controlled code-switching, train/test isolation, and downstream comparison across generators. This combination defines the project's research space.

### 2.3 Project positioning

#### 2.3.1 Identified gap

The available literature leaves three practical gaps.

First, Wolof correction work demonstrates synthetic spelling correction but does not fully represent informal French-Wolof YouTube text. Second, unsupervised normalization methods show how to exploit unpaired noisy corpora, but their behavior is not established for Wolof and must be adapted to multilingual ambiguity. Third, code-mixing and style-transfer methods often optimize generation or language modeling in better-resourced language pairs rather than compare how different synthetic datasets affect correction of authentic held-out Wolof comments.

The project is positioned at the intersection of these gaps. Its novelty is not a claim to invent synthetic noise, word alignment, or lexical mining individually. Its contribution is a **controlled, domain-informed combination and evaluation for a low-resource Wolof setting**.

#### 2.3.2 Baselines and proposed method

The benchmark separates five generation conditions:

1. **Generic rules-only baseline:** uses formal data and documented transformations, but neither YouTube comments nor gold annotations.
2. **Alignment/CMDR baseline:** adds bilingual lexical evidence for controlled code-switching.
3. **External Oolel LLM baseline:** uses the published LLM-generated Wolof pairs, or a separately labeled common-source Oolel reproduction.
4. **Unsupervised YouTube-aware method:** mines high-confidence variants from eligible unpaired comments without manual correction pairs.
5. **Proposed gold-informed linguistic engine:** learns mappings, POS-conditioned error distributions, protected categories, and repeated character transformations only from gold training.

These conditions support interpretable questions. Comparing the rules and Oolel baselines tests explicit transformations against full-sequence LLM generation, although a direct published-dataset comparison remains partially confounded by its different source corpus. Comparing rules with standalone M5 estimates the value of unpaired target-domain data. Comparing standalone M5 with standalone M6 estimates the value of limited supervised linguistic evidence without mixing their inputs. Comparing all methods under the same downstream model tests whether added generator complexity produces useful learning signals rather than merely more elaborate text.

The proposed M6 engine uses a conservative transformation sequence: protect entities, ambiguous entries, and intentional French spans; sample a realistic edit count from gold-training statistics; apply reviewed gold lexical or spacing mappings and recurring POS-conditioned character transformations; and reject unsafe candidates through protection and manual review gates. It does not consume M5 mappings or M1 rules. Each example records its source, seed, changed positions, error types, mapping origins, and confidence. This provenance supports inspection and ablation.

#### 2.3.3 Leakage policy

Using gold data to learn errors is not inherently cheating. In supervised machine learning, training data are explicitly intended to estimate parameters and patterns. Leakage occurs if information from development or test annotations influences mappings, thresholds, rules, prompts, or distributions. The benchmark therefore uses a video-disjoint train/development/test split. Gold-training pairs may inform the standalone M6 generator; development data may select thresholds; test data remain untouched until final evaluation. Unpaired comments from development and test videos are excluded from M5 lexical mining.

This design is stricter than a random comment split because comments from the same video may share topic, vocabulary, repeated phrases, or participants. A video-disjoint split better tests transfer beyond one local context.

#### 2.3.4 Evaluation philosophy

The primary evidence is downstream normalization on real held-out comments. Every project-controlled generator receives the same normalized source pool and produces the same number of examples. The model architecture, tokenizer, hyperparameter budget, stopping rule, gold fine-tuning set, and seeds remain fixed. Each method uses synthetic pretraining followed by identical gold fine-tuning; only the post-gold model is reported. Gold-only and copy controls determine whether synthetic initialization adds value.

After validation, 3,433 of the 3,438 published Oolel rows remain as unique pairs; the other methods are therefore evaluated at a matched 3,433-example size. This controls training volume but not formal-source content, so the Oolel result belongs in a clearly marked external-baseline column or table. Only an Oolel run over the common normalized source pool would qualify for a fully common-source comparison.

`Oolel-Corrector` forms a second, system-level comparison. Its checkpoint is evaluated on exactly the same locked gold-test inputs and scored with the same CER, WER, chrF, exact-match, token-normalization, and overcorrection metrics. Before evaluation, the gold test must be checked for overlap with the corrector's published training dataset. The model is not fine-tuned on test data, and its revision, license, decoding settings, parameter count, hardware, memory use, and inference time are recorded. Its score is reported separately because it answers “Which complete system performs better?” rather than “Which synthetic dataset best trains the fixed seq2seq model?”

Intrinsic and human evaluations explain the results. If the proposed method does not outperform rules, the analysis should examine mapping precision, coverage, edit distributions, overcorrection, benchmark size, and seed variance. This outcome-neutral positioning avoids treating novelty as proof of superiority.

#### 2.3.5 Relationship to the original retrieval objective

Normalization is selected because paired correction targets enable direct and reproducible evaluation. It is nevertheless a means toward the original goal of robust semantic access. Roy et al. provide evidence that normalization can improve information retrieval in other noisy-text settings [7]. If time permits, the project will build a small set of informal queries with manually verified relevant formal passages. Retrieval with the original query will be compared with retrieval after normalization. This secondary experiment can demonstrate transfer while avoiding the confounders of a full RAG system.

## 3. Mission context and contributions

### 3.1 Company and team presentation

### 3.1 Scope of the mission

#### 3.1.1 Initial mission and evolution

The initial mission was motivated by semantic retrieval. The intended application was to query a corpus containing new Wolof data even when user queries were written informally. Early investigation showed that the deeper obstacle was not limited to one retrieval algorithm: the available formal resources and real informal inputs followed different distributions. The same issue affected LLM prompts and machine-translation inputs.

The mission therefore evolved toward synthetic data generation. Rather than optimizing a retrieval pipeline without adequate training and evaluation pairs, the project first addresses the data scarcity that causes multiple downstream systems to fail. Formal Wolof resources are transformed into synthetic informal variants, and the usefulness of each transformation method is measured through normalization of real comments. A retrieval case study remains an optional transfer experiment.

This change is a refinement rather than an abandonment of the initial objective. The project moved from a single application to a reusable formal-informal bridge. The revised mission is:

> Design, implement, and evaluate synthetic data generation methods that improve the robustness of Wolof processing systems to authentic informal and French-Wolof code-switched text.

#### 3.1.2 Assigned tasks and personal contributions

The mission covers the following work packages.

**WP1 — Literature and resource review.** Relevant work on Wolof normalization, lexical normalization, code-switching, synthetic error generation, unsupervised mining, sequence-to-sequence models, and style transfer was collected and critically compared. Available Wolof dictionaries, translation data, and pretrained-model options were inventoried.

**WP2 — Real-data collection and cleaning.** Public YouTube comments were collected with scripts that enumerate video links and download comments. A cleaning notebook removes or normalizes URLs, usernames, timestamps, emojis, extra whitespace, and unusably short rows, while preserving source-video provenance.

**WP3 — Candidate filtering.** Because the cleaned corpus remains too large for manual review, a local Gemma model served as an annotation-candidate filter. The pipeline classifies bounded batches, validates response identifiers, retries failures, stores append-only checkpoints, and can resume. Checkpoints are merged by stable source index, and positive rows retain their video provenance.

**WP4 — Formal-resource and vocabulary preparation.** French-Wolof translation and dictionary data were consolidated. A formal Wolof vocabulary was extracted for candidate lookup, spelling comparison, and annotation assistance.

**WP5 — Synthetic generation prototypes and external baseline selection.** Hand-written informalization rules produced initial formal-informal pairs. CMDR-style bilingual representations and Eflomal alignment were explored for code-switching. A Windows-native IBM Model 1 implementation was added as an alternative to rebuilding Eflomal under Linux. The published Oolel-generated non-standard Wolof dataset was selected as an additional LLM baseline; importing and validating it remain pending.

**WP6 — Reverse normalization prototype.** A first reverse pipeline masks entities, classifies token types, uses reverse lexical mappings and weighted edit distance, and proposes formal Wolof forms. This prototype supports exploration and annotation but is not treated as gold truth.

**WP7 — Gold annotation infrastructure.** A Flask application supports keep, discard, uncertainty, sentence correction, and token-level correspondences. It preserves append-only history and saves stable source identifiers. Annotation guidelines and periodic quality-review requirements were drafted.

**WP8 — Benchmark integrity.** A video-disjoint split tool with minimum-data checks and manifest hashes was implemented. Tests cover critical workflow behavior, and UTF-8 auditing protects Wolof characters.

**WP9 — Remaining experimental contribution.** The remaining work is to run and quality-control the implemented generators, export and review standalone M6, freeze controlled synthetic inputs, train the downstream models, complete quantitative evaluation and ablations, and optionally run the retrieval case study.

#### 3.1.3 Schedule

Table 3.1 presents the mission schedule in functional phases. Calendar dates must be aligned with the actual internship calendar before submission.

| Period | Phase | Principal outputs | Status |
|---|---|---|---|
| `[DATE–DATE]` | Problem framing and literature review | Initial retrieval problem, literature corpus, synthetic-data reformulation | Completed/continuing review |
| `[DATE–DATE]` | Collection and cleaning | Raw and cleaned YouTube corpora; scraping and cleaning workflow | Completed |
| `[DATE–DATE]` | Formal resources and first generators | Vocabulary, rules-only data, CMDR and Eflomal prototypes | Completed as prototypes |
| `[DATE–DATE]` | Filtering and reverse normalization | LLM checkpoints, filtered candidates, reverse pipeline | Completed as working pipeline/prototype |
| `[DATE–DATE]` | Annotation and reliability improvements | Flask tool, gold files, diversified video order, tests, split tooling | In progress |
| `8 August 2026` | Gold benchmark completion | Confirmed normalization policy, 201 kept pairs across 5 videos, locked splits | Completed |
| `[PLANNED DATE–DATE]` | Proposed generation methods | Unsupervised miner, standalone linguistic generator, controlled datasets | In progress |
| `[PLANNED DATE–DATE]` | Modeling and evaluation | Matched seq2seq training, metrics, human review, ablations | Remaining |
| `[PLANNED DATE–DATE]` | Report and handover | Final report, reproducibility package, presentation | In progress/remaining |

#### 3.1.4 Expected deliverables

The mission deliverables are:

- a documented raw and cleaned YouTube comment corpus;
- candidate-classification checkpoints and filtered data with provenance;
- formal Wolof vocabulary and documented translation resources;
- baseline synthetic formal-informal datasets;
- a validated external Oolel LLM-generated baseline with pinned revision and license;
- an external-system evaluation of `Oolel-Corrector` on the locked gold test set;
- lexical alignment artifacts and portability comparison;
- an annotation application and gold annotation files;
- a locked video-disjoint benchmark split;
- an unsupervised YouTube-aware lexical miner;
- the proposed standalone gold-informed linguistic generator;
- controlled downstream model checkpoints and predictions;
- metric tables, human evaluation, error analysis, and ablations;
- code, tests, environment documentation, data lineage, and final report.

#### 3.1.5 Completed contributions at the time of writing

At the current snapshot, the project has produced an end-to-end data-preparation and annotation path. The strongest completed contributions are the collection and cleaning of a large real-world corpus, robust resumable filtering, stable provenance across filtered data, formal vocabulary construction, rules and alignment prototypes, a reverse-normalization assistant, a Flask annotation workflow, a Windows-native IBM Model 1 alternative, video-disjoint split preparation, and basic automated integrity tests.

The central experimental claim is not yet available. Annotation is complete at 201 kept gold pairs across five videos, the video-disjoint split is locked, and all 1,315 gold-train token occurrences have linguistic labels. The standalone M6 implementation exists, but its export/review, the downstream Transformer comparison, and final result tables remain future work. This distinction is important: pipeline completion is an engineering contribution, but it cannot substitute for the final scientific evaluation.

### 3.2 Data and constraints

#### 3.2.1 Data inventory

Table 3.2 lists the principal resources at the time of writing.

| Resource | Volume | Format | Origin and role | Status |
|---|---:|---|---|---|
| Raw YouTube comments | 396,559 records | JSON Lines | Public comments collected from selected videos; source corpus | Completed snapshot |
| Cleaned YouTube comments | 242,174 rows from 193 videos | CSV | Cleaned unpaired target-domain corpus | Completed snapshot |
| LLM-classified comments | 7,250 rows from 5 videos | JSON checkpoints and CSV | Candidate filtering for annotation | Expansion recommended |
| Positive filtered candidates | 5,498 rows from 5 videos | CSV | Informal/code-switched annotation queue | Current |
| Annotation decisions | 359 decisions | CSV | Append-only manual decisions | Annotation stopped |
| Kept gold pairs | 201 pairs from 5 videos | CSV | Formal-informal benchmark pairs | Locked as 142 train, 30 development, and 29 test pairs |
| Token corrections | 918 rows; 614 distinct informal tokens | CSV | Annotation memory and train-only mappings | Ready for train-only analysis after splitting |
| Main French-Wolof parallel corpus | 17,777 sentence pairs | Parquet | Formal source data and alignment | Available |
| Additional bilingual entries | 7,971 rows | CSV | Vocabulary expansion | Available |
| Dictionary translations | 9,626 entries | JSON | Vocabulary and generation | Available |
| Rules-only synthetic corpus | 17,777 pairs | CSV | Existing generation baseline | Prototype/export requires standardization |
| Combined synthetic corpus | 27,403 pairs | CSV | Rules-only pretraining material | Prototype/export requires standardization |
| Oolel non-standard orthography dataset | 3,438 raw / 3,433 accepted pairs | Hugging Face/Parquet | External LLM-generated baseline with `wo`, English, and non-standard fields | Pinned CC BY-SA 4.0 snapshot imported; five duplicate pairs rejected; manual review pending |
| Formal Wolof vocabulary | 32,178 unique entries | Text | Candidate and spelling lookup | Available |
| Eflomal lexicon | 1,989 Wolof entries | Pickle | Wolof-to-French alignment baseline | Available artifact |
| IBM Model 1 trial lexicon | 1,672 entries; 2,775 links from 2,000 pairs | Pickle | Windows-native alignment alternative | Experimental; manual validation needed |
| Reverse-pipeline test export | 111 rows | CSV | Early automatic recovery inspection | Prototype, not gold data |

The corpus has two distinct roles. Formal French-Wolof and dictionary resources provide clean source sentences and lexical candidates. YouTube comments represent the target distribution and supply unpaired evidence, annotation candidates, and the final benchmark. These roles must remain separated during evaluation.

#### 3.2.2 Data quality constraints

The YouTube corpus is large but weakly structured. Comments can be duplicated, context-dependent, extremely short, mostly French, composed of names or emojis, or unrelated to the target task. Video topic and community can create strong local vocabulary patterns. The final annotated gold subset contains 201 kept pairs across five videos. Randomly splitting these comments would overestimate generalization because nearly identical expressions from the same context could appear in training and test data.

Several mitigations are implemented or planned:

- preserve `source_index`, `video_url`, and derived `video_id` throughout the pipeline;
- diversify classification and annotation in round-robin video order;
- remove or detect exact and near duplicates;
- require at least 200 kept pairs across at least 5 videos before splitting;
- create video-disjoint train, development, and test partitions;
- lock row identifiers and hashes in a split manifest;
- exclude test annotations and test-video comments from method development.

Annotation quality is another constraint. Informal Wolof may permit several acceptable formalizations, and the intended meaning may be uncertain without conversational context. French spans can be preserved or translated, but mixing both policies would make the target inconsistent. Before annotation continues at scale, the project must choose one policy. The recommended task is normalization of Wolof while preserving intentional French code-switches. After every 50 kept examples, a sample should be reviewed; before final locking, a second qualified Wolof speaker should review a stratified subset if available.

The formal vocabulary is also incomplete by definition. A token absent from the vocabulary is not necessarily an error. It can be a valid inflection, French word, name, borrowed expression, or rare Wolof form. Unsupervised mining must therefore treat vocabulary membership as candidate evidence rather than ground truth.

#### 3.2.3 Personal data, ethics, and legal review

The raw source consists of publicly visible user-generated comments. Public visibility does not by itself remove privacy and data-protection responsibilities. Usernames, profile information, URLs, names mentioned in text, and combinations of content and video provenance may permit direct or indirect identification. The cleaning stage removes usernames and several obvious identifiers from the modeling text, but the raw collection and source URLs remain more sensitive.

The project should apply purpose limitation, data minimization, storage limitation, integrity, and confidentiality principles consistent with Article 5 of the General Data Protection Regulation [21]. In practical terms:

- retain only fields necessary for the documented research purpose;
- do not attempt to identify or profile comment authors;
- restrict access to raw data and avoid publishing usernames or unnecessary source metadata;
- use anonymized or paraphrased examples in the report when a verbatim comment could be searchable;
- separate internal provenance from public model inputs;
- define a retention and deletion policy with the host organization or school;
- document the lawful basis, institutional policy, and any required ethics or data-protection review;
- verify platform terms and dataset licenses before redistributing raw comments or external corpora.

This report is not a legal determination. `[TO COMPLETE — identify the host organization's or school's validated GDPR/legal basis, reviewer, storage policy, and redistribution decision.]`

Ethical quality also concerns language representation. The classifier and generator may reproduce biases toward particular videos, speakers, topics, or French-Wolof practices. Labeling every non-standard form as an error can stigmatize legitimate community usage. The report should therefore describe the output as a task-specific normalized form, not as a judgment that one speaker's language is invalid. Human evaluation should include native-speaker competence and document disagreements rather than force uncertain cases.

#### 3.2.4 Technical and infrastructure constraints

The main workflow is Windows-native and uses Python notebooks and scripts. The annotation interface and IBM Model 1 alternative run on Windows. Eflomal requires a compiled Linux/WSL environment to rebuild, although the existing `artifacts/lexicon.pkl` can be consumed on Windows. This creates a portability constraint: reproducibility depends either on documenting the WSL build or using the separately validated IBM Model 1 artifact.

LLM-assisted filtering uses a local Ollama model, Gemma 3 12B. Local execution avoids sending comments to a third-party hosted API, but it requires sufficient memory and compute and may produce nondeterministic or malformed classifications. The implemented workflow mitigates this through bounded batches, identifier validation, retries, single-row fallback, checkpointing, and resumability. The LLM output is used only to prioritize manual annotation; it is not treated as a gold label.

Downstream Transformer training has not yet started. mT5 or mBART are candidate architectures, but the final choice depends on available GPU memory, training time, and Wolof behavior. The project should record hardware, library versions, model checkpoint, batch size, precision, runtime, energy or monetary cost where available, and any use of remote compute. If compute is limited, smaller pretrained variants, gradient accumulation, early stopping, and a reduced but identical hyperparameter budget across methods are preferable to an unfair comparison.

Other technical constraints include CSV encoding and Wolof-specific characters, notebook reproducibility, duplicated artifacts, and path differences across environments. Central path configuration, `.editorconfig`, UTF-8 auditing, workflow tests, and data-lineage documentation mitigate these risks. Experimental notebooks should be consolidated into deterministic scripts once the retained methods are fixed.

#### 3.2.5 Budget and time constraints

The dominant cost is human time rather than paid data. Manual annotation requires Wolof competence and careful sentence-level judgment. Mapping and generated-sentence inspection are also necessary because automatic metrics cannot guarantee meaning preservation. The PFE therefore uses LLM filtering to reduce the candidate pool and reuses prior token corrections to accelerate annotation.

The initial blueprint included unsupervised machine translation, iterative back-translation, adversarial CycleGAN training, multilingual Transformers, and two-stage fine-tuning as one pipeline. Implementing all components would exceed the available time and obscure attribution. The revised scope prioritizes a small number of auditable generators and a controlled downstream comparison. Cycle-consistent and fully unsupervised style-transfer models remain related work or future extensions.

`[TO COMPLETE — state the actual financial budget, compute allocation, available GPU/CPU/RAM, internship dates, weekly availability, and final submission deadline.]`

#### 3.2.6 Scientific validity constraints

The final benchmark remains small. The selected minimum of 200 kept pairs enables an operational split, but it does not provide strong statistical power, especially after division by error category. The final 201 pairs produce a dry-run allocation of 142 train, 30 development, and 29 test examples. Results should therefore include seed variation or bootstrap confidence intervals where possible and avoid overinterpreting small differences.

Synthetic methods may also differ in edit difficulty, vocabulary, or sentence selection. The comparison must control the clean source pool, number of examples, model configuration, seeds, gold fine-tuning data, and stopping rule. Standalone M6 may learn from gold training data, but M5 must remain annotation-free and neither method may consume the other's outputs in the primary comparison. This separation answers different research questions and prevents a supervised method from being presented as annotation-free.

Finally, normalization accuracy is evidence for the formal-informal bridge, but it does not automatically prove gains for every downstream task. Retrieval, LLM querying, and translation have their own objectives and failure modes. Claims beyond normalization must either be supported by a focused experiment or clearly presented as future work.

## 4. Methodology and technical approach

This chapter describes the methodology used to construct and evaluate synthetic
formal/informal Wolof pairs. The principal task is **lexical normalization**:
given a Wolof or Wolof–French social-media sentence, the model must produce a
standardized Wolof version while preserving its meaning. Synthetic generation is
therefore evaluated indirectly through the normalization models that it enables.
This is more informative than judging generated sentences only by surface
similarity, but it does not by itself demonstrate an improvement for every
possible downstream application. Semantic retrieval is retained as an optional
secondary experiment after the core normalization benchmark.

The methodology is divided into completed, prototyped, and planned components.
This prevents an experimental intention from being reported as a completed
result. Table 4.1 summarizes the state of the pipeline at the time of writing.

**Table 4.1 – Implementation state of the methodology**

| Component | Current state | Evidence or remaining work |
|---|---|---|
| YouTube extraction and cleaning | Implemented | 396,559 raw and 242,174 cleaned comments |
| Local LLM filtering | Implemented on an initial subset | 7,250 classified; 5,498 retained candidates |
| Gold annotation interface | Implemented; selected stopping point reached | 359 decisions, 201 kept pairs, 918 token corrections |
| Video-disjoint gold split | Implemented and locked | 142 train/3 videos, 30 development/1 video, 29 test/1 video; manifest hashes verified |
| Rule generator | Implemented | Formal/generated pairs exported |
| Eflomal generator | Implemented and exported | 17,777 deterministic M2 pairs, manifest, substitution provenance, and pending 100-row review |
| CMDR-style generator | Implemented and exported | 17,777 deterministic M3 pairs, saved resources, score provenance, and pending 100-row review |
| Oolel data baseline | Imported and validated | 3,433 accepted external pairs, manifest, rejection audit, and pending 100-row review |
| YouTube-aware unsupervised method | Implemented and exported | 17,777 M5 pairs; final manifest gates and two manual reviews remain |
| Gold-informed linguistic engine | Implemented and provisionally exported | 17,777 M6 pairs, 14,215 changed; manual review/finalization and optional ablations remain |
| Seq2seq training and evaluation | Reproducible runner implemented; provisional execution authorized | Pinned mT5-small, visible progress, live logs, versioned/resumable attempts, metrics/tables/plots |
| Oolel-Corrector comparison | Planned | Must run unchanged on the same locked test set |

### 4.1 General solution architecture

The solution uses two data streams. The first consists of **formal resources**:
French–Wolof parallel sentences, dictionary translations, and a formal Wolof
vocabulary. These resources provide clean source sentences to which artificial
informal transformations can be applied. The second consists of **authentic,
unpaired YouTube comments**. These comments provide observations of real
non-standard spelling and code-switching. A small subset is manually normalized
to create gold pairs, while the remaining eligible comments may be used to mine
informal variants without sentence-level supervision.

The streams are separated according to their experimental function. Formal
resources and unpaired training comments may construct synthetic training data.
Gold training annotations may be used only by methods explicitly labelled
supervised, such as the proposed standalone M6 engine. Gold development data may guide
threshold and model choices. Gold test data must be locked and may be opened only
for final evaluation. Neither mining, generation, prompt development, training,
nor early stopping may inspect test targets.

```text
FORMAL RESOURCES                                  AUTHENTIC COMMENTS
parallel + dictionary + vocabulary               YouTube playlists
          |                                              |
   +------+------+                              collect -> clean
   |             |                                      |
 rules      alignment/CMDR                       local LLM filter
   |             |                               /              \
   +------+------+                      manual gold pairs     unpaired comments
          |                                  |                    |
 synthetic paired corpora              train/dev/test      variant mining
          |                                  |                    |
          +---------- separate generators and matched training --+
                                             |
                              identical fresh seq2seq models
                                             |
                        locked metrics + blinded human evaluation
                                             |
                         separate external Oolel-Corrector result
```

**Figure 4.1 – End-to-end experimental architecture.** The gold split, M5/M6
implementations, and the downstream training/evaluation runner are complete.
Manual generator reviews, M5 finalization, M6 export/finalization, long-running
model execution, and the one-time frozen test evaluation remain to be completed.

The comparison reports a **gold-only** control and one **synthetic-plus-gold**
condition per generator. Each method first produces an auditable synthetic
checkpoint, then receives the same gold training data. The intermediate
checkpoint is not scored as a separate benchmark result; the experiment asks
whether each synthetic corpus provides a useful initialization beyond gold-only.
Using gold-train error patterns is not test leakage, but it is supervision and
must not be presented as annotation-free. Leakage would occur if development or
test targets informed mappings, rules, frequencies, or generation thresholds.

#### 4.1.1 Data collection and ingestion

##### Formal resources

The principal formal corpus contains 17,777 French–Wolof sentence pairs in
Parquet format. The Wolof side provides formal sources and the French side
supports alignment-based code-switching. A second file contains 7,971 Wolof
sentences, and dictionary translations provide 9,626 Wolof entries. After
tokenization and deduplication, the current formal vocabulary contains 32,178
unique items. These are working counts and will be recomputed and frozen before
the final experiment.

Inputs are registered in a central configuration module. The project, data, and
artifact roots can be changed through environment variables, reducing dependence
on one Windows directory layout. Parquet, CSV, JSON, and JSONL are read according
to their type, and text is stored as UTF-8 to preserve characters such as `ñ`,
`ŋ`, and `ë`.

##### YouTube comment collection

The authentic corpus was collected from public comments attached to videos in
two selected YouTube playlists. The ingestion script calls `yt-dlp` in
flat-playlist mode, extracts each video URL, removes duplicate URLs, and writes a
video list. A second script uses `youtube-comment-downloader` to retrieve all
available comments for each video. Every raw JSONL record stores the comment text
and source video URL.

Collection is resumable: after a video is processed, its URL is appended to a
ledger and later executions skip it. The raw snapshot contains 396,559 comments.
The current script does not retain timestamps, threads, user identifiers, or an
extraction date per comment. This limits temporal analysis but reduces personal
metadata. Before any release, examples must be anonymized and YouTube's
applicable conditions verified.

The corpus is a research snapshot, not a continuously scheduled crawl. Repeating
the extraction may return different comments. The final report must record the
collection date, playlists, videos attempted and completed, tool versions, and a
checksum of the raw JSONL. Explicit retry and failure logs should be added.

##### External Oolel resources

The benchmark also includes the *Wolof Non-Standard Orthography* dataset from
Soynade Research [22]. Its card reports 3,438 LLM-generated examples under CC
BY-SA 4.0. It contains standard Wolof, non-standard Wolof, and English
information. Revision `ce17c80cff6626a91dc4e38b43de12e005bc240a` was pinned
locally, and the published generation wrapper uses a `<NON_STANDARD>`
instruction. Schema validation accepted 3,433 unique pairs and rejected five
duplicate pairs; all accepted sources differ from their generated outputs.

This is an **external-data baseline**, not automatically a controlled generator
on this project's source sentences. A result may reflect differences in the
formal source corpus, sample size, model, or decoding. The published 3,438 pairs
are therefore marked as external data, and the 3,433 accepted pairs determine
the matched size for the retained M1–M6 methods. The manifest records the pinned revision, CC BY-SA
4.0 license, schema decisions, file hashes, duplicate rejection, and zero exact
overlap with the current gold formal and informal strings. The original prompt,
seed, and decoding parameters remain unavailable, so this condition cannot be
treated as a common-source controlled reproduction.

#### 4.1.2 Preprocessing and feature engineering

##### Cleaning and language filtering

`youtube_data_cleaning.ipynb` removes URLs, `@username` strings, timestamps of
the form `HH:MM` or `HH:MM:SS`, broad emoji ranges, repeated `!` and `?`, and
extra whitespace. It lowercases the remaining text. Empty comments and comments
with two tokens or fewer are discarded.

The notebook then applies `langdetect` to alphabetic material in each token. A
comment is rejected when more than half of its tokens contain no usable
alphabetic material, or when the estimated French-token ratio is at least 0.5.
This reduced 396,559 raw comments to 242,174 cleaned comments from 193 videos.
The output preserves raw text, video URL, cleaned text, and French-ratio estimate.

These are heuristic thresholds. Word-level language identification is unreliable
for short, borrowed, related, or misspelled forms, and the French cutoff may
remove useful code-switched examples. A stratified sample around 0.5 must be
manually audited before freezing the corpus. Exact and near-duplicate handling
must also be documented; cleaning and lowercasing may collapse distinct raw
comments, but the current notebook does not yet provide a complete deduplication
report.

##### Local LLM candidate filtering

Manual review of all cleaned comments is impractical. `classify_wolof.py`
therefore sends batches to Gemma 3 12B through local Ollama. The prompt requests
strict JSON records containing an integer identifier and Boolean label. Positive
means likely informal Wolof or Wolof–French; mostly French comments are negative.

The implementation uses temperature 0, batches of 50, a 300-second timeout,
three retries, and a two-second retry delay. It validates the returned identifier
set and value types. Failed batches are retried individually; if all attempts
fail, the current conservative fallback retains the item. Append-only
checkpoints are written every five batches, and pending rows are selected
round-robin across videos. Labels are joined to source data by the original index
with one-to-one validation.

So far, 7,250 comments from five videos have been classified and 5,498 retained.
The LLM is only a sampling aid, not gold truth. Its precision and recall have not
yet been measured. The final methodology must audit a manually labelled sample,
record the exact model revision, and justify or revise the positive fallback.

##### Gold annotation and error labels

An implemented Flask interface presents retained candidates in deterministic
round-robin video order. The annotator selects `keep`,
`discard_false_positive`, `discard_uninteresting`, or `skip_uncertain`. Kept
records store source index, comment, automatic suggestion, manual formal target,
status, video URL, and token corrections. Annotation files are append-only and
the latest decision is used when a record is revisited.

The interface proposes token corrections using the formal vocabulary, alignment
lexicons, French lexical resources, weighted edit distance, and previously
accepted corrections. Human corrections are ranked first, but suggestions are
never treated as labels. The annotator validates both the sentence and changed
tokens.

The final annotation snapshot contains 359 decisions, 201 kept pairs, 918 token
correction events, and 614 distinct informal forms across five videos. This is
the selected pragmatic stopping point. A second Wolof-speaking reviewer would
still improve reliability, and the small dev/test partitions limit precision.
Agreement should be measured separately for candidate status, target
acceptability, and error category.

The frozen annotation policy preserves intentional French spans while
normalizing orthographic variation. Translating all French would redefine the
task as monolingual translation and is outside the benchmark target. This
policy is documented in the annotation guidelines and is also reflected by the
M6 protection labels.

The planned taxonomy includes Wolof-character substitutions (`ñ/gn`, `ŋ/ng`,
`ë/eu`), phonetic/graphemic substitutions, vowel and consonant length, word
boundaries and hyphens, abbreviation/deletion, insertion, intentional French,
protected content, ambiguity, and other. Code can propose categories from edit
operations and rule patterns, but semantic substitution, borrowing, intent, and
ambiguous boundaries require manual confirmation.

##### Vocabulary and leakage-safe splits

The vocabulary script selects formal Wolof fields, lowercases them, extracts
Unicode word tokens, removes digit-only items, deduplicates, sorts, and writes
UTF-8 text. Its 32,178 forms support candidate ranking but are not an exhaustive
definition of correct Wolof; legitimate inflections and names may be absent.

The split script retains the latest kept annotation with a non-empty target and
assigns complete video groups to train, development, and test using a seeded
greedy approximation to 70/15/15. Its default seed is 2026. It refuses to create
a final split below 200 pairs or five videos and refuses to overwrite locked
files. The manifest stores source/split hashes, ratios, video membership, and
creation time.

This split procedure is implemented and tested. The annotation threshold has
now been reached, and a dry run produces 142 train, 30 development, and 29 test
pairs. No exact cross-split input or pair duplicates and no cross-split inputs
at or above 95 character similarity were found. The normalization policy and
split were frozen on 8 August 2026. Test targets cannot
inform mining, generation, prompts, training, or stopping. The unsupervised
method must also exclude every unpaired comment from development and test
videos, even when that particular comment is not annotated.

### 4.2 Modeling

Every generator produces pairs in the normalization direction
`informal/code-switched Wolof -> formal Wolof`, although generation begins with
a formal sentence and creates its noisy source. The final common schema will
store a source identifier, formal and generated text, method, seed, edits, error
categories, confidence, and provenance. M1–M6 loaders validate the three common
training columns (`source_id`, `source_text`, and `target_text`); method-specific
files and manifests retain richer edit provenance without forcing it into the
Trainer input.

#### 4.2.1 Deterministic phonetic-rule baseline

The implemented rules baseline applies ordered regular expressions to formal
Wolof. It lowercases text, changes hyphenation, reduces selected doubled vowels,
converts `uu` to `ou`, maps `ñ` to `gn`, `ŋ` to `ng`, `ë` to `eu`, and `q`
or `x` to `kh`, and applies context-sensitive rewrites to `c`, `j`, `th`, final
`e`, final `n`, and doubled `g`. Temporary placeholders prevent generated `eu`,
`ou`, and `u` from being transformed again by later expressions.

The latest notebook first runs the multilingual
`Davlan/xlm-roberta-large-ner-hrl` NER model. Named entities, uppercase acronyms
of two or more characters, and numbers are masked; the rules are applied; and
protected substrings are restored. NER uses batches of 32 and currently assumes
GPU device 0.

The method exported 17,777 transformed sentence pairs. An earlier combined file
with dictionary and sentence resources has 27,403 rows. The latter should not be
mixed into sentence-level training without marking dictionary entries. The rules
are deterministic and apply whenever a pattern matches; they do not model the
probability that real writers make each change. Broad operations such as every
final `n` may therefore overgenerate.

Before training, rules must receive stable identifiers and application counts,
while unchanged outputs, duplicates, and protected-span failures must be
measured. A manual sample must be checked for semantic or morphological damage.
A probabilistic version based on gold error frequencies belongs to the
supervised standalone M6 engine, not this baseline.

#### 4.2.2 Eflomal alignment generator

Eflomal [14] estimates links from the 17,777 French–Wolof pairs. The current
notebook tokenizes both sides, writes Eflomal input, and runs batches of 5,000
under Linux/WSL. Forward and reverse files are produced, but the current lexicon
aggregates forward links. For each Wolof token, aligned French counts are
normalized. A candidate is retained if observed at least three times with
probability at least 0.1. The serialized lexicon contains 1,989 Wolof entries and
can be reused from Windows.

For each formal source, French candidates must also occur in its paired French
sentence. The generator selects the highest-probability unused target and
currently attempts to replace approximately 20% of source tokens, with at least
one substitution. Entities, acronyms, and numbers are protected; remaining Wolof
tokens receive the phonetic rules.

The current dictionary-shaped alignment representation remains a methodological
limitation: forward and
reverse links are not symmetrized, and the fixed 20% attempt rate was not
learned from comments. However, the benchmark implementation is now
deterministic (seed 2026), constrains replacements to French tokens from the
paired row, chooses the highest Eflomal probability with lexical tie-breaking,
and logs every substitution. It exported 17,777 pairs, including 17,180 changed
rows, 13,661 rows with alignment edits, and 49,063 alignment substitutions. The
manifest records input/output and lexicon hashes. Its 100-row manual review is
still required before the condition is interpreted as high-quality data.

#### 4.2.3 CMDR-style contextual substitution

The CMDR implementation explores contextual code-switch generation. It extracts
Wolof/French n-grams of length up to three and trains a shared Word2Vec model with
150 dimensions, window 10, minimum count 2, 20 epochs, seed 42, and one worker
for reproducibility. It also builds sentence-level cross-lingual co-occurrence
counts and uses a stable CRC32 UTF-8 hash.

At generation time, French candidates are restricted to n-grams in the paired
French sentence. They must have the same n-gram size, embedding similarity at
least 0.70, and normalized co-occurrence/alignment score at least 0.01. A
Gaussian positional term with standard deviation 0.35 favors similar sentence
locations. The final exploratory cell uses n-grams up to length two and permits
at most two substitutions, followed by entity restoration and Wolof
informalization.

This is **CMDR-style**, not an exact reproduction of a published system. The
deterministic final export contains 17,777 pairs, of which 17,102 changed and
6,176 contain 9,548 accepted alignment substitutions; the shared spelling stage
accounts for the remaining changes. The Word2Vec model, alignment table,
component scores, row-level provenance, hashes, and parameters are saved. Some
distributional neighbors may still be related rather than valid translations,
so the pending 100-row review must be reported and may motivate a pre-test,
documented exclusion rule rather than post-test method selection.

#### 4.2.4 IBM Model 1 portability experiment

An NLTK IBM Model 1 [20] implementation was tested as a Windows-native
alternative to Eflomal. On 2,000 pairs it produced 2,775 accepted links and
1,672 Wolof entries in approximately four seconds with the same minimum count
and probability. Its artifact is compatible with downstream lexicon consumers.

This establishes portability, not equivalent quality. IBM Model 1 is therefore
not a main benchmark method. It will replace Eflomal only if mapping precision
and coverage are first compared on the same manual sample and the choice is made
before test evaluation.

#### 4.2.5 Oolel synthetic-data baseline

The Oolel data [22] were converted to the common normalization direction at the
pinned revision stated above. Automated validation checked wrapper structure,
missing/malformed values, duplicate pairs, unchanged pairs, and exact overlap
with the current gold data without using test targets for cleanup.

Because its 3,433 accepted pairs are fewer and use another formal source pool,
a direct score mixes generation quality, data size, and domain. The primary
table identifies it as external data. Every larger corpus is deterministically
sampled down to 3,433 examples for M7. Five duplicate pairs were rejected from
3,438 raw rows, leaving 3,433 changed accepted pairs; the raw/export hashes and
zero current exact-gold-overlap counts are stored in the manifest. The 100-row
manual review remains incomplete, and the unavailable original prompt seed and
decoding settings remain provenance limitations.

#### 4.2.6 Unsupervised YouTube-aware method

The unsupervised method tests whether real comments can improve generation
without sentence-level gold targets. Its implemented pipeline:

1. build a trusted vocabulary from formal training resources;
2. collect frequent out-of-vocabulary forms from training-eligible comments;
3. retrieve formal candidates using character similarity and reversible
   Wolof-specific graphemic rules;
4. score frequency, context, edit plausibility, and French/entity protection;
5. reject rare, ambiguous, and low-confidence pairs; and
6. invert accepted pairs into formal-to-informal transformations.

A candidate informal form \(u\) and formal form \(s\) may be scored as

\[
S(u,s)=w_c C(u,s)+w_r R(u,s)+w_f F(u)+w_x X(u,s)-w_a A(u,s),
\]

where \(C\) is character similarity, \(R\) indicates a known rewrite family,
\(F\) is corpus frequency, \(X\) is contextual compatibility, and \(A\) penalizes
ambiguity or probable French/entity status. The implemented exploratory weights
are 0.4 for character similarity, 0.3 for rule compatibility, 0.2 for context,
and 0.1 for frequency. Candidates need a total score of at least 0.68 and a
margin of 0.03. These values were fixed without using gold test targets; they
are design choices rather than an optimally tuned result.

Each mapping retains counts, score components, examples, and provenance. The
final mining corpus contains 239,795 eligible comments after excluding both
locked development/test videos. From 19,828 candidates, 114 mappings were
accepted. Applying at most two mappings per formal sentence generated 17,777
pairs, with 1,821 changed and 15,956 unchanged. This low changed rate is an
important limitation and must be interpreted alongside downstream scores.

The dataset and manifest exist, but final M7 use is blocked until the seeded
100-row mapping review and 100-row generation review are completed and the
manifest's finalization list is cleared. Review decisions are quality evidence;
if they are fed back into the mapper, the resulting condition must instead be
labelled semi-supervised and versioned separately.

#### 4.2.7 Standalone gold-informed linguistic engine

The implemented M6 method uses only **gold train** to learn POS-conditioned
error frequencies, an empirical edit-count distribution with an optional cap, reviewed lexical or
spacing mappings, and recurring context-sensitive character transformations.
The finalized evidence contains 1,315 annotated occurrences. After requiring a
reviewed status, Wolof source and target languages, and `protected=no`, 865
occurrences remain eligible, including 588 changed occurrences. The current
thresholds retain 171 mappings and 105 POS-conditioned transformation templates.

Spacing-merge annotations are retained as exact multiword phrase mappings. Their
sentence-like corrected units may have `UNKNOWN` POS without being discarded;
the current profile contains 80 such mappings from 93 observed events.

Curated lookup decisions take precedence over the multilingual seed. French,
entities, ambiguous entries, acronyms, numbers, and sentence-medial title-case
tokens are protected. A stable per-source seed samples an empirical sentence
edit budget and selects non-overlapping eligible transformations. Each output
records positions, error categories, confidence, evidence count, POS, and
mapping source.

M6 deliberately loads neither M5 mappings nor M1 rules. This makes its result
directly comparable with the standalone unsupervised and rule-based conditions.
Optional M5+M6 or M1+M6 combinations may be reported later under distinct names.
Learning training-set errors is ordinary supervised learning, not leakage;
validity depends on never using development/test annotations to define the
generator and on clearly labelling M6 as gold-informed. The uncapped empirical
configuration has provisionally exported 17,777 pairs, including 14,215 changed
rows. Its 100-row manual quality review and manifest finalization remain to be
completed.

#### 4.2.8 Reverse-normalization prototype

An implemented diagnostic module masks protected spans; classifies probable
formal Wolof, translated French, native French, ambiguous, and informal tokens;
reverses alignment substitutions; and ranks vocabulary candidates with weighted
edit distance. A preliminary export contains 111 rows. It supports annotation
but is neither gold truth nor the final learned normalizer. Evaluating it against
synthetic targets made from the same rules would be circular. It may be reported
as a rule/lexicon baseline on the locked human test set.

#### 4.2.9 Downstream seq2seq model

Each controlled corpus trains a fresh LoRA adapter on the same frozen
encoder–decoder. The
selected architecture is `google/mt5-small`, pinned to Hugging Face revision
`73fb5dbe4756edadc8fbe8c769b0a109493acf7a`. mT5 was retained because its
multilingual text-to-text pretraining, SentencePiece vocabulary, and small
variant provide a practical compromise between transfer and the available
8-GB laptop GPU. The fixed revision prevents later changes to the model
repository from silently changing the experiment. No generator receives its
own preferred architecture, adapter configuration, or hyperparameter search.

Full 300-million-parameter fine-tuning was tested and rejected for the main
benchmark after the gold diagnostic required approximately seven minutes per
optimizer step, used 82.5% of the machine's 16-GB RAM, and left the GPU near 20%
utilization. LoRA freezes the shared mT5-small base and adds rank-8 updates to
the attention `q` and `v` projections. The implemented adapter has 344,064
trainable parameters out of 300,520,832 total parameters (0.1145%). This retains
the same pretrained representation while avoiding full-model gradient and
optimizer states.

The model input is the informal sentence prefixed with `normalize Wolof: ` and
the target is the formal sentence. Inputs are truncated to 192 subword tokens
and targets to 96 subword tokens.
The pretrained SentencePiece tokenizer is retained. A character-only
replacement was rejected for the primary benchmark because it would discard
the pretrained embedding interface and introduce a second architectural
variable. Character-level variation is nevertheless represented through the
model's subword segmentation and the synthetic spelling changes in the input.

The benchmark includes:

- a direct no-normalization identity baseline, calculated without training;
- a gold-only adapter;
- one synthetic-pretrained adapter per retained method, used only as an
  auditable prerequisite; and
- the same adapter subsequently continued on identical gold-train data and
  reported as that method's final result.

Architecture, initialization, tokenizer, sample count or explicit size-matching,
source selection, update budget, checkpoint rule, decoding, and seeds are fixed
across conditions. M1–M6 are capped at 3,433 pairs, the number of accepted M4
Oolel pairs. M4 remains an external-source condition because size matching does
not remove its formal-source confound. The initial final design uses seed 2026.
Seeds 2027 and 2028 will be added only if the complete design can be repeated
before any test result is inspected; otherwise the single-seed limitation will
be stated explicitly.

The synthetic prerequisite stage tunes its adapter for one epoch. The two-stage
condition starts from that method's selected adapter checkpoint and continues it
for five epochs on the same 142 gold-training pairs at a lower learning rate.
The frozen base weights remain identical throughout. It is therefore a true
curriculum, not a concatenation of gold and synthetic rows. The 30-item
development split is evaluated with teacher forcing at every epoch, and the
checkpoint with the lowest development loss is retained. Autoregressive
development predictions and the CER, WER, chrF, exact-match, correction, and
overcorrection metrics are then computed once from the restored checkpoint.
This avoids repeatedly decoding the development set while preserving an
identical checkpoint rule across conditions. Training never uses the 29 test
targets.

**Table 4.2 – Frozen M7 training configuration**

| Parameter | Final value |
|---|---|
| Model/tokenizer checkpoint | `google/mt5-small` |
| Exact model revision | `73fb5dbe4756edadc8fbe8c769b0a109493acf7a` |
| Task direction and prefix | informal → formal; `normalize Wolof: ` |
| Maximum input and target length | 192 / 96 subword tokens |
| Optimizer and weight decay | Adafactor; 0.01 |
| Adaptation method | LoRA, sequence-to-sequence task, frozen base |
| LoRA configuration | rank 8, alpha 16, dropout 0.05, targets `q` and `v`, no bias |
| Trainable parameters | 344,064 / 300,520,832 (0.1145%) |
| Synthetic-stage learning rate | $1\times10^{-3}$ |
| Gold fine-tuning learning rate | $5\times10^{-4}$ |
| Physical/effective training batch | 32 / 32, using 1 accumulation step |
| Evaluation batch | 8 |
| Epoch budget | 1 synthetic prerequisite; 5 gold fine-tuning; no early stopping |
| Checkpoint selection | minimum teacher-forced development loss among epoch checkpoints |
| Precision and performance controls | bfloat16 and TF32; strict deterministic kernels disabled while fixed seeds are retained; gradient checkpointing disabled; one retained adapter checkpoint |
| Dropout, clipping, label smoothing | pretrained defaults; Trainer max-gradient norm default; no label smoothing |
| Decoding strategy | greedy decoding (1 beam); maximum 96 tokens; generated once on development after checkpoint selection |
| Seed(s) | 2026; 2027 and 2028 only if the full design is repeated |
| Rows per synthetic condition | 3,433 deterministic sampled pairs |

The implementation is resumable and versioned. Each condition is written to
`checkpoints/m7_seq2seq/<method>/seed_<seed>/<regime>/attempt_<number>/` with a
fingerprint of the data, configuration, and parent synthetic checkpoint. An
unchanged completed run is skipped; an interrupted attempt resumes from its
latest Trainer checkpoint; changed inputs or an explicit rerun create the next
attempt without overwriting earlier evidence. This permits one modified method
to be rerun independently while keeping every comparison auditable.

Each attempt saves the selected model, input/configuration manifest, software
and hardware metadata, development predictions, metric values, file hashes,
training history, and a training-curve figure. The runner also creates a global
run index, run-level and aggregate CSV tables, Markdown result tables, and PNG
comparison plots under `results/m7_seq2seq/`. Locked-test evaluation is a
separate command requiring explicit confirmation of the frozen attempts. It may
resume an interrupted evaluation only for the identical frozen selection and
refuses to replace that selection with later attempts.

At the time of writing, the runner and its tests are complete and the model has
been downloaded. M6 has been provisionally exported, while M5 and M6 retain
unresolved review/finalization gates. The project owner authorized a provisional
execution with these gates bypassed. A `quality_gate_override=true` field is
therefore written into affected manifests, and Chapter 5 must identify this
choice unless the same datasets are subsequently reviewed and finalized.

#### 4.2.10 Oolel-Corrector external baseline

`Oolel-Corrector` [23] is a roughly two-billion-parameter Qwen2 causal model
fine-tuned on Oolel data under AGPL-3.0. It will be run unchanged on the same
locked test inputs, without test fine-tuning or manual cleanup. Exact revision,
prompt, decoding, output parsing, hardware, memory, and latency must be recorded.

This is an **external system comparison**, not a synthetic-data condition. Its
architecture, parameter count, and training history differ, so it belongs in a
separate results table. It answers whether the project models are competitive
with an available corrector, not which synthetic generator is intrinsically
best.

### 4.3 Training and infrastructure

Development is primarily performed on Windows. Eflomal compilation/alignment
runs under WSL/Linux because native Windows installation is difficult. Its
serialized lexicon is then reused by Windows annotation and generation tools.
Gemma 3 12B is served locally through Ollama. M7 training runs natively under
Windows with PyTorch/Transformers on an NVIDIA GeForce RTX 5060 Laptop GPU with
8,151 MiB reported VRAM and driver 581.91. The current interpreter is Python
3.11.0; the modeling environment contains PyTorch 2.8.0+cu129, Transformers
5.5.4, Datasets 2.20.0, and Accelerate 1.13.0. These observed versions must be
recorded again in the final run manifest in case the environment changes.

The project currently depends on Python, pandas, a Parquet engine, NLTK,
Eflomal, Gensim, Transformers, PyTorch, Flask, tqdm,
`youtube-comment-downloader`, `yt-dlp`, `langdetect`, Ollama, SentencePiece, and
Matplotlib. `requirements.txt` specifies compatible version ranges; the run
manifests record the actual modeling-library versions. A fully frozen environment
export should still be archived with the submitted experiment.

`data/` stores registered corpora, `artifacts/` learned lexicons,
classification directories append-only checkpoints, and annotation files
append-only gold decisions/corrections. The experiment runner stores one
versioned attempt per method, seed, and regime under `checkpoints/m7_seq2seq/`.
Aggregate development and frozen-test evidence is written under
`results/m7_seq2seq/`. Final runs record input/configuration fingerprints,
dataset and model revisions, seeds, software versions, GPU identity, peak CUDA
allocation, runtime metrics, prediction hashes, and output paths. Because the
working folder is not currently a Git repository, a source archive identifier
or initialized Git commit must be added manually before submission.

Sixty-two workflow tests currently cover central data paths, split integrity,
alignment compatibility, M5 finalization, M6 isolation and linguistic guards,
M7 metrics and leakage checks, deterministic attempt fingerprints, selective
reruns, gold-only isolation, LoRA configuration, checkpoint-selection policy,
retained-method selection, automatic two-stage training, safe generated-token decoding, and result aggregation. All 62 pass, and the M7
modules compile. The
final package still requires a full notebook-order and UTF-8 audit after the
last results are inserted.

CMDR and Eflomal now write deterministic method datasets, row-level provenance,
review sheets, and manifests. M7 similarly refuses accidental overwrite and
does not rely on notebook display as evidence. M6 still requires its final
export and manifest finalization.

The following information is unavailable in the current files and must be added
manually before submission:

- CPU and installed RAM;
- exact Windows edition/build and WSL distribution/version;
- final compute and storage allocation;
- collection, generation, training, and inference durations;
- energy or monetary cost where available;
- data-collection and experiment dates; and
- any remote compute or hosted services used (currently none recorded for M7).

If compute does not allow three seeds, the reduced design must be decided before
test inspection and applied equally. Bootstrap intervals over test sentences do
not replace training-seed variation, so that limitation must be explicit.

### 4.4 Evaluation and metrics

Evaluation has three levels: intrinsic synthetic-corpus analysis, downstream
normalization metrics, and blinded human judgment. No single metric captures
orthographic correction, naturalness, and meaning preservation.

#### 4.4.1 Intrinsic data analysis

Before training, each synthetic corpus will be profiled by:

- total and unique pairs, exact/cross-method duplicates, and unchanged rate;
- character/token edits and edit-count distribution;
- insertion, deletion, substitution, and annotated error-type coverage;
- vocabulary, mapping, and formal-source coverage;
- French/code-switch rate and switch positions;
- sentence length and protected-span violations;
- rejection rates/reasons and mapping confidence;
- overlap with gold train, development, and test; and
- semantic-corruption rate in a manual sample.

Distributions are required, not only averages. Many unchanged inputs can inflate
accuracy, while an aggressive generator may increase apparent diversity by
damaging meaning.

Approximately 100 outputs per retained method should be reviewed blind by at
least one Wolof speaker, preferably two. The same formal sources should be used
across methods where possible. Reviewers will rate meaning preservation,
plausibility as informal writing, code-switch appropriateness, severity, and
synthetic artifacts. Method labels will be hidden and order randomized.

#### 4.4.2 Automatic normalization metrics

Let \(y\) be the gold formal sentence and \(\hat{y}\) the prediction.
**Character error rate** is

\[
\operatorname{CER}=\frac{S_c+D_c+I_c}{N_c},
\]

where substitutions, deletions, and insertions are divided by the number of gold
characters. CER is well suited to diacritics and short Wolof graphemic changes.
Lower is better.

**Word error rate** applies the same calculation to tokens:

\[
\operatorname{WER}=\frac{S_w+D_w+I_w}{N_w}.
\]

WER measures complete-token and boundary recovery but may count a one-character
mistake as a whole-word failure. It is therefore complementary to CER.

**chrF** is an F-score over character n-gram precision and recall. It provides
partial credit for close strings and is less brittle than exact match. Its
n-gram order, beta, casing, and whitespace handling must be fixed in advance.

**Exact match** is the proportion of complete predictions equal to the target.
Raw and minimally normalized exact match may both be reported if the
normalization is defined before evaluation and used for every system.

**Token normalization precision, recall, and F1** distinguish required from
harmful changes:

\[
P=\frac{TP}{TP+FP},\qquad
R=\frac{TP}{TP+FN},\qquad
F_1=\frac{2PR}{P+R}.
\]

A true positive is a necessary token change corrected to the gold form; a false
negative is a required change left wrong; and a false positive is correct input
material changed incorrectly. Attempted but wrong corrections must count as
errors. Insertions, deletions, and many-to-one boundaries complicate alignment,
so the final scorer needs unit tests and manually verified examples.

**Overcorrection rate** measures incorrect changes to already acceptable
material. It is important because normalization in front of retrieval,
translation, or an LLM must not damage names, facts, numbers, or intentional
French. Protected content and intentional code-switching should be reported
separately where annotation permits.

Metrics will be reported for the full corpus and the subset that actually
requires a change. The copy baseline may score well on unchanged material, so
aggregate accuracy alone would be misleading. Error-category scores will be
shown only when enough test examples support them.

#### 4.4.3 Statistical comparison

For conditions with at least three runs, the report will give the mean and
standard deviation across seeds. Paired bootstrap resampling at sentence level
will estimate confidence intervals for differences on the fixed test set.
Marginal differences will be accompanied by examples and not interpreted only
from rounded scores.

The number of hypotheses will remain tied to the research questions. Given the
minimum gold size, rare error categories may have little statistical power and
will be described qualitatively rather than ranked conclusively.

#### 4.4.4 Human system evaluation

Reference metrics cannot determine whether a different valid normalization
preserves meaning. Reviewers will therefore rate a randomized, anonymized sample
of test predictions for semantic fidelity, formal Wolof acceptability,
intentional French handling, protected-content preservation, and downstream
usefulness. The scale and examples must be written before predictions are shown.

With two reviewers, agreement and adjudicated scores will be reported. With one,
the limitation will be stated and a subset should be re-rated after a delay to
estimate intra-annotator consistency.

#### 4.4.5 Comparisons and ablations

The controlled table will include copy, gold-only, rules, the retained
alignment/CMDR method, Oolel published data, unsupervised YouTube-aware data, and
the proposed standalone linguistic engine. Each row will mark whether it uses unpaired YouTube data,
gold training labels, external formal sources, and size matching. Synthetic-only
and synthetic-plus-gold regimes will be separated.

Oolel-Corrector will appear in a second external-system table beside the best
project model, with architecture, parameter count, original data, license,
hardware, and latency. M6 may be ablated into lexical-mapping-only and
generalized-transformation-only conditions; optional combinations must remain separately named. If both Eflomal and CMDR survive development review, they will
be separate conditions; otherwise the report will give the pre-test rule used to
select one.

#### 4.4.6 Optional retrieval evaluation

If time remains after the normalization benchmark, the original retrieval
motivation can be tested with a formal document collection, informal queries,
and relevance judgments. Raw and normalized queries would be compared using
Recall@\(k\), mean reciprocal rank, and qualitative failures.

This experiment is **not implemented**, and paired relevance judgments do not
currently exist. It must remain future work unless the corpus, queries, qrels,
retriever, and evaluation script are completed. Normalization accuracy alone
will not be presented as proof of retrieval improvement.

#### 4.4.7 Work remaining before final results

The following steps are still required:

1. complete the standalone M6 100-row manual quality review and finalize its
   provisional manifest;
2. complete the M2–M5 review sheets and clear M5's remaining manifest gates;
3. rerun the M7 status command and freeze the final dataset hashes;
4. record CPU/RAM, the exact Windows/WSL versions, storage, dates, and the source
   archive or Git identifier;
5. execute the matched seq2seq training and gold fine-tuning with the frozen
   runner configuration;
6. inspect the generated development tables and plots, and rerun only a method
   when a justified pre-test implementation correction is necessary;
7. freeze the retained attempt paths, then perform the one-time gold-test
   evaluation without further tuning;
8. complete qualitative/error-category analysis and any predeclared ablation;
   and
9. evaluate Oolel-Corrector once on the locked test set after its overlap and
   revision checks.

Until these tasks are complete, Chapter 5 must not contain invented model scores.
Current corpus counts and prototypes may be reported as intermediate engineering
results, but conclusions about comparative quality remain pending.

## 5. Results and analysis

Recommended final length: 10–15 pages. Present only completed results, with numbered figures and tables.

### 5.1 Quantitative results

At the current implementation freeze, the six retained method files exist.
Table 5.1 also preserves M0 as an archived engineering artifact, but M0 is not
trained or included as a synthetic-generation method. A changed output may
still be unnatural or semantically wrong.

**Table 5.1 – Current synthetic-corpus export status before final matching**

| Method | Generator/source | Exported pairs | Changed pairs | Final evidence still required |
|---|---|---:|---:|---|
| M0 | archived identity artifact; not trained | 17,777 | 0 | direct no-normalization score only |
| M1 | Beqi-inspired rules | 17,777 | 17,017 | intrinsic profile/report integration |
| M2 | Eflomal alignment + shared rules | 17,777 | 17,180 | 100-row review |
| M3 | CMDR-style alignment + shared rules | 17,777 | 17,102 | 100-row review |
| M4 | external Oolel LLM-generated data | 3,433 | 3,433 | 100-row review and external-source caveat |
| M5 | unsupervised YouTube mining | 17,777 | 1,821 | mapping/generation reviews and manifest finalization |
| M6 | gold-train linguistic engine | 17,777 | 14,215 | 100-row review and final manifest |

The controlled runner deterministically selects 3,433 rows from every retained
M1–M6 method.
Its readiness output is `results/m7_seq2seq/dataset_status.csv`. The current
execution deliberately bypasses the incomplete review gates and records that
override in each affected manifest; no model score is reported until a run
actually completes.

After training, the run-level development results will be generated in
`results/m7_seq2seq/development_summary.csv`; means and variation will be in
`development_aggregate.csv`; the report-ready Markdown table and figure will be
`development_results.md` and `development_metrics.png`. After the attempts are
frozen, the analogous locked-test artifacts will be `benchmark_summary.csv`,
`test_aggregate.csv`, `test_results.md`, and `test_metrics.png`. Values must be
copied from those artifacts without manual recomputation.

Use two explicitly separated result tables:

1. **Controlled synthetic-data comparison:** fresh copies of the same seq2seq model trained on rules, alignment/CMDR, Oolel-generated data, standalone unsupervised YouTube-aware data, and standalone gold-informed linguistic data, with source-corpus exceptions clearly marked.
2. **External system comparison:** the project's best controlled models and `Oolel-Corrector` evaluated on the same locked gold test set. This table must identify differences in architecture, original training data, parameter count, and inference cost.

### 5.2 Qualitative results

Include representative generations, normalization successes, failures, overcorrections, and error-category examples. Anonymize user content appropriately.

### 5.3 Discussion and critical analysis

Interpret results against the research questions. Discuss practical significance, uncertainty, biases, generalizability, and threats to validity. Explain marginal or negative findings rather than suppressing them.

## 6. Conclusion and outlook

Recommended final length: 3–5 pages.

### 6.1 Summary of contributions

Answer each research question and compare completed contributions with the initial objectives.

### 6.2 Limitations and areas for improvement

Discuss data size, video coverage, annotation subjectivity, synthetic artifacts, compute limits, multilingual ambiguity, and downstream scope.

### 6.3 Perspectives

Propose larger community-reviewed data, retrieval evaluation, robust LLM querying, machine translation, alternative architectures, and responsible release.

## Bibliography

The report uses IEEE numbering. The following working bibliography supports Sections 1–3 and already satisfies the minimum count of 20 references. Before submission, verify formatting against the school's required style and add every source used in Chapters 4–6.

[1] K. Al Sharou, Z. Li, and L. Specia, “Towards a Better Understanding of Noise in Natural Language Processing,” in *Proc. RANLP*, pp. 53–62, 2021, doi: 10.26615/978-954-452-072-4_007.

[2] A. Das and B. Gambäck, “Code-Mixing in Social Media Text: The Last Language Identification Frontier?” *Traitement Automatique des Langues*, vol. 54, no. 3, pp. 41–64, 2013.

[3] C. M. B. Dione, “Finite-State Tokenization for a Deep Wolof LFG Grammar,” *Bergen Language and Linguistics Studies*, vol. 8, no. 1, 2017, doi: 10.15845/bells.v8i1.1340.

[4] D. Mbaye and M. Diallo, “Beqi: Revitalize the Senegalese Wolof Language with a Robust Spelling Corrector,” in *Innovations and Interdisciplinary Solutions for Underserved Areas*, pp. 311–325, 2025, doi: 10.1007/978-3-031-86493-3_25. Preprint: arXiv:2305.08518, 2023.

[5] S. Gouws, D. Hovy, and D. Metzler, “Unsupervised Mining of Lexical Variants from Noisy Text,” in *Proc. First Workshop on Unsupervised Learning in NLP*, pp. 82–90, 2011.

[6] C. Li and Y. Liu, “Improving Text Normalization via Unsupervised Model and Discriminative Reranking,” in *Proc. ACL Student Research Workshop*, pp. 86–93, 2014, doi: 10.3115/v1/P14-3012.

[7] A. Roy, S. Ghosh, K. Ghosh, and S. Ghosh, “An Unsupervised Normalization Algorithm for Noisy Text: A Case Study for Information Retrieval and Stance Detection,” *ACM Journal of Data and Information Quality*, vol. 13, no. 3, Art. 17, pp. 1–25, 2021, doi: 10.1145/3418036.

[8] K. Dekker and R. van der Goot, “Synthetic Data for English Lexical Normalization: How Close Can We Get to Manually Annotated Data?” in *Proc. LREC*, pp. 6300–6309, 2020.

[9] A. Vaswani *et al.*, “Attention Is All You Need,” in *Advances in Neural Information Processing Systems*, vol. 30, 2017.

[10] C. Raffel *et al.*, “Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer,” *Journal of Machine Learning Research*, vol. 21, no. 140, pp. 1–67, 2020.

[11] L. Xue *et al.*, “mT5: A Massively Multilingual Pre-trained Text-to-Text Transformer,” in *Proc. NAACL-HLT*, pp. 483–498, 2021, doi: 10.18653/v1/2021.naacl-main.41.

[12] Y. Liu *et al.*, “Multilingual Denoising Pre-training for Neural Machine Translation,” *Transactions of the Association for Computational Linguistics*, vol. 8, pp. 726–742, 2020, doi: 10.1162/tacl_a_00343.

[13] G. Jawahar, E. M. B. Nagoudi, M. Abdul-Mageed, and L. V. S. Lakshmanan, “Exploring Text-to-Text Transformers for English to Hinglish Machine Translation with Synthetic Code-Mixing,” in *Proc. Fifth Workshop on Computational Approaches to Linguistic Code-Switching*, pp. 36–46, 2021, doi: 10.18653/v1/2021.calcs-1.6.

[14] R. Östling and J. Tiedemann, “Efficient Word Alignment with Markov Chain Monte Carlo,” *The Prague Bulletin of Mathematical Linguistics*, vol. 106, pp. 125–146, 2016, doi: 10.1515/pralin-2016-0013.

[15] D. Jin, Z. Jin, Z. Hu, O. Vechtomova, and R. Mihalcea, “Deep Learning for Text Style Transfer: A Survey,” *Computational Linguistics*, vol. 48, no. 1, pp. 155–205, 2022, doi: 10.1162/coli_a_00426.

[16] J. He, X. Wang, G. Neubig, and T. Berg-Kirkpatrick, “A Probabilistic Formulation of Unsupervised Text Style Transfer,” in *Proc. ICLR*, 2020.

[17] Z. Zhang, S. Ren, S. Liu, J. Wang, P. Chen, M. Li, M. Zhou, and E. Chen, “Style Transfer as Unsupervised Machine Translation,” arXiv:1808.07894, 2018.

[18] C.-Y. Li and N. T. Vu, “Improving Code-Switching Language Modeling with Artificially Generated Texts Using Cycle-Consistent Adversarial Networks,” in *Proc. Interspeech*, pp. 1057–1061, 2020.

[19] K. R. Chandu and A. W. Black, “Style Variation as a Vantage Point for Code-Switching,” in *Proc. Interspeech*, pp. 4761–4765, 2020, doi: 10.21437/Interspeech.2020-2574.

[20] P. F. Brown, V. J. Della Pietra, S. A. Della Pietra, and R. L. Mercer, “The Mathematics of Statistical Machine Translation: Parameter Estimation,” *Computational Linguistics*, vol. 19, no. 2, pp. 263–311, 1993.

[21] European Parliament and Council of the European Union, “Regulation (EU) 2016/679 (General Data Protection Regulation),” *Official Journal of the European Union*, L 119, pp. 1–88, 2016, especially Arts. 5 and 89.

[22] Soynade Research, “Wolof Non-Standard to Standard Parallel Pairs,” Hugging Face Datasets, 2026. [Online]. Available: https://huggingface.co/datasets/soynade-research/Wolof-Non-Standard-Orthography. License: CC BY-SA 4.0. Accessed: Jul. 26, 2026.

[23] Soynade Research, “Oolel-Corrector: A Fine-Tuned Wolof Spelling Corrector,” Hugging Face Models, 2026. [Online]. Available: https://huggingface.co/soynade-research/Oolel-Corrector. License: AGPL-3.0. Accessed: Jul. 26, 2026.

## Appendices

The appendices contain supplementary material rather than information essential to understanding the report.

### Appendix A — Significant code extracts

Include short commented extracts from collection, filtering, splitting, unsupervised mining, standalone linguistic generation, and evaluation. Link to the complete repository where permitted.

### Appendix B — Additional data and results

Include extended descriptive statistics, annotation examples, error taxonomy, additional result tables, and mapping-review forms.

### Appendix C — Technical environment

Include exact library versions, hardware configuration, Windows/WSL differences, installation commands, model identifiers, and reproducibility instructions.
