# Unpaired Wolof pretraining corpus

## Purpose

This corpus is intended for language-representation pretraining before the TNT
normalization objective. It is not another synthetic normalization method and
does not contain informal-to-formal targets. Its two domains are:

- formal Wolof news from the local Defuwaxu snapshot;
- informal and code-switched Wolof comments from YouTube videos that do not
  occur in the Gold normalization benchmark.

## Leakage controls

The builder excludes all five videos represented in Gold train, development,
and test. It also excludes any row whose normalized text, with or without
punctuation, exactly matches a Gold informal source or formal reference. Whole
videos and whole news articles are assigned to only one pretraining split.
The manifest verifies zero document overlap and zero exact text overlap between
train, development, and test.

The existing binary comment labels from the three Gold-training videos are used
only to learn a conservative Wolof-content selector. Labels from the Gold
development and test videos are excluded, and comments that match any Gold
input are removed before fitting that selector. No Gold correction or token
annotation is used. At the selected 0.80 probability threshold, the manifest
records leave-one-training-video-out precision and recall. The deliberately
high threshold prefers corpus cleanliness over retaining every possible
comment.

## Build and outputs

Run from the repository root:

```bash
python -m src.pipelines.build_wolof_pretraining_corpus
```

The command creates these local files under `data/pretraining/`:

- `wolof_pretraining_corpus.parquet`;
- `wolof_pretraining_train.parquet`;
- `wolof_pretraining_dev.parquet`;
- `wolof_pretraining_test.parquet`;
- `wolof_pretraining_review_sample.csv`;
- `wolof_pretraining_manifest.json`.

The Parquet schema is `source_id`, `domain`, `document_id`, `segment_index`,
`text`, `selector_score`, `french_ratio`, and `split`. The selector fields are
empty for news because they only describe comment filtering. Both domains are
segmented at word boundaries to at most 240 characters, leaving room for the
special tokens used by the current 256-position TNT encoder.

## Required manual checks

Before pretraining, review the 150-row sample and fill the two review columns.
In particular, verify that accepted comments contain useful Wolof or relevant
Wolof/French code-switching. Also document how `news_data.parquet` was acquired
and its licence or terms of use. Until that point, the news text and generated
Parquet files must stay local and must not be redistributed.

## Encoder pretraining

`src/modeling/wolof_encoder_pretraining.py` implements a TNT-compatible masked
character encoder. It masks 15% of characters in spans of one to three, uses a
tied prediction head, selects the checkpoint by pretraining-development loss,
and supports epoch-level resume. A weighted sampler exposes the model to 25%
formal-news and 75% informal-comment rows during training without modifying the
stored corpus. The separate pretraining test split is evaluated only after the
best development checkpoint has been selected.
