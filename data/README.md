# Data registry

The project distinguishes canonical benchmark data from raw, external, and
historical material. Paths are registered in `src/config.py`; detailed
provenance is recorded in `docs/Data_lineage.md`.

## Canonical project data

| Path | Role |
| --- | --- |
| `annotations/splits/` | Frozen video-disjoint Gold train/dev/test split: 142/30/29 pairs. |
| `annotations/gold_annotations.csv` | Append-only sentence annotation history. |
| `annotations/token_corrections.csv` | Token-level correction memory. |
| `annotations/token_linguistic_annotations.csv` | Gold-train linguistic enrichment used by M6. |
| `synthetic/*_manifest.json` | Checksums and parameters for locally generated datasets. M5 remains excluded from the final trained comparison. |
| `external_sources.json` | Pinned download registry for external corpora and Lexique. |

## Local-only or licence-sensitive data

Restore all registered external inputs at the exact paths expected by
`src/config.py`:

```bash
python scripts/download_external_data.py --all
python scripts/download_external_data.py --all --verify-only
```

The following paths remain local and are ignored:

- `youtube/`: raw and cleaned public comments plus source URLs;
- `external/*.parquet`: downloaded third-party datasets;
- `translation data/`: downloaded or locally transformed corpora, except its
  source registry document;
- `Lexique400/`: downloaded French lexical data;
- generated synthetic CSVs and review samples.
- `pretraining/*.parquet`: leakage-audited formal-news and informal-comment
  encoder-pretraining splits. The manifest is versioned, but the text remains
  local because the news snapshot's redistribution licence is unresolved.

Rebuild the unpaired pretraining corpus after restoring the local YouTube and
news inputs:

```bash
python -m src.pipelines.build_wolof_pretraining_corpus
```

This command excludes all five Gold videos and every exact normalized Gold
source/reference string, then assigns whole videos/articles to deterministic
90/5/5 train/dev/test splits. Inspect
`pretraining/wolof_pretraining_review_sample.csv` before model pretraining.

The compact Gold annotations and frozen splits are versioned because they are
the irreplaceable evaluation evidence. The larger cleaned comment pool remains
local even though direct identifiers were removed during cleaning.

## Historical material

Folders whose names contain `archives`, along with `splits_v2_corrected/`, are
local provenance snapshots and are not published. The canonical split is
always `annotations/splits/`.

Do not regenerate or tune against the frozen Gold test targets.
