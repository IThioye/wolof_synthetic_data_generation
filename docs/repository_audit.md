# Repository publication audit

**Audit date:** 28 September 2026

This document separates material required for a reproducible public repository
from local working data and historical experiments.

## Cleanup applied

- Added a project-level README and registries for data, notebooks, artifacts,
  results, the report, and the presentation.
- Replaced the broken Eflomal submodule entry with a pinned optional-build guide
  and retained the checksum-verified final M2 lexicon.
- Added consistent text/binary Git attributes and expanded local/generated-file
  ignores.
- Removed the byte-identical duplicate `artifacts/Lexique4.tsv`; the canonical
  copy remains under `data/Lexique400/`.
- Added a pinned, checksum-verifying downloader for external Hugging Face,
  GitHub, and Lexique inputs while preserving the original configured names.
- Removed raw YouTube files, downloaded corpora, locally generated synthetic
  CSVs, archive snapshots, generated Word2Vec binaries, and the classifier log
  from the public tree. Existing local files were preserved and are ignored.
- Removed Python/test caches and moved the loose defence-planning note into
  `legacy/`.
- Verified the reorganized repository with 70 passing tests and a UTF-8 audit
  covering 254 text files.
- Rebuilt the public branch as a clean root commit after creating an ignored
  local recovery bundle of the former history.

## Keep in the public repository

- `src/`, `scripts/`, and `tests/`;
- the canonical notebooks listed in `notebooks/README.md`;
- Gold and synthetic manifests, subject to the data-release decision below;
- the final lightweight metrics, predictions, tables, and plots identified in
  `results/README.md`;
- project documentation under `docs/`;
- the clean report and defence packages under `report/` and `presentation/`.

## Keep locally, but do not publish by default

| Material | Reason |
| --- | --- |
| `checkpoints/` | About 1.5 GB; reproducible trained weights and optimizer state. |
| local `eflomal/` checkout and `.venv` | About 8 GB; optional third-party source plus a Linux environment. |
| `data/youtube/` | Large raw/cleaned comment corpus; the compact cleaned Gold evidence is versioned separately. |
| downloaded external datasets | Restored by `scripts/download_external_data.py`; redistribution is unnecessary. |
| generated synthetic CSVs | Reconstructed by the retained method notebooks; manifests remain versioned. |
| `artifacts/cmdr_word2vec.model*` | Roughly 148 MB of regenerable learned arrays. |
| `Research Papers/` | Copyrighted papers should be cited, not redistributed. |
| `Final Report/`, `Final Presentation/` | Working files, previews, references, and superseded versions. |

## Retain as provenance, but mark as archive

- local annotation and split archive directories;
- local M5 outputs, because M5 was implemented but excluded from the final benchmark;
- removed compact/copy-aware/mT5 development notebooks, whose conclusions are
  summarized in `docs/experiment_failures.md`;
- `legacy/annotation_app.py` and the original report skeleton.

These files explain the research path, but the README and reproduction guide
must not present them as the final method.

## Verified duplication

`artifacts/Lexique4.tsv` and `data/Lexique400/Lexique4.tsv` had identical
SHA-256 hashes. The canonical path is now `data/Lexique400/Lexique4.tsv`; the
duplicate artifact copy can be restored from the local pre-cleanup bundle if
needed.

## Decisions still required before making the repository public

1. Choose a source-code licence. No licence should be inferred from the licences
   of Eflomal, Oolel, Lexique400, or the institutional templates.
2. If model weights are published, use a release asset or a model registry with
   checksums and a model card rather than ordinary Git blobs.

## Recommended public release boundary

The selected release is code + canonical notebooks + compact Gold evidence +
manifests + aggregate results + report/presentation. External corpora,
large comment collections, generated synthetic datasets, checkpoints, and
regenerable binary models remain local.
