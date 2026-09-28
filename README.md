# Synthetic Data Generation for Robust Informal Wolof Normalization

This repository contains an independent aivancity PFE investigating whether
synthetic formal-to-informal Wolof pairs can improve normalization of authentic,
informal, and French-Wolof code-switched YouTube comments.

The project is primarily a **data-generation benchmark**, not a claim to have
built a production Wolof corrector. Each retained generator supplies synthetic
pretraining pairs to the same evaluator; every trained condition is then
fine-tuned on the same small Gold-training split and evaluated on the same
video-disjoint authentic test set.

## Research question

> How well do formal-to-informal synthetic generation strategies improve
> normalization of authentic, code-switched Wolof YouTube comments under the
> same architecture and training protocol?

## Final benchmark

| ID | Method | Role |
| --- | --- | --- |
| M0 | Unchanged input | Direct identity reference; not trained. |
| Gold | Gold only | Evaluator trained only on 142 authentic training pairs. |
| M1 | Beqi-inspired rules | Deterministic Wolof spelling transformations. |
| M2 | Eflomal + M1 | Bilingual alignment-based French substitutions followed by M1. |
| M3 | CMDR-style mixing + M1 | Distributional bilingual substitutions followed by M1. |
| M4 | Oolel data | External LLM-generated non-standard Wolof pairs. |
| M6 | Gold-train linguistic engine | Dominant mappings and POS/context templates learned from Gold train only. |

M5, an unsupervised YouTube lexical-mining prototype, remains documented but is
excluded from the final trained comparison because only 114 mappings passed its
filters and most generated rows remained unchanged.

The final evaluator is a 2.15-million-parameter, character-level,
non-autoregressive edit Transformer inspired by TNT. Each synthetic condition
uses 3,330 matched formal targets, 20 synthetic pretraining epochs, and 20 Gold
fine-tuning epochs. Checkpoints are selected by Gold-development CER; the frozen
29-sentence test set is evaluated only after selection is recorded.

## Repository layout

```text
artifacts/      inspectable mappings, templates, and generation manifests
data/           annotations, frozen splits, synthetic exports, and data registry
docs/           methodology, annotation policy, lineage, and project records
legacy/         superseded application code retained for provenance
notebooks/      experiment controllers and notebook index
presentation/   clean final defence deck and jury preparation material
report/         clean LaTeX source and final report PDF
results/        canonical metrics, predictions, figures, and result registry
scripts/        reporting, scraping, and maintenance entry points
src/            reusable annotation, generation, pipeline, and modeling modules
tests/          workflow, leakage, architecture, and metric tests
```

See [the repository publication audit](docs/repository_audit.md) for the
keep/local/archive decision behind this layout.

## Environment

The principal workflow uses Python 3.11 on Windows. Create an isolated
environment and install the dependencies:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

Download the external datasets into the exact configured paths with:

```bash
python scripts/download_external_data.py --all
```

M2 normally consumes the small frozen `artifacts/lexicon.pkl` included in the
repository. Rebuilding it with Eflomal is optional, requires Linux or WSL, and
is documented in `docs/eflomal_reproducibility.md`. Eflomal itself is not
embedded in this repository.

## Verification

Run the automated tests from the repository root:

```bash
python -m pytest
```

Audit project text encoding with:

```bash
python scripts/audit_utf8.py
```

## Reproducing the final evidence

1. Download the registered external resources and verify their checksums:
   `python scripts/download_external_data.py --all`.
2. Read `data/README.md` and `docs/Data_lineage.md`, then inspect or regenerate
   M1--M6 with their corresponding notebooks. M5 is an
   audit artifact, not a final trained condition.
3. Use `notebooks/m7_tnt_edit_gold_dev_benchmark.ipynb` to inspect or rerun one
   retained benchmark condition. Do not reopen the frozen test during tuning.
4. Rebuild the report tables and figures without retraining:

   ```bash
   python scripts/reporting/build_final_benchmark_analysis.py
   ```

The canonical run and analysis directories are documented in
`results/README.md`. Model checkpoints are intentionally excluded from normal
Git storage.

## Deliverables

- [Final report source and PDF](report/)
- [Final defence presentation and jury notes](presentation/)
- [Final result registry](results/README.md)
- [Notebook workflow index](notebooks/README.md)

## Data, privacy, and licences

Raw comment collections, external datasets, generated synthetic corpora,
checkpoints, and regenerable binary models are excluded. The compact cleaned
Gold annotations and frozen splits are retained as the evaluation evidence.
External sources are restored from their original hosts by the pinned download
registry instead of being redistributed here.

The public branch history was rebuilt after this boundary was fixed, so the old
large blobs are not ancestors of the release commit. A local recovery bundle of
the former history must not be pushed to the public remote.

The repository does not yet declare a source-code licence. One must be selected
explicitly before others can safely reuse the code.

## Generative-AI disclosure

The report contains a dedicated disclosure describing the use of OpenAI Codex,
Google Scholar's AI-assisted discovery features, the local Gemma candidate
filter, and the external Oolel resource. The author performed the annotations,
ran the experiments, verified the outputs, made the methodological decisions,
and remains responsible for the conclusions.
