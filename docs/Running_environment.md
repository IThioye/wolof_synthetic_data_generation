# Running environment

## Supported local setup

The final workflow was executed with Python 3.11 on Windows and an NVIDIA RTX
5060 Laptop GPU. Most data preparation and analysis code also runs on CPU.

Create an isolated environment from the repository root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

Run the tests with:

```powershell
python -m pytest
```

## Eflomal and M2

The retained M2 condition consumes `artifacts/lexicon.pkl`, whose checksum and
provenance are documented in `docs/eflomal_reproducibility.md`. Rebuilding this
artifact with Eflomal is optional and requires Linux or WSL. Eflomal is pinned
by upstream commit in the documentation rather than embedded in this Git
repository. `src/alignment/ibm1_lexicon.py` remains a Windows-native statistical
alignment alternative; it is not the final M2 artifact reported in the
benchmark.

## Annotation applications

Sentence annotation:

```powershell
python -m src.annotation.flask_annotation_app
```

Open `http://127.0.0.1:5000`.

Gold-train linguistic annotation for M6:

```powershell
python -m src.annotation.flask_linguistic_annotation_app
```

Open `http://127.0.0.1:5001`. Do not run the legacy Streamlit and current Flask
interfaces simultaneously because they can target the same annotation CSVs.

## Final training notebook

The retained controller is:

```text
notebooks/m7_tnt_edit_gold_dev_benchmark.ipynb
```

It exposes separate cells per method, visible training progress, resumption, and
the guarded frozen-test operation. The exact final settings are recorded in:

- `results/m7_seq2seq/tnt_edit_gold_dev_v4/protocol.json`;
- `results/m7_seq2seq/tnt_edit_gold_dev_v4/architecture_protocol.json`;
- `results/m7_seq2seq/tnt_edit_gold_dev_v4/frozen_test_manifest.json`.

The earlier `m7_seq2seq_benchmark.ipynb` and compact/copy-aware notebooks are
architecture feasibility records, not the reproduction entry point.

## Paths and local storage

`src/config.py` resolves paths from the repository root and supports these
overrides:

- `PFE_PROJECT_ROOT`;
- `PFE_DATA_DIR`;
- `PFE_ARTIFACTS_DIR`.

Checkpoints, raw comments, external parquet files, papers, and local third-party
environments are excluded by `.gitignore`. See `data/README.md` and
`docs/repository_audit.md` for the publication boundary.
