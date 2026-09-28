# Rebuilding the M2 Eflomal lexicon

The final M2 generator does not require an Eflomal checkout. It consumes the
small frozen alignment lexicon at `artifacts/lexicon.pkl`:

- Eflomal upstream: `https://github.com/robertostling/eflomal`
- pinned commit: `1fe2a43e3667fb2461736ddb48783ab0cf98a171`
- lexicon SHA-256:
  `2ab93bb9e3a55fe1e7dca6b431ab85e094d66cd46c41c7b81600631492ee7ea1`
- number of Wolof source entries: 1,989

The pickle is a trusted project artifact. Do not load a replacement pickle from
an untrusted source, because pickle deserialization can execute code.

## Optional rebuild

Eflomal needs a Unix-like build environment. In Linux or WSL:

```bash
git clone https://github.com/robertostling/eflomal.git
cd eflomal
git checkout 1fe2a43e3667fb2461736ddb48783ab0cf98a171
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
```

From the project repository, first restore the pinned formal parallel corpus:

```bash
python scripts/download_external_data.py formal_parallel
```

Then run `notebooks/m2_eflomal.ipynb`. Its alignment cells prepare the parallel
text, run Eflomal in batches, retain translations meeting the documented count
and probability thresholds, and write `artifacts/lexicon.pkl`. Verify the output
checksum before replacing the frozen artifact. Small floating-point or upstream
build differences should be treated as a new artifact version, not silently
overwritten.

Eflomal is GPL-3.0. It is used as an external tool; its source code is not
copied into this repository.
