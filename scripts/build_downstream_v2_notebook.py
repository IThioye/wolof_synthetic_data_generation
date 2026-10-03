"""Generate the no-training controller notebook for downstream analysis V2."""

from __future__ import annotations

import json
from pathlib import Path


def cell(kind: str, source: str) -> dict:
    payload = {"cell_type": kind, "metadata": {}, "source": source.strip().splitlines(True)}
    if kind == "code":
        payload.update(execution_count=None, outputs=[])
    return payload


CELLS = [
    cell("markdown", """
# Wolof downstream analysis V2 — frozen Laya and TNT

This notebook controls four no-training experiments: clean preservation, M1
severity, cross-method generalization, and detailed V1 error analysis. It never
writes to the frozen V1 result directory and validates model/data hashes before
inference.
"""),
    cell("code", """
from pathlib import Path
import json, subprocess, sys

def find_root():
    for candidate in (Path.cwd(), *Path.cwd().parents):
        if (candidate / "src/downstream/v2_runner.py").is_file():
            return candidate
    raise RuntimeError("Open the notebook inside the repository checkout")

REPO_ROOT = find_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
OUTPUT_DIR = REPO_ROOT / "results/downstream/laya_normalization_v2"
SEED = 2026
RUN_ALL_V2 = False
print(REPO_ROOT)
"""),
    cell("markdown", """
## Frozen-protocol verification

REAL must reproduce V1 exactly. Laya, TNT and official Injongo-test hashes are
checked before any new condition is evaluated.
"""),
    cell("code", """
from src.downstream.v2_protocol import validate_frozen_artifacts
from src.downstream.noise_v2 import M1_SEVERITIES, method_applicability
import pandas as pd

display(validate_frozen_artifacts(REPO_ROOT))
display(pd.DataFrame([severity.__dict__ for severity in M1_SEVERITIES]))
display(pd.DataFrame(method_applicability()))
"""),
    cell("markdown", """
## Execute all four analyses

No fine-tuning occurs. Output is streamed. The runner is deterministic and may
be rerun; it overwrites V2 files only.
"""),
    cell("code", """
command = [
    sys.executable, "-u", "-m", "src.downstream.v2_runner",
    "--repository-root", str(REPO_ROOT),
    "--output-dir", str(OUTPUT_DIR),
    "--seed", str(SEED),
]
print(" ".join(command))
if RUN_ALL_V2:
    process = subprocess.Popen(command, cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in process.stdout:
        print(line, end="", flush=True)
    if process.wait():
        raise RuntimeError("V2 failed; see the traceback immediately above")
else:
    print("Execution disabled; set RUN_ALL_V2=True to rerun.")
"""),
    cell("markdown", """
## Results

The tables below are read-only. `changed_subset_*` is especially important for
M5 because its lexical mappings cover only a small fraction of Injongo.
"""),
    cell("code", """
if (OUTPUT_DIR / "global_summary.json").is_file():
    summary = json.loads((OUTPUT_DIR / "global_summary.json").read_text(encoding="utf-8"))
    display(summary["clean_preservation"])
    display(pd.read_csv(OUTPUT_DIR / "severity/severity_curve.csv"))
    display(pd.read_csv(OUTPUT_DIR / "cross_method/cross_method_summary.csv"))
    display(pd.read_csv(OUTPUT_DIR / "error_analysis/restoration_by_category.csv"))
    display(summary["error_analysis"])
else:
    print("No V2 results yet.")
"""),
    cell("code", """
from IPython.display import Image, display as show
figure = OUTPUT_DIR / "figures/severity_curves.png"
if figure.is_file():
    show(Image(filename=str(figure)))
"""),
]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "notebooks/downstream/laya_downstream_analysis_v2.ipynb"
    output.parent.mkdir(parents=True, exist_ok=True)
    notebook = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
