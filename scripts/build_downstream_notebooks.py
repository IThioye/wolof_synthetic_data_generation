"""Generate the two lightweight downstream notebooks from tested modules."""

from __future__ import annotations

import json
from pathlib import Path


def markdown(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.strip().splitlines(keepends=True),
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip().splitlines(keepends=True),
    }


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


FINE_TUNING_CELLS = [
    markdown(
        """
# Laya multilingual × InjongoIntent Wolof — fine-tuning V1

This notebook adapts Laya's official typed-decisions fine-tuning workflow to
the **official InjongoIntent Wolof train/dev splits**. It trains on REAL train
only, selects and calibrates on dev, and never loads test. The official test is
reserved for the separate frozen benchmark notebook.

Protocol: `train.jsonl → optimization`, `dev.jsonl → selection/calibration`,
`test.jsonl → not used here`.
"""
    ),
    code(
        """
from pathlib import Path
import json, os, sys, subprocess

def find_repository_root():
    for candidate in (Path.cwd(), *Path.cwd().parents):
        if (candidate / "src/downstream").is_dir():
            return candidate
    raise RuntimeError("Open this notebook from inside the repository checkout")

REPO_ROOT = find_repository_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MODEL_NAME = "convaiinnovations/laya-multilingual"
MODEL_REVISION = "e4e9ddf21a7b1903b7acffd8814ad4307bf63a67"
DATA_DIR = REPO_ROOT / "data/external/injongo_intent_wol"
TRAIN_PATH = DATA_DIR / "train.jsonl"
DEV_PATH = DATA_DIR / "dev.jsonl"
TEST_PATH = DATA_DIR / "test.jsonl"  # frozen; intentionally not loaded here
BASE_MODEL_DIR = REPO_ROOT / "artifacts/downstream/laya_multilingual_base"
PREPARED_DIR = REPO_ROOT / "artifacts/downstream/laya_injongo_wolof_v1"
OUTPUT_DIR = REPO_ROOT / "checkpoints/downstream/laya_normalization_v1"
SEED = 2026

INSTALL_DEPENDENCIES = False
RUN_CPU_DRY_RUN = True
RUN_FULL_TRAINING = False  # change to True only when ready for the GPU run
NPROC_PER_NODE = 1         # set 2 on a 2-GPU Kaggle runtime

print("Repository:", REPO_ROOT)
print("Output:", OUTPUT_DIR)
"""
    ),
    markdown(
        """
## Optional dependency installation

The repository requirements are installed first, followed by the pinned Laya
API used by this adaptation. Leave the switch disabled when the environment is
already prepared.
"""
    ),
    code(
        """
if INSTALL_DEPENDENCIES:
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(REPO_ROOT / "requirements.txt")], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(REPO_ROOT / "requirements-downstream.txt")], check=True)
else:
    print("Dependency installation skipped.")
"""
    ),
    markdown(
        """
## Download and validate the official files

All three files are downloaded from a pinned Masakhane repository revision.
Only train and dev are parsed below. No new split or random partition is made.
"""
    ),
    code(
        """
from src.downstream.injongo import download_official_splits, load_split, INJONGO_INTENTS

download_official_splits(DATA_DIR)
train_rows = load_split(TRAIN_PATH, "train")
dev_rows = load_split(DEV_PATH, "dev")

assert len(train_rows) == 2240 and len(dev_rows) == 320
assert set(row["intent"] for row in train_rows) == set(INJONGO_INTENTS)
assert set(row["row_id"] for row in train_rows).isdisjoint(row["row_id"] for row in dev_rows)
print({"train": len(train_rows), "dev": len(dev_rows), "intents": len(INJONGO_INTENTS)})
print("Official test exists but remains unopened:", TEST_PATH.is_file())
"""
    ),
    code(
        """
import pandas as pd

intent_counts = pd.Series([row["intent"] for row in train_rows]).value_counts().sort_index()
display(intent_counts.rename("train_rows").to_frame())
"""
    ),
    markdown(
        """
## Fetch the pinned Laya multilingual checkpoint

This preserves Laya's standalone layout (`encoder/`, `tokenizer/`,
`rl_agent_config.json`, `model.safetensors`) and applies the tokenizer-config
compatibility helper used by the official notebook.
"""
    ),
    code(
        """
from src.downstream.laya_model import snapshot_laya_model, load_laya_tokenizer_config

model_dir = snapshot_laya_model(MODEL_NAME, revision=MODEL_REVISION, local_dir=BASE_MODEL_DIR)
tokenizer, laya_config = load_laya_tokenizer_config(model_dir)
laya_config["max_len"] = 1024
laya_config["head_max_len"] = 256
print("Model directory:", model_dir)
print({key: laya_config.get(key) for key in ("max_len", "head_max_len", "encoder_model")})
"""
    ),
    markdown(
        """
## Inspect the exact typed-choice conversion

The state is the Wolof utterance. A single `choice` question contains the full
40-intent inventory. Empty option descriptions keep the Laya head compact;
the intent labels themselves are still rendered. The conversion fails if even
one option marker is lost or two options collapse under truncation.
"""
    ),
    code(
        """
from src.downstream.injongo import make_laya_question, make_laya_state, build_laya_item

for row in train_rows[:3]:
    item = build_laya_item(row, tokenizer, laya_config)
    print("TEXT:", row["text"])
    print("INTENT:", row["intent"])
    print("LAYA STATE:", make_laya_state(row["text"]))
    print("LAYA QUESTION TYPE:", make_laya_question()["t"])
    print("OPTIONS:", list(make_laya_question()["crit"]))
    print("GOLD:", row["intent"])
    print("ENCODED:", {"tokens": len(item["ids"]), **item["option_stats"]})
    print("-" * 80)
"""
    ),
    markdown(
        """
## Prepare train/dev tensors

The official test file is not read by this step. The manifest records this
explicitly. All 2,240 train examples are retained; unlike Laya's source
notebook, no extra calibration slice is carved out of train because the
official dev split already has that role.
"""
    ),
    code(
        """
import torch
from src.downstream.laya_model import prepare_laya_items

PREPARED_DIR.mkdir(parents=True, exist_ok=True)
train_items = prepare_laya_items(train_rows, tokenizer, laya_config)
dev_items = prepare_laya_items(dev_rows, tokenizer, laya_config)
TRAIN_ITEMS_PATH = PREPARED_DIR / "train_items.pt"
DEV_ITEMS_PATH = PREPARED_DIR / "dev_items.pt"
torch.save(train_items, TRAIN_ITEMS_PATH)
torch.save(dev_items, DEV_ITEMS_PATH)

preparation_manifest = {
    "train_rows": len(train_items), "dev_rows": len(dev_items),
    "test_loaded": False, "intent_count": len(INJONGO_INTENTS),
    "seed": SEED, "model": MODEL_NAME, "model_revision": MODEL_REVISION,
}
(PREPARED_DIR / "preparation_manifest.json").write_text(json.dumps(preparation_manifest, indent=2) + "\\n")
print(preparation_manifest)
"""
    ),
    markdown(
        """
## Mini forward pass (no training)

This loads the complete base checkpoint and checks one CPU forward pass. It can
take roughly a minute on CPU because the encoder has about 322M parameters.
The returned prediction is not meaningful before task fine-tuning.
"""
    ),
    code(
        """
from src.downstream.laya_model import mini_forward_dry_run

if RUN_CPU_DRY_RUN:
    dry_run = mini_forward_dry_run(model_dir, train_rows[0], device="cpu")
    print(json.dumps(dry_run, ensure_ascii=False, indent=2))
else:
    print("CPU dry-run skipped.")
"""
    ),
    markdown(
        """
## GPU fine-tuning — run this yourself

The command reuses the official RLCD + cross-entropy objective, separate
encoder/head learning rates, exploration schedule, mixed precision, gradient
checkpointing, rolling checkpoints and post-training temperature fitting.

Adaptations: REAL Injongo train only; dev evaluation after each epoch; best
checkpoint selected by dev macro-F1; choice temperature fitted on dev; no test
loading. On Windows/local single-GPU, keep `NPROC_PER_NODE=1`. On Kaggle 2×T4,
set it to 2. Training output is streamed directly in the notebook.
"""
    ),
    code(
        """
training_arguments = [
    "--model-dir", str(model_dir),
    "--train-items", str(TRAIN_ITEMS_PATH),
    "--dev-items", str(DEV_ITEMS_PATH),
    "--output-dir", str(OUTPUT_DIR),
    "--epochs", "4", "--micro-batch", "8", "--eval-batch", "16",
    "--gradient-accumulation", "4", "--seed", str(SEED), "--precision", "fp16",
]
if NPROC_PER_NODE == 1:
    # Avoid torch.distributed.run on single-GPU Windows/Jupyter. It adds no
    # training benefit here and may hide the child process's actual stderr.
    train_command = [
        sys.executable, "-u", "-m", "src.downstream.laya_finetune",
        *training_arguments,
    ]
else:
    train_command = [
        sys.executable, "-u", "-m", "torch.distributed.run", "--standalone",
        f"--nproc_per_node={NPROC_PER_NODE}",
        "-m", "src.downstream.laya_finetune", *training_arguments,
    ]
print(" ".join(train_command))
if RUN_FULL_TRAINING:
    process = subprocess.Popen(
        train_command,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    for output_line in process.stdout:
        print(output_line, end="", flush=True)
    return_code = process.wait()
    if return_code:
        raise RuntimeError(
            f"Laya training process failed with exit code {return_code}. "
            "The underlying traceback is printed immediately above."
        )
else:
    print("Training is disabled. Set RUN_FULL_TRAINING=True when ready.")
"""
    ),
    markdown(
        """
## Verify the selected checkpoint

Run after training. The final directory contains the best dev-macro-F1 weights,
dev-fitted choice temperature, tokenizer/config, history and protocol manifest.
"""
    ),
    code(
        """
manifest_path = OUTPUT_DIR / "training_manifest.json"
if manifest_path.is_file():
    training_manifest = json.loads(manifest_path.read_text())
    assert training_manifest["test_used"] is False
    display(training_manifest)
else:
    print("No trained checkpoint yet. Run the GPU cell above first.")
"""
    ),
]


BENCHMARK_CELLS = [
    markdown(
        """
# Frozen Laya REAL / NOISY / NORMALIZED benchmark

This notebook performs the one-time official test evaluation after the Laya
checkpoint is ready. The **same frozen classifier** is used for all conditions:

`REAL = official test`, `NOISY = paired M1 synthetic corruption`,
`NORMALIZED = exactly NOISY passed through the selected TNT Hybrid V3 model`.

The execution switch is disabled by default to keep official test frozen until
all paths and manifests have been reviewed.
"""
    ),
    code(
        """
from pathlib import Path
import json, sys

def find_repository_root():
    for candidate in (Path.cwd(), *Path.cwd().parents):
        if (candidate / "src/downstream").is_dir():
            return candidate
    raise RuntimeError("Open this notebook from inside the repository checkout")

REPO_ROOT = find_repository_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DATA_DIR = REPO_ROOT / "data/external/injongo_intent_wol"
TEST_PATH = DATA_DIR / "test.jsonl"
LAYA_CHECKPOINT = REPO_ROOT / "checkpoints/downstream/laya_normalization_v1"
OUTPUT_DIR = REPO_ROOT / "results/downstream/laya_normalization_v1"
SEED = 2026
RUN_FROZEN_TEST = False  # review readiness below, then change once to True
DEVICE = None            # None selects CUDA when available
print("Laya checkpoint:", LAYA_CHECKPOINT)
print("Results:", OUTPUT_DIR)
"""
    ),
    markdown(
        """
## Readiness and locked TNT selection

The normalizer is not retrained. The repository's locked V3 selection is the
M1-pretrained checkpoint adapted on Gold train with learning rate 5e-5 and
`word_keep_threshold=0.50`, selected on Gold-dev WER. Hashes are checked before
inference. Its local checkpoint must be present because large weights are not
stored in Git.
"""
    ),
    code(
        """
from src.downstream.tnt_normalizer import resolve_selected_tnt_checkpoint

assert LAYA_CHECKPOINT.is_dir(), "Run the fine-tuning notebook first."
training_manifest = json.loads((LAYA_CHECKPOINT / "training_manifest.json").read_text())
assert training_manifest["test_used"] is False
tnt_checkpoint, tnt_selection = resolve_selected_tnt_checkpoint(REPO_ROOT)
display({
    "laya_ready": True,
    "laya_best_epoch": training_manifest["best_epoch"],
    "tnt_checkpoint": str(tnt_checkpoint),
    "tnt_candidate": tnt_selection["candidate"],
    "tnt_selection_split": tnt_selection["selection_split"],
    "tnt_dev_metrics": tnt_selection["development_metrics"],
})
"""
    ),
    markdown(
        """
## One-time official test benchmark

Only this cell loads `test.jsonl`. It creates paired texts, normalizes NOISY in
batches, evaluates the same calibrated Laya checkpoint on all three conditions,
then writes metrics, per-class reports, confusion matrices, predictions and the
four-way paired analysis. No model is updated.
"""
    ),
    code(
        """
from src.downstream.injongo import load_split
from src.downstream.benchmark import run_frozen_benchmark

if RUN_FROZEN_TEST:
    test_rows = load_split(TEST_PATH, "test")
    benchmark = run_frozen_benchmark(
        test_rows,
        laya_checkpoint=LAYA_CHECKPOINT,
        tnt_checkpoint=tnt_checkpoint,
        output_dir=OUTPUT_DIR,
        seed=SEED,
        device=DEVICE,
    )
    display(benchmark["metrics"])
else:
    print("Frozen test remains unopened. Set RUN_FROZEN_TEST=True when the pipeline is final.")
"""
    ),
    markdown(
        """
## Inspect saved results

This section is read-only and becomes available after the benchmark cell.
`recovery_rate` is emitted only when REAL outperforms NOISY, so a zero/negative
noise drop is never hidden by an unstable denominator.
"""
    ),
    code(
        """
import pandas as pd

metrics_path = OUTPUT_DIR / "metrics.json"
if metrics_path.is_file():
    metrics = json.loads(metrics_path.read_text())
    display(pd.DataFrame({condition: values for condition, values in metrics.items() if condition in {"REAL", "NOISY", "NORMALIZED"}}).T)
    display(metrics["deltas"])
    display(metrics["paired"])
    display(pd.read_csv(OUTPUT_DIR / "paired_analysis.csv").head(20))
else:
    print("No benchmark results yet.")
"""
    ),
]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output_dir = root / "notebooks/downstream"
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "laya_injongo_wolof_finetuning.ipynb": notebook(FINE_TUNING_CELLS),
        "laya_normalization_benchmark.ipynb": notebook(BENCHMARK_CELLS),
    }
    for filename, payload in files.items():
        path = output_dir / filename
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=1) + "\n",
            encoding="utf-8",
        )
        print(path)


if __name__ == "__main__":
    main()
