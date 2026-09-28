"""M7 benchmark using a shared TNT-inspired edit evaluator.

This preserves the previous selection route requested by the user: synthetic
pretraining is monitored on Gold-dev, then every condition is fine-tuned on
Gold-train and selected on Gold-dev.  Gold-test remains frozen.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Mapping

import pandas as pd

from src.config import M7_RESULTS_DIR, M7_RUNS_DIR
from src.modeling import compact_benchmark_runner as _base
from src.modeling.tnt_edit_transformer import (
    TntEditArchitecture,
    prepare_tnt_edit_transformer,
)


BENCHMARK_NAME = "tnt_edit_gold_dev_v4"
METHODS = _base.METHODS
CONDITIONS = _base.CONDITIONS
RUNS_DIR = M7_RUNS_DIR / BENCHMARK_NAME
RESULTS_DIR = M7_RESULTS_DIR / BENCHMARK_NAME
SELECTIONS_DIR = RESULTS_DIR / "selections"
TEST_DIR = RESULTS_DIR / "frozen_test"
CompactBenchmarkProtocol = _base.CompactBenchmarkProtocol


def configure_benchmark(name: str) -> None:
    if not name or any(
        character not in "abcdefghijklmnopqrstuvwxyz0123456789_-"
        for character in name
    ):
        raise ValueError("Benchmark name must use lowercase letters, digits, '_' or '-'")
    global BENCHMARK_NAME, RUNS_DIR, RESULTS_DIR, SELECTIONS_DIR, TEST_DIR
    BENCHMARK_NAME = name
    RUNS_DIR = M7_RUNS_DIR / BENCHMARK_NAME
    RESULTS_DIR = M7_RESULTS_DIR / BENCHMARK_NAME
    SELECTIONS_DIR = RESULTS_DIR / "selections"
    TEST_DIR = RESULTS_DIR / "frozen_test"
    _activate_namespace()


def _activate_namespace() -> None:
    _base.BENCHMARK_NAME = BENCHMARK_NAME
    _base.RUNS_DIR = RUNS_DIR
    _base.RESULTS_DIR = RESULTS_DIR
    _base.SELECTIONS_DIR = SELECTIONS_DIR
    _base.TEST_DIR = TEST_DIR


def prepare_benchmark_data(protocol: CompactBenchmarkProtocol):
    _activate_namespace()
    return _base.prepare_benchmark_data(protocol)


def prepare_shared_initialization(
    gold: Mapping[str, pd.DataFrame],
    selected: Mapping[str, pd.DataFrame],
    protocol: CompactBenchmarkProtocol,
    *,
    output_slots_per_source: int = 32,
) -> Path:
    """Create one identical encoder/edit-head initialization for all methods."""
    _activate_namespace()
    architecture = TntEditArchitecture(
        d_model=192,
        attention_heads=4,
        encoder_layers=3,
        feed_forward_size=768,
        dropout=0.1,
        max_position_embeddings=256,
        output_slots_per_source=output_slots_per_source,
        operation_loss_weight=1.0,
        character_loss_weight=1.0,
        label_smoothing=0.1,
    )
    architecture_path = RESULTS_DIR / "architecture_protocol.json"
    payload = asdict(architecture)
    if architecture_path.is_file():
        previous = json.loads(architecture_path.read_text(encoding="utf-8"))
        if previous.get("architecture") != payload:
            raise RuntimeError(
                "This TNT benchmark namespace already contains another architecture"
            )
    else:
        _base.write_json(
            architecture_path,
            {
                "benchmark": BENCHMARK_NAME,
                "architecture": payload,
                "architecture_family": "TNT-inspired encoder edit prediction",
                "synthetic_selection_split": "gold_dev",
                "transfer_selection_split": "gold_dev",
                "final_evaluation_split": "frozen_gold_test",
                "note": (
                    "Synthetic-stage Gold-dev metrics are monitoring/selection only; "
                    "final comparisons use the frozen Gold-test."
                ),
            },
        )
    frames = [gold["train"]] + [selected[method] for method in METHODS]
    return prepare_tnt_edit_transformer(
        frames,
        output_root=RUNS_DIR / "shared_initialization",
        seed=protocol.seed,
        architecture=architecture,
    )


def run_gold_control(*args, **kwargs):
    _activate_namespace()
    return _base.run_gold_control(*args, **kwargs)


def run_synthetic_method(*args, **kwargs):
    _activate_namespace()
    return _base.run_synthetic_method(*args, **kwargs)


def load_selection(*args, **kwargs):
    _activate_namespace()
    return _base.load_selection(*args, **kwargs)


def missing_selections(*args, **kwargs):
    _activate_namespace()
    return _base.missing_selections(*args, **kwargs)


def development_summary(*args, **kwargs):
    _activate_namespace()
    return _base.development_summary(*args, **kwargs)


def evaluate_frozen_test(*args, **kwargs):
    _activate_namespace()
    return _base.evaluate_frozen_test(*args, **kwargs)


def plot_summary(*args, **kwargs):
    _activate_namespace()
    return _base.plot_summary(*args, **kwargs)
