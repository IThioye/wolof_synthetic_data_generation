"""M1 pretraining gate for the diagnostic-driven TNT hybrid V3 model.

V3 is isolated from both the frozen TNT benchmark and the completed V2 gate.
It uses the same M1 sample and optimization budget, selects checkpoints on
Gold-dev, and deliberately leaves Gold-test untouched.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Mapping

import pandas as pd

from src.config import M7_RESULTS_DIR, M7_RUNS_DIR
from src.modeling import compact_benchmark_runner as _base
from src.modeling.metrics import normalization_metrics
from src.modeling.tnt_edit_transformer import OPERATION_NAMES
from src.modeling.tnt_hybrid_experiment import (
    HybridExperimentProtocol,
    _inverse_sqrt_weights,
)
from src.modeling.tnt_hybrid_transformer import BOUNDARY_ACTIONS, WORD_ACTIONS
from src.modeling.tnt_hybrid_v3_transformer import (
    TntHybridV3Architecture,
    hybrid_v3_annotations,
    prepare_tnt_hybrid_v3_transformer,
)


BENCHMARK_NAME = "tnt_hybrid_v3_m1_gate_v1"
RUNS_DIR = M7_RUNS_DIR / BENCHMARK_NAME
RESULTS_DIR = M7_RESULTS_DIR / BENCHMARK_NAME
SELECTIONS_DIR = RESULTS_DIR / "selections"
TEST_DIR = RESULTS_DIR / "frozen_test_disabled"
TNT_V1_RESULTS_DIR = M7_RESULTS_DIR / "tnt_edit_gold_dev_v4"
TNT_V2_RESULTS_DIR = M7_RESULTS_DIR / "tnt_hybrid_m1_gate_v2"
METHOD = "m1"

CompactBenchmarkProtocol = _base.CompactBenchmarkProtocol


def _activate_namespace() -> None:
    _base.BENCHMARK_NAME = BENCHMARK_NAME
    _base.RUNS_DIR = RUNS_DIR
    _base.RESULTS_DIR = RESULTS_DIR
    _base.SELECTIONS_DIR = SELECTIONS_DIR
    _base.TEST_DIR = TEST_DIR


def gate_protocol() -> CompactBenchmarkProtocol:
    """Reuse the exact optimization budget of the frozen TNT M1 run."""
    return CompactBenchmarkProtocol(
        seed=2026,
        matched_unique_rows=3330,
        synthetic_epochs=20.0,
        gold_epochs=20.0,
        synthetic_learning_rate=3e-4,
        gold_learning_rate=1e-4,
        synthetic_batch_size=64,
        gold_batch_size=32,
        evaluation_batch_size=64,
        generated_eval_interval_epochs=5,
    )


def prepare_gate_data(protocol: CompactBenchmarkProtocol):
    _activate_namespace()
    return _base.prepare_benchmark_data(protocol)


def _v3_label_profile(
    gold_train: pd.DataFrame,
    protocol: CompactBenchmarkProtocol,
) -> dict[str, object]:
    """Freeze V3 class weights from Gold-train before synthetic training."""
    profile_protocol = HybridExperimentProtocol(
        seed=protocol.seed,
        output_slots_per_source=16,
        maximum_class_weight=4.0,
        balance_edit_classes=True,
    )
    operation_counts: Counter = Counter()
    boundary_counts: Counter = Counter()
    word_counts: Counter = Counter()
    length_counts: Counter = Counter()
    for source_text, target_text in zip(
        gold_train["source_text"], gold_train["target_text"]
    ):
        source_ids = [ord(character) + 4 for character in source_text] + [2]
        target_ids = [ord(character) + 4 for character in target_text] + [2]
        annotations = hybrid_v3_annotations(
            source_ids,
            target_ids,
            eos_token_id=2,
            pad_token_id=0,
            space_token_id=ord(" ") + 4,
            output_slots_per_source=profile_protocol.output_slots_per_source,
        )
        operation_counts.update(value for value in annotations[0] if value >= 0)
        boundary_counts.update(value for value in annotations[2] if value >= 0)
        word_counts.update(value for value in annotations[4] if value >= 0)
        length_counts.update(value for value in annotations[5] if value >= 0)

    maximum = profile_protocol.maximum_class_weight
    slots = profile_protocol.output_slots_per_source
    return {
        "operation_counts": {
            name: operation_counts.get(index, 0)
            for index, name in enumerate(OPERATION_NAMES)
        },
        "boundary_counts": {
            name: boundary_counts.get(index, 0)
            for index, name in enumerate(BOUNDARY_ACTIONS)
        },
        "word_counts": {
            name: word_counts.get(index, 0)
            for index, name in enumerate(WORD_ACTIONS)
        },
        "segment_length_counts": {
            str(index): length_counts.get(index, 0)
            for index in range(slots + 1)
        },
        "operation_class_weights": _inverse_sqrt_weights(
            operation_counts, len(OPERATION_NAMES), maximum
        ),
        "boundary_class_weights": _inverse_sqrt_weights(
            boundary_counts, len(BOUNDARY_ACTIONS), maximum
        ),
        "word_class_weights": _inverse_sqrt_weights(
            word_counts, len(WORD_ACTIONS), maximum
        ),
        "segment_length_class_weights": _inverse_sqrt_weights(
            length_counts, slots + 1, maximum
        ),
        "computed_from": "complete Gold-train labels only",
        "length_supervision": "substitution and expansion positions only",
        "maximum_class_weight": maximum,
    }


def _v3_architecture(
    gold_train: pd.DataFrame,
    protocol: CompactBenchmarkProtocol,
) -> tuple[TntHybridV3Architecture, dict[str, object]]:
    profile = _v3_label_profile(gold_train, protocol)
    architecture = TntHybridV3Architecture(
        d_model=192,
        attention_heads=4,
        encoder_layers=3,
        feed_forward_size=768,
        dropout=0.1,
        max_position_embeddings=256,
        output_slots_per_source=16,
        operation_loss_weight=1.0,
        character_loss_weight=1.0,
        segment_length_loss_weight=1.0,
        boundary_loss_weight=1.0,
        word_loss_weight=0.5,
        operation_class_weights=tuple(profile["operation_class_weights"]),
        segment_length_class_weights=tuple(
            profile["segment_length_class_weights"]
        ),
        boundary_class_weights=tuple(profile["boundary_class_weights"]),
        word_class_weights=tuple(profile["word_class_weights"]),
        boundary_focal_gamma=2.0,
        label_smoothing=0.1,
        use_boundary_inference=True,
        use_word_keep_gate=True,
        use_hard_lexical_protection=True,
        boundary_confidence_threshold=0.50,
        word_keep_threshold=0.70,
    )
    return architecture, profile


def prepare_gate_initialization(
    gold: Mapping[str, pd.DataFrame],
    selected: Mapping[str, pd.DataFrame],
    protocol: CompactBenchmarkProtocol,
) -> Path:
    """Create one deterministic V3 initialization for the M1 gate."""
    _activate_namespace()
    architecture, profile = _v3_architecture(gold["train"], protocol)
    payload = {
        "benchmark": BENCHMARK_NAME,
        "purpose": "M1-only V3 optimization gate",
        "architecture": asdict(architecture),
        "training_protocol": asdict(protocol),
        "edit_label_profile": profile,
        "synthetic_method": METHOD,
        "selection_split": "Gold-dev, matching TNT V1 and V2",
        "test_policy": "Gold-test is disabled",
        "comparisons": ["tnt_edit_gold_dev_v4", "tnt_hybrid_m1_gate_v2"],
    }
    protocol_path = RESULTS_DIR / "architecture_protocol.json"
    canonical = json.loads(json.dumps(payload, ensure_ascii=False, default=str))
    if protocol_path.is_file():
        previous = json.loads(protocol_path.read_text(encoding="utf-8"))
        for field in ("architecture", "training_protocol"):
            if previous.get(field) != canonical[field]:
                raise RuntimeError(
                    f"Existing {BENCHMARK_NAME} {field} differs from this run"
                )
    _base.write_json(protocol_path, payload)

    frames = [gold["train"]] + [selected[method] for method in _base.METHODS]
    return prepare_tnt_hybrid_v3_transformer(
        frames,
        output_root=RUNS_DIR / "shared_initialization",
        seed=protocol.seed,
        architecture=architecture,
    )


def run_m1_gate(
    gold: Mapping[str, pd.DataFrame],
    selected: Mapping[str, pd.DataFrame],
    shared_initialization: Path,
    protocol: CompactBenchmarkProtocol,
    *,
    rerun: bool = False,
) -> dict[str, object]:
    """Pretrain on M1, then continuously fine-tune on Gold-train."""
    _activate_namespace()
    return _base.run_synthetic_method(
        METHOD,
        selected[METHOD],
        gold,
        shared_initialization,
        protocol,
        rerun=rerun,
    )


def _selection_predictions(
    selection_path: Path,
) -> tuple[dict[str, object], pd.DataFrame]:
    if not selection_path.is_file():
        raise FileNotFoundError(selection_path)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    manifest_path = Path(str(selection["selected_run_manifest"]))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    predictions = pd.read_csv(
        Path(str(manifest["dev_predictions_path"])), keep_default_na=False
    )
    return selection, predictions


def _raw_reference_metrics(
    predictions: pd.DataFrame, gold_dev: pd.DataFrame
) -> dict[str, float | int]:
    aligned = gold_dev[["source_id", "source_text", "target_text"]].merge(
        predictions[["source_id", "prediction"]],
        on="source_id",
        how="left",
        validate="one_to_one",
    )
    if aligned["prediction"].isna().any():
        raise ValueError("Predictions do not cover every Gold-dev source_id")
    return normalization_metrics(
        aligned["source_text"].tolist(),
        aligned["prediction"].tolist(),
        aligned["target_text"].tolist(),
    )


def gate_summary(gold_dev: pd.DataFrame) -> pd.DataFrame:
    """Compare identity and the matched V1, V2, and V3 M1 experiments."""
    rows: list[dict[str, object]] = [
        {
            "condition": "identity",
            **normalization_metrics(
                gold_dev["source_text"].tolist(),
                gold_dev["source_text"].tolist(),
                gold_dev["target_text"].tolist(),
            ),
        }
    ]
    comparisons = (
        ("tnt_v1_m1", TNT_V1_RESULTS_DIR / "selections" / "m1.json"),
        ("tnt_hybrid_v2_m1", TNT_V2_RESULTS_DIR / "selections" / "m1.json"),
        ("tnt_hybrid_v3_m1", SELECTIONS_DIR / "m1.json"),
    )
    for condition, path in comparisons:
        if path.is_file():
            _, predictions = _selection_predictions(path)
            rows.append(
                {
                    "condition": condition,
                    **_raw_reference_metrics(predictions, gold_dev),
                }
            )
    summary = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(RESULTS_DIR / "m1_gate_summary.csv", index=False, encoding="utf-8")
    return summary


def gate_decision(summary: pd.DataFrame) -> dict[str, object]:
    """Report whether V3 improves V2 without consulting Gold-test."""
    indexed = summary.set_index("condition")
    required = {"tnt_hybrid_v2_m1", "tnt_hybrid_v3_m1"}
    if not required.issubset(indexed.index):
        return {"ready": False, "reason": "V2 or V3 M1 predictions are missing"}
    previous = indexed.loc["tnt_hybrid_v2_m1"]
    current = indexed.loc["tnt_hybrid_v3_m1"]
    criteria = {
        "cer_improved_over_v2": bool(current["cer"] < previous["cer"]),
        "wer_not_worse_than_v2": bool(current["wer"] <= previous["wer"]),
        "correction_f1_improved_over_v2": bool(
            current["correction_f1"] > previous["correction_f1"]
        ),
        "overcorrection_reduced_from_v2": bool(
            current["overcorrection_rate"] < previous["overcorrection_rate"]
        ),
        "at_least_one_true_correction": bool(current["correction_tp"] > 0),
    }
    decision = {
        "ready": True,
        "v3_improves_v2_on_all_predeclared_criteria": all(criteria.values()),
        "criteria": criteria,
        "selection_split": "Gold-dev",
        "gold_test_used": False,
    }
    _base.write_json(RESULTS_DIR / "m1_gate_decision.json", decision)
    return decision


def gate_prediction_comparison(gold_dev: pd.DataFrame) -> pd.DataFrame:
    comparison = gold_dev[["source_id", "source_text", "target_text"]].rename(
        columns={"target_text": "reference"}
    )
    paths = (
        ("v1", TNT_V1_RESULTS_DIR / "selections" / "m1.json"),
        ("v2", TNT_V2_RESULTS_DIR / "selections" / "m1.json"),
        ("v3", SELECTIONS_DIR / "m1.json"),
    )
    for label, path in paths:
        if not path.is_file():
            continue
        _, predictions = _selection_predictions(path)
        comparison = comparison.merge(
            predictions[["source_id", "prediction"]].rename(
                columns={"prediction": f"prediction_{label}"}
            ),
            on="source_id",
            validate="one_to_one",
        )
    return comparison
