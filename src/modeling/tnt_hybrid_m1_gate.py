"""M1 synthetic-pretraining gate for the full TNT hybrid architecture.

The experiment mirrors the frozen TNT V1 M1 protocol on Gold-dev, but never
opens Gold-test. Its purpose is to decide whether the hybrid is promising
enough to justify the four-way architecture ablation and full method rerun.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Mapping

import pandas as pd

from src.config import M7_RESULTS_DIR, M7_RUNS_DIR
from src.modeling import compact_benchmark_runner as _base
from src.modeling.metrics import normalization_metrics
from src.modeling.tnt_hybrid_experiment import (
    HybridExperimentProtocol,
    auxiliary_label_profile,
)
from src.modeling.tnt_hybrid_transformer import (
    TntHybridArchitecture,
    prepare_tnt_hybrid_transformer,
)


BENCHMARK_NAME = "tnt_hybrid_m1_gate_v2"
RUNS_DIR = M7_RUNS_DIR / BENCHMARK_NAME
RESULTS_DIR = M7_RESULTS_DIR / BENCHMARK_NAME
SELECTIONS_DIR = RESULTS_DIR / "selections"
TEST_DIR = RESULTS_DIR / "frozen_test_disabled"
PREVIOUS_TNT_RESULTS_DIR = M7_RESULTS_DIR / "tnt_edit_gold_dev_v4"
METHOD = "m1"

CompactBenchmarkProtocol = _base.CompactBenchmarkProtocol


def _activate_namespace() -> None:
    _base.BENCHMARK_NAME = BENCHMARK_NAME
    _base.RUNS_DIR = RUNS_DIR
    _base.RESULTS_DIR = RESULTS_DIR
    _base.SELECTIONS_DIR = SELECTIONS_DIR
    _base.TEST_DIR = TEST_DIR


def gate_protocol() -> CompactBenchmarkProtocol:
    """Return the exact optimization budget used by the frozen TNT V1 run."""
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
    """Recreate the original common-target selection under a new namespace."""
    _activate_namespace()
    return _base.prepare_benchmark_data(protocol)


def _full_hybrid_architecture(
    gold_train: pd.DataFrame,
    protocol: CompactBenchmarkProtocol,
) -> tuple[TntHybridArchitecture, dict[str, object]]:
    profile_protocol = HybridExperimentProtocol(
        seed=protocol.seed,
        output_slots_per_source=16,
        maximum_class_weight=4.0,
        balance_edit_classes=True,
    )
    profile = auxiliary_label_profile(gold_train, profile_protocol)
    profile["computed_from"] = "complete Gold-train labels only"
    architecture = TntHybridArchitecture(
        d_model=192,
        attention_heads=4,
        encoder_layers=3,
        feed_forward_size=768,
        dropout=0.1,
        max_position_embeddings=256,
        output_slots_per_source=16,
        operation_loss_weight=1.0,
        character_loss_weight=1.0,
        operation_class_weights=tuple(profile["operation_class_weights"]),
        boundary_loss_weight=0.5,
        word_loss_weight=0.5,
        boundary_class_weights=tuple(profile["boundary_class_weights"]),
        word_class_weights=tuple(profile["word_class_weights"]),
        label_smoothing=0.1,
        use_boundary_inference=True,
        use_word_keep_gate=True,
        boundary_confidence_threshold=0.60,
        word_keep_threshold=0.80,
    )
    return architecture, profile


def prepare_gate_initialization(
    gold: Mapping[str, pd.DataFrame],
    selected: Mapping[str, pd.DataFrame],
    protocol: CompactBenchmarkProtocol,
) -> Path:
    """Create the full hybrid with the same shared character inventory policy."""
    _activate_namespace()
    architecture, profile = _full_hybrid_architecture(gold["train"], protocol)
    protocol_path = RESULTS_DIR / "architecture_protocol.json"
    payload = {
        "benchmark": BENCHMARK_NAME,
        "purpose": "M1-only architecture gate before hybrid ablations",
        "architecture": asdict(architecture),
        "training_protocol": asdict(protocol),
        "edit_label_profile": profile,
        "class_weight_source": "complete Gold-train only; frozen before M1 training",
        "synthetic_method": METHOD,
        "selection_split": "Gold-dev, matching TNT V1",
        "test_policy": "Gold-test is disabled for this gate",
        "comparison_target": "tnt_edit_gold_dev_v4 M1",
    }
    if protocol_path.is_file():
        previous = json.loads(protocol_path.read_text(encoding="utf-8"))
        # JSON has no tuple type, so dataclass tuples (notably the class
        # weights) round-trip as lists. Compare canonical JSON-compatible
        # structures rather than treating that representation change as a new
        # architecture.
        canonical_payload = json.loads(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
        )
        for field in ("architecture", "training_protocol"):
            if previous.get(field) != canonical_payload[field]:
                raise RuntimeError(
                    f"Existing {BENCHMARK_NAME} {field} differs from this run"
                )
        _base.write_json(protocol_path, payload)
    else:
        _base.write_json(protocol_path, payload)

    # This mirrors TNT V1: one vocabulary/initialization policy shared across
    # all retained synthetic corpora, although only M1 is optimized in the gate.
    frames = [gold["train"]] + [selected[method] for method in _base.METHODS]
    return prepare_tnt_hybrid_transformer(
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
    """Pretrain on M1, fine-tune on Gold-train, and select on Gold-dev."""
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
    prediction_path = Path(str(manifest["dev_predictions_path"]))
    predictions = pd.read_csv(prediction_path, keep_default_na=False)
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
    """Compare identity, frozen TNT V1 M1, and the completed hybrid gate."""
    identity = normalization_metrics(
        gold_dev["source_text"].tolist(),
        gold_dev["source_text"].tolist(),
        gold_dev["target_text"].tolist(),
    )
    _, previous_predictions = _selection_predictions(
        PREVIOUS_TNT_RESULTS_DIR / "selections" / "m1.json"
    )
    previous = _raw_reference_metrics(previous_predictions, gold_dev)
    rows: list[dict[str, object]] = [
        {"condition": "identity", **identity},
        {"condition": "tnt_v1_m1", **previous},
    ]

    hybrid_selection_path = SELECTIONS_DIR / "m1.json"
    if hybrid_selection_path.is_file():
        _, hybrid_predictions = _selection_predictions(hybrid_selection_path)
        hybrid = _raw_reference_metrics(hybrid_predictions, gold_dev)
        rows.append({"condition": "tnt_hybrid_m1", **hybrid})

    summary = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(RESULTS_DIR / "m1_gate_summary.csv", index=False, encoding="utf-8")
    return summary


def gate_decision(summary: pd.DataFrame) -> dict[str, object]:
    """Apply predeclared, multi-metric criteria without opening Gold-test."""
    indexed = summary.set_index("condition")
    if "tnt_hybrid_m1" not in indexed.index:
        return {"ready": False, "reason": "Hybrid M1 gate has not completed"}
    previous = indexed.loc["tnt_v1_m1"]
    hybrid = indexed.loc["tnt_hybrid_m1"]
    criteria = {
        "cer_not_worse_than_v1": bool(hybrid["cer"] <= previous["cer"]),
        "wer_improved_over_v1": bool(hybrid["wer"] < previous["wer"]),
        "correction_f1_improved": bool(
            hybrid["correction_f1"] > previous["correction_f1"]
        ),
        "overcorrection_reduced": bool(
            hybrid["overcorrection_rate"] < previous["overcorrection_rate"]
        ),
        "at_least_one_true_correction": bool(hybrid["correction_tp"] > 0),
    }
    decision = {
        "ready": True,
        "proceed_to_ablation": all(criteria.values()),
        "criteria": criteria,
        "note": (
            "A failed strict gate triggers head diagnostics, not Gold-test "
            "evaluation or automatic full-benchmark training."
        ),
    }
    _base.write_json(RESULTS_DIR / "m1_gate_decision.json", decision)
    return decision


def gate_prediction_comparison(gold_dev: pd.DataFrame) -> pd.DataFrame:
    """Return raw Gold references with V1 and hybrid predictions side by side."""
    _, previous = _selection_predictions(
        PREVIOUS_TNT_RESULTS_DIR / "selections" / "m1.json"
    )
    comparison = gold_dev[["source_id", "source_text", "target_text"]].rename(
        columns={"target_text": "reference"}
    )
    comparison = comparison.merge(
        previous[["source_id", "prediction"]].rename(
            columns={"prediction": "prediction_tnt_v1_m1"}
        ),
        on="source_id",
        validate="one_to_one",
    )
    hybrid_path = SELECTIONS_DIR / "m1.json"
    if hybrid_path.is_file():
        _, hybrid = _selection_predictions(hybrid_path)
        comparison = comparison.merge(
            hybrid[["source_id", "prediction"]].rename(
                columns={"prediction": "prediction_tnt_hybrid_m1"}
            ),
            on="source_id",
            validate="one_to_one",
        )
    return comparison
