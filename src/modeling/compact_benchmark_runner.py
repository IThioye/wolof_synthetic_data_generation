"""Final compact Transformer benchmark orchestration.

The benchmark intentionally keeps architecture and optimization fixed across
the gold control and retained methods M1--M4 and M6.  It creates one deterministic random
initialization, pretrains each synthetic condition independently, and then
fine-tunes every condition on the same locked gold training split.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from src.config import M7_RESULTS_DIR, M7_RUNS_DIR
from src.generation.benchmark_baselines import sha256_file
from src.modeling.benchmark_data import (
    assemble_training_data,
    canonical_text,
    load_locked_gold,
    load_synthetic_pairs,
    require_no_held_out_leakage,
)
from src.modeling.compact_transformer import (
    CompactTransformerArchitecture,
    prepare_compact_transformer,
)
from src.modeling.experiment_runner import run_condition
from src.modeling.metrics import normalization_metrics
from src.modeling.seq2seq_benchmark import Seq2SeqConfig, evaluate_checkpoint


BENCHMARK_NAME = "compact_transformer_continuous_v1"
METHODS = ("m1", "m2", "m3", "m4", "m6")
CONDITIONS = ("gold",) + METHODS
MATCHED_UNIQUE_ROWS = 3330
RUNS_DIR = M7_RUNS_DIR / BENCHMARK_NAME
RESULTS_DIR = M7_RESULTS_DIR / BENCHMARK_NAME
SELECTIONS_DIR = RESULTS_DIR / "selections"
TEST_DIR = RESULTS_DIR / "frozen_test"


@dataclass(frozen=True)
class CompactBenchmarkTrainingConfig(Seq2SeqConfig):
    """Seq2Seq settings extended with the supported scheduler controls."""

    lr_scheduler_type: str = "constant_with_warmup"
    warmup_ratio: float = 0.05


@dataclass(frozen=True)
class CompactBenchmarkProtocol:
    seed: int = 2026
    matched_unique_rows: int = MATCHED_UNIQUE_ROWS
    synthetic_epochs: float = 5.0
    gold_epochs: float = 20.0
    synthetic_learning_rate: float = 3e-4
    gold_learning_rate: float = 1e-4
    synthetic_batch_size: int = 64
    gold_batch_size: int = 32
    evaluation_batch_size: int = 16
    generated_eval_interval_epochs: int = 1

    def validate(self) -> None:
        if self.matched_unique_rows <= 0:
            raise ValueError("matched_unique_rows must be positive")
        if self.synthetic_epochs <= 0:
            raise ValueError("synthetic_epochs must be positive")
        if self.gold_epochs <= 0:
            raise ValueError("gold_epochs must be positive")
        if self.generated_eval_interval_epochs <= 0:
            raise ValueError("generated_eval_interval_epochs must be positive")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def prepare_benchmark_data(
    protocol: CompactBenchmarkProtocol,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], pd.DataFrame]:
    """Load, deduplicate, match, and leakage-check the seven conditions."""
    protocol.validate()
    gold = load_locked_gold()
    selected: dict[str, pd.DataFrame] = {}
    audit_rows: list[dict[str, object]] = []
    unique_by_method: dict[str, pd.DataFrame] = {}
    source_counts: dict[str, dict[str, int]] = {}

    for method in METHODS:
        raw = load_synthetic_pairs(
            method,
            max_rows=None,
            seed=protocol.seed,
            # The user explicitly chose to begin the benchmark before the
            # optional manual synthetic-quality review was complete.
            allow_provisional=True,
        )
        unique = assemble_training_data(raw, None, regime="synthetic_only", seed=protocol.seed)
        source_counts[method] = {
            "loaded_rows": len(raw),
            "unique_pair_rows": len(unique),
        }
        unique["target_key"] = unique["target_text"].map(canonical_text)
        # Some generators can propose more than one informal realization for
        # the same formal input.  Retain one deterministic candidate so every
        # method is trained on the same formal-sentence units.
        unique_by_method[method] = unique.drop_duplicates(
            "target_key", keep="first"
        ).reset_index(drop=True)

    common_targets = set.intersection(
        *(set(frame["target_key"]) for frame in unique_by_method.values())
    )
    if len(common_targets) < protocol.matched_unique_rows:
        raise ValueError(
            f"Only {len(common_targets):,} formal targets occur in every method; "
            f"{protocol.matched_unique_rows:,} are required"
        )
    selected_targets = set(
        pd.Series(sorted(common_targets), dtype=str)
        .sample(n=protocol.matched_unique_rows, random_state=protocol.seed)
        .tolist()
    )
    # Use one shared surface form as well as one shared canonical target.  This
    # prevents a stray case/Unicode variant in a generator artifact from
    # changing the supervision for only that method.
    target_reference = unique_by_method["m4"].set_index("target_key")["target_text"]

    for method in METHODS:
        unique = unique_by_method[method]
        matched = unique.loc[unique["target_key"].isin(selected_targets)].copy()
        matched["target_text"] = matched["target_key"].map(target_reference)
        matched = (
            matched.drop(columns="target_key")
            .sort_values("target_text", kind="stable")
            .reset_index(drop=True)
        )
        if len(matched) != protocol.matched_unique_rows:
            raise RuntimeError(
                f"Common-target selection produced {len(matched):,} rows for {method}"
            )
        dev_audit = require_no_held_out_leakage(matched, gold["dev"])
        test_audit = require_no_held_out_leakage(matched, gold["test"])
        selected[method] = matched
        audit_rows.append(
            {
                "method": method,
                **source_counts[method],
                "unique_target_rows": len(unique),
                "selected_rows": len(matched),
                "common_target_pool": len(common_targets),
                "seed": protocol.seed,
                "dev_input_overlap": dev_audit.input_overlap,
                "dev_pair_overlap": dev_audit.pair_overlap,
                "test_input_overlap": test_audit.input_overlap,
                "test_pair_overlap": test_audit.pair_overlap,
                "source_over_255_chars": int(matched["source_text"].str.len().gt(255).sum()),
                "target_over_255_chars": int(matched["target_text"].str.len().gt(255).sum()),
            }
        )

    audit = pd.DataFrame(audit_rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    audit.to_csv(RESULTS_DIR / "dataset_audit.csv", index=False, encoding="utf-8")
    write_json(
        RESULTS_DIR / "protocol.json",
        {
            "created_at_utc": utc_now(),
            "benchmark": BENCHMARK_NAME,
            "conditions": list(CONDITIONS),
            "trained_synthetic_methods": list(METHODS),
            "m0_policy": "unchanged-input baseline; no trained synthetic condition",
            "protocol": asdict(protocol),
        },
    )
    return gold, selected, audit


def prepare_shared_initialization(
    gold: Mapping[str, pd.DataFrame],
    selected: Mapping[str, pd.DataFrame],
    protocol: CompactBenchmarkProtocol,
) -> Path:
    """Create one immutable vocabulary and random initialization for all runs."""
    architecture = CompactTransformerArchitecture(
        d_model=128,
        attention_heads=4,
        encoder_layers=2,
        decoder_layers=2,
        feed_forward_size=512,
        dropout=0.1,
        max_position_embeddings=256,
        generation_length_ratio=1.25,
        generation_length_margin=16,
        use_incremental_generation=False,
        copy_aware=False,
        copy_position_bias=10.0,
        copy_gate_bias=-3.0,
        copy_regularization_strength=0.0,
        edit_position_weight=1.0,
        label_smoothing=0.0,
        tie_character_embeddings=False,
        scheduled_sampling_probability=0.0,
    )
    frames = [gold["train"]] + [selected[method] for method in METHODS]
    return prepare_compact_transformer(
        frames,
        output_root=RUNS_DIR / "shared_initialization",
        seed=protocol.seed,
        architecture=architecture,
    )


def _training_config(
    model_path: Path,
    *,
    epochs: float,
    learning_rate: float,
    train_batch_size: int,
    protocol: CompactBenchmarkProtocol,
) -> CompactBenchmarkTrainingConfig:
    return CompactBenchmarkTrainingConfig(
        model_name_or_path=str(Path(model_path).resolve()),
        model_revision=None,
        task_prefix="",
        max_source_length=256,
        max_target_length=256,
        learning_rate=learning_rate,
        weight_decay=0.01,
        optim="adamw_torch_fused",
        use_lora=False,
        num_train_epochs=float(epochs),
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=protocol.evaluation_batch_size,
        gradient_accumulation_steps=1,
        generation_num_beams=1,
        # Teacher-forced loss did not track autoregressive normalization
        # quality in the copy-aware pilot.  The gold dev set is only 30 rows,
        # so generated CER can be used at every epoch at acceptable cost.
        checkpoint_selection_metric="cer",
        generate_final_development_metrics=True,
        seed=protocol.seed,
        fp16=False,
        bf16=False,
        tf32=True,
        full_determinism=False,
        gradient_checkpointing=False,
        save_total_limit=1,
        logging_steps=1,
        generated_eval_interval_epochs=protocol.generated_eval_interval_epochs,
        lr_scheduler_type="constant_with_warmup",
        warmup_ratio=0.05,
    )


def _selection_path(condition: str) -> Path:
    return SELECTIONS_DIR / f"{condition}.json"


def load_selection(condition: str) -> dict[str, object] | None:
    path = _selection_path(condition)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _save_selection(
    condition: str,
    best_record: Mapping[str, object],
    records: Sequence[Mapping[str, object]],
    protocol: CompactBenchmarkProtocol,
) -> dict[str, object]:
    metrics = best_record.get("development_metrics", {}) or {}
    payload: dict[str, object] = {
        "selected_at_utc": utc_now(),
        "benchmark": BENCHMARK_NAME,
        "condition": condition,
        "seed": protocol.seed,
        "selection_split": "gold_dev",
        "selection_metric": "cer",
        "selected_checkpoint": best_record["best_model_path"],
        "selected_run_manifest": str(
            (Path(str(best_record["attempt_dir"])) / "run_manifest.json").resolve()
        ),
        "configuration": best_record["configuration"],
        "development_metrics": metrics,
        "candidate_runs": [
            {
                "regime": record.get("regime"),
                "attempt": record.get("attempt"),
                "checkpoint": record.get("best_model_path"),
                "development_metrics": record.get("development_metrics", {}),
            }
            for record in records
        ],
        "protocol": asdict(protocol),
    }
    write_json(_selection_path(condition), payload)
    return payload


def _run_continuous_gold_training(
    *,
    condition: str,
    initial_model: Path,
    parent_manifest: Path,
    gold: Mapping[str, pd.DataFrame],
    protocol: CompactBenchmarkProtocol,
    rerun: bool,
) -> dict[str, object]:
    """Fine-tune without resetting optimizer or scheduler between epochs."""
    config = _training_config(
        initial_model,
        epochs=protocol.gold_epochs,
        learning_rate=protocol.gold_learning_rate,
        train_batch_size=protocol.gold_batch_size,
        protocol=protocol,
    )
    record = run_condition(
        gold["train"],
        gold["dev"],
        method=condition,
        regime="gold_finetune_continuous",
        seed=protocol.seed,
        config=config,
        runs_dir=RUNS_DIR,
        rerun=rerun,
        resume=True,
        parent_manifest_path=parent_manifest,
        quality_gate_override=True,
    )
    metrics = record.get("development_metrics", {}) or {}
    print(
        f"[compact benchmark] {condition}: continuous {protocol.gold_epochs:g}-epoch "
        f"gold run complete; selected dev CER={metrics.get('cer')}",
        flush=True,
    )
    return _save_selection(condition, record, [record], protocol)


def run_gold_control(
    gold: Mapping[str, pd.DataFrame],
    shared_initialization: Path,
    protocol: CompactBenchmarkProtocol,
    *,
    rerun: bool = False,
) -> dict[str, object]:
    """Train the real-gold-only control from the same random initialization."""
    init_manifest = Path(shared_initialization) / "initialization_manifest.json"
    return _run_continuous_gold_training(
        condition="gold",
        initial_model=shared_initialization,
        parent_manifest=init_manifest,
        gold=gold,
        protocol=protocol,
        rerun=rerun,
    )


def run_synthetic_method(
    method: str,
    training: pd.DataFrame,
    gold: Mapping[str, pd.DataFrame],
    shared_initialization: Path,
    protocol: CompactBenchmarkProtocol,
    *,
    rerun: bool = False,
) -> dict[str, object]:
    """Train one isolated synthetic method, then fine-tune it on gold."""
    if method not in METHODS:
        raise ValueError(f"Unknown benchmark method {method!r}; choose from {METHODS}")
    synthetic_config = _training_config(
        shared_initialization,
        epochs=protocol.synthetic_epochs,
        learning_rate=protocol.synthetic_learning_rate,
        train_batch_size=protocol.synthetic_batch_size,
        protocol=protocol,
    )
    synthetic_record = run_condition(
        training,
        gold["dev"],
        method=method,
        regime="synthetic_pretrain",
        seed=protocol.seed,
        config=synthetic_config,
        runs_dir=RUNS_DIR,
        rerun=rerun,
        resume=True,
        parent_manifest_path=Path(shared_initialization) / "initialization_manifest.json",
        quality_gate_override=True,
    )
    return _run_continuous_gold_training(
        condition=method,
        initial_model=Path(str(synthetic_record["best_model_path"])),
        parent_manifest=Path(str(synthetic_record["attempt_dir"])) / "run_manifest.json",
        gold=gold,
        protocol=protocol,
        rerun=rerun,
    )


def development_summary(gold_dev: pd.DataFrame) -> pd.DataFrame:
    """Collect selected dev metrics and the unchanged-input reference baseline."""
    identity = normalization_metrics(
        gold_dev["source_text"].tolist(),
        gold_dev["source_text"].tolist(),
        gold_dev["target_text"].tolist(),
    )
    rows: list[dict[str, object]] = [
        {"condition": "m0_identity", "kind": "reference_baseline", **identity}
    ]
    for condition in CONDITIONS:
        selection = load_selection(condition)
        if selection is None:
            continue
        rows.append(
            {
                "condition": condition,
                "kind": "gold_control" if condition == "gold" else "synthetic_then_gold",
                **(selection.get("development_metrics", {}) or {}),
                "checkpoint": selection.get("selected_checkpoint"),
            }
        )
    summary = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(RESULTS_DIR / "development_summary.csv", index=False, encoding="utf-8")
    return summary


def missing_selections() -> list[str]:
    return [condition for condition in CONDITIONS if load_selection(condition) is None]


def _freeze_payload(protocol: CompactBenchmarkProtocol) -> dict[str, object]:
    selections: dict[str, object] = {}
    for condition in CONDITIONS:
        selection = load_selection(condition)
        if selection is None:
            raise RuntimeError(f"Missing selected checkpoint for {condition}")
        manifest_path = Path(str(selection["selected_run_manifest"]))
        selections[condition] = {
            "selected_checkpoint": selection["selected_checkpoint"],
            "selected_run_manifest": str(manifest_path.resolve()),
            "selected_run_manifest_sha256": sha256_file(manifest_path),
        }
    return {
        "benchmark": BENCHMARK_NAME,
        "seed": protocol.seed,
        "protocol": asdict(protocol),
        "selections": selections,
    }


def evaluate_frozen_test(
    gold_test: pd.DataFrame,
    protocol: CompactBenchmarkProtocol,
    *,
    confirm: bool = False,
) -> pd.DataFrame:
    """Evaluate all finalized conditions once; require an explicit confirmation."""
    if not confirm:
        raise RuntimeError(
            "Frozen test evaluation is disabled. Set confirm=True only after all "
            "seven conditions and the protocol are final."
        )
    missing = missing_selections()
    if missing:
        raise RuntimeError(f"Cannot evaluate the test set; missing selections: {missing}")

    TEST_DIR.mkdir(parents=True, exist_ok=True)
    freeze_path = RESULTS_DIR / "frozen_test_manifest.json"
    payload = _freeze_payload(protocol)
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    payload["selection_fingerprint"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    payload["evaluated_at_utc"] = utc_now()
    if freeze_path.is_file():
        previous = json.loads(freeze_path.read_text(encoding="utf-8"))
        if previous.get("selection_fingerprint") != payload["selection_fingerprint"]:
            raise RuntimeError(
                "The test set was already evaluated with different checkpoint "
                "selections. Refusing to silently replace the frozen comparison."
            )
    else:
        write_json(freeze_path, payload)

    identity = normalization_metrics(
        gold_test["source_text"].tolist(),
        gold_test["source_text"].tolist(),
        gold_test["target_text"].tolist(),
    )
    rows: list[dict[str, object]] = [
        {"condition": "m0_identity", "kind": "reference_baseline", **identity}
    ]
    for condition in CONDITIONS:
        selection = load_selection(condition)
        assert selection is not None
        config = CompactBenchmarkTrainingConfig(**selection["configuration"])
        result = evaluate_checkpoint(
            selection["selected_checkpoint"],
            gold_test,
            output_dir=TEST_DIR / condition,
            config=config,
            split_name="test",
        )
        rows.append(
            {
                "condition": condition,
                "kind": "gold_control" if condition == "gold" else "synthetic_then_gold",
                **result["metrics"],
                "checkpoint": selection["selected_checkpoint"],
                "predictions_path": result["predictions_path"],
                "inference_seconds": result["inference_seconds"],
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(TEST_DIR / "test_summary.csv", index=False, encoding="utf-8")
    return summary


def plot_summary(summary: pd.DataFrame, path: Path, title: str) -> Path | None:
    if summary.empty:
        return None
    import matplotlib.pyplot as plt

    metrics = [metric for metric in ("cer", "wer", "chrf", "correction_f1") if metric in summary]
    if not metrics:
        return None
    figure, axes = plt.subplots(1, len(metrics), figsize=(4.5 * len(metrics), 4.5))
    if len(metrics) == 1:
        axes = [axes]
    for axis, metric in zip(axes, metrics):
        axis.bar(summary["condition"], pd.to_numeric(summary[metric], errors="coerce"))
        axis.set_title(metric.upper())
        axis.tick_params(axis="x", rotation=45)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle(title)
    figure.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path
