"""Resumable, versioned orchestration for the M7 normalization benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pandas as pd

from src.config import (
    M2_EFLOMAL_REVIEW_PATH,
    M3_CMDR_REVIEW_PATH,
    M4_OOLEL_REVIEW_PATH,
    M5_GENERATION_REVIEW_PATH,
    M5_MAPPING_REVIEW_PATH,
    M6_REVIEW_PATH,
    M7_RESULTS_DIR,
    M7_RUNS_DIR,
    M7_SUMMARY_PATH,
)
from src.generation.benchmark_baselines import sha256_file
from src.modeling.benchmark_data import (
    METHOD_MANIFEST_PATHS,
    METHOD_PATHS,
    assemble_training_data,
    load_locked_gold,
    load_synthetic_pairs,
    require_no_held_out_leakage,
)
from src.modeling.seq2seq_benchmark import (
    DEFAULT_MT5_SMALL_REVISION,
    Seq2SeqConfig,
    evaluate_checkpoint,
    train_seq2seq,
    write_benchmark_summary,
)


# M0 remains available as a generated artifact for provenance, but it is not a
# synthetic-error method and is therefore excluded from trained M7 conditions.
DEFAULT_METHODS = tuple(method for method in METHOD_PATHS if method != "m0")
DEFAULT_MATCHED_ROWS = 3433
DEVELOPMENT_SUMMARY_PATH = M7_RESULTS_DIR / "development_summary.csv"
DEVELOPMENT_AGGREGATE_PATH = M7_RESULTS_DIR / "development_aggregate.csv"
DEVELOPMENT_MARKDOWN_PATH = M7_RESULTS_DIR / "development_results.md"
RUN_INDEX_PATH = M7_RESULTS_DIR / "run_index.csv"
DATASET_STATUS_PATH = M7_RESULTS_DIR / "dataset_status.csv"
DEVELOPMENT_PLOT_PATH = M7_RESULTS_DIR / "development_metrics.png"
TEST_PLOT_PATH = M7_RESULTS_DIR / "test_metrics.png"
TEST_AGGREGATE_PATH = M7_RESULTS_DIR / "test_aggregate.csv"
TEST_MARKDOWN_PATH = M7_RESULTS_DIR / "test_results.md"
BENCHMARK_CONFIG_PATH = M7_RESULTS_DIR / "benchmark_config.json"
FROZEN_TEST_MANIFEST_PATH = M7_RESULTS_DIR / "frozen_test_manifest.json"

REVIEW_REQUIREMENTS: dict[str, tuple[tuple[Path, tuple[str, ...]], ...]] = {
    "m2": ((M2_EFLOMAL_REVIEW_PATH, ("meaning_preserved", "code_switch_natural_1_5", "spelling_plausible_1_5", "synthetic_artifact")),),
    "m3": ((M3_CMDR_REVIEW_PATH, ("meaning_preserved", "code_switch_natural_1_5", "spelling_plausible_1_5", "synthetic_artifact")),),
    "m4": ((M4_OOLEL_REVIEW_PATH, ("meaning_preserved", "informal_plausibility_1_5", "code_switch_natural_1_5", "synthetic_artifact")),),
    "m5": (
        (M5_MAPPING_REVIEW_PATH, ("mapping_correct",)),
        (M5_GENERATION_REVIEW_PATH, ("meaning_preserved", "informal_plausibility_1_5")),
    ),
    "m6": ((M6_REVIEW_PATH, ("meaning_preserved", "informal_plausibility_1_5", "severity_1_5", "synthetic_artifact")),),
}


@dataclass(frozen=True)
class BenchmarkRunConfig:
    methods: tuple[str, ...] = DEFAULT_METHODS
    seeds: tuple[int, ...] = (2026,)
    matched_synthetic_rows: int | None = DEFAULT_MATCHED_ROWS
    allow_provisional: bool = False
    base_model: Seq2SeqConfig = field(
        default_factory=lambda: Seq2SeqConfig(
            model_name_or_path="google/mt5-small",
            model_revision=DEFAULT_MT5_SMALL_REVISION,
            max_source_length=192,
            max_target_length=96,
            learning_rate=1e-3,
            optim="adafactor",
            num_train_epochs=3.0,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=8,
            gradient_accumulation_steps=1,
            generation_num_beams=1,
            checkpoint_selection_metric="loss",
            bf16=True,
            gradient_checkpointing=False,
            save_total_limit=1,
            logging_steps=1,
        )
    )
    synthetic_pretrain_epochs: float = 1.0
    gold_finetune_epochs: float = 5.0
    gold_finetune_learning_rate: float = 5e-4
    # Optional output namespace for architecture diagnostics that use the same
    # synthetic method. None preserves all existing paths and fingerprints.
    run_label: str | None = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_dump(path: Path, payload: Mapping[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def frame_fingerprint(data: pd.DataFrame) -> str:
    required = ["source_id", "source_text", "target_text"]
    missing = set(required) - set(data.columns)
    if missing:
        raise ValueError(f"Cannot fingerprint data missing: {sorted(missing)}")
    digest = hashlib.sha256()
    for row in data[required].itertuples(index=False, name=None):
        digest.update(json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def configuration_fingerprint(
    config: Seq2SeqConfig,
    training_sha256: str,
    development_sha256: str,
    parent_manifest_sha256: str | None = None,
) -> str:
    payload = {
        "configuration": asdict(config),
        "training_sha256": training_sha256,
        "development_sha256": development_sha256,
        "parent_manifest_sha256": parent_manifest_sha256,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _review_status(method: str) -> dict[str, object]:
    requirements = REVIEW_REQUIREMENTS.get(method, ())
    if not requirements:
        return {
            "review_required": False,
            "review_complete": True,
            "review_completed_cells": 0,
            "review_required_cells": 0,
            "review_details": "not configured as a method-specific gate",
        }
    required_cells = completed_cells = 0
    details = []
    complete = True
    for path, columns in requirements:
        if not Path(path).is_file():
            complete = False
            details.append(f"missing:{Path(path).name}")
            continue
        data = pd.read_csv(path, keep_default_na=False)
        missing = set(columns) - set(data.columns)
        if missing:
            complete = False
            details.append(f"{Path(path).name}:missing_columns={sorted(missing)}")
            continue
        for column in columns:
            filled = data[column].astype(str).str.strip().ne("")
            required_cells += len(data)
            completed_cells += int(filled.sum())
            complete = complete and bool(filled.all())
        details.append(f"{Path(path).name}:{completed_cells}/{required_cells}")
    return {
        "review_required": True,
        "review_complete": complete,
        "review_completed_cells": completed_cells,
        "review_required_cells": required_cells,
        "review_details": "; ".join(details),
    }


def dataset_status(methods: Sequence[str] = DEFAULT_METHODS) -> pd.DataFrame:
    rows = []
    for method in methods:
        path = Path(METHOD_PATHS[method])
        manifest_path = Path(METHOD_MANIFEST_PATHS.get(method, ""))
        manifest: dict[str, object] = {}
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        data_rows = None
        changed_rows = None
        schema_ready = False
        error = ""
        if path.is_file():
            try:
                data = pd.read_csv(path)
                data_rows = len(data)
                schema_ready = {"source_id", "method", "informal_wolof", "formal_wolof"} <= set(data.columns)
                if {"informal_wolof", "formal_wolof"} <= set(data.columns):
                    changed_rows = int(data["informal_wolof"].astype(str).ne(data["formal_wolof"].astype(str)).sum())
            except Exception as exc:  # status must report rather than hide a broken artifact
                error = str(exc)
        review = _review_status(method)
        pending = manifest.get("finalization_required", []) or []
        rows.append(
            {
                "method": method,
                "dataset_exists": path.is_file(),
                "dataset_path": str(path.resolve()),
                "rows": data_rows,
                "changed_rows": changed_rows,
                "schema_ready": schema_ready,
                "manifest_exists": manifest_path.is_file(),
                "provisional": bool(manifest.get("provisional", False)),
                "manifest_pending_count": len(pending),
                "manifest_pending": "; ".join(str(item) for item in pending),
                **review,
                "status_error": error,
            }
        )
    return pd.DataFrame(rows)


def _attempt_number(path: Path) -> int:
    try:
        return int(path.name.removeprefix("attempt_"))
    except ValueError:
        return -1


def attempt_directories(condition_dir: Path) -> list[Path]:
    condition_dir = Path(condition_dir)
    if not condition_dir.is_dir():
        return []
    return sorted(
        [path for path in condition_dir.glob("attempt_[0-9][0-9][0-9]") if path.is_dir()],
        key=_attempt_number,
    )


def next_attempt_directory(condition_dir: Path) -> Path:
    attempts = attempt_directories(condition_dir)
    number = max((_attempt_number(path) for path in attempts), default=0) + 1
    return Path(condition_dir) / f"attempt_{number:03d}"


def _attempt_manifest(attempt: Path) -> dict[str, object] | None:
    path = Path(attempt) / "run_manifest.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def latest_completed_attempt(condition_dir: Path) -> tuple[Path, dict[str, object]] | None:
    for attempt in reversed(attempt_directories(condition_dir)):
        manifest = _attempt_manifest(attempt)
        if manifest and manifest.get("experiment_status") == "complete":
            return attempt, manifest
    return None


def _matching_completed_attempt(
    condition_dir: Path, experiment_fingerprint: str
) -> tuple[Path, dict[str, object]] | None:
    for attempt in reversed(attempt_directories(condition_dir)):
        manifest = _attempt_manifest(attempt)
        if (
            manifest
            and manifest.get("experiment_status") == "complete"
            and manifest.get("experiment_fingerprint") == experiment_fingerprint
        ):
            return attempt, manifest
    return None


def _last_checkpoint(attempt: Path) -> str | None:
    try:
        from transformers.trainer_utils import get_last_checkpoint

        return get_last_checkpoint(str(attempt))
    except (ImportError, ValueError):
        checkpoints = sorted(
            [path for path in Path(attempt).glob("checkpoint-*") if path.is_dir()],
            key=lambda path: int(path.name.split("-")[-1]),
        )
        return str(checkpoints[-1]) if checkpoints else None


def _matching_incomplete_attempt(condition_dir: Path, fingerprint: str) -> tuple[Path, str] | None:
    for attempt in reversed(attempt_directories(condition_dir)):
        input_path = attempt / "experiment_input.json"
        if not input_path.is_file():
            continue
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        if payload.get("experiment_fingerprint") != fingerprint:
            continue
        checkpoint = _last_checkpoint(attempt)
        if checkpoint:
            return attempt, checkpoint
    return None


def run_condition(
    training: pd.DataFrame,
    development: pd.DataFrame,
    *,
    method: str,
    regime: str,
    seed: int,
    config: Seq2SeqConfig,
    runs_dir: Path = M7_RUNS_DIR,
    rerun: bool = False,
    resume: bool = True,
    parent_manifest_path: Path | None = None,
    quality_gate_override: bool = False,
) -> dict[str, object]:
    training_sha = frame_fingerprint(training)
    development_sha = frame_fingerprint(development)
    parent_sha = sha256_file(parent_manifest_path) if parent_manifest_path else None
    fingerprint = configuration_fingerprint(config, training_sha, development_sha, parent_sha)
    condition_dir = Path(runs_dir) / method / f"seed_{seed}" / regime

    if not rerun:
        completed = _matching_completed_attempt(condition_dir, fingerprint)
        if completed:
            attempt, manifest = completed
            print(
                f"[M7] SKIP {method} / {regime} / seed {seed}: "
                f"completed attempt {_attempt_number(attempt):03d}",
                flush=True,
            )
            return {**manifest, "run_action": "skipped_complete", "attempt_dir": str(attempt.resolve())}

    attempt: Path
    resume_checkpoint: str | None = None
    resumable = _matching_incomplete_attempt(condition_dir, fingerprint) if resume and not rerun else None
    if resumable:
        attempt, resume_checkpoint = resumable
        action = "resumed"
    else:
        attempt = next_attempt_directory(condition_dir)
        attempt.mkdir(parents=True, exist_ok=False)
        action = "trained"

    input_payload = {
        "created_at_utc": utc_now(),
        "method": method,
        "regime": regime,
        "seed": seed,
        "attempt": _attempt_number(attempt),
        "experiment_fingerprint": fingerprint,
        "training_sha256": training_sha,
        "development_sha256": development_sha,
        "parent_manifest_path": str(parent_manifest_path.resolve()) if parent_manifest_path else None,
        "parent_manifest_sha256": parent_sha,
        "quality_gate_override": quality_gate_override,
        "configuration": asdict(config),
    }
    _json_dump(attempt / "experiment_input.json", input_payload)
    print(
        f"\n[M7] START {method} / {regime} / seed {seed} / "
        f"attempt {_attempt_number(attempt):03d} ({action})",
        flush=True,
    )
    print(f"[M7] Output: {attempt.resolve()}", flush=True)
    try:
        manifest = train_seq2seq(
            training,
            development,
            output_dir=attempt,
            config=config,
            resume_from_checkpoint=resume_checkpoint,
        )
    except (Exception, KeyboardInterrupt) as exc:
        _json_dump(
            attempt / "failure.json",
            {**input_payload, "failed_at_utc": utc_now(), "error": repr(exc)},
        )
        raise

    manifest.update(
        {
            "method": method,
            "regime": regime,
            "seed": seed,
            "attempt": _attempt_number(attempt),
            "attempt_dir": str(attempt.resolve()),
            "experiment_status": "complete",
            "experiment_fingerprint": fingerprint,
            "training_sha256": training_sha,
            "development_sha256": development_sha,
            "parent_manifest_path": str(parent_manifest_path.resolve()) if parent_manifest_path else None,
            "parent_manifest_sha256": parent_sha,
            "quality_gate_override": quality_gate_override,
        }
    )
    _json_dump(attempt / "run_manifest.json", manifest)
    _json_dump(
        condition_dir / "latest.json",
        {
            "updated_at_utc": utc_now(),
            "attempt": _attempt_number(attempt),
            "attempt_dir": str(attempt.resolve()),
            "run_manifest": str((attempt / "run_manifest.json").resolve()),
            "experiment_fingerprint": fingerprint,
        },
    )
    development_metrics = manifest.get("development_metrics", {}) or {}
    completion_detail = (
        f"dev CER={development_metrics.get('cer')}"
        if "cer" in development_metrics
        else "checkpoint ready; generated development metrics skipped"
    )
    print(
        f"[M7] COMPLETE {method} / {regime} / seed {seed} / "
        f"attempt {_attempt_number(attempt):03d}; {completion_detail}",
        flush=True,
    )
    return {**manifest, "run_action": action}


def _write_benchmark_configuration(config: BenchmarkRunConfig) -> None:
    M7_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _json_dump(
        BENCHMARK_CONFIG_PATH,
        {
            "updated_at_utc": utc_now(),
            "methods": list(config.methods),
            "seeds": list(config.seeds),
            "matched_synthetic_rows": config.matched_synthetic_rows,
            "allow_provisional": config.allow_provisional,
            "base_model": asdict(config.base_model),
            "synthetic_pretrain_epochs": config.synthetic_pretrain_epochs,
            "gold_finetune_epochs": config.gold_finetune_epochs,
            "gold_finetune_learning_rate": config.gold_finetune_learning_rate,
            "run_label": config.run_label,
        },
    )


def run_training_suite(
    config: BenchmarkRunConfig,
    *,
    regimes: Sequence[str] = ("gold_only", "synthetic_then_gold"),
    rerun: bool = False,
    resume: bool = True,
    runs_dir: Path = M7_RUNS_DIR,
) -> list[dict[str, object]]:
    if config.run_label and (
        not config.run_label.replace("_", "").isalnum()
        or config.run_label.startswith("_")
        or config.run_label.endswith("_")
    ):
        raise ValueError(
            "run_label must contain only letters, digits, and internal underscores"
        )
    regimes = tuple(dict.fromkeys(value.casefold() for value in regimes))
    unknown = set(regimes) - {"gold_only", "synthetic_then_gold"}
    if unknown:
        raise ValueError(f"Unknown regimes: {sorted(unknown)}")
    gold = load_locked_gold()
    held_out = pd.concat([gold["dev"], gold["test"]], ignore_index=True)
    synthetic_regimes_requested = bool(
        {"synthetic_then_gold"} & set(regimes)
    )

    # Preflight every selected synthetic method before spending compute on a
    # partial synthetic suite. A gold-only run has no synthetic dependency.
    preflight_errors = {}
    if synthetic_regimes_requested:
        for method in config.methods:
            try:
                load_synthetic_pairs(
                    method,
                    max_rows=config.matched_synthetic_rows,
                    seed=config.seeds[0],
                    allow_provisional=config.allow_provisional,
                )
            except (FileNotFoundError, ValueError) as exc:
                preflight_errors[method] = str(exc)
    if preflight_errors:
        raise RuntimeError(
            "Selected methods are not training-ready:\n"
            + "\n".join(f"- {method}: {error}" for method, error in preflight_errors.items())
        )
    _write_benchmark_configuration(config)

    records: list[dict[str, object]] = []
    if "gold_only" in regimes:
        for seed in config.seeds:
            training = assemble_training_data(None, gold["train"], regime="gold_only", seed=seed)
            require_no_held_out_leakage(training, held_out)
            run_config = replace(config.base_model, seed=seed)
            records.append(
                run_condition(
                    training,
                    gold["dev"],
                    method="gold",
                    regime="gold_only",
                    seed=seed,
                    config=run_config,
                    runs_dir=runs_dir,
                    rerun=rerun,
                    resume=resume,
                    quality_gate_override=False,
                )
            )

    for method in config.methods if synthetic_regimes_requested else ():
        for seed in config.seeds:
            output_method = (
                f"{method}_{config.run_label}" if config.run_label else method
            )
            synthetic = load_synthetic_pairs(
                method,
                max_rows=config.matched_synthetic_rows,
                seed=seed,
                allow_provisional=config.allow_provisional,
            )
            if "synthetic_then_gold" in regimes:
                print(
                    f"\n[M7] {method} / seed {seed}: STAGE 1/2 — synthetic "
                    f"pretraining ({config.synthetic_pretrain_epochs:g} epoch)",
                    flush=True,
                )
                synthetic_training = assemble_training_data(
                    synthetic, None, regime="synthetic_only", seed=seed
                )
                require_no_held_out_leakage(synthetic_training, held_out)
                pretrain_config = replace(
                    config.base_model,
                    seed=seed,
                    num_train_epochs=config.synthetic_pretrain_epochs,
                    # Loss-only benchmark runs avoid this extra decoding pass.
                    # A CER-selected diagnostic needs the selected stage-1
                    # predictions so synthetic pretraining can be inspected.
                    generate_final_development_metrics=(
                        config.base_model.checkpoint_selection_metric == "cer"
                    ),
                )
                stage1_record = run_condition(
                    synthetic_training,
                    gold["dev"],
                    method=output_method,
                    regime="synthetic_pretrain",
                    seed=seed,
                    config=pretrain_config,
                    runs_dir=runs_dir,
                    rerun=rerun,
                    resume=resume,
                    quality_gate_override=config.allow_provisional,
                )
                stage1_attempt = Path(str(stage1_record["attempt_dir"]))
                stage1_manifest = stage1_attempt / "run_manifest.json"
                checkpoint = stage1_attempt / "best_model"
                if not checkpoint.is_dir():
                    raise FileNotFoundError(f"Missing selected stage-1 model: {checkpoint}")
                print(
                    f"\n[M7] {method} / seed {seed}: STAGE 2/2 — gold fine-tuning "
                    f"({config.gold_finetune_epochs:g} epochs)",
                    flush=True,
                )
                training = assemble_training_data(None, gold["train"], regime="gold_only", seed=seed)
                require_no_held_out_leakage(training, held_out)
                finetune_config = replace(
                    config.base_model,
                    model_name_or_path=str(checkpoint),
                    seed=seed,
                    learning_rate=config.gold_finetune_learning_rate,
                    num_train_epochs=config.gold_finetune_epochs,
                    generate_final_development_metrics=True,
                )
                records.append(
                    run_condition(
                        training,
                        gold["dev"],
                        method=output_method,
                        regime="synthetic_then_gold",
                        seed=seed,
                        config=finetune_config,
                        runs_dir=runs_dir,
                        rerun=rerun,
                        resume=resume,
                        parent_manifest_path=stage1_manifest,
                        quality_gate_override=config.allow_provisional,
                    )
                )
    write_run_outputs(runs_dir=runs_dir)
    return records


def collect_run_index(runs_dir: Path = M7_RUNS_DIR) -> pd.DataFrame:
    rows = []
    runs_dir = Path(runs_dir)
    if not runs_dir.is_dir():
        return pd.DataFrame()
    for attempt in sorted(runs_dir.glob("*/seed_*/*/attempt_*")):
        if not attempt.is_dir():
            continue
        parts = attempt.relative_to(runs_dir).parts
        if len(parts) != 4:
            continue
        method, seed_text, regime, attempt_text = parts
        manifest = _attempt_manifest(attempt)
        failure_path = attempt / "failure.json"
        status = "complete" if manifest and manifest.get("experiment_status") == "complete" else (
            "failed" if failure_path.is_file() else "incomplete"
        )
        rows.append(
            {
                "method": method,
                "seed": int(seed_text.removeprefix("seed_")),
                "regime": regime,
                "attempt": int(attempt_text.removeprefix("attempt_")),
                "status": status,
                "attempt_dir": str(attempt.resolve()),
                "experiment_fingerprint": manifest.get("experiment_fingerprint") if manifest else None,
                "quality_gate_override": manifest.get("quality_gate_override") if manifest else None,
                "cer": (manifest.get("development_metrics", {}) or {}).get("cer") if manifest else None,
                "wer": (manifest.get("development_metrics", {}) or {}).get("wer") if manifest else None,
                "chrf": (manifest.get("development_metrics", {}) or {}).get("chrf") if manifest else None,
                "correction_f1": (manifest.get("development_metrics", {}) or {}).get("correction_f1") if manifest else None,
            }
        )
    return pd.DataFrame(rows)


def collect_latest_development_results(runs_dir: Path = M7_RUNS_DIR) -> pd.DataFrame:
    rows = []
    runs_dir = Path(runs_dir)
    if not runs_dir.is_dir():
        return pd.DataFrame()
    for condition in sorted(runs_dir.glob("*/seed_*/*")):
        if not condition.is_dir() or condition.name.startswith("attempt_"):
            continue
        latest = latest_completed_attempt(condition)
        if latest is None:
            continue
        attempt, manifest = latest
        relative = condition.relative_to(runs_dir).parts
        if len(relative) != 3:
            continue
        method, seed_text, regime = relative
        source_method = method.split("_", 1)[0]
        if method != "gold" and source_method not in DEFAULT_METHODS:
            continue
        if regime not in {"gold_only", "synthetic_then_gold"}:
            continue
        row = {
            "method": method,
            "regime": regime,
            "seed": int(seed_text.removeprefix("seed_")),
            "attempt": _attempt_number(attempt),
            "checkpoint": str((attempt / "best_model").resolve()),
            "run_manifest": str((attempt / "run_manifest.json").resolve()),
            "quality_gate_override": bool(manifest.get("quality_gate_override", False)),
            "training_rows": manifest.get("training_rows"),
            "development_rows": manifest.get("development_rows"),
            "training_mode": manifest.get("training_mode"),
            "trainable_parameter_count": manifest.get("trainable_parameter_count"),
            "trainable_parameter_ratio": manifest.get("trainable_parameter_ratio"),
            "train_runtime_seconds": (manifest.get("train_metrics", {}) or {}).get("train_runtime"),
            "peak_cuda_memory_bytes": (manifest.get("hardware", {}) or {}).get("peak_cuda_memory_bytes"),
        }
        row.update(manifest.get("development_metrics", {}))
        rows.append(row)
    return pd.DataFrame(rows)


def plot_metric_summary(summary: pd.DataFrame, path: Path, *, title: str) -> Path | None:
    if summary.empty:
        return None
    import matplotlib.pyplot as plt
    import numpy as np

    metrics = (
        ("cer", "CER (lower is better)"),
        ("wer", "WER (lower is better)"),
        ("chrf", "chrF (higher is better)"),
        ("correction_f1", "Correction F1 (higher is better)"),
    )
    grouped = summary.groupby(["method", "regime"], sort=True)
    keys = list(grouped.groups)
    labels = [f"{method}\n{regime.replace('_', ' ')}" for method, regime in keys]
    x = np.arange(len(labels))
    figure, axes = plt.subplots(2, 2, figsize=(max(13, len(labels) * 1.05), 9))
    colors = ["#d17a22" if regime == "synthetic_then_gold" else "#3465a4" for _, regime in keys]
    for axis, (metric, label) in zip(axes.flat, metrics):
        means = grouped[metric].mean().tolist() if metric in summary.columns else [math.nan] * len(labels)
        stds = grouped[metric].std().fillna(0).tolist() if metric in summary.columns else [0] * len(labels)
        axis.bar(x, means, yerr=stds, color=colors, alpha=0.88, capsize=3)
        axis.set_title(label)
        axis.set_xticks(x)
        axis.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle(title)
    figure.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(figure)
    return path


def aggregate_results(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    metrics = [
        column for column in (
            "cer", "wer", "chrf", "exact_match", "correction_precision",
            "correction_recall", "correction_f1", "overcorrection_rate",
            "unchanged_sentence_overcorrection_rate",
        ) if column in summary.columns
    ]
    grouped = summary.groupby(["method", "regime"], as_index=False, sort=True)
    means = grouped[metrics].mean().rename(columns={metric: f"{metric}_mean" for metric in metrics})
    counts = grouped.size().rename(columns={"size": "seed_count"})
    result = means.merge(counts, on=["method", "regime"], validate="one_to_one")
    if summary["seed"].nunique() > 1:
        standard = grouped[metrics].std().rename(
            columns={metric: f"{metric}_std" for metric in metrics}
        )
        result = result.merge(standard, on=["method", "regime"], validate="one_to_one")
    return result


def write_markdown_results(summary: pd.DataFrame, path: Path, *, title: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    preferred = [
        column for column in (
            "method", "regime", "seed", "attempt", "cer", "wer", "chrf",
            "exact_match", "correction_precision", "correction_recall",
            "correction_f1", "overcorrection_rate",
            "inference_seconds", "rows_per_second", "peak_cuda_memory_bytes",
        ) if column in summary.columns
    ]
    lines = [f"# {title}", ""]
    if summary.empty:
        lines.append("No completed results yet.")
    else:
        table = summary[preferred].copy()
        for column in table.select_dtypes(include="number").columns:
            if column not in {"seed", "attempt"}:
                table[column] = table[column].map(lambda value: f"{value:.4f}" if pd.notna(value) else "")
        lines.append("| " + " | ".join(preferred) + " |")
        lines.append("| " + " | ".join("---" for _ in preferred) + " |")
        for row in table.itertuples(index=False, name=None):
            lines.append("| " + " | ".join(str(value) for value in row) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_run_outputs(runs_dir: Path = M7_RUNS_DIR) -> dict[str, Path | None]:
    M7_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    status = dataset_status()
    status.to_csv(DATASET_STATUS_PATH, index=False, encoding="utf-8")
    index = collect_run_index(runs_dir)
    index.to_csv(RUN_INDEX_PATH, index=False, encoding="utf-8")
    development = collect_latest_development_results(runs_dir)
    development.to_csv(DEVELOPMENT_SUMMARY_PATH, index=False, encoding="utf-8")
    aggregate = aggregate_results(development)
    aggregate.to_csv(DEVELOPMENT_AGGREGATE_PATH, index=False, encoding="utf-8")
    write_markdown_results(
        development,
        DEVELOPMENT_MARKDOWN_PATH,
        title="M7 development results (model selection only)",
    )
    plot = plot_metric_summary(
        development,
        DEVELOPMENT_PLOT_PATH,
        title="M7 development results by method and training regime",
    )
    return {
        "dataset_status": DATASET_STATUS_PATH,
        "run_index": RUN_INDEX_PATH,
        "development_summary": DEVELOPMENT_SUMMARY_PATH,
        "development_aggregate": DEVELOPMENT_AGGREGATE_PATH,
        "development_markdown": DEVELOPMENT_MARKDOWN_PATH,
        "development_plot": plot,
    }


def evaluate_frozen_suite(
    *,
    methods: Sequence[str],
    seeds: Sequence[int],
    regimes: Sequence[str],
    confirm_frozen: bool,
    runs_dir: Path = M7_RUNS_DIR,
    results_dir: Path = M7_RESULTS_DIR,
) -> pd.DataFrame:
    if not confirm_frozen:
        raise RuntimeError("Locked-test evaluation requires confirm_frozen=True")
    gold = load_locked_gold()
    selected: list[dict[str, object]] = []
    for method in methods:
        method_regimes = ("gold_only",) if method == "gold" else regimes
        for seed in seeds:
            for regime in method_regimes:
                condition = Path(runs_dir) / method / f"seed_{seed}" / regime
                latest = latest_completed_attempt(condition)
                if latest is None:
                    raise FileNotFoundError(f"No completed frozen run: {condition}")
                attempt, manifest = latest
                checkpoint = attempt / "best_model"
                selected.append(
                    {
                        "method": method,
                        "regime": regime,
                        "seed": seed,
                        "attempt": _attempt_number(attempt),
                        "checkpoint": str(checkpoint.resolve()),
                        "run_manifest": str((attempt / "run_manifest.json").resolve()),
                        "run_manifest_sha256": sha256_file(attempt / "run_manifest.json"),
                        "configuration": manifest.get("configuration", {}),
                    }
                )
    results_dir = Path(results_dir)
    frozen_manifest_path = results_dir / FROZEN_TEST_MANIFEST_PATH.name
    if frozen_manifest_path.exists():
        frozen_payload = json.loads(frozen_manifest_path.read_text(encoding="utf-8"))
        if frozen_payload.get("runs") != selected:
            raise RuntimeError(
                "Frozen test selection already exists with different attempts; "
                f"refusing to change it: {frozen_manifest_path}"
            )
    else:
        _json_dump(
            frozen_manifest_path,
            {"frozen_at_utc": utc_now(), "runs": selected},
        )
    records = []
    for item in selected:
        config = Seq2SeqConfig(**item["configuration"])
        result_dir = (
            Path(results_dir)
            / str(item["method"])
            / f"seed_{item['seed']}"
            / str(item["regime"])
            / f"attempt_{int(item['attempt']):03d}"
        )
        metrics_path = result_dir / "test_metrics.json"
        if metrics_path.is_file():
            result = json.loads(metrics_path.read_text(encoding="utf-8"))
            if result.get("checkpoint") != str(Path(str(item["checkpoint"])).resolve()):
                raise RuntimeError(
                    f"Existing test result uses a different checkpoint: {metrics_path}"
                )
        else:
            result = evaluate_checkpoint(
                item["checkpoint"],
                gold["test"],
                output_dir=result_dir,
                config=config,
                split_name="test",
            )
        records.append(
            {
                "method": item["method"],
                "regime": item["regime"],
                "seed": item["seed"],
                "attempt": item["attempt"],
                "checkpoint": item["checkpoint"],
                "model_parameter_count": result.get("model_parameter_count"),
                "inference_seconds": result.get("inference_seconds"),
                "rows_per_second": result.get("rows_per_second"),
                "peak_cuda_memory_bytes": (result.get("hardware", {}) or {}).get(
                    "peak_cuda_memory_bytes"
                ),
                "metrics": result["metrics"],
            }
        )
    summary_path = results_dir / M7_SUMMARY_PATH.name
    aggregate_path = results_dir / TEST_AGGREGATE_PATH.name
    markdown_path = results_dir / TEST_MARKDOWN_PATH.name
    plot_path = results_dir / TEST_PLOT_PATH.name
    summary = write_benchmark_summary(records, summary_path)
    aggregate_results(summary).to_csv(aggregate_path, index=False, encoding="utf-8")
    write_markdown_results(summary, markdown_path, title="M7 locked-test results")
    plot_metric_summary(summary, plot_path, title="M7 locked-test results")
    return summary


def _parse_methods(values: Sequence[str]) -> tuple[str, ...]:
    if not values or any(value.casefold() == "all" for value in values):
        return DEFAULT_METHODS
    methods = tuple(dict.fromkeys(value.casefold() for value in values))
    unknown = set(methods) - set(DEFAULT_METHODS)
    if unknown:
        raise ValueError(f"Unknown methods: {sorted(unknown)}")
    return methods


def _add_selection_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--methods", nargs="+", default=["all"], help="m1 ... m6, or all")
    parser.add_argument("--seeds", nargs="+", type=int, default=[2026])
    parser.add_argument(
        "--matched-rows",
        type=int,
        default=DEFAULT_MATCHED_ROWS,
        help="Equal synthetic rows per method; use 0 for every available row.",
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    status = subparsers.add_parser("status", help="Show dataset, quality-gate, and run status")
    _add_selection_arguments(status)

    train = subparsers.add_parser("train", help="Train/resume selected methods and regimes")
    _add_selection_arguments(train)
    train.add_argument(
        "--regimes",
        nargs="+",
        choices=("gold_only", "synthetic_then_gold"),
        default=["gold_only", "synthetic_then_gold"],
    )
    train.add_argument("--allow-provisional", action="store_true")
    train.add_argument("--rerun", action="store_true", help="Create a new versioned attempt even if complete")
    train.add_argument("--no-resume", action="store_true")
    train.add_argument("--model", default="google/mt5-small")
    train.add_argument(
        "--model-revision",
        default=DEFAULT_MT5_SMALL_REVISION,
        help="Exact Hugging Face revision; use an empty string to follow the model branch",
    )
    train.add_argument("--train-epochs", type=float, default=3.0)
    train.add_argument("--synthetic-pretrain-epochs", type=float, default=1.0)
    train.add_argument("--finetune-epochs", type=float, default=5.0)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--finetune-learning-rate", type=float, default=5e-4)
    train.add_argument(
        "--optimizer",
        choices=("adafactor", "adamw_torch", "adamw_torch_fused"),
        default="adafactor",
    )
    train.add_argument("--train-batch-size", type=int, default=32)
    train.add_argument("--eval-batch-size", type=int, default=8)
    train.add_argument("--gradient-accumulation", type=int, default=1)
    train.add_argument("--generation-num-beams", type=int, default=1)
    train.add_argument(
        "--checkpoint-selection-metric",
        choices=("loss", "cer"),
        default="loss",
        help=(
            "Use fast teacher-forced development loss by default; 'cer' performs "
            "autoregressive development generation after every epoch."
        ),
    )
    train.add_argument("--full-finetune", action="store_true", help="Train every base-model parameter instead of LoRA adapters")
    train.add_argument("--lora-rank", type=int, default=8)
    train.add_argument("--lora-alpha", type=int, default=16)
    train.add_argument("--lora-dropout", type=float, default=0.05)
    train.add_argument("--lora-target-modules", nargs="+", default=["q", "v"])
    train.add_argument("--fp16", action="store_true")
    train.add_argument("--no-bf16", action="store_true")
    train.add_argument("--no-tf32", action="store_true")
    train.add_argument("--full-determinism", action="store_true")
    train.add_argument("--gradient-checkpointing", action="store_true")

    summarize = subparsers.add_parser("summarize", help="Refresh CSV result indexes and plots")

    test = subparsers.add_parser("evaluate-test", help="One-time evaluation of frozen latest runs")
    _add_selection_arguments(test)
    test.add_argument(
        "--regimes",
        nargs="+",
        choices=("synthetic_then_gold",),
        default=["synthetic_then_gold"],
    )
    test.add_argument("--include-gold", action="store_true")
    test.add_argument("--confirm-frozen", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    if args.command == "status":
        methods = _parse_methods(args.methods)
        status = dataset_status(methods)
        outputs = write_run_outputs()
        print(status.to_string(index=False))
        print("\nSaved status/results:")
        for name, path in outputs.items():
            print(f"- {name}: {path}")
        return 0
    if args.command == "summarize":
        outputs = write_run_outputs()
        for name, path in outputs.items():
            print(f"{name}: {path}")
        return 0
    if args.command == "train":
        methods = _parse_methods(args.methods)
        matched_rows = None if args.matched_rows == 0 else args.matched_rows
        base = Seq2SeqConfig(
            model_name_or_path=args.model,
            model_revision=args.model_revision or None,
            max_source_length=192,
            max_target_length=96,
            learning_rate=args.learning_rate,
            optim=args.optimizer,
            use_lora=not args.full_finetune,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            lora_target_modules=tuple(args.lora_target_modules),
            num_train_epochs=args.train_epochs,
            per_device_train_batch_size=args.train_batch_size,
            per_device_eval_batch_size=args.eval_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation,
            generation_num_beams=args.generation_num_beams,
            checkpoint_selection_metric=args.checkpoint_selection_metric,
            seed=args.seeds[0],
            fp16=args.fp16,
            bf16=not args.no_bf16 and not args.fp16,
            tf32=not args.no_tf32,
            full_determinism=args.full_determinism,
            gradient_checkpointing=args.gradient_checkpointing,
            save_total_limit=1,
            logging_steps=1,
        )
        config = BenchmarkRunConfig(
            methods=methods,
            seeds=tuple(args.seeds),
            matched_synthetic_rows=matched_rows,
            allow_provisional=args.allow_provisional,
            base_model=base,
            synthetic_pretrain_epochs=args.synthetic_pretrain_epochs,
            gold_finetune_epochs=args.finetune_epochs,
            gold_finetune_learning_rate=args.finetune_learning_rate,
        )
        records = run_training_suite(
            config,
            regimes=args.regimes,
            rerun=args.rerun,
            resume=not args.no_resume,
        )
        print(pd.DataFrame([
            {
                "method": record.get("method"),
                "regime": record.get("regime"),
                "seed": record.get("seed"),
                "attempt": record.get("attempt"),
                "action": record.get("run_action"),
                "dev_cer": (record.get("development_metrics", {}) or {}).get("cer"),
            }
            for record in records
        ]).to_string(index=False))
        return 0
    if args.command == "evaluate-test":
        methods = list(_parse_methods(args.methods))
        if args.include_gold:
            methods = ["gold", *methods]
        summary = evaluate_frozen_suite(
            methods=methods,
            seeds=args.seeds,
            regimes=args.regimes,
            confirm_frozen=args.confirm_frozen,
        )
        print(summary.to_string(index=False))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
