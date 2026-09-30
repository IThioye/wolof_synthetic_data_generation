"""Gold-train-only ablations for the boundary/word-aware TNT V2 model.

This is an architecture-development experiment, deliberately isolated from the
frozen M7 benchmark.  It creates a video-disjoint inner validation split from
Gold-train and never loads Gold-dev or Gold-test.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import GOLD_TRAIN_PATH, M7_RESULTS_DIR, M7_RUNS_DIR
from src.generation.benchmark_baselines import sha256_file
from src.modeling.benchmark_data import load_gold_split
from src.modeling.seq2seq_benchmark import Seq2SeqConfig, train_seq2seq
from src.modeling.tnt_hybrid_transformer import (
    BOUNDARY_ACTIONS,
    WORD_ACTIONS,
    TntHybridArchitecture,
    hybrid_tnt_annotations,
    prepare_tnt_hybrid_transformer,
)
from src.modeling.tnt_edit_transformer import OPERATION_NAMES


EXPERIMENT_NAME = "tnt_hybrid_gold_train_ablation_v2"
RUNS_DIR = M7_RUNS_DIR / EXPERIMENT_NAME
RESULTS_DIR = M7_RESULTS_DIR / EXPERIMENT_NAME
ABLATIONS = ("character_only", "boundary_only", "word_only", "full_hybrid")


@dataclass(frozen=True)
class HybridExperimentProtocol:
    seed: int = 2026
    epochs: float = 30.0
    learning_rate: float = 3e-4
    train_batch_size: int = 32
    evaluation_batch_size: int = 24
    output_slots_per_source: int = 16
    boundary_loss_weight: float = 0.5
    word_loss_weight: float = 0.5
    boundary_confidence_threshold: float = 0.60
    word_keep_threshold: float = 0.80
    balance_edit_classes: bool = True
    maximum_class_weight: float = 4.0
    generated_eval_interval_epochs: int = 1

    def validate(self) -> None:
        if self.epochs <= 0 or self.learning_rate <= 0:
            raise ValueError("epochs and learning_rate must be positive")
        if self.train_batch_size <= 0 or self.evaluation_batch_size <= 0:
            raise ValueError("batch sizes must be positive")
        if self.output_slots_per_source <= 0:
            raise ValueError("output_slots_per_source must be positive")
        if self.maximum_class_weight < 1:
            raise ValueError("maximum_class_weight must be at least one")


@dataclass(frozen=True)
class HybridTrainingConfig(Seq2SeqConfig):
    """Keep a non-zero learning rate after a short warm-up on the tiny split."""

    lr_scheduler_type: str = "constant_with_warmup"
    warmup_ratio: float = 0.05


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def prepare_inner_gold_split(
    protocol: HybridExperimentProtocol,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Hold out one complete Gold-train video for architecture selection."""
    protocol.validate()
    gold_train = load_gold_split(GOLD_TRAIN_PATH, "train")
    video_counts = gold_train.groupby("video_url", sort=True).size()
    usable_videos = video_counts.loc[video_counts.index.astype(str).str.len() > 0]
    if len(usable_videos) < 2:
        raise ValueError("A video-disjoint inner split requires at least two videos")

    desired_rows = max(1, round(len(gold_train) * 0.20))
    distances = (usable_videos - desired_rows).abs()
    candidates = sorted(distances.loc[distances.eq(distances.min())].index.tolist())
    # Resolve equal-size candidates reproducibly without depending on row order.
    held_out_video = pd.Series(candidates).sample(
        n=1, random_state=protocol.seed
    ).iloc[0]
    inner_validation = gold_train.loc[
        gold_train["video_url"].eq(held_out_video)
    ].reset_index(drop=True)
    inner_training = gold_train.loc[
        ~gold_train["video_url"].eq(held_out_video)
    ].reset_index(drop=True)
    if set(inner_training["video_url"]) & set(inner_validation["video_url"]):
        raise RuntimeError("Video leakage in the inner Gold-train split")

    manifest: dict[str, object] = {
        "created_at_utc": _utc_now(),
        "experiment": EXPERIMENT_NAME,
        "source": str(GOLD_TRAIN_PATH.resolve()),
        "source_sha256": sha256_file(GOLD_TRAIN_PATH),
        "policy": "video-disjoint inner split made only from locked Gold-train",
        "selection_rule": "video size closest to 20% of Gold-train; seeded tie-break",
        "seed": protocol.seed,
        "held_out_video": held_out_video,
        "training_rows": len(inner_training),
        "validation_rows": len(inner_validation),
        "training_videos": sorted(inner_training["video_url"].unique().tolist()),
        "validation_videos": sorted(inner_validation["video_url"].unique().tolist()),
        "gold_dev_loaded": False,
        "gold_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "inner_split_manifest.json", manifest)
    return inner_training, inner_validation, manifest


def _inverse_sqrt_weights(
    counts: Counter, class_count: int, maximum: float
) -> tuple[float, ...]:
    """Return bounded inverse-square-root weights in label-index order."""
    total = sum(counts.values())
    if total == 0:
        return tuple(1.0 for _ in range(class_count))
    values = []
    for label in range(class_count):
        count = counts.get(label, 0)
        raw = math.sqrt(total / (class_count * count)) if count else maximum
        values.append(min(maximum, max(1 / maximum, raw)))
    return tuple(values)


def auxiliary_label_profile(
    training: pd.DataFrame, protocol: HybridExperimentProtocol
) -> dict[str, object]:
    """Measure auxiliary labels and derive weights from inner-training only."""
    boundary_counts: Counter = Counter()
    word_counts: Counter = Counter()
    operation_counts: Counter = Counter()
    for source_text, target_text in zip(
        training["source_text"], training["target_text"]
    ):
        # Unicode code points are sufficient here: the alignment only needs a
        # stable equality-preserving integer representation of each character.
        source_ids = [ord(character) + 4 for character in source_text] + [2]
        target_ids = [ord(character) + 4 for character in target_text] + [2]
        annotations = hybrid_tnt_annotations(
            source_ids,
            target_ids,
            eos_token_id=2,
            pad_token_id=0,
            space_token_id=ord(" ") + 4,
            output_slots_per_source=protocol.output_slots_per_source,
        )
        operation_counts.update(label for label in annotations[0] if label >= 0)
        boundary_counts.update(label for label in annotations[2] if label >= 0)
        word_counts.update(annotations[4])

    operation_weights = _inverse_sqrt_weights(
        operation_counts, len(OPERATION_NAMES), protocol.maximum_class_weight
    )
    boundary_weights = _inverse_sqrt_weights(
        boundary_counts, len(BOUNDARY_ACTIONS), protocol.maximum_class_weight
    )
    word_weights = _inverse_sqrt_weights(
        word_counts, len(WORD_ACTIONS), protocol.maximum_class_weight
    )
    return {
        "operation_counts": {
            action: operation_counts.get(index, 0)
            for index, action in enumerate(OPERATION_NAMES)
        },
        "boundary_counts": {
            action: boundary_counts.get(index, 0)
            for index, action in enumerate(BOUNDARY_ACTIONS)
        },
        "word_counts": {
            action: word_counts.get(index, 0)
            for index, action in enumerate(WORD_ACTIONS)
        },
        "operation_class_weights": operation_weights,
        "boundary_class_weights": boundary_weights,
        "word_class_weights": word_weights,
        "weight_formula": (
            "sqrt(total / (number_of_classes * class_count)), clipped to "
            f"[1/{protocol.maximum_class_weight:g}, "
            f"{protocol.maximum_class_weight:g}]"
        ),
        "computed_from": "inner Gold-training labels only",
    }


def _ablation_architecture(
    name: str,
    protocol: HybridExperimentProtocol,
    label_profile: dict[str, object] | None = None,
) -> TntHybridArchitecture:
    if name not in ABLATIONS:
        raise ValueError(f"Unknown ablation {name!r}; choose from {ABLATIONS}")
    boundary_enabled = name in {"boundary_only", "full_hybrid"}
    word_enabled = name in {"word_only", "full_hybrid"}
    label_profile = label_profile or {}
    return TntHybridArchitecture(
        d_model=192,
        attention_heads=4,
        encoder_layers=3,
        feed_forward_size=768,
        dropout=0.1,
        max_position_embeddings=256,
        output_slots_per_source=protocol.output_slots_per_source,
        operation_loss_weight=1.0,
        character_loss_weight=1.0,
        operation_class_weights=(
            tuple(label_profile.get("operation_class_weights", ()))
            if protocol.balance_edit_classes and label_profile
            else None
        ),
        boundary_loss_weight=(
            protocol.boundary_loss_weight if boundary_enabled else 0.0
        ),
        word_loss_weight=protocol.word_loss_weight if word_enabled else 0.0,
        boundary_class_weights=(
            tuple(label_profile.get("boundary_class_weights", ()))
            if protocol.balance_edit_classes and label_profile
            else None
        ),
        word_class_weights=(
            tuple(label_profile.get("word_class_weights", ()))
            if protocol.balance_edit_classes and label_profile
            else None
        ),
        label_smoothing=0.1,
        use_boundary_inference=boundary_enabled,
        use_word_keep_gate=word_enabled,
        boundary_confidence_threshold=protocol.boundary_confidence_threshold,
        word_keep_threshold=protocol.word_keep_threshold,
    )


def prepare_ablation_initialization(
    name: str,
    inner_training: pd.DataFrame,
    protocol: HybridExperimentProtocol,
) -> Path:
    """Prepare the deterministic initialization for one toggle combination."""
    label_profile = auxiliary_label_profile(inner_training, protocol)
    architecture = _ablation_architecture(name, protocol, label_profile)
    path = prepare_tnt_hybrid_transformer(
        # The held-out video's labels and character inventory stay outside the
        # model preparation step; unknown validation characters remain a fair
        # consequence of this deliberately small inner-training split.
        [inner_training],
        output_root=RUNS_DIR / "initializations" / name,
        seed=protocol.seed,
        architecture=architecture,
    )
    _write_json(
        RESULTS_DIR / f"{name}_protocol.json",
        {
            "created_at_utc": _utc_now(),
            "experiment": EXPERIMENT_NAME,
            "ablation": name,
            "architecture": asdict(architecture),
            "training": asdict(protocol),
            "auxiliary_label_profile": label_profile,
            "selection_split": "video-disjoint inner validation from Gold-train",
            "gold_dev_used": False,
            "gold_test_used": False,
        },
    )
    return path


def _latest_attempt(name: str) -> Path | None:
    root = RUNS_DIR / "ablations" / name
    attempts = sorted(path for path in root.glob("attempt_*")) if root.is_dir() else []
    return attempts[-1] if attempts else None


def _new_attempt(name: str) -> Path:
    root = RUNS_DIR / "ablations" / name
    root.mkdir(parents=True, exist_ok=True)
    existing = [
        int(path.name.removeprefix("attempt_"))
        for path in root.glob("attempt_*")
        if path.name.removeprefix("attempt_").isdigit()
    ]
    return root / f"attempt_{max(existing, default=0) + 1:03d}"


def run_ablation(
    name: str,
    inner_training: pd.DataFrame,
    inner_validation: pd.DataFrame,
    protocol: HybridExperimentProtocol,
    *,
    rerun: bool = False,
    resume: bool = True,
) -> dict[str, object]:
    """Run or resume one ablation without touching any other condition."""
    initialization = prepare_ablation_initialization(
        name, inner_training, protocol
    )
    latest = _latest_attempt(name)
    if latest is not None and (latest / "run_manifest.json").is_file() and not rerun:
        print(f"[TNT-V2] Reusing completed {name}: {latest.resolve()}")
        return json.loads((latest / "run_manifest.json").read_text(encoding="utf-8"))

    attempt = latest if latest is not None and not rerun else _new_attempt(name)
    resume_checkpoint: str | None = None
    if resume and attempt.is_dir():
        checkpoints = sorted(
            attempt.glob("checkpoint-*"),
            key=lambda path: int(path.name.rsplit("-", 1)[-1]),
        )
        if checkpoints:
            resume_checkpoint = str(checkpoints[-1].resolve())
            print(f"[TNT-V2] Resuming {name} from {resume_checkpoint}")

    config = HybridTrainingConfig(
        model_name_or_path=str(initialization.resolve()),
        model_revision=None,
        task_prefix="",
        max_source_length=256,
        max_target_length=256,
        learning_rate=protocol.learning_rate,
        weight_decay=0.01,
        optim="adamw_torch_fused",
        use_lora=False,
        num_train_epochs=protocol.epochs,
        per_device_train_batch_size=protocol.train_batch_size,
        per_device_eval_batch_size=protocol.evaluation_batch_size,
        gradient_accumulation_steps=1,
        generation_num_beams=1,
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
    )
    print(
        f"[TNT-V2] {name}: {len(inner_training)} inner-train rows, "
        f"{len(inner_validation)} inner-validation rows",
        flush=True,
    )
    return train_seq2seq(
        inner_training,
        inner_validation,
        output_dir=attempt,
        config=config,
        resume_from_checkpoint=resume_checkpoint,
    )


def ablation_summary() -> pd.DataFrame:
    """Collect completed inner-validation results; no frozen-set evaluation."""
    rows: list[dict[str, object]] = []
    for name in ABLATIONS:
        attempt = _latest_attempt(name)
        if attempt is None or not (attempt / "run_manifest.json").is_file():
            continue
        manifest = json.loads(
            (attempt / "run_manifest.json").read_text(encoding="utf-8")
        )
        rows.append(
            {
                "ablation": name,
                "attempt": attempt.name,
                **(manifest.get("development_metrics", {}) or {}),
                "checkpoint": manifest.get("best_model_path"),
            }
        )
    summary = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(
        RESULTS_DIR / "inner_validation_summary.csv", index=False, encoding="utf-8"
    )
    return summary


def prediction_comparison() -> pd.DataFrame:
    """Place completed inner-validation predictions side by side for review."""
    comparison: pd.DataFrame | None = None
    for name in ABLATIONS:
        attempt = _latest_attempt(name)
        prediction_path = attempt / "dev_predictions.csv" if attempt else None
        if prediction_path is None or not prediction_path.is_file():
            continue
        frame = pd.read_csv(prediction_path, keep_default_na=False)
        if comparison is None:
            comparison = frame[
                ["source_id", "source_text", "reference"]
            ].copy()
        comparison = comparison.merge(
            frame[["source_id", "prediction"]].rename(
                columns={"prediction": f"prediction_{name}"}
            ),
            on="source_id",
            how="left",
            validate="one_to_one",
        )
    if comparison is None:
        return pd.DataFrame()
    return comparison
