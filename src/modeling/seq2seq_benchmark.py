"""Shared Hugging Face training and inference pipeline for M7."""

from __future__ import annotations

import inspect
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from src.generation.benchmark_baselines import sha256_file
from src.modeling.metrics import generation_metrics, normalization_metrics


DEFAULT_MT5_SMALL_REVISION = "73fb5dbe4756edadc8fbe8c769b0a109493acf7a"


@dataclass(frozen=True)
class Seq2SeqConfig:
    model_name_or_path: str = "google/mt5-small"
    model_revision: str | None = DEFAULT_MT5_SMALL_REVISION
    task_prefix: str = "normalize Wolof: "
    max_source_length: int = 192
    max_target_length: int = 96
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    optim: str = "adafactor"
    use_lora: bool = True
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = ("q", "v")
    num_train_epochs: float = 3.0
    per_device_train_batch_size: int = 24
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    generation_num_beams: int = 1
    checkpoint_selection_metric: str = "loss"
    generate_final_development_metrics: bool = True
    seed: int = 2026
    fp16: bool = False
    bf16: bool = False
    tf32: bool = True
    full_determinism: bool = False
    gradient_checkpointing: bool = False
    save_total_limit: int = 1
    logging_steps: int = 1
    # Generated CER can be much more expensive than the optimization steps for
    # local autoregressive models.  One preserves the historical every-epoch
    # behavior; larger values remain checkpoint selection on generated dev CER.
    generated_eval_interval_epochs: int = 1


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _pretrained_kwargs(model_name_or_path: str, revision: str | None) -> dict[str, str]:
    """Pin Hub models while leaving local fine-tuning checkpoints loadable."""
    if revision and not Path(model_name_or_path).exists():
        return {"revision": revision}
    return {}


def _is_lora_adapter(model_path: Path | str) -> bool:
    return (Path(model_path) / "adapter_config.json").is_file()


def _model_class_for_checkpoint(model_name_or_path: Path | str, fallback):
    """Resolve locally implemented M7 architectures before using AutoModel."""
    from src.modeling.compact_transformer import compact_transformer_model_class
    from src.modeling.tnt_edit_transformer import tnt_edit_model_class

    resolved = compact_transformer_model_class(model_name_or_path, fallback)
    return (
        resolved
        if resolved is not fallback
        else tnt_edit_model_class(model_name_or_path, fallback)
    )


def _load_training_model(config: Seq2SeqConfig, auto_model_class):
    """Load a fresh LoRA model, continue an adapter, or load a full model."""
    if _is_lora_adapter(config.model_name_or_path):
        from peft import PeftConfig, PeftModel

        adapter_path = Path(config.model_name_or_path)
        adapter_config = PeftConfig.from_pretrained(adapter_path)
        base_name = adapter_config.base_model_name_or_path
        revision = getattr(adapter_config, "revision", None) or config.model_revision
        base_model = auto_model_class.from_pretrained(
            base_name, **_pretrained_kwargs(base_name, revision)
        )
        model = PeftModel.from_pretrained(
            base_model, adapter_path, is_trainable=True
        )
        return model, "lora_adapter_continuation", base_name

    model = auto_model_class.from_pretrained(
        config.model_name_or_path,
        **_pretrained_kwargs(config.model_name_or_path, config.model_revision),
    )
    if not config.use_lora:
        return model, "full_finetune", config.model_name_or_path

    from peft import LoraConfig, TaskType, get_peft_model

    adapter_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=list(config.lora_target_modules),
        bias="none",
        revision=config.model_revision,
    )
    return (
        get_peft_model(model, adapter_config),
        "lora_adapter_fresh",
        config.model_name_or_path,
    )


def _load_evaluation_model(checkpoint: Path | str, config: Seq2SeqConfig, auto_model_class):
    if not _is_lora_adapter(checkpoint):
        return auto_model_class.from_pretrained(checkpoint), "full_model"

    from peft import PeftConfig, PeftModel

    adapter_config = PeftConfig.from_pretrained(checkpoint)
    base_name = adapter_config.base_model_name_or_path
    revision = getattr(adapter_config, "revision", None) or config.model_revision
    base_model = auto_model_class.from_pretrained(
        base_name, **_pretrained_kwargs(base_name, revision)
    )
    return PeftModel.from_pretrained(base_model, checkpoint), "lora_adapter"


def _validate_pairs(data: pd.DataFrame, label: str) -> None:
    missing = {"source_id", "source_text", "target_text"} - set(data.columns)
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")
    if data["source_text"].isna().any() or data["target_text"].isna().any():
        raise ValueError(f"{label} contains null text")
    if data["source_id"].duplicated().any():
        raise ValueError(f"{label} contains duplicate source IDs")


def levenshtein_target_annotations(
    source_ids: Sequence[int], target_ids: Sequence[int]
) -> tuple[list[int], list[int]]:
    """Return target edit flags and monotonically aligned source positions.

    Matching characters receive edit flag 0; substitutions and insertions
    receive 1.  A deletion has no target token, so the next emitted token
    (including EOS) is marked as an edit boundary.  Tie breaking is stable:
    diagonal substitution, then deletion, then insertion.
    """
    source = list(source_ids)
    target = list(target_ids)
    source_length, target_length = len(source), len(target)
    costs = [list(range(target_length + 1))]
    backtrace: list[list[str | None]] = [
        [None] + ["insert"] * target_length
    ]
    for source_index in range(1, source_length + 1):
        row = [source_index] + [0] * target_length
        moves: list[str | None] = ["delete"] + [None] * target_length
        for target_index in range(1, target_length + 1):
            if source[source_index - 1] == target[target_index - 1]:
                row[target_index] = costs[source_index - 1][target_index - 1]
                moves[target_index] = "match"
                continue
            candidates = (
                (costs[source_index - 1][target_index - 1] + 1, "substitute"),
                (costs[source_index - 1][target_index] + 1, "delete"),
                (row[target_index - 1] + 1, "insert"),
            )
            row[target_index], moves[target_index] = min(
                candidates, key=lambda candidate: candidate[0]
            )
        costs.append(row)
        backtrace.append(moves)

    operations: list[tuple[str, int | None, int | None]] = []
    source_index, target_index = source_length, target_length
    while source_index or target_index:
        move = backtrace[source_index][target_index]
        if move in {"match", "substitute"}:
            operations.append((move, source_index - 1, target_index - 1))
            source_index -= 1
            target_index -= 1
        elif move == "delete":
            operations.append((move, source_index - 1, None))
            source_index -= 1
        elif move == "insert":
            operations.append((move, None, target_index - 1))
            target_index -= 1
        else:
            raise RuntimeError("Invalid Levenshtein backtrace")
    operations.reverse()

    edit_mask = [0] * target_length
    source_positions = [-1] * target_length
    pending_deletion = False
    for move, aligned_source, aligned_target in operations:
        if move == "delete":
            pending_deletion = True
            continue
        assert aligned_target is not None
        if aligned_source is not None:
            source_positions[aligned_target] = aligned_source
        if move != "match" or pending_deletion:
            edit_mask[aligned_target] = 1
        pending_deletion = False
    # This can only occur when target_ids is empty.  Normal tokenizer outputs
    # include EOS, so ordinary trailing deletions are attached to EOS above.
    return edit_mask, source_positions


def _tokenize_dataset(
    data: pd.DataFrame,
    tokenizer,
    config: Seq2SeqConfig,
    *,
    include_edit_annotations: bool = False,
):
    from datasets import Dataset

    dataset = Dataset.from_pandas(
        data[["source_id", "source_text", "target_text"]], preserve_index=False
    )

    def tokenize(batch):
        inputs = [config.task_prefix + value for value in batch["source_text"]]
        encoded = tokenizer(
            inputs,
            max_length=config.max_source_length,
            truncation=True,
        )
        labels = tokenizer(
            text_target=batch["target_text"],
            max_length=config.max_target_length,
            truncation=True,
        )
        encoded["labels"] = labels["input_ids"]
        if include_edit_annotations:
            annotations = [
                levenshtein_target_annotations(source_ids, target_ids)
                for source_ids, target_ids in zip(
                    encoded["input_ids"], labels["input_ids"]
                )
            ]
            encoded["edit_mask"] = [value[0] for value in annotations]
            encoded["copy_source_positions"] = [value[1] for value in annotations]
        return encoded

    return dataset.map(tokenize, batched=True, desc="Tokenizing M7 pairs")


def _tokenize_tnt_edit_dataset(data: pd.DataFrame, tokenizer, config: Seq2SeqConfig, model):
    """Tokenize pairs and derive method-independent TNT edit supervision."""
    from datasets import Dataset
    from src.modeling.tnt_edit_transformer import tnt_edit_annotations

    dataset = Dataset.from_pandas(
        data[["source_id", "source_text", "target_text"]], preserve_index=False
    )

    def tokenize(batch):
        encoded = tokenizer(
            [config.task_prefix + value for value in batch["source_text"]],
            max_length=config.max_source_length,
            truncation=True,
        )
        targets = tokenizer(
            text_target=batch["target_text"],
            max_length=config.max_target_length,
            truncation=True,
        )
        encoded["labels"] = targets["input_ids"]
        annotations = [
            tnt_edit_annotations(
                source_ids,
                target_ids,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                output_slots_per_source=model.config.output_slots_per_source,
            )
            for source_ids, target_ids in zip(
                encoded["input_ids"], targets["input_ids"]
            )
        ]
        encoded["operation_labels"] = [value[0] for value in annotations]
        encoded["segment_labels"] = [value[1] for value in annotations]
        return encoded

    return dataset.map(tokenize, batched=True, desc="Tokenizing TNT edit pairs")


class EditAwareDataCollator:
    """Pad alignment annotations alongside the standard seq2seq batch."""

    def __init__(self, base_collator, *, padding_side: str = "right"):
        self.base_collator = base_collator
        self.padding_side = padding_side

    def __call__(self, features):
        import torch

        ordinary_features = []
        edit_masks = []
        source_positions = []
        for feature in features:
            feature = dict(feature)
            edit_masks.append(list(feature.pop("edit_mask")))
            source_positions.append(list(feature.pop("copy_source_positions")))
            ordinary_features.append(
                {
                    key: value
                    for key, value in feature.items()
                    if key
                    in {
                        "input_ids",
                        "attention_mask",
                        "token_type_ids",
                        "labels",
                        "decoder_input_ids",
                        "decoder_attention_mask",
                    }
                }
            )
        batch = self.base_collator(ordinary_features)
        target_length = int(batch["labels"].shape[1])

        def pad(values: list[int], fill: int) -> list[int]:
            missing = target_length - len(values)
            if missing < 0:
                raise ValueError("Alignment annotations exceed padded label length")
            padding = [fill] * missing
            return padding + values if self.padding_side == "left" else values + padding

        batch["edit_mask"] = torch.tensor(
            [pad(values, 0) for values in edit_masks], dtype=torch.bool
        )
        batch["copy_source_positions"] = torch.tensor(
            [pad(values, -1) for values in source_positions], dtype=torch.long
        )
        return batch


class TntEditDataCollator:
    """Pad source-anchored operation and output-segment labels."""

    def __init__(self, base_collator, *, output_slots: int, padding_side: str = "right"):
        self.base_collator = base_collator
        self.output_slots = output_slots
        self.padding_side = padding_side

    def __call__(self, features):
        import torch

        ordinary_features = []
        operation_labels = []
        segment_labels = []
        for feature in features:
            feature = dict(feature)
            operation_labels.append(list(feature.pop("operation_labels")))
            segment_labels.append(
                [list(segment) for segment in feature.pop("segment_labels")]
            )
            ordinary_features.append(
                {
                    key: value
                    for key, value in feature.items()
                    if key
                    in {
                        "input_ids",
                        "attention_mask",
                        "token_type_ids",
                        "labels",
                    }
                }
            )
        batch = self.base_collator(ordinary_features)
        source_length = int(batch["input_ids"].shape[1])

        def source_pad(values, fill):
            missing = source_length - len(values)
            if missing < 0:
                raise ValueError("TNT annotations exceed padded source length")
            padding = [fill] * missing
            return padding + values if self.padding_side == "left" else values + padding

        batch["operation_labels"] = torch.tensor(
            [source_pad(values, -100) for values in operation_labels],
            dtype=torch.long,
        )
        empty_segment = [-100] * self.output_slots
        batch["segment_labels"] = torch.tensor(
            [source_pad(values, empty_segment) for values in segment_labels],
            dtype=torch.long,
        )
        return batch


def _decode_predictions(prediction_output, tokenizer) -> tuple[list[str], list[str]]:
    predictions = prediction_output.predictions
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    predictions = _sanitize_token_ids(predictions, tokenizer.pad_token_id)
    labels = _sanitize_token_ids(
        prediction_output.label_ids, tokenizer.pad_token_id
    )
    decoded_predictions = tokenizer.batch_decode(
        predictions, skip_special_tokens=True
    )
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    return [value.strip() for value in decoded_predictions], [value.strip() for value in decoded_labels]


def _sanitize_token_ids(token_ids, pad_token_id: int) -> list[list[int]]:
    """Replace Trainer's negative sequence padding before tokenizer decoding."""
    values = np.asarray(token_ids).copy()
    if values.ndim != 2:
        raise ValueError(
            "Expected generated token IDs with shape (batch, sequence), got "
            f"{values.shape}"
        )
    values[values < 0] = pad_token_id
    return values.astype(np.int64, copy=False).tolist()


def _trainer_metric_function(tokenizer):
    def compute_metrics(evaluation_prediction):
        predictions = evaluation_prediction.predictions
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        predictions = _sanitize_token_ids(predictions, tokenizer.pad_token_id)
        labels = _sanitize_token_ids(
            evaluation_prediction.label_ids, tokenizer.pad_token_id
        )
        decoded_predictions = tokenizer.batch_decode(
            predictions, skip_special_tokens=True
        )
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        return generation_metrics(decoded_predictions, decoded_labels)

    return compute_metrics


def _training_arguments(config: Seq2SeqConfig, output_dir: Path):
    from transformers import Seq2SeqTrainingArguments

    if config.checkpoint_selection_metric not in {"loss", "cer"}:
        raise ValueError(
            "checkpoint_selection_metric must be either 'loss' or 'cer', got "
            f"{config.checkpoint_selection_metric!r}"
        )
    generate_during_epoch_evaluation = config.checkpoint_selection_metric == "cer"
    parameters = inspect.signature(Seq2SeqTrainingArguments).parameters
    values = {
        "output_dir": str(output_dir),
        "do_train": True,
        "do_eval": True,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "optim": config.optim,
        "num_train_epochs": config.num_train_epochs,
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "per_device_eval_batch_size": config.per_device_eval_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "predict_with_generate": generate_during_epoch_evaluation,
        "generation_max_length": config.max_target_length,
        "generation_num_beams": config.generation_num_beams,
        "save_strategy": "epoch",
        "logging_strategy": "steps",
        "logging_steps": config.logging_steps,
        "logging_first_step": True,
        "disable_tqdm": False,
        "log_level": "info",
        "load_best_model_at_end": True,
        "metric_for_best_model": (
            "cer" if generate_during_epoch_evaluation else "eval_loss"
        ),
        "greater_is_better": False,
        "save_total_limit": config.save_total_limit,
        "seed": config.seed,
        "data_seed": config.seed,
        "full_determinism": config.full_determinism,
        "report_to": "none",
        "skip_memory_metrics": False,
        "fp16": config.fp16,
        "bf16": config.bf16,
        "tf32": config.tf32,
        "gradient_checkpointing": config.gradient_checkpointing,
    }
    if "eval_strategy" in parameters:
        values["eval_strategy"] = "epoch"
    else:
        values["evaluation_strategy"] = "epoch"
    # Specialized diagnostic configs may opt into a different schedule without
    # changing the stable base Seq2SeqConfig used by completed M7 runs.
    if "lr_scheduler_type" in parameters and hasattr(config, "lr_scheduler_type"):
        values["lr_scheduler_type"] = getattr(config, "lr_scheduler_type")
    if "warmup_ratio" in parameters and hasattr(config, "warmup_ratio"):
        values["warmup_ratio"] = getattr(config, "warmup_ratio")
    return Seq2SeqTrainingArguments(**values)


def plot_training_history(history: pd.DataFrame, path: Path) -> Path | None:
    """Save per-run loss and development-metric curves when history is available."""
    if history.empty or "step" not in history.columns:
        return None
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    plotted = False
    for column, label in (("loss", "train loss"), ("eval_loss", "dev loss")):
        if column in history.columns:
            values = pd.to_numeric(history[column], errors="coerce")
            mask = values.notna()
            if mask.any():
                axes[0].plot(history.loc[mask, "step"], values.loc[mask], marker="o", label=label)
                plotted = True
    axes[0].set_title("Loss")
    axes[0].set_xlabel("training step")
    axes[0].set_ylabel("loss")
    if plotted:
        axes[0].legend()
    metric_plotted = False
    for column, label in (
        ("eval_cer", "CER"),
        ("eval_wer", "WER"),
        ("eval_correction_f1", "correction F1"),
    ):
        if column in history.columns:
            values = pd.to_numeric(history[column], errors="coerce")
            mask = values.notna()
            if mask.any():
                axes[1].plot(history.loc[mask, "step"], values.loc[mask], marker="o", label=label)
                metric_plotted = True
    axes[1].set_title("Development metrics")
    axes[1].set_xlabel("training step")
    if metric_plotted:
        axes[1].legend()
    else:
        axes[1].text(
            0.5,
            0.5,
            "Generated metrics were not computed per epoch.\n"
            "See run_manifest.json for the final values.",
            ha="center",
            va="center",
            transform=axes[1].transAxes,
        )
        axes[1].set_xticks([])
        axes[1].set_yticks([])
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def train_seq2seq(
    training: pd.DataFrame,
    development: pd.DataFrame,
    *,
    output_dir: Path,
    config: Seq2SeqConfig,
    resume_from_checkpoint: str | None = None,
) -> dict[str, object]:
    """Train one matched M7 condition and evaluate its selected checkpoint on dev."""
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Seq2SeqTrainer,
        TrainerCallback,
    )
    import datasets
    import torch
    import transformers

    _validate_pairs(training, "training data")
    _validate_pairs(development, "development data")
    output_dir = Path(output_dir)
    existing_entries = (
        [path for path in output_dir.iterdir() if path.name != "experiment_input.json"]
        if output_dir.exists()
        else []
    )
    if existing_entries and resume_from_checkpoint is None:
        raise FileExistsError(
            f"Refusing to overwrite non-empty M7 run directory: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    set_global_seed(config.seed)
    print(
        f"[M7] Loading {config.model_name_or_path}"
        + (f" at revision {config.model_revision}" if config.model_revision else ""),
        flush=True,
    )
    pretrained_kwargs = _pretrained_kwargs(
        config.model_name_or_path, config.model_revision
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path, **pretrained_kwargs
    )
    model_class = _model_class_for_checkpoint(
        config.model_name_or_path, AutoModelForSeq2SeqLM
    )
    model, training_mode, base_model_name = _load_training_model(config, model_class)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    print(
        f"[M7] Loaded {parameter_count:,} total parameters; "
        f"training {trainable_parameter_count:,} "
        f"({100 * trainable_parameter_count / parameter_count:.4f}%) via {training_mode}",
        flush=True,
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    print(
        f"[M7] Tokenizing {len(training):,} training and "
        f"{len(development):,} development pairs",
        flush=True,
    )
    is_compact = getattr(model.config, "model_type", None) == "wolof_compact_transformer"
    is_tnt_edit = (
        getattr(model.config, "model_type", None) == "wolof_tnt_edit_transformer"
    )
    include_edit_annotations = is_compact and (
        float(getattr(model.config, "edit_position_weight", 1.0)) != 1.0
        or getattr(model.config, "copy_position_mode", "absolute") == "relative"
    )
    if is_tnt_edit:
        train_dataset = _tokenize_tnt_edit_dataset(training, tokenizer, config, model)
        dev_dataset = _tokenize_tnt_edit_dataset(development, tokenizer, config, model)
    else:
        train_dataset = _tokenize_dataset(
            training,
            tokenizer,
            config,
            include_edit_annotations=include_edit_annotations,
        )
        dev_dataset = _tokenize_dataset(
            development,
            tokenizer,
            config,
            include_edit_annotations=include_edit_annotations,
        )
    base_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    if is_tnt_edit:
        collator = TntEditDataCollator(
            base_collator,
            output_slots=model.config.output_slots_per_source,
            padding_side=tokenizer.padding_side,
        )
    elif include_edit_annotations:
        collator = EditAwareDataCollator(
            base_collator, padding_side=tokenizer.padding_side
        )
    else:
        collator = base_collator
    arguments = _training_arguments(config, output_dir)

    trainer_class = Seq2SeqTrainer
    if is_compact:
        implementation_path = Path(inspect.getfile(model.__class__)).resolve()
        generation_route = (
            "cached incremental"
            if bool(getattr(model.config, "use_incremental_generation", False))
            else "legacy full-prefix"
        )
        print(
            f"[M7] Compact evaluation route: {generation_route}; "
            f"implementation={implementation_path}; "
            f"sha256={sha256_file(implementation_path)[:12]}",
            flush=True,
        )

        class CompactInferenceSeq2SeqTrainer(Seq2SeqTrainer):
            """Make and expose the compact model's actual evaluation path."""

            _compact_evaluation_batch = 0

            def prediction_step(
                self,
                model,
                inputs,
                prediction_loss_only,
                ignore_keys=None,
                **generation_kwargs,
            ):
                timed_generation = bool(
                    self.args.predict_with_generate and not prediction_loss_only
                )
                if timed_generation and torch.cuda.is_available():
                    torch.cuda.synchronize()
                started = time.perf_counter()
                # Custom model generation is not part of optimization.  Force
                # inference mode here as well as on model.generate so Trainer
                # version differences cannot accidentally retain a graph.
                with torch.inference_mode():
                    output = super().prediction_step(
                        model,
                        inputs,
                        prediction_loss_only,
                        ignore_keys=ignore_keys,
                        **generation_kwargs,
                    )
                if timed_generation:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    elapsed = time.perf_counter() - started
                    generated = output[1] if len(output) > 1 else None
                    rows = int(generated.shape[0]) if generated is not None else 0
                    positions = int(generated.shape[1]) if generated is not None else 0
                    self._compact_evaluation_batch += 1
                    print(
                        "[M7] Compact evaluation batch "
                        f"{self._compact_evaluation_batch}: {rows} rows x "
                        f"{positions} positions in {elapsed:.2f}s",
                        flush=True,
                    )
                return output

        trainer_class = CompactInferenceSeq2SeqTrainer

    if is_tnt_edit:
        implementation_path = Path(inspect.getfile(model.__class__)).resolve()
        print(
            "[M7] TNT evaluation route: parallel edit reconstruction; "
            f"implementation={implementation_path}; "
            f"sha256={sha256_file(implementation_path)[:12]}",
            flush=True,
        )

    trainer_parameters = inspect.signature(Seq2SeqTrainer).parameters
    trainer_values = {
        "model": model,
        "args": arguments,
        "train_dataset": train_dataset,
        "eval_dataset": dev_dataset,
        "data_collator": collator,
    }
    if config.checkpoint_selection_metric == "cer":
        trainer_values["compute_metrics"] = _trainer_metric_function(tokenizer)

    live_metrics_path = output_dir / "live_metrics.jsonl"
    live_status_path = output_dir / "live_status.json"

    class LiveMetricsCallback(TrainerCallback):
        def _status(self, state, status: str, **extra) -> None:
            payload = {
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                "status": status,
                "global_step": int(state.global_step),
                "max_steps": int(state.max_steps),
                "epoch": state.epoch,
                **extra,
            }
            live_status_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )

        def on_train_begin(self, args, state, control, **kwargs):
            self._status(state, "training")

        def on_log(self, args, state, control, logs=None, **kwargs):
            payload = {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "global_step": int(state.global_step),
                "max_steps": int(state.max_steps),
                "epoch": state.epoch,
                **(logs or {}),
            }
            with live_metrics_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
            self._status(state, "training", latest_log=logs or {})

        def on_save(self, args, state, control, **kwargs):
            self._status(state, "checkpoint_saved")

        def on_train_end(self, args, state, control, **kwargs):
            self._status(state, "training_complete")

    class GeneratedEvaluationIntervalCallback(TrainerCallback):
        """Keep generated-CER selection while skipping unscheduled epochs."""

        def on_epoch_end(self, args, state, control, **kwargs):
            interval = int(getattr(config, "generated_eval_interval_epochs", 1))
            if interval <= 0:
                raise ValueError("generated_eval_interval_epochs must be positive")
            if config.checkpoint_selection_metric != "cer" or interval == 1:
                return control
            completed_epoch = int(round(float(state.epoch or 0.0)))
            final_epoch = completed_epoch >= math.ceil(float(config.num_train_epochs))
            scheduled = completed_epoch % interval == 0 or final_epoch
            if not scheduled:
                control.should_evaluate = False
                control.should_save = False
            return control

    trainer_values["callbacks"] = [
        LiveMetricsCallback(),
        GeneratedEvaluationIntervalCallback(),
    ]
    if "processing_class" in trainer_parameters:
        trainer_values["processing_class"] = tokenizer
    else:
        trainer_values["tokenizer"] = tokenizer
    trainer = trainer_class(**trainer_values)
    print(
        f"[M7] Training starts: {len(training):,} rows, "
        f"{config.num_train_epochs:g} epochs, physical batch "
        f"{config.per_device_train_batch_size}, accumulation "
        f"{config.gradient_accumulation_steps}",
        flush=True,
    )
    if config.checkpoint_selection_metric == "loss":
        generation_message = (
            "text generation runs once after the best checkpoint is restored"
            if config.generate_final_development_metrics
            else "generated development metrics are skipped for this prerequisite stage"
        )
        print(
            "[M7] Epoch checkpoints are selected by teacher-forced development "
            f"loss; {generation_message}",
            flush=True,
        )
    else:
        evaluation_interval = int(
            getattr(config, "generated_eval_interval_epochs", 1)
        )
        print(
            "[M7] Checkpoints are selected by generated development CER "
            f"every {evaluation_interval} epoch(s)",
            flush=True,
        )
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(str(output_dir / "best_model"))
    tokenizer.save_pretrained(str(output_dir / "best_model"))
    trainer.save_state()
    prediction_path: Path | None = None
    metrics: dict[str, float] = {}
    if config.generate_final_development_metrics:
        print("[M7] Running development prediction and final metrics", flush=True)
        # Epoch evaluation is deliberately loss-only in the fast benchmark policy.
        # Enable autoregressive decoding only for this final development pass.
        trainer.args.predict_with_generate = True
        prediction_output = trainer.predict(
            dev_dataset,
            max_length=config.max_target_length,
            num_beams=config.generation_num_beams,
        )
        predictions, decoded_references = _decode_predictions(
            prediction_output, tokenizer
        )
        metrics = normalization_metrics(
            development["source_text"].tolist(),
            predictions,
            decoded_references,
        )
        prediction_path = output_dir / "dev_predictions.csv"
        pd.DataFrame(
            {
                "source_id": development["source_id"],
                "source_text": development["source_text"],
                "reference": decoded_references,
                "prediction": predictions,
            }
        ).to_csv(prediction_path, index=False, encoding="utf-8")
    else:
        print(
            "[M7] Synthetic prerequisite checkpoint ready; generated development "
            "metrics are deferred until after gold fine-tuning",
            flush=True,
        )
    history_path = output_dir / "training_history.csv"
    history = pd.DataFrame(trainer.state.log_history)
    history.to_csv(history_path, index=False, encoding="utf-8")
    curve_path = plot_training_history(history, output_dir / "training_curves.png")
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": asdict(config),
        "software": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "datasets": datasets.__version__,
            "peft": __import__("peft").__version__ if config.use_lora else None,
        },
        "hardware": {
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "peak_cuda_memory_bytes": (
                int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
            ),
        },
        "model_parameter_count": parameter_count,
        "trainable_parameter_count": trainable_parameter_count,
        "trainable_parameter_ratio": trainable_parameter_count / parameter_count,
        "training_mode": training_mode,
        "base_model_name_or_path": base_model_name,
        "training_rows": len(training),
        "development_rows": len(development),
        "train_metrics": train_result.metrics,
        "development_metrics": metrics,
        "trainer_best_metric": trainer.state.best_metric,
        "best_model_path": str((output_dir / "best_model").resolve()),
        "trainer_best_checkpoint": trainer.state.best_model_checkpoint,
        "training_history_path": str(history_path.resolve()),
        "training_history_sha256": sha256_file(history_path),
        "training_curves_path": str(curve_path.resolve()) if curve_path else None,
        "live_metrics_path": str(live_metrics_path.resolve()),
        "live_status_path": str(live_status_path.resolve()),
        "dev_predictions_path": (
            str(prediction_path.resolve()) if prediction_path is not None else None
        ),
        "dev_predictions_sha256": (
            sha256_file(prediction_path) if prediction_path is not None else None
        ),
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return manifest


def evaluate_checkpoint(
    checkpoint: Path | str,
    evaluation: pd.DataFrame,
    *,
    output_dir: Path,
    config: Seq2SeqConfig,
    split_name: str = "test",
) -> dict[str, object]:
    """Run a frozen checkpoint on one split and save predictions plus metrics."""
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    _validate_pairs(evaluation, f"{split_name} data")
    checkpoint = str(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model_class = _model_class_for_checkpoint(checkpoint, AutoModelForSeq2SeqLM)
    model, checkpoint_type = _load_evaluation_model(
        checkpoint, config, model_class
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    inference_started = time.perf_counter()
    predictions: list[str] = []
    batch_size = config.per_device_eval_batch_size
    with torch.inference_mode():
        for start in range(0, len(evaluation), batch_size):
            texts = [
                config.task_prefix + value
                for value in evaluation["source_text"].iloc[start : start + batch_size]
            ]
            encoded = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=config.max_source_length,
            ).to(device)
            generated = model.generate(
                **encoded,
                max_length=config.max_target_length,
                num_beams=config.generation_num_beams,
            )
            predictions.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - inference_started
    references = evaluation["target_text"].tolist()
    metrics = normalization_metrics(
        evaluation["source_text"].tolist(), predictions, references
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / f"{split_name}_predictions.csv"
    pd.DataFrame(
        {
            "source_id": evaluation["source_id"],
            "source_text": evaluation["source_text"],
            "reference": references,
            "prediction": predictions,
        }
    ).to_csv(prediction_path, index=False, encoding="utf-8")
    result = {
        "checkpoint": str(Path(checkpoint).resolve()),
        "split": split_name,
        "configuration": asdict(config),
        "checkpoint_type": checkpoint_type,
        "model_parameter_count": parameter_count,
        "hardware": {
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "peak_cuda_memory_bytes": (
                int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
            ),
        },
        "inference_seconds": inference_seconds,
        "rows_per_second": len(evaluation) / inference_seconds if inference_seconds else None,
        "metrics": metrics,
        "predictions_path": str(prediction_path.resolve()),
        "predictions_sha256": sha256_file(prediction_path),
    }
    (output_dir / f"{split_name}_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def write_benchmark_summary(records: Sequence[dict[str, object]], path: Path) -> pd.DataFrame:
    rows = []
    for record in records:
        row = {
            "method": record.get("method"),
            "regime": record.get("regime"),
            "seed": record.get("seed"),
            "attempt": record.get("attempt"),
            "checkpoint": record.get("checkpoint"),
            "model_parameter_count": record.get("model_parameter_count"),
            "inference_seconds": record.get("inference_seconds"),
            "rows_per_second": record.get("rows_per_second"),
            "peak_cuda_memory_bytes": record.get("peak_cuda_memory_bytes"),
        }
        row.update(record.get("metrics", {}))
        rows.append(row)
    summary = pd.DataFrame(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(path, index=False, encoding="utf-8")
    return summary
