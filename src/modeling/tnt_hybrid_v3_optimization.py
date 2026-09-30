"""Post-gate optimization utilities for the pretrained TNT hybrid V3 model.

The expensive M1 stage is treated as immutable. Fast inference ablations vary
only reconstruction controls, while optional short Gold adaptations start from
the same completed M1 checkpoint in separate experiment directories.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from src.config import M7_RESULTS_DIR, M7_RUNS_DIR
from src.generation.benchmark_baselines import sha256_file
from src.modeling.benchmark_data import load_locked_gold
from src.modeling.compact_benchmark_runner import CompactBenchmarkTrainingConfig
from src.modeling.experiment_runner import run_condition
from src.modeling.metrics import normalization_metrics
from src.modeling.seq2seq_benchmark import Seq2SeqConfig, evaluate_checkpoint
from src.modeling.tnt_hybrid_v3_transformer import (
    TntHybridV3TransformerForConditionalGeneration,
)


EXPERIMENT_NAME = "tnt_hybrid_v3_optimization_v1"
RUNS_DIR = M7_RUNS_DIR / EXPERIMENT_NAME
RESULTS_DIR = M7_RESULTS_DIR / EXPERIMENT_NAME
INFERENCE_DIR = RESULTS_DIR / "inference_ablation"
GATE_RUNS_DIR = M7_RUNS_DIR / "tnt_hybrid_v3_m1_gate_v1"
SELECTED_ADAPTATION_MANIFEST = (
    RUNS_DIR
    / "m1_v3"
    / "seed_2026"
    / "gold_adapt_lr_5e5_word_keep_t050"
    / "attempt_001"
    / "run_manifest.json"
)
FROZEN_TEST_DIR = RESULTS_DIR / "frozen_test"


@dataclass(frozen=True)
class InferenceVariant:
    name: str
    use_boundary_inference: bool = True
    boundary_confidence_threshold: float = 0.50
    use_word_keep_gate: bool = True
    word_keep_threshold: float = 0.70
    use_hard_lexical_protection: bool = True

    def validate(self) -> None:
        if not self.name or any(character in self.name for character in "\\/:"):
            raise ValueError("Inference variant needs a filesystem-safe name")
        for field in ("boundary_confidence_threshold", "word_keep_threshold"):
            value = float(getattr(self, field))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field} must be in [0, 1]")


INFERENCE_VARIANTS: tuple[InferenceVariant, ...] = (
    InferenceVariant("default"),
    InferenceVariant("no_boundary", use_boundary_inference=False),
    InferenceVariant("boundary_t070", boundary_confidence_threshold=0.70),
    InferenceVariant("boundary_t080", boundary_confidence_threshold=0.80),
    InferenceVariant("no_word_gate", use_word_keep_gate=False),
    InferenceVariant("word_keep_t050", word_keep_threshold=0.50),
    InferenceVariant("word_keep_t085", word_keep_threshold=0.85),
    InferenceVariant(
        "no_boundary_word_t050",
        use_boundary_inference=False,
        word_keep_threshold=0.50,
    ),
)


def load_optimization_data() -> dict[str, pd.DataFrame]:
    return load_locked_gold()


def completed_m1_pretraining() -> tuple[Path, Path]:
    """Return the completed M1 model and its immutable run manifest."""
    latest_path = (
        GATE_RUNS_DIR
        / "m1"
        / "seed_2026"
        / "synthetic_pretrain"
        / "latest.json"
    )
    if not latest_path.is_file():
        raise FileNotFoundError("The completed V3 M1 pretraining run is unavailable")
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    manifest_path = Path(str(latest["run_manifest"]))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("experiment_status") != "complete":
        raise RuntimeError("The V3 M1 pretraining manifest is not complete")
    checkpoint = Path(str(manifest["best_model_path"]))
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)
    return checkpoint, manifest_path


def _apply_inference_variant(model, variant: InferenceVariant) -> None:
    variant.validate()
    for field, value in asdict(variant).items():
        if field != "name":
            setattr(model.config, field, value)


def _generate_predictions(
    model,
    tokenizer,
    evaluation: pd.DataFrame,
    *,
    batch_size: int,
) -> tuple[list[str], float]:
    import torch

    device = next(model.parameters()).device
    predictions: list[str] = []
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(evaluation), batch_size):
            encoded = tokenizer(
                evaluation["source_text"].iloc[start : start + batch_size].tolist(),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=model.config.max_position_embeddings,
            ).to(device)
            generated = model.generate(
                **encoded,
                max_length=model.config.max_position_embeddings,
                num_beams=1,
            )
            predictions.extend(
                value.strip()
                for value in tokenizer.batch_decode(
                    generated, skip_special_tokens=True
                )
            )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return predictions, time.perf_counter() - started


def run_inference_ablation(
    gold_dev: pd.DataFrame,
    *,
    checkpoint: Path | str | None = None,
    variants: Sequence[InferenceVariant] = INFERENCE_VARIANTS,
    batch_size: int = 64,
    output_dir: Path | str | None = None,
) -> pd.DataFrame:
    """Evaluate reconstruction controls without changing trained weights."""
    import torch
    from transformers import AutoTokenizer

    if checkpoint is None:
        checkpoint, _ = completed_m1_pretraining()
    checkpoint = Path(checkpoint)
    output_dir = Path(output_dir) if output_dir is not None else INFERENCE_DIR
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = TntHybridV3TransformerForConditionalGeneration.from_pretrained(checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    output_dir.mkdir(parents=True, exist_ok=True)

    for variant in variants:
        variant.validate()
        if variant.name in seen:
            raise ValueError(f"Duplicate inference variant: {variant.name}")
        seen.add(variant.name)
        _apply_inference_variant(model, variant)
        predictions, seconds = _generate_predictions(
            model, tokenizer, gold_dev, batch_size=batch_size
        )
        metrics = normalization_metrics(
            gold_dev["source_text"].tolist(),
            predictions,
            gold_dev["target_text"].tolist(),
        )
        prediction_path = output_dir / f"{variant.name}_predictions.csv"
        pd.DataFrame(
            {
                "source_id": gold_dev["source_id"],
                "source_text": gold_dev["source_text"],
                "reference": gold_dev["target_text"],
                "prediction": predictions,
            }
        ).to_csv(prediction_path, index=False, encoding="utf-8")
        rows.append(
            {
                "variant": variant.name,
                **asdict(variant),
                **metrics,
                "changed_sentence_rows": sum(
                    prediction != source
                    for prediction, source in zip(
                        predictions, gold_dev["source_text"]
                    )
                ),
                "inference_seconds": seconds,
                "predictions_path": str(prediction_path.resolve()),
            }
        )
        print(
            f"[V3 ablation] {variant.name}: CER={metrics['cer']:.4f}, "
            f"WER={metrics['wer']:.4f}, F1={metrics['correction_f1']:.4f}",
            flush=True,
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(
        output_dir / "summary.csv", index=False, encoding="utf-8"
    )
    (output_dir / "protocol.json").write_text(
        json.dumps(
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_config_sha256": sha256_file(checkpoint / "config.json"),
                "split": "Gold-dev",
                "gold_test_used": False,
                "variants": [asdict(variant) for variant in variants],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return summary


def inference_variant(name: str) -> InferenceVariant:
    matches = [variant for variant in INFERENCE_VARIANTS if variant.name == name]
    if not matches:
        raise ValueError(
            f"Unknown inference variant {name!r}; choose from "
            f"{[variant.name for variant in INFERENCE_VARIANTS]}"
        )
    return matches[0]


def prepare_adaptation_initialization(
    checkpoint: Path | str,
    variant: InferenceVariant,
) -> Path:
    """Copy M1 weights with one explicit inference configuration."""
    from transformers import AutoTokenizer

    checkpoint = Path(checkpoint)
    variant.validate()
    fingerprint_payload = {
        "source_config_sha256": sha256_file(checkpoint / "config.json"),
        "variant": asdict(variant),
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    output_dir = RUNS_DIR / "adaptation_initializations" / f"{variant.name}_{fingerprint}"
    manifest_path = output_dir / "initialization_manifest.json"
    if manifest_path.is_file():
        return output_dir

    model = TntHybridV3TransformerForConditionalGeneration.from_pretrained(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    _apply_inference_variant(model, variant)
    output_dir.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    manifest_path.write_text(
        json.dumps(
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "source_checkpoint": str(checkpoint.resolve()),
                "source_config_sha256": fingerprint_payload[
                    "source_config_sha256"
                ],
                "inference_variant": asdict(variant),
                "purpose": "short Gold adaptation from completed M1 pretraining",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return output_dir


def run_short_gold_adaptation(
    name: str,
    gold: Mapping[str, pd.DataFrame],
    *,
    inference_variant_name: str,
    learning_rate: float,
    epochs: float = 5.0,
    rerun: bool = False,
) -> dict[str, object]:
    """Reuse M1 weights and run one isolated, WER-selected Gold adaptation."""
    if learning_rate <= 0 or epochs <= 0:
        raise ValueError("learning_rate and epochs must be positive")
    checkpoint, _ = completed_m1_pretraining()
    variant = inference_variant(inference_variant_name)
    initialization = prepare_adaptation_initialization(checkpoint, variant)
    configuration = CompactBenchmarkTrainingConfig(
        model_name_or_path=str(initialization.resolve()),
        model_revision=None,
        task_prefix="",
        max_source_length=256,
        max_target_length=256,
        learning_rate=learning_rate,
        weight_decay=0.01,
        optim="adamw_torch_fused",
        use_lora=False,
        num_train_epochs=epochs,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        gradient_accumulation_steps=1,
        generation_num_beams=1,
        checkpoint_selection_metric="wer",
        generate_final_development_metrics=True,
        seed=2026,
        fp16=False,
        bf16=False,
        tf32=True,
        full_determinism=False,
        gradient_checkpointing=False,
        save_total_limit=1,
        logging_steps=1,
        generated_eval_interval_epochs=1,
        lr_scheduler_type="constant_with_warmup",
        warmup_ratio=0.05,
    )
    safe_name = name.strip().replace(" ", "_")
    if not safe_name or any(character in safe_name for character in "\\/:"):
        raise ValueError("Adaptation name must be filesystem-safe")
    return run_condition(
        gold["train"],
        gold["dev"],
        method="m1_v3",
        regime=f"gold_adapt_{safe_name}_{variant.name}",
        seed=2026,
        config=configuration,
        runs_dir=RUNS_DIR,
        rerun=rerun,
        resume=True,
        parent_manifest_path=initialization / "initialization_manifest.json",
        quality_gate_override=True,
    )


def gold_adaptation_summary() -> pd.DataFrame:
    """Collect the immutable M1 control and completed short adaptations."""
    checkpoint, pretraining_manifest = completed_m1_pretraining()
    pretraining = json.loads(pretraining_manifest.read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = [
        {
            "condition": "m1_pretrained_no_gold_adaptation",
            "learning_rate": 0.0,
            "epochs": 0.0,
            **(pretraining.get("development_metrics", {}) or {}),
            "checkpoint": str(checkpoint.resolve()),
        }
    ]
    root = RUNS_DIR / "m1_v3" / "seed_2026"
    if root.is_dir():
        for manifest_path in sorted(root.glob("*/attempt_*/run_manifest.json")):
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            configuration = manifest.get("configuration", {}) or {}
            rows.append(
                {
                    "condition": manifest.get("regime"),
                    "learning_rate": configuration.get("learning_rate"),
                    "epochs": configuration.get("num_train_epochs"),
                    **(manifest.get("development_metrics", {}) or {}),
                    "checkpoint": manifest.get("best_model_path"),
                }
            )
    summary = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(
        RESULTS_DIR / "gold_adaptation_summary.csv",
        index=False,
        encoding="utf-8",
    )
    return summary


def lock_selected_candidate() -> dict[str, object]:
    """Freeze the post-PFE V3 choice before reading Gold-test."""
    if not SELECTED_ADAPTATION_MANIFEST.is_file():
        raise FileNotFoundError(SELECTED_ADAPTATION_MANIFEST)
    run_manifest = json.loads(
        SELECTED_ADAPTATION_MANIFEST.read_text(encoding="utf-8")
    )
    checkpoint = Path(str(run_manifest["best_model_path"]))
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)
    core = {
        "experiment": EXPERIMENT_NAME,
        "candidate": "m1_v3_gold_adapt_lr_5e5_word_keep_t050",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_model_sha256": sha256_file(checkpoint / "model.safetensors"),
        "checkpoint_config_sha256": sha256_file(checkpoint / "config.json"),
        "run_manifest": str(SELECTED_ADAPTATION_MANIFEST.resolve()),
        "run_manifest_sha256": sha256_file(SELECTED_ADAPTATION_MANIFEST),
        "selection_split": "Gold-dev",
        "selection_metric": "WER",
        "selected_epoch": 3,
        "m1_pretraining_epochs": 20,
        "gold_adaptation_epochs_run": 5,
        "gold_adaptation_learning_rate": 5e-5,
        "word_keep_threshold": 0.50,
        "development_metrics": run_manifest.get("development_metrics", {}),
        "test_policy": "one-time post-selection Gold-test evaluation",
    }
    canonical = json.dumps(core, ensure_ascii=False, sort_keys=True)
    payload = {
        **core,
        "selection_fingerprint": hashlib.sha256(
            canonical.encode("utf-8")
        ).hexdigest(),
    }
    path = RESULTS_DIR / "selected_candidate.json"
    if path.is_file():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous.get("selection_fingerprint") != payload["selection_fingerprint"]:
            raise RuntimeError(
                "A different V3 optimization candidate was already locked"
            )
        return previous
    payload["locked_at_utc"] = datetime.now(timezone.utc).isoformat()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


def evaluate_locked_candidate_test(*, confirm: bool = False) -> pd.DataFrame:
    """Evaluate the immutable V3 candidate once on the held-out Gold-test."""
    if not confirm:
        raise RuntimeError(
            "Gold-test remains locked. Pass confirm=True only after explicit approval."
        )
    selection = lock_selected_candidate()
    FROZEN_TEST_DIR.mkdir(parents=True, exist_ok=True)
    freeze_path = FROZEN_TEST_DIR / "frozen_test_manifest.json"
    summary_path = FROZEN_TEST_DIR / "test_comparison.csv"
    if freeze_path.is_file():
        previous = json.loads(freeze_path.read_text(encoding="utf-8"))
        if previous.get("selection_fingerprint") != selection[
            "selection_fingerprint"
        ]:
            raise RuntimeError(
                "Gold-test was already opened for a different V3 candidate"
            )
        if summary_path.is_file():
            print(f"[V3 optimization] Reusing frozen test: {summary_path.resolve()}")
            return pd.read_csv(summary_path, keep_default_na=False)

    freeze_payload = {
        "authorized_at_utc": datetime.now(timezone.utc).isoformat(),
        "authorization": "explicit project-owner confirmation",
        "selection_fingerprint": selection["selection_fingerprint"],
        "selected_candidate": selection,
        "gold_test_opened": True,
    }
    freeze_path.write_text(
        json.dumps(freeze_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    gold = load_locked_gold()
    test = gold["test"]
    checkpoint = Path(str(selection["checkpoint"]))
    evaluation = evaluate_checkpoint(
        checkpoint,
        test,
        output_dir=FROZEN_TEST_DIR / "m1_v3_optimized",
        config=Seq2SeqConfig(
            model_name_or_path=str(checkpoint),
            model_revision=None,
            task_prefix="",
            max_source_length=256,
            max_target_length=256,
            use_lora=False,
            per_device_eval_batch_size=64,
            generation_num_beams=1,
        ),
        split_name="test",
    )
    identity = normalization_metrics(
        test["source_text"].tolist(),
        test["source_text"].tolist(),
        test["target_text"].tolist(),
    )
    rows = [
        {
            "benchmark": EXPERIMENT_NAME,
            "condition": "m0_identity",
            "kind": "reference_baseline",
            **identity,
        },
        {
            "benchmark": EXPERIMENT_NAME,
            "condition": "m1_v3_optimized",
            "kind": "m1_pretrained_then_short_gold_adaptation",
            **evaluation["metrics"],
            "checkpoint": evaluation["checkpoint"],
            "predictions_path": evaluation["predictions_path"],
            "inference_seconds": evaluation["inference_seconds"],
        },
    ]
    previous_summary_path = (
        M7_RESULTS_DIR
        / "tnt_edit_gold_dev_v4"
        / "frozen_test"
        / "test_summary.csv"
    )
    if previous_summary_path.is_file():
        previous_summary = pd.read_csv(
            previous_summary_path, keep_default_na=False
        )
        previous_summary.insert(0, "benchmark", "tnt_edit_gold_dev_v4")
        summary = pd.concat(
            [previous_summary, pd.DataFrame(rows[1:])],
            ignore_index=True,
            sort=False,
        )
    else:
        summary = pd.DataFrame(rows)
    summary.to_csv(summary_path, index=False, encoding="utf-8")
    freeze_payload.update(
        {
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "test_rows": len(test),
            "test_metrics": evaluation["metrics"],
            "test_predictions_sha256": evaluation["predictions_sha256"],
            "comparison_path": str(summary_path.resolve()),
            "comparison_sha256": sha256_file(summary_path),
        }
    )
    freeze_path.write_text(
        json.dumps(freeze_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary
