"""Controlled M1 gate initialized from the unpaired Wolof language encoder.

This experiment is isolated from the completed V3 M1 gate.  Architecture,
synthetic sample, optimization, Gold fine-tuning, and Gold-dev selection stay
unchanged; only the encoder initialization differs.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import pandas as pd
import torch
from transformers import AutoTokenizer

from src.config import (
    ENCODER_PRETRAINING_RESULTS_DIR,
    M7_RESULTS_DIR,
    M7_RUNS_DIR,
)
from src.generation.benchmark_baselines import sha256_file
from src.modeling import compact_benchmark_runner as _base
from src.modeling.metrics import normalization_metrics
from src.modeling.tnt_hybrid_v3_m1_gate import (
    _v3_architecture,
    gate_protocol,
)
from src.modeling.tnt_hybrid_v3_transformer import (
    TntHybridV3TransformerForConditionalGeneration,
    prepare_tnt_hybrid_v3_transformer,
)
from src.modeling.wolof_encoder_pretraining import load_transferable_encoder_state


BENCHMARK_NAME = "tnt_hybrid_v3_language_pretrained_m1_gate_v1"
RUNS_DIR = M7_RUNS_DIR / BENCHMARK_NAME
RESULTS_DIR = M7_RESULTS_DIR / BENCHMARK_NAME
SELECTIONS_DIR = RESULTS_DIR / "selections"
TEST_DIR = RESULTS_DIR / "gold_test_disabled"
CONTROL_RESULTS_DIR = M7_RESULTS_DIR / "tnt_hybrid_v3_m1_gate_v1"
METHOD = "m1"


def _activate_namespace() -> None:
    _base.BENCHMARK_NAME = BENCHMARK_NAME
    _base.RUNS_DIR = RUNS_DIR
    _base.RESULTS_DIR = RESULTS_DIR
    _base.SELECTIONS_DIR = SELECTIONS_DIR
    _base.TEST_DIR = TEST_DIR


def load_transfer_data(protocol=None):
    protocol = protocol or gate_protocol()
    _activate_namespace()
    return _base.prepare_benchmark_data(protocol)


def completed_language_encoder() -> tuple[Path, Path, dict[str, object]]:
    """Return the unique completed, test-audited encoder checkpoint."""
    candidates = []
    for manifest_path in ENCODER_PRETRAINING_RESULTS_DIR.glob(
        "*/training_manifest.json"
    ):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") == "complete" and manifest.get("test_evaluated"):
            candidates.append((manifest_path, manifest))
    if not candidates:
        raise FileNotFoundError(
            "No complete encoder pretraining run with test_evaluated=true was found"
        )
    if len(candidates) > 1:
        raise RuntimeError(
            "Multiple completed encoders exist; select one explicitly before transfer"
        )
    manifest_path, manifest = candidates[0]
    checkpoint = Path(str(manifest["best_checkpoint"]))
    vocabulary_path = checkpoint.parent / "vocabulary.json"
    if not checkpoint.is_file() or not vocabulary_path.is_file():
        raise FileNotFoundError("Encoder checkpoint or vocabulary is missing")
    return checkpoint, vocabulary_path, manifest


def transfer_encoder_state(
    target_model: TntHybridV3TransformerForConditionalGeneration,
    source_state: Mapping[str, torch.Tensor],
    source_vocabulary: Mapping[str, int],
    target_vocabulary: Mapping[str, int],
) -> dict[str, object]:
    """Copy the exact encoder and token-mapped embeddings into TNT V3."""
    position = source_state["position_embedding.weight"]
    if position.shape != target_model.position_embedding.weight.shape:
        raise ValueError(
            "Position embedding mismatch: "
            f"source={tuple(position.shape)}, "
            f"target={tuple(target_model.position_embedding.weight.shape)}"
        )
    encoder_state = {
        key.removeprefix("encoder."): value
        for key, value in source_state.items()
        if key.startswith("encoder.")
    }
    target_encoder_state = target_model.encoder.state_dict()
    if set(encoder_state) != set(target_encoder_state):
        missing = sorted(set(target_encoder_state) - set(encoder_state))
        unexpected = sorted(set(encoder_state) - set(target_encoder_state))
        raise ValueError(
            f"Encoder tensor mismatch; missing={missing}, unexpected={unexpected}"
        )
    for key in encoder_state:
        if encoder_state[key].shape != target_encoder_state[key].shape:
            raise ValueError(f"Encoder tensor shape mismatch for {key}")

    copied_tokens: list[str] = []
    missing_tokens: list[str] = []
    source_embedding = source_state["character_embedding.weight"]
    with torch.no_grad():
        target_model.position_embedding.weight.copy_(position)
        target_model.encoder.load_state_dict(encoder_state, strict=True)
        for token, target_index in sorted(
            target_vocabulary.items(), key=lambda item: item[1]
        ):
            source_index = source_vocabulary.get(token)
            if source_index is None:
                missing_tokens.append(token)
                continue
            target_model.character_embedding.weight[int(target_index)].copy_(
                source_embedding[int(source_index)]
            )
            copied_tokens.append(token)
    exact_parameters = sum(value.numel() for value in encoder_state.values()) + int(
        position.numel()
    )
    embedding_parameters = len(copied_tokens) * target_model.config.d_model
    return {
        "encoder_tensors_copied": len(encoder_state),
        "position_tensors_copied": 1,
        "target_vocabulary_size": len(target_vocabulary),
        "source_vocabulary_size": len(source_vocabulary),
        "embedding_rows_copied": len(copied_tokens),
        "embedding_rows_missing": len(missing_tokens),
        "missing_target_tokens": missing_tokens,
        "exact_encoder_and_position_parameters": exact_parameters,
        "mapped_embedding_parameters": embedding_parameters,
        "total_transferred_parameters": exact_parameters + embedding_parameters,
    }


def _initialization_fingerprint(
    random_base: Path,
    encoder_checkpoint: Path,
    encoder_vocabulary_path: Path,
) -> str:
    payload = {
        "random_base_manifest_sha256": sha256_file(
            random_base / "initialization_manifest.json"
        ),
        "encoder_checkpoint_sha256": sha256_file(encoder_checkpoint),
        "encoder_vocabulary_sha256": sha256_file(encoder_vocabulary_path),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def prepare_transferred_initialization(
    gold: Mapping[str, pd.DataFrame],
    selected: Mapping[str, pd.DataFrame],
    protocol=None,
) -> Path:
    """Build the unchanged V3 model, then replace only its encoder state."""
    protocol = protocol or gate_protocol()
    _activate_namespace()
    architecture, label_profile = _v3_architecture(gold["train"], protocol)
    frames = [gold["train"]] + [selected[method] for method in _base.METHODS]
    random_base = prepare_tnt_hybrid_v3_transformer(
        frames,
        output_root=RUNS_DIR / "shared_random_base",
        seed=protocol.seed,
        architecture=architecture,
    )
    encoder_checkpoint, encoder_vocabulary_path, encoder_manifest = (
        completed_language_encoder()
    )
    fingerprint = _initialization_fingerprint(
        random_base, encoder_checkpoint, encoder_vocabulary_path
    )
    output_dir = RUNS_DIR / "language_pretrained_initialization" / fingerprint[:12]
    manifest_path = output_dir / "initialization_manifest.json"
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("fingerprint") != fingerprint:
            raise RuntimeError("Existing transferred initialization differs")
        print(f"[language transfer] Reusing {output_dir.resolve()}")
        return output_dir

    tokenizer = AutoTokenizer.from_pretrained(random_base)
    model = TntHybridV3TransformerForConditionalGeneration.from_pretrained(random_base)
    source_state = load_transferable_encoder_state(encoder_checkpoint)
    source_vocabulary = json.loads(
        encoder_vocabulary_path.read_text(encoding="utf-8")
    )
    audit = transfer_encoder_state(
        model,
        source_state,
        source_vocabulary,
        tokenizer.get_vocab(),
    )
    if audit["embedding_rows_missing"]:
        raise RuntimeError(
            "Language-pretraining vocabulary does not cover every downstream token: "
            f"{audit['missing_target_tokens']}"
        )
    output_dir.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "fingerprint": fingerprint,
        "benchmark": BENCHMARK_NAME,
        "purpose": "controlled encoder-initialization comparison for M1",
        "random_base": str(random_base.resolve()),
        "random_base_manifest_sha256": sha256_file(
            random_base / "initialization_manifest.json"
        ),
        "language_encoder_checkpoint": str(encoder_checkpoint.resolve()),
        "language_encoder_checkpoint_sha256": sha256_file(encoder_checkpoint),
        "language_encoder_manifest": encoder_manifest,
        "architecture": asdict(architecture),
        "edit_label_profile": label_profile,
        "training_protocol": asdict(protocol),
        "transfer_audit": audit,
        "controlled_difference": (
            "Only character embeddings, positional embeddings, and Transformer "
            "encoder weights are replaced. Normalization heads retain the same "
            "seeded random initialization as the V3 M1 control."
        ),
        "gold_test_policy": "disabled",
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "transfer_protocol.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"[language transfer] Created {output_dir.resolve()} with "
        f"{audit['total_transferred_parameters']:,} transferred parameters"
    )
    return output_dir


def run_language_pretrained_m1(
    gold: Mapping[str, pd.DataFrame],
    selected: Mapping[str, pd.DataFrame],
    initialization: Path,
    protocol=None,
    *,
    rerun: bool = False,
) -> dict[str, object]:
    """Run the unchanged 20-epoch M1 then 20-epoch Gold protocol."""
    protocol = protocol or gate_protocol()
    synthetic_record = run_language_synthetic_stage(
        gold,
        selected,
        initialization,
        protocol,
        rerun=rerun,
    )
    return run_language_gold_stage(
        gold,
        synthetic_record,
        protocol,
        rerun=rerun,
    )


def run_language_synthetic_stage(
    gold: Mapping[str, pd.DataFrame],
    selected: Mapping[str, pd.DataFrame],
    initialization: Path,
    protocol=None,
    *,
    rerun: bool = False,
) -> dict[str, object]:
    """Run or resume only the controlled M1 synthetic stage."""
    protocol = protocol or gate_protocol()
    _activate_namespace()
    configuration = _base._training_config(
        initialization,
        epochs=protocol.synthetic_epochs,
        learning_rate=protocol.synthetic_learning_rate,
        train_batch_size=protocol.synthetic_batch_size,
        protocol=protocol,
    )
    return _base.run_condition(
        selected[METHOD],
        gold["dev"],
        method=METHOD,
        regime="synthetic_pretrain",
        seed=protocol.seed,
        config=configuration,
        runs_dir=RUNS_DIR,
        rerun=rerun,
        resume=True,
        parent_manifest_path=(
            Path(initialization) / "initialization_manifest.json"
        ),
        quality_gate_override=True,
    )


def run_language_gold_stage(
    gold: Mapping[str, pd.DataFrame],
    synthetic_record: Mapping[str, object],
    protocol=None,
    *,
    rerun: bool = False,
) -> dict[str, object]:
    """Run or resume Gold fine-tuning from the completed synthetic stage."""
    protocol = protocol or gate_protocol()
    _activate_namespace()
    return _base._run_continuous_gold_training(
        condition=METHOD,
        initial_model=Path(str(synthetic_record["best_model_path"])),
        parent_manifest=(
            Path(str(synthetic_record["attempt_dir"])) / "run_manifest.json"
        ),
        gold=gold,
        protocol=protocol,
        rerun=rerun,
    )


def control_runtime_reference() -> dict[str, float]:
    """Return previous V3 wall-clock evidence for planning, in minutes."""
    root = (
        M7_RUNS_DIR
        / "tnt_hybrid_v3_m1_gate_v1"
        / METHOD
        / "seed_2026"
    )
    output: dict[str, float] = {}
    for regime in ("synthetic_pretrain", "gold_finetune_continuous"):
        manifests = sorted((root / regime).glob("attempt_*/run_manifest.json"))
        if not manifests:
            continue
        manifest = json.loads(manifests[-1].read_text(encoding="utf-8"))
        runtime = (manifest.get("train_metrics", {}) or {}).get("train_runtime")
        if runtime is not None:
            output[regime] = float(runtime) / 60.0
    output["total"] = sum(output.values())
    return output


def _selection_metrics(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    selection = json.loads(path.read_text(encoding="utf-8"))
    return selection.get("development_metrics", {}) or {}


def development_comparison(gold_dev: pd.DataFrame) -> pd.DataFrame:
    """Compare initialization conditions on Gold-dev without opening Gold-test."""
    identity = normalization_metrics(
        gold_dev["source_text"].tolist(),
        gold_dev["source_text"].tolist(),
        gold_dev["target_text"].tolist(),
    )
    rows: list[dict[str, object]] = [
        {"condition": "identity", "initialization": "none", **identity}
    ]
    control = _selection_metrics(CONTROL_RESULTS_DIR / "selections" / "m1.json")
    if control is not None:
        rows.append(
            {
                "condition": "m1_v3_random_encoder",
                "initialization": "seeded random",
                **control,
            }
        )
    transferred = _selection_metrics(SELECTIONS_DIR / "m1.json")
    if transferred is not None:
        rows.append(
            {
                "condition": "m1_v3_language_pretrained_encoder",
                "initialization": "unpaired formal+informal MLM",
                **transferred,
            }
        )
    output = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output.to_csv(
        RESULTS_DIR / "development_comparison.csv", index=False, encoding="utf-8"
    )
    return output
