"""Self-supervised character-encoder pretraining for the TNT V3 normalizer.

The model mirrors TNT V3's transferable character embeddings, learned
positions, and three-layer Transformer encoder. A tied masked-character head
exists only during language pretraining. Normalization targets are not used by
the objective.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm.auto import tqdm

from src.config import (
    ENCODER_PRETRAINING_RESULTS_DIR,
    ENCODER_PRETRAINING_RUNS_DIR,
    GOLD_SPLITS_DIR,
    M1_BEQI_RULES_PATH,
    PRETRAINING_DEV_PATH,
    PRETRAINING_MANIFEST_PATH,
    PRETRAINING_TEST_PATH,
    PRETRAINING_TRAIN_PATH,
)
from src.generation.benchmark_baselines import sha256_file


SPECIAL_TOKENS = ("<pad>", "<s>", "</s>", "<unk>", "<mask>")


@dataclass(frozen=True)
class EncoderPretrainingConfig:
    seed: int = 2026
    d_model: int = 192
    attention_heads: int = 4
    encoder_layers: int = 3
    feed_forward_size: int = 768
    dropout: float = 0.1
    max_position_embeddings: int = 256
    mask_probability: float = 0.15
    min_mask_span: int = 1
    max_mask_span: int = 3
    train_epochs: int = 5
    train_batch_size: int = 256
    evaluation_batch_size: int = 512
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    gradient_clip_norm: float = 1.0
    formal_sampling_fraction: float = 0.25
    mixed_precision: str = "bf16"
    num_workers: int = 0

    def validate(self) -> None:
        if self.d_model % self.attention_heads:
            raise ValueError("d_model must be divisible by attention_heads")
        if not 0 < self.mask_probability < 1:
            raise ValueError("mask_probability must be between zero and one")
        if not 0 < self.formal_sampling_fraction < 1:
            raise ValueError("formal_sampling_fraction must be between zero and one")
        if self.min_mask_span < 1 or self.max_mask_span < self.min_mask_span:
            raise ValueError("Invalid mask span range")
        if self.train_epochs < 1 or self.train_batch_size < 1:
            raise ValueError("Epoch and batch counts must be positive")
        if self.mixed_precision not in {"none", "fp16", "bf16"}:
            raise ValueError("mixed_precision must be none, fp16, or bf16")


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_pretraining_splits() -> dict[str, pd.DataFrame]:
    paths = {
        "train": PRETRAINING_TRAIN_PATH,
        "dev": PRETRAINING_DEV_PATH,
        "test": PRETRAINING_TEST_PATH,
    }
    data = {name: pd.read_parquet(path) for name, path in paths.items()}
    for name, frame in data.items():
        missing = {"source_id", "domain", "text"} - set(frame.columns)
        if missing:
            raise ValueError(f"{name} split is missing columns: {sorted(missing)}")
        if frame["text"].isna().any() or frame["source_id"].duplicated().any():
            raise ValueError(f"{name} split contains null text or duplicate IDs")
    return data


def downstream_vocabulary_texts() -> list[str]:
    """Reserve IDs for training-only downstream characters without training on them."""
    texts: list[str] = []
    if M1_BEQI_RULES_PATH.is_file():
        m1 = pd.read_csv(M1_BEQI_RULES_PATH)
        for column in ("informal_wolof", "formal_wolof"):
            if column in m1:
                texts.extend(m1[column].dropna().astype(str).tolist())
    gold_path = GOLD_SPLITS_DIR / "gold_train.csv"
    if gold_path.is_file():
        gold = pd.read_csv(gold_path)
        for column in ("comment", "manual_formal_wolof"):
            if column in gold:
                texts.extend(gold[column].dropna().astype(str).tolist())
    return texts


def build_pretraining_vocabulary(
    training_texts: Iterable[str],
    reserved_downstream_texts: Iterable[str] = (),
) -> dict[str, int]:
    characters = sorted(
        {
            character
            for text in [*training_texts, *reserved_downstream_texts]
            for character in str(text)
        }
    )
    vocabulary = {token: index for index, token in enumerate(SPECIAL_TOKENS)}
    for character in characters:
        if character not in vocabulary:
            vocabulary[character] = len(vocabulary)
    return vocabulary


class CharacterCorpusDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        vocabulary: dict[str, int],
        max_length: int,
    ):
        self.source_ids = frame["source_id"].astype(str).tolist()
        self.domains = frame["domain"].astype(str).tolist()
        unknown = vocabulary["<unk>"]
        eos = vocabulary["</s>"]
        self.rows = []
        for text in frame["text"].astype(str):
            encoded = [vocabulary.get(character, unknown) for character in text]
            self.rows.append(encoded[: max_length - 1] + [eos])

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        return {
            "input_ids": self.rows[index],
            "domain": self.domains[index],
            "source_id": self.source_ids[index],
        }


class MaskedCharacterCollator:
    """Apply deterministic span corruption and pad one character batch."""

    def __init__(
        self,
        vocabulary: dict[str, int],
        *,
        mask_probability: float,
        min_span: int,
        max_span: int,
        seed: int,
    ):
        self.vocabulary = vocabulary
        self.mask_probability = mask_probability
        self.min_span = min_span
        self.max_span = max_span
        self.generator = np.random.default_rng(seed)
        self.pad_id = vocabulary["<pad>"]
        self.eos_id = vocabulary["</s>"]
        self.mask_id = vocabulary["<mask>"]
        special_ids = {vocabulary[token] for token in SPECIAL_TOKENS}
        self.random_character_ids = np.asarray(
            sorted(set(vocabulary.values()) - special_ids), dtype=np.int64
        )

    def _mask_positions(self, length: int) -> np.ndarray:
        eligible_length = max(length - 1, 0)
        selected = np.zeros(length, dtype=bool)
        if not eligible_length:
            return selected
        target = max(1, int(round(eligible_length * self.mask_probability)))
        attempts = 0
        while selected[:eligible_length].sum() < target and attempts < target * 20:
            start = int(self.generator.integers(0, eligible_length))
            span = int(self.generator.integers(self.min_span, self.max_span + 1))
            selected[start : min(start + span, eligible_length)] = True
            attempts += 1
        if selected[:eligible_length].sum() > target:
            active = np.flatnonzero(selected[:eligible_length])
            remove = self.generator.choice(
                active, size=int(len(active) - target), replace=False
            )
            selected[remove] = False
        return selected

    def __call__(self, rows: Sequence[dict[str, object]]) -> dict[str, object]:
        maximum = max(len(row["input_ids"]) for row in rows)
        inputs = torch.full((len(rows), maximum), self.pad_id, dtype=torch.long)
        labels = torch.full_like(inputs, -100)
        attention = torch.zeros_like(inputs)
        source_ids: list[str] = []
        domains: list[str] = []
        for row_index, row in enumerate(rows):
            values = np.asarray(row["input_ids"], dtype=np.int64)
            length = len(values)
            masked = self._mask_positions(length)
            corrupted = values.copy()
            active = np.flatnonzero(masked)
            decisions = self.generator.random(len(active))
            corrupted[active[decisions < 0.8]] = self.mask_id
            random_positions = active[(decisions >= 0.8) & (decisions < 0.9)]
            if len(random_positions):
                corrupted[random_positions] = self.generator.choice(
                    self.random_character_ids, size=len(random_positions), replace=True
                )
            inputs[row_index, :length] = torch.from_numpy(corrupted)
            labels[row_index, active] = torch.from_numpy(values[active])
            attention[row_index, :length] = 1
            source_ids.append(str(row["source_id"]))
            domains.append(str(row["domain"]))
        return {
            "input_ids": inputs,
            "attention_mask": attention,
            "labels": labels,
            "source_ids": source_ids,
            "domains": domains,
        }


class MaskedCharacterEncoder(nn.Module):
    """TNT-compatible encoder with a tied temporary masked-character head."""

    def __init__(self, vocabulary_size: int, config: EncoderPretrainingConfig):
        super().__init__()
        self.config = config
        self.vocabulary_size = int(vocabulary_size)
        self.character_embedding = nn.Embedding(
            vocabulary_size, config.d_model, padding_idx=0
        )
        self.position_embedding = nn.Embedding(
            config.max_position_embeddings, config.d_model
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.attention_heads,
            dim_feedforward=config.feed_forward_size,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.encoder_layers)
        self.mlm_bias = nn.Parameter(torch.zeros(vocabulary_size))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)
            else:
                nn.init.zeros_(parameter)
        for module in self.modules():
            if isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        with torch.no_grad():
            self.character_embedding.weight[0].zero_()

    def encode(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        if input_ids.shape[1] > self.config.max_position_embeddings:
            raise ValueError("Input exceeds encoder positional limit")
        positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        hidden = self.character_embedding(input_ids) + self.position_embedding(positions)
        return self.encoder(hidden, src_key_padding_mask=attention_mask.eq(0))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        hidden = self.encode(input_ids, attention_mask)
        logits = functional.linear(hidden, self.character_embedding.weight, self.mlm_bias)
        output = {"logits": logits}
        if labels is not None:
            output["loss"] = functional.cross_entropy(
                logits.reshape(-1, self.vocabulary_size),
                labels.reshape(-1),
                ignore_index=-100,
            )
        return output


def parameter_report(model: MaskedCharacterEncoder) -> dict[str, int | float]:
    groups = {
        "character_embedding": model.character_embedding,
        "position_embedding": model.position_embedding,
        "encoder": model.encoder,
    }
    report = {
        name: sum(parameter.numel() for parameter in module.parameters())
        for name, module in groups.items()
    }
    report["transferable_parameters"] = sum(report.values())
    report["pretraining_parameters"] = sum(
        parameter.numel() for parameter in model.parameters()
    )
    report["fp32_weight_megabytes"] = report["pretraining_parameters"] * 4 / 2**20
    return report


def domain_sampling_weights(
    dataset: CharacterCorpusDataset, formal_fraction: float
) -> torch.Tensor:
    counts = pd.Series(dataset.domains).value_counts().to_dict()
    formal = counts.get("formal_news", 0)
    informal = counts.get("informal_youtube", 0)
    if not formal or not informal:
        raise ValueError("Training data must contain both pretraining domains")
    return torch.tensor(
        [
            formal_fraction / formal
            if domain == "formal_news"
            else (1.0 - formal_fraction) / informal
            for domain in dataset.domains
        ],
        dtype=torch.double,
    )


def _autocast_context(config: EncoderPretrainingConfig, device: torch.device):
    enabled = device.type == "cuda" and config.mixed_precision != "none"
    dtype = torch.bfloat16 if config.mixed_precision == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)


def _loader(
    dataset: CharacterCorpusDataset,
    vocabulary: dict[str, int],
    config: EncoderPretrainingConfig,
    *,
    training: bool,
    epoch: int,
) -> DataLoader:
    collator = MaskedCharacterCollator(
        vocabulary,
        mask_probability=config.mask_probability,
        min_span=config.min_mask_span,
        max_span=config.max_mask_span,
        seed=config.seed + (epoch * 10_007 if training else 500_000),
    )
    sampler = None
    if training:
        generator = torch.Generator().manual_seed(config.seed + epoch)
        sampler = WeightedRandomSampler(
            domain_sampling_weights(dataset, config.formal_sampling_fraction),
            num_samples=len(dataset),
            replacement=True,
            generator=generator,
        )
    return DataLoader(
        dataset,
        batch_size=(
            config.train_batch_size if training else config.evaluation_batch_size
        ),
        sampler=sampler,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collator,
    )


def _batch_metrics(output: dict[str, torch.Tensor], labels: torch.Tensor):
    active = labels.ne(-100)
    count = int(active.sum().item())
    correct = int((output["logits"].argmax(dim=-1)[active] == labels[active]).sum().item())
    return count, correct


@torch.inference_mode()
def evaluate_encoder(
    model: MaskedCharacterEncoder,
    dataset: CharacterCorpusDataset,
    vocabulary: dict[str, int],
    config: EncoderPretrainingConfig,
    device: torch.device,
    *,
    epoch: int,
    description: str,
) -> dict[str, float | int]:
    model.eval()
    loss_sum = 0.0
    masked_count = 0
    correct = 0
    loader = _loader(dataset, vocabulary, config, training=False, epoch=epoch)
    for batch in tqdm(loader, desc=description, leave=False):
        labels = batch["labels"].to(device, non_blocking=True)
        with _autocast_context(config, device):
            output = model(
                batch["input_ids"].to(device, non_blocking=True),
                batch["attention_mask"].to(device, non_blocking=True),
                labels,
            )
        count, batch_correct = _batch_metrics(output, labels)
        loss_sum += float(output["loss"].item()) * count
        masked_count += count
        correct += batch_correct
    return {
        "loss": loss_sum / max(masked_count, 1),
        "masked_accuracy": correct / max(masked_count, 1),
        "masked_characters": masked_count,
        "rows": len(dataset),
    }


def _fingerprint(
    config: EncoderPretrainingConfig,
    vocabulary: dict[str, int],
) -> str:
    corpus_manifest = json.loads(PRETRAINING_MANIFEST_PATH.read_text(encoding="utf-8"))
    split_hashes = {
        split: corpus_manifest["outputs"][split]["sha256"]
        for split in ("train", "dev", "test")
    }
    payload = {
        "config": asdict(config),
        "vocabulary": vocabulary,
        "corpus_split_sha256": split_hashes,
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def prepare_pretraining(
    config: EncoderPretrainingConfig | None = None,
) -> dict[str, object]:
    config = config or EncoderPretrainingConfig()
    config.validate()
    splits = load_pretraining_splits()
    vocabulary = build_pretraining_vocabulary(
        splits["train"]["text"].astype(str), downstream_vocabulary_texts()
    )
    datasets = {
        name: CharacterCorpusDataset(frame, vocabulary, config.max_position_embeddings)
        for name, frame in splits.items()
    }
    set_reproducible_seed(config.seed)
    model = MaskedCharacterEncoder(len(vocabulary), config)
    fingerprint = _fingerprint(config, vocabulary)
    run_dir = ENCODER_PRETRAINING_RUNS_DIR / fingerprint[:12]
    results_dir = ENCODER_PRETRAINING_RESULTS_DIR / fingerprint[:12]
    return {
        "config": config,
        "splits": splits,
        "vocabulary": vocabulary,
        "datasets": datasets,
        "model": model,
        "fingerprint": fingerprint,
        "run_dir": run_dir,
        "results_dir": results_dir,
        "parameter_report": parameter_report(model),
    }


def benchmark_pretraining_speed(
    prepared: dict[str, object], batches: int = 20
) -> dict[str, float | int | str]:
    """Measure throwaway steps without saving or changing the real run."""
    config: EncoderPretrainingConfig = prepared["config"]
    vocabulary: dict[str, int] = prepared["vocabulary"]
    dataset: CharacterCorpusDataset = prepared["datasets"]["train"]
    set_reproducible_seed(config.seed)
    model = MaskedCharacterEncoder(len(vocabulary), config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).train()
    try:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate, fused=device.type == "cuda"
        )
    except TypeError:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    loader = _loader(dataset, vocabulary, config, training=True, epoch=0)
    measured: list[float] = []
    for index, batch in enumerate(
        tqdm(loader, total=min(batches + 2, len(loader)), desc="Speed test")
    ):
        if index >= batches + 2:
            break
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        labels = batch["labels"].to(device, non_blocking=True)
        with _autocast_context(config, device):
            loss = model(
                batch["input_ids"].to(device, non_blocking=True),
                batch["attention_mask"].to(device, non_blocking=True),
                labels,
            )["loss"]
        loss.backward()
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        if index >= 2:
            measured.append(elapsed)
    seconds_per_step = float(np.median(measured))
    steps_per_epoch = math.ceil(len(dataset) / config.train_batch_size)
    training_seconds = seconds_per_step * steps_per_epoch * config.train_epochs
    return {
        "device": str(device),
        "batches_measured": len(measured),
        "seconds_per_step": seconds_per_step,
        "steps_per_epoch": steps_per_epoch,
        "estimated_training_minutes": training_seconds / 60,
        "estimated_total_minutes": training_seconds * 1.25 / 60,
    }


def _save_checkpoint(
    path: Path,
    model: MaskedCharacterEncoder,
    optimizer: torch.optim.Optimizer,
    scheduler,
    *,
    epoch: int,
    history: list[dict[str, object]],
    best_dev_loss: float,
    fingerprint: str,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "history": history,
            "best_dev_loss": best_dev_loss,
            "fingerprint": fingerprint,
        },
        path,
    )


def _plot_history(history: pd.DataFrame, path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history["epoch"], history["train_loss"], marker="o", label="train")
    axes[0].plot(history["epoch"], history["dev_loss"], marker="o", label="dev")
    axes[0].set(title="Masked-character loss", xlabel="Epoch", ylabel="Loss")
    axes[0].legend()
    axes[1].plot(
        history["epoch"], history["train_masked_accuracy"], marker="o", label="train"
    )
    axes[1].plot(
        history["epoch"], history["dev_masked_accuracy"], marker="o", label="dev"
    )
    axes[1].set(title="Masked-character accuracy", xlabel="Epoch", ylabel="Accuracy")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def train_encoder(
    prepared: dict[str, object], *, resume: bool = True
) -> dict[str, object]:
    config: EncoderPretrainingConfig = prepared["config"]
    vocabulary: dict[str, int] = prepared["vocabulary"]
    datasets: dict[str, CharacterCorpusDataset] = prepared["datasets"]
    model: MaskedCharacterEncoder = prepared["model"]
    fingerprint = str(prepared["fingerprint"])
    run_dir = Path(prepared["run_dir"])
    results_dir = Path(prepared["results_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "vocabulary.json").write_text(
        json.dumps(vocabulary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (run_dir / "config.json").write_text(
        json.dumps(asdict(config), indent=2) + "\n", encoding="utf-8"
    )

    set_reproducible_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    model.to(device)
    try:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            fused=device.type == "cuda",
        )
    except TypeError:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
    steps_per_epoch = math.ceil(len(datasets["train"]) / config.train_batch_size)
    total_steps = steps_per_epoch * config.train_epochs
    warmup_steps = max(1, int(total_steps * config.warmup_ratio))

    def schedule(step: int) -> float:
        if step < warmup_steps:
            return max(step / warmup_steps, 1e-3)
        return max((total_steps - step) / max(total_steps - warmup_steps, 1), 0.0)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and config.mixed_precision == "fp16"
    )
    last_path = run_dir / "last_checkpoint.pt"
    best_path = run_dir / "best_encoder.pt"
    history: list[dict[str, object]] = []
    start_epoch = 1
    best_dev_loss = float("inf")
    if resume and last_path.is_file():
        state = torch.load(last_path, map_location="cpu", weights_only=False)
        if state.get("fingerprint") != fingerprint:
            raise RuntimeError("Existing encoder checkpoint has a different fingerprint")
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        history = list(state.get("history", []))
        best_dev_loss = float(state.get("best_dev_loss", float("inf")))
        start_epoch = int(state["epoch"]) + 1
        print(f"Resuming encoder pretraining at epoch {start_epoch}")
    if start_epoch > config.train_epochs:
        print("Encoder pretraining is already complete.")
        return {
            "run_dir": run_dir,
            "results_dir": results_dir,
            "best_checkpoint": best_path,
            "history": pd.DataFrame(history),
            "device": str(device),
        }

    print(
        f"Device={device} | parameters={parameter_report(model)['pretraining_parameters']:,} "
        f"| train rows={len(datasets['train']):,} | steps/epoch={steps_per_epoch:,}",
        flush=True,
    )
    for epoch in range(start_epoch, config.train_epochs + 1):
        model.train()
        loader = _loader(datasets["train"], vocabulary, config, training=True, epoch=epoch)
        loss_sum = 0.0
        masked_count = 0
        correct = 0
        progress = tqdm(loader, desc=f"Epoch {epoch}/{config.train_epochs}")
        for batch in progress:
            optimizer.zero_grad(set_to_none=True)
            labels = batch["labels"].to(device, non_blocking=True)
            with _autocast_context(config, device):
                output = model(
                    batch["input_ids"].to(device, non_blocking=True),
                    batch["attention_mask"].to(device, non_blocking=True),
                    labels,
                )
            scaler.scale(output["loss"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            count, batch_correct = _batch_metrics(output, labels)
            loss_sum += float(output["loss"].item()) * count
            masked_count += count
            correct += batch_correct
            progress.set_postfix(
                loss=f"{loss_sum / max(masked_count, 1):.4f}",
                accuracy=f"{correct / max(masked_count, 1):.3f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )
        dev = evaluate_encoder(
            model,
            datasets["dev"],
            vocabulary,
            config,
            device,
            epoch=epoch,
            description=f"Dev epoch {epoch}",
        )
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / max(masked_count, 1),
            "train_masked_accuracy": correct / max(masked_count, 1),
            "dev_loss": dev["loss"],
            "dev_masked_accuracy": dev["masked_accuracy"],
            "learning_rate": scheduler.get_last_lr()[0],
        }
        history.append(row)
        improved = float(dev["loss"]) < best_dev_loss
        if improved:
            best_dev_loss = float(dev["loss"])
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "dev_metrics": dev,
                    "fingerprint": fingerprint,
                },
                best_path,
            )
        _save_checkpoint(
            last_path,
            model,
            optimizer,
            scheduler,
            epoch=epoch,
            history=history,
            best_dev_loss=best_dev_loss,
            fingerprint=fingerprint,
        )
        history_frame = pd.DataFrame(history)
        history_frame.to_csv(results_dir / "training_history.csv", index=False)
        _plot_history(history_frame, results_dir / "training_curves.png")
        print(
            f"Epoch {epoch}: train loss={row['train_loss']:.4f}, "
            f"dev loss={row['dev_loss']:.4f}, dev accuracy={row['dev_masked_accuracy']:.3f}"
            + (" [best]" if improved else ""),
            flush=True,
        )

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "fingerprint": fingerprint,
        "objective": (
            f"{config.mask_probability:.0%} character span masking with "
            "80/10/10 replacement policy"
        ),
        "test_evaluated": False,
        "configuration": asdict(config),
        "parameters": parameter_report(model),
        "vocabulary_size": len(vocabulary),
        "train_rows": len(datasets["train"]),
        "dev_rows": len(datasets["dev"]),
        "best_dev_loss": best_dev_loss,
        "best_checkpoint": str(best_path.resolve()),
        "corpus_manifest": str(PRETRAINING_MANIFEST_PATH.resolve()),
        "corpus_manifest_sha256": sha256_file(PRETRAINING_MANIFEST_PATH),
    }
    (results_dir / "training_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "run_dir": run_dir,
        "results_dir": results_dir,
        "best_checkpoint": best_path,
        "history": pd.DataFrame(history),
        "device": str(device),
    }


def evaluate_pretraining_test(prepared: dict[str, object]) -> dict[str, object]:
    """Evaluate the selected encoder once on the pretraining test split."""
    config: EncoderPretrainingConfig = prepared["config"]
    vocabulary: dict[str, int] = prepared["vocabulary"]
    datasets: dict[str, CharacterCorpusDataset] = prepared["datasets"]
    run_dir = Path(prepared["run_dir"])
    results_dir = Path(prepared["results_dir"])
    checkpoint_path = run_dir / "best_encoder.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError("Train the encoder before evaluating its test split")
    model = MaskedCharacterEncoder(len(vocabulary), config)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    metrics = evaluate_encoder(
        model,
        datasets["test"],
        vocabulary,
        config,
        device,
        epoch=int(state["epoch"]),
        description="Pretraining test",
    )
    output = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selected_epoch": int(state["epoch"]),
        "checkpoint": str(checkpoint_path.resolve()),
        **metrics,
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "test_metrics.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )
    manifest_path = results_dir / "training_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["test_evaluated"] = True
        manifest["test_metrics"] = metrics
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return output


def load_transferable_encoder_state(checkpoint: Path | str) -> dict[str, torch.Tensor]:
    """Return only state with an exact counterpart in TNT V3."""
    state = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
    model_state = state["model_state_dict"]
    prefixes = ("character_embedding.", "position_embedding.", "encoder.")
    return {key: value for key, value in model_state.items() if key.startswith(prefixes)}
