"""Fine-tune Laya multilingual on official REAL InjongoIntent Wolof train.

This is a local adaptation of Laya's official typed-decisions DDP notebook.
It deliberately uses every official train example for optimization and only
official dev for checkpoint selection and post-selection temperature fitting.
Official test is not accepted by this entry point.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np

from .injongo import INJONGO_INTENTS, collate_laya_items
from .metrics import classification_metrics


def fit_choice_temperature(logits_and_targets: list[tuple[np.ndarray, list[float]]]) -> float:
    import torch
    from laya.common import TEMP_MAX, TEMP_MIN

    if len(logits_and_targets) < 10:
        return 1.0
    option_count = max(len(logits) for logits, _ in logits_and_targets)
    logits = torch.full((len(logits_and_targets), option_count), -1e4)
    targets = torch.zeros((len(logits_and_targets), option_count))
    for index, (row_logits, row_target) in enumerate(logits_and_targets):
        logits[index, : len(row_logits)] = torch.tensor(row_logits)
        targets[index, : len(row_target)] = torch.tensor(row_target)
    log_temperature = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=100)

    def closure():
        optimizer.zero_grad()
        loss = -(
            targets * torch.log_softmax(logits / log_temperature.exp(), dim=-1)
        ).sum(dim=-1).mean()
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.clamp(log_temperature.exp(), TEMP_MIN, TEMP_MAX).item())


def _save_model(model, tokenizer, config: dict, output_dir: Path, metadata: dict) -> None:
    from safetensors.torch import save_file

    output_dir.mkdir(parents=True, exist_ok=True)
    state = {
        key: value.half().contiguous().cpu()
        for key, value in model.state_dict().items()
    }
    save_file(state, str(output_dir / "model.safetensors"))
    model.encoder.config.save_pretrained(output_dir / "encoder")
    tokenizer.save_pretrained(output_dir / "tokenizer")
    (output_dir / "rl_agent_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "checkpoint_meta.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def _evaluate(model, items, tokenizer, device, batch_size: int, autocast_dtype):
    import torch

    model.eval()
    gold: list[str] = []
    predicted: list[str] = []
    logits_and_targets: list[tuple[np.ndarray, list[float]]] = []
    total_loss = 0.0
    total_rows = 0
    with torch.inference_mode():
        for start in range(0, len(items), batch_size):
            chunk = items[start : start + batch_size]
            batch = collate_laya_items(chunk, tokenizer.pad_token_id)
            autocast_enabled = device.type == "cuda" and autocast_dtype is not None
            with torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=autocast_enabled,
            ):
                logits, _ = model(
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                    batch["marker_pos"].to(device),
                    batch["marker_mask"].to(device),
                    batch["qtype"].to(device),
                )
            logits = logits[:, : len(INJONGO_INTENTS)].float()
            labels = batch["label"].to(device)
            loss = torch.nn.functional.cross_entropy(logits, labels, reduction="sum")
            total_loss += float(loss.item())
            total_rows += len(chunk)
            indices = logits.argmax(dim=-1).cpu().tolist()
            for row_index, item in enumerate(chunk):
                gold.append(item["intent"])
                predicted.append(INJONGO_INTENTS[int(indices[row_index])])
                logits_and_targets.append(
                    (
                        logits[row_index].cpu().numpy(),
                        list(item["target"]),
                    )
                )
    metrics, _, _ = classification_metrics(gold, predicted, INJONGO_INTENTS)
    metrics["cross_entropy"] = total_loss / max(1, total_rows)
    model.train()
    return metrics, logits_and_targets


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--train-items", required=True)
    parser.add_argument("--dev-items", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--micro-batch", type=int, default=8)
    parser.add_argument("--eval-batch", type=int, default=16)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--encoder-lr", type=float, default=2.5e-5)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--sigma-start", type=float, default=0.4)
    parser.add_argument("--sigma-end", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--precision", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--disable-gradient-checkpointing", action="store_true")
    return parser.parse_args()


def main() -> None:
    import torch
    import torch.distributed as dist
    from safetensors.torch import load_file
    from torch.nn.parallel import DistributedDataParallel as DDP
    from transformers import AutoTokenizer
    from laya.common import build_model, proper_reward

    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Full Laya fine-tuning requires a CUDA GPU. The notebook has a separate CPU dry-run."
        )

    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if distributed:
        backend = "nccl" if dist.is_nccl_available() else "gloo"
        dist.init_process_group(backend)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)

    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    config = json.loads((model_dir / "rl_agent_config.json").read_text(encoding="utf-8"))
    # Same capacity choices as the official notebook. These values are also
    # used during preprocessing and are persisted with the final checkpoint.
    config["gradient_checkpointing"] = not args.disable_gradient_checkpointing
    config["max_tokens_per_batch"] = 4096
    config["max_len"] = 1024
    config["head_max_len"] = 256

    tokenizer = AutoTokenizer.from_pretrained(model_dir / "tokenizer")
    model = build_model(config, encoder_dir=str(model_dir / "encoder"))
    model.load_state_dict(
        load_file(str(model_dir / "model.safetensors")), strict=True
    )
    if not args.disable_gradient_checkpointing:
        model.encoder.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.head_checkpointing = True
    model.to(device).train()
    wrapped = (
        DDP(model, device_ids=[local_rank], find_unused_parameters=True)
        if distributed
        else model
    )

    train_items = torch.load(args.train_items, weights_only=False)
    dev_items = torch.load(args.dev_items, weights_only=False)
    if not train_items or {item["split"] for item in train_items} != {"train"}:
        raise ValueError("train-items must contain official train only")
    if not dev_items or {item["split"] for item in dev_items} != {"dev"}:
        raise ValueError("dev-items must contain official dev only")

    # No held-out train subset: official dev is the only selection/calibration
    # resource by protocol. Refuse a DDP topology that would require dropping
    # rows merely to equalize ranks.
    if len(train_items) % world_size:
        raise ValueError(
            f"{len(train_items)} train rows are not divisible by {world_size} GPUs; "
            "choose a GPU count that preserves every official train example"
        )
    usable = train_items
    rank_items = usable[rank::world_size]

    encoder_parameters = [
        parameter for name, parameter in wrapped.named_parameters() if "encoder." in name
    ]
    head_parameters = [
        parameter for name, parameter in wrapped.named_parameters() if "encoder." not in name
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_parameters, "lr": args.encoder_lr},
            {"params": head_parameters, "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
    )
    batches_per_epoch = math.ceil(len(rank_items) / args.micro_batch)
    updates_per_epoch = math.ceil(batches_per_epoch / args.gradient_accumulation)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, updates_per_epoch * args.epochs),
        eta_min=1e-6,
    )
    autocast_dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": None,
    }[args.precision]
    scaler = torch.amp.GradScaler("cuda", enabled=args.precision == "fp16")

    if rank == 0:
        print(
            f"Laya Injongo training: {len(usable)} REAL train, {len(dev_items)} dev, "
            f"{world_size} GPU(s), {args.epochs} epochs",
            flush=True,
        )
    history: list[dict] = []
    best_macro_f1 = -1.0
    best_epoch = None
    started = time.time()

    for epoch in range(args.epochs):
        epoch_items = list(rank_items)
        random.Random(args.seed + epoch + rank).shuffle(epoch_items)
        progress = epoch / max(1, args.epochs - 1)
        sigma = args.sigma_start + (args.sigma_end - args.sigma_start) * progress
        optimizer.zero_grad(set_to_none=True)
        accumulated = 0
        epoch_loss = 0.0
        batch_count = 0

        for start in range(0, len(epoch_items), args.micro_batch):
            chunk = epoch_items[start : start + args.micro_batch]
            batch = collate_laya_items(chunk, tokenizer.pad_token_id)
            with torch.autocast(
                device_type="cuda",
                dtype=autocast_dtype,
                enabled=autocast_dtype is not None,
            ):
                logits, activation = wrapped(
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                    batch["marker_pos"].to(device),
                    batch["marker_mask"].to(device),
                    batch["qtype"].to(device),
                )
            logits = logits.float()
            mask = batch["marker_mask"].to(device)
            count = mask.sum(dim=-1, keepdim=True).float()
            target = batch["target"].to(device)
            noise = torch.randn(
                (args.group_size,) + logits.shape, device=device
            ) * sigma * mask
            noise = (noise - noise.sum(dim=-1, keepdim=True) / count) * mask
            sampled_logits = logits.detach().unsqueeze(0) + noise
            sampled_probabilities = torch.softmax(
                sampled_logits.masked_fill(~mask, -1e4), dim=-1
            )
            with torch.no_grad():
                reward = proper_reward(
                    sampled_probabilities,
                    target.unsqueeze(0),
                    batch["qtype"].to(device),
                    mask,
                    w_sph=0.75,
                    w_rps=1.0,
                )
                advantage = reward - reward.mean(dim=0, keepdim=True)
                advantage = advantage / (advantage.std() + 1e-6)
            log_probability = -(
                ((sampled_logits - logits.unsqueeze(0)) ** 2) * mask
            ).sum(dim=-1) / (2 * sigma**2)
            reinforcement_loss = -(advantage * log_probability).mean()
            cross_entropy = -(
                target
                * torch.log_softmax(logits.masked_fill(~mask, -1e4), dim=-1)
            ).sum(dim=-1).mean()
            loss = (
                reinforcement_loss + cross_entropy
            ) / args.gradient_accumulation + 0.0 * activation.sum()
            scaler.scale(loss).backward()
            accumulated += 1
            last_batch = start + args.micro_batch >= len(epoch_items)
            if accumulated % args.gradient_accumulation == 0 or last_batch:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(wrapped.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            epoch_loss += float(loss.item()) * args.gradient_accumulation
            batch_count += 1
            if rank == 0 and (batch_count == 1 or batch_count % 25 == 0):
                print(
                    f"epoch {epoch+1}/{args.epochs} batch {batch_count}/{batches_per_epoch} "
                    f"loss={float(loss.item())*args.gradient_accumulation:.4f} "
                    f"reward={float(reward.mean().item()):.3f}",
                    flush=True,
                )

        if distributed:
            dist.barrier()
        if rank == 0:
            dev_metrics, _ = _evaluate(
                model, dev_items, tokenizer, device, args.eval_batch, autocast_dtype
            )
            epoch_record = {
                "epoch": epoch + 1,
                "train_loss_rank0": epoch_loss / max(1, batch_count),
                "sigma": sigma,
                "dev": dev_metrics,
                "elapsed_seconds": time.time() - started,
            }
            history.append(epoch_record)
            rolling_config = dict(config)
            _save_model(
                model,
                tokenizer,
                rolling_config,
                output_dir / "checkpoint_latest",
                epoch_record,
            )
            if dev_metrics["macro_f1"] > best_macro_f1:
                best_macro_f1 = float(dev_metrics["macro_f1"])
                best_epoch = epoch + 1
                _save_model(
                    model,
                    tokenizer,
                    rolling_config,
                    output_dir / "checkpoint_best",
                    epoch_record,
                )
            (output_dir / "training_history.json").write_text(
                json.dumps(history, indent=2) + "\n", encoding="utf-8"
            )
            print(
                f"epoch {epoch+1} dev accuracy={dev_metrics['accuracy']:.4f} "
                f"macro-F1={dev_metrics['macro_f1']:.4f} "
                f"CE={dev_metrics['cross_entropy']:.4f}",
                flush=True,
            )
        if distributed:
            dist.barrier()

    if rank == 0:
        # Calibrate the checkpoint selected by dev macro-F1, not the final epoch.
        model.load_state_dict(
            load_file(str(output_dir / "checkpoint_best" / "model.safetensors")),
            strict=True,
        )
        best_dev_metrics, dev_logits = _evaluate(
            model, dev_items, tokenizer, device, args.eval_batch, autocast_dtype
        )
        choice_temperature = fit_choice_temperature(dev_logits)
        final_config = dict(config)
        inherited_temperature = final_config.get("temperature", [1.2, 1.2, 1.2])
        if not isinstance(inherited_temperature, list) or len(inherited_temperature) < 3:
            inherited_temperature = [1.2, 1.2, 1.2]
        final_config["temperature"] = [
            choice_temperature,
            float(inherited_temperature[1]),
            float(inherited_temperature[2]),
        ]
        final_config.pop("temperature_by_options", None)
        final_config["fine_tuned"] = True
        final_config["model_name"] = "laya-multilingual-injongo-wolof-intent"
        manifest = {
            "experiment": "laya_downstream_normalization_v1",
            "training_data": "official InjongoIntent/wol train only",
            "selection_data": "official InjongoIntent/wol dev only",
            "test_used": False,
            "selection_metric": "dev_macro_f1",
            "best_epoch": best_epoch,
            "best_dev_metrics": best_dev_metrics,
            "choice_temperature_fitted_on_dev": choice_temperature,
            "train_rows": len(train_items),
            "dev_rows": len(dev_items),
            "intent_count": len(INJONGO_INTENTS),
            "seed": args.seed,
            "hyperparameters": vars(args),
        }
        _save_model(model, tokenizer, final_config, output_dir, manifest)
        (output_dir / "training_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"Saved selected and calibrated model to {output_dir}; "
            "official test was not loaded.",
            flush=True,
        )
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
