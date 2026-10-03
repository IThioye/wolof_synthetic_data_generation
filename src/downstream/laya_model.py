"""Laya model acquisition, preparation, dry-run, and intent inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .injongo import INJONGO_INTENTS, build_laya_item, collate_laya_items


LAYA_MODEL_ID = "convaiinnovations/laya-multilingual"
LAYA_MODEL_REVISION = "e4e9ddf21a7b1903b7acffd8814ad4307bf63a67"


def snapshot_laya_model(
    model_id: str = LAYA_MODEL_ID,
    *,
    revision: str = LAYA_MODEL_REVISION,
    local_dir: str | Path | None = None,
) -> Path:
    """Download the pinned standalone Laya checkpoint layout."""

    from huggingface_hub import snapshot_download
    from laya.agent import _fix_tokenizer_config

    kwargs = {"repo_id": model_id, "revision": revision}
    if local_dir is not None:
        kwargs["local_dir"] = str(local_dir)
    model_dir = Path(snapshot_download(**kwargs))
    _fix_tokenizer_config(str(model_dir))
    return model_dir


def load_laya_assets(model_dir: str | Path, *, load_weights: bool = True):
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    from laya.common import build_model

    model_dir = Path(model_dir)
    config = json.loads((model_dir / "rl_agent_config.json").read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(model_dir / "tokenizer")
    model = build_model(config, encoder_dir=str(model_dir / "encoder"))
    if load_weights:
        model.load_state_dict(
            load_file(str(model_dir / "model.safetensors")), strict=True
        )
    return tokenizer, config, model


def load_laya_tokenizer_config(model_dir: str | Path):
    """Load preprocessing assets without allocating the 322M-parameter model."""

    from transformers import AutoTokenizer

    model_dir = Path(model_dir)
    config = json.loads((model_dir / "rl_agent_config.json").read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(model_dir / "tokenizer")
    return tokenizer, config


def prepare_laya_items(
    rows: Iterable[Mapping],
    tokenizer,
    config: Mapping,
    options: Sequence[str] = INJONGO_INTENTS,
) -> list[dict]:
    return [build_laya_item(row, tokenizer, config, options) for row in rows]


def mini_forward_dry_run(
    model_dir: str | Path,
    row: Mapping,
    *,
    device: str = "cpu",
) -> dict:
    """Load the pinned model and run one forward pass without optimization."""

    import torch

    tokenizer, config, model = load_laya_assets(model_dir)
    item = build_laya_item(row, tokenizer, config)
    batch = collate_laya_items([item], tokenizer.pad_token_id)
    model.to(device).eval()
    with torch.inference_mode():
        logits, _ = model(
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
            batch["marker_pos"].to(device),
            batch["marker_mask"].to(device),
            batch["qtype"].to(device),
        )
    prediction_index = int(logits[0, : len(INJONGO_INTENTS)].argmax().item())
    return {
        "example_id": item["example_id"],
        "input_shape": list(batch["input_ids"].shape),
        "option_count": len(item["markers"]),
        "option_stats": item["option_stats"],
        "gold": item["intent"],
        "unfinetuned_prediction": INJONGO_INTENTS[prediction_index],
        "forward_ok": True,
    }


class LayaIntentClassifier:
    """Batched inference with the exact preprocessing used for fine-tuning."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        device: str | None = None,
        batch_size: int = 16,
        options: Sequence[str] = INJONGO_INTENTS,
    ) -> None:
        import torch

        self.checkpoint = Path(checkpoint).resolve()
        self.tokenizer, self.config, self.model = load_laya_assets(self.checkpoint)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model.to(self.device).eval()
        self.batch_size = int(batch_size)
        self.options = tuple(options)

    def _choice_temperature(self) -> float:
        temperature = self.config.get("temperature", [1.0, 1.0, 1.0])
        if isinstance(temperature, (int, float)):
            return float(temperature)
        return float(temperature[0]) if temperature else 1.0

    def predict(self, rows: Sequence[Mapping], *, text_field: str = "text") -> list[dict]:
        import torch

        prepared_rows = [
            {
                "example_id": str(row["example_id"]),
                "row_id": str(row.get("row_id", row["example_id"])),
                "text": str(row[text_field]),
                "intent": str(row["intent"]),
                "split": str(row.get("split", "test")),
            }
            for row in rows
        ]
        items = prepare_laya_items(
            prepared_rows, self.tokenizer, self.config, self.options
        )
        outputs: list[dict] = []
        temperature = self._choice_temperature()
        with torch.inference_mode():
            for start in range(0, len(items), self.batch_size):
                chunk = items[start : start + self.batch_size]
                batch = collate_laya_items(chunk, self.tokenizer.pad_token_id)
                logits, _ = self.model(
                    batch["input_ids"].to(self.device),
                    batch["attention_mask"].to(self.device),
                    batch["marker_pos"].to(self.device),
                    batch["marker_mask"].to(self.device),
                    batch["qtype"].to(self.device),
                )
                probabilities = torch.softmax(
                    logits[:, : len(self.options)].float() / temperature, dim=-1
                ).cpu()
                for offset, item in enumerate(chunk):
                    index = int(probabilities[offset].argmax().item())
                    outputs.append(
                        {
                            "example_id": item["example_id"],
                            "row_id": item.get("row_id", item["example_id"]),
                            "gold_intent": item["intent"],
                            "prediction": self.options[index],
                            "confidence": float(probabilities[offset, index].item()),
                            "gold_confidence": float(
                                probabilities[offset, int(item["label"])].item()
                            ),
                        }
                    )
        return outputs
