"""Read-only adapter around the selected historical TNT Hybrid V3 checkpoint."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence


DEFAULT_SELECTION_MANIFEST = Path(
    "results/m7_seq2seq/tnt_hybrid_v3_optimization_v1/selected_candidate.json"
)
PORTABLE_SELECTION_REFERENCE = Path(
    "results/downstream/laya_normalization_v1/tnt_selection_reference.json"
)
DEFAULT_RELATIVE_CHECKPOINT = Path(
    "checkpoints/m7_seq2seq/tnt_hybrid_v3_optimization_v1/m1_v3/seed_2026/"
    "gold_adapt_lr_5e5_word_keep_t050/attempt_001/best_model"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_selected_tnt_checkpoint(
    repository_root: str | Path = ".",
    selection_manifest: str | Path = DEFAULT_SELECTION_MANIFEST,
    *,
    verify_hashes: bool = True,
) -> tuple[Path, dict]:
    """Resolve the locked V3 selection without relying on its old absolute path."""

    root = Path(repository_root).resolve()
    manifest_path = Path(selection_manifest)
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    if not manifest_path.is_file():
        fallback_manifest = root / PORTABLE_SELECTION_REFERENCE
        if selection_manifest == DEFAULT_SELECTION_MANIFEST and fallback_manifest.is_file():
            manifest_path = fallback_manifest
        else:
            raise FileNotFoundError(f"Missing TNT selection manifest: {manifest_path}")
    selection = json.loads(manifest_path.read_text(encoding="utf-8"))

    candidates: list[Path] = []
    if selection.get("checkpoint"):
        candidates.append(Path(str(selection["checkpoint"])))
    if selection.get("checkpoint_relative"):
        candidates.append(root / str(selection["checkpoint_relative"]))
    candidates.append(root / DEFAULT_RELATIVE_CHECKPOINT)
    checkpoint = next((path.resolve() for path in candidates if path.is_dir()), None)
    if checkpoint is None:
        raise FileNotFoundError(
            "The selected TNT checkpoint is not present. Expected the original "
            f"path or {root / DEFAULT_RELATIVE_CHECKPOINT}. Checkpoints are local "
            "artifacts and are not downloaded from GitHub."
        )

    if verify_hashes:
        expected = {
            "model.safetensors": selection.get("checkpoint_model_sha256"),
            "config.json": selection.get("checkpoint_config_sha256"),
        }
        for filename, expected_hash in expected.items():
            path = checkpoint / filename
            if not path.is_file():
                raise FileNotFoundError(f"Selected TNT checkpoint lacks {path}")
            observed = _sha256(path)
            if expected_hash and observed != expected_hash:
                raise ValueError(
                    f"Hash mismatch for {path}: expected {expected_hash}, got {observed}"
                )
    return checkpoint, selection


class TntV3Normalizer:
    """Batch inference wrapper; it never trains or mutates the checkpoint."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        device: str | None = None,
        batch_size: int = 32,
    ) -> None:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        from src.modeling.seq2seq_benchmark import _model_class_for_checkpoint

        self.checkpoint = Path(checkpoint).resolve()
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        model_class = _model_class_for_checkpoint(
            self.checkpoint, AutoModelForSeq2SeqLM
        )
        self.model = model_class.from_pretrained(self.checkpoint)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model.to(self.device).eval()
        self.batch_size = int(batch_size)

    def __call__(self, text: str) -> str:
        return self.normalize([text])[0]

    def normalize(self, texts: Sequence[str]) -> list[str]:
        import torch

        predictions: list[str] = []
        with torch.inference_mode():
            for start in range(0, len(texts), self.batch_size):
                batch = [str(value) for value in texts[start : start + self.batch_size]]
                encoded = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.model.config.max_position_embeddings,
                ).to(self.device)
                generated = self.model.generate(
                    **encoded,
                    max_length=self.model.config.max_position_embeddings,
                    num_beams=1,
                )
                predictions.extend(
                    value.strip()
                    for value in self.tokenizer.batch_decode(
                        generated, skip_special_tokens=True
                    )
                )
        return predictions
