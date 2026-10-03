"""Feature extraction for the frozen TNT normalization-routing audit.

The module deliberately separates features available at inference time from
oracle quantities that compare a synthetic input with its clean REAL source.
No model is trained or modified here.
"""

from __future__ import annotations

import math
import re
import unicodedata
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from src.modeling.metrics import evaluation_tokens, normalize_for_evaluation
from src.modeling.tnt_edit_transformer import KEEP
from src.modeling.tnt_hybrid_transformer import (
    BOUNDARY_DELETE,
    BOUNDARY_INSERT_BEFORE,
    BOUNDARY_KEEP,
    BOUNDARY_NONE,
    WORD_KEEP,
)


DEPLOYABLE_SURFACE_FEATURES = (
    "sentence_char_length",
    "sentence_token_count",
    "average_token_length",
    "formal_vocabulary_fraction",
    "oov_fraction",
    "diacritic_rate",
    "repeated_character_rate",
    "character_set_anomaly_rate",
    "multiple_space_count",
    "punctuation_spacing_anomaly_count",
)

DEPLOYABLE_TNT_FEATURES = (
    "tnt_output_char_edits",
    "tnt_output_char_edit_rate",
    "tnt_output_token_edits",
    "tnt_output_token_edit_rate",
    "tnt_output_chrf",
    "tnt_output_changed",
    "tnt_operation_mean_keep_probability",
    "tnt_operation_mean_change_probability",
    "tnt_operation_max_change_probability",
    "tnt_operation_fraction_change_above_050",
    "tnt_operation_predicted_non_keep_fraction",
    "tnt_operation_mean_entropy",
    "tnt_word_mean_keep_probability",
    "tnt_word_mean_change_probability",
    "tnt_word_max_change_probability",
    "tnt_word_fraction_change_above_030",
    "tnt_word_fraction_change_above_050",
    "tnt_word_predicted_edit_fraction",
    "tnt_word_mean_entropy",
    "tnt_boundary_mean_no_change_probability",
    "tnt_boundary_mean_change_probability",
    "tnt_boundary_max_change_probability",
    "tnt_boundary_fraction_change_above_050",
    "tnt_boundary_predicted_change_fraction",
    "tnt_boundary_mean_entropy",
    "tnt_lexically_protected_word_fraction",
)

ORACLE_FEATURES = (
    "synthetic_operation_count",
    "operations_per_token",
    "operations_per_character",
    "oracle_char_edits_from_real",
    "oracle_character_corruption_rate",
    "oracle_token_edits_from_real",
    "oracle_token_corruption_rate",
    "oracle_chrf_to_real",
)


def load_formal_vocabulary(path: str | Path) -> set[str]:
    """Load the existing project vocabulary without introducing a resource."""

    vocabulary_path = Path(path)
    if not vocabulary_path.is_file():
        raise FileNotFoundError(f"Missing existing formal vocabulary: {vocabulary_path}")
    return {
        normalize_for_evaluation(line).casefold()
        for line in vocabulary_path.read_text(encoding="utf-8").splitlines()
        if normalize_for_evaluation(line)
    }


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _diacritic_character(character: str) -> bool:
    decomposed = unicodedata.normalize("NFD", character)
    return any(unicodedata.combining(value) for value in decomposed[1:])


def surface_features(text: str, formal_vocabulary: set[str]) -> dict[str, float | int]:
    normalized = normalize_for_evaluation(text)
    tokens = evaluation_tokens(normalized)
    letters = [character for character in normalized if character.isalpha()]
    known = sum(token.casefold() in formal_vocabulary for token in tokens)
    repeated = sum(
        max(0, len(match.group(0)) - 1)
        for match in re.finditer(r"(.)\1+", normalized.casefold())
    )
    # Wolof orthography, ordinary Latin code-switching, digits, whitespace and
    # Unicode punctuation are accepted. Control/symbol characters are anomalous.
    anomalous = sum(
        unicodedata.category(character).startswith(("C", "S"))
        for character in normalized
    )
    punctuation_spacing = len(re.findall(r"\s+[,.!?;:]", normalized))
    return {
        "sentence_char_length": len(normalized),
        "sentence_token_count": len(tokens),
        "average_token_length": _ratio(sum(map(len, tokens)), len(tokens)),
        "formal_vocabulary_fraction": _ratio(known, len(tokens)),
        "oov_fraction": _ratio(len(tokens) - known, len(tokens)),
        "diacritic_rate": _ratio(sum(_diacritic_character(c) for c in letters), len(letters)),
        "repeated_character_rate": _ratio(repeated, len(letters)),
        "character_set_anomaly_rate": _ratio(anomalous, len(normalized)),
        "multiple_space_count": len(re.findall(r" {2,}", str(text))),
        "punctuation_spacing_anomaly_count": punctuation_spacing,
    }


def binary_entropy(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-12, 1.0)
    return -(clipped * np.log(clipped)).sum(axis=-1)


class TntSignalExtractor:
    """Read frozen V3 heads and aggregate their softmax scores per sentence.

    This is intentionally an inference-side adapter. It does not alter the TNT
    model class or checkpoint and uses the same encoder/projection modules as
    ``generate``.
    """

    def __init__(self, checkpoint: str | Path, *, device: str | None = None, batch_size: int = 32):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        from src.modeling.seq2seq_benchmark import _model_class_for_checkpoint

        self.torch = torch
        self.checkpoint = Path(checkpoint).resolve()
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        model_class = _model_class_for_checkpoint(self.checkpoint, AutoModelForSeq2SeqLM)
        self.model = model_class.from_pretrained(self.checkpoint)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device).eval()
        self.batch_size = int(batch_size)

    @staticmethod
    def _aggregate(probabilities: np.ndarray, change: np.ndarray, predicted_change: np.ndarray, prefix: str) -> dict[str, float]:
        if len(change) == 0:
            return {
                f"{prefix}_mean_change_probability": 0.0,
                f"{prefix}_max_change_probability": 0.0,
                f"{prefix}_fraction_change_above_050": 0.0,
                f"{prefix}_predicted_change_fraction": 0.0,
                f"{prefix}_mean_entropy": 0.0,
            }
        return {
            f"{prefix}_mean_change_probability": float(change.mean()),
            f"{prefix}_max_change_probability": float(change.max()),
            f"{prefix}_fraction_change_above_050": float((change >= 0.5).mean()),
            f"{prefix}_predicted_change_fraction": float(predicted_change.mean()),
            f"{prefix}_mean_entropy": float(binary_entropy(probabilities).mean()),
        }

    def extract(self, texts: Sequence[str]) -> list[dict[str, float]]:
        torch = self.torch
        records: list[dict[str, float]] = []
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
                input_ids = encoded["input_ids"]
                attention_mask = encoded.get("attention_mask")
                hidden = self.model._encode(input_ids, attention_mask)
                operation_probabilities = self.model.operation_projection(hidden).softmax(dim=-1)
                boundary_probabilities = self.model.boundary_projection(hidden).softmax(dim=-1)
                word_ids = self.model._derive_word_ids(input_ids)
                pooled = self.model._pool_words(hidden, word_ids)
                word_features = self.model._derive_word_features(input_ids, word_ids, pooled.shape[1])
                enriched = pooled
                if self.model.config.use_lexical_word_features:
                    for feature_index, embedding in enumerate(self.model.word_feature_embeddings):
                        enriched = enriched + embedding(word_features[:, :, feature_index].long())
                word_probabilities = self.model.word_projection(enriched).softmax(dim=-1)

                for row_index in range(len(batch)):
                    ids = input_ids[row_index]
                    valid_chars = (
                        ids.ne(self.model.config.pad_token_id)
                        & ids.ne(self.model.config.bos_token_id)
                        & ids.ne(self.model.config.eos_token_id)
                        & ids.ne(self.model.config.space_token_id)
                    )
                    op = operation_probabilities[row_index][valid_chars].detach().cpu().numpy()
                    op_change = 1.0 - op[:, KEEP] if len(op) else np.asarray([])
                    operation = self._aggregate(
                        op,
                        op_change,
                        op.argmax(axis=-1) != KEEP if len(op) else np.asarray([]),
                        "tnt_operation",
                    )
                    operation["tnt_operation_mean_keep_probability"] = float(op[:, KEEP].mean()) if len(op) else 1.0
                    operation["tnt_operation_predicted_non_keep_fraction"] = operation.pop("tnt_operation_predicted_change_fraction")

                    word_count = max(int(word_ids[row_index].max().item()) + 1, 0)
                    word = word_probabilities[row_index, :word_count].detach().cpu().numpy()
                    word_change = 1.0 - word[:, WORD_KEEP] if len(word) else np.asarray([])
                    word_record = self._aggregate(
                        word,
                        word_change,
                        word.argmax(axis=-1) != WORD_KEEP if len(word) else np.asarray([]),
                        "tnt_word",
                    )
                    word_record["tnt_word_mean_keep_probability"] = float(word[:, WORD_KEEP].mean()) if len(word) else 1.0
                    word_record["tnt_word_fraction_change_above_030"] = float((word_change >= 0.3).mean()) if len(word) else 0.0
                    word_record["tnt_word_predicted_edit_fraction"] = word_record.pop("tnt_word_predicted_change_fraction")
                    protected = word_features[row_index, :word_count, 2].detach().cpu().numpy()
                    word_record["tnt_lexically_protected_word_fraction"] = float(protected.mean()) if len(protected) else 0.0

                    valid_boundaries = (
                        ids.ne(self.model.config.pad_token_id)
                        & ids.ne(self.model.config.bos_token_id)
                        & ids.ne(self.model.config.eos_token_id)
                    )
                    boundary = boundary_probabilities[row_index][valid_boundaries].detach().cpu().numpy()
                    boundary_ids = ids[valid_boundaries].detach().cpu().numpy()
                    is_space = boundary_ids == self.model.config.space_token_id
                    no_change = np.where(
                        is_space,
                        boundary[:, BOUNDARY_KEEP],
                        boundary[:, BOUNDARY_NONE],
                    ) if len(boundary) else np.asarray([])
                    boundary_change = 1.0 - no_change
                    predicted = boundary.argmax(axis=-1) if len(boundary) else np.asarray([])
                    predicted_change = np.where(
                        is_space,
                        predicted == BOUNDARY_DELETE,
                        predicted == BOUNDARY_INSERT_BEFORE,
                    ) if len(boundary) else np.asarray([])
                    boundary_record = self._aggregate(
                        boundary, boundary_change, predicted_change, "tnt_boundary"
                    )
                    boundary_record["tnt_boundary_mean_no_change_probability"] = float(no_change.mean()) if len(no_change) else 1.0
                    records.append({**operation, **word_record, **boundary_record})
        return records


def feature_families() -> dict[str, list[str]]:
    return {
        "deployable_surface": list(DEPLOYABLE_SURFACE_FEATURES),
        "deployable_tnt": list(DEPLOYABLE_TNT_FEATURES),
        "deployable_all": list(DEPLOYABLE_SURFACE_FEATURES + DEPLOYABLE_TNT_FEATURES),
        "oracle_analysis_only": list(ORACLE_FEATURES),
    }

