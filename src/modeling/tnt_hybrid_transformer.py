"""Boundary- and word-aware extension of the TNT character edit model.

This module is an experimental V2 architecture.  It does not alter the frozen
V1 benchmark.  The character edit heads remain the primary generator, while an
explicit boundary head handles spaces and a pooled word head predicts whether a
source token should be kept or edited.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as functional
from transformers.modeling_outputs import Seq2SeqLMOutput

from src.modeling.beqi_transformer import _training_texts, build_character_vocabulary
from src.modeling.tnt_edit_transformer import (
    DELETE,
    KEEP,
    OPERATION_NAMES,
    TntEditConfig,
    TntEditTransformerForConditionalGeneration,
    _levenshtein_operations,
    _strip_at_eos,
    tnt_edit_annotations,
)


BOUNDARY_NONE = 0
BOUNDARY_KEEP = 1
BOUNDARY_DELETE = 2
BOUNDARY_INSERT_BEFORE = 3
BOUNDARY_ACTIONS = ("none", "keep_space", "delete_space", "insert_space_before")

WORD_KEEP = 0
WORD_EDIT = 1
WORD_ACTIONS = ("keep", "edit")


@dataclass(frozen=True)
class TntHybridArchitecture:
    d_model: int = 192
    attention_heads: int = 4
    encoder_layers: int = 3
    feed_forward_size: int = 768
    dropout: float = 0.1
    max_position_embeddings: int = 256
    output_slots_per_source: int = 16
    operation_loss_weight: float = 1.0
    character_loss_weight: float = 1.0
    operation_class_weights: tuple[float, ...] | None = None
    boundary_loss_weight: float = 0.5
    word_loss_weight: float = 0.5
    boundary_class_weights: tuple[float, ...] | None = None
    word_class_weights: tuple[float, ...] | None = None
    label_smoothing: float = 0.1
    use_boundary_inference: bool = True
    use_word_keep_gate: bool = True
    boundary_confidence_threshold: float = 0.60
    word_keep_threshold: float = 0.80


class TntHybridConfig(TntEditConfig):
    model_type = "wolof_tnt_hybrid_transformer"

    def __init__(
        self,
        boundary_loss_weight: float = 0.5,
        word_loss_weight: float = 0.5,
        operation_class_weights: Sequence[float] | None = None,
        boundary_class_weights: Sequence[float] | None = None,
        word_class_weights: Sequence[float] | None = None,
        use_boundary_inference: bool = True,
        use_word_keep_gate: bool = True,
        boundary_confidence_threshold: float = 0.60,
        word_keep_threshold: float = 0.80,
        space_token_id: int = -1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if boundary_loss_weight < 0 or word_loss_weight < 0:
            raise ValueError("Hybrid auxiliary loss weights must be non-negative")
        for name, value in (
            ("boundary_confidence_threshold", boundary_confidence_threshold),
            ("word_keep_threshold", word_keep_threshold),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        self.boundary_loss_weight = float(boundary_loss_weight)
        self.word_loss_weight = float(word_loss_weight)
        self.operation_class_weights = self._validated_class_weights(
            "operation_class_weights",
            operation_class_weights,
            len(OPERATION_NAMES),
        )
        self.boundary_class_weights = self._validated_class_weights(
            "boundary_class_weights",
            boundary_class_weights,
            len(BOUNDARY_ACTIONS),
        )
        self.word_class_weights = self._validated_class_weights(
            "word_class_weights", word_class_weights, len(WORD_ACTIONS)
        )
        self.use_boundary_inference = bool(use_boundary_inference)
        self.use_word_keep_gate = bool(use_word_keep_gate)
        self.boundary_confidence_threshold = float(boundary_confidence_threshold)
        self.word_keep_threshold = float(word_keep_threshold)
        self.space_token_id = int(space_token_id)
        self.num_boundary_actions = len(BOUNDARY_ACTIONS)
        self.num_word_actions = len(WORD_ACTIONS)

    @staticmethod
    def _validated_class_weights(name, values, expected_length):
        if values is None:
            return None
        weights = [float(value) for value in values]
        if len(weights) != expected_length or any(value <= 0 for value in weights):
            raise ValueError(
                f"{name} must contain {expected_length} positive values"
            )
        return weights


def _alignment_segments(
    source: Sequence[int], target: Sequence[int]
) -> tuple[list[list[int]], list[tuple[str, int | None, int | None]]]:
    operations = _levenshtein_operations(source, target)
    segments: list[list[int]] = [[] for _ in range(len(source) + 1)]
    source_cursor = 0
    for move, aligned_source, aligned_target in operations:
        if move == "insert":
            assert aligned_target is not None
            segments[source_cursor].append(int(target[aligned_target]))
        elif move in {"match", "substitute"}:
            assert aligned_source == source_cursor and aligned_target is not None
            segments[source_cursor].append(int(target[aligned_target]))
            source_cursor += 1
        elif move == "delete":
            assert aligned_source == source_cursor
            source_cursor += 1
        else:
            raise RuntimeError(f"Unsupported alignment move: {move}")
    return segments, operations


def _source_word_ids(source: Sequence[int], space_token_id: int) -> list[int]:
    word_ids: list[int] = []
    current_word = -1
    inside_word = False
    for token_id in source:
        if int(token_id) == space_token_id:
            word_ids.append(-1)
            inside_word = False
        else:
            if not inside_word:
                current_word += 1
                inside_word = True
            word_ids.append(current_word)
    return word_ids


def hybrid_tnt_annotations(
    source_ids: Sequence[int],
    target_ids: Sequence[int],
    *,
    eos_token_id: int,
    pad_token_id: int,
    space_token_id: int,
    output_slots_per_source: int,
) -> tuple[list[int], list[list[int]], list[int], list[int], list[int]]:
    """Create character, boundary, and word-level supervision from one alignment."""
    source = _strip_at_eos(source_ids, eos_token_id)
    target = _strip_at_eos(target_ids, eos_token_id)
    operation_labels, segment_labels = tnt_edit_annotations(
        source_ids,
        target_ids,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        output_slots_per_source=output_slots_per_source,
    )
    # V1 trained every unused output slot as PAD. With 32 slots, a one-character
    # replacement consequently supplied 31 PAD targets and only one content
    # target. V2 supervises the first PAD as the segment terminator and ignores
    # all later slots, preserving a learnable stop signal without dominating
    # the character objective.
    for segment in segment_labels:
        try:
            terminator = segment.index(pad_token_id)
        except ValueError:
            continue
        segment[terminator + 1 :] = [-100] * (len(segment) - terminator - 1)
    raw_segments, alignment = _alignment_segments(source, target)

    boundary_labels: list[int] = []
    for position, source_token in enumerate(source):
        segment = raw_segments[position]
        if int(source_token) == space_token_id:
            boundary_labels.append(
                BOUNDARY_KEEP if space_token_id in segment else BOUNDARY_DELETE
            )
        else:
            boundary_labels.append(
                BOUNDARY_INSERT_BEFORE
                if segment and int(segment[0]) == space_token_id
                else BOUNDARY_NONE
            )
    # EOS is a character-edit anchor for trailing insertions, but not a lexical
    # boundary decision in this V2 experiment.
    boundary_labels.append(-100)

    source_word_ids = _source_word_ids(source, space_token_id)
    word_count = max(source_word_ids, default=-1) + 1
    word_labels = [WORD_KEEP] * word_count

    def mark_word(position: int) -> None:
        candidates = []
        if 0 <= position < len(source_word_ids):
            candidates.append(source_word_ids[position])
        if 0 <= position - 1 < len(source_word_ids):
            candidates.append(source_word_ids[position - 1])
        for word_id in candidates:
            if word_id >= 0:
                word_labels[word_id] = WORD_EDIT
                return

    source_cursor = 0
    for move, aligned_source, aligned_target in alignment:
        if move == "insert":
            assert aligned_target is not None
            if int(target[aligned_target]) != space_token_id:
                mark_word(source_cursor)
        elif move == "substitute":
            assert aligned_source is not None and aligned_target is not None
            if (
                int(source[aligned_source]) != space_token_id
                or int(target[aligned_target]) != space_token_id
            ):
                mark_word(aligned_source)
            source_cursor += 1
        elif move == "delete":
            assert aligned_source is not None
            if int(source[aligned_source]) != space_token_id:
                mark_word(aligned_source)
            source_cursor += 1
        elif move == "match":
            source_cursor += 1

    # Match the source-anchored tensors by adding the EOS slot.
    return (
        operation_labels,
        segment_labels,
        boundary_labels,
        source_word_ids + [-1],
        word_labels,
    )


def _safe_cross_entropy(
    logits,
    labels,
    *,
    label_smoothing: float = 0.0,
    class_weights: Sequence[float] | None = None,
):
    flattened_labels = labels.reshape(-1)
    valid = flattened_labels.ne(-100)
    if not bool(valid.any()):
        return logits.sum() * 0.0
    flattened_logits = logits.reshape(-1, logits.shape[-1])
    return functional.cross_entropy(
        flattened_logits[valid],
        flattened_labels[valid],
        weight=(
            flattened_logits.new_tensor(class_weights)
            if class_weights is not None
            else None
        ),
        label_smoothing=label_smoothing,
    )


class TntHybridTransformerForConditionalGeneration(
    TntEditTransformerForConditionalGeneration
):
    config_class = TntHybridConfig
    base_model_prefix = "tnt_hybrid_transformer"

    def __init__(self, config: TntHybridConfig):
        super().__init__(config)
        self.boundary_projection = nn.Linear(
            config.d_model, config.num_boundary_actions
        )
        self.word_projection = nn.Linear(config.d_model, config.num_word_actions)
        # The parent initializes before these two heads exist. Re-running the
        # standard initializer is deterministic under the shared seed and makes
        # every hybrid parameter follow the same initialization policy.
        self.post_init()

    def _derive_word_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        valid = (
            input_ids.ne(self.config.pad_token_id)
            & input_ids.ne(self.config.bos_token_id)
            & input_ids.ne(self.config.eos_token_id)
            & input_ids.ne(self.config.space_token_id)
        )
        previous_valid = functional.pad(valid[:, :-1], (1, 0), value=False)
        starts = valid & ~previous_valid
        word_ids = starts.long().cumsum(dim=1) - 1
        return torch.where(valid, word_ids, word_ids.new_full((), -1))

    def _pool_words(
        self,
        hidden: torch.Tensor,
        word_ids: torch.Tensor,
        word_count: int | None = None,
    ) -> torch.Tensor:
        if word_count is None:
            word_count = max(int(word_ids.max().item()) + 1, 0)
        pooled = hidden.new_zeros(hidden.shape[0], word_count, hidden.shape[-1])
        counts = hidden.new_zeros(hidden.shape[0], word_count, 1)
        if word_count == 0:
            return pooled
        valid = word_ids.ge(0) & word_ids.lt(word_count)
        safe_ids = word_ids.clamp(min=0, max=word_count - 1)
        pooled.scatter_add_(
            1,
            safe_ids.unsqueeze(-1).expand(-1, -1, hidden.shape[-1]),
            hidden * valid.unsqueeze(-1),
        )
        counts.scatter_add_(
            1,
            safe_ids.unsqueeze(-1),
            valid.unsqueeze(-1).to(hidden.dtype),
        )
        return pooled / counts.clamp_min(1.0)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        operation_labels=None,
        segment_labels=None,
        boundary_labels=None,
        word_ids=None,
        word_labels=None,
        return_dict=True,
        **kwargs,
    ):
        if input_ids is None:
            raise ValueError("input_ids are required")
        hidden = self._encode(input_ids, attention_mask)
        operation_logits = self.operation_projection(hidden)
        segment_logits = self.segment_projection(hidden).view(
            input_ids.shape[0],
            input_ids.shape[1],
            self.config.output_slots_per_source,
            self.config.vocab_size,
        )
        boundary_logits = self.boundary_projection(hidden)
        if word_ids is None:
            word_ids = self._derive_word_ids(input_ids)
        requested_words = word_labels.shape[1] if word_labels is not None else None
        pooled_words = self._pool_words(hidden, word_ids, requested_words)
        word_logits = self.word_projection(pooled_words)

        loss = None
        if operation_labels is not None and segment_labels is not None:
            operation_loss = _safe_cross_entropy(
                operation_logits,
                operation_labels,
                label_smoothing=self.config.label_smoothing,
                class_weights=self.config.operation_class_weights,
            )
            character_loss = _safe_cross_entropy(
                segment_logits,
                segment_labels,
                label_smoothing=self.config.label_smoothing,
            )
            loss = (
                self.config.operation_loss_weight * operation_loss
                + self.config.character_loss_weight * character_loss
            )
            if boundary_labels is not None and self.config.boundary_loss_weight:
                loss = loss + self.config.boundary_loss_weight * _safe_cross_entropy(
                    boundary_logits,
                    boundary_labels,
                    label_smoothing=self.config.label_smoothing,
                    class_weights=self.config.boundary_class_weights,
                )
            if word_labels is not None and self.config.word_loss_weight:
                loss = loss + self.config.word_loss_weight * _safe_cross_entropy(
                    word_logits,
                    word_labels,
                    label_smoothing=self.config.label_smoothing,
                    class_weights=self.config.word_class_weights,
                )
        if not return_dict:
            values = (segment_logits, operation_logits, boundary_logits, word_logits)
            return ((loss,) + values) if loss is not None else values
        return Seq2SeqLMOutput(loss=loss, logits=segment_logits)

    @torch.no_grad()
    def generate(
        self,
        input_ids=None,
        attention_mask=None,
        max_length=None,
        num_beams=1,
        **kwargs,
    ):
        if input_ids is None:
            raise ValueError("input_ids are required")
        if num_beams not in (None, 1):
            raise NotImplementedError("Hybrid TNT reconstruction is deterministic")

        hidden = self._encode(input_ids, attention_mask)
        operations = self.operation_projection(hidden).argmax(dim=-1)
        segments = self.segment_projection(hidden).view(
            input_ids.shape[0],
            input_ids.shape[1],
            self.config.output_slots_per_source,
            self.config.vocab_size,
        ).argmax(dim=-1)
        boundary_confidence = boundaries = None
        if self.config.use_boundary_inference:
            boundary_probabilities = self.boundary_projection(hidden).softmax(dim=-1)
            boundary_confidence, boundaries = boundary_probabilities.max(dim=-1)
        word_ids = word_probabilities = None
        if self.config.use_word_keep_gate:
            word_ids = self._derive_word_ids(input_ids)
            word_logits = self.word_projection(self._pool_words(hidden, word_ids))
            word_probabilities = word_logits.softmax(dim=-1)

        # Transfer each result to the CPU once. Calling int() or float() on a
        # CUDA scalar inside the reconstruction loop forces a synchronization
        # per character and made a 24-row evaluation take several minutes.
        input_rows = input_ids.detach().cpu().tolist()
        operation_rows = operations.detach().cpu().tolist()
        segment_rows = segments.detach().cpu().tolist()
        boundary_rows = boundaries.detach().cpu().tolist() if boundaries is not None else None
        boundary_confidence_rows = (
            boundary_confidence.detach().cpu().tolist()
            if boundary_confidence is not None
            else None
        )
        word_id_rows = word_ids.detach().cpu().tolist() if word_ids is not None else None
        word_probability_rows = (
            word_probabilities.detach().cpu().tolist()
            if word_probabilities is not None
            else None
        )

        limit = int(max_length or self.config.max_position_embeddings)
        ignored = {
            self.config.pad_token_id,
            self.config.bos_token_id,
            self.config.eos_token_id,
        }
        generated_rows: list[list[int]] = []
        for row in range(input_ids.shape[0]):
            output: list[int] = []
            for position, source_token in enumerate(input_rows[row]):
                source_token = int(source_token)
                if source_token == self.config.pad_token_id:
                    break

                boundary = (
                    int(boundary_rows[row][position])
                    if boundary_rows is not None
                    else BOUNDARY_NONE
                )
                boundary_is_confident = bool(
                    boundary_confidence_rows is not None
                    and boundary_confidence_rows[row][position]
                    >= self.config.boundary_confidence_threshold
                )
                if (
                    self.config.use_boundary_inference
                    and source_token == self.config.space_token_id
                    and boundary_is_confident
                    and boundary in {BOUNDARY_KEEP, BOUNDARY_DELETE}
                ):
                    if boundary == BOUNDARY_KEEP and (
                        not output or output[-1] != self.config.space_token_id
                    ):
                        output.append(self.config.space_token_id)
                    continue

                if (
                    self.config.use_boundary_inference
                    and source_token not in ignored
                    and source_token != self.config.space_token_id
                    and boundary_is_confident
                    and boundary == BOUNDARY_INSERT_BEFORE
                    and output
                    and output[-1] != self.config.space_token_id
                ):
                    output.append(self.config.space_token_id)

                operation = int(operation_rows[row][position])
                word_id = (
                    int(word_id_rows[row][position])
                    if word_id_rows is not None
                    else -1
                )
                if (
                    self.config.use_word_keep_gate
                    and word_id >= 0
                    and word_probability_rows is not None
                    and word_id < len(word_probability_rows[row])
                    and word_probability_rows[row][word_id][WORD_KEEP]
                    >= self.config.word_keep_threshold
                ):
                    operation = KEEP

                if operation == KEEP:
                    predicted_tokens = [source_token]
                elif operation == DELETE:
                    predicted_tokens = []
                else:
                    predicted_tokens = []
                    for predicted in segment_rows[row][position]:
                        predicted = int(predicted)
                        if predicted == self.config.pad_token_id:
                            break
                        if predicted not in ignored:
                            predicted_tokens.append(predicted)

                if (
                    self.config.use_boundary_inference
                    and source_token != self.config.space_token_id
                    and boundary_is_confident
                    and boundary in {BOUNDARY_NONE, BOUNDARY_INSERT_BEFORE}
                ):
                    while (
                        predicted_tokens
                        and predicted_tokens[0] == self.config.space_token_id
                    ):
                        predicted_tokens.pop(0)
                output.extend(token for token in predicted_tokens if token not in ignored)
                if len(output) >= max(limit - 1, 0):
                    break

            output = output[: max(limit - 1, 0)] + [self.config.eos_token_id]
            generated_rows.append(output)

        width = max(len(row) for row in generated_rows)
        generated = input_ids.new_full(
            (len(generated_rows), width), self.config.pad_token_id
        )
        for row_index, row_values in enumerate(generated_rows):
            generated[row_index, : len(row_values)] = torch.tensor(
                row_values, dtype=input_ids.dtype, device=input_ids.device
            )
        return generated


def _initialization_fingerprint(
    vocabulary: dict[str, int], architecture: TntHybridArchitecture, seed: int
) -> str:
    payload = {
        "implementation": "wolof_tnt_hybrid_transformer_v2",
        "vocabulary": vocabulary,
        "architecture": asdict(architecture),
        "seed": seed,
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def prepare_tnt_hybrid_transformer(
    training_frames: Sequence[pd.DataFrame],
    *,
    output_root: Path,
    seed: int = 2026,
    architecture: TntHybridArchitecture | None = None,
) -> Path:
    """Create or reuse one deterministic hybrid TNT initialization."""
    from tokenizers import Tokenizer
    from tokenizers.decoders import Fuse
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Split
    from tokenizers.processors import TemplateProcessing
    from transformers import PreTrainedTokenizerFast, set_seed

    architecture = architecture or TntHybridArchitecture()
    texts = _training_texts(training_frames)
    vocabulary = build_character_vocabulary(texts)
    fingerprint = _initialization_fingerprint(vocabulary, architecture, seed)
    output_dir = Path(output_root) / f"tnt_hybrid_char_{fingerprint[:12]}"
    manifest_path = output_dir / "initialization_manifest.json"
    if manifest_path.is_file():
        required = ("config.json", "model.safetensors", "tokenizer.json")
        missing = [name for name in required if not (output_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(f"Incomplete hybrid TNT initialization: {missing}")
        print(f"[TNT-V2] Reusing hybrid initialization: {output_dir.resolve()}")
        return output_dir

    output_dir.mkdir(parents=True, exist_ok=False)
    backend = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    backend.pre_tokenizer = Split("", behavior="isolated")
    backend.decoder = Fuse()
    backend.post_processor = TemplateProcessing(
        single="$A </s>",
        pair="$A </s> $B </s>",
        special_tokens=[("</s>", vocabulary["</s>"])],
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        model_max_length=architecture.max_position_embeddings,
        clean_up_tokenization_spaces=False,
    )
    config = TntHybridConfig(
        vocab_size=len(vocabulary),
        d_model=architecture.d_model,
        attention_heads=architecture.attention_heads,
        encoder_layers=architecture.encoder_layers,
        feed_forward_size=architecture.feed_forward_size,
        dropout=architecture.dropout,
        max_position_embeddings=architecture.max_position_embeddings,
        output_slots_per_source=architecture.output_slots_per_source,
        operation_loss_weight=architecture.operation_loss_weight,
        character_loss_weight=architecture.character_loss_weight,
        operation_class_weights=architecture.operation_class_weights,
        boundary_loss_weight=architecture.boundary_loss_weight,
        word_loss_weight=architecture.word_loss_weight,
        boundary_class_weights=architecture.boundary_class_weights,
        word_class_weights=architecture.word_class_weights,
        label_smoothing=architecture.label_smoothing,
        use_boundary_inference=architecture.use_boundary_inference,
        use_word_keep_gate=architecture.use_word_keep_gate,
        boundary_confidence_threshold=architecture.boundary_confidence_threshold,
        word_keep_threshold=architecture.word_keep_threshold,
        space_token_id=vocabulary.get(" ", -1),
        pad_token_id=vocabulary["<pad>"],
        bos_token_id=vocabulary["<s>"],
        eos_token_id=vocabulary["</s>"],
        decoder_start_token_id=vocabulary["<s>"],
    )
    set_seed(seed)
    model = TntHybridTransformerForConditionalGeneration(config)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    manifest_path.write_text(
        json.dumps(
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "fingerprint": fingerprint,
                "implementation": "Boundary- and word-aware TNT V2",
                "segmentation": "character encoder with pooled whitespace tokens",
                "operations": list(OPERATION_NAMES),
                "boundary_actions": list(BOUNDARY_ACTIONS),
                "word_actions": list(WORD_ACTIONS),
                "architecture": asdict(architecture),
                "seed": seed,
                "parameter_count": parameter_count,
                "training_text_count_used_for_vocabulary": len(texts),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"[TNT-V2] Created hybrid model with {parameter_count:,} parameters: "
        f"{output_dir.resolve()}"
    )
    return output_dir


def is_tnt_hybrid_checkpoint(model_path: Path | str) -> bool:
    config_path = Path(model_path) / "config.json"
    if not config_path.is_file():
        return False
    try:
        return json.loads(config_path.read_text(encoding="utf-8")).get(
            "model_type"
        ) == TntHybridConfig.model_type
    except (OSError, json.JSONDecodeError):
        return False
