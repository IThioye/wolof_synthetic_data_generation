"""TNT-inspired non-autoregressive character edit Transformer.

The model predicts a local edit operation for every source character and, for
rewrite operations, a bounded output segment.  Normalized text is reconstructed
deterministically in source order.  This preserves TNT's central idea (joint
operation and character prediction) while supporting Wolof one-to-many edits
such as ``x -> kh`` and inserted spaces.
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
from transformers import PretrainedConfig, PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import Seq2SeqLMOutput

from src.modeling.beqi_transformer import _training_texts, build_character_vocabulary


KEEP = 0
DELETE = 1
SUBSTITUTE = 2
EXPAND = 3
OPERATION_NAMES = ("keep", "delete", "substitute", "expand")


@dataclass(frozen=True)
class TntEditArchitecture:
    d_model: int = 192
    attention_heads: int = 4
    encoder_layers: int = 3
    feed_forward_size: int = 768
    dropout: float = 0.1
    max_position_embeddings: int = 256
    output_slots_per_source: int = 4
    operation_loss_weight: float = 1.0
    character_loss_weight: float = 1.0
    label_smoothing: float = 0.1


class TntEditConfig(PretrainedConfig):
    model_type = "wolof_tnt_edit_transformer"

    def __init__(
        self,
        vocab_size: int = 128,
        d_model: int = 192,
        attention_heads: int = 4,
        encoder_layers: int = 3,
        feed_forward_size: int = 768,
        dropout: float = 0.1,
        max_position_embeddings: int = 256,
        output_slots_per_source: int = 4,
        operation_loss_weight: float = 1.0,
        character_loss_weight: float = 1.0,
        label_smoothing: float = 0.1,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        decoder_start_token_id: int = 1,
        **kwargs,
    ):
        kwargs.pop("is_encoder_decoder", None)
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            decoder_start_token_id=decoder_start_token_id,
            is_encoder_decoder=True,
            **kwargs,
        )
        if output_slots_per_source <= 0:
            raise ValueError("output_slots_per_source must be positive")
        if operation_loss_weight <= 0 or character_loss_weight <= 0:
            raise ValueError("TNT loss weights must be positive")
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in [0, 1)")
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.hidden_size = d_model
        self.attention_heads = attention_heads
        self.encoder_layers = encoder_layers
        self.feed_forward_size = feed_forward_size
        self.dropout = dropout
        self.max_position_embeddings = max_position_embeddings
        self.output_slots_per_source = output_slots_per_source
        self.operation_loss_weight = operation_loss_weight
        self.character_loss_weight = character_loss_weight
        self.label_smoothing = label_smoothing
        self.num_operations = len(OPERATION_NAMES)


def _strip_at_eos(values: Sequence[int], eos_token_id: int) -> list[int]:
    result = []
    for value in values:
        value = int(value)
        if value == eos_token_id:
            break
        result.append(value)
    return result


def _levenshtein_operations(
    source: Sequence[int], target: Sequence[int]
) -> list[tuple[str, int | None, int | None]]:
    """Return a deterministic Levenshtein alignment."""
    source = list(source)
    target = list(target)
    costs = [list(range(len(target) + 1))]
    backtrace: list[list[str | None]] = [
        [None] + ["insert"] * len(target)
    ]
    for source_index in range(1, len(source) + 1):
        row = [source_index] + [0] * len(target)
        moves: list[str | None] = ["delete"] + [None] * len(target)
        for target_index in range(1, len(target) + 1):
            if source[source_index - 1] == target[target_index - 1]:
                row[target_index] = costs[source_index - 1][target_index - 1]
                moves[target_index] = "match"
            else:
                row[target_index], moves[target_index] = min(
                    (
                        (costs[source_index - 1][target_index - 1] + 1, "substitute"),
                        (costs[source_index - 1][target_index] + 1, "delete"),
                        (row[target_index - 1] + 1, "insert"),
                    ),
                    key=lambda candidate: candidate[0],
                )
        costs.append(row)
        backtrace.append(moves)

    operations: list[tuple[str, int | None, int | None]] = []
    source_index, target_index = len(source), len(target)
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
    return operations


def tnt_edit_annotations(
    source_ids: Sequence[int],
    target_ids: Sequence[int],
    *,
    eos_token_id: int,
    pad_token_id: int,
    output_slots_per_source: int,
) -> tuple[list[int], list[list[int]]]:
    """Align a target into bounded output segments anchored to source positions.

    The final input EOS position is used as a boundary slot for target
    characters inserted after the last source character.
    """
    source = _strip_at_eos(source_ids, eos_token_id)
    target = _strip_at_eos(target_ids, eos_token_id)
    segments: list[list[int]] = [[] for _ in range(len(source) + 1)]
    source_cursor = 0
    for move, aligned_source, aligned_target in _levenshtein_operations(source, target):
        if move == "insert":
            assert aligned_target is not None
            segments[source_cursor].append(target[aligned_target])
        elif move in {"match", "substitute"}:
            assert aligned_source == source_cursor and aligned_target is not None
            segments[source_cursor].append(target[aligned_target])
            source_cursor += 1
        elif move == "delete":
            assert aligned_source == source_cursor
            source_cursor += 1
        else:
            raise RuntimeError(f"Unsupported alignment move: {move}")

    operation_labels: list[int] = []
    segment_labels: list[list[int]] = []
    for position, segment in enumerate(segments):
        if len(segment) > output_slots_per_source:
            raise ValueError(
                "Alignment requires "
                f"{len(segment)} output slots at source position {position}; "
                f"configured maximum is {output_slots_per_source}"
            )
        is_terminal = position == len(source)
        if not segment:
            operation = KEEP if is_terminal else DELETE
            labels = [-100] * output_slots_per_source
        elif not is_terminal and segment == [source[position]]:
            operation = KEEP
            labels = [-100] * output_slots_per_source
        elif len(segment) == 1:
            operation = SUBSTITUTE
            labels = segment + [pad_token_id] * (output_slots_per_source - 1)
        else:
            operation = EXPAND
            labels = segment + [pad_token_id] * (
                output_slots_per_source - len(segment)
            )
        operation_labels.append(operation)
        segment_labels.append(labels)
    return operation_labels, segment_labels


class TntEditTransformerForConditionalGeneration(PreTrainedModel, GenerationMixin):
    """Encoder-only edit predictor with deterministic reconstruction."""

    config_class = TntEditConfig
    base_model_prefix = "tnt_edit_transformer"
    main_input_name = "input_ids"

    def __init__(self, config: TntEditConfig):
        super().__init__(config)
        self.character_embedding = nn.Embedding(
            config.vocab_size, config.d_model, padding_idx=config.pad_token_id
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
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=config.encoder_layers
        )
        self.operation_projection = nn.Linear(config.d_model, config.num_operations)
        self.segment_projection = nn.Linear(
            config.d_model,
            config.output_slots_per_source * config.vocab_size,
        )
        self.post_init()

    def get_input_embeddings(self):
        return self.character_embedding

    def set_input_embeddings(self, value):
        self.character_embedding = value

    def _encode(self, input_ids, attention_mask=None):
        if input_ids.shape[1] > self.config.max_position_embeddings:
            raise ValueError("Input exceeds TNT positional limit")
        positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        hidden = self.character_embedding(input_ids) + self.position_embedding(positions)
        padding = (
            attention_mask.eq(0)
            if attention_mask is not None
            else input_ids.eq(self.config.pad_token_id)
        )
        return self.encoder(hidden, src_key_padding_mask=padding)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        operation_labels=None,
        segment_labels=None,
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
        loss = None
        if operation_labels is not None and segment_labels is not None:
            operation_loss = functional.cross_entropy(
                operation_logits.reshape(-1, self.config.num_operations),
                operation_labels.reshape(-1),
                ignore_index=-100,
                label_smoothing=self.config.label_smoothing,
            )
            character_loss = functional.cross_entropy(
                segment_logits.reshape(-1, self.config.vocab_size),
                segment_labels.reshape(-1),
                ignore_index=-100,
                label_smoothing=self.config.label_smoothing,
            )
            loss = (
                self.config.operation_loss_weight * operation_loss
                + self.config.character_loss_weight * character_loss
            )
        if not return_dict:
            values = (segment_logits, operation_logits)
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
            raise NotImplementedError("TNT edit reconstruction is deterministic")
        hidden = self._encode(input_ids, attention_mask)
        operations = self.operation_projection(hidden).argmax(dim=-1)
        segments = self.segment_projection(hidden).view(
            input_ids.shape[0],
            input_ids.shape[1],
            self.config.output_slots_per_source,
            self.config.vocab_size,
        ).argmax(dim=-1)
        limit = int(max_length or self.config.max_position_embeddings)
        generated_rows: list[list[int]] = []
        ignored = {
            self.config.pad_token_id,
            self.config.bos_token_id,
            self.config.eos_token_id,
        }
        for row in range(input_ids.shape[0]):
            output: list[int] = []
            for position in range(input_ids.shape[1]):
                source_token = int(input_ids[row, position])
                if source_token == self.config.pad_token_id:
                    break
                operation = int(operations[row, position])
                if operation == KEEP:
                    if source_token not in ignored:
                        output.append(source_token)
                elif operation == DELETE:
                    continue
                else:
                    for predicted in segments[row, position].tolist():
                        predicted = int(predicted)
                        if predicted == self.config.pad_token_id:
                            break
                        if predicted not in ignored:
                            output.append(predicted)
                if len(output) >= max(limit - 1, 0):
                    break
            output = output[: max(limit - 1, 0)] + [self.config.eos_token_id]
            generated_rows.append(output)
        width = max(len(row) for row in generated_rows)
        generated = input_ids.new_full(
            (len(generated_rows), width), self.config.pad_token_id
        )
        for row_index, row in enumerate(generated_rows):
            generated[row_index, : len(row)] = torch.tensor(
                row, dtype=input_ids.dtype, device=input_ids.device
            )
        return generated


def _initialization_fingerprint(
    vocabulary: dict[str, int], architecture: TntEditArchitecture, seed: int
) -> str:
    payload = {
        "implementation": "wolof_tnt_edit_transformer_v1",
        "vocabulary": vocabulary,
        "architecture": asdict(architecture),
        "seed": seed,
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def prepare_tnt_edit_transformer(
    training_frames: Sequence[pd.DataFrame],
    *,
    output_root: Path,
    seed: int = 2026,
    architecture: TntEditArchitecture | None = None,
) -> Path:
    """Create or reuse one deterministic TNT-style initialization."""
    from tokenizers import Tokenizer
    from tokenizers.decoders import Fuse
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Split
    from tokenizers.processors import TemplateProcessing
    from transformers import PreTrainedTokenizerFast, set_seed

    architecture = architecture or TntEditArchitecture()
    texts = _training_texts(training_frames)
    vocabulary = build_character_vocabulary(texts)
    fingerprint = _initialization_fingerprint(vocabulary, architecture, seed)
    output_dir = Path(output_root) / f"tnt_edit_char_{fingerprint[:12]}"
    manifest_path = output_dir / "initialization_manifest.json"
    if manifest_path.is_file():
        required = ("config.json", "model.safetensors", "tokenizer.json")
        missing = [name for name in required if not (output_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(f"Incomplete TNT initialization: {missing}")
        print(f"[M7] Reusing TNT edit initialization: {output_dir.resolve()}")
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
    config = TntEditConfig(
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
        label_smoothing=architecture.label_smoothing,
        pad_token_id=vocabulary["<pad>"],
        bos_token_id=vocabulary["<s>"],
        eos_token_id=vocabulary["</s>"],
        decoder_start_token_id=vocabulary["<s>"],
    )
    set_seed(seed)
    model = TntEditTransformerForConditionalGeneration(config)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    manifest_path.write_text(
        json.dumps(
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "fingerprint": fingerprint,
                "implementation": "TNT-inspired non-autoregressive edit Transformer",
                "paper": "Tan et al. (2020), TNT",
                "segmentation": "character-level",
                "operations": list(OPERATION_NAMES),
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
        f"[M7] Created TNT edit Transformer with {parameter_count:,} parameters: "
        f"{output_dir.resolve()}"
    )
    return output_dir


def is_tnt_edit_checkpoint(model_path: Path | str) -> bool:
    config_path = Path(model_path) / "config.json"
    if not config_path.is_file():
        return False
    try:
        return json.loads(config_path.read_text(encoding="utf-8")).get(
            "model_type"
        ) == TntEditConfig.model_type
    except (OSError, json.JSONDecodeError):
        return False


def tnt_edit_model_class(model_path: Path | str, fallback):
    return (
        TntEditTransformerForConditionalGeneration
        if is_tnt_edit_checkpoint(model_path)
        else fallback
    )
