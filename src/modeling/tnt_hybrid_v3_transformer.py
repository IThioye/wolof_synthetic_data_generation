"""Diagnostic-driven V3 of the TNT hybrid normalizer.

V3 separates replacement length from replacement content, applies focal loss
to rare boundary actions, and gives the word head method-independent lexical
features for French words, named entities, and protected tokens.
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

from src.config import LINGUISTIC_ANNOTATIONS_PATH, SENEGALESE_SURNAMES_PATH
from src.modeling.beqi_transformer import _training_texts, build_character_vocabulary
from src.modeling.tnt_edit_transformer import (
    DELETE,
    EXPAND,
    KEEP,
    OPERATION_NAMES,
    SUBSTITUTE,
)
from src.modeling.tnt_hybrid_transformer import (
    BOUNDARY_ACTIONS,
    BOUNDARY_DELETE,
    BOUNDARY_INSERT_BEFORE,
    BOUNDARY_KEEP,
    BOUNDARY_NONE,
    WORD_ACTIONS,
    WORD_KEEP,
    TntHybridConfig,
    TntHybridTransformerForConditionalGeneration,
    _alignment_segments,
    _safe_cross_entropy,
    _strip_at_eos,
    hybrid_tnt_annotations,
)


@dataclass(frozen=True)
class TntHybridV3Architecture:
    d_model: int = 192
    attention_heads: int = 4
    encoder_layers: int = 3
    feed_forward_size: int = 768
    dropout: float = 0.1
    max_position_embeddings: int = 256
    output_slots_per_source: int = 16
    operation_loss_weight: float = 1.0
    character_loss_weight: float = 1.0
    segment_length_loss_weight: float = 1.0
    boundary_loss_weight: float = 1.0
    word_loss_weight: float = 0.5
    operation_class_weights: tuple[float, ...] | None = None
    segment_length_class_weights: tuple[float, ...] | None = None
    boundary_class_weights: tuple[float, ...] | None = None
    word_class_weights: tuple[float, ...] | None = None
    boundary_focal_gamma: float = 2.0
    label_smoothing: float = 0.1
    use_boundary_inference: bool = True
    use_word_keep_gate: bool = True
    use_lexical_word_features: bool = True
    use_hard_lexical_protection: bool = True
    boundary_confidence_threshold: float = 0.50
    word_keep_threshold: float = 0.70


class TntHybridV3Config(TntHybridConfig):
    model_type = "wolof_tnt_hybrid_v3_transformer"

    def __init__(
        self,
        segment_length_loss_weight: float = 1.0,
        segment_length_class_weights: Sequence[float] | None = None,
        boundary_focal_gamma: float = 2.0,
        use_lexical_word_features: bool = True,
        use_hard_lexical_protection: bool = True,
        word_feature_lookup: dict[str, Sequence[int]] | None = None,
        lowercase_token_id_map: dict[str, int] | None = None,
        word_edge_strip_token_ids: Sequence[int] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if segment_length_loss_weight < 0:
            raise ValueError("segment_length_loss_weight must be non-negative")
        if boundary_focal_gamma < 0:
            raise ValueError("boundary_focal_gamma must be non-negative")
        self.segment_length_loss_weight = float(segment_length_loss_weight)
        self.num_segment_lengths = self.output_slots_per_source + 1
        self.segment_length_class_weights = self._validated_class_weights(
            "segment_length_class_weights",
            segment_length_class_weights,
            self.num_segment_lengths,
        )
        self.boundary_focal_gamma = float(boundary_focal_gamma)
        self.use_lexical_word_features = bool(use_lexical_word_features)
        self.use_hard_lexical_protection = bool(use_hard_lexical_protection)
        self.word_feature_lookup = {
            str(key): [int(value) for value in values]
            for key, values in (word_feature_lookup or {}).items()
        }
        self.lowercase_token_id_map = {
            str(key): int(value)
            for key, value in (lowercase_token_id_map or {}).items()
        }
        self.word_edge_strip_token_ids = [
            int(value) for value in (word_edge_strip_token_ids or [])
        ]
        self.num_word_features = 3


def hybrid_v3_annotations(
    source_ids: Sequence[int],
    target_ids: Sequence[int],
    *,
    eos_token_id: int,
    pad_token_id: int,
    space_token_id: int,
    output_slots_per_source: int,
) -> tuple[list[int], list[list[int]], list[int], list[int], list[int], list[int]]:
    """Return V2 labels plus explicit output lengths and content-only slots."""
    base = list(
        hybrid_tnt_annotations(
            source_ids,
            target_ids,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            space_token_id=space_token_id,
            output_slots_per_source=output_slots_per_source,
        )
    )
    source = _strip_at_eos(source_ids, eos_token_id)
    target = _strip_at_eos(target_ids, eos_token_id)
    segments, _ = _alignment_segments(source, target)
    # The length head is consulted only for substitution and expansion.
    # Supervising the overwhelmingly common KEEP positions as length one would
    # replace the old PAD dominance with a new length-one dominance.
    lengths = [
        len(segment) if operation in {SUBSTITUTE, EXPAND} else -100
        for segment, operation in zip(segments, base[0])
    ]

    content_labels = base[1]
    for labels in content_labels:
        for index, token_id in enumerate(labels):
            if token_id == pad_token_id:
                labels[index] = -100
    base[1] = content_labels
    return (*base, lengths)


def _focal_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    gamma: float,
    class_weights: Sequence[float] | None = None,
) -> torch.Tensor:
    flattened_labels = labels.reshape(-1)
    valid = flattened_labels.ne(-100)
    if not bool(valid.any()):
        return logits.sum() * 0.0
    flattened_logits = logits.reshape(-1, logits.shape[-1])[valid]
    valid_labels = flattened_labels[valid]
    weights = (
        flattened_logits.new_tensor(class_weights)
        if class_weights is not None
        else None
    )
    cross_entropy = functional.cross_entropy(
        flattened_logits,
        valid_labels,
        weight=weights,
        reduction="none",
    )
    target_probability = flattened_logits.softmax(dim=-1).gather(
        1, valid_labels.unsqueeze(1)
    ).squeeze(1)
    return (((1.0 - target_probability) ** gamma) * cross_entropy).mean()


class TntHybridV3TransformerForConditionalGeneration(
    TntHybridTransformerForConditionalGeneration
):
    config_class = TntHybridV3Config
    base_model_prefix = "tnt_hybrid_v3_transformer"

    def __init__(self, config: TntHybridV3Config):
        super().__init__(config)
        self.segment_length_projection = nn.Linear(
            config.d_model, config.num_segment_lengths
        )
        self.word_feature_embeddings = nn.ModuleList(
            nn.Embedding(2, config.d_model) for _ in range(config.num_word_features)
        )
        self.post_init()

    def _normalized_word_key(self, token_ids: Sequence[int]) -> str:
        values = [int(value) for value in token_ids]
        strip_ids = set(self.config.word_edge_strip_token_ids)
        while values and values[0] in strip_ids:
            values.pop(0)
        while values and values[-1] in strip_ids:
            values.pop()
        lowercase_map = self.config.lowercase_token_id_map
        values = [lowercase_map.get(str(value), value) for value in values]
        return ",".join(str(value) for value in values)

    def word_features_for_ids(
        self, input_ids: Sequence[int], word_ids: Sequence[int]
    ) -> list[list[int]]:
        word_count = max((int(value) for value in word_ids), default=-1) + 1
        features = [[0, 0, 0] for _ in range(word_count)]
        tokens: list[list[int]] = [[] for _ in range(word_count)]
        for token_id, word_id in zip(input_ids, word_ids):
            word_id = int(word_id)
            if word_id >= 0:
                tokens[word_id].append(int(token_id))
        for word_id, token_values in enumerate(tokens):
            values = self.config.word_feature_lookup.get(
                self._normalized_word_key(token_values)
            )
            if values is not None:
                features[word_id] = [int(value) for value in values]
        return features

    def _derive_word_features(
        self, input_ids: torch.Tensor, word_ids: torch.Tensor, word_count: int
    ) -> torch.Tensor:
        input_rows = input_ids.detach().cpu().tolist()
        word_id_rows = word_ids.detach().cpu().tolist()
        rows = [
            self.word_features_for_ids(input_row, word_id_row)
            for input_row, word_id_row in zip(input_rows, word_id_rows)
        ]
        features = input_ids.new_zeros(
            input_ids.shape[0], word_count, self.config.num_word_features
        )
        for row_index, values in enumerate(rows):
            if values:
                features[row_index, : len(values)] = torch.tensor(
                    values, dtype=torch.long, device=input_ids.device
                )
        return features

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        operation_labels=None,
        segment_labels=None,
        segment_length_labels=None,
        boundary_labels=None,
        word_ids=None,
        word_labels=None,
        word_features=None,
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
        segment_length_logits = self.segment_length_projection(hidden)
        boundary_logits = self.boundary_projection(hidden)
        if word_ids is None:
            word_ids = self._derive_word_ids(input_ids)
        requested_words = word_labels.shape[1] if word_labels is not None else None
        pooled = self._pool_words(hidden, word_ids, requested_words)
        if word_features is None:
            word_features = self._derive_word_features(
                input_ids, word_ids, pooled.shape[1]
            )
        enriched = pooled
        if self.config.use_lexical_word_features:
            for feature_index, embedding in enumerate(self.word_feature_embeddings):
                enriched = enriched + embedding(
                    word_features[:, :, feature_index].long()
                )
        word_logits = self.word_projection(enriched)

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
            if (
                segment_length_labels is not None
                and self.config.segment_length_loss_weight
            ):
                loss = loss + self.config.segment_length_loss_weight * _safe_cross_entropy(
                    segment_length_logits,
                    segment_length_labels,
                    label_smoothing=self.config.label_smoothing,
                    class_weights=self.config.segment_length_class_weights,
                )
            if boundary_labels is not None and self.config.boundary_loss_weight:
                loss = loss + self.config.boundary_loss_weight * _focal_cross_entropy(
                    boundary_logits,
                    boundary_labels,
                    gamma=self.config.boundary_focal_gamma,
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
            values = (
                segment_logits,
                operation_logits,
                segment_length_logits,
                boundary_logits,
                word_logits,
            )
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
        segment_lengths = self.segment_length_projection(hidden).argmax(dim=-1)
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
        word_ids = self._derive_word_ids(input_ids)
        pooled = self._pool_words(hidden, word_ids)
        word_features = self._derive_word_features(
            input_ids, word_ids, pooled.shape[1]
        )
        enriched = pooled
        if self.config.use_lexical_word_features:
            for feature_index, embedding in enumerate(self.word_feature_embeddings):
                enriched = enriched + embedding(
                    word_features[:, :, feature_index].long()
                )
        word_probabilities = self.word_projection(enriched).softmax(dim=-1)

        tensors = [
            input_ids,
            operations,
            segment_lengths,
            segments,
            word_ids,
            word_features,
            word_probabilities,
        ]
        (
            input_rows,
            operation_rows,
            length_rows,
            segment_rows,
            word_id_rows,
            word_feature_rows,
            word_probability_rows,
        ) = [tensor.detach().cpu().tolist() for tensor in tensors]
        boundary_rows = boundaries.detach().cpu().tolist() if boundaries is not None else None
        boundary_confidence_rows = (
            boundary_confidence.detach().cpu().tolist()
            if boundary_confidence is not None
            else None
        )

        limit = int(max_length or self.config.max_position_embeddings)
        ignored = {
            self.config.pad_token_id,
            self.config.bos_token_id,
            self.config.eos_token_id,
        }
        generated_rows: list[list[int]] = []
        for row in range(len(input_rows)):
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
                    source_token == self.config.space_token_id
                    and boundary_is_confident
                    and boundary in {BOUNDARY_KEEP, BOUNDARY_DELETE}
                ):
                    if boundary == BOUNDARY_KEEP and (
                        not output or output[-1] != self.config.space_token_id
                    ):
                        output.append(self.config.space_token_id)
                    continue
                if (
                    source_token not in ignored
                    and source_token != self.config.space_token_id
                    and boundary_is_confident
                    and boundary == BOUNDARY_INSERT_BEFORE
                    and output
                    and output[-1] != self.config.space_token_id
                ):
                    output.append(self.config.space_token_id)

                operation = int(operation_rows[row][position])
                word_id = int(word_id_rows[row][position])
                force_keep = False
                if 0 <= word_id < len(word_probability_rows[row]):
                    features = word_feature_rows[row][word_id]
                    # French/entity status informs the learned word head, but
                    # does not by itself make a surface immutable: reviewed
                    # Gold contains misspelled French words and names too.
                    force_keep = bool(
                        self.config.use_hard_lexical_protection and features[2]
                    )
                    if force_keep or (
                        self.config.use_word_keep_gate
                        and word_probability_rows[row][word_id][WORD_KEEP]
                        >= self.config.word_keep_threshold
                    ):
                        operation = KEEP

                if operation == KEEP:
                    predicted_tokens = [source_token]
                elif operation == DELETE:
                    predicted_tokens = []
                else:
                    output_length = max(1, int(length_rows[row][position]))
                    predicted_tokens = [
                        int(value)
                        for value in segment_rows[row][position][:output_length]
                        if int(value) not in ignored
                    ]
                if (
                    source_token != self.config.space_token_id
                    and boundary_is_confident
                    and boundary in {BOUNDARY_NONE, BOUNDARY_INSERT_BEFORE}
                ):
                    while predicted_tokens and predicted_tokens[0] == self.config.space_token_id:
                        predicted_tokens.pop(0)
                output.extend(predicted_tokens)
                if len(output) >= max(limit - 1, 0):
                    break
            output = output[: max(limit - 1, 0)] + [self.config.eos_token_id]
            generated_rows.append(output)

        width = max(len(row) for row in generated_rows)
        generated = input_ids.new_full(
            (len(generated_rows), width), self.config.pad_token_id
        )
        for row_index, values in enumerate(generated_rows):
            generated[row_index, : len(values)] = torch.tensor(
                values, dtype=input_ids.dtype, device=input_ids.device
            )
        return generated


def _feature_lookup(vocabulary: dict[str, int]) -> tuple[dict[str, list[int]], dict[str, int], list[int]]:
    features: dict[str, list[int]] = {}
    observed_editable_keys: set[str] = set()

    def normalized_key(token: str) -> str | None:
        token = token.strip().strip(".,!?;:()[]{}\"“”‘’…").casefold()
        if not token or any(character.isspace() for character in token):
            return None
        if any(character not in vocabulary for character in token):
            return None
        return ",".join(str(vocabulary[character]) for character in token)

    if LINGUISTIC_ANNOTATIONS_PATH.is_file():
        frame = pd.read_csv(LINGUISTIC_ANNOTATIONS_PATH, keep_default_na=False)
        if "review_status" in frame:
            reviewed = frame.loc[frame["review_status"].eq("reviewed")]
            if not reviewed.empty:
                frame = reviewed
        for record in frame.to_dict("records"):
            language = str(record.get("source_language", "")).casefold()
            target_language = str(record.get("target_language", "")).casefold()
            entity = str(record.get("entity_type", "")).casefold() not in {
                "",
                "unknown",
                "none",
                "o",
            }
            protected = str(record.get("protected", "")).casefold() in {
                "yes",
                "true",
                "1",
                "oui",
            }
            informal = str(record.get("informal_token", ""))
            formal = str(record.get("formal_token", ""))
            surfaces_match = informal.strip().casefold() == formal.strip().casefold()
            for column, token in (
                ("informal_token", informal),
                ("formal_token", formal),
            ):
                key = normalized_key(token)
                if key is None:
                    continue
                if column == "informal_token" and not surfaces_match:
                    observed_editable_keys.add(key)
                # A corrected informal French/entity surface must remain
                # editable. The corrected formal surface, and identity
                # occurrences, are safe candidates for hard protection.
                surface_is_safe = protected and (
                    column == "formal_token" or surfaces_match
                )
                values = [
                    int(language == "fr" or target_language == "fr"),
                    int(entity),
                    int(surface_is_safe),
                ]
                previous = features.setdefault(key, [0, 0, 0])
                features[key] = [max(left, right) for left, right in zip(previous, values)]

        # An explicit correction is stronger evidence than an identity use of
        # the same spelling: keep the lexical flags, but do not hard-protect it.
        for key in observed_editable_keys:
            if key in features:
                features[key][2] = 0

    if SENEGALESE_SURNAMES_PATH.is_file():
        for line in SENEGALESE_SURNAMES_PATH.read_text(encoding="utf-8").splitlines():
            key = normalized_key(line)
            if key is not None:
                previous = features.setdefault(key, [0, 0, 0])
                features[key] = [previous[0], 1, 1]

    lowercase_map = {
        str(token_id): vocabulary.get(character.casefold(), token_id)
        for character, token_id in vocabulary.items()
        if len(character) == 1
    }
    strip_ids = [
        token_id
        for character, token_id in vocabulary.items()
        if len(character) == 1 and not character.isalnum() and character != "'"
    ]
    return features, lowercase_map, strip_ids


def _fingerprint(
    vocabulary: dict[str, int],
    architecture: TntHybridV3Architecture,
    feature_lookup: dict[str, list[int]],
    seed: int,
) -> str:
    payload = {
        "implementation": "wolof_tnt_hybrid_transformer_v3",
        "vocabulary": vocabulary,
        "architecture": asdict(architecture),
        "feature_lookup": feature_lookup,
        "seed": seed,
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def prepare_tnt_hybrid_v3_transformer(
    training_frames: Sequence[pd.DataFrame],
    *,
    output_root: Path,
    seed: int = 2026,
    architecture: TntHybridV3Architecture | None = None,
) -> Path:
    from tokenizers import Tokenizer
    from tokenizers.decoders import Fuse
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Split
    from tokenizers.processors import TemplateProcessing
    from transformers import PreTrainedTokenizerFast, set_seed

    architecture = architecture or TntHybridV3Architecture()
    texts = _training_texts(training_frames)
    vocabulary = build_character_vocabulary(texts)
    feature_lookup, lowercase_map, strip_ids = _feature_lookup(vocabulary)
    fingerprint = _fingerprint(vocabulary, architecture, feature_lookup, seed)
    output_dir = Path(output_root) / f"tnt_hybrid_v3_char_{fingerprint[:12]}"
    manifest_path = output_dir / "initialization_manifest.json"
    if manifest_path.is_file():
        print(f"[TNT-V3] Reusing initialization: {output_dir.resolve()}")
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
    config = TntHybridV3Config(
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
        segment_length_loss_weight=architecture.segment_length_loss_weight,
        boundary_loss_weight=architecture.boundary_loss_weight,
        word_loss_weight=architecture.word_loss_weight,
        operation_class_weights=architecture.operation_class_weights,
        segment_length_class_weights=architecture.segment_length_class_weights,
        boundary_class_weights=architecture.boundary_class_weights,
        word_class_weights=architecture.word_class_weights,
        boundary_focal_gamma=architecture.boundary_focal_gamma,
        label_smoothing=architecture.label_smoothing,
        use_boundary_inference=architecture.use_boundary_inference,
        use_word_keep_gate=architecture.use_word_keep_gate,
        use_lexical_word_features=architecture.use_lexical_word_features,
        use_hard_lexical_protection=architecture.use_hard_lexical_protection,
        boundary_confidence_threshold=architecture.boundary_confidence_threshold,
        word_keep_threshold=architecture.word_keep_threshold,
        word_feature_lookup=feature_lookup,
        lowercase_token_id_map=lowercase_map,
        word_edge_strip_token_ids=strip_ids,
        space_token_id=vocabulary.get(" ", -1),
        pad_token_id=vocabulary["<pad>"],
        bos_token_id=vocabulary["<s>"],
        eos_token_id=vocabulary["</s>"],
        decoder_start_token_id=vocabulary["<s>"],
    )
    set_seed(seed)
    model = TntHybridV3TransformerForConditionalGeneration(config)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    manifest_path.write_text(
        json.dumps(
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "fingerprint": fingerprint,
                "implementation": "TNT hybrid V3",
                "architecture": asdict(architecture),
                "seed": seed,
                "parameter_count": parameter_count,
                "word_feature_entries": len(feature_lookup),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"[TNT-V3] Created {parameter_count:,}-parameter model with "
        f"{len(feature_lookup):,} lexical feature entries: {output_dir.resolve()}"
    )
    return output_dir


def is_tnt_hybrid_v3_checkpoint(model_path: Path | str) -> bool:
    config_path = Path(model_path) / "config.json"
    if not config_path.is_file():
        return False
    try:
        return json.loads(config_path.read_text(encoding="utf-8")).get(
            "model_type"
        ) == TntHybridV3Config.model_type
    except (OSError, json.JSONDecodeError):
        return False
