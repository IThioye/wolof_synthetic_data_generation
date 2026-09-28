"""Sentence-level adaptation of the user's compact Wolof Transformer.

The architecture mirrors Wolof-Auto-Corrector's ``TransformerSpell``: shared
character embeddings, learned positions, PyTorch's encoder-decoder Transformer,
and a character projection head.  The wrapper adds the Hugging Face-compatible
interfaces required by M7 for training, generation, checkpointing, and resume.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import pandas as pd

from src.modeling.beqi_transformer import (
    SPECIAL_TOKENS,
    _training_texts,
    build_character_vocabulary,
)


@dataclass(frozen=True)
class CompactTransformerArchitecture:
    """Defaults from the earlier Wolof-Auto-Corrector Transformer."""

    d_model: int = 64
    attention_heads: int = 4
    encoder_layers: int = 2
    decoder_layers: int = 2
    # Keep the FFN proportional to the attention width.  The earlier 2,048
    # default was 32x d_model and dominated this deliberately small model.
    feed_forward_size: int = 256
    dropout: float = 0.1
    max_position_embeddings: int = 256
    # Optional inference guardrail.  When set, each generated sentence is
    # limited relative to its non-padding source length instead of allowing a
    # failed decoder to fill the global positional limit.
    generation_length_ratio: float | None = None
    generation_length_margin: int = 0
    use_incremental_generation: bool = True
    copy_aware: bool = False
    copy_position_bias: float = 10.0
    # New experiments should prefer content-only copying ("none").  Absolute
    # mode is retained for reproducibility and relative mode is an ablation.
    copy_position_mode: str = "none"
    copy_gate_bias: float = -3.0
    copy_regularization_strength: float = 0.0
    edit_position_weight: float = 1.0
    label_smoothing: float = 0.0
    tie_character_embeddings: bool = True
    scheduled_sampling_probability: float = 0.0


def _shift_labels_right(labels, *, start_token_id: int, pad_token_id: int):
    import torch

    shifted = labels.new_full(labels.shape, pad_token_id)
    shifted[:, 0] = start_token_id
    shifted[:, 1:] = labels[:, :-1]
    shifted.masked_fill_(shifted == -100, pad_token_id)
    return shifted


def _causal_mask(length: int, device):
    import torch

    return torch.triu(
        torch.ones((length, length), dtype=torch.bool, device=device), diagonal=1
    )


def _model_components():
    """Import optional heavy dependencies only when the model is used."""
    import torch
    import torch.nn as nn
    from transformers import PretrainedConfig, PreTrainedModel
    from transformers.generation import GenerationMixin
    from transformers.modeling_outputs import Seq2SeqLMOutput

    return torch, nn, PretrainedConfig, PreTrainedModel, GenerationMixin, Seq2SeqLMOutput


torch, nn, PretrainedConfig, PreTrainedModel, GenerationMixin, Seq2SeqLMOutput = (
    _model_components()
)


class CompactTransformerConfig(PretrainedConfig):
    model_type = "wolof_compact_transformer"

    def __init__(
        self,
        vocab_size: int = 128,
        d_model: int = 64,
        attention_heads: int = 4,
        encoder_layers: int = 2,
        decoder_layers: int = 2,
        feed_forward_size: int = 256,
        dropout: float = 0.1,
        max_position_embeddings: int = 256,
        generation_length_ratio: float | None = None,
        generation_length_margin: int = 0,
        # False preserves generation from checkpoints created before the
        # incremental decoder existed.  New architectures pass True explicitly.
        use_incremental_generation: bool = False,
        copy_aware: bool = False,
        copy_position_bias: float = 10.0,
        # "absolute" preserves the behavior of copy-aware checkpoints created
        # before copy_position_mode existed.  New runs pass their mode explicitly.
        copy_position_mode: str = "absolute",
        copy_gate_bias: float = -3.0,
        copy_regularization_strength: float = 0.0,
        edit_position_weight: float = 1.0,
        label_smoothing: float = 0.0,
        tie_character_embeddings: bool = False,
        scheduled_sampling_probability: float = 0.0,
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
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.hidden_size = d_model
        self.attention_heads = attention_heads
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.feed_forward_size = feed_forward_size
        self.dropout = dropout
        self.max_position_embeddings = max_position_embeddings
        self.generation_length_ratio = generation_length_ratio
        self.generation_length_margin = generation_length_margin
        self.use_incremental_generation = use_incremental_generation
        self.copy_aware = copy_aware
        self.copy_position_bias = copy_position_bias
        if copy_position_mode not in {"absolute", "relative", "none"}:
            raise ValueError(
                "copy_position_mode must be 'absolute', 'relative', or 'none'"
            )
        if edit_position_weight < 1.0:
            raise ValueError("edit_position_weight must be at least 1.0")
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in [0, 1)")
        self.copy_position_mode = copy_position_mode
        self.copy_gate_bias = copy_gate_bias
        self.copy_regularization_strength = copy_regularization_strength
        self.edit_position_weight = edit_position_weight
        self.label_smoothing = label_smoothing
        self.tie_character_embeddings = tie_character_embeddings
        # Transformers uses this standard flag when resolving the declared
        # tied-weight mapping during initialization and checkpoint loading.
        self.tie_word_embeddings = tie_character_embeddings
        self.scheduled_sampling_probability = scheduled_sampling_probability


class CompactTransformerForConditionalGeneration(PreTrainedModel, GenerationMixin):
    """Compact character Transformer with greedy sentence generation."""

    config_class = CompactTransformerConfig
    base_model_prefix = "compact_transformer"
    main_input_name = "input_ids"
    _tied_weights_keys = {
        "output_projection.weight": "character_embedding.weight"
    }

    def __init__(self, config: CompactTransformerConfig):
        super().__init__(config)
        self.pad_idx = config.pad_token_id
        self.character_embedding = nn.Embedding(
            config.vocab_size, config.d_model, padding_idx=config.pad_token_id
        )
        self.position_embedding = nn.Embedding(
            config.max_position_embeddings, config.d_model
        )
        self.transformer = nn.Transformer(
            d_model=config.d_model,
            nhead=config.attention_heads,
            num_encoder_layers=config.encoder_layers,
            num_decoder_layers=config.decoder_layers,
            dim_feedforward=config.feed_forward_size,
            dropout=config.dropout,
            activation="relu",
            batch_first=True,
            norm_first=False,
        )
        self.output_projection = nn.Linear(config.d_model, config.vocab_size)
        self.copy_gate = (
            nn.Linear(config.d_model * 3, 1) if config.copy_aware else None
        )
        self.copy_position_strength = (
            nn.Parameter(torch.tensor(float(config.copy_position_bias)))
            if config.copy_aware
            else None
        )
        self.post_init()
        if self.copy_gate is not None:
            # Begin with a strong but non-exclusive preference for copying.
            # The model can increase the generation probability wherever an
            # insertion or replacement is required.
            nn.init.zeros_(self.copy_gate.weight)
            nn.init.constant_(self.copy_gate.bias, config.copy_gate_bias)

    def get_input_embeddings(self):
        return self.character_embedding

    def set_input_embeddings(self, value):
        self.character_embedding = value
        if getattr(self.config, "tie_character_embeddings", False):
            self.tie_weights()

    def get_output_embeddings(self):
        return self.output_projection

    def set_output_embeddings(self, value):
        self.output_projection = value
        if getattr(self.config, "tie_character_embeddings", False):
            self.tie_weights()

    def _embed(self, token_ids):
        length = token_ids.shape[1]
        if length > self.config.max_position_embeddings:
            raise ValueError(
                f"Sequence length {length} exceeds compact Transformer limit "
                f"{self.config.max_position_embeddings}"
            )
        positions = torch.arange(length, device=token_ids.device).unsqueeze(0)
        return self.character_embedding(token_ids) + self.position_embedding(positions)

    @staticmethod
    def _causal_relative_centers(copy_source_positions, source_length: int):
        """Center step t after the latest aligned source position before t.

        ``copy_source_positions`` is produced only from the training pair's
        Levenshtein alignment.  Insertions use -1 and therefore do not advance
        the running source location.  The shift makes this causal: the current
        target label never determines its own attention center.
        """
        valid_positions = copy_source_positions.clamp(min=-1)
        running_positions = torch.cummax(valid_positions, dim=1).values
        previous_positions = torch.full_like(running_positions, -1)
        previous_positions[:, 1:] = running_positions[:, :-1]
        return (previous_positions + 1).clamp(min=0, max=max(source_length - 1, 0))

    def _output_logits(
        self,
        decoder_output,
        memory,
        input_ids,
        source_padding,
        target_embeddings,
        target_position_offset: int = 0,
        copy_position_centers=None,
        return_generation_gate: bool = False,
    ):
        """Mix normal generation with a source-character pointer distribution."""
        if self.copy_gate is None:
            logits = self.output_projection(decoder_output)
            return (logits, None, None) if return_generation_gate else logits

        import math

        scores = torch.matmul(decoder_output, memory.transpose(1, 2)) / math.sqrt(
            self.config.d_model
        )
        position_mode = getattr(self.config, "copy_position_mode", "absolute")
        if self.copy_position_strength is not None and position_mode != "none":
            import torch.nn.functional as functional

            position_bias = functional.softplus(self.copy_position_strength)
            if position_mode == "absolute":
                position_centers = torch.arange(
                    decoder_output.shape[1], device=decoder_output.device
                ).add(target_position_offset).view(1, -1)
            elif position_mode == "relative":
                if copy_position_centers is None:
                    raise ValueError(
                        "relative copy positioning requires causal position centers"
                    )
                position_centers = copy_position_centers.to(
                    device=decoder_output.device, dtype=torch.long
                )
                if position_centers.ndim == 1:
                    position_centers = position_centers.unsqueeze(1)
                if position_centers.shape != decoder_output.shape[:2]:
                    raise ValueError(
                        "copy_position_centers must match decoder batch/length"
                    )
            else:  # validated by CompactTransformerConfig
                raise ValueError(f"Unsupported copy_position_mode: {position_mode}")
            source_positions = torch.arange(
                memory.shape[1], device=memory.device
            ).view(1, 1, -1)
            scores = scores - position_bias * (
                position_centers.unsqueeze(-1) - source_positions
            ).abs()
        scores = scores.masked_fill(source_padding.unsqueeze(1), torch.finfo(scores.dtype).min)
        copy_attention = torch.softmax(scores, dim=-1)
        copy_probabilities = decoder_output.new_zeros(
            decoder_output.shape[0], decoder_output.shape[1], self.config.vocab_size
        )
        copy_indices = input_ids.unsqueeze(1).expand(
            -1, decoder_output.shape[1], -1
        )
        copy_probabilities.scatter_add_(2, copy_indices, copy_attention)

        generated_probabilities = torch.softmax(
            self.output_projection(decoder_output), dim=-1
        )
        copy_context = torch.matmul(copy_attention, memory)
        generation_gate = torch.sigmoid(
            self.copy_gate(
                torch.cat(
                    [decoder_output, copy_context, target_embeddings], dim=-1
                )
            )
        )
        probabilities = (
            generation_gate * generated_probabilities
            + (1.0 - generation_gate) * copy_probabilities
        )
        # Cross entropy accepts these normalized log probabilities as logits.
        logits = probabilities.clamp_min(1e-9).log()
        if return_generation_gate:
            return logits, generation_gate, copy_attention
        return logits

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        labels=None,
        edit_mask=None,
        copy_source_positions=None,
        return_dict=True,
        **kwargs,
    ):
        import torch.nn.functional as functional

        if input_ids is None:
            raise ValueError("input_ids are required")
        created_decoder_inputs = decoder_input_ids is None
        if created_decoder_inputs:
            if labels is None:
                raise ValueError("decoder_input_ids or labels are required")
            decoder_input_ids = _shift_labels_right(
                labels,
                start_token_id=self.config.decoder_start_token_id,
                pad_token_id=self.config.pad_token_id,
            )
        source_padding = (
            attention_mask.eq(0)
            if attention_mask is not None
            else input_ids.eq(self.config.pad_token_id)
        )
        source = self._embed(input_ids)
        memory = self.transformer.encoder(
            source,
            src_key_padding_mask=source_padding,
        )
        sampling_probability = float(
            getattr(self.config, "scheduled_sampling_probability", 0.0)
        )
        if not 0.0 <= sampling_probability <= 1.0:
            raise ValueError("scheduled_sampling_probability must be between 0 and 1")
        if (
            self.training
            and created_decoder_inputs
            and labels is not None
            and sampling_probability > 0.0
            and decoder_input_ids.shape[1] > 1
        ):
            preliminary_padding = decoder_input_ids.eq(self.config.pad_token_id)
            with torch.no_grad():
                preliminary_target = self._embed(decoder_input_ids)
                preliminary_output = self.transformer.decoder(
                    preliminary_target,
                    memory,
                    tgt_mask=_causal_mask(
                        preliminary_target.shape[1], preliminary_target.device
                    ),
                    tgt_key_padding_mask=preliminary_padding,
                    memory_key_padding_mask=source_padding,
                )
                preliminary_logits = self._output_logits(
                    preliminary_output,
                    memory,
                    input_ids,
                    source_padding,
                    preliminary_target,
                    copy_position_centers=(
                        self._causal_relative_centers(
                            copy_source_positions, memory.shape[1]
                        )
                        if getattr(self.config, "copy_position_mode", "absolute")
                        == "relative"
                        and copy_source_positions is not None
                        else None
                    ),
                )
                preliminary_predictions = preliminary_logits.argmax(dim=-1)
            valid_previous_labels = labels[:, :-1].ne(-100)
            sampling_mask = (
                torch.rand(
                    valid_previous_labels.shape,
                    device=decoder_input_ids.device,
                )
                < sampling_probability
            ) & valid_previous_labels
            mixed_decoder_inputs = decoder_input_ids.clone()
            mixed_decoder_inputs[:, 1:] = torch.where(
                sampling_mask,
                preliminary_predictions[:, :-1],
                mixed_decoder_inputs[:, 1:],
            )
            decoder_input_ids = mixed_decoder_inputs
        target_padding = (
            decoder_attention_mask.eq(0)
            if decoder_attention_mask is not None
            else decoder_input_ids.eq(self.config.pad_token_id)
        )
        target = self._embed(decoder_input_ids)
        output = self.transformer.decoder(
            target,
            memory,
            tgt_mask=_causal_mask(target.shape[1], target.device),
            tgt_key_padding_mask=target_padding,
            memory_key_padding_mask=source_padding,
        )
        relative_centers = None
        if (
            self.copy_gate is not None
            and getattr(self.config, "copy_position_mode", "absolute") == "relative"
        ):
            if copy_source_positions is None:
                raise ValueError(
                    "copy_source_positions are required for relative teacher-forced training"
                )
            relative_centers = self._causal_relative_centers(
                copy_source_positions, memory.shape[1]
            )
        logits, generation_gate, _ = self._output_logits(
            output,
            memory,
            input_ids,
            source_padding,
            target,
            copy_position_centers=relative_centers,
            return_generation_gate=True,
        )
        loss = None
        if labels is not None:
            token_losses = functional.cross_entropy(
                logits.reshape(-1, self.config.vocab_size),
                labels.reshape(-1),
                ignore_index=-100,
                reduction="none",
                label_smoothing=float(getattr(self.config, "label_smoothing", 0.0)),
            )
            valid_labels = labels.reshape(-1).ne(-100)
            position_weights = token_losses.new_ones(token_losses.shape)
            edit_position_weight = float(
                getattr(self.config, "edit_position_weight", 1.0)
            )
            if edit_position_weight < 1.0:
                raise ValueError("edit_position_weight must be at least 1.0")
            if edit_mask is not None and edit_position_weight != 1.0:
                if edit_mask.shape != labels.shape:
                    raise ValueError("edit_mask must have the same shape as labels")
                position_weights = torch.where(
                    edit_mask.reshape(-1).bool(),
                    position_weights.new_full(position_weights.shape, edit_position_weight),
                    position_weights,
                )
            weighted_valid = position_weights * valid_labels.to(position_weights.dtype)
            loss = (token_losses * weighted_valid).sum() / weighted_valid.sum().clamp_min(1.0)
            regularization_strength = float(
                getattr(self.config, "copy_regularization_strength", 0.0)
            )
            if regularization_strength < 0:
                raise ValueError("copy_regularization_strength cannot be negative")
            if regularization_strength and generation_gate is not None:
                aligned_length = min(labels.shape[1], input_ids.shape[1])
                aligned_labels = labels[:, :aligned_length]
                aligned_source = input_ids[:, :aligned_length]
                copyable = (
                    aligned_labels.ne(-100)
                    & aligned_source.ne(self.config.pad_token_id)
                    & aligned_labels.eq(aligned_source)
                )
                if bool(copyable.any()):
                    aligned_gate = generation_gate[:, :aligned_length, 0]
                    copy_penalty = -torch.log1p(
                        -aligned_gate.clamp(max=1.0 - 1e-6)
                    )[copyable].mean()
                    loss = loss + regularization_strength * copy_penalty
        if not return_dict:
            values = (logits,)
            return ((loss,) + values) if loss is not None else values
        return Seq2SeqLMOutput(loss=loss, logits=logits)

    def _incremental_decoder_step(
        self,
        target_step,
        memory,
        source_padding,
        layer_caches,
    ):
        """Decode one position while caching each layer's past self-attention input."""
        output = target_step
        updated_caches = []
        for layer, previous in zip(self.transformer.decoder.layers, layer_caches):
            if layer.norm_first:
                self_attention_input = layer.norm1(output)
                keys_and_values = (
                    torch.cat([previous, self_attention_input], dim=1)
                    if previous is not None
                    else self_attention_input
                )
                updated_caches.append(keys_and_values)
                attended = layer.self_attn(
                    self_attention_input,
                    keys_and_values,
                    keys_and_values,
                    need_weights=False,
                )[0]
                output = output + layer.dropout1(attended)
                cross_input = layer.norm2(output)
                crossed = layer.multihead_attn(
                    cross_input,
                    memory,
                    memory,
                    key_padding_mask=source_padding,
                    need_weights=False,
                )[0]
                output = output + layer.dropout2(crossed)
                feed_forward_input = layer.norm3(output)
                feed_forward = layer.linear2(
                    layer.dropout(
                        layer.activation(layer.linear1(feed_forward_input))
                    )
                )
                output = output + layer.dropout3(feed_forward)
            else:
                self_attention_input = output
                keys_and_values = (
                    torch.cat([previous, self_attention_input], dim=1)
                    if previous is not None
                    else self_attention_input
                )
                updated_caches.append(keys_and_values)
                attended = layer.self_attn(
                    self_attention_input,
                    keys_and_values,
                    keys_and_values,
                    need_weights=False,
                )[0]
                output = layer.norm1(output + layer.dropout1(attended))
                crossed = layer.multihead_attn(
                    output,
                    memory,
                    memory,
                    key_padding_mask=source_padding,
                    need_weights=False,
                )[0]
                output = layer.norm2(output + layer.dropout2(crossed))
                feed_forward = layer.linear2(
                    layer.dropout(layer.activation(layer.linear1(output)))
                )
                output = layer.norm3(output + layer.dropout3(feed_forward))
        if self.transformer.decoder.norm is not None:
            output = self.transformer.decoder.norm(output)
        return output, updated_caches

    def _generate_incrementally(
        self,
        *,
        input_ids,
        memory,
        source_padding,
        row_limits,
        limit: int,
    ):
        """Greedy decoding with cached prefixes and active-row compaction."""
        batch_size = input_ids.shape[0]
        generated = input_ids.new_full(
            (batch_size, limit), self.config.pad_token_id
        )
        generated[:, 0] = self.config.decoder_start_token_id
        if limit <= 1:
            return generated[:, :1]

        active_rows = torch.arange(batch_size, device=input_ids.device)
        active_input_ids = input_ids
        active_memory = memory
        active_source_padding = source_padding
        active_limits = row_limits
        relative_copy_positions = input_ids.new_full((batch_size,), -1)
        current_tokens = input_ids.new_full(
            (batch_size, 1), self.config.decoder_start_token_id
        )
        layer_caches = [None] * len(self.transformer.decoder.layers)
        produced_length = 1

        for target_position in range(limit - 1):
            position_ids = current_tokens.new_full(
                current_tokens.shape, target_position
            )
            target_step = (
                self.character_embedding(current_tokens)
                + self.position_embedding(position_ids)
            )
            decoded_step, layer_caches = self._incremental_decoder_step(
                target_step,
                active_memory,
                active_source_padding,
                layer_caches,
            )
            position_mode = getattr(self.config, "copy_position_mode", "absolute")
            relative_centers = (
                (relative_copy_positions + 1).clamp(
                    min=0, max=max(active_memory.shape[1] - 1, 0)
                )
                if position_mode == "relative"
                else None
            )
            step_logits, generation_gate, copy_attention = self._output_logits(
                decoded_step,
                active_memory,
                active_input_ids,
                active_source_padding,
                target_step,
                target_position_offset=target_position,
                copy_position_centers=relative_centers,
                return_generation_gate=True,
            )
            next_tokens = step_logits[:, -1].argmax(dim=-1)
            generated[active_rows, target_position + 1] = next_tokens
            produced_length = target_position + 2

            if position_mode == "relative":
                assert generation_gate is not None and copy_attention is not None
                copied_rows = generation_gate[:, -1, 0].lt(0.5)
                attended_positions = copy_attention[:, -1].argmax(dim=-1)
                relative_copy_positions = torch.where(
                    copied_rows, attended_positions, relative_copy_positions
                )

            completed = next_tokens.eq(self.config.eos_token_id)
            completed |= produced_length >= active_limits
            retained = ~completed
            if not bool(retained.any()):
                break

            active_rows = active_rows[retained]
            active_input_ids = active_input_ids[retained]
            active_memory = active_memory[retained]
            active_source_padding = active_source_padding[retained]
            active_limits = active_limits[retained]
            relative_copy_positions = relative_copy_positions[retained]
            current_tokens = next_tokens[retained].unsqueeze(1)
            layer_caches = [cache[retained] for cache in layer_caches]

        return generated[:, :produced_length]

    @torch.no_grad()
    def generate(
        self,
        input_ids=None,
        attention_mask=None,
        max_length=None,
        num_beams=1,
        **kwargs,
    ):
        """Greedy decoding compatible with the M7 Seq2SeqTrainer calls."""
        import torch

        if input_ids is None:
            raise ValueError("input_ids are required")
        if num_beams not in (None, 1):
            raise NotImplementedError(
                "The compact diagnostic currently supports greedy decoding only"
            )
        limit = min(
            int(max_length or self.config.max_position_embeddings),
            self.config.max_position_embeddings,
        )
        source_padding = (
            attention_mask.eq(0)
            if attention_mask is not None
            else input_ids.eq(self.config.pad_token_id)
        )
        source = self._embed(input_ids)
        memory = self.transformer.encoder(
            source, src_key_padding_mask=source_padding
        )
        generated = input_ids.new_full(
            (input_ids.shape[0], 1), self.config.decoder_start_token_id
        )
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        relative_copy_positions = input_ids.new_full(
            (input_ids.shape[0],), -1
        )
        length_ratio = getattr(self.config, "generation_length_ratio", None)
        if length_ratio is not None:
            if float(length_ratio) <= 0:
                raise ValueError("generation_length_ratio must be positive")
            margin = int(getattr(self.config, "generation_length_margin", 0))
            if margin < 0:
                raise ValueError("generation_length_margin cannot be negative")
            source_lengths = (
                attention_mask.sum(dim=1)
                if attention_mask is not None
                else input_ids.ne(self.config.pad_token_id).sum(dim=1)
            )
            # The row limit includes the decoder start token.  At least one
            # generated character and one possible EOS position are retained.
            row_limits = (
                torch.ceil(source_lengths.float() * float(length_ratio)).long()
                + margin
                + 1
            ).clamp(min=3, max=limit)
        else:
            row_limits = input_ids.new_full((input_ids.shape[0],), limit)
        if bool(getattr(self.config, "use_incremental_generation", False)):
            return self._generate_incrementally(
                input_ids=input_ids,
                memory=memory,
                source_padding=source_padding,
                row_limits=row_limits,
                limit=limit,
            )
        for _ in range(max(limit - 1, 0)):
            target = self._embed(generated)
            decoded = self.transformer.decoder(
                target,
                memory,
                tgt_mask=_causal_mask(target.shape[1], target.device),
                tgt_key_padding_mask=generated.eq(self.config.pad_token_id),
                memory_key_padding_mask=source_padding,
            )
            position_mode = getattr(self.config, "copy_position_mode", "absolute")
            relative_centers = (
                (relative_copy_positions + 1).clamp(
                    min=0, max=max(memory.shape[1] - 1, 0)
                )
                if position_mode == "relative"
                else None
            )
            step_logits, generation_gate, copy_attention = self._output_logits(
                decoded[:, -1:],
                memory,
                input_ids,
                source_padding,
                target[:, -1:],
                target_position_offset=decoded.shape[1] - 1,
                copy_position_centers=relative_centers,
                return_generation_gate=True,
            )
            next_token = step_logits[:, -1].argmax(dim=-1)
            if position_mode == "relative":
                # Advance only when the mixture attributes this step primarily
                # to copying.  Generated insertions leave the pointer in place.
                assert generation_gate is not None and copy_attention is not None
                copied_rows = generation_gate[:, -1, 0].lt(0.5) & ~finished
                attended_positions = copy_attention[:, -1].argmax(dim=-1)
                relative_copy_positions = torch.where(
                    copied_rows, attended_positions, relative_copy_positions
                )
            next_token = torch.where(
                finished,
                torch.full_like(next_token, self.config.pad_token_id),
                next_token,
            )
            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
            finished |= next_token.eq(self.config.eos_token_id)
            finished |= generated.shape[1] >= row_limits
            if bool(finished.all()):
                break
        return generated


def is_compact_transformer_checkpoint(model_path: Path | str) -> bool:
    path = Path(model_path) / "config.json"
    if not path.is_file():
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("model_type") == (
            CompactTransformerConfig.model_type
        )
    except (OSError, json.JSONDecodeError):
        return False


def compact_transformer_model_class(model_path: Path | str, fallback):
    """Return the local custom class for its checkpoints, otherwise fallback."""
    return (
        CompactTransformerForConditionalGeneration
        if is_compact_transformer_checkpoint(model_path)
        else fallback
    )


def _initialization_fingerprint(
    vocabulary: dict[str, int],
    architecture: CompactTransformerArchitecture,
    seed: int,
    warm_start_sha256: str | None = None,
) -> str:
    architecture_payload = asdict(architecture)
    # Preserve initialization fingerprints from before cached generation was
    # introduced when the legacy path is explicitly requested.
    if not architecture.use_incremental_generation:
        architecture_payload.pop("use_incremental_generation")
    # Keep fingerprints of existing non-copy-aware initializations stable.
    if not architecture.copy_aware:
        architecture_payload.pop("copy_aware")
        architecture_payload.pop("copy_position_bias")
        architecture_payload.pop("copy_position_mode")
        architecture_payload.pop("copy_gate_bias")
        architecture_payload.pop("copy_regularization_strength")
        architecture_payload.pop("scheduled_sampling_probability")
    payload: dict[str, object] = {
        "implementation": "wolof_auto_corrector_compact_sentence_v1",
        "vocabulary": vocabulary,
        "architecture": architecture_payload,
        "seed": seed,
    }
    if warm_start_sha256 is not None:
        payload["warm_start_sha256"] = warm_start_sha256
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def prepare_compact_transformer(
    training_frames: Sequence[pd.DataFrame],
    *,
    output_root: Path,
    seed: int = 2026,
    architecture: CompactTransformerArchitecture | None = None,
    warm_start_checkpoint: Path | None = None,
) -> Path:
    """Create or reuse the deterministic compact character model initialization."""
    from tokenizers import Tokenizer
    from tokenizers.decoders import Fuse
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Split
    from tokenizers.processors import TemplateProcessing
    from transformers import PreTrainedTokenizerFast, set_seed

    architecture = architecture or CompactTransformerArchitecture()
    texts = _training_texts(training_frames)
    vocabulary = build_character_vocabulary(texts)
    warm_start_checkpoint = (
        Path(warm_start_checkpoint) if warm_start_checkpoint is not None else None
    )
    warm_start_sha256 = None
    if warm_start_checkpoint is not None:
        weights_path = warm_start_checkpoint / "model.safetensors"
        if not weights_path.is_file():
            raise FileNotFoundError(f"Warm-start weights are unavailable: {weights_path}")
        warm_start_sha256 = hashlib.sha256(weights_path.read_bytes()).hexdigest()
    fingerprint = _initialization_fingerprint(
        vocabulary, architecture, seed, warm_start_sha256
    )
    output_dir = Path(output_root) / f"compact_transformer_char_{fingerprint[:12]}"
    manifest_path = output_dir / "initialization_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("fingerprint") != fingerprint:
            raise RuntimeError(f"Initialization fingerprint mismatch: {output_dir}")
        required = ("config.json", "model.safetensors", "tokenizer.json")
        missing = [name for name in required if not (output_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(
                f"Incomplete compact Transformer initialization {output_dir}: {missing}"
            )
        print(f"[M7] Reusing compact Transformer initialization: {output_dir.resolve()}")
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
    config = CompactTransformerConfig(
        vocab_size=len(vocabulary),
        d_model=architecture.d_model,
        attention_heads=architecture.attention_heads,
        encoder_layers=architecture.encoder_layers,
        decoder_layers=architecture.decoder_layers,
        feed_forward_size=architecture.feed_forward_size,
        dropout=architecture.dropout,
        max_position_embeddings=architecture.max_position_embeddings,
        generation_length_ratio=architecture.generation_length_ratio,
        generation_length_margin=architecture.generation_length_margin,
        use_incremental_generation=architecture.use_incremental_generation,
        copy_aware=architecture.copy_aware,
        copy_position_bias=architecture.copy_position_bias,
        copy_position_mode=architecture.copy_position_mode,
        copy_gate_bias=architecture.copy_gate_bias,
        copy_regularization_strength=architecture.copy_regularization_strength,
        edit_position_weight=architecture.edit_position_weight,
        label_smoothing=architecture.label_smoothing,
        tie_character_embeddings=architecture.tie_character_embeddings,
        scheduled_sampling_probability=architecture.scheduled_sampling_probability,
        pad_token_id=vocabulary["<pad>"],
        bos_token_id=vocabulary["<s>"],
        eos_token_id=vocabulary["</s>"],
        decoder_start_token_id=vocabulary["<s>"],
    )
    set_seed(seed)
    model = CompactTransformerForConditionalGeneration(config)
    warm_start_missing_keys: list[str] = []
    if warm_start_checkpoint is not None:
        source_tokenizer = PreTrainedTokenizerFast.from_pretrained(
            warm_start_checkpoint
        )
        if source_tokenizer.get_vocab() != tokenizer.get_vocab():
            raise ValueError(
                "Warm-start checkpoint and copy-aware model use different "
                "character vocabularies"
            )
        source_model = CompactTransformerForConditionalGeneration.from_pretrained(
            warm_start_checkpoint
        )
        incompatible = model.load_state_dict(source_model.state_dict(), strict=False)
        warm_start_missing_keys = sorted(incompatible.missing_keys)
        unexpected = sorted(incompatible.unexpected_keys)
        allowed_missing = {
            "copy_gate.weight", "copy_gate.bias", "copy_position_strength"
        }
        if set(warm_start_missing_keys) - allowed_missing or unexpected:
            raise RuntimeError(
                "Unexpected warm-start incompatibility: "
                f"missing={warm_start_missing_keys}, unexpected={unexpected}"
            )
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "fingerprint": fingerprint,
        "implementation": "sentence adaptation of Wolof-Auto-Corrector TransformerSpell",
        "segmentation": "character-level",
        "vocabulary_size": len(vocabulary),
        "architecture": asdict(architecture),
        "seed": seed,
        "parameter_count": parameter_count,
        "training_text_count_used_for_vocabulary": len(texts),
        "warm_start_checkpoint": (
            str(warm_start_checkpoint.resolve())
            if warm_start_checkpoint is not None
            else None
        ),
        "warm_start_model_sha256": warm_start_sha256,
        "warm_start_missing_keys": warm_start_missing_keys,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"[M7] Created compact character Transformer with {parameter_count:,} "
        f"parameters: {output_dir.resolve()}"
    )
    return output_dir
