"""Build a reproducible Beqi-style character-level Transformer initialization.

Beqi used the Transformer-base encoder-decoder from Vaswani et al. through
OpenNMT and found that character-level segmentation was its strongest setup.
This module builds the corresponding randomly initialized model with the
Hugging Face Marian implementation so it can use the existing M7 Trainer,
checkpoint, resume, and evaluation machinery.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


SPECIAL_TOKENS = ("<pad>", "<s>", "</s>", "<unk>")


@dataclass(frozen=True)
class BeqiTransformerArchitecture:
    """Transformer-base dimensions used for the Beqi-style diagnostic."""

    d_model: int = 512
    encoder_layers: int = 6
    decoder_layers: int = 6
    attention_heads: int = 8
    feed_forward_size: int = 2048
    dropout: float = 0.1
    max_position_embeddings: int = 256


def _training_texts(frames: Sequence[pd.DataFrame]) -> list[str]:
    texts: list[str] = []
    for frame in frames:
        missing = {"source_text", "target_text"} - set(frame.columns)
        if missing:
            raise ValueError(
                f"Character-vocabulary data is missing columns: {sorted(missing)}"
            )
        for column in ("source_text", "target_text"):
            if frame[column].isna().any():
                raise ValueError(f"Character-vocabulary data contains null {column}")
            texts.extend(frame[column].astype(str).tolist())
    if not texts:
        raise ValueError("At least one training sentence is required")
    return texts


def build_character_vocabulary(texts: Iterable[str]) -> dict[str, int]:
    """Return a deterministic character vocabulary with fixed special IDs."""
    characters = sorted({character for text in texts for character in str(text)})
    vocabulary = {token: index for index, token in enumerate(SPECIAL_TOKENS)}
    for character in characters:
        if character not in vocabulary:
            vocabulary[character] = len(vocabulary)
    return vocabulary


def _initialization_fingerprint(
    vocabulary: dict[str, int], architecture: BeqiTransformerArchitecture, seed: int
) -> str:
    payload = {
        "implementation": "marian_vanilla_transformer_character_level_v1",
        "vocabulary": vocabulary,
        "architecture": asdict(architecture),
        "seed": seed,
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def prepare_beqi_transformer(
    training_frames: Sequence[pd.DataFrame],
    *,
    output_root: Path,
    seed: int = 2026,
    architecture: BeqiTransformerArchitecture | None = None,
) -> Path:
    """Create or reuse a deterministic, randomly initialized Beqi-style model.

    Only training frames should be supplied. The directory name incorporates
    the character vocabulary, architecture, and seed, so changed training text
    creates a distinct initialization without overwriting earlier experiments.
    """
    from tokenizers import Tokenizer
    from tokenizers.decoders import Fuse
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Split
    from tokenizers.processors import TemplateProcessing
    from transformers import (
        MarianConfig,
        MarianMTModel,
        PreTrainedTokenizerFast,
        set_seed,
    )

    architecture = architecture or BeqiTransformerArchitecture()
    texts = _training_texts(training_frames)
    vocabulary = build_character_vocabulary(texts)
    fingerprint = _initialization_fingerprint(vocabulary, architecture, seed)
    output_dir = Path(output_root) / f"beqi_transformer_char_{fingerprint[:12]}"
    manifest_path = output_dir / "initialization_manifest.json"

    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("fingerprint") != fingerprint:
            raise RuntimeError(f"Initialization fingerprint mismatch: {output_dir}")
        required = ("config.json", "model.safetensors", "tokenizer.json")
        missing = [name for name in required if not (output_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(
                f"Incomplete Beqi-style initialization {output_dir}: {missing}"
            )
        print(f"[M7] Reusing Beqi-style initialization: {output_dir.resolve()}")
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

    config = MarianConfig(
        vocab_size=len(vocabulary),
        d_model=architecture.d_model,
        encoder_layers=architecture.encoder_layers,
        decoder_layers=architecture.decoder_layers,
        encoder_attention_heads=architecture.attention_heads,
        decoder_attention_heads=architecture.attention_heads,
        encoder_ffn_dim=architecture.feed_forward_size,
        decoder_ffn_dim=architecture.feed_forward_size,
        activation_function="relu",
        dropout=architecture.dropout,
        attention_dropout=architecture.dropout,
        activation_dropout=architecture.dropout,
        max_position_embeddings=architecture.max_position_embeddings,
        static_position_embeddings=True,
        scale_embedding=True,
        normalize_before=False,
        normalize_embedding=False,
        share_encoder_decoder_embeddings=True,
        pad_token_id=vocabulary["<pad>"],
        bos_token_id=vocabulary["<s>"],
        eos_token_id=vocabulary["</s>"],
        decoder_start_token_id=vocabulary["<pad>"],
        forced_eos_token_id=vocabulary["</s>"],
    )
    set_seed(seed)
    model = MarianMTModel(config)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "fingerprint": fingerprint,
        "implementation": "Hugging Face MarianMTModel initialized from scratch",
        "scope": "Beqi-style architecture diagnostic; not an exact Beqi reproduction",
        "segmentation": "character-level",
        "special_tokens": list(SPECIAL_TOKENS),
        "vocabulary_size": len(vocabulary),
        "architecture": asdict(architecture),
        "seed": seed,
        "parameter_count": parameter_count,
        "training_text_count_used_for_vocabulary": len(texts),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"[M7] Created Beqi-style character Transformer with "
        f"{parameter_count:,} parameters: {output_dir.resolve()}"
    )
    return output_dir
