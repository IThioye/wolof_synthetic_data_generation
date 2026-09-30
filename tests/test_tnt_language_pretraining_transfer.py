import torch

from src.modeling.tnt_hybrid_v3_language_pretraining_gate import (
    transfer_encoder_state,
)
from src.modeling.tnt_hybrid_v3_transformer import (
    TntHybridV3Config,
    TntHybridV3TransformerForConditionalGeneration,
)
from src.modeling.wolof_encoder_pretraining import (
    EncoderPretrainingConfig,
    MaskedCharacterEncoder,
)


def test_language_encoder_transfer_maps_embeddings_and_exact_encoder():
    source_vocabulary = {
        "<pad>": 0,
        "<s>": 1,
        "</s>": 2,
        "<unk>": 3,
        "<mask>": 4,
        "a": 5,
        "b": 6,
        " ": 7,
    }
    target_vocabulary = {
        "<pad>": 0,
        "<s>": 1,
        "</s>": 2,
        "<unk>": 3,
        "a": 4,
        "b": 5,
        " ": 6,
    }
    architecture = EncoderPretrainingConfig(
        d_model=16,
        attention_heads=4,
        encoder_layers=1,
        feed_forward_size=32,
        max_position_embeddings=32,
        mixed_precision="none",
    )
    source = MaskedCharacterEncoder(len(source_vocabulary), architecture)
    target_config = TntHybridV3Config(
        vocab_size=len(target_vocabulary),
        d_model=16,
        attention_heads=4,
        encoder_layers=1,
        feed_forward_size=32,
        max_position_embeddings=32,
        output_slots_per_source=2,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        decoder_start_token_id=1,
        space_token_id=6,
    )
    target = TntHybridV3TransformerForConditionalGeneration(target_config)
    source_state = {
        key: value.detach().clone()
        for key, value in source.state_dict().items()
        if key.startswith(("character_embedding.", "position_embedding.", "encoder."))
    }
    audit = transfer_encoder_state(
        target, source_state, source_vocabulary, target_vocabulary
    )
    assert audit["embedding_rows_missing"] == 0
    assert audit["embedding_rows_copied"] == len(target_vocabulary)
    assert torch.equal(
        target.position_embedding.weight, source.position_embedding.weight
    )
    for token, target_index in target_vocabulary.items():
        source_index = source_vocabulary[token]
        assert torch.equal(
            target.character_embedding.weight[target_index],
            source.character_embedding.weight[source_index],
        )
