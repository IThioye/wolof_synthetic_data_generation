import pandas as pd
import torch

from src.modeling.wolof_encoder_pretraining import (
    CharacterCorpusDataset,
    EncoderPretrainingConfig,
    MaskedCharacterCollator,
    MaskedCharacterEncoder,
    build_pretraining_vocabulary,
    domain_sampling_weights,
    parameter_report,
)


def _frame():
    return pd.DataFrame(
        {
            "source_id": ["news:1", "youtube:1", "youtube:2"],
            "domain": ["formal_news", "informal_youtube", "informal_youtube"],
            "text": ["Ñu dem", "dagn dem", "waaw kay"],
        }
    )


def test_masking_is_reproducible_and_never_masks_eos():
    frame = _frame()
    vocabulary = build_pretraining_vocabulary(frame["text"])
    dataset = CharacterCorpusDataset(frame, vocabulary, 32)
    kwargs = dict(mask_probability=0.4, min_span=1, max_span=2, seed=17)
    first = MaskedCharacterCollator(vocabulary, **kwargs)([dataset[0], dataset[1]])
    second = MaskedCharacterCollator(vocabulary, **kwargs)([dataset[0], dataset[1]])
    assert torch.equal(first["input_ids"], second["input_ids"])
    assert torch.equal(first["labels"], second["labels"])
    eos = vocabulary["</s>"]
    for row, labels in zip(first["input_ids"], first["labels"]):
        eos_position = int((row == eos).nonzero()[0])
        assert labels[eos_position] == -100
        assert labels.ne(-100).any()


def test_encoder_forward_and_parameter_report_match_transfer_components():
    config = EncoderPretrainingConfig(
        d_model=16,
        attention_heads=4,
        encoder_layers=1,
        feed_forward_size=32,
        max_position_embeddings=32,
        mixed_precision="none",
    )
    vocabulary = build_pretraining_vocabulary(_frame()["text"])
    dataset = CharacterCorpusDataset(_frame(), vocabulary, 32)
    batch = MaskedCharacterCollator(
        vocabulary, mask_probability=0.3, min_span=1, max_span=2, seed=5
    )([dataset[0], dataset[1]])
    model = MaskedCharacterEncoder(len(vocabulary), config)
    output = model(batch["input_ids"], batch["attention_mask"], batch["labels"])
    assert output["logits"].shape[:2] == batch["input_ids"].shape
    assert torch.isfinite(output["loss"])
    report = parameter_report(model)
    assert report["pretraining_parameters"] == (
        report["transferable_parameters"] + len(vocabulary)
    )


def test_domain_weights_target_requested_probability_mass():
    frame = _frame()
    vocabulary = build_pretraining_vocabulary(frame["text"])
    dataset = CharacterCorpusDataset(frame, vocabulary, 32)
    weights = domain_sampling_weights(dataset, formal_fraction=0.25)
    formal_mass = weights[0] / weights.sum()
    assert torch.isclose(formal_mass, torch.tensor(0.25, dtype=torch.double))
