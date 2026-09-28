import torch

from src.modeling.compact_transformer import (
    CompactTransformerArchitecture,
    CompactTransformerForConditionalGeneration,
    prepare_compact_transformer,
)
from src.modeling.seq2seq_benchmark import levenshtein_target_annotations
from src.modeling.tnt_edit_transformer import (
    EXPAND,
    KEEP,
    TntEditConfig,
    TntEditTransformerForConditionalGeneration,
    tnt_edit_annotations,
)


def _frame(source: str, target: str):
    import pandas as pd

    return pd.DataFrame(
        {"source_id": ["row-1"], "source_text": [source], "target_text": [target]}
    )


def test_target_annotations_cover_insertions_and_deletion_boundaries():
    assert levenshtein_target_annotations([1, 2], [1, 3, 2]) == (
        [0, 1, 0],
        [0, -1, 1],
    )
    # The deletion itself has no target position, so the following target is
    # deliberately weighted as the boundary at which the source was skipped.
    assert levenshtein_target_annotations([1, 2, 3], [1, 3]) == (
        [0, 1],
        [0, 2],
    )


def test_relative_copy_model_uses_edit_mask_and_preserves_tied_weights(tmp_path):
    from transformers import AutoTokenizer

    architecture = CompactTransformerArchitecture(
        d_model=16,
        attention_heads=4,
        encoder_layers=1,
        decoder_layers=1,
        feed_forward_size=64,
        max_position_embeddings=32,
        generation_length_ratio=1.25,
        generation_length_margin=2,
        copy_aware=True,
        copy_position_mode="relative",
        edit_position_weight=3.0,
        label_smoothing=0.1,
        tie_character_embeddings=True,
    )
    model_dir = prepare_compact_transformer(
        [_frame("nanga", "naka")],
        output_root=tmp_path,
        seed=7,
        architecture=architecture,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = CompactTransformerForConditionalGeneration.from_pretrained(model_dir)
    source = tokenizer("nanga", return_tensors="pt")
    labels = tokenizer(text_target="naka", return_tensors="pt")["input_ids"]
    edit_mask, source_positions = levenshtein_target_annotations(
        source["input_ids"][0].tolist(), labels[0].tolist()
    )
    output = model(
        **source,
        labels=labels,
        edit_mask=torch.tensor([edit_mask], dtype=torch.bool),
        copy_source_positions=torch.tensor([source_positions]),
    )
    model.eval()
    model.config.use_incremental_generation = False
    legacy_generated = model.generate(**source, max_length=16)
    model.config.use_incremental_generation = True
    generated = model.generate(**source, max_length=16)

    assert output.loss.isfinite()
    assert generated.shape[0] == 1
    assert torch.equal(generated, legacy_generated)
    assert model.character_embedding.weight.data_ptr() == (
        model.output_projection.weight.data_ptr()
    )
    assert model.config.copy_position_mode == "relative"
    assert model.config.edit_position_weight == 3.0
    assert model.config.label_smoothing == 0.1


def test_tnt_annotations_keep_identity_and_encode_insertions():
    identity_operations, identity_segments = tnt_edit_annotations(
        [4, 2],
        [4, 2],
        eos_token_id=2,
        pad_token_id=0,
        output_slots_per_source=3,
    )
    assert identity_operations == [KEEP, KEEP]
    assert identity_segments == [[-100, -100, -100], [-100, -100, -100]]

    operations, segments = tnt_edit_annotations(
        [4, 2],
        [5, 4, 2],
        eos_token_id=2,
        pad_token_id=0,
        output_slots_per_source=3,
    )
    assert operations == [EXPAND, KEEP]
    assert segments == [[5, 4, 0], [-100, -100, -100]]


def test_tnt_model_computes_joint_loss_and_non_autoregressive_output():
    config = TntEditConfig(
        vocab_size=8,
        d_model=16,
        attention_heads=4,
        encoder_layers=1,
        feed_forward_size=32,
        max_position_embeddings=16,
        output_slots_per_source=3,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        decoder_start_token_id=1,
    )
    model = TntEditTransformerForConditionalGeneration(config)
    input_ids = torch.tensor([[4, 2]])
    attention_mask = torch.ones_like(input_ids)
    operation_labels, segment_labels = tnt_edit_annotations(
        input_ids[0].tolist(),
        [5, 4, 2],
        eos_token_id=2,
        pad_token_id=0,
        output_slots_per_source=3,
    )
    result = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=torch.tensor([[5, 4, 2]]),
        operation_labels=torch.tensor([operation_labels]),
        segment_labels=torch.tensor([segment_labels]),
    )
    model.eval()
    generated = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=12,
    )

    assert result.loss.isfinite()
    assert result.logits.shape == (1, 2, 3, 8)
    assert generated.shape[0] == 1
    assert generated.shape[1] <= 7  # at most two source positions x three slots + EOS
    assert generated[0, -1].item() == config.eos_token_id
