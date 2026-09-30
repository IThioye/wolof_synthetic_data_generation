import torch

from src.modeling.compact_transformer import (
    CompactTransformerArchitecture,
    CompactTransformerForConditionalGeneration,
    prepare_compact_transformer,
)
from src.modeling.seq2seq_benchmark import (
    Seq2SeqConfig,
    TntEditDataCollator,
    _tokenize_tnt_edit_dataset,
    levenshtein_target_annotations,
    train_seq2seq,
)
from src.modeling.tnt_edit_transformer import (
    EXPAND,
    KEEP,
    SUBSTITUTE,
    TntEditConfig,
    TntEditTransformerForConditionalGeneration,
    tnt_edit_annotations,
)
from src.modeling.tnt_hybrid_transformer import (
    BOUNDARY_DELETE,
    BOUNDARY_INSERT_BEFORE,
    BOUNDARY_KEEP,
    BOUNDARY_NONE,
    WORD_EDIT,
    WORD_KEEP,
    TntHybridConfig,
    TntHybridArchitecture,
    TntHybridTransformerForConditionalGeneration,
    hybrid_tnt_annotations,
    prepare_tnt_hybrid_transformer,
)
from src.modeling.tnt_hybrid_v3_transformer import (
    TntHybridV3Architecture,
    TntHybridV3Config,
    TntHybridV3TransformerForConditionalGeneration,
    hybrid_v3_annotations,
    prepare_tnt_hybrid_v3_transformer,
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


def test_hybrid_tnt_annotations_separate_spacing_from_lexical_edits():
    inserted = hybrid_tnt_annotations(
        [4, 5, 2],
        [4, 3, 5, 2],
        eos_token_id=2,
        pad_token_id=0,
        space_token_id=3,
        output_slots_per_source=3,
    )
    assert inserted[2] == [BOUNDARY_NONE, BOUNDARY_INSERT_BEFORE, -100]
    assert inserted[3] == [0, 0, -1]
    # A pure boundary edit should not make the lexical gate rewrite the word.
    assert inserted[4] == [WORD_KEEP]

    deleted = hybrid_tnt_annotations(
        [4, 3, 5, 2],
        [4, 5, 2],
        eos_token_id=2,
        pad_token_id=0,
        space_token_id=3,
        output_slots_per_source=3,
    )
    assert deleted[2] == [BOUNDARY_NONE, BOUNDARY_DELETE, BOUNDARY_NONE, -100]
    assert deleted[3] == [0, -1, 1, -1]
    assert deleted[4] == [WORD_KEEP, WORD_KEEP]


def test_hybrid_tnt_annotations_mark_the_affected_word():
    annotations = hybrid_tnt_annotations(
        [4, 5, 3, 6, 2],
        [4, 7, 3, 6, 2],
        eos_token_id=2,
        pad_token_id=0,
        space_token_id=3,
        output_slots_per_source=3,
    )
    assert annotations[2] == [
        BOUNDARY_NONE,
        BOUNDARY_NONE,
        BOUNDARY_KEEP,
        BOUNDARY_NONE,
        -100,
    ]
    assert annotations[3] == [0, 0, -1, 1, -1]
    assert annotations[4] == [WORD_EDIT, WORD_KEEP]
    assert annotations[1][1] == [7, 0, -100]


def test_hybrid_tnt_model_computes_all_losses_and_generates():
    config = TntHybridConfig(
        vocab_size=9,
        d_model=16,
        attention_heads=4,
        encoder_layers=1,
        feed_forward_size=32,
        max_position_embeddings=16,
        output_slots_per_source=3,
        space_token_id=3,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        decoder_start_token_id=1,
        boundary_loss_weight=0.5,
        word_loss_weight=0.5,
        boundary_class_weights=(0.5, 1.0, 2.0, 3.0),
        word_class_weights=(1.0, 1.0),
        operation_class_weights=(0.5, 2.0, 1.5, 3.0),
        use_boundary_inference=True,
        use_word_keep_gate=True,
    )
    model = TntHybridTransformerForConditionalGeneration(config)
    input_ids = torch.tensor([[4, 5, 3, 6, 2]])
    attention_mask = torch.ones_like(input_ids)
    annotations = hybrid_tnt_annotations(
        input_ids[0].tolist(),
        [4, 7, 3, 6, 2],
        eos_token_id=2,
        pad_token_id=0,
        space_token_id=3,
        output_slots_per_source=3,
    )
    result = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=torch.tensor([[4, 7, 3, 6, 2]]),
        operation_labels=torch.tensor([annotations[0]]),
        segment_labels=torch.tensor([annotations[1]]),
        boundary_labels=torch.tensor([annotations[2]]),
        word_ids=torch.tensor([annotations[3]]),
        word_labels=torch.tensor([annotations[4]]),
    )
    model.eval()
    generated = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=16,
    )

    assert result.loss.isfinite()
    assert result.logits.shape == (1, 5, 3, 9)
    assert generated.shape[0] == 1
    assert generated[0, -1].item() == config.eos_token_id
    assert model.config.use_boundary_inference is True
    assert model.config.use_word_keep_gate is True
    assert model.config.boundary_class_weights == [0.5, 1.0, 2.0, 3.0]
    assert model.config.operation_class_weights == [0.5, 2.0, 1.5, 3.0]


def test_hybrid_tnt_tokenizer_and_collator_supply_auxiliary_labels(tmp_path):
    import pandas as pd
    from transformers import AutoTokenizer, DataCollatorForSeq2Seq

    pairs = pd.DataFrame(
        {
            "source_id": ["one", "two"],
            "source_text": ["ab", "a b"],
            "target_text": ["a b", "ac"],
        }
    )
    architecture = TntHybridArchitecture(
        d_model=16,
        attention_heads=4,
        encoder_layers=1,
        feed_forward_size=32,
        max_position_embeddings=16,
        output_slots_per_source=3,
    )
    model_dir = prepare_tnt_hybrid_transformer(
        [pairs], output_root=tmp_path, seed=11, architecture=architecture
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = TntHybridTransformerForConditionalGeneration.from_pretrained(model_dir)
    dataset = _tokenize_tnt_edit_dataset(
        pairs,
        tokenizer,
        Seq2SeqConfig(
            model_name_or_path=str(model_dir),
            model_revision=None,
            task_prefix="",
            max_source_length=16,
            max_target_length=16,
            use_lora=False,
        ),
        model,
    )
    collator = TntEditDataCollator(
        DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
        output_slots=3,
        padding_side=tokenizer.padding_side,
    )
    batch = collator([dataset[0], dataset[1]])
    result = model(**batch)

    assert batch["boundary_labels"].shape == batch["input_ids"].shape
    assert batch["word_ids"].shape == batch["input_ids"].shape
    assert batch["word_labels"].shape[0] == 2
    assert result.loss.isfinite()


def test_hybrid_tnt_runs_through_shared_trainer(tmp_path):
    import pandas as pd

    training = pd.DataFrame(
        {
            "source_id": ["a", "b", "c", "d"],
            "source_text": ["ab", "a b", "ba", "b a"],
            "target_text": ["a b", "ab", "ba", "ba"],
        }
    )
    development = pd.DataFrame(
        {
            "source_id": ["e", "f"],
            "source_text": ["ab", "b a"],
            # ``c`` is deliberately absent from the training vocabulary. The
            # development CSV and metrics must still use the original Gold
            # reference rather than silently dropping its <unk> character.
            "target_text": ["a b", "c"],
        }
    )
    model_dir = prepare_tnt_hybrid_transformer(
        [training],
        output_root=tmp_path / "initialization",
        seed=3,
        architecture=TntHybridArchitecture(
            d_model=16,
            attention_heads=4,
            encoder_layers=1,
            feed_forward_size=32,
            max_position_embeddings=16,
            output_slots_per_source=3,
        ),
    )
    manifest = train_seq2seq(
        training,
        development,
        output_dir=tmp_path / "run",
        config=Seq2SeqConfig(
            model_name_or_path=str(model_dir),
            model_revision=None,
            task_prefix="",
            max_source_length=16,
            max_target_length=16,
            learning_rate=1e-3,
            optim="adamw_torch",
            use_lora=False,
            num_train_epochs=1,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            checkpoint_selection_metric="cer",
            seed=3,
            tf32=False,
            logging_steps=1,
        ),
    )

    assert (tmp_path / "run" / "best_model" / "config.json").is_file()
    assert (tmp_path / "run" / "dev_predictions.csv").is_file()
    assert manifest["development_metrics"]["rows"] == 2
    predictions = pd.read_csv(
        tmp_path / "run" / "dev_predictions.csv", keep_default_na=False
    )
    assert predictions.loc[predictions["source_id"].eq("f"), "reference"].item() == "c"


def test_hybrid_v3_separates_edit_length_from_content_and_keep_positions():
    annotations = hybrid_v3_annotations(
        [4, 5, 2],
        [4, 7, 2],
        eos_token_id=2,
        pad_token_id=0,
        space_token_id=3,
        output_slots_per_source=3,
    )

    assert annotations[0] == [KEEP, SUBSTITUTE, KEEP]
    assert annotations[1][0] == [-100, -100, -100]
    assert annotations[1][1] == [7, -100, -100]
    # Length is useful only where the operation head requests generated
    # content. KEEP and EOS anchors cannot dominate this objective.
    assert annotations[5] == [-100, 1, -100]

    expansion = hybrid_v3_annotations(
        [4, 2],
        [5, 4, 2],
        eos_token_id=2,
        pad_token_id=0,
        space_token_id=3,
        output_slots_per_source=3,
    )
    assert expansion[5] == [2, -100]
    assert expansion[1][0] == [5, 4, -100]


def test_hybrid_v3_model_computes_length_and_feature_aware_losses():
    config = TntHybridV3Config(
        vocab_size=10,
        d_model=16,
        attention_heads=4,
        encoder_layers=1,
        feed_forward_size=32,
        max_position_embeddings=16,
        output_slots_per_source=3,
        space_token_id=3,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        decoder_start_token_id=1,
        segment_length_class_weights=(1.0, 1.0, 1.0, 1.0),
        boundary_focal_gamma=2.0,
        word_feature_lookup={"4,5": [1, 0, 1]},
        use_hard_lexical_protection=True,
    )
    model = TntHybridV3TransformerForConditionalGeneration(config)
    input_ids = torch.tensor([[4, 5, 3, 6, 2]])
    annotations = hybrid_v3_annotations(
        input_ids[0].tolist(),
        [4, 7, 3, 6, 2],
        eos_token_id=2,
        pad_token_id=0,
        space_token_id=3,
        output_slots_per_source=3,
    )
    word_features = model.word_features_for_ids(
        input_ids[0].tolist(), annotations[3]
    )
    result = model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        labels=torch.tensor([[4, 7, 3, 6, 2]]),
        operation_labels=torch.tensor([annotations[0]]),
        segment_labels=torch.tensor([annotations[1]]),
        segment_length_labels=torch.tensor([annotations[5]]),
        boundary_labels=torch.tensor([annotations[2]]),
        word_ids=torch.tensor([annotations[3]]),
        word_labels=torch.tensor([annotations[4]]),
        word_features=torch.tensor([word_features]),
    )
    generated = model.generate(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        max_length=16,
    )

    assert result.loss.isfinite()
    assert result.logits.shape == (1, 5, 3, 10)
    assert word_features[0] == [1, 0, 1]
    assert generated.shape[0] == 1
    assert generated[0, -1].item() == config.eos_token_id


def test_hybrid_v3_tokenizer_and_collator_supply_new_supervision(tmp_path):
    import pandas as pd
    from transformers import AutoTokenizer, DataCollatorForSeq2Seq

    pairs = pd.DataFrame(
        {
            "source_id": ["one", "two"],
            "source_text": ["ab", "a b"],
            "target_text": ["ac", "ab"],
        }
    )
    model_dir = prepare_tnt_hybrid_v3_transformer(
        [pairs],
        output_root=tmp_path,
        seed=13,
        architecture=TntHybridV3Architecture(
            d_model=16,
            attention_heads=4,
            encoder_layers=1,
            feed_forward_size=32,
            max_position_embeddings=16,
            output_slots_per_source=3,
        ),
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = TntHybridV3TransformerForConditionalGeneration.from_pretrained(model_dir)
    dataset = _tokenize_tnt_edit_dataset(
        pairs,
        tokenizer,
        Seq2SeqConfig(
            model_name_or_path=str(model_dir),
            model_revision=None,
            task_prefix="",
            max_source_length=16,
            max_target_length=16,
            use_lora=False,
        ),
        model,
    )
    collator = TntEditDataCollator(
        DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
        output_slots=3,
        padding_side=tokenizer.padding_side,
    )
    batch = collator([dataset[0], dataset[1]])
    result = model(**batch)

    assert batch["segment_length_labels"].shape == batch["input_ids"].shape
    assert batch["word_features"].shape[:2] == batch["word_labels"].shape
    assert batch["word_features"].shape[2] == 3
    assert result.loss.isfinite()


def test_hybrid_v3_runs_through_shared_trainer(tmp_path):
    import pandas as pd

    training = pd.DataFrame(
        {
            "source_id": ["a", "b"],
            "source_text": ["ab", "a b"],
            "target_text": ["ac", "ab"],
        }
    )
    development = pd.DataFrame(
        {
            "source_id": ["c"],
            "source_text": ["ab"],
            "target_text": ["ac"],
        }
    )
    model_dir = prepare_tnt_hybrid_v3_transformer(
        [training],
        output_root=tmp_path / "initialization",
        seed=17,
        architecture=TntHybridV3Architecture(
            d_model=16,
            attention_heads=4,
            encoder_layers=1,
            feed_forward_size=32,
            max_position_embeddings=16,
            output_slots_per_source=3,
        ),
    )
    manifest = train_seq2seq(
        training,
        development,
        output_dir=tmp_path / "run",
        config=Seq2SeqConfig(
            model_name_or_path=str(model_dir),
            model_revision=None,
            task_prefix="",
            max_source_length=16,
            max_target_length=16,
            learning_rate=1e-3,
            optim="adamw_torch",
            use_lora=False,
            num_train_epochs=1,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=1,
            checkpoint_selection_metric="cer",
            seed=17,
            tf32=False,
            logging_steps=1,
        ),
    )

    assert manifest["development_metrics"]["rows"] == 1
    assert (tmp_path / "run" / "best_model" / "config.json").is_file()


def test_hybrid_v3_french_feature_does_not_freeze_a_corrected_surface(
    tmp_path, monkeypatch
):
    import pandas as pd
    import src.modeling.tnt_hybrid_v3_transformer as v3

    annotations_path = tmp_path / "annotations.csv"
    pd.DataFrame(
        [
            {
                "informal_token": "bonjor",
                "formal_token": "bonjour",
                "source_language": "fr",
                "target_language": "fr",
                "entity_type": "UNKNOWN",
                "protected": "yes",
                "review_status": "reviewed",
            }
        ]
    ).to_csv(annotations_path, index=False)
    monkeypatch.setattr(v3, "LINGUISTIC_ANNOTATIONS_PATH", annotations_path)
    monkeypatch.setattr(v3, "SENEGALESE_SURNAMES_PATH", tmp_path / "missing.txt")
    vocabulary = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3}
    for character in sorted(set("bonjoru")):
        vocabulary[character] = len(vocabulary)

    features, _, _ = v3._feature_lookup(vocabulary)

    def key(token):
        return ",".join(str(vocabulary[character]) for character in token)

    assert features[key("bonjor")] == [1, 0, 0]
    assert features[key("bonjour")] == [1, 0, 1]
