import json
import math
import threading

import numpy as np
import pandas as pd
import pytest

import src.annotation.annotation_core as annotation_core
from src.annotation.lookup_suggestions import SuggestionDecision
from src.annotation.annotation_core import (
    AnnotationService,
    align_sentence_to_source,
    classify_suggestion_action,
    interleave_comments_by_video,
    replace_surface_words,
    surface_word_tokens,
)
from src.annotation.linguistic_annotation_core import (
    LinguisticAnnotationService,
    build_occurrences,
    derive_transformations,
    infer_error_type,
    infer_error_types,
)
from src.alignment.ibm1_lexicon import train_lexicon
from config import CLEAN_COMMENTS_PATH, PROJECT_ROOT
from src.generation.benchmark_baselines import (
    METHOD_M0,
    METHOD_M1,
    build_beqi_rule_pairs,
    build_identity_pairs,
    informalize_beqi_rules,
)
from src.generation.oolel_baseline import add_overlap_flags, prepare_oolel
from src.generation.alignment_benchmarks import (
    generate_m2_pair,
    generate_m3_pair,
    stable_token_hash,
)
from src.generation.youtube_variant_mining import (
    MiningConfig,
    apply_mined_variants,
    build_mapping_index,
    canonical_signature,
    finalize_m5_manifest,
    likely_capitalized_tokens,
    mine_variant_mappings,
    word_tokens,
)
from src.generation.gold_linguistic_engine import (
    LinguisticConfig,
    LinguisticProfile,
    build_token_lookup,
    generate_m6_pair,
    learn_linguistic_profile,
    write_m6_outputs,
)
from src.modeling.benchmark_data import (
    assemble_training_data,
    audit_training_leakage,
    load_synthetic_pairs,
    require_no_held_out_leakage,
)
from src.modeling.beqi_transformer import (
    BeqiTransformerArchitecture,
    build_character_vocabulary,
    prepare_beqi_transformer,
)
from src.modeling.compact_transformer import (
    CompactTransformerArchitecture,
    CompactTransformerForConditionalGeneration,
    compact_transformer_model_class,
    prepare_compact_transformer,
)
from src.modeling.metrics import generation_metrics, normalization_metrics
from src.modeling.experiment_runner import (
    BenchmarkRunConfig,
    DEFAULT_METHODS,
    aggregate_results,
    collect_latest_development_results,
    frame_fingerprint,
    next_attempt_directory,
    run_condition,
    run_training_suite,
    write_markdown_results,
)
from src.modeling.seq2seq_benchmark import (
    Seq2SeqConfig,
    _is_lora_adapter,
    _sanitize_token_ids,
    _training_arguments,
)
from src.pipelines.classify_wolof import rows_in_diverse_video_order
from src.pipelines.get_filtered_dataset import build_filtered_dataset
from src.pipelines.prepare_gold_splits import assign_video_disjoint_splits
from src.pipelines.reconstruct_streamlit_sentences import reconstruct_sentence
from src.pipelines.prepare_wolof_lookup_seed import build_seed_lookup
from src.pipelines.prepare_multilingual_lookup_seed import build_multilingual_seed
from src.pipelines.refresh_linguistic_auto_annotations import (
    classify_refresh_rows,
    normalize_preserved_name_autos,
)
from src.pipelines.refresh_ambiguous_lookup_annotations import (
    ambiguous_seed_keys,
    stale_ambiguous_identity_mask,
)
from src.pipelines.audit_post_annotation_consistency import (
    build_linguistic_review,
    build_split_review,
)


def test_config_points_to_repository_data():
    assert (PROJECT_ROOT / "pfe_report_skeleton.md").exists()
    assert CLEAN_COMMENTS_PATH == PROJECT_ROOT / "data" / "youtube" / "clean_data.csv"


def test_external_download_registry_preserves_configured_paths():
    registry = json.loads(
        (PROJECT_ROOT / "data" / "external_sources.json").read_text(encoding="utf-8")
    )
    assert registry["formal_parallel"]["target_path"] == (
        "data/translation data/train-00000-of-00001.parquet"
    )
    assert registry["oolel"]["target_path"] == "data/external/oolel_raw.parquet"
    assert registry["wolof_translation_csv"]["target_path"] == (
        "data/translation data/wolof.csv"
    )
    assert registry["lexique4"]["target_path"] == "data/Lexique400/Lexique4.tsv"


def test_post_annotation_audit_finds_split_and_linguistic_drift(tmp_path):
    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    payload = [{"token": "lii", "final_correction": "li"}]
    columns = {
        "source_index": ["10"], "comment": ["lii"],
        "manual_formal_wolof": ["lii"], "video_url": ["video-a"],
        "token_corrections_json": [json.dumps(payload)],
    }
    for split in ("train", "dev", "test"):
        frame = pd.DataFrame(columns)
        if split != "train":
            frame["manual_formal_wolof"] = "li"
        frame.to_csv(split_dir / f"gold_{split}.csv", index=False)

    split_review = build_split_review(split_dir)
    assert split_review[["split", "source_index"]].to_dict("records") == [
        {"split": "train", "source_index": "10"}
    ]

    gold_train = pd.read_csv(
        split_dir / "gold_train.csv", keep_default_na=False, dtype=str
    )
    annotations = pd.DataFrame([
        {
            "occurrence_id": "10:0", "informal_token": "lii",
            "formal_token": "lii", "pos": "DET",
            "error_types_json": '["identity"]', "notes": "AUTO_IDENTITY_LOOKUP",
        },
        {
            "occurrence_id": "obsolete:0", "informal_token": "x",
            "formal_token": "x", "pos": "NOUN",
            "error_types_json": '["identity"]', "notes": "",
        },
    ])
    linguistic_review = build_linguistic_review(gold_train, annotations)
    assert set(linguistic_review["issue"]) == {
        "token_pair_changed_in_gold_train", "orphan_occurrence_id"
    }


def test_filtered_dataset_joins_original_non_prefix_indices():
    clean = pd.DataFrame(
        {
            "clean_comment": ["zero", "one", "two", "three", "four"],
            "video_url": ["a", "a", "b", "b", "c"],
        }
    )
    checkpoints = pd.DataFrame(
        {
            "index": [4, 1, 3],
            "is_informally_code_switched": ["True", "False", "True"],
        }
    )

    filtered = build_filtered_dataset(clean, checkpoints)

    assert filtered["source_index"].tolist() == [3, 4]
    assert filtered["clean_comment"].tolist() == ["three", "four"]
    assert filtered["video_url"].tolist() == ["b", "c"]


def test_annotation_and_classification_orders_are_video_diverse():
    comments = pd.DataFrame(
        {
            "source_index": [0, 1, 2, 3, 4],
            "clean_comment": ["a0", "a1", "b0", "b1", "c0"],
            "video_url": ["a", "a", "b", "b", "c"],
        }
    )
    interleaved = interleave_comments_by_video(comments)
    assert interleaved["source_index"].tolist() == [0, 2, 4, 1, 3]

    classification_order = rows_in_diverse_video_order(
        comments.set_index("source_index"), already_done=set()
    )
    assert [source_index for source_index, _ in classification_order] == [0, 2, 4, 1, 3]


def test_annotation_progress_tracks_eligible_gold_pairs_and_videos():
    service = AnnotationService.__new__(AnnotationService)
    service.lock = threading.RLock()
    service.comments = pd.DataFrame({"source_index": [0, 1, 2, 3]})
    service.annotated_ids = {0, 1, 2}
    service.latest_by_id = {
        0: {
            "status": "keep",
            "manual_formal_wolof": "formal zero",
            "video_url": "video-a",
        },
        1: {
            "status": "discard_uninteresting",
            "manual_formal_wolof": "",
            "video_url": "video-a",
        },
        # A keep without a formal target is not eligible for the gold split.
        2: {"status": "keep", "manual_formal_wolof": "", "video_url": "video-b"},
        4: {
            "status": "keep",
            "manual_formal_wolof": "formal four",
            "video_url": "video-b",
        },
    }

    progress = service.progress()

    assert progress["saved"] == 3
    assert progress["remaining"] == 1
    assert progress["kept"] == 2
    assert progress["gold_remaining"] == progress["minimum_kept"] - 2
    assert progress["kept_videos"] == 2
    assert progress["gold_videos_remaining"] == progress["minimum_videos"] - 2


def test_sentence_annotation_alignment_reconstructs_insertions_and_spacing_edits():
    source = ["dama", "bagn", "ko"]
    target = "man dama bañ ko léegi"

    aligned = align_sentence_to_source(source, target)

    assert " ".join(part for part in aligned if part) == target
    assert aligned[0].startswith("man ")
    assert aligned[-1].endswith(" léegi")


def test_sentence_annotation_surface_replacement_preserves_punctuation_and_spacing():
    source = "Waaw,  dama bagn ko!"
    tokens = surface_word_tokens(source)

    replaced = replace_surface_words(source, ["Waaw", "damaa", "bañ", "ko"])

    assert tokens == ["Waaw", "dama", "bagn", "ko"]
    assert replaced == "Waaw,  damaa bañ ko!"


def test_sentence_annotation_action_distinguishes_source_suggestion_and_edit():
    assert classify_suggestion_action("dama", "damaa", "dama") == "kept_source"
    assert classify_suggestion_action("dama", "damaa", "damaa") == "accepted_suggestion"
    assert classify_suggestion_action("dama", "damaa", "dama dem") == "edited"


def test_sentence_annotation_does_not_auto_translate_a_french_span():
    service = AnnotationService.__new__(AnnotationService)
    service.lock = threading.RLock()
    service.learned_corrections = {}
    service.base_token_suggestions = lambda _token: SuggestionDecision(
        category="french_exact",
        auto="bonjour",
        candidates=(),
        accepted=False,
        review_required=True,
        reason="Exact French lookup entry; left unchanged",
    )

    suggestion, rows = service.analyze("bonjour!")

    assert suggestion == "bonjour!"
    assert rows[0]["suggestions"] == ["bonjour"]


def test_sentence_annotation_save_keeps_sentence_and_token_evidence_synchronized(
    tmp_path, monkeypatch
):
    gold_path = tmp_path / "gold.csv"
    token_path = tmp_path / "tokens.csv"
    event_path = tmp_path / "events.csv"
    monkeypatch.setattr(annotation_core, "GOLD_OUTPUT_PATH", gold_path)
    monkeypatch.setattr(annotation_core, "TOKEN_OUTPUT_PATH", token_path)
    monkeypatch.setattr(
        annotation_core, "SENTENCE_ANNOTATION_EVENTS_PATH", event_path
    )

    service = AnnotationService.__new__(AnnotationService)
    service.lock = threading.RLock()
    service.comments = pd.DataFrame(
        {
            "source_index": [7],
            "clean_comment": ["dama bagn ko!"],
            "video_url": ["video-new"],
        }
    )
    service.annotated_ids = set()
    service.campaign_annotated_ids = set()
    service.latest_by_id = {}
    service.learned_corrections = {}
    service.analyze = lambda _text: (
        "damaa bañ ko!",
        [
            {"token": "dama", "category": "informal_wolof"},
            {"token": "bagn", "category": "informal_wolof"},
            {"token": "ko", "category": "wolof_formal"},
        ],
    )

    service.save(
        {
            "source_index": 7,
            "status": "keep",
            "manual_sentence": "man damaa bañ ko léegi!",
            "elapsed_ms": 1234,
            "token_rows": [
                {"token": "dama", "selected": "damaa"},
                {"token": "bagn", "selected": "bañ"},
                {"token": "ko", "selected": "ko"},
            ],
        }
    )

    saved = pd.read_csv(gold_path, keep_default_na=False).iloc[0]
    assert saved["manual_formal_wolof"] == "man damaa bañ ko léegi!"
    assert reconstruct_sentence(saved["token_corrections_json"]) == saved[
        "manual_formal_wolof"
    ]
    event = pd.read_csv(event_path).iloc[0]
    assert event["suggestion_action"] == "edited"
    assert event["elapsed_ms"] == 1234


def test_linguistic_occurrences_preserve_gold_token_order_and_error_hints():
    token_payload = [
        {
            "token": "buneh",
            "category": "informal_wolof",
            "selected_candidate": "bu ne",
            "final_correction": "bu ne",
        },
        {
            "token": "TIDIANE",
            "category": "french_native",
            "selected_candidate": "TIDIANE",
            "final_correction": "tidiane",
        },
    ]
    gold = pd.DataFrame(
        {
            "source_index": ["10"],
            "comment": ["buneh TIDIANE"],
            "manual_formal_wolof": ["bu ne tidiane"],
            "video_url": ["video-a"],
            "token_corrections_json": [json.dumps(token_payload)],
        }
    )

    occurrences = build_occurrences(gold)

    assert occurrences["occurrence_id"].tolist() == ["10:0", "10:1"]
    assert occurrences["formal_token"].tolist() == ["bu ne", "tidiane"]
    assert infer_error_type("buneh", "bu ne") == "spacing_merge"
    assert infer_error_types("buneh", "bu ne") == ("spacing_merge", "insertion")
    transformations = derive_transformations("bu ne", "buneh")
    assert {step["operation"] for step in transformations} == {"delete", "insert"}
    assert infer_error_type("TIDIANE", "tidiane") == "capitalization"
    assert infer_error_types("weurseuk", "wërsëg") == ("diacritic", "substitution")


def test_linguistic_service_uses_optional_lookup_and_saves_append_only(tmp_path):
    token_payload = [
        {
            "token": "xale",
            "category": "wolof_formal",
            "selected_candidate": "xale",
            "final_correction": "xale",
        }
    ]
    gold_path = tmp_path / "gold_train.csv"
    pd.DataFrame(
        {
            "source_index": ["20"],
            "comment": ["xale"],
            "manual_formal_wolof": ["xale"],
            "video_url": ["video-a"],
            "token_corrections_json": [json.dumps(token_payload)],
        }
    ).to_csv(gold_path, index=False)
    seed_path = tmp_path / "formal_token_lookup_seed.csv"
    pd.DataFrame(
        {
            "formal_token": ["xale"],
            "lemma": ["xale"],
            "pos": ["NOUN"],
            "target_language": ["wo"],
            "entity_type": ["NONE"],
            "protected": ["no"],
            "review_status": ["reviewed"],
        }
    ).to_csv(seed_path, index=False)
    annotations_path = tmp_path / "occurrences.csv"
    curated_path = tmp_path / "curated.csv"

    service = LinguisticAnnotationService(
        gold_train_path=gold_path,
        annotations_path=annotations_path,
        seed_lookup_path=seed_path,
        curated_lookup_path=curated_path,
        auto_save_safe_identities=False,
    )
    record = service.record_at(0)
    assert record["prefill"]["lookup_source"] == "seed_lookup"
    assert record["prefill"]["pos"] == "NOUN"
    assert record["prefill"]["source_language"] == "wo"
    assert record["prefill"]["target_language"] == "wo"
    assert record["prefill"]["entity_type"] == "UNKNOWN"
    assert record["prefill"]["protected"] == "no"
    assert record["prefill"]["review_status"] == "reviewed"
    assert record["lexical_fields_locked"]
    assert not record["pos_editable"]

    result = service.save(
        {
            "occurrence_id": "20:0",
            "informal_token": "xale",
            "formal_token": "xale",
            "lemma": "xale",
            "pos": "NOUN",
            "source_language": "wo",
            "target_language": "wo",
            "entity_type": "NONE",
            "protected": "no",
            "error_types": ["identity"],
            "review_status": "reviewed",
            "notes": "",
            "lexical_reusable": True,
            "lookup_source": "seed_lookup",
        }
    )

    assert result["progress"]["saved"] == 1
    assert service.next_unannotated_position(0) is None
    assert len(pd.read_csv(annotations_path)) == 1
    assert not curated_path.exists()
    saved = pd.read_csv(annotations_path).iloc[0]
    assert saved["entity_type"] == "UNKNOWN"
    assert saved["protected"] == "no"
    assert not bool(saved["lexical_reusable"])
    assert json.loads(saved["error_types_json"]) == ["identity"]
    assert json.loads(saved["transformation_steps_json"]) == []
    assert service.record_at(0)["prefill"]["lookup_source"] == "seed_lookup"


def test_linguistic_service_auto_saves_safe_lookup_identity(tmp_path):
    payload = [{
        "token": "xale", "category": "wolof_formal",
        "selected_candidate": "xale", "final_correction": "xale",
    }]
    gold_path = tmp_path / "gold.csv"
    pd.DataFrame({
        "source_index": ["30"], "comment": ["xale"],
        "manual_formal_wolof": ["xale"], "video_url": ["video-a"],
        "token_corrections_json": [json.dumps(payload)],
    }).to_csv(gold_path, index=False)
    seed_path = tmp_path / "seed.csv"
    pd.DataFrame({
        "formal_token": ["xale"], "lemma": ["xale"], "pos": ["NOUN"],
        "target_language": ["wo"], "entity_type": ["UNKNOWN"],
        "protected": ["no"], "review_status": ["reviewed"],
        "ambiguous_pos": ["false"], "pos_candidates_json": ['["NOUN"]'],
    }).to_csv(seed_path, index=False)
    annotations_path = tmp_path / "annotations.csv"

    service = LinguisticAnnotationService(
        gold_train_path=gold_path,
        annotations_path=annotations_path,
        seed_lookup_path=seed_path,
        curated_lookup_path=tmp_path / "curated.csv",
    )

    assert service.progress()["saved"] == 1
    assert service.progress()["auto_saved_identity_on_startup"] == 1
    saved = pd.read_csv(
        annotations_path, keep_default_na=False, dtype=str
    ).iloc[0]
    assert saved["error_type"] == "identity"
    assert saved["source_language"] == "wo"
    assert saved["notes"] == "AUTO_IDENTITY_LOOKUP"


def test_linguistic_service_auto_saves_spacing_merge_and_advances(tmp_path):
    spacing_payload = [{
        "token": "buneh", "category": "informal_wolof",
        "selected_candidate": "bu ne", "final_correction": "bu ne",
    }]
    unresolved_payload = [{
        "token": "xal", "category": "informal_wolof",
        "selected_candidate": "xale", "final_correction": "xale",
    }]
    gold_path = tmp_path / "gold.csv"
    pd.DataFrame({
        "source_index": ["34", "35"],
        "comment": ["buneh", "xal"],
        "manual_formal_wolof": ["bu ne", "xale"],
        "video_url": ["video-a", "video-b"],
        "token_corrections_json": [
            json.dumps(spacing_payload), json.dumps(unresolved_payload)
        ],
    }).to_csv(gold_path, index=False)
    annotations_path = tmp_path / "annotations.csv"

    service = LinguisticAnnotationService(
        gold_train_path=gold_path,
        annotations_path=annotations_path,
        seed_lookup_path=tmp_path / "missing_seed.csv",
        curated_lookup_path=tmp_path / "curated.csv",
    )

    progress = service.progress()
    assert progress["saved"] == 1
    assert progress["auto_saved_spacing_merge_on_startup"] == 1
    assert service.next_unannotated_position() == 1
    saved = pd.read_csv(
        annotations_path, keep_default_na=False, dtype=str
    ).iloc[0]
    assert saved["occurrence_id"] == "34:0"
    assert saved["source_index"] == "34"
    assert saved["token_position"] == "0"
    assert saved["pos"] == "UNKNOWN"
    assert saved["source_language"] == "wo"
    assert saved["target_language"] == "wo"
    assert saved["entity_type"] == "UNKNOWN"
    assert saved["protected"] == "no"
    assert saved["review_status"] == "reviewed"
    assert saved["notes"] == "AUTO_SPACING_MERGE"
    assert saved["lookup_source"] == "spacing_merge_policy"
    assert json.loads(saved["error_types_json"]) == [
        "spacing_merge", "insertion"
    ]
    assert saved["saved_utc"]


def test_reusable_manual_identity_immediately_propagates_to_other_occurrences(tmp_path):
    payload = [{"token": "bët", "final_correction": "bët"}]
    gold_path = tmp_path / "gold.csv"
    pd.DataFrame({
        "source_index": ["50", "51"],
        "comment": ["bët", "bët"],
        "manual_formal_wolof": ["bët", "bët"],
        "video_url": ["video-a", "video-b"],
        "token_corrections_json": [json.dumps(payload), json.dumps(payload)],
    }).to_csv(gold_path, index=False)
    annotations_path = tmp_path / "annotations.csv"
    service = LinguisticAnnotationService(
        gold_train_path=gold_path,
        annotations_path=annotations_path,
        seed_lookup_path=tmp_path / "missing_seed.csv",
        curated_lookup_path=tmp_path / "curated.csv",
    )

    result = service.save({
        "occurrence_id": "50:0", "informal_token": "bët", "formal_token": "bët",
        "lemma": "", "pos": "UNKNOWN", "source_language": "wo",
        "target_language": "wo", "entity_type": "UNKNOWN", "protected": "no",
        "error_types": ["identity"], "review_status": "reviewed", "notes": "",
        "lexical_reusable": True,
    })

    assert result["propagated_identity_count"] == 1
    assert result["progress"]["saved"] == 2
    assert service.next_unannotated_position() is None
    saved = pd.read_csv(annotations_path, keep_default_na=False, dtype=str)
    assert saved["occurrence_id"].tolist() == ["50:0", "51:0"]
    assert saved["source_index"].tolist() == ["50", "51"]
    assert saved["token_position"].tolist() == ["0", "0"]
    assert saved["saved_utc"].str.len().gt(0).all()
    propagated = saved.iloc[1]
    assert propagated["pos"] == "UNKNOWN"
    assert propagated["notes"] == "AUTO_IDENTITY_LOOKUP"
    assert propagated["lookup_source"] == "curated_lookup"


def test_two_matching_reviews_propagate_exact_changed_mapping_default(tmp_path):
    token_payload = [{"token": "bet", "final_correction": "bët"}]
    gold_path = tmp_path / "gold.csv"
    pd.DataFrame({
        "source_index": ["60", "61", "62", "63"],
        "comment": ["bet"] * 4,
        "manual_formal_wolof": ["bët"] * 4,
        "video_url": ["a", "b", "c", "d"],
        "token_corrections_json": [json.dumps(token_payload)] * 4,
    }).to_csv(gold_path, index=False)
    annotations_path = tmp_path / "annotations.csv"
    service = LinguisticAnnotationService(
        gold_train_path=gold_path,
        annotations_path=annotations_path,
        seed_lookup_path=tmp_path / "missing_seed.csv",
        curated_lookup_path=tmp_path / "curated.csv",
    )

    def review(occurrence_id):
        return service.save({
            "occurrence_id": occurrence_id,
            "informal_token": "bet", "formal_token": "bët", "lemma": "bët",
            "pos": "NOUN", "source_language": "wo", "target_language": "wo",
            "entity_type": "UNKNOWN", "protected": "no",
            "error_types": ["diacritic", "substitution"],
            "review_status": "reviewed", "notes": "", "lexical_reusable": False,
        })

    first = review("60:0")
    assert first["propagated_confirmed_mapping_count"] == 0
    second = review("61:0")

    assert second["propagated_confirmed_mapping_count"] == 2
    assert second["progress"]["saved"] == 4
    saved = pd.read_csv(annotations_path, keep_default_na=False, dtype=str)
    assert saved["occurrence_id"].tolist() == ["60:0", "61:0", "62:0", "63:0"]
    automatic = saved.iloc[2:]
    assert set(automatic["notes"]) == {"AUTO_CONFIRMED_MAPPING_DEFAULT"}
    assert set(automatic["lookup_source"]) == {"confirmed_mapping_default"}
    assert set(automatic["source_index"]) == {"62", "63"}
    assert automatic["saved_utc"].str.len().gt(0).all()
    assert all(
        json.loads(value) == ["diacritic", "substitution"]
        for value in automatic["error_types_json"]
    )


def test_linguistic_service_limits_ambiguous_pos_to_dictionary_candidates(tmp_path):
    payload = [{
        "token": "xol", "category": "wolof_formal",
        "selected_candidate": "xol", "final_correction": "xol",
    }]
    gold_path = tmp_path / "gold.csv"
    pd.DataFrame({
        "source_index": ["31"], "comment": ["xol"],
        "manual_formal_wolof": ["xol"], "video_url": ["video-a"],
        "token_corrections_json": [json.dumps(payload)],
    }).to_csv(gold_path, index=False)
    seed_path = tmp_path / "seed.csv"
    pd.DataFrame({
        "formal_token": ["xol"], "lemma": ["xol"], "pos": ["UNKNOWN"],
        "target_language": ["wo"], "entity_type": ["UNKNOWN"],
        "protected": ["no"], "review_status": ["reviewed"],
        "ambiguous_pos": ["true"],
        "pos_candidates_json": ['["NOUN", "VERB"]'],
    }).to_csv(seed_path, index=False)
    service = LinguisticAnnotationService(
        gold_train_path=gold_path,
        annotations_path=tmp_path / "annotations.csv",
        seed_lookup_path=seed_path,
        curated_lookup_path=tmp_path / "curated.csv",
    )
    record = service.record_at(0)
    assert record["options"]["pos"] == ("UNKNOWN", "NOUN", "VERB")
    assert record["pos_editable"]
    assert record["source_language_locked"]
    assert not record["reusable_allowed"]
    assert not record["prefill"]["lexical_reusable"]
    assert record["options"]["error"] == ("identity",)

    bad_payload = {
        "occurrence_id": "31:0", "informal_token": "xol", "formal_token": "xol",
        "lemma": "xol", "pos": "ADJ", "source_language": "wo",
        "target_language": "wo", "entity_type": "UNKNOWN", "protected": "no",
        "error_types": ["identity"], "review_status": "reviewed", "notes": "",
        "lexical_reusable": True,
    }
    with pytest.raises(ValueError, match="lookup candidates"):
        service.save(bad_payload)


def test_linguistic_service_enforces_person_proper_noun_and_protection(tmp_path):
    token_payload = [{
        "token": "Aicha", "category": "informal_wolof",
        "selected_candidate": "Aicha", "final_correction": "Aicha",
    }]
    gold_path = tmp_path / "gold.csv"
    pd.DataFrame({
        "source_index": ["32"], "comment": ["Aicha"],
        "manual_formal_wolof": ["Aicha"], "video_url": ["video-a"],
        "token_corrections_json": [json.dumps(token_payload)],
    }).to_csv(gold_path, index=False)
    annotations_path = tmp_path / "annotations.csv"
    service = LinguisticAnnotationService(
        gold_train_path=gold_path,
        annotations_path=annotations_path,
        seed_lookup_path=tmp_path / "missing_seed.csv",
        curated_lookup_path=tmp_path / "curated.csv",
    )

    service.save({
        "occurrence_id": "32:0", "informal_token": "Aicha", "formal_token": "Aicha",
        "lemma": "Aicha", "pos": "NOUN", "source_language": "wo",
        "target_language": "wo", "entity_type": "PERSON", "protected": "no",
        "error_types": ["identity"], "review_status": "reviewed", "notes": "",
        "lexical_reusable": True,
    })

    saved = pd.read_csv(annotations_path, keep_default_na=False).iloc[0]
    assert saved["pos"] == "PROPN"
    assert saved["entity_type"] == "PERSON"
    assert saved["protected"] == "yes"


def test_wolof_dictionary_adapter_preserves_pos_ambiguity():
    dictionary = pd.DataFrame(
        {
            "word": ["xale", "am", "am", "Marem"],
            "pos": ["n.", "n.", "v.t.", "np."],
            "translation": ["enfant", "avoir", "avoir", "nom propre"],
            "pos full name": ["nom", "nom", "verbe transitif", "nom propre"],
            "categorie": ["nom", "nom", "verbe", "nom"],
        }
    )

    seed = build_seed_lookup(dictionary).set_index("formal_token")

    assert seed.loc["xale", "pos"] == "NOUN"
    assert seed.loc["Marem", "pos"] == "PROPN"
    assert seed.loc["Marem", "entity_type"] == "UNKNOWN"
    assert seed.loc["Marem", "protected"] == "no"
    assert seed.loc["Marem", "review_status"] == "reviewed"
    assert seed.loc["am", "pos"] == "UNKNOWN"
    assert seed.loc["am", "ambiguous_pos"] == "true"
    assert json.loads(seed.loc["am", "pos_candidates_json"]) == ["NOUN", "VERB"]


def test_multilingual_seed_preserves_wolof_french_language_ambiguity():
    wolof = pd.DataFrame({
        "word": ["la"], "pos": ["pn."], "translation": ["toi"],
        "pos full name": ["pronom"], "categorie": ["pronom"],
    })
    lexique = pd.DataFrame({
        "1_Mot": ["la", "bonjour", "bonjour"],
        "4_Lemme": ["le", "bonjour", "bonjour"],
        "5_Cgram": ["ART:def", "NOM", "ONO"],
    })
    payload = [
        {"token": "la", "final_correction": "la"},
        {"token": "bonjour", "final_correction": "bonjour"},
    ]
    gold = pd.DataFrame({"token_corrections_json": [json.dumps(payload)]})

    combined, stats = build_multilingual_seed(wolof, lexique, gold)
    lookup = combined.set_index("formal_token")

    assert lookup.loc["la", "target_language"] == "unknown"
    assert lookup.loc["la", "source"] == "lookup_table_wolof.csv+Lexique4.tsv"
    assert lookup.loc["la", "language_ambiguous"] == "true"
    assert json.loads(lookup.loc["la", "language_candidates_json"]) == ["wo", "fr"]
    assert json.loads(lookup.loc["la", "pos_candidates_by_language_json"]) == {
        "wo": ["PRON"], "fr": ["DET"]
    }
    assert lookup.loc["bonjour", "target_language"] == "fr"
    assert lookup.loc["bonjour", "protected"] == "yes"
    assert lookup.loc["bonjour", "pos"] == "UNKNOWN"
    assert json.loads(lookup.loc["bonjour", "pos_candidates_json"]) == ["INTJ", "NOUN"]
    assert stats["french_gold_entries"] == 1
    assert stats["ambiguous_language_entries"] == 1


def test_language_ambiguous_lookup_requires_contextual_language_and_pos(tmp_path):
    payload = [{"token": "la", "final_correction": "la"}]
    gold_path = tmp_path / "gold.csv"
    pd.DataFrame({
        "source_index": ["70", "71"], "comment": ["la", "la"],
        "manual_formal_wolof": ["la", "la"], "video_url": ["a", "b"],
        "token_corrections_json": [json.dumps(payload), json.dumps(payload)],
    }).to_csv(gold_path, index=False)
    seed_path = tmp_path / "seed.csv"
    pd.DataFrame({
        "formal_token": ["la"], "lemma": ["la"], "pos": ["UNKNOWN"],
        "target_language": ["unknown"], "entity_type": ["UNKNOWN"],
        "protected": ["uncertain"], "review_status": ["uncertain"],
        "source": ["lookup_table_wolof.csv+Lexique4.tsv"],
        "language_candidates_json": ['["wo", "fr"]'],
        "language_ambiguous": ["true"], "ambiguous_pos": ["true"],
        "pos_candidates_json": ['["DET", "PRON"]'],
        "pos_candidates_by_language_json": ['{"wo": ["PRON"], "fr": ["DET"]}'],
    }).to_csv(seed_path, index=False)
    annotations_path = tmp_path / "annotations.csv"
    service = LinguisticAnnotationService(
        gold_train_path=gold_path,
        annotations_path=annotations_path,
        seed_lookup_path=seed_path,
        curated_lookup_path=tmp_path / "curated.csv",
    )

    assert service.progress()["saved"] == 0
    record = service.record_at(0)
    assert record["target_language_editable"]
    assert not record["source_language_locked"]
    assert not record["reusable_allowed"]
    assert record["options"]["language"] == ("unknown", "wo", "fr")
    assert record["options"]["pos"] == ("UNKNOWN", "PRON", "DET")

    base = {
        "occurrence_id": "70:0", "informal_token": "la", "formal_token": "la",
        "lemma": "la", "source_language": "fr", "target_language": "fr",
        "entity_type": "UNKNOWN", "protected": "yes", "error_types": ["identity"],
        "review_status": "reviewed", "notes": "", "lexical_reusable": True,
    }
    with pytest.raises(ValueError, match="not valid for selected language"):
        service.save({**base, "pos": "PRON"})
    result = service.save({**base, "pos": "DET"})
    assert result["progress"]["saved"] == 1
    assert not (tmp_path / "curated.csv").exists()
    saved = pd.read_csv(annotations_path, keep_default_na=False).iloc[0]
    assert saved["source_language"] == "fr"
    assert saved["target_language"] == "fr"
    assert not bool(saved["lexical_reusable"])


def test_ambiguity_refresh_removes_lookup_identity_autos_only():
    seed = pd.DataFrame({
        "formal_token": ["la", "dem", "xale"],
        "ambiguous_pos": ["false", "true", "false"],
        "language_ambiguous": ["true", "false", "false"],
    })
    annotations = pd.DataFrame({
        "formal_token": ["la", "dem", "la", "dem", "xale"],
        "notes": [
            "AUTO_IDENTITY_LOOKUP",
            "AUTO_IDENTITY_LOOKUP",
            "",
            "AUTO_CONFIRMED_MAPPING_DEFAULT",
            "AUTO_IDENTITY_LOOKUP",
        ],
    })

    keys = ambiguous_seed_keys(seed)
    remove = stale_ambiguous_identity_mask(annotations, keys)

    assert keys == {"la", "dem"}
    assert remove.tolist() == [True, True, False, False, False]


def test_multilingual_seed_gives_reviewed_name_inventory_precedence():
    wolof = pd.DataFrame({
        "word": ["Awa"], "pos": ["n."], "translation": ["nom"],
        "pos full name": ["nom"], "categorie": ["nom"],
    })
    lexique = pd.DataFrame({
        "1_Mot": ["Awa"], "4_Lemme": ["Awa"], "5_Cgram": ["NOM"],
    })
    payload = [{"token": "Awa", "final_correction": "Awa"}]
    gold = pd.DataFrame({"token_corrections_json": [json.dumps(payload)]})

    combined, stats = build_multilingual_seed(wolof, lexique, gold, ["Awa"])
    awa = combined.loc[combined["formal_token"].str.casefold() == "awa"].iloc[0]

    assert awa["pos"] == "PROPN"
    assert awa["target_language"] == "unknown"
    assert awa["entity_type"] == "PERSON"
    assert awa["protected"] == "yes"
    assert awa["source"] == "senegalese_surnames.txt"
    assert stats["name_entries"] == 1
    assert stats["wolof_entries"] == 0
    assert stats["french_gold_entries"] == 0


def test_name_lookup_auto_saves_person_with_unknown_language(tmp_path):
    payload = [{"token": "Awa", "final_correction": "Awa"}]
    gold_path = tmp_path / "gold.csv"
    pd.DataFrame({
        "source_index": ["41"], "comment": ["Awa"],
        "manual_formal_wolof": ["Awa"], "video_url": ["video-a"],
        "token_corrections_json": [json.dumps(payload)],
    }).to_csv(gold_path, index=False)
    seed_path = tmp_path / "seed.csv"
    pd.DataFrame({
        "formal_token": ["Awa"], "lemma": ["Awa"], "pos": ["PROPN"],
        "target_language": ["unknown"], "entity_type": ["PERSON"],
        "protected": ["yes"], "review_status": ["reviewed"],
        "source": ["senegalese_surnames.txt"], "ambiguous_pos": ["false"],
        "pos_candidates_json": ['["PROPN"]'],
    }).to_csv(seed_path, index=False)
    annotations_path = tmp_path / "annotations.csv"

    service = LinguisticAnnotationService(
        gold_train_path=gold_path,
        annotations_path=annotations_path,
        seed_lookup_path=seed_path,
        curated_lookup_path=tmp_path / "curated.csv",
    )

    saved = pd.read_csv(annotations_path, keep_default_na=False).iloc[0]
    assert saved["pos"] == "PROPN"
    assert saved["source_language"] == "unknown"
    assert saved["target_language"] == "unknown"
    assert saved["entity_type"] == "PERSON"
    assert saved["protected"] == "yes"
    assert saved["notes"] == "AUTO_IDENTITY_NAME_LOOKUP"
    record = service.record_at(0)
    assert record["prefill"]["notes"] == ""
    assert record["source_language_locked"]


def test_refresh_removes_only_untouched_stale_auto_rows():
    base = {
        "lemma": "bonjour", "pos": "NOUN", "source_language": "fr",
        "target_language": "fr", "entity_type": "UNKNOWN", "protected": "yes",
        "error_type": "identity", "error_types_json": '["identity"]',
        "transformation_steps_json": "[]", "review_status": "reviewed",
        "notes": "AUTO_IDENTITY_LOOKUP", "lexical_reusable": "false",
        "lookup_source": "french_seed_lookup",
    }
    annotations = pd.DataFrame([
        {**base, "occurrence_id": "1:0", "formal_token": "bonjour"},
        {**base, "occurrence_id": "2:0", "formal_token": "Awa", "lemma": "Awa",
         "pos": "PROPN", "source_language": "unknown", "target_language": "unknown"},
        {**base, "occurrence_id": "3:0", "formal_token": "bonjour", "pos": "PROPN"},
        {**base, "occurrence_id": "4:0", "formal_token": "bonjour", "notes": "human"},
        {**base, "occurrence_id": "5:0", "formal_token": "Oumy", "lemma": "Oumy",
         "pos": "PROPN", "source_language": "wo", "target_language": "wo",
         "protected": "no", "lookup_source": "curated_lookup"},
    ])
    seed = {
        "bonjour": {"lemma": "bonjour", "pos": "NOUN", "target_language": "fr"},
        "awa": {"lemma": "Awa", "pos": "NOUN", "target_language": "fr"},
    }

    remove, modified = classify_refresh_rows(annotations, seed, {}, {"awa"})
    migrated = normalize_preserved_name_autos(annotations, {"awa"}, remove)

    assert annotations.loc[remove, "occurrence_id"].tolist() == ["1:0", "5:0"]
    assert set(annotations.loc[modified, "occurrence_id"]) == {"2:0", "3:0"}
    assert annotations.loc[migrated, "occurrence_id"].tolist() == ["2:0"]
    awa = annotations.loc[annotations["occurrence_id"] == "2:0"].iloc[0]
    assert awa["entity_type"] == "PERSON"
    assert awa["target_language"] == "unknown"
    assert awa["lookup_source"] == "name_seed_lookup"


def test_french_seed_prefills_languages_protection_and_spelling_error(tmp_path):
    token_payload = [
        {"token": "bonjour", "category": "french_native", "final_correction": "bonjour"},
        {"token": "bonjor", "category": "french_native", "final_correction": "bonjour"},
    ]
    gold_path = tmp_path / "gold.csv"
    pd.DataFrame({
        "source_index": ["40"], "comment": ["bonjour bonjor"],
        "manual_formal_wolof": ["bonjour bonjour"], "video_url": ["video-a"],
        "token_corrections_json": [json.dumps(token_payload)],
    }).to_csv(gold_path, index=False)
    seed_path = tmp_path / "seed.csv"
    pd.DataFrame({
        "formal_token": ["bonjour"], "lemma": ["bonjour"], "pos": ["NOUN"],
        "target_language": ["fr"], "entity_type": ["UNKNOWN"],
        "protected": ["yes"], "review_status": ["reviewed"],
        "source": ["Lexique4.tsv"], "ambiguous_pos": ["false"],
        "pos_candidates_json": ['["NOUN"]'],
    }).to_csv(seed_path, index=False)
    service = LinguisticAnnotationService(
        gold_train_path=gold_path,
        annotations_path=tmp_path / "annotations.csv",
        seed_lookup_path=seed_path,
        curated_lookup_path=tmp_path / "curated.csv",
    )

    assert service.progress()["auto_saved_identity_on_startup"] == 1
    changed = service.record_at(1)
    assert changed["prefill"]["source_language"] == "fr"
    assert changed["prefill"]["target_language"] == "fr"
    assert changed["prefill"]["protected"] == "yes"
    assert changed["prefill"]["pos"] == "NOUN"
    assert "french_spelling" in changed["prefill"]["error_types"]
    assert changed["source_language_locked"]


def test_m0_identity_and_m1_rules_share_sources_but_not_transformations():
    sources = pd.DataFrame(
        {
            "source_id": ["parallel:0", "parallel:1"],
            "source_row": [0, 1],
            "formal_wolof": ["xam ñu", "BCEAO xam 2026"],
        }
    )

    identity = build_identity_pairs(sources)
    rules = build_beqi_rule_pairs(sources)

    assert identity["method"].eq(METHOD_M0).all()
    assert rules["method"].eq(METHOD_M1).all()
    assert identity["source_id"].tolist() == rules["source_id"].tolist()
    assert identity["informal_wolof"].tolist() == ["xam ñu", "BCEAO xam 2026"]
    assert rules["informal_wolof"].tolist() == ["kham gnou", "BCEAO kham 2026"]
    assert rules["changed"].all()


def test_m1_reports_linguistic_edits_without_counting_placeholder_restoration():
    transformed, counts = informalize_beqi_rules("ñu")

    assert transformed == "gnou"
    assert counts == {"enye_to_gn": 1, "u_to_ou": 1}


def test_m4_accepts_schema_alias_cleans_wrappers_and_rejects_bad_rows():
    raw = pd.DataFrame(
        {
            "en": ["know", "know duplicate", "same", "bad wrapper", "missing"],
            "wo": ["xam", "xam", "jamm", "ñu dem", None],
            "non_standardized": [
                "<NON_STANDARD> kham </NON_STANDARD>",
                "<NON_STANDARD> kham </NON_STANDARD>",
                "<NON_STANDARD> jamm </NON_STANDARD>",
                "<NON_STANDARD> gnou dem",
                None,
            ],
        }
    )

    accepted, audit = prepare_oolel(raw, revision="test-revision")

    assert accepted["informal_wolof"].tolist() == ["kham"]
    assert accepted["formal_wolof"].tolist() == ["xam"]
    assert len(audit) == 5
    assert audit["accepted"].tolist() == [True, False, False, False, False]
    assert "duplicate_pair" in audit.loc[1, "rejection_reasons"]
    assert "unchanged_normalized" in audit.loc[2, "rejection_reasons"]
    assert "malformed_wrapper" in audit.loc[3, "rejection_reasons"]
    assert "empty_formal" in audit.loc[4, "rejection_reasons"]


def test_m4_overlap_flags_are_diagnostic_and_normalized():
    raw = pd.DataFrame(
        {
            "wo": ["Xam  naa"],
            "non_standard": ["<NON_STANDARD> kham naa </NON_STANDARD>"],
        }
    )
    _accepted, audit = prepare_oolel(raw, revision="test-revision")

    flagged = add_overlap_flags(
        audit,
        common_formal=pd.Series(["xam naa"]),
        gold_formal=pd.Series(["XAM NAA"]),
        gold_informal=pd.Series(["kham naa"]),
    )

    assert bool(flagged.loc[0, "overlap_common_formal"])
    assert bool(flagged.loc[0, "overlap_current_gold_formal"])
    assert bool(flagged.loc[0, "overlap_current_gold_informal"])


def test_m5_mines_rule_compatible_variant_from_unpaired_context():
    config = MiningConfig(min_frequency=3, acceptance_score=0.68)
    comments = ["kham naa"] * 5 + ["dem naa"]
    formal_sentences = ["xam naa"] * 5 + ["dem naa"]

    candidates = mine_variant_mappings(
        comments,
        formal_sentences,
        formal_vocab={"xam", "naa", "dem"},
        french_vocab=set(),
        config=config,
    )

    mapping = candidates.loc[candidates["informal_token"].eq("kham")].iloc[0]
    assert mapping["formal_token"] == "xam"
    assert mapping["status"] == "accepted"
    assert mapping["rule_similarity"] == 1.0
    assert canonical_signature("kham") == canonical_signature("xam")


def test_m5_capitalization_guard_rejects_probable_entity():
    protected = likely_capitalized_tokens(
        ["Merci Marem", "Bonjour Marem", "Ak Marem"],
        min_count=3,
        minimum_ratio=0.6,
    )
    candidates = mine_variant_mappings(
        ["marem"] * 4,
        ["mareem"],
        formal_vocab={"mareem"},
        french_vocab=set(),
        protected_tokens=protected,
        config=MiningConfig(min_frequency=3),
    )

    assert "marem" in protected
    assert candidates.loc[0, "reason"] == "likely_entity_by_capitalization"


def test_m5_generation_is_deterministic_and_preserves_punctuation():
    accepted = pd.DataFrame(
        {
            "informal_token": ["kham", "xame"],
            "formal_token": ["xam", "xam"],
            "frequency": [10, 5],
            "score": [0.9, 0.8],
        }
    )
    mapping_index = build_mapping_index(accepted)

    first, first_edits = apply_mined_variants(
        "Dama xam, te xam!", "parallel:1", mapping_index, seed=2026, max_edits=2
    )
    second, second_edits = apply_mined_variants(
        "Dama xam, te xam!", "parallel:1", mapping_index, seed=2026, max_edits=2
    )

    assert first == second
    assert first_edits == second_edits
    assert first.endswith("!") and "," in first
    assert len(first_edits) == 2
    assert "xam" not in word_tokens(first)


def test_m2_exports_alignment_and_shared_rule_metadata_with_punctuation():
    row = generate_m2_pair(
        "xam ñu.",
        "savoir nous.",
        {"xam": {"savoir": 0.9}},
        source_id="parallel:0",
        source_row=0,
    )

    assert row["informal_wolof"] == "savoir gnou."
    assert row["alignment_edit_count"] == 1
    assert row["spelling_edit_count"] == 2
    assert '"alignment_probability": 0.9' in row["alignment_edits"]
    assert row["informal_wolof"].endswith(".")


def test_m3_records_cmdr_scores_and_avoids_unaligned_candidates():
    class FakeVectors:
        allowed = {"xam", "savoir", "ñu", "nous"}

        def __contains__(self, item):
            return item in self.allowed

        def similarity(self, left, right):
            return 0.9

    class FakeModel:
        wv = FakeVectors()

    row = generate_m3_pair(
        "xam ñu!",
        "savoir nous!",
        FakeModel(),
        {"xam": {"savoir": 0.5}},
        source_id="parallel:0",
        source_row=0,
        max_n=1,
        max_substitutions=2,
    )

    assert row["informal_wolof"] == "savoir gnou!"
    assert row["alignment_edit_count"] == 1
    assert '"embedding_similarity": 0.9' in row["alignment_edits"]
    assert '"alignment_score": 0.5' in row["alignment_edits"]


def test_cmdr_token_hash_is_process_independent():
    assert stable_token_hash("Wolof_ñu") == 2265038418


def test_gold_splits_do_not_share_videos():
    kept = pd.DataFrame(
        {
            "source_index": range(12),
            "video_url": [f"video-{index // 2}" for index in range(12)],
            "comment": [f"informal-{index}" for index in range(12)],
            "manual_formal_wolof": [f"formal-{index}" for index in range(12)],
            "status": ["keep"] * 12,
        }
    )

    splits = assign_video_disjoint_splits(kept, seed=2026)

    assert sum(len(split) for split in splits.values()) == len(kept)
    video_sets = [set(split["video_url"]) for split in splits.values()]
    assert all(video_sets)
    assert video_sets[0].isdisjoint(video_sets[1])
    assert video_sets[0].isdisjoint(video_sets[2])
    assert video_sets[1].isdisjoint(video_sets[2])


def test_ibm1_builds_eflomal_compatible_lexicon():
    parallel = pd.DataFrame(
        {
            "wolof": ["kër", "kër", "kër", "téere", "téere", "téere"],
            "french": ["maison", "maison", "maison", "livre", "livre", "livre"],
        }
    )

    lexicon, _model, bitext = train_lexicon(
        parallel, iterations=3, min_count=1, min_prob=0.1
    )

    assert len(bitext) == 6
    assert lexicon["kër"]["maison"] == 1.0
    assert lexicon["téere"]["livre"] == 1.0


def test_m7_perfect_predictions_have_perfect_metrics():
    metrics = normalization_metrics(
        ["man dem"],
        ["man dem na"],
        ["man dem na"],
    )

    assert metrics["cer"] == 0.0
    assert metrics["wer"] == 0.0
    assert metrics["chrf"] == 100.0
    assert metrics["exact_match"] == 1.0
    assert metrics["correction_f1"] == 1.0
    assert metrics["overcorrection_rate"] == 0.0


def test_m7_metrics_detect_overcorrection():
    metrics = normalization_metrics(["jàmm"], ["jàmm rekk"], ["jàmm"])

    assert metrics["correction_fp"] == 1
    assert metrics["overcorrection_rate"] == 1.0
    assert metrics["unchanged_sentence_overcorrection_rate"] == 1.0


def test_m7_generation_metrics_reject_length_mismatch():
    try:
        generation_metrics(["a"], ["a", "b"])
    except ValueError as error:
        assert "equal length" in str(error)
    else:
        raise AssertionError("Length mismatch should fail")


def test_m7_synthetic_plus_gold_retains_authentic_duplicate_first():
    synthetic = pd.DataFrame(
        {
            "source_id": ["synthetic:1", "synthetic:2"],
            "method": ["m1", "m1"],
            "source_text": ["noisy", "another"],
            "target_text": ["formal", "target"],
            "source_type": ["synthetic", "synthetic"],
        }
    )
    gold = pd.DataFrame(
        {
            "source_id": ["gold:train:1"],
            "method": ["gold_train"],
            "source_text": ["noisy"],
            "target_text": ["formal"],
            "source_type": ["gold"],
        }
    )

    combined = assemble_training_data(synthetic, gold, regime="synthetic_plus_gold")

    assert len(combined) == 2
    assert combined.loc[combined.source_text.eq("noisy"), "source_type"].item() == "gold"


def test_m7_leakage_audit_blocks_held_out_input_overlap():
    training = pd.DataFrame({"source_text": ["same"], "target_text": ["train target"]})
    held_out = pd.DataFrame({"source_text": [" same "], "target_text": ["test target"]})

    audit = audit_training_leakage(training, held_out)

    assert audit.input_overlap == 1
    assert audit.target_overlap == 0
    assert audit.unsafe
    try:
        require_no_held_out_leakage(training, held_out)
    except ValueError as error:
        assert "Unsafe held-out overlap" in str(error)
    else:
        raise AssertionError("Held-out input leakage should fail")


def test_m7_loads_common_synthetic_schema(tmp_path):
    path = tmp_path / "m9.csv"
    pd.DataFrame(
        {
            "source_id": ["parallel:0"],
            "method": ["m9"],
            "informal_wolof": ["  noisy   text "],
            "formal_wolof": ["formal text"],
        }
    ).to_csv(path, index=False)

    loaded = load_synthetic_pairs("m9", path=path)

    assert loaded.loc[0, "source_text"] == "noisy text"
    assert loaded.loc[0, "target_text"] == "formal text"
    assert loaded.loc[0, "source_type"] == "synthetic"


def test_m7_rejects_provisional_synthetic_manifest(tmp_path):
    path = tmp_path / "m9.csv"
    manifest_path = tmp_path / "m9_manifest.json"
    pd.DataFrame(
        {
            "source_id": ["parallel:0"],
            "method": ["m9"],
            "informal_wolof": ["noisy"],
            "formal_wolof": ["formal"],
        }
    ).to_csv(path, index=False)
    manifest_path.write_text(
        '{"provisional": true, "leakage_status": "held-out videos not excluded"}',
        encoding="utf-8",
    )

    try:
        load_synthetic_pairs("m9", path=path, manifest_path=manifest_path)
    except ValueError as error:
        assert "provisional" in str(error)
        assert "held-out videos not excluded" in str(error)
    else:
        raise AssertionError("Provisional data should not enter final M7 training")


def _m6_annotation_rows(rows):
    records = []
    for index, row in enumerate(rows):
        formal = row["formal_token"]
        informal = row["informal_token"]
        default_error = "identity" if formal.casefold() == informal.casefold() else "substitution"
        error = row.get("error_type", default_error)
        records.append(
            {
                "occurrence_id": f"{row['source_index']}:{index}",
                "source_index": str(row["source_index"]),
                "informal_token": informal,
                "formal_token": formal,
                "pos": row.get("pos", "NOUN"),
                "source_language": row.get("source_language", "wo"),
                "target_language": row.get("target_language", "wo"),
                "protected": row.get("protected", "no"),
                "error_type": error,
                "error_types_json": json.dumps([error]),
                "transformation_steps_json": json.dumps(row.get("steps", [])),
                "review_status": row.get("review_status", "reviewed"),
                "lexical_reusable": row.get("lexical_reusable", True),
            }
        )
    return pd.DataFrame(records)


def _m6_lookup(**entries):
    return {
        token: {
            "pos": {properties.get("pos", "NOUN")},
            "target_language": properties.get("target_language", "wo"),
            "entity_type": properties.get("entity_type", "UNKNOWN"),
            "protected": properties.get("protected", "no"),
        }
        for token, properties in entries.items()
    }


def test_m6_rejects_annotations_outside_locked_gold_train():
    annotations = _m6_annotation_rows(
        [
            {"source_index": 1, "formal_token": "xol", "informal_token": "khol"},
            {"source_index": 2, "formal_token": "wax", "informal_token": "wakh"},
        ]
    )

    with pytest.raises(ValueError, match="outside locked gold train"):
        learn_linguistic_profile(annotations, gold_train_source_ids=[1])


def test_m6_learns_only_reviewed_unprotected_wolof_evidence():
    annotations = _m6_annotation_rows(
        [
            {"source_index": 1, "formal_token": "xol", "informal_token": "khol"},
            {
                "source_index": 1,
                "formal_token": "bonne",
                "informal_token": "bon",
                "source_language": "fr",
                "target_language": "fr",
                "protected": "yes",
            },
            {
                "source_index": 2,
                "formal_token": "nit",
                "informal_token": "nitt",
                "review_status": "uncertain",
            },
        ]
    )

    profile = learn_linguistic_profile(annotations, gold_train_source_ids=[1, 2])

    assert set(profile.mappings.informal_token) == {"khol"}
    assert profile.mappings.mapping_source.eq("gold_train_linguistic_mapping").all()
    assert profile.training_summary["uses_m5_outputs"] is False
    assert profile.training_summary["uses_m1_rules"] is False


def test_m6_applies_a_gold_linguistic_mapping_without_m5_or_m1():
    annotations = _m6_annotation_rows(
        [
            {
                "source_index": 1,
                "formal_token": "xol",
                "informal_token": "khol",
                "error_type": "substitution",
            }
        ]
    )
    profile = learn_linguistic_profile(
        annotations,
        gold_train_source_ids=[1],
        config=LinguisticConfig(min_transformation_count=99),
    )

    row = generate_m6_pair(
        "Xol bi!",
        source_id="parallel:0",
        source_row=0,
        profile=profile,
        token_lookup=_m6_lookup(xol={}, bi={"pos": "DET"}),
        config=LinguisticConfig(min_transformation_count=99),
    )

    assert row["informal_wolof"] == "khol bi!"
    assert row["lexical_mapping_edit_count"] == 1
    assert row["transformation_edit_count"] == 0
    assert "youtube" not in row["applied_transformations"]
    assert "beqi" not in row["applied_transformations"]


def test_m6_generalizes_a_repeated_transformation_only_to_matching_pos():
    step_x_to_kh = {
        "operation": "replace",
        "formal_text": "x",
        "informal_text": "kh",
        "formal_start": 0,
        "formal_end": 1,
    }
    annotations = _m6_annotation_rows(
        [
            {
                "source_index": 1,
                "formal_token": "xol",
                "informal_token": "khol",
                "pos": "NOUN",
                "error_type": "substitution",
                "steps": [step_x_to_kh],
                "lexical_reusable": False,
            },
            {
                "source_index": 2,
                "formal_token": "xon",
                "informal_token": "khon",
                "pos": "NOUN",
                "error_type": "substitution",
                "steps": [step_x_to_kh],
                "lexical_reusable": False,
            },
        ]
    )
    profile = learn_linguistic_profile(annotations, gold_train_source_ids=[1, 2])

    noun = generate_m6_pair(
        "xob",
        source_id="parallel:noun",
        source_row=0,
        profile=profile,
        token_lookup=_m6_lookup(xob={"pos": "NOUN"}),
    )
    verb = generate_m6_pair(
        "xob",
        source_id="parallel:verb",
        source_row=1,
        profile=profile,
        token_lookup=_m6_lookup(xob={"pos": "VERB"}),
    )

    assert noun["informal_wolof"] == "khob"
    assert noun["transformation_edit_count"] == 1
    assert verb["informal_wolof"] == "xob"


def test_m6_zero_empirical_budget_does_not_force_an_available_mapping():
    annotations = _m6_annotation_rows(
        [{"source_index": 1, "formal_token": "xol", "informal_token": "khol"}]
    )
    learned = learn_linguistic_profile(annotations, gold_train_source_ids=[1])
    profile = LinguisticProfile(
        mappings=learned.mappings,
        transformations=learned.transformations,
        edit_budgets=(0,),
        pos_error_profile=learned.pos_error_profile,
        training_summary=learned.training_summary,
    )

    row = generate_m6_pair(
        "xol",
        source_id="parallel:0",
        source_row=0,
        profile=profile,
        token_lookup=_m6_lookup(xol={}),
    )

    assert row["informal_wolof"] == "xol"
    assert row["edit_count"] == 0


def test_m6_lookup_prefers_curated_protection_over_seed():
    curated = pd.DataFrame(
        {
            "formal_token": ["Awa"],
            "pos": ["PROPN"],
            "target_language": ["unknown"],
            "entity_type": ["PERSON"],
            "protected": ["yes"],
            "review_status": ["reviewed"],
        }
    )
    seed = pd.DataFrame(
        {
            "formal_token": ["Awa"],
            "pos": ["NOUN"],
            "target_language": ["wo"],
            "entity_type": ["UNKNOWN"],
            "protected": ["no"],
        }
    )

    lookup = build_token_lookup(curated, seed)

    assert lookup["awa"]["protected"] == "yes"
    assert lookup["awa"]["entity_type"] == "PERSON"


def test_m6_generation_is_deterministic_for_the_same_source_id():
    annotations = _m6_annotation_rows(
        [{"source_index": 1, "formal_token": "xol", "informal_token": "khol"}]
    )
    profile = learn_linguistic_profile(annotations, gold_train_source_ids=[1])
    arguments = {
        "formal": "xol xol",
        "source_id": "parallel:fixed",
        "source_row": 0,
        "profile": profile,
        "token_lookup": _m6_lookup(xol={}),
    }

    first = generate_m6_pair(**arguments)
    second = generate_m6_pair(**arguments)

    assert first == second


def test_m6_manifest_explicitly_excludes_m5_and_m1(tmp_path):
    annotations = _m6_annotation_rows(
        [{"source_index": 1, "formal_token": "xol", "informal_token": "khol"}]
    )
    profile = learn_linguistic_profile(annotations, gold_train_source_ids=[1])
    row = generate_m6_pair(
        "xol",
        source_id="parallel:0",
        source_row=0,
        profile=profile,
        token_lookup=_m6_lookup(xol={}),
    )
    input_path = tmp_path / "linguistic_annotations.csv"
    annotations.to_csv(input_path, index=False)

    manifest = write_m6_outputs(
        pd.DataFrame([row]),
        profile,
        output_path=tmp_path / "pairs.csv",
        mappings_path=tmp_path / "mappings.csv",
        transformations_path=tmp_path / "transformations.csv",
        profile_path=tmp_path / "profile.json",
        review_path=tmp_path / "review.csv",
        manifest_path=tmp_path / "manifest.json",
        input_paths={"linguistic_annotations": input_path},
        config=LinguisticConfig(),
        review_size=1,
    )

    assert manifest["uses_m5_outputs"] is False
    assert manifest["uses_m1_rules"] is False
    assert all("m5" not in name.casefold() for name in manifest["inputs"])


def test_m6_none_edit_cap_preserves_the_empirical_sentence_count():
    annotations = _m6_annotation_rows(
        [
            {
                "source_index": 1,
                "formal_token": f"formal{index}",
                "informal_token": f"informal{index}",
            }
            for index in range(5)
        ]
    )

    uncapped = learn_linguistic_profile(
        annotations,
        gold_train_source_ids=[1],
        config=LinguisticConfig(max_edits_per_sentence=None),
    )
    capped = learn_linguistic_profile(
        annotations,
        gold_train_source_ids=[1],
        config=LinguisticConfig(max_edits_per_sentence=3),
    )

    assert uncapped.edit_budgets == (5,)
    assert capped.edit_budgets == (3,)


def test_m6_spacing_merge_is_a_phrase_mapping_despite_unknown_pos():
    annotations = _m6_annotation_rows(
        [
            {
                "source_index": 1,
                "formal_token": "bu ne",
                "informal_token": "buneh",
                "pos": "UNKNOWN",
                "error_type": "spacing_merge",
                "lexical_reusable": False,
            }
        ]
    )
    profile = learn_linguistic_profile(annotations, gold_train_source_ids=[1])

    row = generate_m6_pair(
        "bu ne fi",
        source_id="parallel:spacing",
        source_row=0,
        profile=profile,
        token_lookup=_m6_lookup(
            bu={"pos": "DET"}, ne={"pos": "VERB"}, fi={"pos": "ADV"}
        ),
    )

    assert row["informal_wolof"] == "buneh fi"
    assert row["lexical_mapping_edit_count"] == 1
    assert "spacing_merge" in json.loads(row["error_types"])


def _m7_frame(source_text="informal", target_text="formal"):
    return pd.DataFrame(
        {
            "source_id": ["row:1"],
            "source_text": [source_text],
            "target_text": [target_text],
        }
    )


def test_m7_frame_fingerprint_is_stable_and_content_sensitive():
    original = _m7_frame()
    identical = original.copy()
    modified = _m7_frame(source_text="modified")

    assert frame_fingerprint(original) == frame_fingerprint(identical)
    assert frame_fingerprint(original) != frame_fingerprint(modified)


def test_m7_defaults_to_fixed_lora_adapter_configuration(tmp_path):
    config = Seq2SeqConfig()
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")

    assert config.use_lora is True
    assert config.lora_rank == 8
    assert config.lora_target_modules == ("q", "v")
    assert config.max_target_length == 96
    assert config.per_device_train_batch_size == 24
    assert config.per_device_eval_batch_size == 8
    assert config.gradient_accumulation_steps == 1
    assert config.generation_num_beams == 1
    assert config.checkpoint_selection_metric == "loss"
    assert config.tf32 is True
    assert config.full_determinism is False
    assert config.gradient_checkpointing is False
    assert _is_lora_adapter(adapter)


def test_m7_excludes_identity_artifact_from_trained_methods():
    assert DEFAULT_METHODS == ("m1", "m2", "m3", "m4", "m5", "m6")


def test_m7_loss_checkpoint_policy_avoids_generation_during_epoch_eval(tmp_path):
    arguments = _training_arguments(Seq2SeqConfig(), tmp_path)

    assert arguments.predict_with_generate is False
    assert arguments.metric_for_best_model == "eval_loss"
    assert arguments.greater_is_better is False


def test_m7_cer_checkpoint_policy_remains_available(tmp_path):
    config = Seq2SeqConfig(checkpoint_selection_metric="cer")
    arguments = _training_arguments(config, tmp_path)

    assert arguments.predict_with_generate is True
    assert arguments.metric_for_best_model == "cer"


def test_m7_wer_checkpoint_policy_uses_generated_predictions(tmp_path):
    config = Seq2SeqConfig(checkpoint_selection_metric="wer")
    arguments = _training_arguments(config, tmp_path)

    assert arguments.predict_with_generate is True
    assert arguments.metric_for_best_model == "wer"
    assert arguments.greater_is_better is False


def test_m7_generated_negative_padding_is_safe_for_tokenizer_decode():
    assert _sanitize_token_ids(
        np.array([[7, 1, -100], [9, -1, 1]]), pad_token_id=0
    ) == [[7, 1, 0], [9, 0, 1]]


def test_beqi_character_vocabulary_is_deterministic_and_preserves_spaces():
    first = build_character_vocabulary(["na nga", "jàmm"])
    second = build_character_vocabulary(["jàmm", "na nga"])

    assert first == second
    assert first["<pad>"] == 0
    assert " " in first
    assert "à" in first


def test_beqi_transformer_initialization_round_trips_locally(tmp_path):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    architecture = BeqiTransformerArchitecture(
        d_model=32,
        encoder_layers=1,
        decoder_layers=1,
        attention_heads=4,
        feed_forward_size=64,
        max_position_embeddings=48,
    )
    model_dir = prepare_beqi_transformer(
        [_m7_frame("nanga def", "naka nga def")],
        output_root=tmp_path,
        seed=7,
        architecture=architecture,
    )
    reused = prepare_beqi_transformer(
        [_m7_frame("nanga def", "naka nga def")],
        output_root=tmp_path,
        seed=7,
        architecture=architecture,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    encoded = tokenizer("nanga def")

    assert reused == model_dir
    assert tokenizer.decode(encoded["input_ids"], skip_special_tokens=True) == "nanga def"
    assert model.config.encoder_layers == 1
    assert model.config.decoder_layers == 1


def test_compact_transformer_round_trips_and_generates_locally(tmp_path):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    architecture = CompactTransformerArchitecture(
        d_model=16,
        attention_heads=4,
        encoder_layers=1,
        decoder_layers=1,
        feed_forward_size=32,
        max_position_embeddings=32,
        generation_length_ratio=0.5,
        generation_length_margin=1,
    )
    model_dir = prepare_compact_transformer(
        [_m7_frame("nanga def", "naka nga def")],
        output_root=tmp_path,
        seed=7,
        architecture=architecture,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model_class = compact_transformer_model_class(
        model_dir, AutoModelForSeq2SeqLM
    )
    model = model_class.from_pretrained(model_dir)
    encoded = tokenizer("nanga def", return_tensors="pt")
    labels = tokenizer(text_target="naka nga def", return_tensors="pt")["input_ids"]
    output = model(**encoded, labels=labels)
    generated = model.generate(**encoded, max_length=16, num_beams=1)

    assert isinstance(model, CompactTransformerForConditionalGeneration)
    assert tokenizer.decode(encoded["input_ids"][0], skip_special_tokens=True) == "nanga def"
    assert output.logits.shape[:2] == labels.shape
    assert output.loss.isfinite()
    assert generated.shape[0] == 1
    expected_cap = int(math.ceil(encoded["attention_mask"].sum() * 0.5)) + 2
    assert generated.shape[1] <= expected_cap
    assert model.config.encoder_layers == 1
    assert model.config.generation_length_ratio == 0.5

    copy_architecture = CompactTransformerArchitecture(
        d_model=16,
        attention_heads=4,
        encoder_layers=1,
        decoder_layers=1,
        feed_forward_size=32,
        max_position_embeddings=32,
        generation_length_ratio=0.5,
        generation_length_margin=1,
        copy_aware=True,
        copy_regularization_strength=0.05,
        scheduled_sampling_probability=0.25,
    )
    copy_dir = prepare_compact_transformer(
        [_m7_frame("nanga def", "naka nga def")],
        output_root=tmp_path,
        seed=7,
        architecture=copy_architecture,
        warm_start_checkpoint=model_dir,
    )
    copy_model = CompactTransformerForConditionalGeneration.from_pretrained(copy_dir)
    copy_output = copy_model(**encoded, labels=labels)
    copy_generated = copy_model.generate(**encoded, max_length=16, num_beams=1)

    assert copy_model.config.copy_aware is True
    assert copy_model.config.copy_regularization_strength == 0.05
    assert copy_model.config.scheduled_sampling_probability == 0.25
    assert copy_output.loss.isfinite()
    assert copy_generated.shape[1] <= expected_cap
    assert torch.equal(
        model.character_embedding.weight,
        copy_model.character_embedding.weight,
    )


def test_m7_attempt_numbers_are_versioned_without_overwrite(tmp_path):
    condition = tmp_path / "m1" / "seed_2026" / "synthetic_then_gold"
    assert next_attempt_directory(condition).name == "attempt_001"
    (condition / "attempt_001").mkdir(parents=True)
    assert next_attempt_directory(condition).name == "attempt_002"


def test_m7_run_condition_skips_same_input_and_versions_modified_input(
    tmp_path, monkeypatch
):
    def fake_train(training, development, *, output_dir, config, resume_from_checkpoint=None):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "best_model").mkdir(exist_ok=True)
        return {
            "configuration": config.__dict__,
            "training_rows": len(training),
            "development_rows": len(development),
            "train_metrics": {"train_runtime": 1.0},
            "development_metrics": {
                "cer": 0.1,
                "wer": 0.2,
                "chrf": 80.0,
                "correction_f1": 0.5,
            },
            "hardware": {"peak_cuda_memory_bytes": 123},
        }

    monkeypatch.setattr(
        "src.modeling.experiment_runner.train_seq2seq", fake_train
    )
    config = Seq2SeqConfig(model_name_or_path="fake")
    development = _m7_frame("dev input", "dev target")

    first = run_condition(
        _m7_frame(),
        development,
        method="m1",
        regime="synthetic_then_gold",
        seed=2026,
        config=config,
        runs_dir=tmp_path,
    )
    skipped = run_condition(
        _m7_frame(),
        development,
        method="m1",
        regime="synthetic_then_gold",
        seed=2026,
        config=config,
        runs_dir=tmp_path,
    )
    modified = run_condition(
        _m7_frame(source_text="new synthetic input"),
        development,
        method="m1",
        regime="synthetic_then_gold",
        seed=2026,
        config=config,
        runs_dir=tmp_path,
    )
    forced = run_condition(
        _m7_frame(source_text="new synthetic input"),
        development,
        method="m1",
        regime="synthetic_then_gold",
        seed=2026,
        config=config,
        runs_dir=tmp_path,
        rerun=True,
    )

    assert first["attempt"] == 1
    assert first["run_action"] == "trained"
    assert skipped["attempt"] == 1
    assert skipped["run_action"] == "skipped_complete"
    assert modified["attempt"] == 2
    assert forced["attempt"] == 3
    assert (tmp_path / "m1/seed_2026/synthetic_then_gold/attempt_001").is_dir()
    assert (tmp_path / "m1/seed_2026/synthetic_then_gold/attempt_003").is_dir()

    summary = collect_latest_development_results(tmp_path)
    assert len(summary) == 1
    assert summary.iloc[0].attempt == 3


def test_m7_aggregate_and_markdown_results_are_report_ready(tmp_path):
    summary = pd.DataFrame(
        {
            "method": ["m1", "m1"],
            "regime": ["synthetic_then_gold", "synthetic_then_gold"],
            "seed": [1, 2],
            "attempt": [1, 1],
            "cer": [0.2, 0.4],
            "wer": [0.3, 0.5],
            "chrf": [70.0, 74.0],
            "correction_f1": [0.4, 0.6],
        }
    )

    aggregate = aggregate_results(summary)
    markdown_path = tmp_path / "results.md"
    write_markdown_results(summary, markdown_path, title="Results")

    assert aggregate.iloc[0].cer_mean == pytest.approx(0.3)
    assert aggregate.iloc[0].cer_std > 0
    assert aggregate.iloc[0].seed_count == 2
    text = markdown_path.read_text(encoding="utf-8")
    assert "# Results" in text
    assert "| method | regime |" in text


def test_m7_gold_only_run_does_not_require_synthetic_datasets(tmp_path, monkeypatch):
    gold = {
        "train": _m7_frame("gold train input", "gold train target"),
        "dev": _m7_frame("gold dev input", "gold dev target"),
        "test": _m7_frame("gold test input", "gold test target"),
    }
    gold["dev"]["source_id"] = ["dev:1"]
    gold["test"]["source_id"] = ["test:1"]
    monkeypatch.setattr("src.modeling.experiment_runner.load_locked_gold", lambda: gold)
    monkeypatch.setattr(
        "src.modeling.experiment_runner.load_synthetic_pairs",
        lambda *args, **kwargs: pytest.fail("gold-only must not load synthetic data"),
    )
    monkeypatch.setattr(
        "src.modeling.experiment_runner._write_benchmark_configuration", lambda config: None
    )
    monkeypatch.setattr("src.modeling.experiment_runner.write_run_outputs", lambda **kwargs: {})
    monkeypatch.setattr(
        "src.modeling.experiment_runner.run_condition",
        lambda *args, **kwargs: {"method": kwargs["method"], "regime": kwargs["regime"]},
    )

    records = run_training_suite(
        BenchmarkRunConfig(methods=("missing",), seeds=(2026,)),
        regimes=("gold_only",),
        runs_dir=tmp_path,
    )

    assert records == [{"method": "gold", "regime": "gold_only"}]


def test_m7_synthetic_then_gold_builds_prerequisite_and_reports_only_final(
    tmp_path, monkeypatch
):
    gold = {
        "train": _m7_frame("gold train input", "gold train target"),
        "dev": _m7_frame("gold dev input", "gold dev target"),
        "test": _m7_frame("gold test input", "gold test target"),
    }
    gold["dev"]["source_id"] = ["dev:1"]
    gold["test"]["source_id"] = ["test:1"]
    synthetic = _m7_frame("synthetic input", "synthetic target")
    calls = []

    monkeypatch.setattr("src.modeling.experiment_runner.load_locked_gold", lambda: gold)
    monkeypatch.setattr(
        "src.modeling.experiment_runner.load_synthetic_pairs",
        lambda *args, **kwargs: synthetic,
    )
    monkeypatch.setattr(
        "src.modeling.experiment_runner._write_benchmark_configuration", lambda config: None
    )
    monkeypatch.setattr("src.modeling.experiment_runner.write_run_outputs", lambda **kwargs: {})

    def fake_run(training, development, **kwargs):
        attempt = (
            tmp_path
            / kwargs["method"]
            / f"seed_{kwargs['seed']}"
            / kwargs["regime"]
            / "attempt_001"
        )
        (attempt / "best_model").mkdir(parents=True, exist_ok=True)
        (attempt / "run_manifest.json").write_text("{}", encoding="utf-8")
        calls.append({**kwargs, "training_rows": len(training)})
        return {
            "method": kwargs["method"],
            "regime": kwargs["regime"],
            "attempt_dir": str(attempt),
            "experiment_status": "complete",
            "development_metrics": (
                {"cer": 0.1} if kwargs["regime"] == "synthetic_then_gold" else {}
            ),
        }

    monkeypatch.setattr("src.modeling.experiment_runner.run_condition", fake_run)
    records = run_training_suite(
        BenchmarkRunConfig(
            methods=("m1",),
            seeds=(2026,),
            base_model=Seq2SeqConfig(model_name_or_path="fake"),
            synthetic_pretrain_epochs=1.0,
            gold_finetune_epochs=5.0,
        ),
        regimes=("synthetic_then_gold",),
        runs_dir=tmp_path,
    )

    assert [call["regime"] for call in calls] == [
        "synthetic_pretrain",
        "synthetic_then_gold",
    ]
    assert calls[0]["config"].num_train_epochs == 1.0
    assert calls[0]["config"].generate_final_development_metrics is False
    assert calls[1]["config"].num_train_epochs == 5.0
    assert calls[1]["config"].generate_final_development_metrics is True
    assert calls[1]["config"].model_name_or_path.endswith("best_model")
    assert [record["regime"] for record in records] == ["synthetic_then_gold"]

    calls.clear()
    run_training_suite(
        BenchmarkRunConfig(
            methods=("m1",),
            seeds=(2026,),
            base_model=Seq2SeqConfig(
                model_name_or_path="fake", checkpoint_selection_metric="cer"
            ),
            synthetic_pretrain_epochs=3.0,
            gold_finetune_epochs=15.0,
        ),
        regimes=("synthetic_then_gold",),
        runs_dir=tmp_path,
    )
    assert calls[0]["config"].num_train_epochs == 3.0
    assert calls[0]["config"].generate_final_development_metrics is True
    assert calls[1]["config"].num_train_epochs == 15.0


def test_m5_review_finalization_records_quality_and_clears_gate(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    mapping_path = tmp_path / "mapping.csv"
    generation_path = tmp_path / "generation.csv"
    manifest_path.write_text(
        '{"provisional": false, "finalization_required": ["review"]}', encoding="utf-8"
    )
    pd.DataFrame({"mapping_correct": [True, False]}).to_csv(mapping_path, index=False)
    pd.DataFrame(
        {"meaning_preserved": [True, False], "informal_plausibility_1_5": [5, 3]}
    ).to_csv(generation_path, index=False)

    manifest = finalize_m5_manifest(manifest_path, mapping_path, generation_path)

    assert manifest["mapping_review_precision"] == 0.5
    assert manifest["generation_meaning_preservation_rate"] == 0.5
    assert manifest["generation_mean_plausibility"] == 4.0
    assert manifest["finalization_required"] == []
