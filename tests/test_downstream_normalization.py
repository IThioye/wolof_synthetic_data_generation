from __future__ import annotations

from pathlib import Path

import pytest

from src.downstream.injongo import (
    INJONGO_INTENTS,
    load_official_splits,
    make_laya_question,
    validate_official_splits,
)
from src.downstream.metrics import condition_deltas, paired_analysis, paired_category
from src.downstream.noise import build_noisy_records
from src.downstream.tnt_normalizer import resolve_selected_tnt_checkpoint


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_laya_choice_contains_complete_stable_intent_inventory():
    question = make_laya_question()
    assert question["t"] == "choice"
    assert tuple(question["crit"]) == INJONGO_INTENTS
    assert len(question["crit"]) == 40


def test_m1_noise_is_deterministic_and_preserves_pairing_fields():
    source = [
        {
            "example_id": "test-1",
            "row_id": "test:travel_suggestion:test-1",
            "text": "Ñu dem ci kër gi",
            "intent": "travel_suggestion",
            "split": "test",
            "source": "fixture",
        }
    ]
    first = build_noisy_records(source, seed=2026)
    second = build_noisy_records(source, seed=2026)
    assert first == second
    assert first[0]["example_id"] == "test-1"
    assert first[0]["intent"] == "travel_suggestion"
    assert first[0]["real_text"] == source[0]["text"]
    assert first[0]["noise_kind"] == "synthetic noisy Wolof"


@pytest.mark.parametrize(
    ("noisy", "normalized", "expected"),
    [
        ("wrong", "gold", "HELPED"),
        ("gold", "gold", "UNCHANGED_CORRECT"),
        ("wrong", "other", "UNRESOLVED"),
        ("gold", "wrong", "HARMED"),
    ],
)
def test_paired_category(noisy, normalized, expected):
    assert paired_category("gold", noisy, normalized) == expected


def test_paired_rates_and_recovery_formula():
    rows = []
    for index, (noisy, normalized) in enumerate(
        [("wrong", "gold"), ("gold", "gold"), ("wrong", "other"), ("gold", "wrong")]
    ):
        rows.append(
            {
                "example_id": str(index),
                "row_id": str(index),
                "real_text": "r",
                "noisy_text": "n",
                "normalized_text": "z",
                "gold_intent": "gold",
                "real_prediction": "gold",
                "noisy_prediction": noisy,
                "normalized_prediction": normalized,
            }
        )
    enriched, rates = paired_analysis(rows)
    assert {row["category"] for row in enriched} == {
        "HELPED",
        "UNCHANGED_CORRECT",
        "UNRESOLVED",
        "HARMED",
    }
    assert rates["help_rate"] == pytest.approx(0.25)
    assert rates["harm_rate"] == pytest.approx(0.25)
    assert condition_deltas(0.9, 0.5, 0.7)["recovery_rate"] == pytest.approx(0.5)
    assert condition_deltas(0.5, 0.6, 0.7)["recovery_rate"] is None


def test_local_official_injongo_splits_when_downloaded():
    root = REPOSITORY_ROOT / "data/external/injongo_intent_wol"
    if not (root / "train.jsonl").is_file():
        pytest.skip("External official splits are downloaded on demand")
    splits = load_official_splits(root)
    summary = validate_official_splits(splits)
    assert summary["sizes"] == {"train": 2240, "dev": 320, "test": 638}
    assert summary["intent_count"] == 40
    assert set(summary["train_intent_counts"].values()) == {56}


def test_selected_tnt_checkpoint_is_hash_locked_when_present():
    manifest = (
        REPOSITORY_ROOT
        / "results/m7_seq2seq/tnt_hybrid_v3_optimization_v1/selected_candidate.json"
    )
    if not manifest.is_file():
        pytest.skip("Historical local result manifest is not versioned")
    checkpoint, selection = resolve_selected_tnt_checkpoint(REPOSITORY_ROOT)
    assert checkpoint.is_dir()
    assert selection["candidate"] == "m1_v3_gold_adapt_lr_5e5_word_keep_t050"
    assert selection["selection_split"] == "Gold-dev"
