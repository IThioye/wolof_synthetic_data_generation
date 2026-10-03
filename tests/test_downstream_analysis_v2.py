from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.downstream.injongo import load_official_splits
from src.downstream.metrics import paired_analysis
from src.downstream.noise_v2 import (
    M1_SEVERITIES,
    build_m1_severity_records,
    method_applicability,
)
from src.downstream.v2_protocol import (
    V1_LAYA_SHA256,
    V1_REAL_ACCURACY,
    V1_REAL_MACRO_F1,
    V1_TNT_SHA256,
    assert_v1_real_baseline,
    validate_frozen_artifacts,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def official_splits():
    path = ROOT / "data/external/injongo_intent_wol"
    if not (path / "train.jsonl").is_file():
        pytest.skip("Official Injongo splits are downloaded on demand")
    return load_official_splits(path)


def test_v1_real_reference_and_drift_guard():
    assert_v1_real_baseline(
        {"accuracy": V1_REAL_ACCURACY, "macro_f1": V1_REAL_MACRO_F1}
    )
    with pytest.raises(RuntimeError, match="REAL baseline drift"):
        assert_v1_real_baseline(
            {"accuracy": V1_REAL_ACCURACY - 0.01, "macro_f1": V1_REAL_MACRO_F1}
        )


def test_frozen_checkpoint_hashes_are_unchanged_when_local():
    if not (ROOT / "checkpoints/downstream/laya_normalization_v1/model.safetensors").is_file():
        pytest.skip("Local frozen checkpoints are intentionally ignored by Git")
    frozen = validate_frozen_artifacts(ROOT)
    assert frozen["laya_model_sha256"] == V1_LAYA_SHA256
    assert frozen["tnt_model_sha256"] == V1_TNT_SHA256


def test_official_splits_remain_disjoint(official_splits):
    keys = {
        split: {(row["intent"], row["example_id"]) for row in rows}
        for split, rows in official_splits.items()
    }
    assert keys["train"].isdisjoint(keys["dev"])
    assert keys["train"].isdisjoint(keys["test"])
    assert keys["dev"].isdisjoint(keys["test"])


def test_controlled_m1_is_deterministic_paired_and_ordered(official_splits):
    test = official_splits["test"]
    conditions = {
        severity.name: build_m1_severity_records(test, severity, seed=2026)
        for severity in M1_SEVERITIES
    }
    assert conditions["light"] == build_m1_severity_records(
        test, M1_SEVERITIES[0], seed=2026
    )
    expected_pairing = [(row["row_id"], row["intent"]) for row in test]
    for rows in conditions.values():
        assert [(row["row_id"], row["intent"]) for row in rows] == expected_pairing
    means = {
        name: sum(row["operation_count"] for row in rows) / len(rows)
        for name, rows in conditions.items()
    }
    assert means["light"] < means["medium"] < means["heavy"]
    assert means["light"] == pytest.approx(2.0, abs=0.15)
    assert means["medium"] == pytest.approx(5.0, abs=0.15)
    assert means["heavy"] == pytest.approx(9.31191222570533)


def test_m1_heavy_reproduces_v1_text_when_outputs_exist(official_splits):
    v1_path = ROOT / "results/downstream/laya_normalization_v1/paired_text_conditions.csv"
    if not v1_path.is_file():
        pytest.skip("Frozen V1 text output is unavailable")
    import pandas as pd

    observed = build_m1_severity_records(
        official_splits["test"], M1_SEVERITIES[-1], seed=2026
    )
    expected = pd.read_csv(v1_path).set_index("row_id")["noisy_text"].to_dict()
    assert all(row["noisy_text"] == expected[row["row_id"]] for row in observed)


def test_paired_category_counts_always_sum_to_rows():
    rows = []
    outcomes = [("wrong", "gold"), ("gold", "gold"), ("wrong", "other"), ("gold", "wrong")]
    for index, (noisy, normalized) in enumerate(outcomes):
        rows.append(
            {
                "example_id": str(index), "row_id": str(index),
                "real_text": "real", "noisy_text": "noisy", "normalized_text": "norm",
                "gold_intent": "gold", "real_prediction": "gold",
                "noisy_prediction": noisy, "normalized_prediction": normalized,
            }
        )
    _, rates = paired_analysis(rows)
    assert sum(rates["counts"].values()) == rates["rows"] == 4


def test_cross_method_applicability_is_explicit():
    statuses = {row["method"]: row for row in method_applicability()}
    assert statuses["M1"]["applicable"] is True
    assert statuses["M5"]["applicable"] is True
    assert statuses["M6"]["applicable"] is True
    for method in ("M2", "M3", "M4"):
        assert statuses[method]["applicable"] is False
        assert statuses[method]["reason"]
