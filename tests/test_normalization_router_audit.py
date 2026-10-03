from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.downstream.routing_audit import (
    CONDITIONS,
    _condition_source_paths,
    _rates,
    _standardize_condition,
)
from src.downstream.routing_baselines import (
    RoutingEvaluation,
    evaluate_decisions,
    grouped_threshold_router,
)
from src.downstream.routing_features import (
    DEPLOYABLE_SURFACE_FEATURES,
    DEPLOYABLE_TNT_FEATURES,
    ORACLE_FEATURES,
    TntSignalExtractor,
    feature_families,
    surface_features,
)
from src.downstream.v2_protocol import TNT_CHECKPOINT


ROOT = Path(__file__).resolve().parents[1]


def test_existing_v2_conditions_reconstruct_expected_routing_labels():
    frames = []
    for name, paths in _condition_source_paths(ROOT).items():
        frame = _standardize_condition(name, paths)
        assert len(frame) == 638
        # Injongo example_id values repeat across intents; row_id is the unique
        # observation key, while grouping on example_id is deliberately stricter.
        assert frame["row_id"].nunique() == 638
        assert frame["routing_label"].isin(
            ["HELPED", "HARMED", "NEUTRAL_CORRECT", "NEUTRAL_WRONG"]
        ).all()
        assert (
            frame["normalize_beneficial"]
            == (frame["routing_label"] == "HELPED").astype(int)
        ).all()
        frames.append(frame)
    assert {frame["condition"].iloc[0] for frame in frames} == set(CONDITIONS)
    expected_ids = set(frames[0]["example_id"])
    assert all(set(frame["example_id"]) == expected_ids for frame in frames)


def test_severity_help_harm_counts_match_frozen_v2():
    paths = _condition_source_paths(ROOT)
    frames = pd.concat(
        [_standardize_condition(name, paths[name]) for name in ("REAL", "M1_LIGHT", "M1_MEDIUM", "M1_HEAVY")],
        ignore_index=True,
    )
    rates = _rates(frames, ["condition"]).set_index("condition")
    assert rates.loc["REAL", "helped"] == 5
    assert rates.loc["REAL", "harmed"] == 24
    assert rates.loc["M1_LIGHT", "helped"] == 24
    assert rates.loc["M1_LIGHT", "harmed"] == 37
    assert rates.loc["M1_MEDIUM", "helped"] == 58
    assert rates.loc["M1_MEDIUM", "harmed"] == 40
    assert rates.loc["M1_HEAVY", "helped"] == 110
    assert rates.loc["M1_HEAVY", "harmed"] == 35


def test_feature_families_keep_oracle_separate_from_deployable():
    families = feature_families()
    deployable = set(families["deployable_all"])
    oracle = set(families["oracle_analysis_only"])
    assert deployable
    assert oracle
    assert deployable.isdisjoint(oracle)
    assert set(DEPLOYABLE_SURFACE_FEATURES).issubset(deployable)
    assert set(DEPLOYABLE_TNT_FEATURES).issubset(deployable)
    assert set(ORACLE_FEATURES) == oracle


def test_surface_features_do_not_treat_oov_as_automatically_erroneous():
    values = surface_features("Defal Xam-Xam 2026", {"defal"})
    assert values["sentence_token_count"] == 3
    assert values["formal_vocabulary_fraction"] == pytest.approx(1 / 3)
    assert values["oov_fraction"] == pytest.approx(2 / 3)
    assert "error" not in values


def test_routing_evaluation_uses_before_or_after_prediction():
    frame = pd.DataFrame(
        {
            "gold_intent": ["alarm", "alarm"],
            "laya_prediction_before": ["alarm", "weather"],
            "laya_prediction_after": ["weather", "alarm"],
            "routing_label": ["HARMED", "HELPED"],
        }
    )
    result = evaluate_decisions(
        frame,
        RoutingEvaluation("perfect", "test", np.asarray([False, True])),
    )
    assert result["accuracy"] == 1.0
    assert result["helped_routed_to_tnt"] == 1
    assert result["harmed_successfully_skipped"] == 1


def test_grouped_threshold_router_produces_one_decision_per_variant():
    rows = []
    for group in range(20):
        for condition in ("clean", "heavy"):
            rows.append(
                {
                    "example_id": f"id-{group}",
                    "routing_group_id": f"id-{group}",
                    "score": float(condition == "heavy"),
                    "correct_before": condition == "clean",
                    "correct_after": condition == "heavy",
                    "routing_label": "NEUTRAL_CORRECT" if condition == "clean" else "HELPED",
                }
            )
    frame = pd.DataFrame(rows)
    evaluation, folds = grouped_threshold_router(frame, ["score"], folds=5)
    assert len(evaluation.decisions) == len(frame)
    assert len(folds) == 5
    assert (folds["train_groups"] + folds["test_groups"] == 20).all()


@pytest.mark.skipif(not (ROOT / TNT_CHECKPOINT / "model.safetensors").is_file(), reason="local frozen checkpoint unavailable")
def test_frozen_tnt_keep_signals_are_extractable_without_training():
    extractor = TntSignalExtractor(ROOT / TNT_CHECKPOINT, device="cpu", batch_size=1)
    signals = extractor.extract(["Déffal ab fàttali."])[0]
    assert 0.0 <= signals["tnt_operation_mean_keep_probability"] <= 1.0
    assert 0.0 <= signals["tnt_word_mean_keep_probability"] <= 1.0
    assert 0.0 <= signals["tnt_boundary_mean_no_change_probability"] <= 1.0

