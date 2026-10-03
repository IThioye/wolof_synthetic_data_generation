"""Simple, leakage-controlled routing baselines for frozen paired outcomes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from src.downstream.injongo import INJONGO_INTENTS
from src.downstream.metrics import classification_metrics


@dataclass(frozen=True)
class RoutingEvaluation:
    strategy: str
    feature_set: str
    decisions: np.ndarray


def evaluate_decisions(frame: pd.DataFrame, evaluation: RoutingEvaluation) -> dict:
    normalize = np.asarray(evaluation.decisions, dtype=bool)
    predictions = np.where(
        normalize,
        frame["laya_prediction_after"].astype(str).to_numpy(),
        frame["laya_prediction_before"].astype(str).to_numpy(),
    )
    metrics, _, _ = classification_metrics(
        frame["gold_intent"].astype(str).tolist(),
        predictions.tolist(),
        INJONGO_INTENTS,
    )
    categories = frame["routing_label"].astype(str).to_numpy()
    return {
        "strategy": evaluation.strategy,
        "feature_set": evaluation.feature_set,
        "rows": len(frame),
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "number_normalized": int(normalize.sum()),
        "normalization_rate": float(normalize.mean()),
        "helped_routed_to_tnt": int(((categories == "HELPED") & normalize).sum()),
        "harmed_routed_to_tnt": int(((categories == "HARMED") & normalize).sum()),
        "helped_incorrectly_skipped": int(((categories == "HELPED") & ~normalize).sum()),
        "harmed_successfully_skipped": int(((categories == "HARMED") & ~normalize).sum()),
    }


def baseline_evaluations(frame: pd.DataFrame) -> list[RoutingEvaluation]:
    category = frame["routing_label"].astype(str).to_numpy()
    return [
        RoutingEvaluation("always_skip", "none", np.zeros(len(frame), dtype=bool)),
        RoutingEvaluation("always_normalize", "none", np.ones(len(frame), dtype=bool)),
        RoutingEvaluation("oracle_help_only", "oracle_outcome", category == "HELPED"),
        RoutingEvaluation(
            "oracle_choose_best",
            "oracle_outcome",
            (category == "HELPED"),
        ),
    ]


def _candidate_thresholds(values: np.ndarray) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return np.asarray([0.0])
    return np.unique(
        np.concatenate(
            ([finite.min() - 1e-12], np.quantile(finite, np.linspace(0.02, 0.98, 49)), [finite.max() + 1e-12])
        )
    )


def _best_threshold(frame: pd.DataFrame, features: Sequence[str]) -> tuple[str, float, bool]:
    """Choose feature, threshold and direction by downstream train accuracy."""

    best: tuple[float, float, str, float, bool] | None = None
    before = frame["correct_before"].astype(bool).to_numpy()
    after = frame["correct_after"].astype(bool).to_numpy()
    for feature in features:
        values = frame[feature].astype(float).to_numpy()
        for threshold in _candidate_thresholds(values):
            for greater in (True, False):
                decision = values >= threshold if greater else values <= threshold
                correct = np.where(decision, after, before)
                score = float(correct.mean())
                normalization_rate = float(decision.mean())
                candidate = (score, -normalization_rate, feature, float(threshold), greater)
                if best is None or candidate[:2] > best[:2]:
                    best = candidate
    if best is None:
        raise ValueError("No routing features supplied")
    return best[2], best[3], best[4]


def grouped_threshold_router(
    frame: pd.DataFrame,
    features: Sequence[str],
    *,
    folds: int = 5,
    seed: int = 2026,
    strategy: str = "grouped_threshold",
    feature_set: str = "deployable",
) -> tuple[RoutingEvaluation, pd.DataFrame]:
    from sklearn.model_selection import GroupKFold

    del seed  # GroupKFold is deterministic; retained in the public protocol.
    decisions = np.zeros(len(frame), dtype=bool)
    fold_rows = []
    splitter = GroupKFold(n_splits=folds)
    dummy = np.zeros(len(frame))
    group_column = "routing_group_id" if "routing_group_id" in frame else "example_id"
    groups = frame[group_column].astype(str).to_numpy()
    for fold, (train_indices, test_indices) in enumerate(splitter.split(dummy, groups=groups)):
        train = frame.iloc[train_indices]
        feature, threshold, greater = _best_threshold(train, features)
        values = frame.iloc[test_indices][feature].astype(float).to_numpy()
        decisions[test_indices] = values >= threshold if greater else values <= threshold
        fold_rows.append(
            {
                "fold": fold,
                "selected_feature": feature,
                "threshold": threshold,
                "direction": "greater_or_equal" if greater else "less_or_equal",
                "group_column": group_column,
                "train_groups": train[group_column].nunique(),
                "test_groups": frame.iloc[test_indices][group_column].nunique(),
            }
        )
    return RoutingEvaluation(strategy, feature_set, decisions), pd.DataFrame(fold_rows)


def _select_probability_threshold(frame: pd.DataFrame, probabilities: np.ndarray) -> float:
    before = frame["correct_before"].astype(bool).to_numpy()
    after = frame["correct_after"].astype(bool).to_numpy()
    best = (-1.0, 0.0, 1.0)
    for threshold in _candidate_thresholds(probabilities):
        decision = probabilities >= threshold
        score = float(np.where(decision, after, before).mean())
        candidate = (score, -float(decision.mean()), float(threshold))
        if candidate[:2] > best[:2]:
            best = candidate
    return best[2]


def grouped_classifier_router(
    frame: pd.DataFrame,
    features: Sequence[str],
    *,
    model_name: str,
    folds: int = 5,
    inner_folds: int = 4,
    seed: int = 2026,
    feature_set: str = "deployable",
) -> tuple[RoutingEvaluation, pd.DataFrame]:
    """Nested grouped CV; threshold selection never sees an outer test group."""

    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier

    def make_model(offset: int):
        if model_name == "logistic_regression":
            estimator = LogisticRegression(
                class_weight="balanced", max_iter=2000, random_state=seed + offset
            )
            return Pipeline(
                [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", estimator)]
            )
        if model_name == "decision_tree":
            estimator = DecisionTreeClassifier(
                max_depth=3,
                min_samples_leaf=30,
                class_weight="balanced",
                random_state=seed + offset,
            )
            return Pipeline(
                [("impute", SimpleImputer(strategy="median")), ("model", estimator)]
            )
        raise ValueError(f"Unknown router model: {model_name}")

    X = frame[list(features)].astype(float)
    y = frame["normalize_beneficial"].astype(int).to_numpy()
    group_column = "routing_group_id" if "routing_group_id" in frame else "example_id"
    groups = frame[group_column].astype(str).to_numpy()
    outer = GroupKFold(n_splits=folds)
    decisions = np.zeros(len(frame), dtype=bool)
    fold_rows = []
    for fold, (train_indices, test_indices) in enumerate(outer.split(X, y, groups)):
        inner_frame = frame.iloc[train_indices].reset_index(drop=True)
        inner_X = X.iloc[train_indices].reset_index(drop=True)
        inner_y = y[train_indices]
        inner_groups = groups[train_indices]
        inner_probabilities = np.zeros(len(train_indices), dtype=float)
        inner = GroupKFold(n_splits=inner_folds)
        for inner_fold, (fit_indices, validation_indices) in enumerate(
            inner.split(inner_X, inner_y, inner_groups)
        ):
            model = make_model(fold * 100 + inner_fold)
            model.fit(inner_X.iloc[fit_indices], inner_y[fit_indices])
            inner_probabilities[validation_indices] = model.predict_proba(
                inner_X.iloc[validation_indices]
            )[:, 1]
        threshold = _select_probability_threshold(inner_frame, inner_probabilities)
        final_model = make_model(fold)
        final_model.fit(X.iloc[train_indices], y[train_indices])
        probabilities = final_model.predict_proba(X.iloc[test_indices])[:, 1]
        decisions[test_indices] = probabilities >= threshold
        fold_rows.append(
            {
                "fold": fold,
                "model": model_name,
                "group_column": group_column,
                "threshold": threshold,
                "train_groups": len(set(groups[train_indices])),
                "test_groups": len(set(groups[test_indices])),
                "positive_train_rows": int(y[train_indices].sum()),
            }
        )
    return (
        RoutingEvaluation(f"grouped_{model_name}", feature_set, decisions),
        pd.DataFrame(fold_rows),
    )

