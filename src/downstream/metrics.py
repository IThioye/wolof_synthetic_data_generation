"""Intent-classification and paired normalization-effect metrics."""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np


PAIRED_CATEGORIES = (
    "HELPED",
    "UNCHANGED_CORRECT",
    "UNRESOLVED",
    "HARMED",
)


def classification_metrics(
    gold: Sequence[str],
    predicted: Sequence[str],
    labels: Sequence[str],
) -> tuple[dict, list[dict], np.ndarray]:
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        precision_recall_fscore_support,
    )

    if len(gold) != len(predicted):
        raise ValueError("gold and predicted lengths differ")
    labels = list(labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        gold, predicted, labels=labels, zero_division=0
    )
    per_class = [
        {
            "intent": label,
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, label in enumerate(labels)
    ]
    metrics = {
        "rows": len(gold),
        "accuracy": float(accuracy_score(gold, predicted)),
        "macro_f1": float(np.mean(f1)),
    }
    return metrics, per_class, confusion_matrix(gold, predicted, labels=labels)


def paired_category(gold: str, noisy_prediction: str, normalized_prediction: str) -> str:
    noisy_correct = noisy_prediction == gold
    normalized_correct = normalized_prediction == gold
    if not noisy_correct and normalized_correct:
        return "HELPED"
    if noisy_correct and normalized_correct:
        return "UNCHANGED_CORRECT"
    if not noisy_correct and not normalized_correct:
        return "UNRESOLVED"
    return "HARMED"


def paired_analysis(rows: Iterable[Mapping]) -> tuple[list[dict], dict]:
    """Attach the four predeclared paired outcomes and calculate their rates."""

    output: list[dict] = []
    counts = {category: 0 for category in PAIRED_CATEGORIES}
    for row in rows:
        required = {
            "example_id",
            "row_id",
            "real_text",
            "noisy_text",
            "normalized_text",
            "gold_intent",
            "real_prediction",
            "noisy_prediction",
            "normalized_prediction",
        }
        missing = required.difference(row)
        if missing:
            raise ValueError(f"Paired row missing fields: {sorted(missing)}")
        category = paired_category(
            str(row["gold_intent"]),
            str(row["noisy_prediction"]),
            str(row["normalized_prediction"]),
        )
        enriched = dict(row)
        enriched["category"] = category
        output.append(enriched)
        counts[category] += 1
    denominator = len(output)
    rates = {
        "rows": denominator,
        "help_rate": counts["HELPED"] / denominator if denominator else 0.0,
        "harm_rate": counts["HARMED"] / denominator if denominator else 0.0,
        "unchanged_correct_rate": (
            counts["UNCHANGED_CORRECT"] / denominator if denominator else 0.0
        ),
        "unresolved_rate": counts["UNRESOLVED"] / denominator if denominator else 0.0,
        "counts": counts,
    }
    return output, rates


def condition_deltas(real: float, noisy: float, normalized: float) -> dict[str, float | None]:
    drop = real - noisy
    gain = normalized - noisy
    return {
        "real": float(real),
        "noisy": float(noisy),
        "normalized": float(normalized),
        "noise_drop": float(drop),
        "normalization_gain": float(gain),
        "recovery_rate": float(gain / drop) if drop > 0 else None,
    }
