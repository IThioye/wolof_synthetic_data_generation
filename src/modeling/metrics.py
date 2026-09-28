"""Normalization metrics used by every M7 condition."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from difflib import SequenceMatcher
from typing import Sequence


TOKEN_RE = re.compile(r"\w+(?:[-’']\w+)*|[^\w\s]", flags=re.UNICODE)


def normalize_for_evaluation(text: object) -> str:
    normalized = unicodedata.normalize("NFC", str(text) if text is not None else "")
    return " ".join(normalized.split())


def evaluation_tokens(text: object) -> list[str]:
    return TOKEN_RE.findall(normalize_for_evaluation(text).casefold())


def levenshtein_distance(left: Sequence, right: Sequence) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_item in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_item in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_item != right_item),
                )
            )
        previous = current
    return previous[-1]


def _ngram_counts(text: str, order: int) -> Counter[str]:
    compact = "".join(text.split())
    return Counter(compact[index : index + order] for index in range(len(compact) - order + 1))


def corpus_chrf(
    predictions: Sequence[str],
    references: Sequence[str],
    *,
    char_order: int = 6,
    beta: float = 2.0,
) -> float:
    """Calculate corpus chrF (character n-grams, order 6, beta 2)."""
    precisions: list[float] = []
    recalls: list[float] = []
    for order in range(1, char_order + 1):
        matches = predicted_total = reference_total = 0
        for prediction, reference in zip(predictions, references):
            predicted = _ngram_counts(normalize_for_evaluation(prediction), order)
            expected = _ngram_counts(normalize_for_evaluation(reference), order)
            matches += sum((predicted & expected).values())
            predicted_total += sum(predicted.values())
            reference_total += sum(expected.values())
        if predicted_total or reference_total:
            precisions.append(matches / predicted_total if predicted_total else 0.0)
            recalls.append(matches / reference_total if reference_total else 0.0)
    precision = sum(precisions) / len(precisions) if precisions else 0.0
    recall = sum(recalls) / len(recalls) if recalls else 0.0
    beta_squared = beta**2
    denominator = beta_squared * precision + recall
    return 100.0 * (1 + beta_squared) * precision * recall / denominator if denominator else 0.0


def _edit_signatures(source: str, target: str) -> set[tuple[int, int, tuple[str, ...]]]:
    source_tokens = evaluation_tokens(source)
    target_tokens = evaluation_tokens(target)
    matcher = SequenceMatcher(a=source_tokens, b=target_tokens, autojunk=False)
    return {
        (source_start, source_end, tuple(target_tokens[target_start:target_end]))
        for operation, source_start, source_end, target_start, target_end in matcher.get_opcodes()
        if operation != "equal"
    }


def generation_metrics(predictions: Sequence[str], references: Sequence[str]) -> dict[str, float | int]:
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have equal length")
    normalized_predictions = [normalize_for_evaluation(value) for value in predictions]
    normalized_references = [normalize_for_evaluation(value) for value in references]
    char_edits = sum(
        levenshtein_distance(list(prediction), list(reference))
        for prediction, reference in zip(normalized_predictions, normalized_references)
    )
    reference_characters = sum(len(reference) for reference in normalized_references)
    word_edits = sum(
        levenshtein_distance(evaluation_tokens(prediction), evaluation_tokens(reference))
        for prediction, reference in zip(normalized_predictions, normalized_references)
    )
    reference_words = sum(len(evaluation_tokens(reference)) for reference in normalized_references)
    exact = sum(
        prediction == reference
        for prediction, reference in zip(normalized_predictions, normalized_references)
    )
    return {
        "rows": len(predictions),
        "cer": char_edits / reference_characters if reference_characters else 0.0,
        "wer": word_edits / reference_words if reference_words else 0.0,
        "chrf": corpus_chrf(normalized_predictions, normalized_references),
        "exact_match": exact / len(predictions) if predictions else 0.0,
    }


def normalization_metrics(
    sources: Sequence[str],
    predictions: Sequence[str],
    references: Sequence[str],
) -> dict[str, float | int]:
    if not (len(sources) == len(predictions) == len(references)):
        raise ValueError("Sources, predictions, and references must have equal length")
    result = generation_metrics(predictions, references)
    true_positive = false_positive = false_negative = 0
    overcorrected_unchanged = unchanged_rows = 0
    for source, prediction, reference in zip(sources, predictions, references):
        gold_edits = _edit_signatures(source, reference)
        predicted_edits = _edit_signatures(source, prediction)
        true_positive += len(gold_edits & predicted_edits)
        false_positive += len(predicted_edits - gold_edits)
        false_negative += len(gold_edits - predicted_edits)
        if normalize_for_evaluation(source) == normalize_for_evaluation(reference):
            unchanged_rows += 1
            overcorrected_unchanged += normalize_for_evaluation(prediction) != normalize_for_evaluation(reference)
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    precision = true_positive / precision_denominator if precision_denominator else 0.0
    recall = true_positive / recall_denominator if recall_denominator else 0.0
    result.update(
        {
            "correction_tp": true_positive,
            "correction_fp": false_positive,
            "correction_fn": false_negative,
            "correction_precision": precision,
            "correction_recall": recall,
            "correction_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "overcorrection_rate": false_positive / precision_denominator if precision_denominator else 0.0,
            "unchanged_reference_rows": unchanged_rows,
            "unchanged_sentence_overcorrection_rate": (
                overcorrected_unchanged / unchanged_rows if unchanged_rows else 0.0
            ),
        }
    )
    return result
