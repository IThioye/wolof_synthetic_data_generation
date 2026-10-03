"""Text restoration, lexical-cue, and downstream error analysis utilities."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from src.modeling.metrics import (
    corpus_chrf,
    evaluation_tokens,
    generation_metrics,
    levenshtein_distance,
    normalize_for_evaluation,
)


def pair_text_metrics(reference: str, candidate: str) -> dict[str, float | int | bool]:
    reference = normalize_for_evaluation(reference)
    candidate = normalize_for_evaluation(candidate)
    reference_tokens = evaluation_tokens(reference)
    candidate_tokens = evaluation_tokens(candidate)
    char_edits = levenshtein_distance(list(candidate), list(reference))
    word_edits = levenshtein_distance(candidate_tokens, reference_tokens)
    matcher = SequenceMatcher(a=reference_tokens, b=candidate_tokens, autojunk=False)
    equal_tokens = sum(
        source_end - source_start
        for operation, source_start, source_end, _, _ in matcher.get_opcodes()
        if operation == "equal"
    )
    token_denominator = max(len(reference_tokens), len(candidate_tokens), 1)
    return {
        "char_edits": char_edits,
        "cer": char_edits / max(len(reference), 1),
        "word_edits": word_edits,
        "wer": word_edits / max(len(reference_tokens), 1),
        "chrf": corpus_chrf([candidate], [reference]),
        "exact": candidate == reference,
        "token_identity_rate": equal_tokens / token_denominator,
        "modified_token_count": word_edits,
    }


def token_edit_details(source: str, target: str) -> dict[str, object]:
    source_tokens = evaluation_tokens(source)
    target_tokens = evaluation_tokens(target)
    matcher = SequenceMatcher(a=source_tokens, b=target_tokens, autojunk=False)
    removed: list[str] = []
    added: list[str] = []
    substitutions: list[dict[str, list[str]]] = []
    operations: list[dict[str, object]] = []
    for operation, source_start, source_end, target_start, target_end in matcher.get_opcodes():
        if operation == "equal":
            continue
        old = source_tokens[source_start:source_end]
        new = target_tokens[target_start:target_end]
        removed.extend(old)
        added.extend(new)
        if operation == "replace":
            substitutions.append({"source": old, "target": new})
        operations.append(
            {
                "operation": operation,
                "source": old,
                "target": new,
                "source_start": source_start,
                "target_start": target_start,
            }
        )
    return {
        "removed_tokens": removed,
        "added_tokens": added,
        "substitutions": substitutions,
        "token_operations": operations,
        "spacing_changed": len(source.split()) != len(target.split()),
    }


def corpus_restoration_metrics(reference: Sequence[str], candidate: Sequence[str]) -> dict:
    metrics = dict(generation_metrics(candidate, reference))
    row_metrics = [pair_text_metrics(real, output) for real, output in zip(reference, candidate)]
    metrics.update(
        {
            "sentence_identity_rate": float(np.mean([row["exact"] for row in row_metrics])),
            "token_identity_rate": float(np.mean([row["token_identity_rate"] for row in row_metrics])),
            "changed_sentences": int(sum(not row["exact"] for row in row_metrics)),
            "mean_character_edits": float(np.mean([row["char_edits"] for row in row_metrics])),
            "mean_modified_tokens": float(np.mean([row["modified_token_count"] for row in row_metrics])),
        }
    )
    return metrics


def learn_intent_cues(
    train_rows: Sequence[Mapping],
    *,
    top_k: int = 25,
    smoothing: float = 0.5,
) -> dict[str, list[dict[str, float | int | str]]]:
    """Learn transparent token log-odds cues from official train only."""

    by_intent: dict[str, Counter[str]] = defaultdict(Counter)
    global_counts: Counter[str] = Counter()
    for row in train_rows:
        tokens = [token for token in evaluation_tokens(row["text"]) if len(token) > 1]
        by_intent[str(row["intent"])].update(tokens)
        global_counts.update(tokens)
    vocabulary_size = max(1, len(global_counts))
    output: dict[str, list[dict[str, float | int | str]]] = {}
    total_tokens = sum(global_counts.values())
    for intent, counts in by_intent.items():
        intent_total = sum(counts.values())
        other_total = total_tokens - intent_total
        candidates = []
        for token, count in counts.items():
            other_count = global_counts[token] - count
            in_rate = (count + smoothing) / (intent_total + smoothing * vocabulary_size)
            out_rate = (other_count + smoothing) / (other_total + smoothing * vocabulary_size)
            candidates.append(
                {
                    "token": token,
                    "intent_count": int(count),
                    "other_count": int(other_count),
                    "log_odds": float(math.log(in_rate / out_rate)),
                }
            )
        candidates.sort(key=lambda row: (-float(row["log_odds"]), -int(row["intent_count"]), str(row["token"])))
        output[intent] = candidates[:top_k]
    return dict(sorted(output.items()))


def _cue_sets(cues: Mapping[str, Sequence[Mapping]]) -> dict[str, set[str]]:
    return {
        intent: {str(item["token"]) for item in items}
        for intent, items in cues.items()
    }


def build_detailed_error_rows(
    paired: pd.DataFrame,
    *,
    intent_cues: Mapping[str, Sequence[Mapping]],
    formal_vocabulary: set[str],
) -> pd.DataFrame:
    cue_sets = _cue_sets(intent_cues)
    records: list[dict] = []
    for row in paired.to_dict(orient="records"):
        real_text = str(row["real_text"])
        noisy_text = str(row["noisy_text"])
        normalized_text = str(row["normalized_text"])
        noisy_metrics = pair_text_metrics(real_text, noisy_text)
        normalized_metrics = pair_text_metrics(real_text, normalized_text)
        edit_details = token_edit_details(noisy_text, normalized_text)
        real_tokens = set(evaluation_tokens(real_text))
        noisy_tokens = set(evaluation_tokens(noisy_text))
        normalized_tokens = set(evaluation_tokens(normalized_text))
        cues = cue_sets.get(str(row["gold_intent"]), set())
        real_cues = cues & real_tokens
        cues_lost_by_noise = real_cues - noisy_tokens
        cues_restored = cues_lost_by_noise & normalized_tokens
        cues_not_restored = cues_lost_by_noise - normalized_tokens
        cues_harmed = (real_cues & noisy_tokens) - normalized_tokens
        added_real_words = sorted(set(edit_details["added_tokens"]) & formal_vocabulary)
        delta_cer = float(noisy_metrics["cer"]) - float(normalized_metrics["cer"])
        delta_wer = float(noisy_metrics["wer"]) - float(normalized_metrics["wer"])
        delta_chrf = float(normalized_metrics["chrf"]) - float(noisy_metrics["chrf"])
        confidence_delta = float(row["normalized_gold_confidence"]) - float(row["noisy_gold_confidence"])

        flags = {
            "normalization_insufficient": (
                (delta_cer > 0 or delta_wer > 0)
                and (float(normalized_metrics["cer"]) > 0.10 or float(normalized_metrics["wer"]) > 0.25)
            ),
            "wrong_or_harmful_normalization": delta_cer < -0.01 or delta_wer < -0.05,
            "discriminative_cue_not_restored": bool(cues_not_restored),
            "close_to_real_but_laya_wrong": (
                float(normalized_metrics["cer"]) <= 0.10
                and float(normalized_metrics["wer"]) <= 0.25
            ),
            "highly_distorted_or_ambiguous": (
                float(noisy_metrics["cer"]) >= 0.35
                or float(noisy_metrics["wer"]) >= 0.70
            ),
        }
        taxonomy = "F_other_or_unclear"
        if str(row["category"]) == "UNRESOLVED":
            if flags["wrong_or_harmful_normalization"]:
                taxonomy = "B_wrong_normalization_or_harmful_edit"
            elif flags["close_to_real_but_laya_wrong"]:
                taxonomy = "D_close_to_real_but_laya_still_wrong"
            elif flags["highly_distorted_or_ambiguous"]:
                taxonomy = "E_highly_distorted_or_ambiguous"
            elif flags["discriminative_cue_not_restored"]:
                taxonomy = "C_discriminative_cue_not_restored"
            elif flags["normalization_insufficient"]:
                taxonomy = "A_normalization_insufficient"

        records.append(
            {
                **row,
                "noisy_cer": noisy_metrics["cer"],
                "normalized_cer": normalized_metrics["cer"],
                "delta_cer": delta_cer,
                "noisy_wer": noisy_metrics["wer"],
                "normalized_wer": normalized_metrics["wer"],
                "delta_wer": delta_wer,
                "noisy_chrf": noisy_metrics["chrf"],
                "normalized_chrf": normalized_metrics["chrf"],
                "delta_chrf": delta_chrf,
                "normalized_equals_real": normalized_metrics["exact"],
                "noisy_gold_confidence": float(row["noisy_gold_confidence"]),
                "normalized_gold_confidence": float(row["normalized_gold_confidence"]),
                "real_gold_confidence": float(row["real_gold_confidence"]),
                "gold_confidence_delta": confidence_delta,
                "removed_tokens_json": json.dumps(edit_details["removed_tokens"], ensure_ascii=False),
                "added_tokens_json": json.dumps(edit_details["added_tokens"], ensure_ascii=False),
                "token_operations_json": json.dumps(edit_details["token_operations"], ensure_ascii=False),
                "spacing_changed": bool(edit_details["spacing_changed"]),
                "real_word_substitution_candidate": bool(added_real_words),
                "added_real_words_json": json.dumps(added_real_words, ensure_ascii=False),
                "real_cues_json": json.dumps(sorted(real_cues), ensure_ascii=False),
                "cues_restored_json": json.dumps(sorted(cues_restored), ensure_ascii=False),
                "cues_not_restored_json": json.dumps(sorted(cues_not_restored), ensure_ascii=False),
                "cues_harmed_json": json.dumps(sorted(cues_harmed), ensure_ascii=False),
                **flags,
                "unresolved_taxonomy": taxonomy,
            }
        )
    return pd.DataFrame(records)


def grouped_restoration_statistics(analysis: pd.DataFrame) -> pd.DataFrame:
    rows = []
    columns = ("delta_cer", "delta_wer", "delta_chrf", "gold_confidence_delta")
    for category, group in analysis.groupby("category", sort=False):
        row: dict[str, object] = {"category": category, "rows": len(group)}
        for column in columns:
            values = group[column].astype(float)
            row[f"{column}_mean"] = float(values.mean())
            row[f"{column}_median"] = float(values.median())
            row[f"{column}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def restoration_confidence_correlations(analysis: pd.DataFrame) -> dict:
    from scipy.stats import pearsonr, spearmanr

    output = {}
    confidence = analysis["gold_confidence_delta"].astype(float)
    for column in ("delta_cer", "delta_wer", "delta_chrf"):
        values = analysis[column].astype(float)
        pearson = pearsonr(values, confidence)
        spearman = spearmanr(values, confidence)
        output[column] = {
            "pearson_r": float(pearson.statistic),
            "pearson_p": float(pearson.pvalue),
            "spearman_rho": float(spearman.statistic),
            "spearman_p": float(spearman.pvalue),
            "interpretation": "association only; not a causal estimate",
        }
    return output
