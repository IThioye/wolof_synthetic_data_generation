"""Post-hoc head diagnostics for a trained TNT hybrid checkpoint."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Sequence

import pandas as pd

from src.config import LINGUISTIC_ANNOTATIONS_PATH, M7_RESULTS_DIR
from src.modeling.tnt_edit_transformer import OPERATION_NAMES
from src.modeling.tnt_hybrid_transformer import (
    BOUNDARY_ACTIONS,
    WORD_ACTIONS,
    BOUNDARY_DELETE,
    BOUNDARY_INSERT_BEFORE,
    BOUNDARY_KEEP,
    WORD_KEEP,
    TntHybridTransformerForConditionalGeneration,
    hybrid_tnt_annotations,
)


GATE_RESULTS_DIR = M7_RESULTS_DIR / "tnt_hybrid_m1_gate_v2"
DIAGNOSTIC_DIR = GATE_RESULTS_DIR / "head_diagnostics"


def _mode(values: pd.Series, default: str = "unknown") -> str:
    cleaned = values.fillna("").astype(str).str.strip()
    cleaned = cleaned.loc[cleaned.ne("")]
    if cleaned.empty:
        return default
    counts = cleaned.value_counts()
    return str(sorted(counts.loc[counts.eq(counts.max())].index)[0])


def _token_key(value: object) -> str:
    normalized = str(value or "").casefold().strip()
    return re.sub(r"^\W+|\W+$", "", normalized, flags=re.UNICODE)


def _linguistic_lookup() -> dict[str, dict[str, str]]:
    if not LINGUISTIC_ANNOTATIONS_PATH.is_file():
        return {}
    frame = pd.read_csv(LINGUISTIC_ANNOTATIONS_PATH, keep_default_na=False)
    if "review_status" in frame:
        reviewed = frame.loc[frame["review_status"].eq("reviewed")]
        if not reviewed.empty:
            frame = reviewed
    rows: list[dict[str, str]] = []
    for token_column in ("informal_token", "formal_token"):
        if token_column not in frame:
            continue
        working = frame.copy()
        working["lookup_key"] = working[token_column].map(_token_key)
        for key, group in working.loc[working["lookup_key"].ne("")].groupby(
            "lookup_key", sort=False
        ):
            rows.append(
                {
                    "lookup_key": key,
                    "language": _mode(group.get("source_language", pd.Series(dtype=str))),
                    "entity_type": _mode(group.get("entity_type", pd.Series(dtype=str))),
                    "protected": _mode(group.get("protected", pd.Series(dtype=str))),
                    "pos": _mode(group.get("pos", pd.Series(dtype=str))),
                }
            )
    if not rows:
        return {}
    lookup_frame = pd.DataFrame(rows)
    result: dict[str, dict[str, str]] = {}
    for key, group in lookup_frame.groupby("lookup_key", sort=False):
        result[str(key)] = {
            "language": _mode(group["language"]),
            "entity_type": _mode(group["entity_type"]),
            "protected": _mode(group["protected"]),
            "pos": _mode(group["pos"]),
        }
    return result


def _selected_checkpoint() -> Path:
    selection_path = GATE_RESULTS_DIR / "selections" / "m1.json"
    if not selection_path.is_file():
        raise FileNotFoundError("The completed hybrid M1 selection is unavailable")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    checkpoint = Path(str(selection["selected_checkpoint"]))
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)
    return checkpoint


def _confusion(
    frame: pd.DataFrame,
    *,
    gold_column: str,
    prediction_column: str,
    labels: Sequence[str],
) -> pd.DataFrame:
    table = pd.crosstab(frame[gold_column], frame[prediction_column], dropna=False)
    return table.reindex(index=labels, columns=labels, fill_value=0)


def _class_metrics(
    frame: pd.DataFrame,
    *,
    gold_column: str,
    prediction_column: str,
    labels: Sequence[str],
) -> list[dict[str, object]]:
    rows = []
    for label in labels:
        gold = frame[gold_column].eq(label)
        predicted = frame[prediction_column].eq(label)
        true_positive = int((gold & predicted).sum())
        gold_count = int(gold.sum())
        predicted_count = int(predicted.sum())
        rows.append(
            {
                "label": label,
                "gold_count": gold_count,
                "predicted_count": predicted_count,
                "true_positive": true_positive,
                "precision": (
                    true_positive / predicted_count if predicted_count else 0.0
                ),
                "recall": true_positive / gold_count if gold_count else 0.0,
            }
        )
    return rows


def _confidence_summary(
    frame: pd.DataFrame, correct_column: str, confidence_column: str
) -> dict[str, float | int]:
    correct = frame.loc[frame[correct_column]]
    incorrect = frame.loc[~frame[correct_column]]
    return {
        "correct_count": len(correct),
        "incorrect_count": len(incorrect),
        "mean_confidence_correct": (
            float(correct[confidence_column].mean()) if len(correct) else 0.0
        ),
        "mean_confidence_incorrect": (
            float(incorrect[confidence_column].mean()) if len(incorrect) else 0.0
        ),
    }


def run_head_diagnostics(
    gold_dev: pd.DataFrame,
    *,
    checkpoint: Path | str | None = None,
    batch_size: int = 16,
) -> dict[str, object]:
    """Export head predictions and confusion summaries on Gold-dev only."""
    import torch
    from transformers import AutoTokenizer

    checkpoint = Path(checkpoint) if checkpoint is not None else _selected_checkpoint()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = TntHybridTransformerForConditionalGeneration.from_pretrained(checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    linguistic_lookup = _linguistic_lookup()

    operation_rows: list[dict[str, object]] = []
    segment_rows: list[dict[str, object]] = []
    boundary_rows: list[dict[str, object]] = []
    word_rows: list[dict[str, object]] = []

    for start in range(0, len(gold_dev), batch_size):
        batch = gold_dev.iloc[start : start + batch_size]
        encoded = tokenizer(
            batch["source_text"].tolist(),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=model.config.max_position_embeddings,
        ).to(device)
        with torch.inference_mode():
            hidden = model._encode(encoded["input_ids"], encoded.get("attention_mask"))
            operation_probabilities = model.operation_projection(hidden).softmax(dim=-1)
            operation_confidence, operation_predictions = operation_probabilities.max(
                dim=-1
            )
            segment_probabilities = model.segment_projection(hidden).view(
                encoded["input_ids"].shape[0],
                encoded["input_ids"].shape[1],
                model.config.output_slots_per_source,
                model.config.vocab_size,
            ).softmax(dim=-1)
            segment_confidence, segment_predictions = segment_probabilities.max(dim=-1)
            boundary_probabilities = model.boundary_projection(hidden).softmax(dim=-1)
            boundary_confidence, boundary_predictions = boundary_probabilities.max(
                dim=-1
            )
            word_ids = model._derive_word_ids(encoded["input_ids"])
            word_probabilities = model.word_projection(
                model._pool_words(hidden, word_ids)
            ).softmax(dim=-1)

        input_rows = encoded["input_ids"].detach().cpu().tolist()
        attention_rows = encoded["attention_mask"].detach().cpu().tolist()
        operation_probability_rows = operation_probabilities.detach().cpu().tolist()
        operation_prediction_rows = operation_predictions.detach().cpu().tolist()
        operation_confidence_rows = operation_confidence.detach().cpu().tolist()
        segment_prediction_rows = segment_predictions.detach().cpu().tolist()
        segment_confidence_rows = segment_confidence.detach().cpu().tolist()
        boundary_prediction_rows = boundary_predictions.detach().cpu().tolist()
        boundary_confidence_rows = boundary_confidence.detach().cpu().tolist()
        word_id_rows = word_ids.detach().cpu().tolist()
        word_probability_rows = word_probabilities.detach().cpu().tolist()

        for local_index, (_, record) in enumerate(batch.iterrows()):
            source_length = int(sum(attention_rows[local_index]))
            source_ids = input_rows[local_index][:source_length]
            target_ids = tokenizer(
                text_target=str(record["target_text"]),
                truncation=True,
                max_length=model.config.max_position_embeddings,
            )["input_ids"]
            annotations = hybrid_tnt_annotations(
                source_ids,
                target_ids,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                space_token_id=model.config.space_token_id,
                output_slots_per_source=model.config.output_slots_per_source,
            )
            operation_gold, segment_gold, boundary_gold, gold_word_ids, word_gold = annotations
            source_words = str(record["source_text"]).split()

            for position in range(source_length):
                token_id = int(source_ids[position])
                token = tokenizer.convert_ids_to_tokens(token_id)
                raw_operation = int(operation_prediction_rows[local_index][position])
                word_id = int(word_id_rows[local_index][position])
                keep_probability = (
                    float(word_probability_rows[local_index][word_id][WORD_KEEP])
                    if 0 <= word_id < len(word_probability_rows[local_index])
                    else 0.0
                )
                keep_gate = bool(
                    word_id >= 0
                    and keep_probability >= model.config.word_keep_threshold
                )
                effective_operation = 0 if keep_gate else raw_operation
                gold_operation = int(operation_gold[position])
                operation_rows.append(
                    {
                        "source_id": record["source_id"],
                        "character_position": position,
                        "character": token,
                        "word_id": word_id,
                        "gold_operation": OPERATION_NAMES[gold_operation],
                        "predicted_operation_raw": OPERATION_NAMES[raw_operation],
                        "predicted_operation_effective": OPERATION_NAMES[
                            effective_operation
                        ],
                        "prediction_confidence": float(
                            operation_confidence_rows[local_index][position]
                        ),
                        "word_keep_probability": keep_probability,
                        "word_keep_gate_applied": keep_gate,
                    }
                )
                for slot, gold_token_id in enumerate(segment_gold[position]):
                    if int(gold_token_id) == -100:
                        continue
                    predicted_token_id = int(
                        segment_prediction_rows[local_index][position][slot]
                    )
                    segment_rows.append(
                        {
                            "source_id": record["source_id"],
                            "character_position": position,
                            "slot": slot,
                            "gold_operation": OPERATION_NAMES[gold_operation],
                            "target_kind": (
                                "terminator"
                                if int(gold_token_id) == tokenizer.pad_token_id
                                else "content"
                            ),
                            "gold_token_id": int(gold_token_id),
                            "gold_token": tokenizer.convert_ids_to_tokens(
                                int(gold_token_id)
                            ),
                            "predicted_token_id": predicted_token_id,
                            "predicted_token": tokenizer.convert_ids_to_tokens(
                                predicted_token_id
                            ),
                            "prediction_confidence": float(
                                segment_confidence_rows[local_index][position][slot]
                            ),
                            "correct": predicted_token_id == int(gold_token_id),
                        }
                    )

                predicted_boundary = int(
                    boundary_prediction_rows[local_index][position]
                )
                boundary_is_confident = bool(
                    boundary_confidence_rows[local_index][position]
                    >= model.config.boundary_confidence_threshold
                )
                accepted_action = "none"
                if boundary_is_confident:
                    if token_id == model.config.space_token_id and predicted_boundary in {
                        BOUNDARY_KEEP,
                        BOUNDARY_DELETE,
                    }:
                        accepted_action = BOUNDARY_ACTIONS[predicted_boundary]
                    elif (
                        token_id != model.config.space_token_id
                        and predicted_boundary == BOUNDARY_INSERT_BEFORE
                    ):
                        accepted_action = BOUNDARY_ACTIONS[predicted_boundary]
                gold_boundary = int(boundary_gold[position])
                if gold_boundary >= 0:
                    boundary_rows.append(
                        {
                            "source_id": record["source_id"],
                            "character_position": position,
                            "character": token,
                            "gold_boundary": BOUNDARY_ACTIONS[gold_boundary],
                            "predicted_boundary": BOUNDARY_ACTIONS[
                                predicted_boundary
                            ],
                            "accepted_boundary_action": accepted_action,
                            "prediction_confidence": float(
                                boundary_confidence_rows[local_index][position]
                            ),
                        }
                    )

            for word_id, gold_label in enumerate(word_gold):
                probabilities = word_probability_rows[local_index][word_id]
                prediction = int(max(range(len(probabilities)), key=probabilities.__getitem__))
                source_word = (
                    source_words[word_id] if word_id < len(source_words) else ""
                )
                lookup = linguistic_lookup.get(_token_key(source_word), {})
                keep_probability = float(probabilities[WORD_KEEP])
                positions = [
                    index
                    for index, value in enumerate(gold_word_ids)
                    if value == word_id
                ]
                raw_edit_positions = sum(
                    int(operation_prediction_rows[local_index][position] != 0)
                    for position in positions
                )
                gate_applied = keep_probability >= model.config.word_keep_threshold
                word_rows.append(
                    {
                        "source_id": record["source_id"],
                        "word_id": word_id,
                        "source_word": source_word,
                        "gold_word_action": WORD_ACTIONS[int(gold_label)],
                        "predicted_word_action": WORD_ACTIONS[prediction],
                        "keep_probability": keep_probability,
                        "edit_probability": float(probabilities[1]),
                        "keep_gate_applied": gate_applied,
                        "raw_character_edit_positions": raw_edit_positions,
                        "effective_character_edit_positions": (
                            0 if gate_applied else raw_edit_positions
                        ),
                        "lookup_language": lookup.get("language", "unknown"),
                        "lookup_entity_type": lookup.get(
                            "entity_type", "unknown"
                        ),
                        "lookup_protected": lookup.get("protected", "unknown"),
                        "lookup_pos": lookup.get("pos", "unknown"),
                    }
                )

    operations = pd.DataFrame(operation_rows)
    segments = pd.DataFrame(segment_rows)
    boundaries = pd.DataFrame(boundary_rows)
    words = pd.DataFrame(word_rows)
    operations["raw_correct"] = operations["gold_operation"].eq(
        operations["predicted_operation_raw"]
    )
    operations["effective_correct"] = operations["gold_operation"].eq(
        operations["predicted_operation_effective"]
    )
    boundaries["correct"] = boundaries["gold_boundary"].eq(
        boundaries["predicted_boundary"]
    )
    words["correct"] = words["gold_word_action"].eq(
        words["predicted_word_action"]
    )

    DIAGNOSTIC_DIR.mkdir(parents=True, exist_ok=True)
    operation_path = DIAGNOSTIC_DIR / "operation_predictions.csv"
    segment_path = DIAGNOSTIC_DIR / "segment_predictions.csv"
    boundary_path = DIAGNOSTIC_DIR / "boundary_predictions.csv"
    word_path = DIAGNOSTIC_DIR / "word_predictions.csv"
    operations.to_csv(operation_path, index=False, encoding="utf-8")
    segments.to_csv(segment_path, index=False, encoding="utf-8")
    boundaries.to_csv(boundary_path, index=False, encoding="utf-8")
    words.to_csv(word_path, index=False, encoding="utf-8")

    raw_operation_confusion = _confusion(
        operations,
        gold_column="gold_operation",
        prediction_column="predicted_operation_raw",
        labels=OPERATION_NAMES,
    )
    effective_operation_confusion = _confusion(
        operations,
        gold_column="gold_operation",
        prediction_column="predicted_operation_effective",
        labels=OPERATION_NAMES,
    )
    boundary_confusion = _confusion(
        boundaries,
        gold_column="gold_boundary",
        prediction_column="predicted_boundary",
        labels=BOUNDARY_ACTIONS,
    )
    word_confusion = _confusion(
        words,
        gold_column="gold_word_action",
        prediction_column="predicted_word_action",
        labels=WORD_ACTIONS,
    )
    raw_operation_confusion.to_csv(
        DIAGNOSTIC_DIR / "operation_confusion_raw.csv", encoding="utf-8"
    )
    effective_operation_confusion.to_csv(
        DIAGNOSTIC_DIR / "operation_confusion_effective.csv", encoding="utf-8"
    )
    boundary_confusion.to_csv(
        DIAGNOSTIC_DIR / "boundary_confusion.csv", encoding="utf-8"
    )
    word_confusion.to_csv(DIAGNOSTIC_DIR / "word_confusion.csv", encoding="utf-8")

    known_french = words["lookup_language"].eq("fr")
    known_entities = ~words["lookup_entity_type"].str.casefold().isin(
        {"", "unknown", "none", "o"}
    )
    known_protected = words["lookup_protected"].str.casefold().isin(
        {"yes", "true", "1", "oui"}
    )

    def subset_word_summary(mask: pd.Series) -> dict[str, float | int]:
        subset = words.loc[mask]
        return {
            "rows": len(subset),
            "keep_gate_rate": (
                float(subset["keep_gate_applied"].mean()) if len(subset) else 0.0
            ),
            "raw_character_edit_positions": int(
                subset["raw_character_edit_positions"].sum()
            ),
            "effective_character_edit_positions": int(
                subset["effective_character_edit_positions"].sum()
            ),
        }

    content_segments = segments.loc[segments["target_kind"].eq("content")]
    terminator_segments = segments.loc[segments["target_kind"].eq("terminator")]
    segment_groups = segments.groupby(
        ["source_id", "character_position"], sort=False
    )["correct"].all()

    summary: dict[str, object] = {
        "checkpoint": str(checkpoint.resolve()),
        "split": "Gold-dev",
        "rows": len(gold_dev),
        "thresholds": {
            "word_keep": model.config.word_keep_threshold,
            "boundary": model.config.boundary_confidence_threshold,
        },
        "operation": {
            "positions": len(operations),
            "raw_accuracy": float(operations["raw_correct"].mean()),
            "effective_accuracy_after_word_gate": float(
                operations["effective_correct"].mean()
            ),
            "word_gate_overrides": int(
                operations["word_keep_gate_applied"].sum()
            ),
            "raw_per_class": _class_metrics(
                operations,
                gold_column="gold_operation",
                prediction_column="predicted_operation_raw",
                labels=OPERATION_NAMES,
            ),
            "effective_per_class": _class_metrics(
                operations,
                gold_column="gold_operation",
                prediction_column="predicted_operation_effective",
                labels=OPERATION_NAMES,
            ),
            "confidence": _confidence_summary(
                operations, "raw_correct", "prediction_confidence"
            ),
        },
        "segment": {
            "supervised_slots": len(segments),
            "overall_accuracy": float(segments["correct"].mean()),
            "content_slots": len(content_segments),
            "content_accuracy": float(content_segments["correct"].mean()),
            "terminator_slots": len(terminator_segments),
            "terminator_accuracy": float(terminator_segments["correct"].mean()),
            "exact_supervised_segments": int(segment_groups.sum()),
            "supervised_segments": len(segment_groups),
            "exact_segment_rate": float(segment_groups.mean()),
            "content_confidence": _confidence_summary(
                content_segments, "correct", "prediction_confidence"
            ),
        },
        "boundary": {
            "positions": len(boundaries),
            "raw_accuracy": float(boundaries["correct"].mean()),
            "gold_actions": boundaries["gold_boundary"].value_counts().to_dict(),
            "predicted_actions": boundaries["predicted_boundary"]
            .value_counts()
            .to_dict(),
            "accepted_actions": boundaries["accepted_boundary_action"]
            .value_counts()
            .to_dict(),
            "per_class": _class_metrics(
                boundaries,
                gold_column="gold_boundary",
                prediction_column="predicted_boundary",
                labels=BOUNDARY_ACTIONS,
            ),
            "confidence": _confidence_summary(
                boundaries, "correct", "prediction_confidence"
            ),
        },
        "word": {
            "words": len(words),
            "accuracy": float(words["correct"].mean()),
            "gold_actions": words["gold_word_action"].value_counts().to_dict(),
            "predicted_actions": words["predicted_word_action"]
            .value_counts()
            .to_dict(),
            "keep_gate_applied_words": int(words["keep_gate_applied"].sum()),
            "keep_gate_rate": float(words["keep_gate_applied"].mean()),
            "gold_keep_blocked": int(
                (
                    words["gold_word_action"].eq("keep")
                    & words["keep_gate_applied"]
                ).sum()
            ),
            "gold_edit_incorrectly_blocked": int(
                (
                    words["gold_word_action"].eq("edit")
                    & words["keep_gate_applied"]
                ).sum()
            ),
            "per_class": _class_metrics(
                words,
                gold_column="gold_word_action",
                prediction_column="predicted_word_action",
                labels=WORD_ACTIONS,
            ),
            "known_french": subset_word_summary(known_french),
            "known_entities": subset_word_summary(known_entities),
            "known_protected": subset_word_summary(known_protected),
        },
        "artifacts": {
            "operation_predictions": str(operation_path.resolve()),
            "segment_predictions": str(segment_path.resolve()),
            "boundary_predictions": str(boundary_path.resolve()),
            "word_predictions": str(word_path.resolve()),
        },
        "lookup_note": (
            "Language/entity/protection flags are lookup matches from reviewed "
            "Gold-train annotations; unknown Gold-dev tokens remain unknown."
        ),
    }
    summary_path = DIAGNOSTIC_DIR / "head_diagnostics_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    summary["summary_path"] = str(summary_path.resolve())
    return summary
