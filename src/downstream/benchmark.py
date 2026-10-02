"""Frozen REAL/NOISY/NORMALIZED intent-classification benchmark."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from .injongo import INJONGO_INTENTS
from .laya_model import LayaIntentClassifier
from .metrics import (
    classification_metrics,
    condition_deltas,
    paired_analysis,
)
from .noise import build_noisy_records
from .tnt_normalizer import TntV3Normalizer


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prediction_map(rows: Sequence[Mapping]) -> dict[str, Mapping]:
    mapped = {str(row["row_id"]): row for row in rows}
    if len(mapped) != len(rows):
        raise ValueError("Duplicate prediction row_id")
    return mapped


def run_frozen_benchmark(
    test_rows: Sequence[Mapping],
    *,
    laya_checkpoint: str | Path,
    tnt_checkpoint: str | Path,
    output_dir: str | Path = "results/downstream/laya_normalization_v1",
    seed: int = 2026,
    laya_batch_size: int = 16,
    tnt_batch_size: int = 32,
    device: str | None = None,
) -> dict:
    """Evaluate one frozen classifier on all three paired conditions."""

    if not test_rows or {str(row["split"]) for row in test_rows} != {"test"}:
        raise ValueError("run_frozen_benchmark accepts official test rows only")
    if len({str(row["row_id"]) for row in test_rows}) != len(test_rows):
        raise ValueError("Duplicate test row_id")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    laya_checkpoint = Path(laya_checkpoint).resolve()
    tnt_checkpoint = Path(tnt_checkpoint).resolve()
    noisy_rows = build_noisy_records(test_rows, seed=seed)

    normalizer = TntV3Normalizer(
        tnt_checkpoint, device=device, batch_size=tnt_batch_size
    )
    normalized_texts = normalizer.normalize(
        [str(row["noisy_text"]) for row in noisy_rows]
    )
    normalized_rows = [
        {
            "example_id": row["example_id"],
            "row_id": row["row_id"],
            "text": normalized_text,
            "intent": row["intent"],
            "split": "test",
        }
        for row, normalized_text in zip(noisy_rows, normalized_texts)
    ]
    real_rows = [
        {
            "example_id": str(row["example_id"]),
            "row_id": str(row["row_id"]),
            "text": str(row["text"]),
            "intent": str(row["intent"]),
            "split": "test",
        }
        for row in test_rows
    ]
    noisy_for_model = [
        {
            "example_id": row["example_id"],
            "row_id": row["row_id"],
            "text": row["noisy_text"],
            "intent": row["intent"],
            "split": "test",
        }
        for row in noisy_rows
    ]

    classifier = LayaIntentClassifier(
        laya_checkpoint, device=device, batch_size=laya_batch_size
    )
    condition_rows = {
        "REAL": real_rows,
        "NOISY": noisy_for_model,
        "NORMALIZED": normalized_rows,
    }
    predictions = {
        condition: classifier.predict(rows)
        for condition, rows in condition_rows.items()
    }

    metrics: dict[str, dict] = {}
    for condition, prediction_rows in predictions.items():
        gold = [str(row["gold_intent"]) for row in prediction_rows]
        predicted = [str(row["prediction"]) for row in prediction_rows]
        summary, per_class, confusion = classification_metrics(
            gold, predicted, INJONGO_INTENTS
        )
        metrics[condition] = summary
        pd.DataFrame(per_class).to_csv(
            output_dir / f"per_class_{condition.lower()}.csv", index=False
        )
        confusion_frame = pd.DataFrame(
            confusion, index=INJONGO_INTENTS, columns=INJONGO_INTENTS
        )
        confusion_frame.index.name = "gold_intent"
        confusion_frame.to_csv(output_dir / f"confusion_{condition.lower()}.csv")
        pd.DataFrame(prediction_rows).to_csv(
            output_dir / f"predictions_{condition.lower()}.csv", index=False
        )

    metrics["deltas"] = {
        metric_name: condition_deltas(
            metrics["REAL"][metric_name],
            metrics["NOISY"][metric_name],
            metrics["NORMALIZED"][metric_name],
        )
        for metric_name in ("accuracy", "macro_f1")
    }

    real_predictions = _prediction_map(predictions["REAL"])
    noisy_predictions = _prediction_map(predictions["NOISY"])
    normalized_predictions = _prediction_map(predictions["NORMALIZED"])
    paired_input: list[dict] = []
    for noisy, normalized_text in zip(noisy_rows, normalized_texts):
        example_id = str(noisy["example_id"])
        row_id = str(noisy["row_id"])
        paired_input.append(
            {
                "example_id": example_id,
                "row_id": row_id,
                "real_text": noisy["real_text"],
                "noisy_text": noisy["noisy_text"],
                "normalized_text": normalized_text,
                "gold_intent": noisy["intent"],
                "real_prediction": real_predictions[row_id]["prediction"],
                "noisy_prediction": noisy_predictions[row_id]["prediction"],
                "normalized_prediction": normalized_predictions[row_id]["prediction"],
            }
        )
    paired_rows, paired_rates = paired_analysis(paired_input)
    pd.DataFrame(paired_rows).to_csv(output_dir / "paired_analysis.csv", index=False)

    noisy_export = pd.DataFrame(noisy_rows)
    noisy_export["noise_operations"] = noisy_export["noise_operations"].map(
        lambda value: json.dumps(value, ensure_ascii=False, sort_keys=True)
    )
    noisy_export["normalized_text"] = normalized_texts
    noisy_export.to_csv(output_dir / "paired_text_conditions.csv", index=False)

    metrics["paired"] = paired_rates
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "experiment": "laya_downstream_normalization_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "classifier_training_condition": "REAL train only",
            "evaluation_split": "official InjongoIntent/wol test",
            "conditions": ["REAL", "NOISY", "NORMALIZED"],
            "same_frozen_classifier": True,
            "noise_description": "synthetic noisy Wolof generated by existing M1 rules",
            "seed": seed,
        },
        "rows": len(test_rows),
        "laya_checkpoint": str(laya_checkpoint),
        "laya_model_sha256": _sha256(laya_checkpoint / "model.safetensors"),
        "tnt_checkpoint": str(tnt_checkpoint),
        "tnt_model_sha256": _sha256(tnt_checkpoint / "model.safetensors"),
        "outputs": sorted(path.name for path in output_dir.iterdir() if path.is_file()),
    }
    (output_dir / "benchmark_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {"metrics": metrics, "manifest": manifest}
