"""Run frozen downstream clean, severity, generalization, and error analyses."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from .injongo import INJONGO_INTENTS, load_split
from .laya_model import LayaIntentClassifier
from .metrics import classification_metrics, condition_deltas, paired_analysis
from .noise_v2 import (
    M1_SEVERITIES,
    build_m1_severity_records,
    build_m5_records,
    build_m6_records,
    method_applicability,
)
from .text_error_analysis import (
    build_detailed_error_rows,
    corpus_restoration_metrics,
    grouped_restoration_statistics,
    learn_intent_cues,
    pair_text_metrics,
    restoration_confidence_correlations,
)
from .tnt_normalizer import TntV3Normalizer
from .v2_protocol import (
    INJONGO_TEST,
    LAYA_CHECKPOINT,
    TNT_CHECKPOINT,
    V1_RESULTS_DIR,
    V2_RESULTS_DIR,
    assert_v1_real_baseline,
    current_git_commit,
    sha256_file,
    validate_frozen_artifacts,
)


def _write_json(path: Path, payload: Mapping | Sequence) -> None:
    def clean(value):
        if isinstance(value, Mapping):
            return {str(key): clean(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(item) for item in value]
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(clean(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _model_rows(records: Sequence[Mapping], text_field: str) -> list[dict]:
    return [
        {
            "example_id": str(row["example_id"]),
            "row_id": str(row["row_id"]),
            "text": str(row[text_field]),
            "intent": str(row["intent"]),
            "split": "test",
        }
        for row in records
    ]


def _prediction_map(predictions: Sequence[Mapping]) -> dict[str, Mapping]:
    mapped = {str(row["row_id"]): row for row in predictions}
    if len(mapped) != len(predictions):
        raise RuntimeError("Duplicate row_id in predictions")
    return mapped


def _validate_pairing(reference: Sequence[Mapping], candidate: Sequence[Mapping]) -> None:
    expected = [(str(row["row_id"]), str(row["intent"])) for row in reference]
    observed = [(str(row["row_id"]), str(row["intent"])) for row in candidate]
    if observed != expected:
        raise RuntimeError("Condition changed row ordering, row IDs, or gold intents")


def _score_condition(
    name: str,
    predictions: list[dict],
    output_dir: Path,
) -> tuple[dict, pd.DataFrame]:
    gold = [str(row["gold_intent"]) for row in predictions]
    predicted = [str(row["prediction"]) for row in predictions]
    metrics, per_class, confusion = classification_metrics(
        gold, predicted, INJONGO_INTENTS
    )
    pd.DataFrame(predictions).to_csv(
        output_dir / "predictions" / f"{name.lower()}.csv", index=False
    )
    per_class_frame = pd.DataFrame(per_class)
    per_class_frame.to_csv(
        output_dir / "per_class" / f"{name.lower()}.csv", index=False
    )
    confusion_frame = pd.DataFrame(
        confusion, index=INJONGO_INTENTS, columns=INJONGO_INTENTS
    )
    confusion_frame.index.name = "gold_intent"
    confusion_frame.to_csv(
        output_dir / "confusion" / f"{name.lower()}.csv"
    )
    return metrics, per_class_frame


def _paired_rows(
    source_records: Sequence[Mapping],
    normalized_texts: Sequence[str],
    predictions: Mapping[str, Sequence[Mapping]],
    noisy_condition: str,
    normalized_condition: str,
    real_condition: str = "REAL",
) -> tuple[list[dict], dict]:
    real_map = _prediction_map(predictions[real_condition])
    noisy_map = _prediction_map(predictions[noisy_condition])
    normalized_map = _prediction_map(predictions[normalized_condition])
    rows = []
    for source, normalized_text in zip(source_records, normalized_texts):
        row_id = str(source["row_id"])
        rows.append(
            {
                "example_id": str(source["example_id"]),
                "row_id": row_id,
                "real_text": str(source["real_text"]),
                "noisy_text": str(source["noisy_text"]),
                "normalized_text": str(normalized_text),
                "gold_intent": str(source["intent"]),
                "real_prediction": str(real_map[row_id]["prediction"]),
                "noisy_prediction": str(noisy_map[row_id]["prediction"]),
                "normalized_prediction": str(normalized_map[row_id]["prediction"]),
                "real_gold_confidence": float(real_map[row_id]["gold_confidence"]),
                "noisy_gold_confidence": float(noisy_map[row_id]["gold_confidence"]),
                "normalized_gold_confidence": float(normalized_map[row_id]["gold_confidence"]),
            }
        )
    return paired_analysis(rows)


def _noise_statistics(records: Sequence[Mapping]) -> dict:
    real = [str(row["real_text"]) for row in records]
    noisy = [str(row["noisy_text"]) for row in records]
    counts = [int(row["operation_count"]) for row in records]
    metrics = corpus_restoration_metrics(real, noisy)
    metrics.update(
        {
            "mean_operations": float(statistics.mean(counts)),
            "median_operations": float(statistics.median(counts)),
            "total_operations": int(sum(counts)),
            "changed_rows": int(sum(bool(row["changed"]) for row in records)),
            "changed_rate": float(sum(bool(row["changed"]) for row in records) / len(records)),
            "modified_token_count": int(sum(pair_text_metrics(r, n)["modified_token_count"] for r, n in zip(real, noisy))),
        }
    )
    return metrics


def _per_intent_comparison(
    noisy_per_class: pd.DataFrame,
    normalized_per_class: pd.DataFrame,
    paired_rows: pd.DataFrame,
    *,
    method: str,
) -> pd.DataFrame:
    noisy = noisy_per_class.set_index("intent")
    normalized = normalized_per_class.set_index("intent")
    categories = pd.crosstab(paired_rows["gold_intent"], paired_rows["category"])
    records = []
    for intent in INJONGO_INTENTS:
        record = {
            "method": method,
            "intent": intent,
            "support": int(noisy.loc[intent, "support"]),
            "noisy_precision": float(noisy.loc[intent, "precision"]),
            "noisy_recall": float(noisy.loc[intent, "recall"]),
            "noisy_f1": float(noisy.loc[intent, "f1"]),
            "normalized_precision": float(normalized.loc[intent, "precision"]),
            "normalized_recall": float(normalized.loc[intent, "recall"]),
            "normalized_f1": float(normalized.loc[intent, "f1"]),
            "f1_gain": float(normalized.loc[intent, "f1"] - noisy.loc[intent, "f1"]),
        }
        for category in ("HELPED", "HARMED", "UNCHANGED_CORRECT", "UNRESOLVED"):
            record[category.lower()] = int(
                categories.loc[intent, category]
                if intent in categories.index and category in categories.columns
                else 0
            )
        records.append(record)
    return pd.DataFrame(records)


def _clean_analysis(
    test_rows: Sequence[Mapping],
    tnt_real: Sequence[str],
    metrics: Mapping[str, Mapping],
    predictions: Mapping[str, Sequence[Mapping]],
    per_class: Mapping[str, pd.DataFrame],
    output_dir: Path,
) -> dict:
    real_map = _prediction_map(predictions["REAL"])
    normalized_map = _prediction_map(predictions["TNT_REAL"])
    paired = []
    for row, normalized_text in zip(test_rows, tnt_real):
        row_id = str(row["row_id"])
        real_correct = real_map[row_id]["prediction"] == row["intent"]
        normalized_correct = normalized_map[row_id]["prediction"] == row["intent"]
        if real_correct and normalized_correct:
            category = "UNCHANGED_CORRECT"
        elif real_correct and not normalized_correct:
            category = "HARMED_CLEAN"
        elif not real_correct and normalized_correct:
            category = "HELPED_CLEAN"
        else:
            category = "UNCHANGED_WRONG"
        text_metrics = pair_text_metrics(str(row["text"]), str(normalized_text))
        paired.append(
            {
                "example_id": row["example_id"],
                "row_id": row_id,
                "intent": row["intent"],
                "real_text": row["text"],
                "tnt_real_text": normalized_text,
                "real_prediction": real_map[row_id]["prediction"],
                "tnt_real_prediction": normalized_map[row_id]["prediction"],
                "real_gold_confidence": real_map[row_id]["gold_confidence"],
                "tnt_real_gold_confidence": normalized_map[row_id]["gold_confidence"],
                "category": category,
                **text_metrics,
            }
        )
    frame = pd.DataFrame(paired)
    frame.to_csv(output_dir / "clean" / "clean_paired_analysis.csv", index=False)
    frame.sort_values(
        ["char_edits", "word_edits"], ascending=False
    ).head(100).to_csv(
        output_dir / "clean" / "most_modified_clean_sentences.csv", index=False
    )
    counts = frame["category"].value_counts().to_dict()
    category_by_intent = pd.crosstab(frame["intent"], frame["category"])
    real_class = per_class["REAL"].set_index("intent")
    tnt_class = per_class["TNT_REAL"].set_index("intent")
    clean_by_intent = []
    for intent in INJONGO_INTENTS:
        record = {
            "intent": intent,
            "real_precision": float(real_class.loc[intent, "precision"]),
            "real_recall": float(real_class.loc[intent, "recall"]),
            "real_f1": float(real_class.loc[intent, "f1"]),
            "tnt_real_precision": float(tnt_class.loc[intent, "precision"]),
            "tnt_real_recall": float(tnt_class.loc[intent, "recall"]),
            "tnt_real_f1": float(tnt_class.loc[intent, "f1"]),
            "f1_delta": float(tnt_class.loc[intent, "f1"] - real_class.loc[intent, "f1"]),
        }
        for category in ("UNCHANGED_CORRECT", "HARMED_CLEAN", "HELPED_CLEAN", "UNCHANGED_WRONG"):
            record[category.lower()] = int(
                category_by_intent.loc[intent, category]
                if intent in category_by_intent.index and category in category_by_intent.columns
                else 0
            )
        clean_by_intent.append(record)
    pd.DataFrame(clean_by_intent).to_csv(
        output_dir / "clean" / "clean_by_intent.csv", index=False
    )
    rows = len(frame)
    text_summary = corpus_restoration_metrics(
        frame["real_text"].tolist(), frame["tnt_real_text"].tolist()
    )
    result = {
        "real": metrics["REAL"],
        "tnt_real": metrics["TNT_REAL"],
        "clean_accuracy_delta": metrics["TNT_REAL"]["accuracy"] - metrics["REAL"]["accuracy"],
        "clean_macro_f1_delta": metrics["TNT_REAL"]["macro_f1"] - metrics["REAL"]["macro_f1"],
        "clean_harm_rate": counts.get("HARMED_CLEAN", 0) / rows,
        "clean_help_rate": counts.get("HELPED_CLEAN", 0) / rows,
        "net_clean_gain": counts.get("HELPED_CLEAN", 0) - counts.get("HARMED_CLEAN", 0),
        "paired_counts": counts,
        "text_preservation": text_summary,
    }
    _write_json(output_dir / "clean" / "clean_summary.json", result)
    return result


def _plot_severity(severity: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    order = ["light", "medium", "heavy"]
    frame = severity.set_index("severity").loc[order]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(frame["mean_operations"], frame["noisy_accuracy"], marker="o", label="NOISY accuracy")
    axes[0].plot(frame["mean_operations"], frame["normalized_accuracy"], marker="o", label="NORMALIZED accuracy")
    axes[0].axhline(float(frame["real_accuracy"].iloc[0]), color="black", linestyle="--", label="REAL accuracy")
    axes[0].set(xlabel="Mean M1 operations / sentence", ylabel="Accuracy", title="Intent performance by M1 severity")
    axes[0].legend()
    axes[1].plot(frame["mean_operations"], frame["accuracy_recovery_rate"], marker="o", label="Accuracy recovery")
    axes[1].plot(frame["mean_operations"], frame["macro_f1_recovery_rate"], marker="o", label="Macro-F1 recovery")
    axes[1].set(xlabel="Mean M1 operations / sentence", ylabel="Recovery rate", title="Normalization recovery by severity")
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output_dir / "figures" / "severity_curves.png", dpi=180)
    plt.close(figure)


def run_v2(
    repository_root: str | Path = ".",
    *,
    output_dir: str | Path = V2_RESULTS_DIR,
    seed: int = 2026,
    device: str | None = None,
) -> dict:
    root = Path(repository_root).resolve()
    output_dir = Path(output_dir)
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    for subdir in ("conditions", "predictions", "per_class", "confusion", "clean", "severity", "cross_method", "error_analysis", "figures"):
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    print("[V2] Validating frozen artifacts...", flush=True)
    frozen = validate_frozen_artifacts(root)
    train_rows = load_split(root / "data/external/injongo_intent_wol/train.jsonl", "train")
    test_rows = load_split(root / INJONGO_TEST, "test")
    real_records = [
        {
            "example_id": row["example_id"], "row_id": row["row_id"],
            "intent": row["intent"], "split": "test", "real_text": row["text"],
            "text": row["text"],
        }
        for row in test_rows
    ]

    print("[V2] Generating controlled and cross-method noise...", flush=True)
    severity_records = {
        severity.name: build_m1_severity_records(test_rows, severity, seed=seed)
        for severity in M1_SEVERITIES
    }
    v1_texts = pd.read_csv(root / V1_RESULTS_DIR / "paired_text_conditions.csv").set_index("row_id")
    heavy_by_id = {row["row_id"]: row["noisy_text"] for row in severity_records["heavy"]}
    if any(str(v1_texts.loc[row_id, "noisy_text"]) != text for row_id, text in heavy_by_id.items()):
        raise RuntimeError("M1-heavy no longer reproduces the exact V1 NOISY text")
    cross_records = {
        "m1": severity_records["heavy"],
        "m5": build_m5_records(test_rows, seed=seed),
        "m6": build_m6_records(test_rows, seed=seed),
    }
    for records in [*severity_records.values(), *cross_records.values()]:
        _validate_pairing(test_rows, records)

    noise_sets: dict[str, Sequence[Mapping]] = {
        "LIGHT": severity_records["light"],
        "MEDIUM": severity_records["medium"],
        "HEAVY": severity_records["heavy"],
        "M5": cross_records["m5"],
        "M6": cross_records["m6"],
    }
    for name, records in noise_sets.items():
        export = pd.DataFrame(records)
        export["noise_operations"] = export["noise_operations"].map(
            lambda value: json.dumps(value, ensure_ascii=False, sort_keys=True)
        )
        export.to_csv(output_dir / "conditions" / f"noisy_{name.lower()}.csv", index=False)

    print("[V2] Running frozen TNT on REAL and each noisy condition...", flush=True)
    normalizer = TntV3Normalizer(root / TNT_CHECKPOINT, device=device, batch_size=32)
    normalized_texts: dict[str, list[str]] = {
        "TNT_REAL": normalizer.normalize([str(row["text"]) for row in test_rows])
    }
    for name, records in noise_sets.items():
        normalized_texts[f"NORMALIZED_{name}"] = normalizer.normalize(
            [str(row["noisy_text"]) for row in records]
        )
    del normalizer
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    condition_rows: dict[str, list[dict]] = {
        "REAL": _model_rows(real_records, "text"),
        "TNT_REAL": [
            {"example_id": row["example_id"], "row_id": row["row_id"], "text": text, "intent": row["intent"], "split": "test"}
            for row, text in zip(test_rows, normalized_texts["TNT_REAL"])
        ],
    }
    for name, records in noise_sets.items():
        condition_rows[f"NOISY_{name}"] = _model_rows(records, "noisy_text")
        condition_rows[f"NORMALIZED_{name}"] = [
            {"example_id": row["example_id"], "row_id": row["row_id"], "text": text, "intent": row["intent"], "split": "test"}
            for row, text in zip(records, normalized_texts[f"NORMALIZED_{name}"])
        ]
    for rows in condition_rows.values():
        _validate_pairing(test_rows, rows)

    print(f"[V2] Running frozen Laya on {len(condition_rows)} conditions...", flush=True)
    classifier = LayaIntentClassifier(root / LAYA_CHECKPOINT, device=device, batch_size=16)
    predictions: dict[str, list[dict]] = {}
    metrics: dict[str, dict] = {}
    per_class: dict[str, pd.DataFrame] = {}
    for index, (name, rows) in enumerate(condition_rows.items(), start=1):
        print(f"[V2] Laya {index}/{len(condition_rows)}: {name}", flush=True)
        predictions[name] = classifier.predict(rows)
        metrics[name], per_class[name] = _score_condition(
            name, predictions[name], output_dir
        )
    assert_v1_real_baseline(metrics["REAL"])
    _write_json(output_dir / "all_condition_metrics.json", metrics)

    print("[V2] Clean preservation analysis...", flush=True)
    clean_summary = _clean_analysis(
        test_rows, normalized_texts["TNT_REAL"], metrics, predictions, per_class, output_dir
    )

    print("[V2] Severity analysis...", flush=True)
    severity_summary_rows = []
    severity_intents = []
    for severity in M1_SEVERITIES:
        key = severity.name.upper()
        paired_rows, paired_rates = _paired_rows(
            severity_records[severity.name],
            normalized_texts[f"NORMALIZED_{key}"],
            predictions,
            f"NOISY_{key}",
            f"NORMALIZED_{key}",
        )
        paired_frame = pd.DataFrame(paired_rows)
        paired_frame.to_csv(output_dir / "severity" / f"paired_{severity.name}.csv", index=False)
        stats = _noise_statistics(severity_records[severity.name])
        accuracy_delta = condition_deltas(metrics["REAL"]["accuracy"], metrics[f"NOISY_{key}"]["accuracy"], metrics[f"NORMALIZED_{key}"]["accuracy"])
        f1_delta = condition_deltas(metrics["REAL"]["macro_f1"], metrics[f"NOISY_{key}"]["macro_f1"], metrics[f"NORMALIZED_{key}"]["macro_f1"])
        severity_summary_rows.append(
            {
                "severity": severity.name,
                "target_mean_operations": severity.target_mean_operations,
                "application_probability": severity.application_probability,
                "mean_operations": stats["mean_operations"],
                "median_operations": stats["median_operations"],
                "changed_rows": stats["changed_rows"],
                "changed_rate": stats["changed_rate"],
                "corruption_cer": stats["cer"],
                "corruption_wer": stats["wer"],
                "corruption_chrf": stats["chrf"],
                "modified_token_count": stats["modified_token_count"],
                "real_accuracy": metrics["REAL"]["accuracy"],
                "noisy_accuracy": metrics[f"NOISY_{key}"]["accuracy"],
                "normalized_accuracy": metrics[f"NORMALIZED_{key}"]["accuracy"],
                "accuracy_noise_drop": accuracy_delta["noise_drop"],
                "accuracy_normalization_gain": accuracy_delta["normalization_gain"],
                "accuracy_recovery_rate": accuracy_delta["recovery_rate"],
                "real_macro_f1": metrics["REAL"]["macro_f1"],
                "noisy_macro_f1": metrics[f"NOISY_{key}"]["macro_f1"],
                "normalized_macro_f1": metrics[f"NORMALIZED_{key}"]["macro_f1"],
                "macro_f1_noise_drop": f1_delta["noise_drop"],
                "macro_f1_normalization_gain": f1_delta["normalization_gain"],
                "macro_f1_recovery_rate": f1_delta["recovery_rate"],
                **{name: value for name, value in paired_rates.items() if name != "counts"},
                **{f"count_{name.lower()}": value for name, value in paired_rates["counts"].items()},
            }
        )
        severity_intents.append(
            _per_intent_comparison(
                per_class[f"NOISY_{key}"], per_class[f"NORMALIZED_{key}"],
                paired_frame, method=f"m1_{severity.name}"
            )
        )
    severity_summary = pd.DataFrame(severity_summary_rows)
    severity_summary.to_csv(output_dir / "severity" / "severity_curve.csv", index=False)
    pd.concat(severity_intents, ignore_index=True).to_csv(
        output_dir / "severity" / "severity_by_intent.csv", index=False
    )
    _plot_severity(severity_summary, output_dir)

    print("[V2] Cross-method analysis...", flush=True)
    applicability = pd.DataFrame(method_applicability())
    applicability.to_csv(output_dir / "cross_method" / "method_applicability.csv", index=False)
    cross_summary_rows = []
    cross_intents = []
    for method, records, key in (
        ("M1", cross_records["m1"], "HEAVY"),
        ("M5", cross_records["m5"], "M5"),
        ("M6", cross_records["m6"], "M6"),
    ):
        paired_rows, paired_rates = _paired_rows(
            records, normalized_texts[f"NORMALIZED_{key}"], predictions,
            f"NOISY_{key}", f"NORMALIZED_{key}"
        )
        paired_frame = pd.DataFrame(paired_rows)
        paired_frame.to_csv(output_dir / "cross_method" / f"paired_{method.lower()}.csv", index=False)
        stats = _noise_statistics(records)
        changed_ids = {str(row["row_id"]) for row in records if bool(row["changed"])}
        noisy_map = _prediction_map(predictions[f"NOISY_{key}"])
        normalized_map = _prediction_map(predictions[f"NORMALIZED_{key}"])
        changed_noisy_correct = []
        changed_normalized_correct = []
        for row in records:
            row_id = str(row["row_id"])
            if row_id not in changed_ids:
                continue
            gold = str(row["intent"])
            changed_noisy_correct.append(noisy_map[row_id]["prediction"] == gold)
            changed_normalized_correct.append(normalized_map[row_id]["prediction"] == gold)
        changed_helped = sum(
            not noisy and normalized
            for noisy, normalized in zip(changed_noisy_correct, changed_normalized_correct)
        )
        changed_harmed = sum(
            noisy and not normalized
            for noisy, normalized in zip(changed_noisy_correct, changed_normalized_correct)
        )
        accuracy_delta = condition_deltas(metrics["REAL"]["accuracy"], metrics[f"NOISY_{key}"]["accuracy"], metrics[f"NORMALIZED_{key}"]["accuracy"])
        f1_delta = condition_deltas(metrics["REAL"]["macro_f1"], metrics[f"NOISY_{key}"]["macro_f1"], metrics[f"NORMALIZED_{key}"]["macro_f1"])
        cross_summary_rows.append(
            {
                "method": method,
                "n_examples": len(records),
                "changed_rows": stats["changed_rows"],
                "mean_operations": stats["mean_operations"],
                "changed_subset_rows": len(changed_noisy_correct),
                "changed_subset_noisy_accuracy": (
                    float(sum(changed_noisy_correct) / len(changed_noisy_correct))
                    if changed_noisy_correct else None
                ),
                "changed_subset_normalized_accuracy": (
                    float(sum(changed_normalized_correct) / len(changed_normalized_correct))
                    if changed_normalized_correct else None
                ),
                "changed_subset_normalization_gain": (
                    float(
                        sum(changed_normalized_correct) / len(changed_normalized_correct)
                        - sum(changed_noisy_correct) / len(changed_noisy_correct)
                    ) if changed_noisy_correct else None
                ),
                "changed_subset_helped": int(changed_helped),
                "changed_subset_harmed": int(changed_harmed),
                "real_accuracy": metrics["REAL"]["accuracy"],
                "noisy_accuracy": metrics[f"NOISY_{key}"]["accuracy"],
                "normalized_accuracy": metrics[f"NORMALIZED_{key}"]["accuracy"],
                "noise_drop": accuracy_delta["noise_drop"],
                "normalization_gain": accuracy_delta["normalization_gain"],
                "recovery_rate": accuracy_delta["recovery_rate"],
                "real_macro_f1": metrics["REAL"]["macro_f1"],
                "noisy_macro_f1": metrics[f"NOISY_{key}"]["macro_f1"],
                "normalized_macro_f1": metrics[f"NORMALIZED_{key}"]["macro_f1"],
                "macro_f1_noise_drop": f1_delta["noise_drop"],
                "macro_f1_normalization_gain": f1_delta["normalization_gain"],
                "macro_f1_recovery_rate": f1_delta["recovery_rate"],
                **paired_rates["counts"],
            }
        )
        cross_intents.append(
            _per_intent_comparison(
                per_class[f"NOISY_{key}"], per_class[f"NORMALIZED_{key}"], paired_frame, method=method
            )
        )
    cross_summary = pd.DataFrame(cross_summary_rows)
    cross_summary.to_csv(output_dir / "cross_method" / "cross_method_summary.csv", index=False)
    pd.concat(cross_intents, ignore_index=True).to_csv(
        output_dir / "cross_method" / "cross_method_by_intent.csv", index=False
    )

    print("[V2] Detailed V1-compatible error analysis...", flush=True)
    heavy_paired, heavy_rates = _paired_rows(
        severity_records["heavy"], normalized_texts["NORMALIZED_HEAVY"],
        predictions, "NOISY_HEAVY", "NORMALIZED_HEAVY"
    )
    heavy_frame = pd.DataFrame(heavy_paired)
    expected_counts = {"HELPED": 110, "HARMED": 35, "UNCHANGED_CORRECT": 275, "UNRESOLVED": 218}
    if heavy_rates["counts"] != expected_counts:
        raise RuntimeError(
            f"V1 paired categories drifted: expected {expected_counts}, got {heavy_rates['counts']}"
        )
    cues = learn_intent_cues(train_rows)
    _write_json(output_dir / "error_analysis" / "intent_cues_train_only.json", cues)
    formal_vocabulary = {
        token for row in train_rows for token in __import__("src.modeling.metrics", fromlist=["evaluation_tokens"]).evaluation_tokens(row["text"])
    }
    lookup_path = root / "data/annotations/formal_token_lookup.csv"
    if lookup_path.is_file():
        formal_vocabulary.update(
            pd.read_csv(lookup_path)["formal_token"].dropna().astype(str).str.casefold()
        )
    detailed = build_detailed_error_rows(
        heavy_frame, intent_cues=cues, formal_vocabulary=formal_vocabulary
    )
    detailed.to_csv(output_dir / "error_analysis" / "v1_detailed_error_analysis.csv", index=False)
    error_by_intent = (
        detailed.groupby("gold_intent")
        .agg(
            rows=("row_id", "size"),
            delta_cer_mean=("delta_cer", "mean"),
            delta_wer_mean=("delta_wer", "mean"),
            delta_chrf_mean=("delta_chrf", "mean"),
            gold_confidence_delta_mean=("gold_confidence_delta", "mean"),
            spacing_changed=("spacing_changed", "sum"),
            cue_not_restored=("discriminative_cue_not_restored", "sum"),
            real_word_substitution_candidates=("real_word_substitution_candidate", "sum"),
        )
        .reset_index()
    )
    error_categories = pd.crosstab(detailed["gold_intent"], detailed["category"])
    error_by_intent = error_by_intent.merge(
        error_categories.reset_index(), on="gold_intent", how="left"
    )
    error_by_intent.to_csv(
        output_dir / "error_analysis" / "error_analysis_by_intent.csv", index=False
    )
    unresolved = detailed.loc[detailed["category"].eq("UNRESOLVED")].copy()
    harmed = detailed.loc[detailed["category"].eq("HARMED")].copy()
    helped = detailed.loc[detailed["category"].eq("HELPED")].copy()
    unresolved.to_csv(output_dir / "error_analysis" / "unresolved_review.csv", index=False)
    harmed.sort_values(["delta_cer", "gold_confidence_delta"]).to_csv(
        output_dir / "error_analysis" / "harmed_largest_textual_degradation.csv", index=False
    )
    harmed.sort_values("gold_confidence_delta").to_csv(
        output_dir / "error_analysis" / "harmed_largest_confidence_drop.csv", index=False
    )
    helped.loc[~helped["normalized_equals_real"]].sort_values(
        "delta_cer", ascending=False
    ).to_csv(output_dir / "error_analysis" / "helped_without_exact_reconstruction.csv", index=False)
    grouped = grouped_restoration_statistics(detailed)
    grouped.to_csv(output_dir / "error_analysis" / "restoration_by_category.csv", index=False)
    correlations = restoration_confidence_correlations(detailed)
    _write_json(output_dir / "error_analysis" / "restoration_confidence_correlations.json", correlations)
    taxonomy_counts = unresolved["unresolved_taxonomy"].value_counts().to_dict()
    timezone_frame = detailed.loc[detailed["gold_intent"].eq("timezone")].copy()
    timezone_switch = timezone_frame.loc[
        timezone_frame["noisy_prediction"].eq("timezone")
        & timezone_frame["normalized_prediction"].eq("weather")
    ]
    timezone_real_oreer = timezone_switch["real_text"].str.casefold().str.contains("oreer", regex=False)
    timezone_normalized_missing_oreer = ~timezone_switch["normalized_text"].str.casefold().str.contains("oreer", regex=False)
    timezone_summary = {
        "timezone_rows": len(timezone_frame),
        "noisy_timezone_to_normalized_weather": len(timezone_switch),
        "switch_rows_with_real_oreer": int(timezone_real_oreer.sum()),
        "switch_rows_where_normalized_lacks_oreer": int(timezone_normalized_missing_oreer.sum()),
        "switch_rows_real_oreer_not_restored": int((timezone_real_oreer & timezone_normalized_missing_oreer).sum()),
    }
    _write_json(output_dir / "error_analysis" / "timezone_fiso_oreer_analysis.json", timezone_summary)

    global_summary = {
        "clean_preservation": clean_summary,
        "noise_severity": severity_summary.to_dict(orient="records"),
        "cross_method_generalization": cross_summary.to_dict(orient="records"),
        "method_applicability": method_applicability(),
        "error_analysis": {
            "paired_counts": heavy_rates["counts"],
            "unresolved_taxonomy_counts": taxonomy_counts,
            "helped_without_exact_reconstruction": int((~helped["normalized_equals_real"]).sum()),
            "timezone": timezone_summary,
            "correlations": correlations,
        },
    }
    _write_json(output_dir / "global_summary.json", global_summary)
    manifest = {
        "experiment": "laya_downstream_analysis_v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit_at_execution": current_git_commit(root),
        "seed": seed,
        "frozen_artifacts": frozen,
        "laya_config_sha256": sha256_file(root / LAYA_CHECKPOINT / "rl_agent_config.json"),
        "tnt_config_sha256": sha256_file(root / TNT_CHECKPOINT / "config.json"),
        "v1_metrics_sha256": sha256_file(root / V1_RESULTS_DIR / "metrics.json"),
        "real_baseline_reproduced": metrics["REAL"],
        "m1_severity": [severity.__dict__ for severity in M1_SEVERITIES],
        "test_rows": len(test_rows),
        "no_training_performed": True,
    }
    _write_json(output_dir / "experiment_manifest.json", manifest)
    print(f"[V2] Complete: {output_dir}", flush=True)
    return global_summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", default=".")
    parser.add_argument("--output-dir", default=str(V2_RESULTS_DIR))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_args()
    run_v2(
        arguments.repository_root,
        output_dir=arguments.output_dir,
        seed=arguments.seed,
        device=arguments.device,
    )
