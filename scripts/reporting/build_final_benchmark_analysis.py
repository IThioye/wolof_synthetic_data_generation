"""Build reproducible tables and figures for the final PFE benchmark report.

This script is analysis-only: it reads the locked gold splits, frozen test
predictions, final benchmark manifests, and already-exported synthetic corpora.
It never trains a model or changes the frozen benchmark artifacts.
"""

from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.modeling.benchmark_data import (  # noqa: E402
    assemble_training_data,
    canonical_text,
    load_locked_gold,
    load_synthetic_pairs,
)
from src.modeling.metrics import (  # noqa: E402
    evaluation_tokens,
    generation_metrics,
    levenshtein_distance,
    normalize_for_evaluation,
)


BENCHMARK_DIR = ROOT / "results" / "m7_seq2seq" / "tnt_edit_gold_dev_v4"
FROZEN_DIR = BENCHMARK_DIR / "frozen_test"
OUTPUT_DIR = ROOT / "results" / "tnt_edit_report_analysis"
ANNOTATIONS_PATH = ROOT / "data" / "annotations" / "token_linguistic_annotations.csv"
M6_PROFILE_PATH = ROOT / "artifacts" / "m6_gold_linguistic_profile.json"
METHODS = ("m1", "m2", "m3", "m4", "m6")
CONDITIONS = ("m0_identity", "gold", *METHODS)
METHOD_LABELS = {
    "m0_identity": "M0 identity",
    "gold": "Gold only",
    "m1": "M1 rules",
    "m2": "M2 Eflomal",
    "m3": "M3 CMDR",
    "m4": "M4 Oolel",
    "m6": "M6 gold-informed",
}
COLORS = {
    "m0_identity": "#718096",
    "gold": "#1f4e79",
    "m1": "#d17a22",
    "m2": "#dc9b3a",
    "m3": "#7c5aa6",
    "m4": "#2d8b7b",
    "m6": "#b3485a",
}


def write_csv(frame: pd.DataFrame, name: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT_DIR / name, index=False, encoding="utf-8")


def parse_json_list(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return [text]
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    if isinstance(parsed, dict):
        return [str(item) for item in parsed]
    return [str(parsed)]


def selected_synthetic_data() -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Reconstruct the exact common-target sample used by the final runner."""
    unique_by_method: dict[str, pd.DataFrame] = {}
    raw_by_method: dict[str, pd.DataFrame] = {}
    for method in METHODS:
        raw = load_synthetic_pairs(method, max_rows=None, seed=2026, allow_provisional=True)
        unique = assemble_training_data(raw, None, regime="synthetic_only", seed=2026)
        unique["target_key"] = unique["target_text"].map(canonical_text)
        unique_by_method[method] = unique.drop_duplicates("target_key", keep="first").reset_index(drop=True)
        raw_by_method[method] = pd.read_csv(
            ROOT / "data" / "synthetic" / {
                "m1": "m1_beqi_rules_pairs.csv",
                "m2": "m2_eflomal_rules_pairs.csv",
                "m3": "m3_cmdr_rules_pairs.csv",
                "m4": "m4_oolel_external_pairs.csv",
                "m6": "m6_gold_linguistic_pairs.csv",
            }[method]
        )

    common_targets = set.intersection(*(set(frame["target_key"]) for frame in unique_by_method.values()))
    selected_targets = set(
        pd.Series(sorted(common_targets), dtype=str).sample(n=3330, random_state=2026).tolist()
    )
    target_reference = unique_by_method["m4"].set_index("target_key")["target_text"]
    selected: dict[str, pd.DataFrame] = {}
    selected_raw: dict[str, pd.DataFrame] = {}
    for method in METHODS:
        frame = unique_by_method[method]
        matched = frame.loc[frame["target_key"].isin(selected_targets)].copy()
        matched["target_text"] = matched["target_key"].map(target_reference)
        matched = matched.drop(columns="target_key").sort_values("target_text", kind="stable").reset_index(drop=True)
        if len(matched) != 3330:
            raise RuntimeError(f"Unexpected matched row count for {method}: {len(matched)}")
        selected[method] = matched
        selected_raw[method] = matched[["source_id"]].merge(
            raw_by_method[method], on="source_id", how="left", validate="one_to_one"
        )
    return selected, selected_raw


def corpus_profile() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    gold = load_locked_gold()
    selected, selected_raw = selected_synthetic_data()
    rows: list[dict[str, object]] = []

    gold_train = gold["train"]
    gold_metrics = generation_metrics(
        gold_train["source_text"].tolist(), gold_train["target_text"].tolist()
    )
    gold_char_edits = [
        levenshtein_distance(
            list(normalize_for_evaluation(source)), list(normalize_for_evaluation(target))
        )
        for source, target in zip(gold_train["source_text"], gold_train["target_text"])
    ]
    gold_word_edits = [
        levenshtein_distance(evaluation_tokens(source), evaluation_tokens(target))
        for source, target in zip(gold_train["source_text"], gold_train["target_text"])
    ]
    rows.append(
        {
            "condition": "gold_train",
            "label": "Gold train",
            "rows": len(gold_train),
            "changed_rows": int(
                sum(canonical_text(s) != canonical_text(t) for s, t in zip(gold_train["source_text"], gold_train["target_text"]))
            ),
            "changed_percent": 100.0 * sum(
                canonical_text(s) != canonical_text(t) for s, t in zip(gold_train["source_text"], gold_train["target_text"])
            ) / len(gold_train),
            "source_chars_mean": gold_train["source_text"].str.len().mean(),
            "target_chars_mean": gold_train["target_text"].str.len().mean(),
            "source_target_length_ratio": gold_train["source_text"].str.len().sum() / gold_train["target_text"].str.len().sum(),
            "character_edits_mean": np.mean(gold_char_edits),
            "word_edits_mean": np.mean(gold_word_edits),
            **gold_metrics,
        }
    )
    for method, frame in selected.items():
        metrics = generation_metrics(
            frame["source_text"].tolist(), frame["target_text"].tolist()
        )
        changed = sum(canonical_text(s) != canonical_text(t) for s, t in zip(frame["source_text"], frame["target_text"]))
        char_edits = [
            levenshtein_distance(
                list(normalize_for_evaluation(source)), list(normalize_for_evaluation(target))
            )
            for source, target in zip(frame["source_text"], frame["target_text"])
        ]
        word_edits = [
            levenshtein_distance(evaluation_tokens(source), evaluation_tokens(target))
            for source, target in zip(frame["source_text"], frame["target_text"])
        ]
        rows.append(
            {
                "condition": method,
                "label": METHOD_LABELS[method],
                "rows": len(frame),
                "changed_rows": int(changed),
                "changed_percent": 100.0 * changed / len(frame),
                "source_chars_mean": frame["source_text"].str.len().mean(),
                "target_chars_mean": frame["target_text"].str.len().mean(),
                "source_target_length_ratio": frame["source_text"].str.len().sum() / frame["target_text"].str.len().sum(),
                "character_edits_mean": np.mean(char_edits),
                "word_edits_mean": np.mean(word_edits),
                **metrics,
            }
        )
    profile = pd.DataFrame(rows)
    write_csv(profile, "corpus_transformation_profile.csv")

    # A vertical stack remains legible after LaTeX scales the figure to one
    # text column.  The previous 1x3 layout compressed labels and placed value
    # annotations too close to the upper grid/axes lines.
    fig, axes = plt.subplots(3, 1, figsize=(10.6, 8.4))
    labels = profile["label"].tolist()
    colors = ["#1f4e79"] + [COLORS[key] for key in METHODS]
    for axis, metric, title in (
        (axes[0], "cer", "Source-to-target CER (lower = closer)"),
        (axes[1], "wer", "Source-to-target WER (lower = closer)"),
        (axes[2], "chrf", "Source-to-target chrF (higher = closer)"),
    ):
        values = profile[metric].to_numpy()
        positions = np.arange(len(labels))
        bars = axis.barh(positions, values, color=colors, height=0.66)
        axis.set_title(title, fontsize=14, loc="left", pad=8, fontweight="bold")
        axis.set_yticks(positions, labels, fontsize=11)
        axis.invert_yaxis()
        axis.tick_params(axis="x", labelsize=11)
        axis.grid(axis="x", alpha=0.18, linewidth=0.8)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        label_space = max(float(values.max()) * 0.18, 0.04)
        axis.set_xlim(0, float(values.max()) + label_space)
        for bar, value in zip(bars, values):
            axis.text(
                float(value) + label_space * 0.10,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.2f}",
                ha="left",
                va="center",
                fontsize=11,
                fontweight="bold",
                clip_on=False,
            )
    fig.suptitle("Intrinsic distance between informal sources and formal targets", fontsize=17, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.965), h_pad=1.15)
    fig.savefig(OUTPUT_DIR / "corpus_transformation_profile.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return profile, selected_raw


def gold_annotation_analysis() -> None:
    data = pd.read_csv(ANNOTATIONS_PATH, keep_default_na=False)
    data["changed"] = [canonical_text(left) != canonical_text(right) for left, right in zip(data["informal_token"], data["formal_token"])]
    data["labels"] = data["error_types_json"].map(parse_json_list)

    labels = Counter(label for values in data["labels"] for label in set(values))
    label_frame = pd.DataFrame(
        [
            {
                "error_label": label,
                "occurrences": count,
                "percent_of_annotated_occurrences": 100.0 * count / len(data),
            }
            for label, count in labels.most_common()
        ]
    )
    write_csv(label_frame, "gold_error_labels.csv")

    pos_frame = (
        data.groupby("pos", dropna=False)
        .agg(occurrences=("occurrence_id", "size"), changed_occurrences=("changed", "sum"))
        .reset_index()
    )
    pos_frame["change_rate_percent"] = 100.0 * pos_frame["changed_occurrences"] / pos_frame["occurrences"]
    pos_frame = pos_frame.sort_values(["occurrences", "pos"], ascending=[False, True])
    write_csv(pos_frame, "gold_pos_change_rates.csv")

    language_frame = (
        data.groupby(["source_language", "target_language"], dropna=False)
        .agg(occurrences=("occurrence_id", "size"), changed_occurrences=("changed", "sum"))
        .reset_index()
    )
    language_frame["percent"] = 100.0 * language_frame["occurrences"] / len(data)
    write_csv(language_frame, "gold_language_pairs.csv")

    categorical_rows = []
    for field in ("review_status", "protected", "entity_type", "lookup_source"):
        for value, count in data[field].value_counts(dropna=False).items():
            categorical_rows.append(
                {"field": field, "value": str(value), "occurrences": int(count), "percent": 100.0 * count / len(data)}
            )
    write_csv(pd.DataFrame(categorical_rows), "gold_annotation_categorical_summary.csv")

    plot_labels = label_frame.loc[label_frame["error_label"] != "identity"].head(12).iloc[::-1].copy()
    plot_labels["display_label"] = plot_labels["error_label"].str.replace("_", " ", regex=False)
    fig, axis = plt.subplots(figsize=(9.4, 6.4))
    bars = axis.barh(plot_labels["display_label"], plot_labels["occurrences"], color="#1f4e79", height=0.68)
    axis.set_title("Most frequent gold-training transformation labels", fontsize=16, pad=12, fontweight="bold")
    axis.set_xlabel("Annotated occurrences (multi-label counts)", fontsize=13, labelpad=8)
    axis.tick_params(axis="both", labelsize=12)
    axis.grid(axis="x", alpha=0.18, linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    label_space = max(float(plot_labels["occurrences"].max()) * 0.13, 12.0)
    axis.set_xlim(0, float(plot_labels["occurrences"].max()) + label_space)
    for bar, value in zip(bars, plot_labels["occurrences"]):
        axis.text(
            float(value) + label_space * 0.08,
            bar.get_y() + bar.get_height() / 2,
            str(int(value)),
            va="center",
            fontsize=12,
            fontweight="bold",
            clip_on=False,
        )
    fig.tight_layout(pad=1.2)
    fig.savefig(OUTPUT_DIR / "gold_error_labels.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    plot_pos = pos_frame.loc[pos_frame["occurrences"] >= 10].sort_values("change_rate_percent")
    fig, axis = plt.subplots(figsize=(9.2, 6.2))
    bars = axis.barh(plot_pos["pos"], plot_pos["change_rate_percent"], color="#b3485a", height=0.68)
    axis.set_title("Gold-training change rate by POS (at least 10 occurrences)", fontsize=16, pad=12, fontweight="bold")
    axis.set_xlabel("Changed occurrences (%)", fontsize=13, labelpad=8)
    axis.tick_params(axis="both", labelsize=12)
    axis.set_xlim(0, 110)
    axis.grid(axis="x", alpha=0.18, linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    for bar, value in zip(bars, plot_pos["change_rate_percent"]):
        axis.text(
            min(float(value) + 1.5, 103.0),
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}%",
            va="center",
            fontsize=12,
            fontweight="bold",
            clip_on=False,
        )
    fig.tight_layout(pad=1.2)
    fig.savefig(OUTPUT_DIR / "gold_pos_change_rates.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def method_transformation_analysis(selected_raw: dict[str, pd.DataFrame]) -> None:
    rows: list[dict[str, object]] = []
    label_rows: list[dict[str, object]] = []
    for method, frame in selected_raw.items():
        for column in (
            "edit_count",
            "alignment_edit_count",
            "spelling_edit_count",
            "lexical_mapping_edit_count",
            "transformation_edit_count",
        ):
            if column in frame:
                numeric = pd.to_numeric(frame[column], errors="coerce").fillna(0)
                rows.append(
                    {
                        "method": method,
                        "label": METHOD_LABELS[method],
                        "measure": column,
                        "total": float(numeric.sum()),
                        "mean_per_row": float(numeric.mean()),
                        "rows_with_measure": int((numeric > 0).sum()),
                        "rows_with_measure_percent": 100.0 * float((numeric > 0).mean()),
                    }
                )
        if "error_types" in frame:
            counts = Counter(label for value in frame["error_types"] for label in set(parse_json_list(value)))
            for label, count in counts.most_common():
                label_rows.append(
                    {
                        "method": method,
                        "method_label": METHOD_LABELS[method],
                        "error_label": label,
                        "rows": count,
                        "percent_of_matched_rows": 100.0 * count / len(frame),
                    }
                )
    write_csv(pd.DataFrame(rows), "matched_method_edit_mechanisms.csv")
    write_csv(pd.DataFrame(label_rows), "matched_method_error_labels.csv")


def row_char_components(frame: pd.DataFrame, prediction_column: str) -> tuple[np.ndarray, np.ndarray]:
    edits = []
    lengths = []
    for prediction, reference in zip(frame[prediction_column], frame["reference"]):
        prediction = normalize_for_evaluation(prediction)
        reference = normalize_for_evaluation(reference)
        edits.append(levenshtein_distance(list(prediction), list(reference)))
        lengths.append(len(reference))
    return np.asarray(edits, dtype=np.int64), np.asarray(lengths, dtype=np.int64)


def prediction_analysis() -> None:
    summary = pd.read_csv(FROZEN_DIR / "test_summary.csv")
    write_csv(summary, "frozen_test_metrics.csv")
    frames: dict[str, pd.DataFrame] = {}
    model_rows: list[dict[str, object]] = []
    baseline_frame: pd.DataFrame | None = None
    baseline_edits: np.ndarray | None = None

    for condition in CONDITIONS:
        if condition == "m0_identity":
            base = pd.read_csv(FROZEN_DIR / "gold" / "test_predictions.csv")
            frame = base[["source_id", "source_text", "reference"]].copy()
            frame["prediction"] = frame["source_text"]
        else:
            frame = pd.read_csv(FROZEN_DIR / condition / "test_predictions.csv", keep_default_na=False)
        frames[condition] = frame
        char_edits, ref_lengths = row_char_components(frame, "prediction")
        source_edits = np.asarray(
            [
                levenshtein_distance(list(normalize_for_evaluation(s)), list(normalize_for_evaluation(r)))
                for s, r in zip(frame["source_text"], frame["reference"])
            ],
            dtype=np.int64,
        )
        if condition == "m0_identity":
            baseline_frame = frame
            baseline_edits = char_edits
        predictions = frame["prediction"].map(normalize_for_evaluation)
        token_lists = predictions.map(evaluation_tokens)
        dominant_share = token_lists.map(
            lambda tokens: max(Counter(tokens).values()) / len(tokens) if tokens else 0.0
        )
        single_character_share = token_lists.map(
            lambda tokens: sum(len(token) == 1 for token in tokens) / len(tokens) if tokens else 0.0
        )
        row = {
            "condition": condition,
            "label": METHOD_LABELS[condition],
            "rows": len(frame),
            "empty_predictions": int(predictions.eq("").sum()),
            "unique_predictions": int(predictions.nunique()),
            "prediction_chars_mean": predictions.str.len().mean(),
            "reference_chars_mean": frame["reference"].map(normalize_for_evaluation).str.len().mean(),
            "prediction_reference_length_ratio": predictions.str.len().sum() / frame["reference"].map(normalize_for_evaluation).str.len().sum(),
            "prediction_tokens_mean": token_lists.map(len).mean(),
            "reference_tokens_mean": frame["reference"].map(evaluation_tokens).map(len).mean(),
            "dominant_token_ge_50_percent_rows": int(((dominant_share >= 0.5) & (token_lists.map(len) >= 4)).sum()),
            "single_character_tokens_ge_50_percent_rows": int(
                ((single_character_share >= 0.5) & (token_lists.map(len) >= 4)).sum()
            ),
            "rows_better_than_identity_char_edits": int((char_edits < source_edits).sum()),
            "rows_equal_to_identity_char_edits": int((char_edits == source_edits).sum()),
            "rows_worse_than_identity_char_edits": int((char_edits > source_edits).sum()),
            "median_sentence_cer": float(np.median(char_edits / np.maximum(ref_lengths, 1))),
        }
        model_rows.append(row)
    write_csv(pd.DataFrame(model_rows), "frozen_test_prediction_diagnostics.csv")

    if baseline_frame is None or baseline_edits is None:
        raise RuntimeError("Identity baseline was not constructed")
    baseline_lengths = np.asarray([len(normalize_for_evaluation(x)) for x in baseline_frame["reference"]], dtype=np.int64)
    rng = np.random.default_rng(2026)
    samples = rng.integers(0, len(baseline_frame), size=(5000, len(baseline_frame)))
    baseline_boot = baseline_edits[samples].sum(axis=1) / baseline_lengths[samples].sum(axis=1)
    bootstrap_rows = []
    for condition, frame in frames.items():
        edits, lengths = row_char_components(frame, "prediction")
        boot = edits[samples].sum(axis=1) / lengths[samples].sum(axis=1)
        delta = boot - baseline_boot
        bootstrap_rows.append(
            {
                "condition": condition,
                "label": METHOD_LABELS[condition],
                "cer": edits.sum() / lengths.sum(),
                "cer_ci95_low": np.quantile(boot, 0.025),
                "cer_ci95_high": np.quantile(boot, 0.975),
                "cer_delta_vs_identity": (edits.sum() - baseline_edits.sum()) / lengths.sum(),
                "delta_ci95_low": np.quantile(delta, 0.025),
                "delta_ci95_high": np.quantile(delta, 0.975),
            }
        )
    bootstrap = pd.DataFrame(bootstrap_rows)
    write_csv(bootstrap, "frozen_test_cer_bootstrap.csv")

    ordered = summary.set_index("condition").loc[list(CONDITIONS)].reset_index()
    # Stacking the metrics vertically gives each condition label and value room
    # at report scale.  Values are written beyond the bar ends rather than on
    # top of horizontal axes/grid lines.
    fig, axes = plt.subplots(3, 1, figsize=(10.6, 9.2))
    labels = [METHOD_LABELS[key] for key in ordered["condition"]]
    colors = [COLORS[key] for key in ordered["condition"]]
    for axis, metric, title in (
        (axes[0], "cer", "CER (lower is better)"),
        (axes[1], "wer", "WER (lower is better)"),
        (axes[2], "chrf", "chrF (higher is better)"),
    ):
        values = ordered[metric].to_numpy()
        positions = np.arange(len(labels))
        bars = axis.barh(positions, values, color=colors, height=0.66)
        axis.set_title(title, fontsize=14, loc="left", pad=8, fontweight="bold")
        axis.set_yticks(positions, labels, fontsize=11)
        axis.invert_yaxis()
        axis.tick_params(axis="x", labelsize=11)
        axis.grid(axis="x", alpha=0.18, linewidth=0.8)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        identity_value = float(ordered.loc[ordered["condition"] == "m0_identity", metric].iloc[0])
        axis.axvline(identity_value, color="#718096", linestyle="--", linewidth=1.0, alpha=0.75, zorder=0)
        label_space = max(float(values.max()) * 0.18, 0.04)
        axis.set_xlim(0, float(values.max()) + label_space)
        for bar, value in zip(bars, values):
            decimals = 2 if metric == "chrf" else 3
            axis.text(
                float(value) + label_space * 0.10,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.{decimals}f}",
                ha="left",
                va="center",
                fontsize=11,
                fontweight="bold",
                clip_on=False,
            )
    fig.suptitle("Frozen gold-test normalization results (29 sentences)", fontsize=17, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.965), h_pad=1.15)
    fig.savefig(OUTPUT_DIR / "frozen_test_metrics_report.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def training_analysis() -> None:
    rows = []
    for condition in ("gold", *METHODS):
        selection = json.loads((BENCHMARK_DIR / "selections" / f"{condition}.json").read_text(encoding="utf-8"))
        gold_manifest_path = Path(selection["selected_run_manifest"])
        gold_manifest = json.loads(gold_manifest_path.read_text(encoding="utf-8"))
        gold_history = pd.read_csv(gold_manifest["training_history_path"])
        gold_eval = gold_history.loc[pd.to_numeric(gold_history.get("eval_cer"), errors="coerce").notna()].copy()
        best_eval = gold_eval.loc[pd.to_numeric(gold_eval["eval_cer"], errors="coerce").idxmin()]
        synthetic_runtime = 0.0
        synthetic_best_epoch = np.nan
        if condition != "gold":
            parent_path = Path(gold_manifest["parent_manifest_path"])
            parent = json.loads(parent_path.read_text(encoding="utf-8"))
            synthetic_runtime = float(parent.get("train_metrics", {}).get("train_runtime", 0.0))
            synthetic_history = pd.read_csv(parent["training_history_path"])
            synthetic_eval = synthetic_history.loc[
                pd.to_numeric(synthetic_history.get("eval_cer"), errors="coerce").notna()
            ].copy()
            if not synthetic_eval.empty:
                synthetic_best_epoch = float(
                    synthetic_eval.loc[pd.to_numeric(synthetic_eval["eval_cer"], errors="coerce").idxmin(), "epoch"]
                )
        gold_runtime = float(gold_manifest.get("train_metrics", {}).get("train_runtime", 0.0))
        rows.append(
            {
                "condition": condition,
                "label": METHOD_LABELS[condition],
                "synthetic_runtime_seconds": synthetic_runtime,
                "gold_runtime_seconds": gold_runtime,
                "total_runtime_seconds": synthetic_runtime + gold_runtime,
                "total_runtime_hours": (synthetic_runtime + gold_runtime) / 3600.0,
                "synthetic_best_epoch": synthetic_best_epoch,
                "gold_best_epoch": float(best_eval["epoch"]),
                "best_development_cer": float(best_eval["eval_cer"]),
            }
        )
    write_csv(pd.DataFrame(rows), "training_runtime_and_selection.csv")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    profile, selected_raw = corpus_profile()
    gold_annotation_analysis()
    method_transformation_analysis(selected_raw)
    prediction_analysis()
    training_analysis()
    metadata = {
        "created_from_frozen_benchmark": str(BENCHMARK_DIR.relative_to(ROOT)),
        "analysis_seed": 2026,
        "bootstrap_samples": 5000,
        "matched_synthetic_rows_per_method": 3330,
        "frozen_test_rows": 29,
        "notes": [
            "Bootstrap intervals reflect test-sample uncertainty only, not training-seed variance.",
            "Gold error labels are multi-valued; label counts therefore do not sum to the number of occurrences.",
            "Source-to-target corpus metrics describe perturbation intensity, not semantic quality.",
        ],
    }
    (OUTPUT_DIR / "analysis_manifest.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(profile.to_string(index=False))
    print(f"\nWrote report analyses to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
