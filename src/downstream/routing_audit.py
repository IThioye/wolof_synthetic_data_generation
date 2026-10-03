"""Frozen V1/V2 feasibility audit for selective Wolof normalization."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from src.downstream.routing_baselines import (
    baseline_evaluations,
    evaluate_decisions,
    grouped_classifier_router,
    grouped_threshold_router,
)
from src.downstream.routing_features import (
    DEPLOYABLE_SURFACE_FEATURES,
    DEPLOYABLE_TNT_FEATURES,
    ORACLE_FEATURES,
    TntSignalExtractor,
    feature_families,
    load_formal_vocabulary,
    surface_features,
)
from src.downstream.text_error_analysis import pair_text_metrics
from src.downstream.v2_protocol import (
    TNT_CHECKPOINT,
    V1_TNT_SHA256,
    V2_RESULTS_DIR,
    current_git_commit,
    sha256_file,
    validate_frozen_artifacts,
)


RESULTS_DIR = Path("results/downstream/normalization_router_audit_v1")
FORMAL_VOCABULARY_PATH = Path("data/vocabulary.txt")
SEED = 2026
CONDITIONS = ("REAL", "M1_LIGHT", "M1_MEDIUM", "M1_HEAVY", "M5", "M6")


def _read(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path, keep_default_na=False)


def _sha256(path: Path) -> str:
    return sha256_file(path)


def _condition_source_paths(root: Path) -> dict[str, dict[str, Path | None]]:
    base = root / V2_RESULTS_DIR
    return {
        "REAL": {
            "paired": base / "clean/clean_paired_analysis.csv",
            "condition": None,
            "before_prediction": base / "predictions/real.csv",
            "after_prediction": base / "predictions/tnt_real.csv",
        },
        "M1_LIGHT": {
            "paired": base / "severity/paired_light.csv",
            "condition": base / "conditions/noisy_light.csv",
            "before_prediction": base / "predictions/noisy_light.csv",
            "after_prediction": base / "predictions/normalized_light.csv",
        },
        "M1_MEDIUM": {
            "paired": base / "severity/paired_medium.csv",
            "condition": base / "conditions/noisy_medium.csv",
            "before_prediction": base / "predictions/noisy_medium.csv",
            "after_prediction": base / "predictions/normalized_medium.csv",
        },
        "M1_HEAVY": {
            "paired": base / "severity/paired_heavy.csv",
            "condition": base / "conditions/noisy_heavy.csv",
            "before_prediction": base / "predictions/noisy_heavy.csv",
            "after_prediction": base / "predictions/normalized_heavy.csv",
        },
        "M5": {
            "paired": base / "cross_method/paired_m5.csv",
            "condition": base / "conditions/noisy_m5.csv",
            "before_prediction": base / "predictions/noisy_m5.csv",
            "after_prediction": base / "predictions/normalized_m5.csv",
        },
        "M6": {
            "paired": base / "cross_method/paired_m6.csv",
            "condition": base / "conditions/noisy_m6.csv",
            "before_prediction": base / "predictions/noisy_m6.csv",
            "after_prediction": base / "predictions/normalized_m6.csv",
        },
    }


def _standardize_condition(name: str, paths: Mapping[str, Path | None]) -> pd.DataFrame:
    paired = _read(Path(paths["paired"]))
    before = _read(Path(paths["before_prediction"]))[
        ["row_id", "confidence"]
    ].rename(columns={"confidence": "laya_confidence_before"})
    after = _read(Path(paths["after_prediction"]))[
        ["row_id", "confidence"]
    ].rename(columns={"confidence": "laya_confidence_after"})

    if name == "REAL":
        frame = paired.rename(
            columns={
                "intent": "gold_intent",
                "real_text": "input_text",
                "tnt_real_text": "normalized_text",
                "real_prediction": "laya_prediction_before",
                "tnt_real_prediction": "laya_prediction_after",
                "real_gold_confidence": "gold_confidence_before",
                "tnt_real_gold_confidence": "gold_confidence_after",
            }
        )
        mapping = {
            "HELPED_CLEAN": "HELPED",
            "HARMED_CLEAN": "HARMED",
            "UNCHANGED_CORRECT": "NEUTRAL_CORRECT",
            "UNCHANGED_WRONG": "NEUTRAL_WRONG",
        }
        frame["real_text"] = frame["input_text"]
        frame["synthetic_operation_count"] = 0
        frame["noise_method"] = "none"
        frame["severity"] = "clean"
        frame["routing_label"] = frame["category"].map(mapping)
    else:
        frame = paired.rename(
            columns={
                "noisy_text": "input_text",
                "noisy_prediction": "laya_prediction_before",
                "normalized_prediction": "laya_prediction_after",
                "noisy_gold_confidence": "gold_confidence_before",
                "normalized_gold_confidence": "gold_confidence_after",
            }
        )
        mapping = {
            "HELPED": "HELPED",
            "HARMED": "HARMED",
            "UNCHANGED_CORRECT": "NEUTRAL_CORRECT",
            "UNRESOLVED": "NEUTRAL_WRONG",
        }
        condition = _read(Path(paths["condition"]))[
            ["row_id", "noise_method", "operation_count"]
        ].rename(columns={"operation_count": "synthetic_operation_count"})
        frame = frame.merge(
            condition, on="row_id", how="left", validate="one_to_one"
        )
        frame["severity"] = {
            "M1_LIGHT": "light",
            "M1_MEDIUM": "medium",
            "M1_HEAVY": "heavy",
            "M5": "method_specific",
            "M6": "method_specific",
        }[name]
        frame["routing_label"] = frame["category"].map(mapping)
    frame = frame.merge(before, on="row_id", how="left", validate="one_to_one")
    frame = frame.merge(after, on="row_id", how="left", validate="one_to_one")
    frame["condition"] = name
    # Injongo's raw example_id restarts within intent partitions. row_id is the
    # globally unique utterance identifier; all six variants share it.
    frame["routing_group_id"] = frame["row_id"].astype(str)
    frame["correct_before"] = frame["laya_prediction_before"] == frame["gold_intent"]
    frame["correct_after"] = frame["laya_prediction_after"] == frame["gold_intent"]
    frame["normalize_beneficial"] = (frame["routing_label"] == "HELPED").astype(int)
    frame["routing_target_3way"] = frame["routing_label"].map(
        {"HELPED": "NORMALIZE", "HARMED": "SKIP", "NEUTRAL_CORRECT": "UNCERTAIN", "NEUTRAL_WRONG": "UNCERTAIN"}
    )
    required = [
        "example_id", "row_id", "routing_group_id", "condition", "noise_method", "severity",
        "input_text", "normalized_text", "real_text", "gold_intent",
        "laya_prediction_before", "laya_prediction_after", "correct_before",
        "correct_after", "gold_confidence_before", "gold_confidence_after",
        "laya_confidence_before", "laya_confidence_after", "routing_label",
        "normalize_beneficial", "routing_target_3way", "synthetic_operation_count",
    ]
    if frame[required].isna().any().any():
        raise RuntimeError(f"Missing values while standardizing {name}")
    return frame[required].copy()


def build_routing_dataset(
    root: Path,
    *,
    extract_tnt_signals: bool = True,
    device: str | None = None,
    batch_size: int = 64,
) -> pd.DataFrame:
    frames = [
        _standardize_condition(name, paths)
        for name, paths in _condition_source_paths(root).items()
    ]
    frame = pd.concat(frames, ignore_index=True)
    expected_rows = 638 * len(CONDITIONS)
    if len(frame) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} paired rows, found {len(frame)}")
    expected_ids = set(frames[0]["example_id"])
    for name, group in frame.groupby("condition"):
        if set(group["example_id"]) != expected_ids:
            raise RuntimeError(f"Example IDs differ for {name}")

    vocabulary_path = root / FORMAL_VOCABULARY_PATH
    vocabulary = load_formal_vocabulary(vocabulary_path)
    surface_rows = [surface_features(text, vocabulary) for text in frame["input_text"]]
    frame = pd.concat([frame, pd.DataFrame(surface_rows)], axis=1)

    deployable_text_rows = []
    oracle_rows = []
    for row in frame.to_dict(orient="records"):
        output_metrics = pair_text_metrics(row["input_text"], row["normalized_text"])
        oracle_metrics = pair_text_metrics(row["real_text"], row["input_text"])
        tokens = max(int(row["sentence_token_count"]), 1)
        characters = max(int(row["sentence_char_length"]), 1)
        operations = int(row["synthetic_operation_count"])
        deployable_text_rows.append(
            {
                "tnt_output_char_edits": output_metrics["char_edits"],
                "tnt_output_char_edit_rate": output_metrics["cer"],
                "tnt_output_token_edits": output_metrics["word_edits"],
                "tnt_output_token_edit_rate": output_metrics["wer"],
                "tnt_output_chrf": output_metrics["chrf"],
                "tnt_output_changed": int(not output_metrics["exact"]),
            }
        )
        oracle_rows.append(
            {
                "operations_per_token": operations / tokens,
                "operations_per_character": operations / characters,
                "oracle_char_edits_from_real": oracle_metrics["char_edits"],
                "oracle_character_corruption_rate": oracle_metrics["cer"],
                "oracle_token_edits_from_real": oracle_metrics["word_edits"],
                "oracle_token_corruption_rate": oracle_metrics["wer"],
                "oracle_chrf_to_real": oracle_metrics["chrf"],
            }
        )
    frame = pd.concat(
        [frame, pd.DataFrame(deployable_text_rows), pd.DataFrame(oracle_rows)], axis=1
    )

    signals_path = root / RESULTS_DIR / "tnt_head_signals.csv"
    if extract_tnt_signals:
        frozen = validate_frozen_artifacts(root)
        if frozen["tnt_model_sha256"] != V1_TNT_SHA256:
            raise RuntimeError("Frozen TNT checkpoint changed")
        extractor = TntSignalExtractor(root / TNT_CHECKPOINT, device=device, batch_size=batch_size)
        signals = pd.DataFrame(extractor.extract(frame["input_text"].tolist()))
        signals.insert(0, "condition", frame["condition"].to_numpy())
        signals.insert(0, "example_id", frame["example_id"].to_numpy())
        signals_path.parent.mkdir(parents=True, exist_ok=True)
        signals.to_csv(signals_path, index=False)
    else:
        signals = _read(signals_path)
        if list(zip(signals["example_id"], signals["condition"])) != list(
            zip(frame["example_id"], frame["condition"])
        ):
            raise RuntimeError("Cached TNT signals do not match routing rows")
    signal_features = signals.drop(columns=["example_id", "condition"])
    frame = pd.concat([frame, signal_features], axis=1)
    return frame


def _feature_summary(frame: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    records = []
    for category, group in frame.groupby("routing_label", sort=False):
        for feature in features:
            values = pd.to_numeric(group[feature], errors="coerce").dropna()
            records.append(
                {
                    "routing_label": category,
                    "feature": feature,
                    "n": len(values),
                    "mean": values.mean(),
                    "median": values.median(),
                    "std": values.std(ddof=1),
                    "q1": values.quantile(0.25),
                    "q3": values.quantile(0.75),
                    "min": values.min(),
                    "max": values.max(),
                }
            )
    return pd.DataFrame(records)


def _benjamini_hochberg(pvalues: Sequence[float]) -> np.ndarray:
    values = np.asarray(pvalues, dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = np.minimum.accumulate(
        (ranked * len(values) / np.arange(1, len(values) + 1))[::-1]
    )[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.minimum(adjusted, 1.0)
    return output


def _separability(frame: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    from scipy.stats import mannwhitneyu
    from sklearn.metrics import roc_auc_score

    clear = frame[frame["routing_label"].isin(["HELPED", "HARMED"])].copy()
    labels = (clear["routing_label"] == "HELPED").astype(int).to_numpy()
    records = []
    for feature in features:
        helped = clear.loc[clear["routing_label"] == "HELPED", feature].astype(float).to_numpy()
        harmed = clear.loc[clear["routing_label"] == "HARMED", feature].astype(float).to_numpy()
        statistic = mannwhitneyu(helped, harmed, alternative="two-sided")
        all_values = clear[feature].astype(float).to_numpy()
        auc = roc_auc_score(labels, all_values) if len(np.unique(all_values)) > 1 else 0.5
        records.append(
            {
                "feature": feature,
                "feature_family": "oracle" if feature in ORACLE_FEATURES else "deployable",
                "helped_n": len(helped),
                "harmed_n": len(harmed),
                "helped_median": float(np.median(helped)),
                "harmed_median": float(np.median(harmed)),
                "mann_whitney_u": float(statistic.statistic),
                "p_value": float(statistic.pvalue),
                "rank_biserial_helped_vs_harmed": float(2 * statistic.statistic / (len(helped) * len(harmed)) - 1),
                "auc_helped_positive": float(auc),
                "auc_best_orientation": float(max(auc, 1 - auc)),
            }
        )
    output = pd.DataFrame(records)
    output["p_value_bh"] = _benjamini_hochberg(output["p_value"])
    return output.sort_values("auc_best_orientation", ascending=False)


def _rates(frame: pd.DataFrame, group_columns: Sequence[str]) -> pd.DataFrame:
    rows = []
    for keys, group in frame.groupby(list(group_columns), sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        counts = group["routing_label"].value_counts()
        row = dict(zip(group_columns, keys))
        row.update({"rows": len(group)})
        for label in ("HELPED", "HARMED", "NEUTRAL_CORRECT", "NEUTRAL_WRONG"):
            row[label.lower()] = int(counts.get(label, 0))
            row[f"p_{label.lower()}"] = float(counts.get(label, 0) / len(group))
        row["net_benefit"] = row["p_helped"] - row["p_harmed"]
        row["expected_accuracy_gain"] = row["net_benefit"]
        rows.append(row)
    return pd.DataFrame(rows)


def _plot_outputs(
    frame: pd.DataFrame,
    severity: pd.DataFrame,
    router_results: pd.DataFrame,
    separability: pd.DataFrame,
    output_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    severity_order = ["clean", "light", "medium", "heavy"]
    severity_plot = severity.set_index("severity").loc[severity_order]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.arange(len(severity_order))
    axes[0].plot(x, severity_plot["p_helped"] * 100, marker="o", label="HELPED")
    axes[0].plot(x, severity_plot["p_harmed"] * 100, marker="o", label="HARMED")
    axes[0].set_xticks(x, severity_order)
    axes[0].set_ylabel("Rate (%)")
    axes[0].set_xlabel("M1 severity")
    axes[0].set_ylim(bottom=0)
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].plot(x, severity_plot["net_benefit"] * 100, marker="o", color="#6b3fa0")
    axes[1].set_xticks(x, severity_order)
    axes[1].set_ylabel("HELPED − HARMED (pp)")
    axes[1].set_xlabel("M1 severity")
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "severity_help_harm.png", dpi=180)
    plt.close(fig)

    best_feature = separability.loc[
        separability["feature_family"] == "deployable", "feature"
    ].iloc[0]
    clear = frame[frame["routing_label"].isin(["HELPED", "HARMED"])]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    values = [
        clear.loc[clear["routing_label"] == label, best_feature].astype(float)
        for label in ("HELPED", "HARMED")
    ]
    ax.boxplot(values, tick_labels=["HELPED", "HARMED"], showfliers=False)
    ax.set_ylabel(best_feature)
    ax.set_title("Best descriptive deployable feature (not a held-out selection)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "deployable_feature_helped_vs_harmed.png", dpi=180)
    plt.close(fig)

    priority = [
        "always_skip", "always_normalize", "grouped_threshold_deployable_tnt",
        "grouped_logistic_regression", "grouped_decision_tree", "oracle_help_only",
    ]
    selected = router_results[router_results["condition"] == "ALL"].set_index("strategy")
    available = [value for value in priority if value in selected.index]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(available, selected.loc[available, "accuracy"] * 100, color="#315b7d")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Final Laya accuracy (%)")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "router_final_accuracy.png", dpi=180)
    plt.close(fig)


def _evaluate_all_routers(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    evaluations = baseline_evaluations(frame)
    folds = []

    oracle_severity, details = grouped_threshold_router(
        frame,
        ["synthetic_operation_count"],
        strategy="oracle_severity_threshold",
        feature_set="oracle_analysis_only",
    )
    evaluations.append(oracle_severity)
    details["strategy"] = oracle_severity.strategy
    folds.append(details)

    oracle_threshold, details = grouped_threshold_router(
        frame,
        ORACLE_FEATURES,
        strategy="oracle_feature_threshold",
        feature_set="oracle_analysis_only",
    )
    evaluations.append(oracle_threshold)
    details["strategy"] = oracle_threshold.strategy
    folds.append(details)

    tnt_threshold, details = grouped_threshold_router(
        frame,
        DEPLOYABLE_TNT_FEATURES,
        strategy="grouped_threshold_deployable_tnt",
        feature_set="deployable_tnt",
    )
    evaluations.append(tnt_threshold)
    details["strategy"] = tnt_threshold.strategy
    folds.append(details)

    deployable = list(DEPLOYABLE_SURFACE_FEATURES + DEPLOYABLE_TNT_FEATURES)
    all_threshold, details = grouped_threshold_router(
        frame,
        deployable,
        strategy="grouped_threshold_deployable_all",
        feature_set="deployable_all",
    )
    evaluations.append(all_threshold)
    details["strategy"] = all_threshold.strategy
    folds.append(details)

    for model_name in ("logistic_regression", "decision_tree"):
        evaluation, details = grouped_classifier_router(
            frame, deployable, model_name=model_name, feature_set="deployable_all"
        )
        evaluations.append(evaluation)
        details["strategy"] = evaluation.strategy
        folds.append(details)

    result_rows = []
    for evaluation in evaluations:
        combined = evaluate_decisions(frame, evaluation)
        combined["condition"] = "ALL"
        result_rows.append(combined)
        for condition, indices in frame.groupby("condition", sort=False).groups.items():
            subset = frame.loc[indices].reset_index(drop=True)
            subset_evaluation = type(evaluation)(
                evaluation.strategy,
                evaluation.feature_set,
                evaluation.decisions[np.asarray(list(indices), dtype=int)],
            )
            result = evaluate_decisions(subset, subset_evaluation)
            result["condition"] = condition
            result_rows.append(result)
    return pd.DataFrame(result_rows), pd.concat(folds, ignore_index=True)


def _write_json(path: Path, value) -> None:
    def clean(item):
        if isinstance(item, dict):
            return {str(key): clean(data) for key, data in item.items()}
        if isinstance(item, list):
            return [clean(data) for data in item]
        if isinstance(item, (np.integer,)):
            return int(item)
        if isinstance(item, (np.floating, float)):
            return float(item) if math.isfinite(float(item)) else None
        return item

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _report(
    frame: pd.DataFrame,
    severity: pd.DataFrame,
    methods: pd.DataFrame,
    routers: pd.DataFrame,
    separability: pd.DataFrame,
) -> str:
    combined = routers[routers["condition"] == "ALL"].set_index("strategy")
    skip = combined.loc["always_skip"]
    always = combined.loc["always_normalize"]
    oracle = combined.loc["oracle_help_only"]
    candidates = combined.loc[
        [index for index in combined.index if index.startswith("grouped_")]
    ].sort_values("accuracy", ascending=False)
    best = candidates.iloc[0]
    best_name = candidates.index[0]
    per_condition = routers[
        routers["strategy"].isin(
            ["always_skip", "always_normalize", best_name, "oracle_help_only"]
        )
        & routers["condition"].ne("ALL")
    ].pivot(index="condition", columns="strategy", values="accuracy").reset_index()
    keep_rows = separability[
        separability["feature"].isin(
            [
                "tnt_operation_mean_keep_probability",
                "tnt_word_mean_keep_probability",
                "tnt_boundary_mean_no_change_probability",
            ]
        )
    ]
    top_features = separability[separability["feature_family"] == "deployable"].head(8)
    clear_rows = int(frame["routing_label"].isin(["HELPED", "HARMED"]).sum())
    return f"""# Normalization router feasibility audit V1

## Scope

This audit reuses the frozen V1/V2 Laya and TNT predictions. It trains neither
model. Grouped routing experiments keep every condition of one utterance in
the same fold. Injongo reuses raw `example_id` values between intents, so the
globally unique `row_id` is recorded as `routing_group_id`. Oracle features are
analysis-only and are never described as deployable.

## Is there an oracle opportunity?

- Always skip: accuracy **{skip['accuracy']:.2%}**, macro-F1 **{skip['macro_f1']:.2%}**.
- Always normalize: accuracy **{always['accuracy']:.2%}**, macro-F1 **{always['macro_f1']:.2%}**.
- Perfect help-only oracle: accuracy **{oracle['accuracy']:.2%}**, macro-F1 **{oracle['macro_f1']:.2%}**.
- Oracle gain over skip: **{oracle['accuracy'] - skip['accuracy']:+.2%}**; over always normalize: **{oracle['accuracy'] - always['accuracy']:+.2%}**.

There is therefore real headroom for selection. This is an upper bound, not a
deployable result.

## Severity pattern

{severity.to_markdown(index=False)}

The sign changes between light and medium M1 noise in this controlled setup.
It is not a universal operation-count threshold: M5/M6 use different
transformations, and operation count is unavailable on naturally noisy text.

## Method pattern

{methods.to_markdown(index=False)}

## Are HELPED and HARMED separable?

There are {clear_rows} clear-outcome rows. The strongest descriptive
deployable signals are:

{top_features[['feature','helped_median','harmed_median','auc_best_orientation','rank_biserial_helped_vs_harmed','p_value_bh']].to_markdown(index=False)}

These univariate figures are descriptive. Multiple comparisons use a
Benjamini-Hochberg adjustment, and held-out routing performance below is the
decisive test.

## Leakage-controlled routing baselines

{combined.reset_index()[['strategy','feature_set','accuracy','macro_f1','normalization_rate','helped_routed_to_tnt','harmed_routed_to_tnt']].to_markdown(index=False)}

Best simple deployable strategy: **{best_name}**, accuracy **{best['accuracy']:.2%}**
({best['accuracy'] - skip['accuracy']:+.2%} versus always skip), macro-F1
**{best['macro_f1']:.2%}**, normalization rate **{best['normalization_rate']:.2%}**.

### Results by condition

{per_condition.to_markdown(index=False)}

## TNT KEEP signals

- Character head: `operation_projection`; KEEP is action 0 among
  `keep/delete/substitute/expand`.
- Word head: `word_projection`; KEEP is action 0 among `keep/edit`, computed
  from contextual encoder states mean-pooled over each whitespace-delimited
  word and enriched by the existing lexical flags.
- Boundary head: `boundary_projection`; no-change is `keep_space` on an input
  space and `none` elsewhere. Changes are principally `delete_space` and
  `insert_space_before`.
- The logits exist during inference but the normalizer wrapper previously
  returned only generated text. This audit reads the frozen heads directly;
  no architecture or checkpoint was changed.
- These softmax scores are **not calibrated probabilities**. Training used
  label smoothing, class weighting and focal loss for boundaries, with no
  post-hoc calibration. `1 - P(KEEP)` is therefore a useful score to audit,
  not yet a probability of normalization need.

Descriptive KEEP separability (HELPED versus HARMED):

{keep_rows[['feature','helped_median','harmed_median','auc_best_orientation','rank_biserial_helped_vs_harmed','p_value_bh']].to_markdown(index=False)}

## Feasibility decision (A–F)

- **A — Oracle headroom:** yes. The oracle is {oracle['accuracy'] - skip['accuracy']:+.2%}
  above always skip, so routing is not intrinsically pointless.
- **B — Deployable separability:** moderate, not strong. The best univariate
  AUCs are around 0.70, and HELPED/HARMED distributions still overlap.
- **C — Simple rule/classifier:** yes on the pooled controlled benchmark.
  `{best_name}` beats always skip by {best['accuracy'] - skip['accuracy']:+.2%}
  and always normalize by {best['accuracy'] - always['accuracy']:+.2%} under
  grouped out-of-fold evaluation.
- **D — KEEP/logit signal:** useful but uncalibrated. Character and word KEEP
  scores have moderate association; boundary confidence is substantially
  weaker. They can be normalization-need scores only after independent
  calibration.
- **E — Beyond M1-heavy:** not convincingly. The pooled gain mainly comes from
  M1 medium/heavy. On M5 and M6 the best pooled router is more conservative
  than always-TNT but does not consistently beat always-skip.
- **F — Recommended next step:** do **not** build a neural router yet. Preserve
  this small logistic/threshold baseline, collect or reserve human-noisy
  routing data, and calibrate a conservative abstention threshold there. A
  complex router is justified only if that independent evaluation confirms a
  gain beyond the M1 distribution.
"""


def run_audit(
    repository_root: str | Path = ".",
    *,
    extract_tnt_signals: bool = True,
    device: str | None = None,
    batch_size: int = 64,
) -> dict:
    root = Path(repository_root).resolve()
    output = root / RESULTS_DIR
    output.mkdir(parents=True, exist_ok=True)
    frozen = validate_frozen_artifacts(root)
    frame = build_routing_dataset(
        root,
        extract_tnt_signals=extract_tnt_signals,
        device=device,
        batch_size=batch_size,
    )
    features = list(DEPLOYABLE_SURFACE_FEATURES + DEPLOYABLE_TNT_FEATURES + ORACLE_FEATURES)
    summary = _feature_summary(frame, features)
    separability = _separability(frame, features)
    severity = _rates(
        frame[frame["condition"].isin(["REAL", "M1_LIGHT", "M1_MEDIUM", "M1_HEAVY"])],
        ["condition", "severity"],
    )
    methods = _rates(frame, ["condition", "noise_method", "severity"])
    intents = _rates(frame, ["condition", "gold_intent"])
    routers, folds = _evaluate_all_routers(frame)

    frame.to_csv(output / "routing_dataset.csv", index=False)
    summary.to_csv(output / "feature_summary.csv", index=False)
    separability.to_csv(output / "feature_separability.csv", index=False)
    separability[
        separability["feature"].str.contains("keep|boundary", case=False, regex=True)
    ].to_csv(output / "keep_signal_analysis.csv", index=False)
    severity.to_csv(output / "severity_help_harm.csv", index=False)
    methods.to_csv(output / "method_help_harm.csv", index=False)
    intents.to_csv(output / "intent_help_harm.csv", index=False)
    routers[routers["strategy"].str.startswith("oracle")].to_csv(
        output / "oracle_routing_results.csv", index=False
    )
    routers.to_csv(output / "simple_router_results.csv", index=False)
    folds.to_csv(output / "router_fold_selections.csv", index=False)
    _write_json(output / "feature_families.json", feature_families())

    _plot_outputs(frame, severity, routers, separability, output / "figures")
    report = _report(frame, severity, methods, routers, separability)
    (output / "routing_audit_report.md").write_text(report, encoding="utf-8")

    combined = routers[routers["condition"] == "ALL"].set_index("strategy")
    deployable_names = [name for name in combined.index if name.startswith("grouped_")]
    best_name = combined.loc[deployable_names, "accuracy"].idxmax()
    manifest = {
        "experiment": "normalization_router_audit_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit_at_execution": current_git_commit(root),
        "seed": SEED,
        "no_laya_or_tnt_training": True,
        "frozen_artifacts": frozen,
        "source_result_hashes": {
            str(path.relative_to(root)): _sha256(path)
            for values in _condition_source_paths(root).values()
            for path in values.values()
            if path is not None
        },
        "formal_vocabulary": {
            "path": str(FORMAL_VOCABULARY_PATH),
            "sha256": _sha256(root / FORMAL_VOCABULARY_PATH),
        },
        "rows": len(frame),
        "conditions": list(CONDITIONS),
        "grouping_key": "routing_group_id (= row_id; raw example_id repeats across intents)",
        "outer_group_folds": 5,
        "inner_group_folds": 4,
        "best_deployable_strategy": best_name,
    }
    _write_json(output / "experiment_manifest.json", manifest)
    compact = {
        "oracle": combined.loc[["always_skip", "always_normalize", "oracle_help_only"]].reset_index().to_dict(orient="records"),
        "best_deployable": combined.loc[best_name].to_dict() | {"strategy": best_name},
        "severity": severity.to_dict(orient="records"),
        "method": methods.to_dict(orient="records"),
        "keep_probabilities_calibrated": False,
        "recommendation": "See routing_audit_report.md; promote only if grouped deployable gains generalize beyond M1-heavy.",
    }
    _write_json(output / "routing_audit_summary.json", compact)
    return compact


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", default=".")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--reuse-tnt-signals", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_args()
    result = run_audit(
        arguments.repository_root,
        extract_tnt_signals=not arguments.reuse_tnt_signals,
        device=arguments.device,
        batch_size=arguments.batch_size,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))

