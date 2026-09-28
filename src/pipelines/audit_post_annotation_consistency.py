"""Audit corrected gold splits against token and linguistic annotations.

This command is read-only with respect to source data. It writes compact review
CSVs containing only rows that require a project-owner decision.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.annotation.linguistic_annotation_core import build_occurrences
from src.config import ANNOTATIONS_DIR, GOLD_SPLITS_DIR, LINGUISTIC_ANNOTATIONS_PATH
from src.pipelines.reconstruct_streamlit_sentences import normalize, reconstruct_sentence


DEFAULT_SPLIT_REVIEW = ANNOTATIONS_DIR / "gold_split_consistency_review.csv"
DEFAULT_LINGUISTIC_REVIEW = (
    ANNOTATIONS_DIR / "linguistic_annotation_reconciliation_review.csv"
)
SPLIT_REVIEW_COLUMNS = (
    "split", "source_index", "video_url", "comment",
    "current_manual_formal_wolof", "sentence_reconstructed_from_tokens",
    "decision", "approved_formal_sentence", "reviewer_notes",
)
LINGUISTIC_REVIEW_COLUMNS = (
    "issue", "occurrence_id", "annotation_informal", "annotation_formal",
    "current_informal", "current_formal", "annotation_pos",
    "annotation_error_types", "annotation_notes", "decision", "reviewer_notes",
)


def build_split_review(split_dir: Path) -> pd.DataFrame:
    records: list[dict[str, str]] = []
    for split in ("train", "dev", "test"):
        frame = pd.read_csv(
            split_dir / f"gold_{split}.csv",
            keep_default_na=False,
            dtype={"source_index": "string"},
        )
        for row in frame.to_dict("records"):
            reconstructed = reconstruct_sentence(row["token_corrections_json"])
            if normalize(reconstructed) == normalize(row["manual_formal_wolof"]):
                continue
            records.append(
                {
                    "split": split,
                    "source_index": str(row["source_index"]),
                    "video_url": str(row["video_url"]),
                    "comment": str(row["comment"]),
                    "current_manual_formal_wolof": str(row["manual_formal_wolof"]),
                    "sentence_reconstructed_from_tokens": reconstructed,
                    "decision": "",
                    "approved_formal_sentence": "",
                    "reviewer_notes": "",
                }
            )
    return pd.DataFrame(records, columns=SPLIT_REVIEW_COLUMNS)


def build_linguistic_review(
    gold_train: pd.DataFrame, annotations: pd.DataFrame
) -> pd.DataFrame:
    occurrences = build_occurrences(gold_train).set_index("occurrence_id")
    latest = annotations.drop_duplicates("occurrence_id", keep="last").set_index(
        "occurrence_id"
    )
    records: list[dict[str, str]] = []
    for occurrence_id, row in latest.iterrows():
        if occurrence_id not in occurrences.index:
            records.append(
                {
                    "issue": "orphan_occurrence_id",
                    "occurrence_id": str(occurrence_id),
                    "annotation_informal": str(row.get("informal_token", "")),
                    "annotation_formal": str(row.get("formal_token", "")),
                    "current_informal": "",
                    "current_formal": "",
                    "annotation_pos": str(row.get("pos", "")),
                    "annotation_error_types": str(row.get("error_types_json", "")),
                    "annotation_notes": str(row.get("notes", "")),
                    "decision": "",
                    "reviewer_notes": "",
                }
            )
            continue
        current = occurrences.loc[occurrence_id]
        if (
            str(row.get("informal_token", "")) == str(current["informal_token"])
            and str(row.get("formal_token", "")) == str(current["formal_token"])
        ):
            continue
        records.append(
            {
                "issue": "token_pair_changed_in_gold_train",
                "occurrence_id": str(occurrence_id),
                "annotation_informal": str(row.get("informal_token", "")),
                "annotation_formal": str(row.get("formal_token", "")),
                "current_informal": str(current["informal_token"]),
                "current_formal": str(current["formal_token"]),
                "annotation_pos": str(row.get("pos", "")),
                "annotation_error_types": str(row.get("error_types_json", "")),
                "annotation_notes": str(row.get("notes", "")),
                "decision": "",
                "reviewer_notes": "",
            }
        )
    for occurrence_id in sorted(set(occurrences.index) - set(latest.index)):
        current = occurrences.loc[occurrence_id]
        records.append(
            {
                "issue": "missing_annotation",
                "occurrence_id": str(occurrence_id),
                "annotation_informal": "",
                "annotation_formal": "",
                "current_informal": str(current["informal_token"]),
                "current_formal": str(current["formal_token"]),
                "annotation_pos": "",
                "annotation_error_types": "",
                "annotation_notes": "",
                "decision": "",
                "reviewer_notes": "",
            }
        )
    return pd.DataFrame(records, columns=LINGUISTIC_REVIEW_COLUMNS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path, default=GOLD_SPLITS_DIR)
    parser.add_argument("--annotations", type=Path, default=LINGUISTIC_ANNOTATIONS_PATH)
    parser.add_argument("--split-review", type=Path, default=DEFAULT_SPLIT_REVIEW)
    parser.add_argument(
        "--linguistic-review", type=Path, default=DEFAULT_LINGUISTIC_REVIEW
    )
    args = parser.parse_args()

    split_review = build_split_review(args.split_dir)
    gold_train = pd.read_csv(
        args.split_dir / "gold_train.csv",
        keep_default_na=False,
        dtype={"source_index": "string"},
    )
    annotations = pd.read_csv(
        args.annotations, keep_default_na=False, dtype={"source_index": "string"}
    )
    linguistic_review = build_linguistic_review(gold_train, annotations)

    args.split_review.parent.mkdir(parents=True, exist_ok=True)
    split_review.to_csv(args.split_review, index=False, encoding="utf-8-sig")
    linguistic_review.to_csv(
        args.linguistic_review, index=False, encoding="utf-8-sig"
    )
    by_split = split_review["split"].value_counts().to_dict()
    print(
        f"Wrote {len(split_review)} split inconsistencies {by_split} and "
        f"{len(linguistic_review)} linguistic reconciliation cases."
    )


if __name__ == "__main__":
    main()
