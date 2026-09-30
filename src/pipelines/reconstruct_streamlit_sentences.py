"""Reconstruct stale Streamlit sentence annotations from token corrections.

The legacy Streamlit interface did not synchronize its full-sentence text area
when an annotator changed token-level candidates.  This utility identifies that
specific failure signature and produces a review sheet without changing the
annotation history or locked gold splits.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ANNOTATIONS = PROJECT_ROOT / "data" / "annotations" / "gold_annotations.csv"
DEFAULT_SPLITS = PROJECT_ROOT / "data" / "annotations" / "splits"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "annotations"
    / "streamlit_sentence_reconstruction_review.csv"
)


def normalize(value: object) -> str:
    """Normalize whitespace and case for consistency comparisons."""
    if value is None or pd.isna(value):
        return ""
    return " ".join(str(value).split()).casefold()


def reconstruct_sentence(payload_text: object) -> str:
    """Join the annotator's final token choices in their original order."""
    payload = json.loads(str(payload_text or "[]"))
    # Optimized sentence-level annotations store the reviewed surface sentence
    # explicitly.  This preserves punctuation while retaining token mappings.
    for correction in payload:
        sentence_target = str(correction.get("sentence_target") or "").strip()
        if sentence_target:
            return sentence_target
    tokens = []
    for correction in payload:
        final = str(correction.get("final_correction") or "").strip()
        selected = str(correction.get("selected_candidate") or "").strip()
        original = str(correction.get("token") or "").strip()
        tokens.append(final or selected or original)
    return " ".join(token for token in tokens if token).strip()


def load_split_membership(split_dir: Path) -> dict[str, str]:
    membership: dict[str, str] = {}
    for split in ("train", "dev", "test"):
        path = split_dir / f"gold_{split}.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path, dtype={"source_index": "string"})
        for source_index in frame["source_index"].dropna():
            membership[str(source_index)] = split
    return membership


def build_review(annotations: pd.DataFrame, split_dir: Path) -> pd.DataFrame:
    required = {
        "source_index",
        "comment",
        "auto_suggestion",
        "manual_formal_wolof",
        "status",
        "video_url",
        "token_corrections_json",
    }
    missing = required - set(annotations.columns)
    if missing:
        raise ValueError(f"Gold annotations are missing columns: {sorted(missing)}")

    latest = annotations.drop_duplicates("source_index", keep="last").copy()
    latest = latest.loc[latest["status"].eq("keep")].copy()
    latest["reconstructed_manual_formal_wolof"] = latest[
        "token_corrections_json"
    ].map(reconstruct_sentence)

    # This is the exact legacy failure signature: the sentence remained equal
    # to the automatic suggestion although the saved token choices imply a
    # different final sentence.
    affected = latest.loc[
        latest.apply(
            lambda row: normalize(row["manual_formal_wolof"])
            == normalize(row["auto_suggestion"])
            and normalize(row["reconstructed_manual_formal_wolof"])
            != normalize(row["manual_formal_wolof"]),
            axis=1,
        )
    ].copy()

    membership = load_split_membership(split_dir)
    affected["split"] = affected["source_index"].astype(str).map(membership).fillna("")
    affected["review_status"] = ""
    affected["reviewer_notes"] = ""
    affected = affected.rename(columns={"manual_formal_wolof": "stale_manual_formal_wolof"})

    columns = [
        "source_index",
        "split",
        "video_url",
        "comment",
        "auto_suggestion",
        "stale_manual_formal_wolof",
        "reconstructed_manual_formal_wolof",
        "review_status",
        "reviewer_notes",
        "token_corrections_json",
    ]
    return affected[columns].sort_values("source_index").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    annotations = pd.read_csv(args.annotations, dtype={"source_index": "string"})
    review = build_review(annotations, args.split_dir)
    non_train = review.loc[review["split"].ne("train")]
    if not non_train.empty:
        raise RuntimeError(
            "The detected failure signature includes held-out rows; refusing to export."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    review.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"Wrote {len(review)} reconstructed training sentences to {args.output}")


if __name__ == "__main__":
    main()
