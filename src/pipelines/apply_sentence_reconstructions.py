"""Append approved reconstructed sentences to the gold annotation history."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ANNOTATIONS = PROJECT_ROOT / "data" / "annotations" / "gold_annotations.csv"
DEFAULT_REVIEW = (
    PROJECT_ROOT
    / "data"
    / "annotations"
    / "streamlit_sentence_reconstruction_review.csv"
)
DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "annotations"
    / "streamlit_sentence_reconstruction_application.json"
)


def normalize(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return " ".join(str(value).split()).casefold()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--approve-all",
        action="store_true",
        help="Record explicit project-owner approval for every reconstructed row.",
    )
    args = parser.parse_args()

    if not args.approve_all:
        raise ValueError("Pass --approve-all only after the project owner approves the review.")

    review = pd.read_csv(args.review, dtype={"source_index": "string"}, keep_default_na=False)
    required_review = {
        "source_index",
        "split",
        "auto_suggestion",
        "stale_manual_formal_wolof",
        "reconstructed_manual_formal_wolof",
        "review_status",
        "reviewer_notes",
    }
    missing = required_review - set(review.columns)
    if missing:
        raise ValueError(f"Review file is missing columns: {sorted(missing)}")
    if review.empty or review["source_index"].duplicated().any():
        raise ValueError("Review must contain unique, non-empty source rows.")
    if not review["split"].eq("train").all():
        raise ValueError("Only training rows may be repaired by this workflow.")
    if review["reconstructed_manual_formal_wolof"].map(normalize).eq("").any():
        raise ValueError("Every reconstructed sentence must be non-empty.")

    with args.annotations.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        history = list(reader)
    if not fieldnames:
        raise ValueError("Gold annotation file has no header.")

    latest: dict[str, dict[str, str]] = {}
    for row in history:
        latest[str(row["source_index"])] = row

    before_hash = sha256(args.annotations)
    appended: list[dict[str, str]] = []
    already_applied: list[str] = []
    for review_row in review.to_dict("records"):
        source_index = str(review_row["source_index"])
        current = latest.get(source_index)
        if current is None:
            raise ValueError(f"Unknown source_index in review: {source_index}")
        proposed = str(review_row["reconstructed_manual_formal_wolof"]).strip()
        if normalize(current["manual_formal_wolof"]) == normalize(proposed):
            already_applied.append(source_index)
            continue
        if normalize(current["manual_formal_wolof"]) != normalize(
            review_row["stale_manual_formal_wolof"]
        ):
            raise RuntimeError(
                f"Annotation {source_index} changed after review creation; refusing to append."
            )
        if normalize(current["auto_suggestion"]) != normalize(
            review_row["auto_suggestion"]
        ):
            raise RuntimeError(
                f"Automatic suggestion changed for {source_index}; refusing to append."
            )
        corrected = dict(current)
        corrected["manual_formal_wolof"] = proposed
        appended.append(corrected)

    if appended:
        with args.annotations.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writerows(appended)

    review["review_status"] = "approved"
    review.loc[review["reviewer_notes"].eq(""), "reviewer_notes"] = (
        "Approved by project owner; reconstructed from saved token corrections."
    )
    review.to_csv(args.review, index=False, encoding="utf-8-sig")

    manifest = {
        "applied_utc": datetime.now(timezone.utc).isoformat(),
        "annotations": str(args.annotations.resolve()),
        "annotations_sha256_before": before_hash,
        "annotations_sha256_after": sha256(args.annotations),
        "review": str(args.review.resolve()),
        "review_sha256": sha256(args.review),
        "approved_rows": len(review),
        "appended_rows": len(appended),
        "already_applied_rows": len(already_applied),
        "source_indices": review["source_index"].tolist(),
        "policy": "Append corrected records; preserve the original annotation history.",
    }
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Approved {len(review)} rows; appended {len(appended)} corrected history records "
        f"({len(already_applied)} already applied)."
    )


if __name__ == "__main__":
    main()
