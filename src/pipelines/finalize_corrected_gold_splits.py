"""Synchronize corrected locked splits to history and refresh their manifest.

Dry-run is the default. ``--apply`` appends changed split rows to the annotation
history, verifies that the latest history exactly reproduces the existing split
membership/content, and atomically writes a new integrity manifest. It never
reassigns a source row to another split.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import (
    GOLD_ANNOTATIONS_PATH,
    GOLD_SPLIT_MANIFEST_PATH,
    GOLD_SPLITS_DIR,
)
from src.pipelines.prepare_gold_splits import SPLIT_RATIOS, latest_kept_annotations
from src.pipelines.reconstruct_streamlit_sentences import normalize, reconstruct_sentence


SPLIT_NAMES = ("train", "dev", "test")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_and_validate_splits(split_dir: Path) -> dict[str, pd.DataFrame]:
    splits = {
        name: pd.read_csv(
            split_dir / f"gold_{name}.csv",
            keep_default_na=False,
            dtype={"source_index": "string"},
        )
        for name in SPLIT_NAMES
    }
    columns = list(splits["train"].columns)
    all_ids: set[str] = set()
    videos: dict[str, set[str]] = {}
    for name, frame in splits.items():
        if list(frame.columns) != columns:
            raise ValueError(f"{name} columns do not match train columns")
        ids = set(frame["source_index"].astype(str))
        if frame["source_index"].duplicated().any() or all_ids & ids:
            raise ValueError(f"Duplicate source IDs found in or across {name}")
        all_ids |= ids
        videos[name] = set(frame["video_url"].astype(str))
        mismatches = []
        for row in frame.to_dict("records"):
            reconstructed = reconstruct_sentence(row["token_corrections_json"])
            if normalize(reconstructed) != normalize(row["manual_formal_wolof"]):
                mismatches.append(str(row["source_index"]))
        if mismatches:
            raise ValueError(
                f"{name} has sentence/token inconsistencies: {mismatches}"
            )
    for index, left in enumerate(SPLIT_NAMES):
        for right in SPLIT_NAMES[index + 1 :]:
            if videos[left] & videos[right]:
                raise ValueError(f"Video leakage between {left} and {right}")
    return splits


def rows_needing_history_sync(
    history: pd.DataFrame, splits: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    latest = history.drop_duplicates("source_index", keep="last").set_index(
        "source_index"
    )
    records = []
    for split, frame in splits.items():
        for row in frame.to_dict("records"):
            source_index = str(row["source_index"])
            if source_index not in latest.index:
                raise ValueError(f"Split row {source_index} is absent from annotation history")
            if any(
                str(row[column]) != str(latest.at[source_index, column])
                for column in frame.columns
                if column != "source_index"
            ):
                records.append({**row, "_split": split})
    return pd.DataFrame(records, columns=[*history.columns, "_split"])


def verify_history_reproduces_splits(
    history: pd.DataFrame, splits: dict[str, pd.DataFrame]
) -> None:
    kept = latest_kept_annotations(history).set_index("source_index")
    split_ids = {
        str(value)
        for frame in splits.values()
        for value in frame["source_index"]
    }
    if set(kept.index.astype(str)) != split_ids:
        raise RuntimeError("Latest kept annotation membership differs from locked splits")
    for frame in splits.values():
        for row in frame.to_dict("records"):
            source_index = str(row["source_index"])
            for column in frame.columns:
                if column == "source_index":
                    continue
                if str(kept.at[source_index, column]) != str(row[column]):
                    raise RuntimeError(
                        f"History does not reproduce split row {source_index}, field {column}"
                    )


def build_manifest(
    annotations_path: Path,
    split_dir: Path,
    splits: dict[str, pd.DataFrame],
    previous_manifest_hash: str,
) -> dict[str, object]:
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "revision_reason": "Post-annotation corrections approved by project owner",
        "previous_manifest_sha256": previous_manifest_hash,
        "source": str(annotations_path.resolve()),
        "source_sha256": sha256(annotations_path),
        "seed": 2026,
        "ratios": SPLIT_RATIOS,
        "splits": {
            name: {
                "rows": len(frame),
                "videos": sorted(frame["video_url"].unique().tolist()),
                "sha256": sha256(split_dir / f"gold_{name}.csv"),
            }
            for name, frame in splits.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=GOLD_ANNOTATIONS_PATH)
    parser.add_argument("--split-dir", type=Path, default=GOLD_SPLITS_DIR)
    parser.add_argument("--manifest", type=Path, default=GOLD_SPLIT_MANIFEST_PATH)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    splits = load_and_validate_splits(args.split_dir)
    history = pd.read_csv(
        args.annotations, keep_default_na=False, dtype={"source_index": "string"}
    )
    pending = rows_needing_history_sync(history, splits)
    print(
        "Rows needing history synchronization: "
        + str(pending["_split"].value_counts().to_dict() if not pending.empty else {})
    )
    if not args.apply:
        print("Dry run only; no files were changed.")
        return

    fieldnames = list(history.columns)
    if not pending.empty:
        with args.annotations.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            for row in pending.to_dict("records"):
                writer.writerow({name: row[name] for name in fieldnames})

    updated = pd.read_csv(
        args.annotations, keep_default_na=False, dtype={"source_index": "string"}
    )
    verify_history_reproduces_splits(updated, splits)
    previous_hash = sha256(args.manifest) if args.manifest.exists() else ""
    manifest = build_manifest(
        args.annotations, args.split_dir, splits, previous_hash
    )
    temporary = args.manifest.with_suffix(".tmp.json")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(args.manifest)
    print(
        f"Appended {len(pending)} corrected history rows and refreshed "
        f"{args.manifest}."
    )


if __name__ == "__main__":
    main()
