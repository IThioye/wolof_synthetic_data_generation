"""Create immutable, video-disjoint gold train/dev/test files after annotation."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import datetime, timezone

import pandas as pd


from pathlib import Path
import sys
external_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(external_dir))

from config import (
    ANNOTATIONS_DIR,
    GOLD_ANNOTATIONS_PATH,
    MIN_GOLD_KEPT,
    MIN_GOLD_VIDEOS,
)


SPLIT_RATIOS = {"train": 0.70, "dev": 0.15, "test": 0.15}
REQUIRED_COLUMNS = {
    "source_index",
    "comment",
    "manual_formal_wolof",
    "status",
    "video_url",
}


def latest_kept_annotations(annotations: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(annotations.columns)
    if missing:
        raise ValueError(f"Gold annotations are missing columns: {sorted(missing)}")

    latest = annotations.drop_duplicates("source_index", keep="last").copy()
    latest["manual_formal_wolof"] = latest["manual_formal_wolof"].fillna("").str.strip()
    latest["video_url"] = latest["video_url"].fillna("").str.strip()
    kept = latest.loc[
        latest["status"].eq("keep") & latest["manual_formal_wolof"].ne("")
    ].copy()
    return kept.sort_values("source_index").reset_index(drop=True)


def assign_video_disjoint_splits(
    kept: pd.DataFrame, seed: int = 2026
) -> dict[str, pd.DataFrame]:
    if kept["video_url"].eq("").any():
        raise ValueError("Every kept annotation needs a video_url before splitting")

    group_sizes = kept.groupby("video_url").size().to_dict()
    if len(group_sizes) < len(SPLIT_RATIOS):
        raise ValueError("At least three videos are required for video-disjoint splits")

    rng = random.Random(seed)
    groups = list(group_sizes)
    rng.shuffle(groups)
    groups.sort(key=lambda group: group_sizes[group], reverse=True)

    total = len(kept)
    targets = {name: total * ratio for name, ratio in SPLIT_RATIOS.items()}
    counts = {name: 0 for name in SPLIT_RATIOS}
    assignments: dict[str, str] = {}

    for group in groups:
        # Relative deficit gives small splits a fair chance while retaining all
        # comments from one video in exactly one partition.
        split = max(
            SPLIT_RATIOS,
            key=lambda name: (targets[name] - counts[name]) / targets[name],
        )
        assignments[group] = split
        counts[split] += group_sizes[group]

    result = {}
    for split_name in SPLIT_RATIOS:
        split_groups = {
            group for group, assigned in assignments.items() if assigned == split_name
        }
        result[split_name] = kept.loc[kept["video_url"].isin(split_groups)].copy()
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_ready(kept: pd.DataFrame, min_kept: int, min_videos: int) -> None:
    kept_count = len(kept)
    video_count = kept["video_url"].replace("", pd.NA).nunique()
    problems = []
    if kept_count < min_kept:
        problems.append(f"{kept_count} kept annotations; need at least {min_kept}")
    if video_count < min_videos:
        problems.append(f"{video_count} represented videos; need at least {min_videos}")
    if kept["video_url"].eq("").any():
        problems.append("one or more kept annotations have no video_url")
    if problems:
        raise ValueError("Benchmark is not ready: " + "; ".join(problems))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=GOLD_ANNOTATIONS_PATH)
    parser.add_argument("--output-dir", type=Path, default=ANNOTATIONS_DIR / "splits")
    parser.add_argument("--min-kept", type=int, default=MIN_GOLD_KEPT)
    parser.add_argument("--min-videos", type=int, default=MIN_GOLD_VIDEOS)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    annotations = pd.read_csv(args.input)
    status_counts = annotations.drop_duplicates("source_index", keep="last")[
        "status"
    ].value_counts()
    kept = latest_kept_annotations(annotations)
    video_count = kept["video_url"].replace("", pd.NA).nunique()
    print("Latest annotation statuses:")
    print(status_counts.to_string())
    print(f"Eligible kept pairs: {len(kept):,}")
    print(f"Videos represented by kept pairs: {video_count:,}")

    try:
        validate_ready(kept, args.min_kept, args.min_videos)
        splits = assign_video_disjoint_splits(kept, args.seed)
    except ValueError as exc:
        print(exc)
        print("No split files were written. Continue diverse annotation first.")
        return 2
    for name, split in splits.items():
        print(
            f"{name}: {len(split):,} pairs from {split['video_url'].nunique():,} videos"
        )

    if args.dry_run:
        print("Dry run only; no split files were written.")
        return 0

    output_paths = {
        name: args.output_dir / f"gold_{name}.csv" for name in SPLIT_RATIOS
    }
    manifest_path = args.output_dir / "split_manifest.json"
    existing = [path for path in [*output_paths.values(), manifest_path] if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite locked benchmark files: "
            + ", ".join(str(path) for path in existing)
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, split in splits.items():
        split.to_csv(output_paths[name], index=False, encoding="utf-8")

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(args.input.resolve()),
        "source_sha256": sha256(args.input),
        "seed": args.seed,
        "ratios": SPLIT_RATIOS,
        "splits": {
            name: {
                "rows": len(split),
                "videos": sorted(split["video_url"].unique().tolist()),
                "sha256": sha256(output_paths[name]),
            }
            for name, split in splits.items()
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Locked split files written to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
