"""Remove stale identity autos created from an ambiguous lookup entry.

The current lookup seed is authoritative for identifying ambiguity.  Human
reviews and two-confirmation mapping defaults are retained.  ``--apply`` makes
an archive before replacing the active occurrence CSV.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.annotation.linguistic_annotation_core import clean, lookup_key
from src.config import (
    FORMAL_TOKEN_LOOKUP_PATH,
    FORMAL_TOKEN_LOOKUP_SEED_PATH,
    LINGUISTIC_ANNOTATIONS_PATH,
)
from src.pipelines.refresh_linguistic_auto_annotations import _sha256


def ambiguous_seed_keys(seed: pd.DataFrame) -> set[str]:
    ambiguous_pos = seed.get("ambiguous_pos", pd.Series("false", index=seed.index))
    ambiguous_language = seed.get(
        "language_ambiguous", pd.Series("false", index=seed.index)
    )
    mask = ambiguous_pos.map(clean).eq("true") | ambiguous_language.map(clean).eq("true")
    return {
        lookup_key(value)
        for value in seed.loc[mask, "formal_token"]
        if lookup_key(value)
    }


def stale_ambiguous_identity_mask(
    annotations: pd.DataFrame, ambiguous_keys: set[str]
) -> pd.Series:
    notes = annotations.get("notes", pd.Series("", index=annotations.index)).map(clean)
    formal_keys = annotations["formal_token"].map(lookup_key)
    return notes.str.startswith("AUTO_IDENTITY_") & formal_keys.isin(ambiguous_keys)


def refresh(
    annotations_path: Path,
    seed_path: Path,
    curated_path: Path,
    *,
    apply: bool,
) -> dict[str, object]:
    annotations = pd.read_csv(annotations_path, keep_default_na=False, dtype=str)
    seed = pd.read_csv(seed_path, keep_default_na=False, dtype=str)
    keys = ambiguous_seed_keys(seed)
    remove = stale_ambiguous_identity_mask(annotations, keys)
    removed = annotations.loc[remove]
    stats: dict[str, object] = {
        "ambiguous_seed_tokens": len(keys),
        "rows_before": len(annotations),
        "stale_identity_autos_removed": int(remove.sum()),
        "human_and_mapping_rows_preserved": int((~remove).sum()),
        "rows_after": int((~remove).sum()),
        "removed_by_lookup_source": removed.get(
            "lookup_source", pd.Series(dtype=str)
        ).value_counts().to_dict(),
    }
    if not apply:
        return stats

    stamp = datetime.now(timezone.utc).strftime("pre_ambiguity_refresh_%Y%m%dT%H%M%SZ")
    archive = annotations_path.parent / "linguistic_annotation_archives" / stamp
    archive.mkdir(parents=True, exist_ok=False)
    archived = []
    for path in (annotations_path, seed_path, curated_path):
        if not path.exists():
            continue
        destination = archive / path.name
        shutil.copy2(path, destination)
        archived.append({
            "source": str(path.resolve()),
            "archived_as": destination.name,
            "sha256": _sha256(destination),
        })

    temporary = annotations_path.with_suffix(".ambiguity-refresh.tmp.csv")
    annotations.loc[~remove].to_csv(temporary, index=False, encoding="utf-8")
    temporary.replace(annotations_path)
    stats["archive"] = str(archive.resolve())
    stats["archived_files"] = archived
    stats["applied_utc"] = datetime.now(timezone.utc).isoformat()
    (archive / "refresh_manifest.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=LINGUISTIC_ANNOTATIONS_PATH)
    parser.add_argument("--seed", type=Path, default=FORMAL_TOKEN_LOOKUP_SEED_PATH)
    parser.add_argument("--curated", type=Path, default=FORMAL_TOKEN_LOOKUP_PATH)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    print(json.dumps(
        refresh(args.annotations, args.seed, args.curated, apply=args.apply),
        indent=2,
        ensure_ascii=False,
    ))


if __name__ == "__main__":
    main()
