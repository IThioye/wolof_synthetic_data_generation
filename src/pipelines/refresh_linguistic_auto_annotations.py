"""Remove stale automatic identity annotations without losing human edits.

The command is intentionally dry-run by default.  With ``--apply``, it first
archives the annotation and lookup CSV files, then removes only automatic rows
whose linguistic fields still exactly match the lookup record that created
them.  Human rows and automatic rows subsequently changed by a reviewer are
preserved.
"""

from __future__ import annotations

import argparse
import hashlib
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
    SENEGALESE_SURNAMES_PATH,
)


AUTO_PREFIX = "AUTO_IDENTITY_"
COMPARISON_FIELDS = (
    "lemma",
    "pos",
    "source_language",
    "target_language",
    "entity_type",
    "protected",
    "error_type",
    "error_types_json",
    "transformation_steps_json",
    "review_status",
    "lexical_reusable",
    "lookup_source",
)


def _read_lookup(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    frame = pd.read_csv(path, keep_default_na=False, dtype=str)
    if frame.empty or "formal_token" not in frame.columns:
        return {}
    latest: dict[str, dict[str, str]] = {}
    for row in frame.to_dict("records"):
        key = lookup_key(row.get("formal_token"))
        if key:
            latest[key] = {name: clean(value) for name, value in row.items()}
    return latest


def _expected_auto_row(
    row: dict[str, str],
    seed_lookup: dict[str, dict[str, str]],
    curated_lookup: dict[str, dict[str, str]],
) -> dict[str, str] | None:
    source = clean(row.get("lookup_source"))
    key = lookup_key(row.get("formal_token"))
    lexical = curated_lookup.get(key) if source == "curated_lookup" else seed_lookup.get(key)
    if not lexical:
        return None

    if source == "name_seed_lookup":
        language, entity_type, protected = "unknown", "PERSON", "yes"
    elif source == "french_seed_lookup":
        language, entity_type, protected = "fr", "UNKNOWN", "yes"
    elif source == "seed_lookup":
        language, entity_type, protected = "wo", "UNKNOWN", "no"
    elif source == "curated_lookup":
        language = clean(lexical.get("target_language"))
        entity_type = clean(lexical.get("entity_type")) or "UNKNOWN"
        protected = clean(lexical.get("protected")) or "uncertain"
    else:
        return None

    return {
        "lemma": clean(lexical.get("lemma")) or clean(row.get("formal_token")),
        "pos": clean(lexical.get("pos")),
        "source_language": language,
        "target_language": language,
        "entity_type": entity_type,
        "protected": protected,
        "error_type": "identity",
        "error_types_json": '["identity"]',
        "transformation_steps_json": "[]",
        "review_status": "reviewed",
        "lexical_reusable": "false",
        "lookup_source": source,
    }


def classify_refresh_rows(
    annotations: pd.DataFrame,
    seed_lookup: dict[str, dict[str, str]],
    curated_lookup: dict[str, dict[str, str]],
    name_keys: set[str],
) -> tuple[pd.Series, pd.Series]:
    """Return masks for removable stale autos and preserved modified autos."""
    remove = pd.Series(False, index=annotations.index, dtype=bool)
    modified = pd.Series(False, index=annotations.index, dtype=bool)
    for index, raw in annotations.iterrows():
        row = {name: clean(value) for name, value in raw.items()}
        if not row.get("notes", "").startswith(AUTO_PREFIX):
            continue
        expected = _expected_auto_row(row, seed_lookup, curated_lookup)
        untouched = expected is not None and all(
            row.get(field, "") == expected[field] for field in COMPARISON_FIELDS
        )
        source = row.get("lookup_source")
        stale = source == "french_seed_lookup" or lookup_key(
            row.get("formal_token")
        ) in name_keys
        # A curated auto row can always be recreated from its separately saved
        # human lookup entry.  Removing it is safe even when that lookup was
        # revised after the automatic propagation occurred.
        if source == "curated_lookup" or (untouched and stale):
            remove.loc[index] = True
        elif not untouched:
            modified.loc[index] = True
    return remove, modified


def normalize_preserved_name_autos(
    annotations: pd.DataFrame, name_keys: set[str], remove: pd.Series
) -> pd.Series:
    """Complete partial manual name corrections without changing French choices.

    A reviewer may have corrected only the POS and language of an automatic
    French row before PERSON automation existed.  If both languages were set
    to unknown, the row is clearly no longer asserting French; complete its
    name fields.  Rows explicitly retained as French remain untouched (notably
    context-dependent forms such as ``Cheikh``).
    """
    migrated = pd.Series(False, index=annotations.index, dtype=bool)
    for index, raw in annotations.iterrows():
        if remove.loc[index]:
            continue
        if not clean(raw.get("notes")).startswith(AUTO_PREFIX):
            continue
        if lookup_key(raw.get("formal_token")) not in name_keys:
            continue
        if clean(raw.get("pos")) != "PROPN":
            continue
        if clean(raw.get("source_language")).casefold() != "unknown":
            continue
        if clean(raw.get("target_language")).casefold() != "unknown":
            continue
        annotations.loc[index, "source_language"] = "unknown"
        annotations.loc[index, "target_language"] = "unknown"
        annotations.loc[index, "entity_type"] = "PERSON"
        annotations.loc[index, "protected"] = "yes"
        annotations.loc[index, "lookup_source"] = "name_seed_lookup"
        annotations.loc[index, "notes"] = "AUTO_IDENTITY_NAME_LOOKUP"
        migrated.loc[index] = True
    return migrated


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def refresh(
    annotations_path: Path,
    seed_path: Path,
    curated_path: Path,
    names_path: Path,
    *,
    apply: bool,
) -> dict[str, object]:
    annotations = pd.read_csv(annotations_path, keep_default_na=False, dtype=str)
    names = {
        lookup_key(value)
        for value in names_path.read_text(encoding="utf-8-sig").splitlines()
        if lookup_key(value)
    }
    remove, modified = classify_refresh_rows(
        annotations,
        _read_lookup(seed_path),
        _read_lookup(curated_path),
        names,
    )
    migrated = normalize_preserved_name_autos(annotations, names, remove)
    stats: dict[str, object] = {
        "rows_before": len(annotations),
        "rows_to_remove": int(remove.sum()),
        "modified_auto_rows_preserved": int(modified.sum()),
        "partial_name_rows_completed": int(migrated.sum()),
        "rows_after": int((~remove).sum()),
        "removed_occurrence_ids": annotations.loc[remove, "occurrence_id"].tolist(),
    }
    if not apply:
        return stats

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = annotations_path.parent / "linguistic_annotation_archives" / stamp
    archive.mkdir(parents=True, exist_ok=False)
    archived: list[dict[str, str]] = []
    for path in (annotations_path, seed_path, curated_path, names_path):
        if not path.exists():
            continue
        destination = archive / path.name
        shutil.copy2(path, destination)
        archived.append({
            "source": str(path.resolve()),
            "archived_as": destination.name,
            "sha256": _sha256(destination),
        })

    kept = annotations.loc[~remove].copy()
    temporary = annotations_path.with_suffix(".refresh.tmp.csv")
    kept.to_csv(temporary, index=False, encoding="utf-8")
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
    parser.add_argument("--names", type=Path, default=SENEGALESE_SURNAMES_PATH)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    stats = refresh(
        args.annotations, args.seed, args.curated, args.names, apply=args.apply
    )
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
