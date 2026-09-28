"""Preparation and quality auditing for the external Oolel M4 dataset."""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.generation.benchmark_baselines import sha256_file


OOLEL_DATASET_ID = "soynade-research/Wolof-Non-Standard-Orthography"
OOLEL_REVISION = "ce17c80cff6626a91dc4e38b43de12e005bc240a"
OOLEL_LICENSE = "CC BY-SA 4.0"
METHOD_M4 = "m4_oolel_llm_external"
NONSTANDARD_COLUMN_CANDIDATES = ("non_standardized", "non_standard")
OPEN_TAG_RE = re.compile(r"<\s*NON_STANDARD\s*>", flags=re.IGNORECASE)
CLOSE_TAG_RE = re.compile(r"<\s*/\s*NON_STANDARD\s*>", flags=re.IGNORECASE)
ANY_TAG_RE = re.compile(r"<[^>]+>")


def _clean_text(value: object) -> str:
    if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def normalized_key(value: object) -> str:
    """Create a conservative key for duplicates, unchanged rows, and overlap."""
    text = unicodedata.normalize("NFKC", _clean_text(value))
    return re.sub(r"\s+", " ", text).strip().casefold()


def clean_nonstandard(value: object) -> tuple[str, str]:
    """Remove the documented wrapper and report whether it was well formed."""
    text = _clean_text(value)
    open_count = len(OPEN_TAG_RE.findall(text))
    close_count = len(CLOSE_TAG_RE.findall(text))
    cleaned = OPEN_TAG_RE.sub("", text)
    cleaned = CLOSE_TAG_RE.sub("", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if open_count == close_count == 1:
        status = "valid"
    elif open_count == close_count == 0:
        status = "absent"
    else:
        status = "malformed"
    return cleaned, status


def resolve_nonstandard_column(columns: list[str] | pd.Index) -> str:
    for candidate in NONSTANDARD_COLUMN_CANDIDATES:
        if candidate in columns:
            return candidate
    raise ValueError(
        "Could not find the non-standard Wolof column. Expected one of "
        f"{NONSTANDARD_COLUMN_CANDIDATES}; received {list(columns)}"
    )


def _edit_distance(source: str, target: str) -> int:
    try:
        from rapidfuzz.distance import Levenshtein
    except ImportError as exc:
        raise RuntimeError(
            "M4 edit counts require rapidfuzz. Install it with: pip install rapidfuzz"
        ) from exc
    return int(Levenshtein.distance(source, target))


def prepare_oolel(
    raw: pd.DataFrame,
    *,
    revision: str = OOLEL_REVISION,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return accepted common-schema pairs and an audit table for every row."""
    if "wo" not in raw.columns:
        raise ValueError(f"Oolel data needs a 'wo' column; received {list(raw.columns)}")
    nonstandard_column = resolve_nonstandard_column(raw.columns)

    rows = []
    for position, row in raw.reset_index(drop=True).iterrows():
        formal = _clean_text(row.get("wo"))
        informal, wrapper_status = clean_nonstandard(row.get(nonstandard_column))
        english = _clean_text(row.get("en"))
        formal_key = normalized_key(formal)
        informal_key = normalized_key(informal)
        reasons = []
        if not formal:
            reasons.append("empty_formal")
        if not informal:
            reasons.append("empty_informal")
        if wrapper_status == "malformed" or ANY_TAG_RE.search(informal):
            reasons.append("malformed_wrapper")
        if formal_key and formal_key == informal_key:
            reasons.append("unchanged_normalized")

        rows.append(
            {
                "source_id": f"oolel:{position}",
                "source_row": position,
                "source_corpus": f"{OOLEL_DATASET_ID}@{revision}",
                "method": METHOD_M4,
                "seed": "not_reported",
                "informal_wolof": informal,
                "formal_wolof": formal,
                "english": english,
                "changed": formal_key != informal_key,
                "edit_count": _edit_distance(informal, formal),
                "error_types": json.dumps(
                    ["external_llm_generation_unspecified"], ensure_ascii=False
                ),
                "applied_rules": "{}",
                "wrapper_status": wrapper_status,
                "formal_key": formal_key,
                "informal_key": informal_key,
                "rejection_reasons": reasons,
            }
        )

    audit = pd.DataFrame(rows)
    duplicate_pair = audit.duplicated(["formal_key", "informal_key"], keep="first")
    for index in audit.index[duplicate_pair]:
        audit.at[index, "rejection_reasons"] = [
            *audit.at[index, "rejection_reasons"],
            "duplicate_pair",
        ]
    audit["accepted"] = audit["rejection_reasons"].map(len).eq(0)
    audit["rejection_reasons"] = audit["rejection_reasons"].map(
        lambda reasons: json.dumps(reasons, ensure_ascii=False)
    )

    accepted_columns = [
        "source_id",
        "source_row",
        "source_corpus",
        "method",
        "seed",
        "informal_wolof",
        "formal_wolof",
        "english",
        "changed",
        "edit_count",
        "error_types",
        "applied_rules",
    ]
    accepted = audit.loc[audit["accepted"], accepted_columns].reset_index(drop=True)
    return accepted, audit


def add_overlap_flags(
    audit: pd.DataFrame,
    *,
    common_formal: pd.Series,
    gold_formal: pd.Series | None = None,
    gold_informal: pd.Series | None = None,
) -> pd.DataFrame:
    """Flag exact normalized overlap without using gold to alter M4 text."""
    flagged = audit.copy()
    common_keys = {normalized_key(value) for value in common_formal if normalized_key(value)}
    gold_formal_keys = {
        normalized_key(value) for value in (gold_formal if gold_formal is not None else [])
        if normalized_key(value)
    }
    gold_informal_keys = {
        normalized_key(value) for value in (gold_informal if gold_informal is not None else [])
        if normalized_key(value)
    }
    flagged["overlap_common_formal"] = flagged["formal_key"].isin(common_keys)
    flagged["overlap_current_gold_formal"] = flagged["formal_key"].isin(gold_formal_keys)
    flagged["overlap_current_gold_informal"] = flagged["informal_key"].isin(
        gold_informal_keys
    )
    return flagged


def rejection_counts(audit: pd.DataFrame) -> Counter[str]:
    counts: Counter[str] = Counter()
    for encoded in audit.loc[~audit["accepted"], "rejection_reasons"]:
        counts.update(json.loads(encoded))
    return counts


def write_m4_outputs(
    accepted: pd.DataFrame,
    audit: pd.DataFrame,
    *,
    raw_path: Path,
    output_path: Path,
    rejected_path: Path,
    review_path: Path,
    manifest_path: Path,
    review_size: int = 100,
    review_seed: int = 2026,
) -> dict[str, object]:
    """Export M4, all rejected rows, a manual-review sheet, and a manifest."""
    for path in (output_path, rejected_path, review_path, manifest_path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    accepted.to_csv(output_path, index=False, encoding="utf-8")
    audit.loc[~audit["accepted"]].to_csv(rejected_path, index=False, encoding="utf-8")

    sample_size = min(review_size, len(accepted))
    review = accepted.sample(n=sample_size, random_state=review_seed).copy()
    review["meaning_preserved"] = ""
    review["informal_plausibility_1_5"] = ""
    review["code_switch_natural_1_5"] = ""
    review["synthetic_artifact"] = ""
    review["review_notes"] = ""
    review.to_csv(review_path, index=False, encoding="utf-8-sig")

    manifest = {
        "method": METHOD_M4,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_id": OOLEL_DATASET_ID,
        "revision": OOLEL_REVISION,
        "license": OOLEL_LICENSE,
        "raw_path": str(Path(raw_path).resolve()),
        "raw_sha256": sha256_file(Path(raw_path)),
        "output_path": str(Path(output_path).resolve()),
        "output_sha256": sha256_file(Path(output_path)),
        "raw_rows": len(audit),
        "accepted_rows": len(accepted),
        "rejected_rows": int((~audit["accepted"]).sum()),
        "changed_rows": int(accepted["changed"].sum()),
        "rejection_counts": dict(rejection_counts(audit)),
        "wrapper_status_counts": audit["wrapper_status"].value_counts().to_dict(),
        "exact_common_formal_overlap": int(
            audit.get("overlap_common_formal", pd.Series(dtype=bool)).sum()
        ),
        "exact_current_gold_formal_overlap": int(
            audit.get("overlap_current_gold_formal", pd.Series(dtype=bool)).sum()
        ),
        "exact_current_gold_informal_overlap": int(
            audit.get("overlap_current_gold_informal", pd.Series(dtype=bool)).sum()
        ),
        "manual_review_rows": sample_size,
        "manual_review_seed": review_seed,
        "limitations": [
            "External formal source corpus; not automatically common-source controlled.",
            "Original generation seed, prompt, decoding parameters, and row-level rules are not reported.",
            "Automatic validation cannot establish semantic preservation or naturalness.",
            "Gold-overlap checks must be repeated after the final split is locked.",
        ],
    }
    Path(manifest_path).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest
