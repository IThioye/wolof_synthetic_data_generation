"""M0 identity and M1 Beqi-inspired synthetic benchmark generators.

The M1 rule inventory is the shared spelling component previously duplicated in
the Eflomal and CMDR notebooks. It is described as Beqi-inspired because it is
not a complete reimplementation of the published Beqi corrector.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pandas as pd


METHOD_M0 = "m0_identity"
METHOD_M1 = "m1_beqi_inspired_rules"
COMMON_SOURCE_CORPUS = "french_wolof_parallel"
DEFAULT_BENCHMARK_SEED = 2026

# Order is significant: later transformations receive the output of earlier
# ones. These are the same expressions used in eflomal.ipynb and cmdr.ipynb.
BEQI_INSPIRED_RULES: tuple[tuple[str, str, str], ...] = (
    ("join_word_a_word", r"(\w+)-a-(\w+)", r"\1a\2"),
    ("hyphen_to_space", r"-", " "),
    ("reduce_double_vowel", r"([aieo])\1", r"\1"),
    ("uu_to_ou", r"uu", "__OU__"),
    ("enye_to_gn", r"ñ+", "gn"),
    ("eng_to_ng", r"ŋ+", "ng"),
    ("schwa_to_eu", r"ë+", "__EU__"),
    ("ee_acute_to_e", r"ée", "e"),
    ("soft_g_add_u", r"g([ieéèêë])", r"g__U__\1"),
    ("o_acute_to_o", r"ó", "o"),
    ("a_diacritic_to_a", r"(à|á|â|ä)", "a"),
    ("u_before_blt_to_ou", r"u([blt]+)", r"__OU__\1"),
    ("u_to_ou", r"u+", "__OU__"),
    ("q_to_kh", r"q", "kh"),
    ("x_to_kh", r"x", "kh"),
    ("c_vowel_to_th", r"c([aeiouy]+)", r"th\1"),
    ("final_cc_to_thie", r"c{2}\b", "thie"),
    ("j_eao_to_di", r"j([eaoë]{1,2})", r"di\1"),
    ("j_i_to_dj", r"j(i+)", r"dj\1"),
    ("j_u_to_dio", r"j(u+)", r"dio\1"),
    ("th_vowel_add_i", r"th([aeouy]+)", r"thi\1"),
    ("final_e_to_e_acute", r"e\b", "é"),
    ("final_n_to_ne", r"n\b", "ne"),
    ("double_g_to_g", r"gg", "g"),
    ("restore_eu_placeholder", r"__EU__", "eu"),
    ("restore_ou_placeholder", r"__OU__", "ou"),
    ("restore_u_placeholder", r"__U__", "u"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_formal_sources(path: Path, wolof_column: str = "wolof") -> pd.DataFrame:
    """Load the common formal source pool while retaining stable row identity."""
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        raw = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        raw = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported formal source format: {path.suffix}")
    if wolof_column not in raw.columns:
        raise ValueError(
            f"Expected column {wolof_column!r}; available columns: {list(raw.columns)}"
        )

    sources = pd.DataFrame(
        {
            "source_row": raw.index.astype(int),
            "formal_wolof": raw[wolof_column].fillna("").astype(str).str.strip(),
        }
    )
    sources = sources.loc[sources["formal_wolof"].ne("")].copy()
    sources["source_id"] = sources["source_row"].map(lambda row: f"parallel:{row}")
    return sources[["source_id", "source_row", "formal_wolof"]].reset_index(drop=True)


def build_identity_pairs(sources: pd.DataFrame) -> pd.DataFrame:
    """Create M0 pairs without modifying the formal source text."""
    output = sources[["source_id", "source_row", "formal_wolof"]].copy()
    output.insert(2, "source_corpus", COMMON_SOURCE_CORPUS)
    output.insert(3, "method", METHOD_M0)
    output.insert(4, "seed", DEFAULT_BENCHMARK_SEED)
    output.insert(5, "informal_wolof", output["formal_wolof"])
    output["changed"] = False
    output["edit_count"] = 0
    output["error_types"] = "[]"
    output["applied_rules"] = "[]"
    return output


def _merge_spans(spans: Iterable[tuple[int, int]], text_length: int) -> list[list[int]]:
    valid = sorted(
        (max(0, int(start)), min(text_length, int(end)))
        for start, end in spans
        if int(start) < int(end)
    )
    merged: list[list[int]] = []
    for start, end in valid:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return merged


def mask_protected(
    text: str, entity_spans: Sequence[tuple[int, int]] | None = None
) -> tuple[str, Mapping[str, str]]:
    """Mask optional NER spans, uppercase acronyms, and numeric expressions."""
    spans = list(entity_spans or [])
    spans.extend(
        (match.start(), match.end())
        for match in re.finditer(r"\b[A-ZÀ-Ý0-9]{2,}\b", text)
    )
    spans.extend(
        (match.start(), match.end())
        for match in re.finditer(r"\b\d+(?:[.,]\d+)?\b", text)
    )
    merged = _merge_spans(spans, len(text))
    saved: dict[str, str] = {}
    masked = text
    # Numeric-only sentinels are unaffected by the spelling rules.
    for index, (start, end) in enumerate(reversed(merged)):
        sentinel = f"99887766{index:08d}66778899"
        while sentinel in text:
            sentinel = f"9{sentinel}9"
        saved[sentinel] = text[start:end]
        masked = masked[:start] + sentinel + masked[end:]
    return masked, saved


def informalize_beqi_rules(
    text: str, entity_spans: Sequence[tuple[int, int]] | None = None
) -> tuple[str, Counter[str]]:
    """Apply the shared ordered M1 rule inventory and return edit metadata."""
    if not isinstance(text, str) or not text.strip():
        return text, Counter()

    transformed, saved = mask_protected(text, entity_spans)
    counts: Counter[str] = Counter()
    lowered = transformed.lower()
    if lowered != transformed:
        counts["lowercase"] += 1
    transformed = lowered

    for rule_id, pattern, replacement in BEQI_INSPIRED_RULES:
        transformed, count = re.subn(pattern, replacement, transformed)
        # Placeholder restoration is implementation bookkeeping rather than a
        # second linguistic edit, so it is excluded from per-row edit counts.
        if count and not rule_id.startswith("restore_"):
            counts[rule_id] += count

    for sentinel, original in saved.items():
        transformed = transformed.replace(sentinel, original)
    transformed = re.sub(r"\s+", " ", transformed).strip()
    return transformed, counts


def build_beqi_rule_pairs(
    sources: pd.DataFrame,
    entity_spans_by_source: Mapping[str, Sequence[tuple[int, int]]] | None = None,
) -> pd.DataFrame:
    """Create M1 standalone rules-only pairs from the common source pool."""
    spans = entity_spans_by_source or {}
    rows = []
    for source in sources.itertuples(index=False):
        informal, counts = informalize_beqi_rules(
            source.formal_wolof, spans.get(source.source_id)
        )
        rows.append(
            {
                "source_id": source.source_id,
                "source_row": source.source_row,
                "source_corpus": COMMON_SOURCE_CORPUS,
                "method": METHOD_M1,
                "seed": DEFAULT_BENCHMARK_SEED,
                "informal_wolof": informal,
                "formal_wolof": source.formal_wolof,
                "changed": informal != source.formal_wolof,
                "edit_count": sum(counts.values()),
                "error_types": json.dumps(sorted(counts), ensure_ascii=False),
                "applied_rules": json.dumps(counts, ensure_ascii=False, sort_keys=True),
            }
        )
    return pd.DataFrame(rows)


def write_dataset_and_manifest(
    dataset: pd.DataFrame,
    output_path: Path,
    manifest_path: Path,
    *,
    method: str,
    input_path: Path,
    parameters: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Write a benchmark CSV and a checksum-bearing JSON manifest."""
    output_path = Path(output_path)
    manifest_path = Path(manifest_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False, encoding="utf-8")

    manifest: dict[str, object] = {
        "method": method,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_path": str(Path(input_path).resolve()),
        "input_sha256": sha256_file(Path(input_path)),
        "output_path": str(output_path.resolve()),
        "output_sha256": sha256_file(output_path),
        "rows": len(dataset),
        "unique_sources": int(dataset["source_id"].nunique()),
        "changed_rows": int(dataset["changed"].sum()),
        "unchanged_rows": int((~dataset["changed"]).sum()),
        "parameters": dict(parameters or {}),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest
