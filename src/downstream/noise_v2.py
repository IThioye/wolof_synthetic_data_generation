"""Controlled M1 severity and applicable cross-method noise adapters."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from src.generation.benchmark_baselines import (
    BEQI_INSPIRED_RULES,
    informalize_beqi_rules,
    mask_protected,
)
from src.generation.gold_linguistic_engine import (
    LinguisticConfig,
    LinguisticProfile,
    build_token_lookup,
    generate_m6_pair,
)
from src.generation.youtube_variant_mining import (
    apply_mined_variants,
    build_mapping_index,
)


V1_HEAVY_MEAN_OPERATIONS = 9.31191222570533


@dataclass(frozen=True)
class M1Severity:
    name: str
    target_mean_operations: float
    application_probability: float
    description: str


M1_SEVERITIES = (
    M1Severity(
        "light",
        2.0,
        2.0 / V1_HEAVY_MEAN_OPERATIONS,
        "Each eligible M1 occurrence is retained by a deterministic hash threshold of 2/9.31.",
    ),
    M1Severity(
        "medium",
        5.0,
        5.0 / V1_HEAVY_MEAN_OPERATIONS,
        "Each eligible M1 occurrence is retained by a deterministic hash threshold of 5/9.31.",
    ),
    M1Severity(
        "heavy",
        V1_HEAVY_MEAN_OPERATIONS,
        1.0,
        "Exact V1 M1 generator: every ordered rule occurrence is applied.",
    ),
)


def _uniform(seed: int, row_id: str, rule_id: str, occurrence: int, match_text: str) -> float:
    payload = f"m1-severity-v2:{seed}:{row_id}:{rule_id}:{occurrence}:{match_text}"
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def informalize_m1_controlled(
    text: str,
    *,
    row_id: str,
    seed: int,
    application_probability: float,
) -> tuple[str, Counter[str]]:
    """Apply the unchanged ordered M1 rule family at controlled occurrence rate.

    Heavy delegates to the historical implementation exactly. Light/medium
    retain individual eligible occurrences using stable hash decisions; they do
    not impose a per-sentence edit quota and therefore preserve natural
    variation in how many M1 opportunities each sentence contains.
    """

    if not 0.0 <= application_probability <= 1.0:
        raise ValueError("application_probability must be in [0, 1]")
    if application_probability >= 1.0:
        return informalize_beqi_rules(text)
    if not isinstance(text, str) or not text.strip():
        return text, Counter()

    transformed, saved = mask_protected(text)
    counts: Counter[str] = Counter()
    if transformed.lower() != transformed and _uniform(
        seed, row_id, "lowercase", 0, transformed
    ) < application_probability:
        transformed = transformed.lower()
        counts["lowercase"] += 1

    for rule_id, pattern, replacement in BEQI_INSPIRED_RULES:
        if rule_id.startswith("restore_"):
            transformed = re.sub(pattern, replacement, transformed)
            continue
        occurrence = 0

        def choose(match: re.Match) -> str:
            nonlocal occurrence
            current = occurrence
            occurrence += 1
            if _uniform(seed, row_id, rule_id, current, match.group(0)) < application_probability:
                counts[rule_id] += 1
                return match.expand(replacement)
            return match.group(0)

        transformed = re.sub(pattern, choose, transformed)

    for sentinel, original in saved.items():
        transformed = transformed.replace(sentinel, original)
    return re.sub(r"\s+", " ", transformed).strip(), counts


def build_m1_severity_records(
    rows: Sequence[Mapping],
    severity: M1Severity,
    *,
    seed: int = 2026,
) -> list[dict]:
    output: list[dict] = []
    for row in rows:
        noisy, operations = informalize_m1_controlled(
            str(row["text"]),
            row_id=str(row["row_id"]),
            seed=seed,
            application_probability=severity.application_probability,
        )
        output.append(
            {
                "example_id": str(row["example_id"]),
                "row_id": str(row["row_id"]),
                "intent": str(row["intent"]),
                "split": str(row["split"]),
                "real_text": str(row["text"]),
                "noisy_text": noisy,
                "noise_method": f"m1_{severity.name}",
                "noise_operations": dict(sorted(operations.items())),
                "operation_count": int(sum(operations.values())),
                "changed": noisy != str(row["text"]),
                "seed": seed,
            }
        )
    return output


def build_m5_records(
    rows: Sequence[Mapping],
    *,
    mappings_path: str | Path = "artifacts/m5_youtube_variant_mappings.csv",
    seed: int = 2026,
    max_edits: int = 2,
) -> list[dict]:
    mappings = pd.read_csv(mappings_path)
    mapping_index = build_mapping_index(mappings)
    output: list[dict] = []
    for row in rows:
        noisy, edits = apply_mined_variants(
            str(row["text"]),
            str(row["row_id"]),
            mapping_index,
            seed=seed,
            max_edits=max_edits,
        )
        output.append(
            {
                "example_id": str(row["example_id"]),
                "row_id": str(row["row_id"]),
                "intent": str(row["intent"]),
                "split": str(row["split"]),
                "real_text": str(row["text"]),
                "noisy_text": noisy,
                "noise_method": "m5_youtube_mined_variants",
                "noise_operations": edits,
                "operation_count": len(edits),
                "changed": bool(edits),
                "seed": seed,
            }
        )
    return output


def _load_m6_profile(
    profile_path: str | Path,
    mappings_path: str | Path,
    transformations_path: str | Path,
) -> LinguisticProfile:
    raw = json.loads(Path(profile_path).read_text(encoding="utf-8"))
    budgets = tuple(
        int(value)
        for value, count in raw["edit_budget_distribution"].items()
        for _ in range(int(count))
    )
    return LinguisticProfile(
        mappings=pd.read_csv(mappings_path),
        transformations=pd.read_csv(transformations_path),
        edit_budgets=budgets,
        pos_error_profile=raw["pos_error_profile"],
        training_summary=raw["training_summary"],
    )


def build_m6_records(
    rows: Sequence[Mapping],
    *,
    artifacts_dir: str | Path = "artifacts",
    annotations_dir: str | Path = "data/annotations",
    seed: int = 2026,
) -> list[dict]:
    artifacts_dir = Path(artifacts_dir)
    annotations_dir = Path(annotations_dir)
    profile = _load_m6_profile(
        artifacts_dir / "m6_gold_linguistic_profile.json",
        artifacts_dir / "m6_gold_linguistic_mappings.csv",
        artifacts_dir / "m6_gold_linguistic_transformations.csv",
    )
    token_lookup = build_token_lookup(
        pd.read_csv(annotations_dir / "formal_token_lookup.csv"),
        pd.read_csv(annotations_dir / "formal_token_lookup_seed.csv"),
    )
    config = LinguisticConfig(max_edits_per_sentence=None, seed=seed)
    output: list[dict] = []
    for source_row, row in enumerate(rows):
        generated = generate_m6_pair(
            str(row["text"]),
            source_id=str(row["row_id"]),
            source_row=source_row,
            profile=profile,
            token_lookup=token_lookup,
            config=config,
        )
        edits = json.loads(str(generated["applied_rules"]))
        output.append(
            {
                "example_id": str(row["example_id"]),
                "row_id": str(row["row_id"]),
                "intent": str(row["intent"]),
                "split": str(row["split"]),
                "real_text": str(row["text"]),
                "noisy_text": str(generated["informal_wolof"]),
                "noise_method": "m6_gold_linguistic_engine",
                "noise_operations": edits,
                "operation_count": int(generated["edit_count"]),
                "lexical_mapping_edit_count": int(generated["lexical_mapping_edit_count"]),
                "transformation_edit_count": int(generated["transformation_edit_count"]),
                "changed": bool(generated["changed"]),
                "seed": seed,
            }
        )
    return output


def method_applicability() -> list[dict]:
    return [
        {
            "method": "M0",
            "applicable": False,
            "status": "identity_reference",
            "reason": "M0 makes no corruption and is represented by REAL.",
            "intent_preservation": "exact",
        },
        {
            "method": "M1",
            "applicable": True,
            "status": "evaluated",
            "reason": "Deterministic Beqi-inspired orthographic generator accepts arbitrary Wolof text.",
            "intent_preservation": "expected but synthetic; every V1 row was heavily modified",
        },
        {
            "method": "M2",
            "applicable": False,
            "status": "not_applicable",
            "reason": "Requires the paired French translation of each source row to constrain Eflomal substitutions; Injongo supplies no French translation.",
            "intent_preservation": "cannot be established without adding a translation model",
        },
        {
            "method": "M3",
            "applicable": False,
            "status": "not_applicable",
            "reason": "Requires paired French text for CMDR candidate scoring and positions; fabricating it would change the method and add a confound.",
            "intent_preservation": "cannot be established without adding a translation model",
        },
        {
            "method": "M4",
            "applicable": False,
            "status": "not_applicable",
            "reason": "Oolel is an external set of generated formal/informal pairs, not a conditional corruption function for Injongo rows.",
            "intent_preservation": "not transferable row-wise to Injongo",
        },
        {
            "method": "M5",
            "applicable": True,
            "status": "evaluated_low_coverage",
            "reason": "Accepted YouTube-mined lexical mappings can be applied token-wise to arbitrary formal text.",
            "intent_preservation": "likely for spelling variants, but mapping errors remain possible",
        },
        {
            "method": "M6",
            "applicable": True,
            "status": "evaluated_provisional",
            "reason": "Gold-train lexical mappings and POS/context character templates accept arbitrary Wolof text with protection rules.",
            "intent_preservation": "designed for orthographic noise; real-word/context errors remain possible",
        },
    ]
