"""M6 standalone gold-train linguistic synthetic generator.

M6 learns lexical variants, character-edit templates, POS-conditioned error
frequencies, protection decisions, and sentence edit budgets exclusively from
the finalized gold-training linguistic annotations. It intentionally consumes
neither M5 mappings nor M1 rules; combinations belong in separately named
ablation conditions.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from src.generation.benchmark_baselines import (
    COMMON_SOURCE_CORPUS,
    DEFAULT_BENCHMARK_SEED,
    sha256_file,
)
from src.generation.youtube_variant_mining import WORD_RE, word_tokens


METHOD_M6 = "m6_gold_linguistic_engine"
GENERALIZABLE_ERROR_TYPES = (
    "diacritic",
    "deletion",
    "insertion",
    "substitution",
    "repetition",
)
TRUE_VALUES = {"1", "true", "yes", "y"}

MAPPING_COLUMNS = (
    "formal_token",
    "informal_token",
    "count",
    "formal_total",
    "dominance",
    "confidence",
    "mapping_source",
    "pos_candidates_json",
    "error_types_json",
)
TRANSFORMATION_COLUMNS = (
    "pos",
    "error_type",
    "operation",
    "formal_text",
    "informal_text",
    "left_context",
    "right_context",
    "position_kind",
    "count",
    "trigger_total",
    "dominance",
    "confidence",
    "error_weight",
    "mapping_source",
)


@dataclass(frozen=True)
class LinguisticConfig:
    min_mapping_count: int = 1
    min_mapping_dominance: float = 0.50
    min_transformation_count: int = 2
    min_transformation_dominance: float = 0.60
    # None removes the artificial cap while retaining the empirical number of
    # changed occurrences observed per gold-training sentence.
    max_edits_per_sentence: int | None = 3
    seed: int = DEFAULT_BENCHMARK_SEED
    allowed_target_language: str = "wo"
    allowed_source_language: str = "wo"
    generalizable_error_types: tuple[str, ...] = GENERALIZABLE_ERROR_TYPES


@dataclass
class LinguisticProfile:
    mappings: pd.DataFrame
    transformations: pd.DataFrame
    edit_budgets: tuple[int, ...]
    pos_error_profile: dict[str, dict[str, object]]
    training_summary: dict[str, object]


def _clean(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return unicodedata.normalize("NFC", " ".join(str(value).strip().split()))


def _key(value: object) -> str:
    return _clean(value).casefold()


def _truthy(value: object) -> bool:
    return _key(value) in TRUE_VALUES


def _json_list(value: object, fallback: object = "") -> list[str]:
    text = _clean(value)
    values: list[str] = []
    if text:
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = []
        if isinstance(parsed, list):
            values = [_key(item) for item in parsed if _key(item)]
    fallback_value = _key(fallback)
    if not values and fallback_value:
        values = [fallback_value]
    return list(dict.fromkeys(values))


def _parse_steps(value: object) -> list[dict[str, object]]:
    text = _clean(value)
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def _empty_frame(columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=list(columns))


def _validate_annotations(
    annotations: pd.DataFrame,
    gold_train_source_ids: Sequence[object],
) -> pd.DataFrame:
    required = {
        "occurrence_id",
        "source_index",
        "informal_token",
        "formal_token",
        "pos",
        "source_language",
        "target_language",
        "protected",
        "error_type",
        "error_types_json",
        "transformation_steps_json",
        "review_status",
        "lexical_reusable",
    }
    missing = required - set(annotations.columns)
    if missing:
        raise ValueError(f"Linguistic annotations are missing columns: {sorted(missing)}")

    latest = annotations.copy()
    latest["occurrence_id"] = latest["occurrence_id"].map(_clean)
    latest = latest.loc[latest["occurrence_id"].ne("")]
    latest = latest.drop_duplicates("occurrence_id", keep="last").reset_index(drop=True)
    allowed_ids = {_clean(value) for value in gold_train_source_ids}
    source_keys = latest["source_index"].map(_clean)
    outside = sorted(set(source_keys) - allowed_ids)
    if outside:
        raise ValueError(
            "M6 annotations contain source IDs outside locked gold train: "
            + ", ".join(outside[:10])
        )
    return latest


def _eligible_rows(
    annotations: pd.DataFrame,
    config: LinguisticConfig,
) -> pd.DataFrame:
    rows = annotations.copy()
    rows["formal_key"] = rows["formal_token"].map(_key)
    rows["informal_key"] = rows["informal_token"].map(_key)
    rows["pos_key"] = rows["pos"].map(lambda value: _clean(value).upper() or "UNKNOWN")
    rows["changed"] = rows["formal_key"].ne(rows["informal_key"])
    return rows.loc[
        rows["review_status"].map(_key).eq("reviewed")
        & rows["target_language"].map(_key).eq(config.allowed_target_language)
        & rows["source_language"].map(_key).eq(config.allowed_source_language)
        & rows["protected"].map(_key).eq("no")
        & rows["formal_key"].ne("")
        & rows["informal_key"].ne("")
    ].copy()


def _learn_pos_error_profile(rows: pd.DataFrame) -> dict[str, dict[str, object]]:
    profile: dict[str, dict[str, object]] = {}
    for pos, group in rows.groupby("pos_key", sort=True):
        changed = group.loc[group["changed"]]
        counts: Counter[str] = Counter()
        for row in changed.itertuples(index=False):
            counts.update(
                error for error in _json_list(row.error_types_json, row.error_type)
                if error not in {"identity", "uncertain", "multiple"}
            )
        total_errors = sum(counts.values())
        profile[str(pos)] = {
            "eligible_occurrences": int(len(group)),
            "changed_occurrences": int(len(changed)),
            "change_rate": float(len(changed) / len(group)) if len(group) else 0.0,
            "error_counts": dict(sorted(counts.items())),
            "error_probabilities": {
                error: float(count / total_errors)
                for error, count in sorted(counts.items())
            } if total_errors else {},
        }
    return profile


def _learn_mappings(rows: pd.DataFrame, config: LinguisticConfig) -> pd.DataFrame:
    buckets: dict[tuple[str, str], dict[str, object]] = {}
    formal_totals: Counter[str] = Counter()
    for row in rows.loc[rows["changed"]].itertuples(index=False):
        errors = _json_list(row.error_types_json, row.error_type)
        reusable = _truthy(row.lexical_reusable) or "spacing_merge" in errors
        if not reusable:
            continue
        pair = (row.formal_key, row.informal_key)
        bucket = buckets.setdefault(
            pair,
            {
                "formal_token": _clean(row.formal_token),
                "informal_token": _clean(row.informal_token),
                "count": 0,
                "pos": set(),
                "errors": set(),
            },
        )
        bucket["count"] = int(bucket["count"]) + 1
        bucket["pos"].add(row.pos_key)
        bucket["errors"].update(errors)
        formal_totals[row.formal_key] += 1

    records = []
    for (formal_key, _), bucket in buckets.items():
        count = int(bucket["count"])
        formal_total = int(formal_totals[formal_key])
        dominance = count / formal_total if formal_total else 0.0
        if count < config.min_mapping_count or dominance < config.min_mapping_dominance:
            continue
        records.append(
            {
                "formal_token": bucket["formal_token"],
                "informal_token": bucket["informal_token"],
                "count": count,
                "formal_total": formal_total,
                "dominance": dominance,
                "confidence": dominance,
                "mapping_source": "gold_train_linguistic_mapping",
                "pos_candidates_json": json.dumps(sorted(bucket["pos"]), ensure_ascii=False),
                "error_types_json": json.dumps(sorted(bucket["errors"]), ensure_ascii=False),
            }
        )
    if not records:
        return _empty_frame(MAPPING_COLUMNS)
    return pd.DataFrame(records, columns=MAPPING_COLUMNS).sort_values(
        ["count", "dominance", "formal_token", "informal_token"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)


def _position_kind(start: int, end: int, length: int) -> str:
    if start == 0:
        return "prefix"
    if end == length:
        return "suffix"
    return "internal"


def _primary_error(row: object, allowed: set[str]) -> str:
    primary = _key(getattr(row, "error_type", ""))
    if primary in allowed:
        return primary
    return next(
        (
            value for value in _json_list(
                getattr(row, "error_types_json", ""), primary
            ) if value in allowed
        ),
        "",
    )


def _learn_transformations(
    rows: pd.DataFrame,
    pos_profile: Mapping[str, Mapping[str, object]],
    config: LinguisticConfig,
) -> pd.DataFrame:
    allowed = set(config.generalizable_error_types)
    buckets: Counter[tuple[str, ...]] = Counter()
    trigger_totals: Counter[tuple[str, ...]] = Counter()
    for row in rows.loc[rows["changed"]].itertuples(index=False):
        formal = row.formal_key
        if " " in formal:
            continue
        error_type = _primary_error(row, allowed)
        if not error_type:
            continue
        for step in _parse_steps(row.transformation_steps_json):
            operation = _key(step.get("operation"))
            if operation not in {"insert", "delete", "replace"}:
                continue
            formal_text = _key(step.get("formal_text"))
            informal_text = _key(step.get("informal_text"))
            if formal_text == informal_text or (not formal_text and not informal_text):
                continue
            try:
                start = int(step.get("formal_start", 0))
                end = int(step.get("formal_end", start))
            except (TypeError, ValueError):
                continue
            if start < 0 or end < start or end > len(formal):
                continue
            left = formal[start - 1 : start] if start else "^"
            right = formal[end : end + 1] if end < len(formal) else "$"
            position = _position_kind(start, end, len(formal))
            signature = (
                row.pos_key,
                error_type,
                operation,
                formal_text,
                informal_text,
                left,
                right,
                position,
            )
            trigger = (
                row.pos_key,
                error_type,
                operation,
                formal_text,
                left,
                right,
                position,
            )
            buckets[signature] += 1
            trigger_totals[trigger] += 1

    records = []
    for signature, count in buckets.items():
        pos, error_type, operation, formal_text, informal_text, left, right, position = signature
        trigger = (pos, error_type, operation, formal_text, left, right, position)
        total = trigger_totals[trigger]
        dominance = count / total if total else 0.0
        if (
            count < config.min_transformation_count
            or dominance < config.min_transformation_dominance
        ):
            continue
        probabilities = pos_profile.get(pos, {}).get("error_probabilities", {})
        error_weight = float(probabilities.get(error_type, 0.0))
        records.append(
            {
                "pos": pos,
                "error_type": error_type,
                "operation": operation,
                "formal_text": formal_text,
                "informal_text": informal_text,
                "left_context": left,
                "right_context": right,
                "position_kind": position,
                "count": int(count),
                "trigger_total": int(total),
                "dominance": float(dominance),
                "confidence": float(dominance),
                "error_weight": error_weight,
                "mapping_source": "gold_train_linguistic_transformation",
            }
        )
    if not records:
        return _empty_frame(TRANSFORMATION_COLUMNS)
    return pd.DataFrame(records, columns=TRANSFORMATION_COLUMNS).sort_values(
        ["count", "dominance", "pos", "error_type"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)


def learn_linguistic_profile(
    annotations: pd.DataFrame,
    gold_train_source_ids: Sequence[object],
    *,
    config: LinguisticConfig = LinguisticConfig(),
) -> LinguisticProfile:
    """Learn the complete M6 profile from locked gold-train annotations only."""
    latest = _validate_annotations(annotations, gold_train_source_ids)
    eligible = _eligible_rows(latest, config)
    pos_profile = _learn_pos_error_profile(eligible)
    mappings = _learn_mappings(eligible, config)
    transformations = _learn_transformations(eligible, pos_profile, config)

    ids = [_clean(value) for value in gold_train_source_ids]
    changed_counts = (
        eligible.loc[eligible["changed"]].groupby(
            eligible.loc[eligible["changed"], "source_index"].map(_clean)
        ).size().to_dict()
    )
    budgets = tuple(
        (
            int(changed_counts.get(source_id, 0))
            if config.max_edits_per_sentence is None
            else min(
                config.max_edits_per_sentence,
                int(changed_counts.get(source_id, 0)),
            )
        )
        for source_id in ids
    )
    summary = {
        "annotation_rows": int(len(latest)),
        "eligible_occurrences": int(len(eligible)),
        "eligible_changed_occurrences": int(eligible["changed"].sum()),
        "gold_train_source_count": int(len(set(ids))),
        "mapping_count": int(len(mappings)),
        "transformation_count": int(len(transformations)),
        "uses_m5_outputs": False,
        "uses_m1_rules": False,
    }
    return LinguisticProfile(
        mappings=mappings,
        transformations=transformations,
        edit_budgets=budgets,
        pos_error_profile=pos_profile,
        training_summary=summary,
    )


def build_token_lookup(
    curated_lookup: pd.DataFrame,
    seed_lookup: pd.DataFrame | None = None,
) -> dict[str, dict[str, object]]:
    """Build conservative source-token properties; curated rows override seed rows."""
    required = {"formal_token", "pos", "target_language", "protected"}
    if not curated_lookup.empty:
        missing = required - set(curated_lookup.columns)
        if missing:
            raise ValueError(f"Curated lookup is missing columns: {sorted(missing)}")
    seed = seed_lookup if seed_lookup is not None else pd.DataFrame()
    if not seed.empty:
        missing = required - set(seed.columns)
        if missing:
            raise ValueError(f"Seed lookup is missing columns: {sorted(missing)}")

    result: dict[str, dict[str, object]] = {}
    curated = curated_lookup.copy()
    if not curated.empty:
        curated["key"] = curated["formal_token"].map(_key)
        if "review_status" in curated.columns:
            reviewed = curated["review_status"].map(_key).eq("reviewed")
            curated = curated.loc[reviewed]
        for key, group in curated.loc[curated["key"].ne("")].groupby("key", sort=False):
            row = group.iloc[-1]
            result[key] = {
                "pos": {_clean(row["pos"]).upper() or "UNKNOWN"},
                "target_language": _key(row["target_language"]) or "unknown",
                "entity_type": _clean(row.get("entity_type", "UNKNOWN")).upper(),
                "protected": _key(row["protected"]) or "uncertain",
                "source": "curated_gold_train_lookup",
            }

    if seed.empty:
        return result
    seed = seed.copy()
    seed["key"] = seed["formal_token"].map(_key)
    for key, group in seed.loc[seed["key"].ne("")].groupby("key", sort=False):
        if key in result:
            continue
        pos_values = {_clean(value).upper() or "UNKNOWN" for value in group["pos"]}
        languages = {_key(value) or "unknown" for value in group["target_language"]}
        protected_values = {_key(value) or "uncertain" for value in group["protected"]}
        ambiguous = len(pos_values) != 1 or len(languages) != 1
        if "ambiguous_pos" in group.columns:
            ambiguous = ambiguous or any(_truthy(value) for value in group["ambiguous_pos"])
        if "language_ambiguous" in group.columns:
            ambiguous = ambiguous or any(_truthy(value) for value in group["language_ambiguous"])
        result[key] = {
            "pos": pos_values,
            "target_language": next(iter(languages)) if len(languages) == 1 else "unknown",
            "entity_type": (
                _clean(group.iloc[-1].get("entity_type", "UNKNOWN")).upper()
                or "UNKNOWN"
            ),
            "protected": "yes" if ambiguous else (
                "yes" if "yes" in protected_values else next(iter(protected_values))
            ),
            "source": "multilingual_seed_lookup",
        }
    return result


def _stable_rng(seed: int, source_id: str) -> random.Random:
    digest = hashlib.sha256(f"m6-isolated:{seed}:{source_id}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _mapping_index(mappings: pd.DataFrame) -> dict[tuple[str, ...], list[dict[str, object]]]:
    index: dict[tuple[str, ...], list[dict[str, object]]] = defaultdict(list)
    for row in mappings.itertuples(index=False):
        key = tuple(word_tokens(row.formal_token))
        if not key:
            continue
        index[key].append(
            {
                "informal": _clean(row.informal_token),
                "confidence": float(row.confidence),
                "count": int(row.count),
                "error_types": _json_list(row.error_types_json),
                "source": "gold_train_linguistic_mapping",
            }
        )
    return dict(index)


def _transformation_index(transformations: pd.DataFrame) -> dict[str, list[object]]:
    index: dict[str, list[object]] = defaultdict(list)
    for transformation in transformations.itertuples(index=False):
        index[str(transformation.pos)].append(transformation)
    return dict(index)


def _position_matches(start: int, end: int, length: int, expected: str) -> bool:
    return _position_kind(start, end, length) == expected


def _apply_template(token: str, template: object) -> str | None:
    normalized = _key(token)
    formal_text = _key(template.formal_text)
    informal_text = _key(template.informal_text)
    operation = _key(template.operation)
    candidates: list[tuple[int, int]] = []
    if operation == "insert":
        candidates = [(position, position) for position in range(len(normalized) + 1)]
    elif formal_text:
        start = normalized.find(formal_text)
        while start >= 0:
            candidates.append((start, start + len(formal_text)))
            start = normalized.find(formal_text, start + 1)
    for start, end in candidates:
        left = normalized[start - 1 : start] if start else "^"
        right = normalized[end : end + 1] if end < len(normalized) else "$"
        if left != _clean(template.left_context) or right != _clean(template.right_context):
            continue
        if not _position_matches(start, end, len(normalized), _key(template.position_kind)):
            continue
        transformed = normalized[:start] + informal_text + normalized[end:]
        if transformed != normalized:
            return transformed
    return None


def _token_is_protected(
    token: str,
    properties: Mapping[str, object] | None,
    *,
    token_position: int,
) -> bool:
    if len(token) > 1 and token.isupper():
        return True
    # Sentence-medial title case is a conservative entity guard. Sentence-initial
    # capitalization remains eligible because it is normal orthography.
    if token_position > 0 and token.istitle():
        return True
    if properties is None:
        return False
    return (
        _key(properties.get("protected")) != "no"
        or _key(properties.get("target_language")) != "wo"
        or _key(properties.get("entity_type")) not in {"", "unknown", "none"}
    )


def _candidate_weight(candidate: Mapping[str, object]) -> float:
    confidence = max(0.01, float(candidate.get("confidence", 0.0)))
    count = max(1, int(candidate.get("count", 1)))
    error_weight = max(0.01, float(candidate.get("error_weight", 1.0)))
    lexical_bonus = 2.0 if candidate.get("source") == "gold_train_linguistic_mapping" else 1.0
    return lexical_bonus * confidence * math.log1p(count) * error_weight


def generate_m6_pair(
    formal: str,
    *,
    source_id: str,
    source_row: int,
    profile: LinguisticProfile,
    token_lookup: Mapping[str, Mapping[str, object]],
    config: LinguisticConfig = LinguisticConfig(),
    _mapping_index_cache: Mapping[tuple[str, ...], Sequence[Mapping[str, object]]] | None = None,
    _transformation_index_cache: Mapping[str, Sequence[object]] | None = None,
) -> dict[str, object]:
    """Generate one isolated M6 example with deterministic train-learned edits."""
    matches = list(WORD_RE.finditer(formal))
    tokens = [match.group(0) for match in matches]
    keys = [_key(token) for token in tokens]
    properties = [token_lookup.get(key) for key in keys]
    protected_positions = {
        index for index, token in enumerate(tokens)
        if _token_is_protected(
            token, properties[index], token_position=index
        )
    }
    candidates: list[dict[str, object]] = []

    mapping_index = _mapping_index_cache or _mapping_index(profile.mappings)
    max_phrase = max((len(key) for key in mapping_index), default=1)
    for size in range(max_phrase, 0, -1):
        for start in range(len(keys) - size + 1):
            positions = set(range(start, start + size))
            if positions & protected_positions:
                continue
            variants = mapping_index.get(tuple(keys[start : start + size]), [])
            for variant in variants:
                candidates.append(
                    {
                        "word_start": start,
                        "word_end": start + size,
                        "char_start": matches[start].start(),
                        "char_end": matches[start + size - 1].end(),
                        "formal": formal[matches[start].start() : matches[start + size - 1].end()],
                        **variant,
                        "error_weight": 1.0,
                    }
                )

    transformations_by_pos = (
        _transformation_index_cache or _transformation_index(profile.transformations)
    )
    for index, token in enumerate(tokens):
        if index in protected_positions:
            continue
        token_properties = properties[index]
        if token_properties is None:
            continue
        for pos in token_properties.get("pos", {"UNKNOWN"}):
            for template in transformations_by_pos.get(str(pos), []):
                informal = _apply_template(token, template)
                if informal is None:
                    continue
                candidates.append(
                    {
                        "word_start": index,
                        "word_end": index + 1,
                        "char_start": matches[index].start(),
                        "char_end": matches[index].end(),
                        "formal": token,
                        "informal": informal,
                        "confidence": float(template.confidence),
                        "count": int(template.count),
                        "error_weight": float(template.error_weight),
                        "error_types": [str(template.error_type)],
                        "source": "gold_train_linguistic_transformation",
                        "operation": str(template.operation),
                        "formal_text": str(template.formal_text),
                        "informal_text": str(template.informal_text),
                        "left_context": str(template.left_context),
                        "right_context": str(template.right_context),
                        "position_kind": str(template.position_kind),
                        "pos": str(template.pos),
                    }
                )

    rng = _stable_rng(config.seed, source_id)
    budget = rng.choice(profile.edit_budgets) if profile.edit_budgets else 0
    budget = max(0, int(budget))
    if config.max_edits_per_sentence is not None:
        budget = min(config.max_edits_per_sentence, budget)
    if budget == 0:
        candidates = []
    for candidate in candidates:
        weight = _candidate_weight(candidate)
        candidate["selection_key"] = rng.random() ** (1.0 / max(weight, 1e-9))
    candidates.sort(
        key=lambda row: (
            -float(row["selection_key"]),
            int(row["word_start"]),
            str(row["informal"]),
        )
    )

    selected: list[dict[str, object]] = []
    occupied: set[int] = set()
    for candidate in candidates:
        positions = set(range(int(candidate["word_start"]), int(candidate["word_end"])))
        if positions & occupied:
            continue
        selected.append({key: value for key, value in candidate.items() if key != "selection_key"})
        occupied.update(positions)
        if len(selected) >= budget:
            break

    informal = formal
    for edit in sorted(selected, key=lambda row: int(row["char_start"]), reverse=True):
        start, end = int(edit["char_start"]), int(edit["char_end"])
        informal = informal[:start] + str(edit["informal"]) + informal[end:]
    informal = re.sub(r"\s+", " ", informal).strip()
    error_types = sorted(
        {
            error
            for edit in selected
            for error in edit.get("error_types", [])
            if error
        }
    )
    lexical_count = sum(
        edit["source"] == "gold_train_linguistic_mapping" for edit in selected
    )
    template_count = sum(
        edit["source"] == "gold_train_linguistic_transformation" for edit in selected
    )
    serialized = json.dumps(selected, ensure_ascii=False, sort_keys=True)
    return {
        "source_id": source_id,
        "source_row": source_row,
        "source_corpus": COMMON_SOURCE_CORPUS,
        "method": METHOD_M6,
        "seed": config.seed,
        "informal_wolof": informal,
        "formal_wolof": formal,
        "changed": informal != formal,
        "edit_count": len(selected),
        "linguistic_edit_count": len(selected),
        "lexical_mapping_edit_count": lexical_count,
        "transformation_edit_count": template_count,
        "error_types": json.dumps(error_types, ensure_ascii=False),
        "applied_rules": serialized,
        "applied_transformations": serialized,
    }


def build_m6_pairs(
    sources: pd.DataFrame,
    profile: LinguisticProfile,
    token_lookup: Mapping[str, Mapping[str, object]],
    *,
    config: LinguisticConfig = LinguisticConfig(),
) -> pd.DataFrame:
    required = {"source_id", "source_row", "formal_wolof"}
    missing = required - set(sources.columns)
    if missing:
        raise ValueError(f"Formal sources are missing columns: {sorted(missing)}")
    mapping_index = _mapping_index(profile.mappings)
    transformation_index = _transformation_index(profile.transformations)
    return pd.DataFrame(
        [
            generate_m6_pair(
                str(row.formal_wolof),
                source_id=str(row.source_id),
                source_row=int(row.source_row),
                profile=profile,
                token_lookup=token_lookup,
                config=config,
                _mapping_index_cache=mapping_index,
                _transformation_index_cache=transformation_index,
            )
            for row in sources.itertuples(index=False)
        ]
    )


def write_m6_outputs(
    pairs: pd.DataFrame,
    profile: LinguisticProfile,
    *,
    output_path: Path,
    mappings_path: Path,
    transformations_path: Path,
    profile_path: Path,
    review_path: Path,
    manifest_path: Path,
    input_paths: Mapping[str, Path],
    config: LinguisticConfig,
    review_size: int = 100,
) -> dict[str, object]:
    outputs = (
        output_path,
        mappings_path,
        transformations_path,
        profile_path,
        review_path,
        manifest_path,
    )
    for path in outputs:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(output_path, index=False, encoding="utf-8")
    profile.mappings.to_csv(mappings_path, index=False, encoding="utf-8")
    profile.transformations.to_csv(transformations_path, index=False, encoding="utf-8")
    profile_payload = {
        "method": METHOD_M6,
        "configuration": asdict(config),
        "training_summary": profile.training_summary,
        "edit_budget_distribution": {
            str(key): int(value) for key, value in sorted(Counter(profile.edit_budgets).items())
        },
        "pos_error_profile": profile.pos_error_profile,
    }
    Path(profile_path).write_text(
        json.dumps(profile_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    changed = pairs.loc[pairs["changed"]]
    review = changed.sample(min(review_size, len(changed)), random_state=config.seed).copy()
    review["meaning_preserved"] = ""
    review["informal_plausibility_1_5"] = ""
    review["severity_1_5"] = ""
    review["synthetic_artifact"] = ""
    review["review_notes"] = ""
    review.to_csv(review_path, index=False, encoding="utf-8-sig")

    manifest = {
        "method": METHOD_M6,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "provisional": True,
        "learning_scope": "finalized locked gold train linguistic annotations only",
        "uses_m5_outputs": False,
        "uses_m1_rules": False,
        "leakage_status": "no development/test annotations or M5 outputs consumed",
        "rows": int(len(pairs)),
        "changed_rows": int(pairs["changed"].sum()),
        "unchanged_rows": int((~pairs["changed"]).sum()),
        "lexical_mapping_edits": int(pairs["lexical_mapping_edit_count"].sum()),
        "transformation_edits": int(pairs["transformation_edit_count"].sum()),
        "configuration": asdict(config),
        "training_summary": profile.training_summary,
        "output_path": str(Path(output_path).resolve()),
        "output_sha256": sha256_file(Path(output_path)),
        "review_path": str(Path(review_path).resolve()),
        "manual_review_rows": int(len(review)),
        "artifacts": {
            "mappings": {
                "path": str(Path(mappings_path).resolve()),
                "sha256": sha256_file(Path(mappings_path)),
            },
            "transformations": {
                "path": str(Path(transformations_path).resolve()),
                "sha256": sha256_file(Path(transformations_path)),
            },
            "profile": {
                "path": str(Path(profile_path).resolve()),
                "sha256": sha256_file(Path(profile_path)),
            },
        },
        "inputs": {
            name: {"path": str(Path(path).resolve()), "sha256": sha256_file(Path(path))}
            for name, path in input_paths.items()
        },
        "finalization_required": [
            "Complete all M6 manual review fields.",
            "Finalize the M6 manifest before M7 training.",
        ],
    }
    Path(manifest_path).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def finalize_m6_manifest(manifest_path: Path, review_path: Path) -> dict[str, object]:
    review = pd.read_csv(review_path)
    required = {
        "meaning_preserved",
        "informal_plausibility_1_5",
        "severity_1_5",
        "synthetic_artifact",
        "review_notes",
    }
    missing = required - set(review.columns)
    if missing:
        raise ValueError(f"M6 review is missing columns: {sorted(missing)}")
    for column in required - {"review_notes"}:
        if review[column].fillna("").astype(str).str.strip().eq("").any():
            raise ValueError(f"M6 review is incomplete: {column}")
    plausibility = pd.to_numeric(review["informal_plausibility_1_5"], errors="raise")
    severity = pd.to_numeric(review["severity_1_5"], errors="raise")
    if not plausibility.between(1, 5).all() or not severity.between(1, 5).all():
        raise ValueError("M6 plausibility and severity ratings must be between 1 and 5")

    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    manifest["provisional"] = False
    manifest["review_sha256"] = sha256_file(Path(review_path))
    manifest["review_mean_plausibility"] = float(plausibility.mean())
    manifest["review_mean_severity"] = float(severity.mean())
    manifest["finalized_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["finalization_required"] = []
    Path(manifest_path).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest
