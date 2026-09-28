"""Benchmark-ready M2 Eflomal and M3 CMDR generation utilities."""

from __future__ import annotations

import json
import math
import pickle
import random
import re
import zlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from src.generation.benchmark_baselines import (
    COMMON_SOURCE_CORPUS,
    DEFAULT_BENCHMARK_SEED,
    informalize_beqi_rules,
    mask_protected,
    sha256_file,
)


METHOD_M2 = "m2_eflomal_plus_beqi_rules"
METHOD_M3 = "m3_cmdr_plus_beqi_rules"
WORD_RE = re.compile(r"[^\W\d_]+(?:[-’'][^\W\d_]+)*", flags=re.UNICODE)


def stable_token_hash(value: str) -> int:
    """Return a process-independent hash for reproducible Gensim initialization."""
    return zlib.crc32(value.encode("utf-8"))


def word_tokens(text: object) -> list[str]:
    if text is None or pd.isna(text):
        return []
    return [match.group(0).casefold() for match in WORD_RE.finditer(str(text))]


def make_ngrams(tokens: Sequence[str], max_n: int) -> list[str]:
    return [
        "_".join(tokens[start : start + size])
        for size in range(1, max_n + 1)
        for start in range(len(tokens) - size + 1)
    ]


def _restore(text: str, saved: Mapping[str, str]) -> str:
    for sentinel, value in saved.items():
        text = text.replace(sentinel, value)
    return text


def _sentinel(index: int, occupied: str) -> str:
    value = f"88776655{index:08d}55667788"
    while value in occupied:
        value = f"8{value}8"
    return value


def _apply_substitutions_and_rules(
    masked_text: str,
    substitutions: Sequence[dict[str, object]],
    protected_values: Mapping[str, str],
) -> tuple[str, Counter[str]]:
    """Protect inserted French, apply the exact shared M1 rules, then restore."""
    working = masked_text
    inserted: dict[str, str] = {}
    for index, substitution in enumerate(
        sorted(substitutions, key=lambda row: int(row["char_start"]), reverse=True)
    ):
        start = int(substitution["char_start"])
        end = int(substitution["char_end"])
        sentinel = _sentinel(index, working)
        inserted[sentinel] = str(substitution["target_french"])
        working = working[:start] + sentinel + working[end:]

    informal, rule_counts = informalize_beqi_rules(working)
    informal = _restore(informal, inserted)
    informal = _restore(informal, protected_values)
    return re.sub(r"\s+", " ", informal).strip(), rule_counts


def _common_row(
    *,
    source_id: str,
    source_row: int,
    formal: str,
    informal: str,
    method: str,
    substitutions: list[dict[str, object]],
    rule_counts: Counter[str],
    seed: int,
) -> dict[str, object]:
    error_types = []
    if substitutions:
        error_types.append("alignment_code_switch")
    if rule_counts:
        error_types.append("beqi_inspired_orthographic_rules")
    return {
        "source_id": source_id,
        "source_row": source_row,
        "source_corpus": COMMON_SOURCE_CORPUS,
        "method": method,
        "seed": seed,
        "informal_wolof": informal,
        "formal_wolof": formal,
        "changed": informal != formal,
        "edit_count": len(substitutions) + sum(rule_counts.values()),
        "alignment_edit_count": len(substitutions),
        "spelling_edit_count": sum(rule_counts.values()),
        "error_types": json.dumps(error_types, ensure_ascii=False),
        "applied_rules": json.dumps(rule_counts, ensure_ascii=False, sort_keys=True),
        "alignment_edits": json.dumps(substitutions, ensure_ascii=False),
    }


def generate_m2_pair(
    formal: str,
    french: str,
    lexicon: Mapping[str, Mapping[str, float]],
    *,
    source_id: str,
    source_row: int,
    substitution_ratio: float = 0.20,
    entity_spans: Sequence[tuple[int, int]] | None = None,
    seed: int = DEFAULT_BENCHMARK_SEED,
) -> dict[str, object]:
    masked, protected = mask_protected(formal, entity_spans)
    matches = list(WORD_RE.finditer(masked))
    french_words = set(word_tokens(french))
    desired = max(1, round(len(matches) * substitution_ratio)) if matches else 0
    candidates = []
    for word_index, match in enumerate(matches):
        source = match.group(0).casefold()
        for target, probability in lexicon.get(source, {}).items():
            target_key = str(target).casefold()
            if target_key in french_words:
                candidates.append(
                    {
                        "word_start": word_index,
                        "word_end": word_index + 1,
                        "char_start": match.start(),
                        "char_end": match.end(),
                        "source_wolof": match.group(0),
                        "target_french": target_key,
                        "alignment_probability": float(probability),
                    }
                )
    candidates.sort(
        key=lambda row: (
            -float(row["alignment_probability"]),
            int(row["word_start"]),
            str(row["target_french"]),
        )
    )
    selected = []
    used_source = set()
    used_target = set()
    for candidate in candidates:
        source_position = int(candidate["word_start"])
        target = str(candidate["target_french"])
        if source_position in used_source or target in used_target:
            continue
        selected.append(candidate)
        used_source.add(source_position)
        used_target.add(target)
        if len(selected) >= desired:
            break

    informal, rule_counts = _apply_substitutions_and_rules(masked, selected, protected)
    return _common_row(
        source_id=source_id,
        source_row=source_row,
        formal=formal,
        informal=informal,
        method=METHOD_M2,
        substitutions=selected,
        rule_counts=rule_counts,
        seed=seed,
    )


def generate_m2_dataset(
    parallel: pd.DataFrame,
    lexicon: Mapping[str, Mapping[str, float]],
    *,
    substitution_ratio: float = 0.20,
    entities_by_row: Mapping[int, Sequence[tuple[int, int]]] | None = None,
    seed: int = DEFAULT_BENCHMARK_SEED,
) -> pd.DataFrame:
    entities = entities_by_row or {}
    rows = []
    for source_row, record in parallel.reset_index(drop=True).iterrows():
        rows.append(
            generate_m2_pair(
                str(record["wolof"]),
                str(record["french"]),
                lexicon,
                source_id=f"parallel:{source_row}",
                source_row=source_row,
                substitution_ratio=substitution_ratio,
                entity_spans=entities.get(source_row),
                seed=seed,
            )
        )
    return pd.DataFrame(rows)


def build_alignment_table(
    parallel: pd.DataFrame,
    *,
    min_count: int = 3,
) -> dict[str, dict[str, float]]:
    cooccurrence: dict[str, Counter[str]] = defaultdict(Counter)
    for record in parallel.itertuples(index=False):
        for wolof in set(word_tokens(record.wolof)):
            cooccurrence[wolof].update(set(word_tokens(record.french)))
    table = {}
    for wolof, counts in cooccurrence.items():
        total = sum(counts.values())
        table[wolof] = {
            french: count / total
            for french, count in counts.items()
            if count >= min_count
        }
    return table


def train_cmdr_model(
    parallel: pd.DataFrame,
    *,
    max_n: int = 3,
    vector_size: int = 150,
    window: int = 10,
    min_count: int = 2,
    epochs: int = 20,
    seed: int = DEFAULT_BENCHMARK_SEED,
):
    from gensim.models import Word2Vec

    corpus = [
        make_ngrams(word_tokens(row.wolof), max_n)
        + make_ngrams(word_tokens(row.french), max_n)
        for row in parallel.itertuples(index=False)
    ]
    return Word2Vec(
        sentences=corpus,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=1,
        epochs=epochs,
        seed=seed,
        hashfxn=stable_token_hash,
    )


def save_cmdr_resources(model, alignment_table: dict, model_path: Path, table_path: Path):
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    Path(table_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))
    with Path(table_path).open("wb") as handle:
        pickle.dump(alignment_table, handle)


def load_cmdr_resources(model_path: Path, table_path: Path):
    from gensim.models import Word2Vec

    model = Word2Vec.load(str(model_path))
    with Path(table_path).open("rb") as handle:
        table = pickle.load(handle)
    return model, table


def _alignment_score(
    wolof_ngram: str,
    french_ngram: str,
    table: Mapping[str, Mapping[str, float]],
) -> float:
    scores = [
        max((table.get(source, {}).get(target, 0.0) for target in french_ngram.split("_")), default=0.0)
        for source in wolof_ngram.split("_")
    ]
    return sum(scores) / len(scores) if scores else 0.0


def _position_score(source_position: float, target_position: float, sigma: float) -> float:
    return math.exp(-((source_position - target_position) ** 2) / (2 * sigma**2))


def generate_m3_pair(
    formal: str,
    french: str,
    model,
    alignment_table: Mapping[str, Mapping[str, float]],
    *,
    source_id: str,
    source_row: int,
    max_n: int = 2,
    max_substitutions: int = 2,
    embedding_threshold: float = 0.70,
    alignment_threshold: float = 0.01,
    position_sigma: float = 0.35,
    entity_spans: Sequence[tuple[int, int]] | None = None,
    seed: int = DEFAULT_BENCHMARK_SEED,
) -> dict[str, object]:
    masked, protected = mask_protected(formal, entity_spans)
    matches = list(WORD_RE.finditer(masked))
    wolof = [match.group(0).casefold() for match in matches]
    french_tokens = word_tokens(french)
    ngram_penalty = {1: 1.0, 2: 0.85, 3: 0.70}
    candidates = []
    for size in range(max_n, 0, -1):
        for start in range(len(wolof) - size + 1):
            source_ngram = "_".join(wolof[start : start + size])
            if source_ngram not in model.wv:
                continue
            source_position = (start + size / 2) / max(1, len(wolof))
            for target_start in range(len(french_tokens) - size + 1):
                target_ngram = "_".join(french_tokens[target_start : target_start + size])
                if target_ngram not in model.wv or target_ngram == source_ngram:
                    continue
                embedding = float(model.wv.similarity(source_ngram, target_ngram))
                if embedding < embedding_threshold:
                    continue
                alignment = _alignment_score(source_ngram, target_ngram, alignment_table)
                if alignment < alignment_threshold:
                    continue
                target_position = (target_start + size / 2) / max(1, len(french_tokens))
                positional = _position_score(source_position, target_position, position_sigma)
                combined = embedding * (1 + alignment) * positional * ngram_penalty[size]
                candidates.append(
                    {
                        "word_start": start,
                        "word_end": start + size,
                        "char_start": matches[start].start(),
                        "char_end": matches[start + size - 1].end(),
                        "source_wolof": " ".join(wolof[start : start + size]),
                        "target_french": " ".join(french_tokens[target_start : target_start + size]),
                        "embedding_similarity": embedding,
                        "alignment_score": alignment,
                        "position_score": positional,
                        "combined_score": combined,
                    }
                )
    candidates.sort(
        key=lambda row: (
            -float(row["combined_score"]),
            -(int(row["word_end"]) - int(row["word_start"])),
            int(row["word_start"]),
            str(row["target_french"]),
        )
    )
    selected = []
    occupied = set()
    used_targets = set()
    for candidate in candidates:
        span = set(range(int(candidate["word_start"]), int(candidate["word_end"])))
        target = str(candidate["target_french"])
        if span & occupied or target in used_targets:
            continue
        selected.append(candidate)
        occupied.update(span)
        used_targets.add(target)
        if len(selected) >= max_substitutions:
            break

    informal, rule_counts = _apply_substitutions_and_rules(masked, selected, protected)
    return _common_row(
        source_id=source_id,
        source_row=source_row,
        formal=formal,
        informal=informal,
        method=METHOD_M3,
        substitutions=selected,
        rule_counts=rule_counts,
        seed=seed,
    )


def generate_m3_dataset(
    parallel: pd.DataFrame,
    model,
    alignment_table: Mapping[str, Mapping[str, float]],
    *,
    entities_by_row: Mapping[int, Sequence[tuple[int, int]]] | None = None,
    seed: int = DEFAULT_BENCHMARK_SEED,
    **parameters,
) -> pd.DataFrame:
    entities = entities_by_row or {}
    rows = []
    for source_row, record in parallel.reset_index(drop=True).iterrows():
        rows.append(
            generate_m3_pair(
                str(record["wolof"]),
                str(record["french"]),
                model,
                alignment_table,
                source_id=f"parallel:{source_row}",
                source_row=source_row,
                entity_spans=entities.get(source_row),
                seed=seed,
                **parameters,
            )
        )
    return pd.DataFrame(rows)


def write_alignment_benchmark(
    pairs: pd.DataFrame,
    *,
    output_path: Path,
    review_path: Path,
    manifest_path: Path,
    method: str,
    input_path: Path,
    artifacts: Mapping[str, Path],
    parameters: Mapping[str, object],
    review_size: int = 100,
    review_seed: int = DEFAULT_BENCHMARK_SEED,
) -> dict[str, object]:
    for path in (output_path, review_path, manifest_path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(output_path, index=False, encoding="utf-8")
    candidates = pairs.loc[pairs["changed"]]
    review = candidates.sample(min(review_size, len(candidates)), random_state=review_seed).copy()
    review["meaning_preserved"] = ""
    review["code_switch_natural_1_5"] = ""
    review["spelling_plausible_1_5"] = ""
    review["synthetic_artifact"] = ""
    review["review_notes"] = ""
    review.to_csv(review_path, index=False, encoding="utf-8-sig")

    manifest = {
        "method": method,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_path": str(Path(input_path).resolve()),
        "input_sha256": sha256_file(Path(input_path)),
        "output_path": str(Path(output_path).resolve()),
        "output_sha256": sha256_file(Path(output_path)),
        "rows": len(pairs),
        "changed_rows": int(pairs["changed"].sum()),
        "unchanged_rows": int((~pairs["changed"]).sum()),
        "rows_with_alignment_edits": int(pairs["alignment_edit_count"].gt(0).sum()),
        "alignment_edits": int(pairs["alignment_edit_count"].sum()),
        "spelling_edits": int(pairs["spelling_edit_count"].sum()),
        "artifacts": {},
        "parameters": dict(parameters),
        "manual_review_rows": len(review),
    }
    for name, path_value in artifacts.items():
        path = Path(path_value)
        record = {"path": str(path.resolve()), "sha256": sha256_file(path)}
        sidecars = sorted(path.parent.glob(f"{path.name}.*.npy"))
        if sidecars:
            record["sidecars"] = [
                {"path": str(sidecar.resolve()), "sha256": sha256_file(sidecar)}
                for sidecar in sidecars
            ]
        manifest["artifacts"][name] = record
    Path(manifest_path).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest
