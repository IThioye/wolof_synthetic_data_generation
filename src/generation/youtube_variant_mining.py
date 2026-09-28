"""Unsupervised YouTube lexical-variant mining and M5 generation.

No gold normalization targets are consumed. A final run must exclude every
comment from gold development and test videos; provisional runs are labelled in
their method name and manifest.
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
from typing import Iterable, Mapping, Sequence

import pandas as pd
from rapidfuzz.distance import Levenshtein

from src.generation.benchmark_baselines import (
    COMMON_SOURCE_CORPUS,
    DEFAULT_BENCHMARK_SEED,
    sha256_file,
)


METHOD_M5 = "m5_youtube_mined_variants"
WORD_RE = re.compile(r"[^\W\d_]+(?:[-’'][^\W\d_]+)*", flags=re.UNICODE)


@dataclass(frozen=True)
class MiningConfig:
    min_frequency: int = 5
    min_token_length: int = 3
    max_token_length: int = 30
    context_window: int = 2
    max_context_items: int = 50
    candidate_pool_limit: int = 200
    max_length_difference: int = 3
    minimum_char_similarity: float = 0.50
    acceptance_score: float = 0.68
    minimum_margin: float = 0.03
    character_weight: float = 0.40
    rule_weight: float = 0.30
    context_weight: float = 0.20
    frequency_weight: float = 0.10
    max_edits_per_sentence: int = 2
    capitalization_min_count: int = 3
    capitalization_ratio: float = 0.60
    seed: int = DEFAULT_BENCHMARK_SEED


def word_tokens(text: object) -> list[str]:
    if text is None or pd.isna(text):
        return []
    normalized = unicodedata.normalize("NFKC", str(text)).casefold()
    return WORD_RE.findall(normalized)


def canonical_signature(token: str) -> str:
    """Collapse common formal/informal Wolof spellings for candidate retrieval."""
    text = unicodedata.normalize("NFKC", token).casefold()
    replacements = (
        ("thie", "c"),
        ("thi", "c"),
        ("gn", "ñ"),
        ("ng", "ŋ"),
        ("kh", "x"),
        ("dio", "j"),
        ("dj", "j"),
        ("eu", "ë"),
        ("ou", "u"),
    )
    for source, target in replacements:
        text = text.replace(source, target)
    text = "".join(
        char for char in unicodedata.normalize("NFD", text)
        if unicodedata.category(char) != "Mn"
    )
    text = re.sub(r"([a-zà-ÿ])\1+", r"\1", text)
    return text


def character_ngrams(token: str, size: int = 3) -> set[str]:
    padded = f"^{token}$"
    if len(padded) <= size:
        return {padded}
    return {padded[index : index + size] for index in range(len(padded) - size + 1)}


def _similarity(left: str, right: str) -> float:
    return float(Levenshtein.normalized_similarity(left, right))


def _weighted_context_similarity(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    keys = set(left) | set(right)
    numerator = sum(min(left[key], right[key]) for key in keys)
    denominator = sum(max(left[key], right[key]) for key in keys)
    return numerator / denominator if denominator else 0.0


def _trim_counter(counter: Counter[str], limit: int) -> Counter[str]:
    return Counter(dict(counter.most_common(limit)))


class FormalCandidateIndex:
    def __init__(self, formal_words: Iterable[str], french_words: Iterable[str] = ()):
        french = {word.casefold() for word in french_words}
        self.words = sorted(
            {
                word.casefold().strip()
                for word in formal_words
                if word and word.strip() and word.casefold().strip() not in french
            }
        )
        self.signatures: dict[str, list[str]] = defaultdict(list)
        self.ngrams: dict[str, list[str]] = defaultdict(list)
        for word in self.words:
            signature = canonical_signature(word)
            self.signatures[signature].append(word)
            for ngram in character_ngrams(signature):
                self.ngrams[ngram].append(word)

    def candidates(self, informal: str, config: MiningConfig) -> list[str]:
        signature = canonical_signature(informal)
        candidates = set(self.signatures.get(signature, []))
        shared_counts: Counter[str] = Counter()
        for ngram in character_ngrams(signature):
            shared_counts.update(self.ngrams.get(ngram, []))
        for word, _shared in shared_counts.most_common(config.candidate_pool_limit * 2):
            if abs(len(word) - len(informal)) <= config.max_length_difference:
                candidates.add(word)
            if len(candidates) >= config.candidate_pool_limit:
                break
        return sorted(candidates)


def count_tokens(texts: Iterable[object]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(word_tokens(text))
    return counts


def likely_capitalized_tokens(
    raw_texts: Iterable[object],
    *,
    min_count: int,
    minimum_ratio: float,
    ignore_first_token: bool = False,
) -> set[str]:
    """Find probable names/entities using capitalization in the pre-lowercase text."""
    totals: Counter[str] = Counter()
    capitalized: Counter[str] = Counter()
    for text in raw_texts:
        if text is None or pd.isna(text):
            continue
        for position, match in enumerate(
            WORD_RE.finditer(unicodedata.normalize("NFKC", str(text)))
        ):
            if ignore_first_token and position == 0:
                continue
            surface = match.group(0)
            key = surface.casefold()
            totals[key] += 1
            if surface[:1].isupper():
                capitalized[key] += 1
    return {
        token
        for token, total in totals.items()
        if total >= min_count and capitalized[token] / total >= minimum_ratio
    }


def collect_contexts(
    texts: Iterable[object],
    target_tokens: set[str],
    *,
    window: int,
    max_items: int,
) -> dict[str, Counter[str]]:
    contexts: dict[str, Counter[str]] = defaultdict(Counter)
    for text in texts:
        tokens = word_tokens(text)
        for index, token in enumerate(tokens):
            if token not in target_tokens:
                continue
            start = max(0, index - window)
            end = min(len(tokens), index + window + 1)
            for context_index in range(start, end):
                if context_index != index:
                    contexts[token][canonical_signature(tokens[context_index])] += 1
    return {
        token: _trim_counter(counter, max_items) for token, counter in contexts.items()
    }


def mine_variant_mappings(
    comments: Sequence[object],
    formal_sentences: Sequence[object],
    formal_vocab: Iterable[str],
    french_vocab: Iterable[str],
    protected_tokens: Iterable[str] = (),
    config: MiningConfig = MiningConfig(),
) -> pd.DataFrame:
    """Rank one best formal candidate per observed YouTube OOV token."""
    formal_words = {word.casefold().strip() for word in formal_vocab if word.strip()}
    french_words = {word.casefold().strip() for word in french_vocab if word.strip()}
    protected_words = {
        word.casefold().strip() for word in protected_tokens if word and word.strip()
    }
    frequencies = count_tokens(comments)
    index = FormalCandidateIndex(formal_words, french_words)

    observed_oov = {
        token
        for token, frequency in frequencies.items()
        if token not in formal_words and frequency >= config.min_frequency
    }
    eligible = {
        token
        for token in observed_oov
        if config.min_token_length <= len(token) <= config.max_token_length
        and token not in french_words
        and token not in protected_words
        and not re.search(r"(.)\1{3,}", token)
    }
    comment_contexts = collect_contexts(
        comments,
        eligible,
        window=config.context_window,
        max_items=config.max_context_items,
    )
    candidate_targets = set()
    candidate_pools: dict[str, list[str]] = {}
    for token in eligible:
        pool = index.candidates(token, config)
        candidate_pools[token] = pool
        candidate_targets.update(pool)
    formal_contexts = collect_contexts(
        formal_sentences,
        candidate_targets,
        window=config.context_window,
        max_items=config.max_context_items,
    )

    maximum_frequency = max(frequencies.values(), default=1)
    rows = []
    for token in sorted(observed_oov, key=lambda value: (-frequencies[value], value)):
        frequency = frequencies[token]
        base = {
            "informal_token": token,
            "frequency": frequency,
            "formal_token": "",
            "score": 0.0,
            "second_score": 0.0,
            "margin": 0.0,
            "char_similarity": 0.0,
            "rule_similarity": 0.0,
            "context_similarity": 0.0,
            "frequency_score": math.log1p(frequency) / math.log1p(maximum_frequency),
            "candidate_count": 0,
            "status": "rejected",
            "reason": "",
        }
        if len(token) < config.min_token_length:
            rows.append({**base, "reason": "too_short"})
            continue
        if len(token) > config.max_token_length:
            rows.append({**base, "reason": "too_long"})
            continue
        if token in french_words:
            rows.append({**base, "reason": "likely_french"})
            continue
        if token in protected_words:
            rows.append({**base, "reason": "likely_entity_by_capitalization"})
            continue
        if re.search(r"(.)\1{3,}", token):
            rows.append({**base, "reason": "excessive_repetition"})
            continue

        scored = []
        informal_signature = canonical_signature(token)
        for formal in candidate_pools.get(token, []):
            char_score = _similarity(token, formal)
            rule_score = _similarity(informal_signature, canonical_signature(formal))
            context_score = _weighted_context_similarity(
                comment_contexts.get(token, Counter()),
                formal_contexts.get(formal, Counter()),
            )
            frequency_score = base["frequency_score"]
            total = (
                config.character_weight * char_score
                + config.rule_weight * rule_score
                + config.context_weight * context_score
                + config.frequency_weight * frequency_score
            )
            scored.append((total, formal, char_score, rule_score, context_score))
        scored.sort(reverse=True)
        if not scored:
            rows.append({**base, "reason": "no_candidate"})
            continue

        best = scored[0]
        second_score = scored[1][0] if len(scored) > 1 else 0.0
        margin = best[0] - second_score if len(scored) > 1 else best[0]
        result = {
            **base,
            "formal_token": best[1],
            "score": best[0],
            "second_score": second_score,
            "margin": margin,
            "char_similarity": best[2],
            "rule_similarity": best[3],
            "context_similarity": best[4],
            "candidate_count": len(scored),
        }
        if best[2] < config.minimum_char_similarity:
            result["reason"] = "low_character_similarity"
        elif best[0] < config.acceptance_score:
            result["reason"] = "low_score"
        elif margin < config.minimum_margin:
            result["reason"] = "ambiguous_margin"
        else:
            result["status"] = "accepted"
            result["reason"] = "accepted"
        rows.append(result)

    return pd.DataFrame(rows)


def build_mapping_index(accepted: pd.DataFrame) -> dict[str, list[dict[str, object]]]:
    required = {"informal_token", "formal_token", "frequency", "score"}
    missing = required - set(accepted.columns)
    if missing:
        raise ValueError(f"Accepted mappings are missing columns: {sorted(missing)}")
    mapping: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in accepted.itertuples(index=False):
        mapping[row.formal_token].append(
            {
                "informal": row.informal_token,
                "frequency": int(row.frequency),
                "score": float(row.score),
            }
        )
    for variants in mapping.values():
        variants.sort(key=lambda row: (-row["score"], -row["frequency"], row["informal"]))
    return dict(mapping)


def _stable_rng(seed: int, source_id: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{source_id}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def apply_mined_variants(
    text: str,
    source_id: str,
    mapping_index: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    seed: int,
    max_edits: int,
) -> tuple[str, list[dict[str, object]]]:
    """Replace eligible formal tokens while retaining punctuation and spacing."""
    matches = list(WORD_RE.finditer(text))
    eligible = []
    for position, match in enumerate(matches):
        original = match.group(0)
        key = original.casefold()
        if key in mapping_index and not original.isupper():
            eligible.append(position)
    rng = _stable_rng(seed, source_id)
    rng.shuffle(eligible)
    selected = set(eligible[:max_edits])

    pieces = []
    edits = []
    cursor = 0
    for position, match in enumerate(matches):
        pieces.append(text[cursor : match.start()])
        original = match.group(0)
        replacement = original
        if position in selected:
            variants = mapping_index[original.casefold()]
            weights = [
                max(1e-9, float(item["score"]) * math.log1p(int(item["frequency"])))
                for item in variants
            ]
            choice = rng.choices(list(variants), weights=weights, k=1)[0]
            replacement = str(choice["informal"])
            edits.append(
                {
                    "position": position,
                    "formal": original,
                    "informal": replacement,
                    "mapping_score": float(choice["score"]),
                    "youtube_frequency": int(choice["frequency"]),
                }
            )
        pieces.append(replacement)
        cursor = match.end()
    pieces.append(text[cursor:])
    return "".join(pieces), edits


def build_m5_pairs(
    sources: pd.DataFrame,
    accepted_mappings: pd.DataFrame,
    *,
    config: MiningConfig = MiningConfig(),
    provisional: bool,
) -> pd.DataFrame:
    mapping_index = build_mapping_index(accepted_mappings)
    method = f"{METHOD_M5}_provisional" if provisional else METHOD_M5
    rows = []
    for source in sources.itertuples(index=False):
        informal, edits = apply_mined_variants(
            source.formal_wolof,
            source.source_id,
            mapping_index,
            seed=config.seed,
            max_edits=config.max_edits_per_sentence,
        )
        rows.append(
            {
                "source_id": source.source_id,
                "source_row": source.source_row,
                "source_corpus": COMMON_SOURCE_CORPUS,
                "method": method,
                "seed": config.seed,
                "informal_wolof": informal,
                "formal_wolof": source.formal_wolof,
                "changed": bool(edits),
                "edit_count": len(edits),
                "error_types": json.dumps(
                    ["youtube_mined_lexical_variant"] if edits else [],
                    ensure_ascii=False,
                ),
                "applied_rules": json.dumps(edits, ensure_ascii=False),
            }
        )
    return pd.DataFrame(rows)


def write_m5_outputs(
    candidates: pd.DataFrame,
    pairs: pd.DataFrame,
    *,
    candidates_path: Path,
    accepted_path: Path,
    mapping_review_path: Path,
    generation_review_path: Path,
    pairs_path: Path,
    manifest_path: Path,
    comments_path: Path,
    formal_path: Path,
    vocabulary_path: Path,
    excluded_videos: set[str],
    eligible_comment_rows: int,
    provisional: bool,
    config: MiningConfig,
    review_size: int = 100,
) -> dict[str, object]:
    for path in (
        candidates_path,
        accepted_path,
        mapping_review_path,
        generation_review_path,
        pairs_path,
        manifest_path,
    ):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    accepted = candidates.loc[candidates["status"].eq("accepted")].copy()
    candidates.to_csv(candidates_path, index=False, encoding="utf-8")
    accepted.to_csv(accepted_path, index=False, encoding="utf-8")
    pairs.to_csv(pairs_path, index=False, encoding="utf-8")

    mapping_review = accepted.sample(
        min(review_size, len(accepted)), random_state=config.seed
    ).copy()
    mapping_review["mapping_correct"] = ""
    mapping_review["review_notes"] = ""
    mapping_review.to_csv(mapping_review_path, index=False, encoding="utf-8-sig")

    changed_pairs = pairs.loc[pairs["changed"]]
    generation_review = changed_pairs.sample(
        min(review_size, len(changed_pairs)), random_state=config.seed
    ).copy()
    generation_review["meaning_preserved"] = ""
    generation_review["informal_plausibility_1_5"] = ""
    generation_review["review_notes"] = ""
    generation_review.to_csv(
        generation_review_path, index=False, encoding="utf-8-sig"
    )

    manifest = {
        "method": METHOD_M5,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "provisional": provisional,
        "leakage_status": (
            "PROVISIONAL: gold dev/test videos are not yet locked"
            if provisional
            else "gold dev/test videos excluded"
        ),
        "comments_path": str(Path(comments_path).resolve()),
        "comments_sha256": sha256_file(Path(comments_path)),
        "formal_path": str(Path(formal_path).resolve()),
        "formal_sha256": sha256_file(Path(formal_path)),
        "vocabulary_path": str(Path(vocabulary_path).resolve()),
        "vocabulary_sha256": sha256_file(Path(vocabulary_path)),
        "eligible_comment_rows": eligible_comment_rows,
        "excluded_video_count": len(excluded_videos),
        "candidate_mappings": len(candidates),
        "accepted_mappings": len(accepted),
        "generated_rows": len(pairs),
        "changed_rows": int(pairs["changed"].sum()),
        "unchanged_rows": int((~pairs["changed"]).sum()),
        "configuration": asdict(config),
        "manual_mapping_review_rows": len(mapping_review),
        "manual_generation_review_rows": len(generation_review),
        "finalization_required": [
            "Repeat mining after gold dev/test videos are locked and excluded.",
            "Review mapping and generation samples without using gold test targets.",
            "Report mapping coverage and unchanged-row rate.",
        ],
    }
    Path(manifest_path).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def finalize_m5_manifest(
    manifest_path: Path,
    mapping_review_path: Path,
    generation_review_path: Path,
) -> dict[str, object]:
    """Finalize a leakage-safe M5 run after both manual reviews are complete."""
    mapping_review = pd.read_csv(mapping_review_path)
    generation_review = pd.read_csv(generation_review_path)
    mapping_required = {"mapping_correct"}
    generation_required = {"meaning_preserved", "informal_plausibility_1_5"}
    if mapping_required - set(mapping_review.columns):
        raise ValueError("M5 mapping review is missing mapping_correct")
    if generation_required - set(generation_review.columns):
        raise ValueError("M5 generation review is missing required ratings")
    mapping_values = mapping_review["mapping_correct"].fillna("").astype(str).str.strip().str.casefold()
    meaning_values = generation_review["meaning_preserved"].fillna("").astype(str).str.strip().str.casefold()
    plausibility = pd.to_numeric(
        generation_review["informal_plausibility_1_5"], errors="coerce"
    )
    if mapping_values.eq("").any() or meaning_values.eq("").any() or plausibility.isna().any():
        raise ValueError("M5 reviews are incomplete")
    if (~plausibility.between(1, 5)).any():
        raise ValueError("M5 plausibility ratings must be between 1 and 5")
    true_values = {"true", "1", "yes", "y", "correct"}
    false_values = {"false", "0", "no", "n", "incorrect"}
    if not mapping_values.isin(true_values | false_values).all():
        raise ValueError("M5 mapping_correct must use a true/false verdict")
    if not meaning_values.isin(true_values | false_values).all():
        raise ValueError("M5 meaning_preserved must use a true/false verdict")
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest.get("provisional", True):
        raise ValueError("A provisional M5 run cannot be finalized")
    manifest["mapping_review_precision"] = float(mapping_values.isin(true_values).mean())
    manifest["generation_meaning_preservation_rate"] = float(
        meaning_values.isin(true_values).mean()
    )
    manifest["generation_mean_plausibility"] = float(plausibility.mean())
    manifest["mapping_review_sha256"] = sha256_file(Path(mapping_review_path))
    manifest["generation_review_sha256"] = sha256_file(Path(generation_review_path))
    manifest["review_finalized_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["finalization_required"] = []
    Path(manifest_path).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest
