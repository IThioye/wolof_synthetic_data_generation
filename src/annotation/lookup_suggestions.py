"""Confidence-aware token suggestions from the Wolof and French lookups."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein


TOKEN_PATTERN = re.compile(r"^\w+(?:[-'’]\w+)*$", flags=re.UNICODE)
PHONETIC_REWRITES = (
    ("kh", "x"),
    ("eu", "ë"),
    ("gn", "ñ"),
    ("th", "c"),
    ("dj", "j"),
    ("ou", "u"),
)
CHAR_EQUIVALENCES = {
    ("a", "à"), ("a", "á"),
    ("e", "é"), ("e", "è"), ("e", "ë"),
    ("i", "ï"), ("o", "ó"), ("u", "ú"),
    ("n", "ñ"), ("n", "ŋ"), ("c", "ç"),
}
CHAR_EQUIVALENCES |= {(right, left) for left, right in CHAR_EQUIVALENCES}


@dataclass(frozen=True)
class Candidate:
    word: str
    score: float
    pos: str = ""
    definition: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "word": self.word,
            "score": round(self.score, 4),
            "pos": self.pos,
            "definition": self.definition,
        }


@dataclass(frozen=True)
class SuggestionDecision:
    category: str
    auto: str
    candidates: tuple[Candidate, ...]
    accepted: bool
    review_required: bool
    reason: str
    french_score: float | None = None
    french_candidate: str | None = None


def _clean(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return " ".join(str(value).split())


def _single_token_words(values) -> set[str]:
    words = set()
    for value in values:
        word = _clean(value).casefold()
        if word and TOKEN_PATTERN.fullmatch(word):
            words.add(word)
    return words


def wolof_character_distance(left: str, right: str) -> float:
    """Levenshtein distance with a small cost for Wolof diacritic variants."""
    left, right = left.casefold(), right.casefold()
    rows, columns = len(left), len(right)
    previous = [float(index) for index in range(columns + 1)]
    for row in range(1, rows + 1):
        current = [float(row)] + [0.0] * columns
        for column in range(1, columns + 1):
            lchar, rchar = left[row - 1], right[column - 1]
            if lchar == rchar:
                substitution = 0.0
            elif (lchar, rchar) in CHAR_EQUIVALENCES:
                substitution = 0.35
            else:
                substitution = 1.0
            current[column] = min(
                previous[column] + 1.0,
                current[column - 1] + 1.0,
                previous[column - 1] + substitution,
            )
        previous = current
    return previous[-1]


def orthographic_variants(token: str, max_variants: int = 24) -> tuple[tuple[str, int], ...]:
    """Generate conservative informal-to-standard Wolof spelling variants."""
    best_steps = {token.casefold(): 0}
    frontier = [token.casefold()]
    for informal, formal in PHONETIC_REWRITES:
        next_frontier = list(frontier)
        for variant in frontier:
            if informal not in variant:
                continue
            rewritten = variant.replace(informal, formal)
            steps = best_steps[variant] + 1
            if steps < best_steps.get(rewritten, math.inf):
                best_steps[rewritten] = steps
                next_frontier.append(rewritten)
        frontier = list(dict.fromkeys(next_frontier))[:max_variants]
    return tuple(sorted(best_steps.items(), key=lambda item: (item[1], item[0])))


class LookupSuggestionIndex:
    """Nearest lookup headwords with French/ambiguity rejection gates."""

    def __init__(
        self,
        wolof_lookup: pd.DataFrame,
        french_words,
        *,
        protected_words=(),
        max_distance_ratio: float = 0.24,
        min_candidate_margin: float = 0.06,
        french_ambiguity_margin: float = 0.04,
    ):
        self.max_distance_ratio = float(max_distance_ratio)
        self.min_candidate_margin = float(min_candidate_margin)
        self.french_ambiguity_margin = float(french_ambiguity_margin)
        self.wolof_words = _single_token_words(wolof_lookup.get("word", []))
        self.french_words = _single_token_words(french_words)
        self.protected_words = {str(word).strip().casefold() for word in protected_words if str(word).strip()}
        self.by_length: dict[int, list[str]] = defaultdict(list)
        self.french_by_length: dict[int, list[str]] = defaultdict(list)
        self.metadata: dict[str, dict[str, str]] = {}
        self._cache: dict[str, SuggestionDecision] = {}

        for word in sorted(self.wolof_words):
            self.by_length[len(word)].append(word)
        for word in sorted(self.french_words):
            self.french_by_length[len(word)].append(word)

        for word, rows in wolof_lookup.groupby(
            wolof_lookup["word"].fillna("").astype(str).str.strip().str.casefold(),
            sort=False,
        ):
            if not word or word not in self.wolof_words:
                continue
            pos_values = sorted({_clean(value) for value in rows.get("pos", []) if _clean(value)})
            definitions = []
            for value in rows.get("translation", []):
                definition = _clean(value)
                if definition and definition not in definitions:
                    definitions.append(definition)
            self.metadata[word] = {
                "pos": " / ".join(pos_values[:4]),
                "definition": "; ".join(definitions[:3]),
            }

    @classmethod
    def from_files(
        cls,
        wolof_lookup_path: Path,
        french_lookup_path: Path,
        curated_lookup_path: Path,
        surnames_path: Path,
        **kwargs,
    ) -> "LookupSuggestionIndex":
        wolof = pd.read_csv(wolof_lookup_path, low_memory=False)
        french = pd.read_csv(french_lookup_path, sep="\t", usecols=["1_Mot"], low_memory=False)
        protected = set()
        if surnames_path.exists():
            protected.update(surnames_path.read_text(encoding="utf-8").splitlines())
        if curated_lookup_path.exists():
            curated = pd.read_csv(curated_lookup_path, keep_default_na=False)
            if "formal_token" in curated.columns:
                protected_mask = pd.Series(False, index=curated.index)
                if "protected" in curated.columns:
                    protected_mask |= curated["protected"].astype(str).str.casefold().eq("yes")
                if "entity_type" in curated.columns:
                    protected_mask |= ~curated["entity_type"].astype(str).str.upper().isin({"", "UNKNOWN"})
                protected.update(curated.loc[protected_mask, "formal_token"].astype(str))
        return cls(wolof, french["1_Mot"], protected_words=protected, **kwargs)

    def _wolof_candidates(self, token: str, limit: int = 6) -> tuple[Candidate, ...]:
        variants = orthographic_variants(token)
        shortlist = set()
        for variant, _steps in variants:
            pool = []
            for length in range(max(1, len(variant) - 2), len(variant) + 3):
                pool.extend(self.by_length.get(length, ()))
            # RapidFuzz performs the broad scan in compiled code.  The custom
            # Wolof-weighted distance is then applied only to this shortlist.
            shortlist.update(
                match[0]
                for match in process.extract(
                    variant,
                    pool,
                    scorer=Levenshtein.normalized_distance,
                    score_cutoff=0.55,
                    limit=48,
                )
            )

        scored: dict[str, float] = {}
        for candidate in shortlist:
            for variant, steps in variants:
                distance = wolof_character_distance(variant, candidate) + 0.12 * steps
                # Normalize against the observed token.  Using the longer side
                # would make an equally distant longer candidate look better;
                # adding a separate length penalty would instead suppress real
                # contractions such as a four-character form -> three letters.
                ratio = distance / max(len(token), 1)
                if ratio <= 0.48:
                    scored[candidate] = min(scored.get(candidate, math.inf), ratio)
        ranked = sorted(scored.items(), key=lambda item: (item[1], item[0]))
        candidates = []
        for word, score in ranked[:limit]:
            meta = self.metadata.get(word, {})
            candidates.append(
                Candidate(
                    word=word,
                    score=score,
                    pos=meta.get("pos", ""),
                    definition=meta.get("definition", ""),
                )
            )
        return tuple(candidates)

    def _nearest_french(self, token: str) -> tuple[str, float] | None:
        pool = []
        for length in range(max(1, len(token) - 2), len(token) + 3):
            pool.extend(self.french_by_length.get(length, ()))
        if not pool:
            return None
        match = process.extractOne(
            token,
            pool,
            scorer=Levenshtein.normalized_distance,
            score_cutoff=min(0.5, self.max_distance_ratio + 0.12),
        )
        return (str(match[0]), float(match[1])) if match else None

    def suggest(self, token: str) -> SuggestionDecision:
        surface = str(token)
        key = surface.casefold()
        if key in self._cache:
            cached = self._cache[key]
            return SuggestionDecision(
                category=cached.category,
                auto=surface if cached.auto == key else cached.auto,
                candidates=cached.candidates,
                accepted=cached.accepted,
                review_required=cached.review_required,
                reason=cached.reason,
                french_score=cached.french_score,
                french_candidate=cached.french_candidate,
            )

        in_wolof = key in self.wolof_words
        in_french = key in self.french_words
        if key in self.protected_words or any(character.isdigit() for character in key):
            decision = SuggestionDecision("protected", key, (), False, False, "Protected name, entity, or number")
        elif in_wolof and in_french:
            decision = SuggestionDecision("ambiguous_exact", key, (), False, True, "Exact entry in both Wolof and French lookups")
        elif in_french:
            decision = SuggestionDecision("french_exact", key, (), False, True, "Exact French lookup entry; left unchanged")
        elif in_wolof:
            meta = self.metadata.get(key, {})
            exact = Candidate(key, 0.0, meta.get("pos", ""), meta.get("definition", ""))
            decision = SuggestionDecision("wolof_formal", key, (exact,), False, False, "Exact formal Wolof lookup entry")
        else:
            candidates = self._wolof_candidates(key)
            french_match = self._nearest_french(key)
            french_candidate = french_match[0] if french_match else None
            french_score = french_match[1] if french_match else None
            allowed = self.max_distance_ratio if len(key) >= 4 else min(0.18, self.max_distance_ratio)
            if not candidates or candidates[0].score > allowed:
                if french_score is not None and french_score <= allowed:
                    decision = SuggestionDecision(
                        "possible_french",
                        key,
                        candidates,
                        False,
                        True,
                        "A nearby French form passed the threshold while no Wolof candidate did",
                        french_score,
                        french_candidate,
                    )
                else:
                    decision = SuggestionDecision("unresolved", key, candidates, False, True, "No Wolof lookup candidate passed the distance threshold", french_score, french_candidate)
            else:
                best = candidates[0]
                second_score = candidates[1].score if len(candidates) > 1 else math.inf
                wolof_margin = second_score - best.score
                french_competitive = french_score is not None and french_score <= best.score + self.french_ambiguity_margin
                if french_competitive:
                    category = "possible_french" if french_score + self.french_ambiguity_margin < best.score else "ambiguous_near"
                    reason = f"French candidate '{french_candidate}' is as plausible as the Wolof candidate"
                    decision = SuggestionDecision(category, key, candidates, False, True, reason, french_score, french_candidate)
                elif wolof_margin < self.min_candidate_margin:
                    decision = SuggestionDecision("ambiguous_wolof", key, candidates, False, True, "Several Wolof candidates have nearly equal distance", french_score, french_candidate)
                else:
                    decision = SuggestionDecision(
                        "informal_wolof",
                        best.word,
                        candidates,
                        True,
                        True,
                        f"Close Wolof candidate passed the automatic threshold ({best.score:.2f}); verify in context",
                        french_score,
                        french_candidate,
                    )

        self._cache[key] = decision
        if surface != key and decision.auto == key:
            return SuggestionDecision(
                decision.category,
                surface,
                decision.candidates,
                decision.accepted,
                decision.review_required,
                decision.reason,
                decision.french_score,
                decision.french_candidate,
            )
        return decision
