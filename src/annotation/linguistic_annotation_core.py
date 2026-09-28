"""Gold-train linguistic annotation service for the isolated M6 generator."""

from __future__ import annotations

import csv
import difflib
import json
import threading
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import (
    FORMAL_TOKEN_LOOKUP_PATH,
    FORMAL_TOKEN_LOOKUP_SEED_PATH,
    GOLD_TRAIN_PATH,
    LINGUISTIC_ANNOTATIONS_PATH,
)


POS_OPTIONS = (
    "UNKNOWN", "NOUN", "VERB", "ADJ", "ADV", "PROPN", "PRON", "DET",
    "ADP", "AUX", "CCONJ", "SCONJ", "PART", "INTJ", "NUM", "PUNCT",
    "SYM", "X",
)
LANGUAGE_OPTIONS = ("unknown", "wo", "fr", "mixed", "other")
ENTITY_OPTIONS = (
    "UNKNOWN", "NONE", "PERSON", "LOCATION", "ORGANIZATION", "EVENT",
    "PRODUCT", "WORK", "OTHER",
)
PROTECTED_OPTIONS = ("uncertain", "no", "yes", "contextual")
ERROR_OPTIONS = (
    "identity",
    "capitalization",
    "diacritic",
    "spacing_split",
    "spacing_merge",
    "abbreviation",
    "repetition",
    "deletion",
    "insertion",
    "substitution",
    "phonetic_spelling",
    "french_spelling",
    "lexical_replacement",
    "punctuation",
    "multiple",
    "other",
    "uncertain",
)
ERROR_DEFINITIONS = {
    "identity": "The informal and formal units are exactly the same; no synthetic error occurred.",
    "capitalization": "Only uppercase/lowercase usage differs from the normalized form.",
    "diacritic": "A normalized accent or Wolof diacritic was removed, added, or changed.",
    "spacing_split": "The informal form introduces a word boundary that is absent from the formal unit.",
    "spacing_merge": "The informal form removes a word boundary and joins formal words.",
    "abbreviation": "The informal form is a substantially shortened version of the formal unit.",
    "repetition": "One or more characters or syllable-like sequences are repeated informally.",
    "deletion": "One or more non-space characters from the formal unit are absent informally.",
    "insertion": "One or more non-space characters are added to the informal form.",
    "substitution": "A character or character sequence is replaced by another sequence.",
    "phonetic_spelling": "The informal spelling represents pronunciation rather than standard orthography.",
    "french_spelling": "A French token is present but written with a non-standard or misspelled French form.",
    "lexical_replacement": "The informal and formal units are different lexical choices, not only spelling variants.",
    "punctuation": "Punctuation is inserted, deleted, or replaced.",
    "multiple": "A complex transformation is not adequately described by the more specific selected labels.",
    "other": "A known transformation does not fit the available labels.",
    "uncertain": "The correct interpretation cannot be determined confidently.",
}
REVIEW_OPTIONS = ("reviewed", "uncertain")

OCCURRENCE_COLUMNS = (
    "occurrence_id",
    "source_index",
    "token_position",
    "informal_token",
    "formal_token",
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
    "notes",
    "lexical_reusable",
    "lookup_source",
    "saved_utc",
)

LOOKUP_COLUMNS = (
    "formal_token",
    "lemma",
    "pos",
    "target_language",
    "entity_type",
    "protected",
    "review_status",
    "source",
    "source_index",
    "updated_utc",
)

LEXICAL_FIELDS = (
    "lemma", "pos", "target_language", "entity_type", "protected", "review_status"
)


def clean(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return " ".join(str(value).strip().split())


def lookup_key(value: object) -> str:
    return clean(value).casefold()


def without_diacritics(value: str) -> str:
    decomposed = unicodedata.normalize("NFD", value)
    return "".join(char for char in decomposed if unicodedata.category(char) != "Mn")


def derive_transformations(formal: str, informal: str) -> list[dict[str, object]]:
    """Return a deterministic character edit script in generation direction."""
    formal = clean(formal)
    informal = clean(informal)
    steps: list[dict[str, object]] = []
    matcher = difflib.SequenceMatcher(a=formal, b=informal, autojunk=False)
    operation_names = {"replace": "replace", "delete": "delete", "insert": "insert"}
    for tag, formal_start, formal_end, informal_start, informal_end in matcher.get_opcodes():
        if tag == "equal":
            continue
        steps.append(
            {
                "operation": operation_names[tag],
                "formal_text": formal[formal_start:formal_end],
                "informal_text": informal[informal_start:informal_end],
                "formal_start": formal_start,
                "formal_end": formal_end,
                "informal_start": informal_start,
                "informal_end": informal_end,
            }
        )
    return steps


def _ordered_error_types(values: set[str]) -> tuple[str, ...]:
    return tuple(option for option in ERROR_OPTIONS if option in values)


def infer_error_types(informal: str, formal: str) -> tuple[str, ...]:
    """Provide conservative multi-label hints; the annotator is authoritative."""
    informal = clean(informal)
    formal = clean(formal)
    if informal == formal:
        return ("identity",)
    if informal.casefold() == formal.casefold():
        return ("capitalization",)

    labels: set[str] = set()
    if formal.count(" ") > informal.count(" "):
        labels.add("spacing_merge")
    elif formal.count(" ") < informal.count(" "):
        labels.add("spacing_split")

    folded_informal = informal.casefold()
    folded_formal = formal.casefold()
    informal_marks = sum(
        unicodedata.category(char) == "Mn"
        for char in unicodedata.normalize("NFD", folded_informal)
    )
    formal_marks = sum(
        unicodedata.category(char) == "Mn"
        for char in unicodedata.normalize("NFD", folded_formal)
    )
    if informal_marks != formal_marks:
        labels.add("diacritic")
    if without_diacritics(folded_informal) == without_diacritics(folded_formal):
        return _ordered_error_types(labels)

    if len(folded_informal) <= max(1, int(len(folded_formal) * 0.65)):
        labels.add("abbreviation")

    for step in derive_transformations(formal, informal):
        formal_text = str(step["formal_text"])
        informal_text = str(step["informal_text"])
        nonspace_formal = formal_text.replace(" ", "")
        nonspace_informal = informal_text.replace(" ", "")
        operation = step["operation"]
        if operation == "delete" and nonspace_formal:
            labels.add("deletion")
        elif operation == "insert" and nonspace_informal:
            labels.add("insertion")
            start = int(step["informal_start"])
            end = int(step["informal_end"])
            left = informal[start - 1] if start > 0 else ""
            right = informal[end] if end < len(informal) else ""
            if (
                any(char and char in {left, right} for char in nonspace_informal)
                or len(set(nonspace_informal)) < len(nonspace_informal)
            ):
                labels.add("repetition")
        elif operation == "replace" and (nonspace_formal or nonspace_informal):
            labels.add("substitution")

        changed_characters = formal_text + informal_text
        if any(unicodedata.category(char).startswith("P") for char in changed_characters):
            labels.add("punctuation")

    if not labels:
        labels.add("other")
    return _ordered_error_types(labels)


def infer_error_type(informal: str, formal: str) -> str:
    """Backward-compatible primary hint for older callers."""
    return infer_error_types(informal, formal)[0]


def parse_error_types(value: object, fallback: object = "") -> tuple[str, ...]:
    try:
        parsed = json.loads(clean(value) or "[]")
    except (json.JSONDecodeError, TypeError):
        parsed = []
    if not isinstance(parsed, list):
        parsed = []
    values = {clean(item) for item in parsed if clean(item) in ERROR_OPTIONS}
    fallback_value = clean(fallback)
    if not values and fallback_value in ERROR_OPTIONS:
        values.add(fallback_value)
    return _ordered_error_types(values)


def append_row(path: Path, row: dict[str, object], columns: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        if not exists:
            writer.writeheader()
        writer.writerow({column: row.get(column, "") for column in columns})


def read_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path, dtype="string", keep_default_na=False)


def build_occurrences(gold_train: pd.DataFrame) -> pd.DataFrame:
    required = {
        "source_index", "comment", "manual_formal_wolof", "video_url",
        "token_corrections_json",
    }
    missing = required - set(gold_train.columns)
    if missing:
        raise ValueError(f"Gold train is missing columns: {sorted(missing)}")

    records: list[dict[str, object]] = []
    for sentence_position, row in gold_train.reset_index(drop=True).iterrows():
        source_index = clean(row["source_index"])
        try:
            payload = json.loads(row["token_corrections_json"] or "[]")
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid token JSON for source {source_index}") from exc
        for token_position, token_row in enumerate(payload):
            informal = clean(token_row.get("token"))
            selected = clean(token_row.get("selected_candidate"))
            formal = clean(token_row.get("final_correction")) or selected or informal
            if not informal or not formal:
                continue
            records.append(
                {
                    "occurrence_id": f"{source_index}:{token_position}",
                    "source_index": source_index,
                    "sentence_position": int(sentence_position),
                    "token_position": int(token_position),
                    "informal_token": informal,
                    "formal_token": formal,
                    "changed": lookup_key(informal) != lookup_key(formal),
                    "automatic_category": clean(token_row.get("category")),
                    "comment": clean(row["comment"]),
                    "formal_sentence": clean(row["manual_formal_wolof"]),
                    "video_url": clean(row["video_url"]),
                }
            )
    occurrences = pd.DataFrame(records)
    if occurrences.empty:
        raise ValueError("Gold train contains no token occurrences")
    if occurrences["occurrence_id"].duplicated().any():
        raise ValueError("Gold train produced duplicate occurrence identifiers")
    return occurrences


def load_lookup(path: Path, *, source_name: str) -> dict[str, dict[str, object]]:
    frame = read_optional_csv(path)
    if frame.empty:
        return {}
    required = {"formal_token", *LEXICAL_FIELDS}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path.name} is missing lookup columns: {sorted(missing)}")
    latest: dict[str, dict[str, str]] = {}
    for row in frame.to_dict("records"):
        key = lookup_key(row["formal_token"])
        if not key:
            continue
        effective_source = source_name
        if source_name == "seed_lookup":
            resource_name = clean(row.get("source")).casefold()
            if resource_name == "senegalese_surnames.txt":
                effective_source = "name_seed_lookup"
            elif "+lexique4.tsv" in resource_name:
                effective_source = "ambiguous_language_seed_lookup"
            elif clean(row.get("target_language")) == "fr":
                effective_source = "french_seed_lookup"
        lexical = {
            field: clean(row.get(field)) for field in LEXICAL_FIELDS
        } | {"lookup_source": effective_source}
        evidence = {}
        for source_field, evidence_field in (
            ("pos_candidates_json", "pos_candidates"),
            ("dictionary_pos_json", "dictionary_pos"),
            ("dictionary_categories_json", "dictionary_categories"),
            ("translations_json", "translations"),
            ("lemmas_json", "lemmas"),
            ("language_candidates_json", "language_candidates"),
        ):
            raw_value = clean(row.get(source_field))
            if not raw_value:
                continue
            try:
                values = json.loads(raw_value)
            except json.JSONDecodeError:
                values = []
            if isinstance(values, list):
                evidence[evidence_field] = [clean(value) for value in values if clean(value)]
        if clean(row.get("entry_count")):
            evidence["entry_count"] = clean(row.get("entry_count"))
        if clean(row.get("ambiguous_pos")):
            evidence["ambiguous_pos"] = clean(row.get("ambiguous_pos")) == "true"
        if clean(row.get("language_ambiguous")):
            evidence["language_ambiguous"] = (
                clean(row.get("language_ambiguous")) == "true"
            )
        raw_pos_by_language = clean(row.get("pos_candidates_by_language_json"))
        if raw_pos_by_language:
            try:
                parsed_pos_by_language = json.loads(raw_pos_by_language)
            except json.JSONDecodeError:
                parsed_pos_by_language = {}
            if isinstance(parsed_pos_by_language, dict):
                evidence["pos_candidates_by_language"] = {
                    clean(language): [
                        clean(value) for value in values if clean(value)
                    ]
                    for language, values in parsed_pos_by_language.items()
                    if isinstance(values, list)
                }
        lexical["lookup_evidence"] = evidence
        latest[key] = lexical
    return latest


class LinguisticAnnotationService:
    def __init__(
        self,
        *,
        gold_train_path: Path = GOLD_TRAIN_PATH,
        annotations_path: Path = LINGUISTIC_ANNOTATIONS_PATH,
        seed_lookup_path: Path = FORMAL_TOKEN_LOOKUP_SEED_PATH,
        curated_lookup_path: Path = FORMAL_TOKEN_LOOKUP_PATH,
        auto_save_safe_identities: bool = True,
    ) -> None:
        self.gold_train_path = Path(gold_train_path)
        self.annotations_path = Path(annotations_path)
        self.seed_lookup_path = Path(seed_lookup_path)
        self.curated_lookup_path = Path(curated_lookup_path)
        self.lock = threading.RLock()

        gold_train = pd.read_csv(
            self.gold_train_path, dtype={"source_index": "string"}, keep_default_na=False
        )
        self.occurrences = build_occurrences(gold_train)
        self.by_id = self.occurrences.set_index("occurrence_id").to_dict("index")
        self.latest_annotations = self._load_latest_annotations()
        self.lookup: dict[str, dict[str, object]] = {}
        self.lookup.update(load_lookup(self.seed_lookup_path, source_name="seed_lookup"))
        curated = load_lookup(self.curated_lookup_path, source_name="curated_lookup")
        for key, value in curated.items():
            seed_value = self.lookup.get(key)
            seed_evidence = (seed_value or {}).get("lookup_evidence") or {}
            if seed_value and (
                seed_evidence.get("ambiguous_pos")
                or seed_evidence.get("language_ambiguous")
            ):
                # A single contextual choice must not turn a genuinely
                # ambiguous dictionary form into a global lexical rule.
                seed_evidence = dict(seed_evidence)
                seed_evidence["prior_curated_choice"] = {
                    field: clean(value.get(field)) for field in LEXICAL_FIELDS
                }
                seed_value["lookup_evidence"] = seed_evidence
                continue
            if seed_value and not value.get("lookup_evidence"):
                evidence = dict(seed_evidence)
                # A concrete human-curated POS resolves any ambiguity recorded
                # by the seed dictionary while retaining the candidate list as
                # provenance.
                if clean(value.get("pos")) not in {"", "UNKNOWN"}:
                    evidence["ambiguous_pos"] = False
                value["lookup_evidence"] = evidence
            self.lookup[key] = value
        self.auto_saved_identity_count = 0
        self.auto_saved_spacing_merge_count = 0
        self.auto_saved_confirmed_mapping_count = 0
        if auto_save_safe_identities:
            self.auto_saved_identity_count = self._auto_save_safe_identities()
            self.auto_saved_spacing_merge_count = (
                self._auto_save_spacing_merges()
            )
            self.auto_saved_confirmed_mapping_count = (
                self._auto_save_confirmed_mapping_defaults()
            )

    @staticmethod
    def _pos_options_for_lookup(lexical: dict[str, object]) -> tuple[str, ...]:
        if not lexical:
            return POS_OPTIONS
        evidence = lexical.get("lookup_evidence") or {}
        candidates = tuple(
            option
            for option in POS_OPTIONS
            if option in set(evidence.get("pos_candidates") or [])
        )
        if (evidence.get("ambiguous_pos") or lexical.get("pos") == "UNKNOWN") and candidates:
            # Keep UNKNOWN selected until the annotator makes an explicit
            # contextual decision; HTML selects otherwise display candidate 1.
            return ("UNKNOWN", *candidates)
        pos = clean(lexical.get("pos"))
        if pos in POS_OPTIONS and pos != "UNKNOWN":
            return (pos,)
        return POS_OPTIONS

    def _auto_save_safe_identities(self, formal_key: str | None = None) -> int:
        """Persist identity occurrences needing no human lexical decision."""
        appended = 0
        with self.lock:
            for raw_occurrence_id, occurrence in self.by_id.items():
                occurrence_id = clean(raw_occurrence_id)
                if occurrence_id in self.latest_annotations:
                    continue
                informal = clean(occurrence["informal_token"])
                formal = clean(occurrence["formal_token"])
                if informal != formal:
                    continue
                if formal_key is not None and lookup_key(formal) != formal_key:
                    continue
                lexical = self.lookup.get(lookup_key(formal))
                if not lexical:
                    continue
                evidence = lexical.get("lookup_evidence") or {}
                pos = clean(lexical.get("pos"))
                lookup_source = clean(lexical.get("lookup_source"))
                curated_reviewed = (
                    lookup_source == "curated_lookup"
                    and clean(lexical.get("review_status")) == "reviewed"
                )
                if (
                    pos not in POS_OPTIONS
                    or (pos == "UNKNOWN" and not curated_reviewed)
                    or (bool(evidence.get("ambiguous_pos")) and not curated_reviewed)
                    or (bool(evidence.get("language_ambiguous")) and not curated_reviewed)
                ):
                    continue
                lemma = clean(lexical.get("lemma")) or formal
                saved_utc = datetime.now(timezone.utc).isoformat()
                language = clean(lexical.get("target_language"))
                curated_person = (
                    lookup_source == "curated_lookup"
                    and pos == "PROPN"
                    and clean(lexical.get("entity_type")) == "PERSON"
                    and clean(lexical.get("protected")) == "yes"
                )
                if (
                    language not in {"wo", "fr"}
                    and lookup_source != "name_seed_lookup"
                    and not curated_person
                    and not curated_reviewed
                ):
                    continue
                if lookup_source == "name_seed_lookup":
                    language, entity_type, protected = "unknown", "PERSON", "yes"
                elif curated_person:
                    language, entity_type, protected = "unknown", "PERSON", "yes"
                elif lookup_source == "french_seed_lookup":
                    entity_type, protected = "UNKNOWN", "yes"
                elif lookup_source == "seed_lookup":
                    entity_type, protected = "UNKNOWN", "no"
                else:
                    entity_type = clean(lexical.get("entity_type")) or "UNKNOWN"
                    protected = clean(lexical.get("protected")) or "uncertain"
                row = {
                    "occurrence_id": occurrence_id,
                    "source_index": clean(occurrence["source_index"]),
                    "token_position": int(occurrence["token_position"]),
                    "informal_token": informal,
                    "formal_token": formal,
                    "lemma": lemma,
                    "pos": pos,
                    "source_language": language,
                    "target_language": language,
                    "entity_type": entity_type,
                    "protected": protected,
                    "error_type": "identity",
                    "error_types_json": json.dumps(["identity"], ensure_ascii=False),
                    "transformation_steps_json": "[]",
                    "review_status": "reviewed",
                    "notes": (
                        "AUTO_IDENTITY_NAME_LOOKUP"
                        if lookup_source == "name_seed_lookup"
                        else "AUTO_IDENTITY_LOOKUP"
                    ),
                    "lexical_reusable": "false",
                    "lookup_source": clean(lexical.get("lookup_source")) or "seed_lookup",
                    "saved_utc": saved_utc,
                }
                append_row(self.annotations_path, row, OCCURRENCE_COLUMNS)
                self.latest_annotations[occurrence_id] = {
                    key: clean(value) for key, value in row.items()
                }
                appended += 1
        return appended

    def _auto_save_spacing_merges(self) -> int:
        """Persist unannotated spacing merges using the approved Wolof policy."""
        appended = 0
        with self.lock:
            for raw_occurrence_id, occurrence in self.by_id.items():
                occurrence_id = clean(raw_occurrence_id)
                if occurrence_id in self.latest_annotations:
                    continue
                informal = clean(occurrence["informal_token"])
                formal = clean(occurrence["formal_token"])
                error_types = infer_error_types(informal, formal)
                if "spacing_merge" not in error_types:
                    continue
                row = {
                    "occurrence_id": occurrence_id,
                    "source_index": clean(occurrence["source_index"]),
                    "token_position": int(occurrence["token_position"]),
                    "informal_token": informal,
                    "formal_token": formal,
                    # A normalized multiword unit does not have a reliable
                    # single-token lemma or POS without additional analysis.
                    "lemma": "",
                    "pos": "UNKNOWN",
                    "source_language": "wo",
                    "target_language": "wo",
                    "entity_type": "UNKNOWN",
                    "protected": "no",
                    "error_type": error_types[0],
                    "error_types_json": json.dumps(error_types, ensure_ascii=False),
                    "transformation_steps_json": json.dumps(
                        derive_transformations(formal, informal), ensure_ascii=False
                    ),
                    "review_status": "reviewed",
                    "notes": "AUTO_SPACING_MERGE",
                    "lexical_reusable": "false",
                    "lookup_source": "spacing_merge_policy",
                    "saved_utc": datetime.now(timezone.utc).isoformat(),
                }
                append_row(self.annotations_path, row, OCCURRENCE_COLUMNS)
                self.latest_annotations[occurrence_id] = {
                    key: clean(value) for key, value in row.items()
                }
                appended += 1
        return appended

    @staticmethod
    def _default_signature(row: dict[str, object]) -> tuple[object, ...]:
        """Properties that must agree before an occurrence default is trusted."""
        return (
            clean(row.get("lemma")),
            clean(row.get("pos")),
            clean(row.get("source_language")),
            clean(row.get("target_language")),
            clean(row.get("entity_type")),
            clean(row.get("protected")),
            parse_error_types(row.get("error_types_json"), row.get("error_type")),
        )

    def _confirmed_default_for_mapping(
        self, informal: str, formal: str
    ) -> dict[str, str] | None:
        confirmations = [
            row
            for row in self.latest_annotations.values()
            if clean(row.get("informal_token")) == informal
            and clean(row.get("formal_token")) == formal
            and clean(row.get("review_status")) == "reviewed"
            and not clean(row.get("notes")).startswith("AUTO_")
        ]
        signatures: dict[tuple[object, ...], list[dict[str, str]]] = {}
        for row in confirmations:
            signatures.setdefault(self._default_signature(row), []).append(row)
        ranked = sorted(signatures.values(), key=len, reverse=True)
        if not ranked or len(ranked[0]) < 2:
            return None
        # Do not select an arbitrary default when two competing annotations
        # have equal support.
        if len(ranked) > 1 and len(ranked[0]) == len(ranked[1]):
            return None
        return ranked[0][-1]

    def _auto_save_confirmed_mapping_defaults(
        self, mapping: tuple[str, str] | None = None
    ) -> int:
        """Propagate an exact mapping after two matching human confirmations."""
        if mapping is None:
            mappings = {
                (
                    clean(occurrence["informal_token"]),
                    clean(occurrence["formal_token"]),
                )
                for occurrence in self.by_id.values()
            }
        else:
            mappings = {(clean(mapping[0]), clean(mapping[1]))}

        appended = 0
        with self.lock:
            for informal, formal in mappings:
                default = self._confirmed_default_for_mapping(informal, formal)
                if default is None:
                    continue
                error_types = parse_error_types(
                    default.get("error_types_json"), default.get("error_type")
                )
                if not error_types:
                    continue
                for raw_occurrence_id, occurrence in self.by_id.items():
                    occurrence_id = clean(raw_occurrence_id)
                    if occurrence_id in self.latest_annotations:
                        continue
                    if clean(occurrence["informal_token"]) != informal:
                        continue
                    if clean(occurrence["formal_token"]) != formal:
                        continue
                    row = {
                        "occurrence_id": occurrence_id,
                        "source_index": clean(occurrence["source_index"]),
                        "token_position": int(occurrence["token_position"]),
                        "informal_token": informal,
                        "formal_token": formal,
                        "lemma": clean(default.get("lemma")),
                        "pos": clean(default.get("pos")),
                        "source_language": clean(default.get("source_language")),
                        "target_language": clean(default.get("target_language")),
                        "entity_type": clean(default.get("entity_type")),
                        "protected": clean(default.get("protected")),
                        "error_type": error_types[0],
                        "error_types_json": json.dumps(error_types, ensure_ascii=False),
                        "transformation_steps_json": json.dumps(
                            derive_transformations(formal, informal), ensure_ascii=False
                        ),
                        "review_status": "reviewed",
                        "notes": "AUTO_CONFIRMED_MAPPING_DEFAULT",
                        "lexical_reusable": "false",
                        "lookup_source": "confirmed_mapping_default",
                        "saved_utc": datetime.now(timezone.utc).isoformat(),
                    }
                    append_row(self.annotations_path, row, OCCURRENCE_COLUMNS)
                    self.latest_annotations[occurrence_id] = {
                        key: clean(value) for key, value in row.items()
                    }
                    appended += 1
        return appended

    def _load_latest_annotations(self) -> dict[str, dict[str, str]]:
        frame = read_optional_csv(self.annotations_path)
        if frame.empty:
            return {}
        optional_new = {"error_types_json", "transformation_steps_json"}
        missing = (set(OCCURRENCE_COLUMNS) - optional_new) - set(frame.columns)
        if missing:
            raise ValueError(
                f"{self.annotations_path.name} is missing columns: {sorted(missing)}"
            )
        schema_upgraded = False
        if "error_types_json" not in frame.columns:
            frame["error_types_json"] = frame["error_type"].map(
                lambda value: json.dumps([clean(value)], ensure_ascii=False)
            )
            schema_upgraded = True
        if "transformation_steps_json" not in frame.columns:
            frame["transformation_steps_json"] = frame.apply(
                lambda row: json.dumps(
                    derive_transformations(row["formal_token"], row["informal_token"]),
                    ensure_ascii=False,
                ),
                axis=1,
            )
            schema_upgraded = True
        if schema_upgraded:
            frame = frame[[*OCCURRENCE_COLUMNS]]
            frame.to_csv(self.annotations_path, index=False, encoding="utf-8")
        latest: dict[str, dict[str, str]] = {}
        for row in frame.to_dict("records"):
            occurrence_id = clean(row.get("occurrence_id"))
            if occurrence_id:
                latest[occurrence_id] = {key: clean(value) for key, value in row.items()}
        return latest

    def progress(self) -> dict[str, int]:
        valid_ids = set(self.by_id)
        saved = len(valid_ids & set(self.latest_annotations))
        formal_keys = {lookup_key(value) for value in self.occurrences["formal_token"]}
        covered = len(formal_keys & set(self.lookup))
        return {
            "total": len(self.occurrences),
            "saved": saved,
            "remaining": len(self.occurrences) - saved,
            "unique_formal_tokens": len(formal_keys),
            "lookup_covered": covered,
            "lookup_missing": len(formal_keys) - covered,
            "auto_saved_identity_on_startup": self.auto_saved_identity_count,
            "auto_saved_spacing_merge_on_startup": (
                self.auto_saved_spacing_merge_count
            ),
            "auto_saved_confirmed_mapping_on_startup": (
                self.auto_saved_confirmed_mapping_count
            ),
        }

    def next_unannotated_position(self, after: int = -1) -> int | None:
        total = len(self.occurrences)
        if not total:
            return None
        start = (int(after) + 1) % total
        for offset in range(total):
            position = (start + offset) % total
            occurrence_id = self.occurrences.iloc[position]["occurrence_id"]
            if occurrence_id not in self.latest_annotations:
                return position
        return None

    def record_at(self, position: int) -> dict[str, object]:
        position = int(position)
        if position < 0 or position >= len(self.occurrences):
            raise IndexError("Token occurrence position is outside the gold training data")
        occurrence = self.occurrences.iloc[position].to_dict()
        occurrence_id = occurrence["occurrence_id"]
        existing = self.latest_annotations.get(occurrence_id)
        lexical = self.lookup.get(lookup_key(occurrence["formal_token"]), {})
        transformations = derive_transformations(
            clean(occurrence["formal_token"]), clean(occurrence["informal_token"])
        )
        automatic_error_types = infer_error_types(
            clean(occurrence["informal_token"]), clean(occurrence["formal_token"])
        )

        prefill = {
            "lemma": clean(occurrence["formal_token"]) if " " not in clean(occurrence["formal_token"]) else "",
            "pos": "UNKNOWN",
            "source_language": "unknown",
            "target_language": "unknown",
            "entity_type": "UNKNOWN",
            "protected": "uncertain",
            "error_types": automatic_error_types,
            "review_status": "reviewed",
            "notes": "",
            "lexical_reusable": True,
            "lookup_source": "none",
            "lookup_evidence": {},
        }
        if lexical:
            prefill.update(lexical)
            prefill["source_language"] = lexical.get("target_language", "unknown")
            if lexical.get("lookup_source") == "ambiguous_language_seed_lookup":
                prefill["target_language"] = "unknown"
                prefill["source_language"] = "unknown"
                prefill["entity_type"] = "UNKNOWN"
                prefill["protected"] = "uncertain"
                prefill["review_status"] = "uncertain"
            elif lexical.get("lookup_source") == "seed_lookup":
                # The supplied resource is a Wolof dictionary. Its definitions
                # are written in French, but the headword language is Wolof.
                prefill["target_language"] = "wo"
                prefill["source_language"] = "wo"
                prefill["entity_type"] = "UNKNOWN"
                prefill["protected"] = "no"
            elif lexical.get("lookup_source") == "name_seed_lookup":
                prefill["target_language"] = "unknown"
                prefill["source_language"] = "unknown"
                prefill["pos"] = "PROPN"
                prefill["entity_type"] = "PERSON"
                prefill["protected"] = "yes"
            elif lexical.get("lookup_source") == "french_seed_lookup":
                prefill["target_language"] = "fr"
                prefill["source_language"] = "fr"
                prefill["entity_type"] = "UNKNOWN"
                prefill["protected"] = "yes"
            if (
                prefill.get("target_language") == "fr"
                and clean(occurrence["informal_token"])
                != clean(occurrence["formal_token"])
            ):
                prefill["error_types"] = _ordered_error_types(
                    set(prefill["error_types"]) | {"french_spelling"}
                )
            if lexical.get("lookup_source") != "ambiguous_language_seed_lookup":
                prefill["review_status"] = "reviewed"
        if existing:
            prefill.update(
                {
                    key: existing.get(key, prefill.get(key, ""))
                    for key in (
                        "lemma", "pos", "source_language", "target_language",
                        "entity_type", "protected", "review_status",
                        "notes", "lookup_source",
                    )
                }
            )
            prefill["error_types"] = parse_error_types(
                existing.get("error_types_json"), existing.get("error_type")
            ) or automatic_error_types
            prefill["lexical_reusable"] = existing.get("lexical_reusable") == "true"
            # The marker belongs to the automated history row, not to a human
            # annotation.  Hiding it here means a subsequent manual save no
            # longer looks like an untouched automatic decision.
            if clean(existing.get("notes")).startswith("AUTO_"):
                prefill["notes"] = ""

        lookup_evidence = prefill.get("lookup_evidence") or {}
        language_ambiguous = bool(lookup_evidence.get("language_ambiguous"))
        seed_ambiguous = bool(lexical) and clean(
            lexical.get("lookup_source")
        ) != "curated_lookup" and (
            bool(lookup_evidence.get("ambiguous_pos")) or language_ambiguous
        )
        pos_editable = (
            not lexical
            or prefill.get("pos") == "UNKNOWN"
            or bool(lookup_evidence.get("ambiguous_pos"))
        )
        lexical_fields_locked = bool(lexical)
        is_identity = clean(occurrence["informal_token"]) == clean(
            occurrence["formal_token"]
        )
        identity_lookup = is_identity and lexical_fields_locked
        french_lookup = bool(lexical) and clean(lexical.get("target_language")) == "fr"
        name_lookup = bool(lexical) and lexical.get("lookup_source") == "name_seed_lookup"
        pos_options = self._pos_options_for_lookup(lexical)
        if lexical_fields_locked and not pos_editable and not existing:
            prefill["lexical_reusable"] = False
        if seed_ambiguous:
            prefill["lexical_reusable"] = False

        language_candidates = tuple(
            value
            for value in LANGUAGE_OPTIONS
            if value in set(lookup_evidence.get("language_candidates") or [])
        )
        language_options = (
            ("unknown", *language_candidates)
            if language_ambiguous and language_candidates
            else LANGUAGE_OPTIONS
        )

        return {
            "position": position,
            **occurrence,
            "changed": bool(occurrence["changed"]),
            "already_annotated": existing is not None,
            "lexical_fields_locked": lexical_fields_locked,
            "lemma_editable": not lexical,
            "pos_editable": pos_editable,
            "identity_lookup": identity_lookup,
            "target_language_editable": not lexical or language_ambiguous,
            "entity_editable": not lexical or language_ambiguous,
            "protected_editable": not lexical or language_ambiguous,
            "reusable_allowed": not seed_ambiguous,
            "source_language_locked": (
                (identity_lookup and not language_ambiguous)
                or french_lookup
                or name_lookup
            ),
            "transformation_direction": "formal_to_informal",
            "transformation_steps": transformations,
            "prefill": prefill,
            "options": {
                "pos": pos_options,
                "language": language_options,
                "entity": ENTITY_OPTIONS,
                "protected": PROTECTED_OPTIONS,
                "error": ("identity",) if is_identity else ERROR_OPTIONS,
                "error_definitions": ERROR_DEFINITIONS,
                "review": REVIEW_OPTIONS,
            },
        }

    @staticmethod
    def _require_choice(payload: dict[str, object], field: str, choices: tuple[str, ...]) -> str:
        value = clean(payload.get(field))
        if value not in choices:
            raise ValueError(f"Invalid {field}: {value!r}")
        return value

    def save(self, payload: dict[str, object]) -> dict[str, object]:
        occurrence_id = clean(payload.get("occurrence_id"))
        occurrence = self.by_id.get(occurrence_id)
        if occurrence is None:
            raise ValueError("Unknown gold-train token occurrence")
        if clean(payload.get("informal_token")) != clean(occurrence["informal_token"]):
            raise ValueError("Submitted informal token does not match the gold split")
        if clean(payload.get("formal_token")) != clean(occurrence["formal_token"]):
            raise ValueError("Submitted formal token does not match the gold split")

        submitted_error_types = payload.get("error_types")
        if not isinstance(submitted_error_types, list):
            legacy_error = clean(payload.get("error_type"))
            submitted_error_types = [legacy_error] if legacy_error else []
        invalid_error_types = {
            clean(value) for value in submitted_error_types if clean(value) not in ERROR_OPTIONS
        }
        if invalid_error_types:
            raise ValueError(f"Invalid error types: {sorted(invalid_error_types)}")
        selected_error_types = _ordered_error_types(
            {clean(value) for value in submitted_error_types}
        )
        if not selected_error_types:
            raise ValueError("Select at least one error type")
        is_identity = clean(occurrence["informal_token"]) == clean(
            occurrence["formal_token"]
        )
        if "identity" in selected_error_types and len(selected_error_types) > 1:
            raise ValueError("Identity cannot be combined with another error type")
        if "identity" in selected_error_types and not is_identity:
            raise ValueError("Identity cannot label a changed token occurrence")
        if is_identity and selected_error_types != ("identity",):
            raise ValueError("An unchanged token occurrence must use identity")

        transformations = derive_transformations(
            clean(occurrence["formal_token"]), clean(occurrence["informal_token"])
        )

        lexical = self.lookup.get(lookup_key(occurrence["formal_token"]), {})
        lookup_evidence = lexical.get("lookup_evidence") or {}
        language_ambiguous = bool(lookup_evidence.get("language_ambiguous"))
        seed_ambiguous = bool(lexical) and clean(
            lexical.get("lookup_source")
        ) != "curated_lookup" and (
            bool(lookup_evidence.get("ambiguous_pos")) or language_ambiguous
        )
        pos_editable = (
            not lexical
            or lexical.get("pos") == "UNKNOWN"
            or bool(lookup_evidence.get("ambiguous_pos"))
        )
        submitted_pos = self._require_choice(payload, "pos", POS_OPTIONS)
        allowed_pos_options = self._pos_options_for_lookup(lexical)
        if submitted_pos not in allowed_pos_options:
            raise ValueError(
                f"POS {submitted_pos!r} is not one of the lookup candidates: "
                f"{list(allowed_pos_options)}"
            )
        submitted_source_language = self._require_choice(
            payload, "source_language", LANGUAGE_OPTIONS
        )
        submitted_target_language = self._require_choice(
            payload, "target_language", LANGUAGE_OPTIONS
        )
        submitted_entity_type = self._require_choice(
            payload, "entity_type", ENTITY_OPTIONS
        )
        submitted_protected = self._require_choice(
            payload, "protected", PROTECTED_OPTIONS
        )
        if submitted_entity_type == "PERSON":
            if "PROPN" not in allowed_pos_options:
                raise ValueError("A PERSON entity requires PROPN in the allowed POS choices")
            submitted_pos = "PROPN"
            submitted_protected = "yes"
        if lexical:
            lemma = clean(lexical.get("lemma"))
            pos = submitted_pos if pos_editable else clean(lexical.get("pos"))
            if lexical.get("lookup_source") == "ambiguous_language_seed_lookup":
                candidates = set(lookup_evidence.get("language_candidates") or [])
                if submitted_target_language not in candidates:
                    raise ValueError(
                        "Select one of the dictionary language candidates: "
                        f"{sorted(candidates)}"
                    )
                pos_by_language = lookup_evidence.get(
                    "pos_candidates_by_language"
                ) or {}
                language_pos = set(
                    pos_by_language.get(submitted_target_language) or []
                )
                if (
                    submitted_pos != "UNKNOWN"
                    and language_pos
                    and submitted_pos not in language_pos
                ):
                    raise ValueError(
                        f"POS {submitted_pos!r} is not valid for selected "
                        f"language {submitted_target_language!r}: "
                        f"{sorted(language_pos)}"
                    )
                target_language = submitted_target_language
                entity_type = submitted_entity_type
                protected = submitted_protected
            elif lexical.get("lookup_source") == "seed_lookup":
                target_language = "wo"
                entity_type = "UNKNOWN"
                protected = "no"
            elif lexical.get("lookup_source") == "name_seed_lookup":
                pos = "PROPN"
                target_language = "unknown"
                entity_type = "PERSON"
                protected = "yes"
            elif lexical.get("lookup_source") == "french_seed_lookup":
                target_language = "fr"
                entity_type = "UNKNOWN"
                protected = "yes"
            else:
                target_language = clean(lexical.get("target_language"))
                entity_type = clean(lexical.get("entity_type"))
                protected = clean(lexical.get("protected"))
            lookup_source = clean(lexical.get("lookup_source"))
        else:
            lemma = clean(payload.get("lemma"))
            pos = submitted_pos
            target_language = submitted_target_language
            entity_type = submitted_entity_type
            protected = submitted_protected
            lookup_source = "manual"
        identity_lookup = is_identity and bool(lexical)
        person_name_lookup = bool(lexical) and (
            lexical.get("lookup_source") == "name_seed_lookup"
            or (
                clean(lexical.get("pos")) == "PROPN"
                and clean(lexical.get("target_language")) == "unknown"
                and clean(lexical.get("entity_type")) == "PERSON"
                and clean(lexical.get("protected")) == "yes"
            )
        )
        if language_ambiguous and identity_lookup:
            source_language = target_language
        elif person_name_lookup:
            source_language = "unknown"
        elif lexical and clean(lexical.get("target_language")) == "fr":
            source_language = "fr"
        elif identity_lookup:
            source_language = "wo"
        else:
            source_language = submitted_source_language
        review_status = (
            "reviewed"
            if identity_lookup and not seed_ambiguous
            else self._require_choice(payload, "review_status", REVIEW_OPTIONS)
        )
        lexical_reusable = bool(payload.get("lexical_reusable")) and (
            not lexical or pos_editable
        ) and not seed_ambiguous

        saved_utc = datetime.now(timezone.utc).isoformat()
        row = {
            "occurrence_id": occurrence_id,
            "source_index": clean(occurrence["source_index"]),
            "token_position": int(occurrence["token_position"]),
            "informal_token": clean(occurrence["informal_token"]),
            "formal_token": clean(occurrence["formal_token"]),
            "lemma": lemma,
            "pos": pos,
            "source_language": source_language,
            "target_language": target_language,
            "entity_type": entity_type,
            "protected": protected,
            # Keep a primary value for compatibility while the JSON list is the
            # authoritative multi-label representation.
            "error_type": selected_error_types[0],
            "error_types_json": json.dumps(selected_error_types, ensure_ascii=False),
            "transformation_steps_json": json.dumps(transformations, ensure_ascii=False),
            "review_status": review_status,
            "notes": clean(payload.get("notes")),
            "lexical_reusable": str(lexical_reusable).lower(),
            "lookup_source": lookup_source,
            "saved_utc": saved_utc,
        }
        if not row["lemma"] and row["pos"] not in {"PUNCT", "SYM", "X", "UNKNOWN"}:
            raise ValueError("Lemma is required for a classified lexical token")

        lookup_update = None
        with self.lock:
            append_row(self.annotations_path, row, OCCURRENCE_COLUMNS)
            self.latest_annotations[occurrence_id] = {key: clean(value) for key, value in row.items()}
            if lexical_reusable:
                lookup_row = {
                    "formal_token": row["formal_token"],
                    "lemma": row["lemma"],
                    "pos": row["pos"],
                    "target_language": row["target_language"],
                    "entity_type": row["entity_type"],
                    "protected": row["protected"],
                    "review_status": row["review_status"],
                    "source": "manual_occurrence",
                    "source_index": row["source_index"],
                    "updated_utc": saved_utc,
                }
                append_row(self.curated_lookup_path, lookup_row, LOOKUP_COLUMNS)
                lookup_update: dict[str, object] = {
                    field: row[field] for field in LEXICAL_FIELDS
                } | {"lookup_source": "curated_lookup"}
                lookup_update["lookup_evidence"] = self.lookup.get(
                    lookup_key(row["formal_token"]), {}
                ).get("lookup_evidence", {})
                self.lookup[lookup_key(row["formal_token"])] = lookup_update

        propagated_identity_count = 0
        if lookup_update is not None:
            propagated_identity_count = self._auto_save_safe_identities(
                lookup_key(row["formal_token"])
            )
        propagated_confirmed_mapping_count = (
            self._auto_save_confirmed_mapping_defaults(
                (row["informal_token"], row["formal_token"])
            )
        )

        return {
            "lookup_update": lookup_update,
            "propagated_identity_count": propagated_identity_count,
            "propagated_confirmed_mapping_count": (
                propagated_confirmed_mapping_count
            ),
            "progress": self.progress(),
        }
