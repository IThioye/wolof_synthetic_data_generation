import csv
import difflib
import json
import re
import threading
from datetime import datetime, timezone

from functools import lru_cache
import pandas as pd

from pathlib import Path
import sys
external_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(external_dir))


from config import (
    ANNOTATION_FRENCH_AMBIGUITY_MARGIN,
    ANNOTATION_TARGET_KEPT,
    ANNOTATION_MAX_DISTANCE_RATIO,
    ANNOTATION_MIN_CANDIDATE_MARGIN,
    BASE_GOLD_ANNOTATIONS_PATH,
    BASE_TOKEN_CORRECTIONS_PATH,
    CHECKPOINT_DATA_PATH as CHECKPOINT_PATH,
    CLEAN_COMMENTS_PATH,
    FILTERED_COMMENTS_PATH as DEFAULT_COMMENTS_PATH,
    FRENCH_WORDLIST_PATH,
    FORMAL_TOKEN_LOOKUP_PATH,
    GOLD_ANNOTATIONS_PATH as GOLD_OUTPUT_PATH,
    MIN_GOLD_KEPT,
    MIN_GOLD_VIDEOS,
    SENEGALESE_SURNAMES_PATH,
    SENTENCE_ANNOTATION_EVENTS_PATH,
    TOKEN_CORRECTIONS_PATH as TOKEN_OUTPUT_PATH,
    WOLOF_LOOKUP_SOURCE_PATH,
)

try:
    from .lookup_suggestions import LookupSuggestionIndex
except ImportError:
    from annotation.lookup_suggestions import LookupSuggestionIndex

GOLD_COLUMNS = [
    "source_index",
    "comment",
    "auto_suggestion",
    "manual_formal_wolof",
    "status",
    "video_url",
    "token_corrections_json",
]

TOKEN_COLUMNS = [
    "source_index",
    "informal_token",
    "formal_token",
    "category",
    "comment",
]

EVENT_COLUMNS = [
    "saved_utc",
    "source_index",
    "status",
    "suggestion_action",
    "elapsed_ms",
    "source_word_count",
    "target_word_count",
    "changed_token_count",
    "output_file",
]

VALID_STATUSES = {
    "keep",
    "discard_false_positive",
    "discard_uninteresting",
    "skip_uncertain",
}


def interleave_comments_by_video(comments):
    """Round-robin comments across videos while keeping a deterministic order."""
    if comments.empty or "video_url" not in comments.columns:
        return comments.reset_index(drop=True)

    ordered = comments.copy()
    missing_video = ordered["video_url"].isna() | ordered["video_url"].eq("")
    ordered["_video_group"] = ordered["video_url"].astype(str)
    ordered.loc[missing_video, "_video_group"] = ordered.loc[
        missing_video, "source_index"
    ].map(lambda value: f"missing-video-{value}")
    ordered["_video_order"] = pd.factorize(ordered["_video_group"], sort=False)[0]
    ordered["_within_video"] = ordered.groupby("_video_group", sort=False).cumcount()
    ordered = ordered.sort_values(
        ["_within_video", "_video_order", "source_index"], kind="stable"
    )
    return ordered.drop(
        columns=["_video_group", "_video_order", "_within_video"]
    ).reset_index(drop=True)


def load_comments():
    if CHECKPOINT_PATH.exists() and CLEAN_COMMENTS_PATH.exists():
        clean_df = pd.read_csv(CLEAN_COMMENTS_PATH).reset_index()
        checkpoint_df = pd.read_csv(CHECKPOINT_PATH)
        merged = clean_df.merge(checkpoint_df, on="index", how="inner")
        merged = merged.loc[merged["is_informally_code_switched"] == True].copy()
        merged = merged.rename(columns={"index": "source_index"})
        selected = merged[["source_index", "clean_comment", "video_url"]]
        return interleave_comments_by_video(selected)

    df = pd.read_csv(DEFAULT_COMMENTS_PATH)
    if "source_index" not in df.columns:
        df = df.reset_index().rename(columns={"index": "source_index"})
    if "video_url" not in df.columns:
        df["video_url"] = ""
    return interleave_comments_by_video(
        df[["source_index", "clean_comment", "video_url"]]
    )


def _load_csv_if_present(path):
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, keep_default_na=False)
    except (pd.errors.EmptyDataError, UnicodeDecodeError):
        return pd.DataFrame()


def load_existing_annotations():
    """Load canonical history plus an optional active campaign history.

    When ``PFE_GOLD_ANNOTATIONS_PATH`` points at a new campaign file, rows that
    were already reviewed in the canonical history remain skipped.  A newer
    campaign row wins if a source row is deliberately revisited.
    """
    paths = [BASE_GOLD_ANNOTATIONS_PATH]
    if GOLD_OUTPUT_PATH.resolve() != BASE_GOLD_ANNOTATIONS_PATH.resolve():
        paths.append(GOLD_OUTPUT_PATH)
    frames = [_load_csv_if_present(path) for path in paths]
    frames = [frame for frame in frames if "source_index" in frame.columns]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _add_equivalent(learned, informal, formal):
    informal = str(informal).strip()
    formal = str(formal).strip()
    if (
        not informal
        or not formal
        or informal.casefold() == "nan"
        or formal.casefold() == "nan"
        or informal == formal
    ):
        return
    equivalents = learned.setdefault(informal.casefold(), [])
    if formal not in equivalents:
        equivalents.append(formal)


def load_learned_corrections(existing_annotations=None):
    """Load newest-first equivalents without discarding historical CSV work."""
    learned = {}

    token_paths = [BASE_TOKEN_CORRECTIONS_PATH]
    if TOKEN_OUTPUT_PATH.resolve() != BASE_TOKEN_CORRECTIONS_PATH.resolve():
        token_paths.append(TOKEN_OUTPUT_PATH)
    correction_frames = [_load_csv_if_present(path) for path in token_paths]
    correction_frames = [
        frame
        for frame in correction_frames
        if {"informal_token", "formal_token"}.issubset(frame.columns)
    ]
    corrections = (
        pd.concat(correction_frames, ignore_index=True)
        if correction_frames
        else pd.DataFrame()
    )
    if not corrections.empty:
        if {"informal_token", "formal_token"}.issubset(corrections.columns):
            for row in corrections.iloc[::-1].itertuples(index=False):
                _add_equivalent(learned, row.informal_token, row.formal_token)

    # Recover any corrections present in gold history but absent from the token file.
    if existing_annotations is None:
        existing_annotations = load_existing_annotations()
    if not existing_annotations.empty and "token_corrections_json" in existing_annotations.columns:
        for row in existing_annotations.iloc[::-1].to_dict("records"):
            if row.get("status") != "keep":
                continue
            try:
                payload = json.loads(row.get("token_corrections_json") or "[]")
            except (TypeError, json.JSONDecodeError):
                continue
            for correction in payload:
                _add_equivalent(
                    learned,
                    correction.get("token", ""),
                    correction.get("final_correction", ""),
                )

    return learned


def load_lookup_suggestion_index():
    return LookupSuggestionIndex.from_files(
        WOLOF_LOOKUP_SOURCE_PATH,
        FRENCH_WORDLIST_PATH,
        FORMAL_TOKEN_LOOKUP_PATH,
        SENEGALESE_SURNAMES_PATH,
        max_distance_ratio=ANNOTATION_MAX_DISTANCE_RATIO,
        min_candidate_margin=ANNOTATION_MIN_CANDIDATE_MARGIN,
        french_ambiguity_margin=ANNOTATION_FRENCH_AMBIGUITY_MARGIN,
    )


def append_dict(path, row, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with open(path, "a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


WORD_PATTERN = re.compile(r"\w+(?:[-'’]\w+)*", flags=re.UNICODE)


def surface_word_tokens(text):
    """Return lexical tokens without lowercasing or discarding their surface form."""
    return [match.group(0) for match in WORD_PATTERN.finditer(str(text))]


def replace_surface_words(text, replacements):
    """Replace lexical spans while preserving the source punctuation and spacing."""
    matches = list(WORD_PATTERN.finditer(str(text)))
    if len(matches) != len(replacements):
        raise ValueError("Replacement count does not match the source word count")
    pieces = []
    cursor = 0
    for match, replacement in zip(matches, replacements):
        pieces.extend((str(text)[cursor : match.start()], str(replacement)))
        cursor = match.end()
    pieces.append(str(text)[cursor:])
    return "".join(pieces)


def _distribute_replacement(source_count, target_tokens):
    """Allocate a replacement block while preserving exact target word order."""
    if source_count <= 0:
        return []
    if not target_tokens:
        return [""] * source_count
    if source_count == 1:
        return [" ".join(target_tokens)]
    if len(target_tokens) == 1:
        return [target_tokens[0], *([""] * (source_count - 1))]

    allocated = []
    cursor = 0
    for source_offset in range(source_count):
        remaining_sources = source_count - source_offset
        remaining_targets = len(target_tokens) - cursor
        take = max(1, remaining_targets - (remaining_sources - 1))
        if remaining_targets <= 0:
            allocated.append("")
            continue
        allocated.append(" ".join(target_tokens[cursor : cursor + take]))
        cursor += take
    return allocated


def align_sentence_to_source(source_tokens, target_sentence):
    """Map a reviewed target sentence back to source-token correction slots.

    The mapping is intended for annotation evidence, not for scoring.  Its
    output always reconstructs the target *word sequence*.  Inserted words are
    attached to the preceding source token (or the next token at sentence
    start), and unequal replacement blocks are distributed monotonically.
    """
    source_tokens = [str(token) for token in source_tokens]
    target_tokens = surface_word_tokens(target_sentence)
    aligned = list(source_tokens)
    prefixes = [""] * len(source_tokens)
    suffixes = [""] * len(source_tokens)
    matcher = difflib.SequenceMatcher(
        a=[token.casefold() for token in source_tokens],
        b=[token.casefold() for token in target_tokens],
        autojunk=False,
    )
    for tag, source_start, source_end, target_start, target_end in matcher.get_opcodes():
        if tag == "equal":
            for offset, target_token in enumerate(target_tokens[target_start:target_end]):
                aligned[source_start + offset] = target_token
        elif tag == "delete":
            aligned[source_start:source_end] = [""] * (source_end - source_start)
        elif tag == "replace":
            aligned[source_start:source_end] = _distribute_replacement(
                source_end - source_start,
                target_tokens[target_start:target_end],
            )
        elif tag == "insert":
            inserted = " ".join(target_tokens[target_start:target_end])
            if source_start > 0:
                suffixes[source_start - 1] = " ".join(
                    part for part in (suffixes[source_start - 1], inserted) if part
                )
            elif aligned:
                prefixes[0] = " ".join(part for part in (inserted, prefixes[0]) if part)
    return [
        " ".join(part for part in (prefix, token, suffix) if part)
        for prefix, token, suffix in zip(prefixes, aligned, suffixes)
    ]


def classify_suggestion_action(comment, auto_sentence, manual_sentence):
    normalize = lambda value: " ".join(str(value).split()).casefold()
    manual = normalize(manual_sentence)
    if manual == normalize(comment):
        return "kept_source"
    if manual == normalize(auto_sentence):
        return "accepted_suggestion"
    return "edited"


class AnnotationService:
    def __init__(self):
        self.comments = load_comments()
        self.lookup_index = load_lookup_suggestion_index()
        self.lock = threading.RLock()

        existing = load_existing_annotations()
        active_existing = _load_csv_if_present(GOLD_OUTPUT_PATH)
        self.campaign_annotated_ids = (
            set(active_existing["source_index"].astype(int))
            if "source_index" in active_existing.columns
            else set()
        )
        if not existing.empty:
            existing = existing.copy()
            existing["source_index"] = existing["source_index"].astype(int)
            self.annotated_ids = set(existing["source_index"])
            self.latest_by_id = (
                existing.drop_duplicates("source_index", keep="last")
                .set_index("source_index")
                .to_dict("index")
            )
        else:
            self.annotated_ids = set()
            self.latest_by_id = {}
        self.learned_corrections = load_learned_corrections(existing)

    @lru_cache(maxsize=50_000)
    def base_token_suggestions(self, token):
        return self.lookup_index.suggest(token)

    def analyze(self, text):
        source_tokens = surface_word_tokens(text)
        rows = []
        auto_tokens = []

        for token in source_tokens:
            lookup_token = token.casefold()
            with self.lock:
                remembered = list(self.learned_corrections.get(lookup_token, []))
            if len(token) > 1 and token.isupper():
                rows.append(
                    {
                        "token": token,
                        "category": "protected",
                        "suggestions": [token],
                        "auto": token,
                        "learned": bool(remembered),
                        "changed": False,
                        "accepted": False,
                        "review_required": False,
                        "reason": "Uppercase acronym or protected form",
                        "candidate_details": {},
                    }
                )
                auto_tokens.append(token)
                continue
            decision = self.base_token_suggestions(lookup_token)
            guarded = decision.category in {
                "protected",
                "french_exact",
                "ambiguous_exact",
                "possible_french",
                "ambiguous_near",
            }
            if remembered and not guarded and len(remembered) == 1:
                category = "learned_correction"
                auto = remembered[0]
                accepted = True
                review_required = False
                reason = "One previously confirmed human correction"
            elif remembered and not guarded:
                category = "ambiguous_learned"
                auto = token
                accepted = False
                review_required = True
                reason = "Earlier annotations contain several corrections for this token"
            else:
                category = decision.category
                auto = token if guarded else decision.auto
                accepted = decision.accepted and not guarded
                review_required = decision.review_required
                reason = decision.reason

            suggestions = [token]
            suggestions.extend(remembered)
            if decision.french_candidate:
                suggestions.append(decision.french_candidate)
            suggestions.extend(candidate.word for candidate in decision.candidates)
            if auto not in suggestions:
                suggestions.insert(1, auto)
            suggestions = list(dict.fromkeys(suggestions))[:8]
            candidate_details = {
                candidate.word: candidate.as_dict()
                for candidate in decision.candidates
            }
            if decision.french_candidate and decision.french_score is not None:
                candidate_details[decision.french_candidate] = {
                    "word": decision.french_candidate,
                    "score": round(decision.french_score, 4),
                    "pos": "French lookup",
                    "definition": "Review only; never applied automatically",
                }

            rows.append(
                {
                    "token": token,
                    "category": category,
                    "suggestions": suggestions[:8],
                    "auto": auto,
                    "learned": bool(remembered),
                    "changed": auto != token,
                    "accepted": accepted,
                    "review_required": review_required,
                    "reason": reason,
                    "candidate_details": candidate_details,
                }
            )
            auto_tokens.append(auto)
        return replace_surface_words(text, auto_tokens), rows

    def progress(self):
        with self.lock:
            total = len(self.comments)
            saved = sum(
                int(source_index) in self.annotated_ids
                for source_index in self.comments["source_index"]
            )
            eligible_kept = [
                row
                for row in self.latest_by_id.values()
                if row.get("status") == "keep"
                and str(row.get("manual_formal_wolof", "")).strip()
            ]
            kept = len(eligible_kept)
            campaign_ids = getattr(self, "campaign_annotated_ids", self.annotated_ids)
            campaign_saved = sum(
                int(source_index) in campaign_ids
                for source_index in self.comments["source_index"]
            )
            kept_videos = len(
                {
                    str(row.get("video_url", "")).strip()
                    for row in eligible_kept
                    if str(row.get("video_url", "")).strip()
                    and str(row.get("video_url", "")).strip() != "nan"
                }
            )
        return {
            "total": total,
            "saved": saved,
            "remaining": total - saved,
            "kept": kept,
            "minimum_kept": MIN_GOLD_KEPT,
            "annotation_target": ANNOTATION_TARGET_KEPT,
            "gold_remaining": max(MIN_GOLD_KEPT - kept, 0),
            "annotation_remaining": max(ANNOTATION_TARGET_KEPT - kept, 0),
            "kept_videos": kept_videos,
            "minimum_videos": MIN_GOLD_VIDEOS,
            "gold_videos_remaining": max(MIN_GOLD_VIDEOS - kept_videos, 0),
            "campaign_saved": campaign_saved,
            "output_file": str(GOLD_OUTPUT_PATH),
        }

    def next_unannotated_position(self, after=-1):
        total = len(self.comments)
        if not total:
            return None
        start = (int(after) + 1) % total
        with self.lock:
            for offset in range(total):
                position = (start + offset) % total
                source_index = int(self.comments.iloc[position]["source_index"])
                if source_index not in self.annotated_ids:
                    return position
        return None

    def record_at(self, position):
        position = int(position)
        if position < 0 or position >= len(self.comments):
            raise IndexError("Comment position is outside the dataset")

        current = self.comments.iloc[position]
        source_index = int(current["source_index"])
        comment = str(current["clean_comment"])
        video_url = str(current.get("video_url", ""))
        if video_url == "nan":
            video_url = ""
        auto_sentence, token_rows = self.analyze(comment)
        with self.lock:
            latest = self.latest_by_id.get(source_index)
            latest_status = latest.get("status", "") if latest else ""
        return {
            "position": position,
            "source_index": source_index,
            "comment": comment,
            "video_url": video_url,
            "auto_sentence": auto_sentence,
            "token_rows": token_rows,
            "already_annotated": latest is not None,
            "latest_status": latest_status,
            "suggested_change_count": sum(row["changed"] for row in token_rows),
            "review_token_count": sum(row["review_required"] for row in token_rows),
        }

    def save(self, payload):
        source_index = int(payload["source_index"])
        status = payload.get("status", "")
        if status not in VALID_STATUSES:
            raise ValueError("Invalid annotation status")

        matching = self.comments.loc[self.comments["source_index"] == source_index]
        if matching.empty:
            raise ValueError("Unknown source index")
        current = matching.iloc[0]
        comment = str(current["clean_comment"])
        video_url = str(current.get("video_url", ""))
        if video_url == "nan":
            video_url = ""

        token_rows = payload.get("token_rows")
        if not isinstance(token_rows, list):
            raise ValueError("token_rows must be a list")
        server_auto_sentence, analysis_rows = self.analyze(comment)
        expected_tokens = [row["token"] for row in analysis_rows]
        received_tokens = [str(row.get("token", "")) for row in token_rows]
        if received_tokens != expected_tokens:
            raise ValueError("Submitted tokens do not match this comment")

        normalized_rows = []
        for row, analysis in zip(token_rows, analysis_rows):
            normalized_rows.append(
                {
                    "token": str(row.get("token", "")),
                    "category": analysis["category"],
                    "selected": str(row.get("selected", "")),
                    "manual": str(row.get("manual", "")).strip(),
                }
            )

        auto_sentence = server_auto_sentence
        manual_sentence = str(payload.get("manual_sentence", "")).strip()
        if status == "keep" and not manual_sentence:
            raise ValueError("A kept comment requires a reviewed formal sentence")

        aligned_corrections = align_sentence_to_source(expected_tokens, manual_sentence)
        for row, aligned in zip(normalized_rows, aligned_corrections):
            row["manual"] = aligned

        token_payload = [
            {
                "token": row["token"],
                "category": row["category"],
                "selected_candidate": row["selected"],
                "final_correction": row["manual"],
                **({"sentence_target": manual_sentence} if index == 0 else {}),
            }
            for index, row in enumerate(normalized_rows)
        ]
        suggestion_action = classify_suggestion_action(
            comment, auto_sentence, manual_sentence
        )
        changed_token_count = sum(
            source.casefold() != target.casefold()
            for source, target in zip(expected_tokens, aligned_corrections)
        )
        learned_updates = {}

        with self.lock:
            append_dict(
                GOLD_OUTPUT_PATH,
                {
                    "source_index": source_index,
                    "comment": comment,
                    "auto_suggestion": auto_sentence,
                    "manual_formal_wolof": manual_sentence,
                    "status": status,
                    "video_url": video_url,
                    "token_corrections_json": json.dumps(
                        token_payload, ensure_ascii=False
                    ),
                },
                GOLD_COLUMNS,
            )

            append_dict(
                SENTENCE_ANNOTATION_EVENTS_PATH,
                {
                    "saved_utc": datetime.now(timezone.utc).isoformat(),
                    "source_index": source_index,
                    "status": status,
                    "suggestion_action": suggestion_action,
                    "elapsed_ms": max(0, int(payload.get("elapsed_ms", 0) or 0)),
                    "source_word_count": len(expected_tokens),
                    "target_word_count": len(surface_word_tokens(manual_sentence)),
                    "changed_token_count": changed_token_count,
                    "output_file": str(GOLD_OUTPUT_PATH),
                },
                EVENT_COLUMNS,
            )

            if status == "keep":
                for row in normalized_rows:
                    correction = row["manual"]
                    token = row["token"]
                    if not correction or correction == token:
                        continue
                    append_dict(
                        TOKEN_OUTPUT_PATH,
                        {
                            "source_index": source_index,
                            "informal_token": token,
                            "formal_token": correction,
                            "category": row["category"],
                            "comment": comment,
                        },
                        TOKEN_COLUMNS,
                    )
                    equivalents = self.learned_corrections.setdefault(
                        token.casefold(), []
                    )
                    if correction in equivalents:
                        equivalents.remove(correction)
                    equivalents.insert(0, correction)
                    learned_updates[token.casefold()] = correction

            self.annotated_ids.add(source_index)
            self.campaign_annotated_ids.add(source_index)
            self.latest_by_id[source_index] = {
                "status": status,
                "manual_formal_wolof": manual_sentence,
                "video_url": video_url,
            }

        return learned_updates
