import csv
import json
import pickle
import threading

from functools import lru_cache
import pandas as pd

from pathlib import Path
import sys
external_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(external_dir))


from config import (
    CHECKPOINT_DATA_PATH as CHECKPOINT_PATH,
    CLEAN_COMMENTS_PATH,
    FILTERED_COMMENTS_PATH as DEFAULT_COMMENTS_PATH,
    FRENCH_WORDLIST_PATH,
    GOLD_ANNOTATIONS_PATH as GOLD_OUTPUT_PATH,
    LEXICON_PATH,
    MIN_GOLD_KEPT,
    MIN_GOLD_VIDEOS,
    TOKEN_CORRECTIONS_PATH as TOKEN_OUTPUT_PATH,
    TRAIN_PARQUET_PATH,
    WOLOF_VOCAB_PATH,
)

from normalization.reverse_code_switched import (
    VocabIndex,
    best_reverse_translation,
    build_french_vocab_from_corpus,
    build_reverse_lexicon,
    classify_token,
    generate_candidates,
    mask_protected,
    weighted_distance,
    word_tokens,
)

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


def load_existing_annotations():
    if not GOLD_OUTPUT_PATH.exists():
        return pd.DataFrame()
    try:
        annotations = pd.read_csv(GOLD_OUTPUT_PATH)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    if "source_index" not in annotations.columns:
        return pd.DataFrame()
    return annotations


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

    if TOKEN_OUTPUT_PATH.exists():
        try:
            corrections = pd.read_csv(TOKEN_OUTPUT_PATH)
        except (pd.errors.EmptyDataError, UnicodeDecodeError):
            corrections = pd.DataFrame()
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


def load_resources():
    with open(LEXICON_PATH, "rb") as handle:
        lexicon = pickle.load(handle)
    reverse_lexicon = build_reverse_lexicon(lexicon)

    with open(WOLOF_VOCAB_PATH, encoding="utf-8") as handle:
        vocab_index = VocabIndex(handle.read().splitlines())

    train_df = pd.read_parquet(TRAIN_PARQUET_PATH)
    french_vocab_corpus = build_french_vocab_from_corpus(train_df["french"])

    french_vocab_general = set()
    if FRENCH_WORDLIST_PATH.exists():
        lexique = pd.read_csv(FRENCH_WORDLIST_PATH, sep="\t")
        french_vocab_general = set(lexique["1_Mot"].str.lower().dropna())

    return vocab_index, french_vocab_corpus, french_vocab_general, reverse_lexicon


def nearest_vocab_candidates(token, vocab_index, limit=6):
    """Rank nearby vocabulary in one pass instead of rescanning per rewrite."""
    scored = {}
    generated = generate_candidates(token, max_candidates=40)
    generated.add(token)

    for candidate in generated:
        if candidate in vocab_index:
            scored[candidate] = min(
                scored.get(candidate, 999.0), weighted_distance(token, candidate)
            )

    # The old implementation called find_best_match for every generated rewrite;
    # every call scanned the same vocabulary buckets. Scan those buckets once.
    pool = vocab_index.candidates_near_length(len(token), window=1)
    threshold = max(1.0, len(token) * 0.34)
    for word in pool:
        distance = weighted_distance(token, word)
        if distance <= threshold:
            score = distance + 0.05 * abs(len(word) - len(token))
            scored[word] = min(scored.get(word, 999.0), score)

    ranked = sorted(
        scored.items(),
        key=lambda item: (item[1], abs(len(item[0]) - len(token)), item[0]),
    )
    return [word for word, _ in ranked[:limit]]


def generated_token_suggestions(
    token,
    vocab_index,
    french_vocab_corpus,
    french_vocab_general,
    reverse_lexicon,
):
    category = classify_token(
        token,
        vocab_index,
        french_vocab_corpus,
        french_vocab_general,
        reverse_lexicon,
    )
    suggestions = [token]

    if category == "french_substituted":
        translations = reverse_lexicon.get(token, {})
        suggestions.extend(
            word
            for word, _ in sorted(
                translations.items(), key=lambda item: item[1], reverse=True
            )
        )
        best = best_reverse_translation(token, reverse_lexicon)
        if best:
            suggestions.insert(1, best)
    elif category == "informal_wolof":
        suggestions.extend(nearest_vocab_candidates(token, vocab_index))
    # Formal and ambiguous tokens already have an exact vocabulary match. Their
    # original form is sufficient unless a human correction has been learned.

    deduped = []
    for suggestion in suggestions:
        if suggestion and suggestion not in deduped:
            deduped.append(suggestion)
    return category, tuple(deduped[:8])


def append_dict(path, row, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with open(path, "a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


class AnnotationService:
    def __init__(self):
        self.comments = load_comments()
        self.resources = load_resources()
        self.lock = threading.RLock()

        existing = load_existing_annotations()
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
        return generated_token_suggestions(token, *self.resources)

    def analyze(self, text):
        masked, _saved = mask_protected(text, entities=[])
        rows = []
        auto_tokens = []

        for token in word_tokens(masked):
            with self.lock:
                remembered = list(self.learned_corrections.get(token.casefold(), []))

            if remembered:
                # A human correction is more valuable than another fuzzy search.
                # Avoid the expensive distance calculation for already learned words.
                category = classify_token(token, *self.resources)
                suggestions = list(remembered)
                if token not in suggestions:
                    suggestions.append(token)
                auto = remembered[0]
            else:
                category, generated = self.base_token_suggestions(token)
                suggestions = list(generated)
                auto = (
                    suggestions[1]
                    if len(suggestions) > 1
                    and category in {"informal_wolof", "french_substituted"}
                    else suggestions[0]
                )

            rows.append(
                {
                    "token": token,
                    "category": category,
                    "suggestions": suggestions[:8],
                    "auto": auto,
                    "learned": bool(remembered),
                }
            )
            auto_tokens.append(auto)
        return " ".join(auto_tokens), rows

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
            "gold_remaining": max(MIN_GOLD_KEPT - kept, 0),
            "kept_videos": kept_videos,
            "minimum_videos": MIN_GOLD_VIDEOS,
            "gold_videos_remaining": max(MIN_GOLD_VIDEOS - kept_videos, 0),
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
        expected_tokens = [row["token"] for row in self.analyze(comment)[1]]
        received_tokens = [str(row.get("token", "")) for row in token_rows]
        if received_tokens != expected_tokens:
            raise ValueError("Submitted tokens do not match this comment")

        normalized_rows = []
        for row in token_rows:
            normalized_rows.append(
                {
                    "token": str(row.get("token", "")),
                    "category": str(row.get("category", "")),
                    "selected": str(row.get("selected", "")),
                    "manual": str(row.get("manual", "")).strip(),
                }
            )

        token_payload = [
            {
                "token": row["token"],
                "category": row["category"],
                "selected_candidate": row["selected"],
                "final_correction": row["manual"],
            }
            for row in normalized_rows
        ]
        auto_sentence = str(payload.get("auto_sentence", ""))
        manual_sentence = str(payload.get("manual_sentence", "")).strip()
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
            self.latest_by_id[source_index] = {
                "status": status,
                "manual_formal_wolof": manual_sentence,
                "video_url": video_url,
            }

        return learned_updates
