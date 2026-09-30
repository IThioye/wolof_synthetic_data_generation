"""Build leakage-audited unpaired Wolof pretraining splits.

The corpus combines formal news text with informal YouTube comments.  It never
uses a Gold correction as training text.  All videos represented in the Gold
normalization benchmark are excluded from the YouTube corpus, and all exact
Gold source/reference strings are excluded from both domains.

The existing comment labels are used only to fit a conservative Wolof-content
selector.  They are not normalization labels and the Gold sentences themselves
are removed before the selector is fitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import unicodedata
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support

from src.config import (
    CHECKPOINT_DATA_PATH,
    CLEAN_COMMENTS_PATH,
    GOLD_SPLITS_DIR,
    NEWS_DATA_PATH,
    PRETRAINING_CORPUS_PATH,
    PRETRAINING_DATA_DIR,
    PRETRAINING_DEV_PATH,
    PRETRAINING_MANIFEST_PATH,
    PRETRAINING_TEST_PATH,
    PRETRAINING_TRAIN_PATH,
    PROJECT_ROOT,
)
from src.pipelines.get_filtered_dataset import _coerce_boolean_labels


ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060\ufe0e\ufe0f\ufeff]")
WHITESPACE_RE = re.compile(r"\s+")
URL_RE = re.compile(r"(?:https?://|www\.)\S+", flags=re.IGNORECASE)
TOKEN_RE = re.compile(r"\S+")


@dataclass(frozen=True)
class CorpusBuildConfig:
    seed: int = 2026
    selector_threshold: float = 0.80
    train_fraction: float = 0.90
    dev_fraction: float = 0.05
    test_fraction: float = 0.05
    min_characters: int = 8
    min_alphabetic_characters: int = 5
    min_tokens: int = 2
    # TNT V3 has 256 position embeddings. Keep room for tokenizer special tokens.
    max_segment_characters: int = 240
    selector_max_features: int = 60_000
    selector_min_document_frequency: int = 2
    selector_c: float = 2.0

    def validate(self) -> None:
        fractions = self.train_fraction + self.dev_fraction + self.test_fraction
        if not math.isclose(fractions, 1.0, abs_tol=1e-9):
            raise ValueError("Train/dev/test fractions must sum to 1.0")
        if not 0.0 < self.selector_threshold < 1.0:
            raise ValueError("selector_threshold must be between zero and one")
        if self.max_segment_characters < self.min_characters:
            raise ValueError("max_segment_characters must exceed min_characters")


def normalize_text(value: object) -> str:
    """Normalize encoding/spacing while preserving case, spelling and accents."""
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFC", str(value))
    text = ZERO_WIDTH_RE.sub("", text)
    text = "".join(
        " " if character.isspace() else ""
        if unicodedata.category(character) == "Cc"
        else character
        for character in text
    )
    return WHITESPACE_RE.sub(" ", text).strip()


def canonical_text(value: object) -> str:
    """Return the exact-overlap key used for deduplication and leakage checks."""
    return normalize_text(value).casefold()


def compact_text(value: object) -> str:
    """Return a punctuation-insensitive overlap key without changing outputs."""
    return "".join(
        character
        for character in canonical_text(value)
        if character.isalnum()
    )


def usable_text(text: str, config: CorpusBuildConfig) -> bool:
    without_urls = URL_RE.sub("", text).strip()
    alphabetic = sum(character.isalpha() for character in without_urls)
    return (
        len(without_urls) >= config.min_characters
        and alphabetic >= config.min_alphabetic_characters
        and len(TOKEN_RE.findall(without_urls)) >= config.min_tokens
    )


def segment_text(text: str, max_characters: int) -> list[str]:
    """Split long text at whitespace, never silently truncating it."""
    text = normalize_text(text)
    if len(text) <= max_characters:
        return [text] if text else []
    segments: list[str] = []
    current: list[str] = []
    current_length = 0
    for token in TOKEN_RE.findall(text):
        if len(token) > max_characters:
            if current:
                segments.append(" ".join(current))
                current, current_length = [], 0
            segments.extend(
                token[start : start + max_characters]
                for start in range(0, len(token), max_characters)
            )
            continue
        projected = current_length + len(token) + (1 if current else 0)
        if current and projected > max_characters:
            segments.append(" ".join(current))
            current, current_length = [token], len(token)
        else:
            current.append(token)
            current_length = projected
    if current:
        segments.append(" ".join(current))
    return segments


def _selector_vectorizer(config: CorpusBuildConfig) -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=config.selector_min_document_frequency,
        max_features=config.selector_max_features,
        sublinear_tf=True,
        dtype=np.float32,
    )


def _selector_classifier(config: CorpusBuildConfig) -> LogisticRegression:
    return LogisticRegression(
        C=config.selector_c,
        class_weight="balanced",
        max_iter=500,
        random_state=config.seed,
    )


def train_content_selector(
    labeled: pd.DataFrame,
    config: CorpusBuildConfig,
) -> tuple[TfidfVectorizer, LogisticRegression, dict[str, object]]:
    """Fit and group-audit the existing Wolof-content labels."""
    required = {"text", "label", "document_id"}
    missing = required - set(labeled.columns)
    if missing:
        raise ValueError(f"Selector labels are missing columns: {sorted(missing)}")
    if labeled["label"].nunique() != 2:
        raise ValueError("Selector training data must contain both classes")

    probabilities = np.full(len(labeled), np.nan, dtype=float)
    groups = labeled["document_id"].astype(str).to_numpy()
    labels = labeled["label"].astype(bool).to_numpy()
    texts = labeled["text"].astype(str).to_numpy()

    for held_out_group in sorted(set(groups)):
        training_mask = groups != held_out_group
        validation_mask = ~training_mask
        if labels[training_mask].min() == labels[training_mask].max():
            continue
        vectorizer = _selector_vectorizer(config)
        training_vectors = vectorizer.fit_transform(texts[training_mask])
        classifier = _selector_classifier(config)
        classifier.fit(training_vectors, labels[training_mask])
        probabilities[validation_mask] = classifier.predict_proba(
            vectorizer.transform(texts[validation_mask])
        )[:, 1]

    valid = ~np.isnan(probabilities)
    predictions = probabilities[valid] >= config.selector_threshold
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels[valid], predictions, average="binary", zero_division=0
    )
    audit = {
        "protocol": "leave-one-video-out",
        "rows": int(len(labeled)),
        "videos": int(labeled["document_id"].nunique()),
        "positive_rows": int(labels.sum()),
        "threshold": config.selector_threshold,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accepted_rows": int(predictions.sum()),
        "evaluated_rows": int(valid.sum()),
    }

    vectorizer = _selector_vectorizer(config)
    vectors = vectorizer.fit_transform(texts)
    classifier = _selector_classifier(config)
    classifier.fit(vectors, labels)
    return vectorizer, classifier, audit


def prepare_selector_labels(
    clean_comments: pd.DataFrame,
    checkpoint_data: pd.DataFrame,
    forbidden_keys: set[str],
    allowed_videos: set[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    checkpoints = checkpoint_data.copy()
    checkpoints["index"] = pd.to_numeric(checkpoints["index"], errors="raise").astype(int)
    checkpoints["label"] = _coerce_boolean_labels(
        checkpoints["is_informally_code_switched"]
    )
    checkpoints = checkpoints.drop_duplicates("index", keep="last")
    comments = clean_comments.reset_index(names="source_index")
    joined = checkpoints.merge(
        comments[["source_index", "clean_comment", "video_url"]],
        left_on="index",
        right_on="source_index",
        how="inner",
        validate="one_to_one",
    )
    joined["text"] = joined["clean_comment"].map(normalize_text)
    joined["canonical"] = joined["text"].map(canonical_text)
    if allowed_videos is None:
        allowed_video_mask = pd.Series(True, index=joined.index)
    else:
        allowed_video_mask = joined["video_url"].astype(str).isin(allowed_videos)
    gold_overlap = joined["canonical"].isin(forbidden_keys)
    output = joined.loc[
        allowed_video_mask & ~gold_overlap, ["text", "label", "video_url"]
    ].rename(
        columns={"video_url": "document_id"}
    )
    return output.reset_index(drop=True), {
        "checkpoint_rows": int(len(checkpoints)),
        "joined_rows": int(len(joined)),
        "non_training_video_rows_excluded": int((~allowed_video_mask).sum()),
        "gold_text_rows_found_in_all_labeled_videos": int(gold_overlap.sum()),
        "gold_text_rows_excluded_before_selector_fit": int(
            (allowed_video_mask & gold_overlap).sum()
        ),
        "selector_training_rows": int(len(output)),
    }


def prepare_youtube(
    clean_comments: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    classifier: LogisticRegression,
    gold_videos: set[str],
    forbidden_keys: set[str],
    forbidden_compact_keys: set[str],
    config: CorpusBuildConfig,
) -> tuple[pd.DataFrame, dict[str, int]]:
    data = clean_comments.reset_index(names="source_index").copy()
    counts = {"input_rows": int(len(data))}
    gold_video_mask = data["video_url"].astype(str).isin(gold_videos)
    counts["gold_video_rows_excluded"] = int(gold_video_mask.sum())
    data = data.loc[~gold_video_mask].copy()
    counts["rows_after_gold_video_exclusion"] = int(len(data))

    data["text"] = data["clean_comment"].map(normalize_text)
    usable = data["text"].map(lambda value: usable_text(value, config))
    counts["unusable_rows_excluded"] = int((~usable).sum())
    data = data.loc[usable].copy()
    data["canonical"] = data["text"].map(canonical_text)
    data["compact"] = data["text"].map(compact_text)
    gold_overlap = data["canonical"].isin(forbidden_keys) | data["compact"].isin(
        forbidden_compact_keys
    )
    counts["gold_text_overlap_rows_excluded"] = int(gold_overlap.sum())
    data = data.loc[~gold_overlap].copy()

    probabilities = classifier.predict_proba(vectorizer.transform(data["text"]))[:, 1]
    data["selector_score"] = probabilities
    accepted = data["selector_score"] >= config.selector_threshold
    counts["selector_rejected_rows"] = int((~accepted).sum())
    data = data.loc[accepted].copy()
    counts["accepted_comments_before_segmentation"] = int(len(data))
    data["segments"] = data["text"].map(
        lambda value: segment_text(value, config.max_segment_characters)
    )
    data = data.explode("segments", ignore_index=True)
    data["text"] = data["segments"].astype(str)
    data["segment_index"] = data.groupby("source_index", sort=False).cumcount()
    counts["segments_before_deduplication"] = int(len(data))
    data["canonical"] = data["text"].map(canonical_text)
    data["compact"] = data["text"].map(compact_text)
    segment_gold_overlap = data["canonical"].isin(forbidden_keys) | data[
        "compact"
    ].isin(forbidden_compact_keys)
    counts["gold_text_overlap_segments_excluded"] = int(segment_gold_overlap.sum())
    data = data.loc[~segment_gold_overlap].copy()
    duplicate = data.duplicated("canonical", keep="first")
    counts["within_domain_duplicates_excluded"] = int(duplicate.sum())
    data = data.loc[~duplicate].copy()
    data["domain"] = "informal_youtube"
    data["document_id"] = data["video_url"].astype(str)
    data["source_id"] = (
        "youtube:"
        + data["source_index"].astype(str)
        + ":"
        + data["segment_index"].astype(str)
    )
    counts["output_rows"] = int(len(data))
    counts["output_documents"] = int(data["document_id"].nunique())
    return data[
        [
            "source_id",
            "domain",
            "document_id",
            "segment_index",
            "text",
            "selector_score",
            "french_ratio",
            "canonical",
            "compact",
        ]
    ].reset_index(drop=True), counts


def prepare_news(
    news: pd.DataFrame,
    forbidden_keys: set[str],
    forbidden_compact_keys: set[str],
    config: CorpusBuildConfig,
) -> tuple[pd.DataFrame, dict[str, int]]:
    required = {"url", "text"}
    missing = required - set(news.columns)
    if missing:
        raise ValueError(f"News data is missing columns: {sorted(missing)}")
    rows: list[dict[str, object]] = []
    for source_row, row in news.reset_index(drop=True).iterrows():
        normalized = normalize_text(row["text"])
        for segment_index, segment in enumerate(
            segment_text(normalized, config.max_segment_characters)
        ):
            rows.append(
                {
                    "source_id": f"news:{source_row}:{segment_index}",
                    "domain": "formal_news",
                    "document_id": str(row["url"]),
                    "segment_index": segment_index,
                    "text": segment,
                    "selector_score": np.nan,
                    "french_ratio": np.nan,
                }
            )
    data = pd.DataFrame(rows)
    counts = {
        "input_rows": int(len(news)),
        "input_documents": int(news["url"].nunique()),
        "segments_before_filtering": int(len(data)),
    }
    usable = data["text"].map(lambda value: usable_text(value, config))
    counts["unusable_segments_excluded"] = int((~usable).sum())
    data = data.loc[usable].copy()
    data["canonical"] = data["text"].map(canonical_text)
    data["compact"] = data["text"].map(compact_text)
    gold_overlap = data["canonical"].isin(forbidden_keys) | data["compact"].isin(
        forbidden_compact_keys
    )
    counts["gold_text_overlap_segments_excluded"] = int(gold_overlap.sum())
    data = data.loc[~gold_overlap].copy()
    duplicate = data.duplicated("canonical", keep="first")
    counts["within_domain_duplicates_excluded"] = int(duplicate.sum())
    data = data.loc[~duplicate].copy()
    counts["output_rows"] = int(len(data))
    counts["output_documents"] = int(data["document_id"].nunique())
    return data.reset_index(drop=True), counts


def assign_group_splits(
    data: pd.DataFrame,
    config: CorpusBuildConfig,
) -> pd.Series:
    """Assign whole documents to approximately proportional row splits."""
    if data.empty:
        raise ValueError("Cannot split an empty domain")
    group_sizes = data.groupby("document_id", sort=True).size().rename("rows").reset_index()
    if len(group_sizes) < 3:
        raise ValueError("Each domain needs at least three documents for grouped splitting")
    rng = np.random.default_rng(config.seed)
    group_sizes["tie_breaker"] = rng.random(len(group_sizes))
    group_sizes = group_sizes.sort_values(
        ["rows", "tie_breaker", "document_id"], ascending=[False, True, True]
    )
    fractions = {
        "train": config.train_fraction,
        "dev": config.dev_fraction,
        "test": config.test_fraction,
    }
    targets = {name: len(data) * fraction for name, fraction in fractions.items()}
    current = {name: 0 for name in fractions}
    assignments: dict[str, str] = {}
    for row in group_sizes.itertuples(index=False):
        candidates = sorted(
            fractions,
            key=lambda name: (
                (current[name] + row.rows) / targets[name],
                name,
            ),
        )
        chosen = candidates[0]
        assignments[str(row.document_id)] = chosen
        current[chosen] += int(row.rows)
    splits = data["document_id"].astype(str).map(assignments)
    if splits.isna().any() or set(splits) != {"train", "dev", "test"}:
        raise AssertionError("Grouped split assignment failed")
    return splits


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_record(path: Path) -> dict[str, object]:
    resolved = path.resolve()
    try:
        display_path = resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        display_path = resolved.as_posix()
    return {
        "path": display_path,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _domain_split_summary(data: pd.DataFrame) -> list[dict[str, object]]:
    summary = (
        data.assign(character_count=data["text"].str.len())
        .groupby(["split", "domain"], sort=True)
        .agg(
            rows=("source_id", "size"),
            documents=("document_id", "nunique"),
            characters=("character_count", "sum"),
        )
        .reset_index()
    )
    return summary.to_dict("records")


def load_gold_splits(directory: Path) -> pd.DataFrame:
    frames = []
    for split in ("train", "dev", "test"):
        path = directory / f"gold_{split}.csv"
        frame = pd.read_csv(path)
        frame["gold_split"] = split
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def build_corpus(
    clean_comments: pd.DataFrame,
    checkpoint_data: pd.DataFrame,
    news: pd.DataFrame,
    gold: pd.DataFrame,
    config: CorpusBuildConfig,
) -> tuple[pd.DataFrame, dict[str, object]]:
    config.validate()
    gold_text_columns = [column for column in ("comment", "manual_formal_wolof") if column in gold]
    gold_texts = [normalize_text(value) for column in gold_text_columns for value in gold[column]]
    forbidden_keys = {canonical_text(value) for value in gold_texts if value}
    forbidden_compact_keys = {compact_text(value) for value in gold_texts if value}
    forbidden_compact_keys.discard("")
    gold_videos = set(gold["video_url"].dropna().astype(str))
    selector_videos = set(
        gold.loc[gold["gold_split"] == "train", "video_url"].dropna().astype(str)
    )

    labels, label_counts = prepare_selector_labels(
        clean_comments, checkpoint_data, forbidden_keys, selector_videos
    )
    vectorizer, classifier, selector_audit = train_content_selector(labels, config)
    youtube, youtube_counts = prepare_youtube(
        clean_comments,
        vectorizer,
        classifier,
        gold_videos,
        forbidden_keys,
        forbidden_compact_keys,
        config,
    )
    formal_news, news_counts = prepare_news(
        news, forbidden_keys, forbidden_compact_keys, config
    )
    corpus = pd.concat([formal_news, youtube], ignore_index=True)
    cross_domain_duplicate = corpus.duplicated("canonical", keep="first")
    cross_domain_duplicates = int(cross_domain_duplicate.sum())
    corpus = corpus.loc[~cross_domain_duplicate].copy()
    corpus["split"] = ""
    for domain in sorted(corpus["domain"].unique()):
        mask = corpus["domain"] == domain
        corpus.loc[mask, "split"] = assign_group_splits(corpus.loc[mask], config).values
    corpus = corpus.sort_values(
        ["split", "domain", "document_id", "source_id"], kind="stable"
    ).reset_index(drop=True)

    document_overlap = {}
    text_overlap = {}
    for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
        left_rows = corpus[corpus["split"] == left]
        right_rows = corpus[corpus["split"] == right]
        key = f"{left}_{right}"
        document_overlap[key] = int(
            len(set(left_rows["document_id"]) & set(right_rows["document_id"]))
        )
        text_overlap[key] = int(
            len(set(left_rows["canonical"]) & set(right_rows["canonical"]))
        )
    remaining_gold_text_overlap = int(
        (
            corpus["canonical"].isin(forbidden_keys)
            | corpus["compact"].isin(forbidden_compact_keys)
        ).sum()
    )
    remaining_gold_video_rows = int(corpus["document_id"].isin(gold_videos).sum())
    if any(document_overlap.values()) or any(text_overlap.values()):
        raise AssertionError("Leakage detected between pretraining splits")
    if remaining_gold_text_overlap or remaining_gold_video_rows:
        raise AssertionError("Gold normalization data leaked into the corpus")

    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "unpaired Wolof encoder pretraining corpus",
        "configuration": asdict(config),
        "gold_exclusion": {
            "rows": int(len(gold)),
            "videos": sorted(gold_videos),
            "selector_training_videos": sorted(selector_videos),
            "source_and_reference_text_keys": int(len(forbidden_keys)),
            "compact_text_keys": int(len(forbidden_compact_keys)),
        },
        "selector_label_preparation": label_counts,
        "selector_validation": selector_audit,
        "youtube_processing": youtube_counts,
        "news_processing": news_counts,
        "cross_domain_duplicates_excluded": cross_domain_duplicates,
        "final_rows": int(len(corpus)),
        "final_documents": int(corpus["document_id"].nunique()),
        "final_characters": int(corpus["text"].str.len().sum()),
        "maximum_text_characters": int(corpus["text"].str.len().max()),
        "split_domain_summary": _domain_split_summary(corpus),
        "leakage_checks": {
            "document_overlap_between_splits": document_overlap,
            "text_overlap_between_splits": text_overlap,
            "rows_from_gold_videos": remaining_gold_video_rows,
            "rows_matching_gold_source_or_reference": remaining_gold_text_overlap,
            "passed": True,
        },
        "source_notes": {
            "youtube": (
                "Cleaned public comments outside all five Gold videos. Selection uses "
                "the pre-existing informal-Wolof/code-switch classifier labels only; "
                "no normalization target is used."
            ),
            "news": (
                "Local defuwaxu.com news snapshot. Provenance and redistribution "
                "licence are not yet documented; keep the text local until resolved."
            ),
        },
    }
    export_columns = [
        "source_id",
        "domain",
        "document_id",
        "segment_index",
        "text",
        "selector_score",
        "french_ratio",
        "split",
    ]
    return corpus[export_columns], manifest


def _write_review_sample(corpus: pd.DataFrame, path: Path, seed: int) -> None:
    samples = []
    for (split, domain), frame in corpus.groupby(["split", "domain"], sort=True):
        sample_size = min(25, len(frame))
        sample = frame.sample(sample_size, random_state=seed).copy()
        sample["review_is_wolof_or_relevant_codeswitch"] = ""
        sample["review_notes"] = ""
        samples.append(sample)
    pd.concat(samples, ignore_index=True).to_csv(path, index=False, encoding="utf-8")


def write_corpus(
    corpus: pd.DataFrame,
    manifest: dict[str, object],
    output_dir: Path = PRETRAINING_DATA_DIR,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "corpus": output_dir / PRETRAINING_CORPUS_PATH.name,
        "train": output_dir / PRETRAINING_TRAIN_PATH.name,
        "dev": output_dir / PRETRAINING_DEV_PATH.name,
        "test": output_dir / PRETRAINING_TEST_PATH.name,
    }
    corpus.to_parquet(output_paths["corpus"], index=False)
    for split in ("train", "dev", "test"):
        corpus.loc[corpus["split"] == split].reset_index(drop=True).to_parquet(
            output_paths[split], index=False
        )
    review_path = output_dir / "wolof_pretraining_review_sample.csv"
    _write_review_sample(corpus, review_path, int(manifest["configuration"]["seed"]))
    manifest["outputs"] = {
        name: _path_record(path) for name, path in output_paths.items()
    }
    manifest["outputs"]["review_sample"] = _path_record(review_path)
    manifest_path = output_dir / PRETRAINING_MANIFEST_PATH.name
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def build_from_config(config: CorpusBuildConfig) -> dict[str, object]:
    clean_comments = pd.read_csv(CLEAN_COMMENTS_PATH)
    checkpoint_data = pd.read_csv(CHECKPOINT_DATA_PATH)
    news = pd.read_parquet(NEWS_DATA_PATH)
    gold = load_gold_splits(GOLD_SPLITS_DIR)
    corpus, manifest = build_corpus(
        clean_comments, checkpoint_data, news, gold, config
    )
    manifest["inputs"] = {
        "clean_comments": _path_record(CLEAN_COMMENTS_PATH),
        "checkpoint_labels": _path_record(CHECKPOINT_DATA_PATH),
        "news": _path_record(NEWS_DATA_PATH),
        "gold_train": _path_record(GOLD_SPLITS_DIR / "gold_train.csv"),
        "gold_dev": _path_record(GOLD_SPLITS_DIR / "gold_dev.csv"),
        "gold_test": _path_record(GOLD_SPLITS_DIR / "gold_test.csv"),
    }
    return write_corpus(corpus, manifest)


def parse_args(arguments: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--selector-threshold", type=float, default=0.80)
    parser.add_argument("--max-segment-characters", type=int, default=240)
    return parser.parse_args(arguments)


def main(arguments: Iterable[str] | None = None) -> None:
    args = parse_args(arguments)
    config = CorpusBuildConfig(
        seed=args.seed,
        selector_threshold=args.selector_threshold,
        max_segment_characters=args.max_segment_characters,
    )
    manifest = build_from_config(config)
    print(json.dumps({
        "final_rows": manifest["final_rows"],
        "final_documents": manifest["final_documents"],
        "split_domain_summary": manifest["split_domain_summary"],
        "selector_validation": manifest["selector_validation"],
        "leakage_checks": manifest["leakage_checks"],
        "manifest": str(PRETRAINING_MANIFEST_PATH.resolve()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
