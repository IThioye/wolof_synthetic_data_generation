import pandas as pd

from src.pipelines.build_wolof_pretraining_corpus import (
    CorpusBuildConfig,
    assign_group_splits,
    canonical_text,
    compact_text,
    normalize_text,
    prepare_news,
    prepare_selector_labels,
    segment_text,
)


def test_normalization_preserves_spelling_and_removes_layout_noise():
    assert normalize_text("  Ñu\u200b   dem\nfa  ") == "Ñu dem fa"
    assert canonical_text(" ÑU  dem ") == "ñu dem"
    assert compact_text("Ñu-dem!") == "ñudem"


def test_segment_text_never_truncates_content():
    text = "ab cd ef gh ij"
    segments = segment_text(text, max_characters=5)
    assert segments == ["ab cd", "ef gh", "ij"]
    assert " ".join(segments) == text


def test_prepare_news_filters_gold_overlap_and_duplicates():
    news = pd.DataFrame(
        {
            "url": ["article-1", "article-1", "article-2", "article-3"],
            "text": [
                "Lii mooy xibaar bi tey",
                "Lii mooy xibaar bi tey",
                "Gold reference bi fii",
                "Ñu dem ca dëkk ba suba",
            ],
        }
    )
    config = CorpusBuildConfig(max_segment_characters=50)
    output, counts = prepare_news(
        news,
        {canonical_text("Gold reference bi fii")},
        {compact_text("Gold reference bi fii")},
        config,
    )
    assert output["text"].tolist() == ["Lii mooy xibaar bi tey", "Ñu dem ca dëkk ba suba"]
    assert counts["gold_text_overlap_segments_excluded"] == 1
    assert counts["within_domain_duplicates_excluded"] == 1


def test_selector_labels_exclude_held_out_videos_and_gold_text():
    clean = pd.DataFrame(
        {
            "clean_comment": ["gold input here", "usable Wolof row", "held out row"],
            "video_url": ["train-video", "train-video", "test-video"],
        }
    )
    labels = pd.DataFrame(
        {
            "index": [0, 1, 2],
            "is_informally_code_switched": [True, True, False],
        }
    )
    output, counts = prepare_selector_labels(
        clean,
        labels,
        {canonical_text("gold input here")},
        {"train-video"},
    )
    assert output["text"].tolist() == ["usable Wolof row"]
    assert counts["non_training_video_rows_excluded"] == 1
    assert counts["gold_text_rows_excluded_before_selector_fit"] == 1


def test_group_split_is_deterministic_disjoint_and_contains_all_splits():
    data = pd.DataFrame(
        {
            "document_id": [f"document-{index // 3}" for index in range(90)],
            "text": [f"row {index}" for index in range(90)],
        }
    )
    config = CorpusBuildConfig(seed=17)
    first = assign_group_splits(data, config)
    second = assign_group_splits(data, config)
    assert first.tolist() == second.tolist()
    assert set(first) == {"train", "dev", "test"}
    assigned = data.assign(split=first).groupby("document_id")["split"].nunique()
    assert assigned.max() == 1
