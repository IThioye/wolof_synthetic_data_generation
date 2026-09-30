"""Central path configuration for the Wolof PFE project.

All paths are absolute, so scripts work regardless of the directory from which
they are launched. Set PFE_PROJECT_ROOT, PFE_DATA_DIR, or PFE_ARTIFACTS_DIR to
override the default locations without editing application code.
"""

import os
from pathlib import Path


def _configured_path(environment_name: str, default: Path) -> Path:
    value = os.environ.get(environment_name)
    return Path(value).expanduser().resolve() if value else default.resolve()


# Project directories. This module lives in <project>/src, so the repository
# root is its parent directory rather than the directory containing this file.
PROJECT_ROOT = _configured_path("PFE_PROJECT_ROOT", Path(__file__).resolve().parent.parent)
DATA_DIR = _configured_path("PFE_DATA_DIR", PROJECT_ROOT / "data")
ARTIFACTS_DIR = _configured_path(
    "PFE_ARTIFACTS_DIR", PROJECT_ROOT / "artifacts"
)
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
SCRAPING_DIR = PROJECT_ROOT / "scripts" / "scraping"
ANNOTATIONS_DIR = DATA_DIR / "annotations"
GOLD_SPLITS_DIR = ANNOTATIONS_DIR / "splits"
TRANSLATION_DATA_DIR = DATA_DIR / "translation data"
SYNTHETIC_DATA_DIR = DATA_DIR / "synthetic"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
PRETRAINING_DATA_DIR = DATA_DIR / "pretraining"

# Main comment datasets and classifier outputs
YOUTUBE_DATA_DIR = DATA_DIR / "youtube"
COMMENTS_JSON_PATH = YOUTUBE_DATA_DIR / "comments_data.json"
CLEAN_COMMENTS_PATH = YOUTUBE_DATA_DIR / "clean_data.csv"
CLASSIFIED_COMMENTS_PATH = YOUTUBE_DATA_DIR / "clean_data_classified.csv"
CHECKPOINT_DATA_PATH = DATA_DIR / "checkpoint_data.csv"
FILTERED_COMMENTS_PATH = DATA_DIR / "filtered_data.csv"
CLASSIFIER_LOG_PATH = PROJECT_ROOT / "classify_wolof.log"
NEWS_DATA_PATH = DATA_DIR / "news_data.parquet"
PRETRAINING_CORPUS_PATH = PRETRAINING_DATA_DIR / "wolof_pretraining_corpus.parquet"
PRETRAINING_TRAIN_PATH = PRETRAINING_DATA_DIR / "wolof_pretraining_train.parquet"
PRETRAINING_DEV_PATH = PRETRAINING_DATA_DIR / "wolof_pretraining_dev.parquet"
PRETRAINING_TEST_PATH = PRETRAINING_DATA_DIR / "wolof_pretraining_test.parquet"
PRETRAINING_MANIFEST_PATH = (
    PRETRAINING_DATA_DIR / "wolof_pretraining_manifest.json"
)
ENCODER_PRETRAINING_RUNS_DIR = CHECKPOINTS_DIR / "wolof_encoder_pretraining_v1"
ENCODER_PRETRAINING_RESULTS_DIR = (
    PROJECT_ROOT / "results" / "wolof_encoder_pretraining_v1"
)

# Annotation outputs.  The canonical files remain the defaults, while the
# environment overrides make it possible to run a new annotation campaign
# without modifying the frozen benchmark history.
BASE_GOLD_ANNOTATIONS_PATH = ANNOTATIONS_DIR / "gold_annotations.csv"
BASE_TOKEN_CORRECTIONS_PATH = ANNOTATIONS_DIR / "token_corrections.csv"
GOLD_ANNOTATIONS_PATH = _configured_path(
    "PFE_GOLD_ANNOTATIONS_PATH", BASE_GOLD_ANNOTATIONS_PATH
)
TOKEN_CORRECTIONS_PATH = _configured_path(
    "PFE_TOKEN_CORRECTIONS_PATH", BASE_TOKEN_CORRECTIONS_PATH
)
SENTENCE_ANNOTATION_EVENTS_PATH = _configured_path(
    "PFE_SENTENCE_ANNOTATION_EVENTS_PATH",
    ANNOTATIONS_DIR / "sentence_annotation_events.csv",
)

# Gold-train linguistic annotation for the isolated M6 generation engine.
# The seed lookup is optional and read-only; the Flask application works
# without it and stores reviewed reusable entries in the curated lookup.
LINGUISTIC_ANNOTATIONS_PATH = ANNOTATIONS_DIR / "token_linguistic_annotations.csv"
FORMAL_TOKEN_LOOKUP_SEED_PATH = ANNOTATIONS_DIR / "formal_token_lookup_seed.csv"
FORMAL_TOKEN_LOOKUP_PATH = ANNOTATIONS_DIR / "formal_token_lookup.csv"
WOLOF_LOOKUP_SOURCE_PATH = DATA_DIR / "lookup_table_wolof.csv"
SENEGALESE_SURNAMES_PATH = DATA_DIR / "senegalese_surnames.txt"

# Minimum operational size for creating the locked, video-disjoint gold split.
# Both the split command and annotation interface import these values so the UI
# cannot report a different target from the dataset-readiness guard.
MIN_GOLD_KEPT = 200
MIN_GOLD_VIDEOS = 5
# The first benchmark minimum has already been reached.  This larger, editable
# target is only a progress aid for the follow-up annotation campaign; it does
# not change any train/dev/test split automatically.
ANNOTATION_TARGET_KEPT = int(os.environ.get("PFE_ANNOTATION_TARGET_KEPT", "1000"))
ANNOTATION_MAX_DISTANCE_RATIO = float(
    # Gold-train calibration: 0.10 keeps only very close orthographic matches.
    # More distant candidates remain visible for manual review.
    os.environ.get("PFE_ANNOTATION_MAX_DISTANCE_RATIO", "0.10")
)
ANNOTATION_MIN_CANDIDATE_MARGIN = float(
    os.environ.get("PFE_ANNOTATION_MIN_CANDIDATE_MARGIN", "0.08")
)
ANNOTATION_FRENCH_AMBIGUITY_MARGIN = float(
    os.environ.get("PFE_ANNOTATION_FRENCH_AMBIGUITY_MARGIN", "0.04")
)

# Reusable language resources and generated artifacts
LEXICON_PATH = ARTIFACTS_DIR / "lexicon.pkl"
FRENCH_WORDLIST_PATH = DATA_DIR / "Lexique400" / "Lexique4.tsv"
WOLOF_VOCAB_PATH = DATA_DIR / "vocabulary.txt"

# Parallel/external translation data
TRAIN_PARQUET_PATH = TRANSLATION_DATA_DIR / "train-00000-of-00001.parquet"
WOLOF_TRANSLATIONS_PATH = TRANSLATION_DATA_DIR / "wolof.csv"
DICTIONARY_TRANSLATIONS_PATH = (
    TRANSLATION_DATA_DIR / "dictionnary_translations.json"
)
TRANSLATION_INFORMAL_PATH = (
    TRANSLATION_DATA_DIR / "translation_informal_wolof.csv"
)
COMBINED_INFORMAL_PATH = TRANSLATION_DATA_DIR / "combined_informal_wolof.csv"

# Controlled synthetic-data benchmark outputs
M0_IDENTITY_PATH = SYNTHETIC_DATA_DIR / "m0_identity_pairs.csv"
M1_BEQI_RULES_PATH = SYNTHETIC_DATA_DIR / "m1_beqi_rules_pairs.csv"
M0_IDENTITY_MANIFEST_PATH = SYNTHETIC_DATA_DIR / "m0_identity_manifest.json"
M1_BEQI_RULES_MANIFEST_PATH = SYNTHETIC_DATA_DIR / "m1_beqi_rules_manifest.json"
M2_EFLOMAL_PATH = SYNTHETIC_DATA_DIR / "m2_eflomal_rules_pairs.csv"
M2_EFLOMAL_REVIEW_PATH = SYNTHETIC_DATA_DIR / "m2_eflomal_review_sample.csv"
M2_EFLOMAL_MANIFEST_PATH = SYNTHETIC_DATA_DIR / "m2_eflomal_manifest.json"
M3_CMDR_PATH = SYNTHETIC_DATA_DIR / "m3_cmdr_rules_pairs.csv"
M3_CMDR_REVIEW_PATH = SYNTHETIC_DATA_DIR / "m3_cmdr_review_sample.csv"
M3_CMDR_MANIFEST_PATH = SYNTHETIC_DATA_DIR / "m3_cmdr_manifest.json"
CMDR_MODEL_PATH = ARTIFACTS_DIR / "cmdr_word2vec.model"
CMDR_ALIGNMENT_PATH = ARTIFACTS_DIR / "cmdr_alignment_table.pkl"
M4_OOLEL_RAW_PATH = EXTERNAL_DATA_DIR / "oolel_raw.parquet"
M4_OOLEL_RAW_METADATA_PATH = EXTERNAL_DATA_DIR / "oolel_raw_metadata.json"
M4_OOLEL_PATH = SYNTHETIC_DATA_DIR / "m4_oolel_external_pairs.csv"
M4_OOLEL_REJECTED_PATH = SYNTHETIC_DATA_DIR / "m4_oolel_rejected_rows.csv"
M4_OOLEL_REVIEW_PATH = SYNTHETIC_DATA_DIR / "m4_oolel_manual_review_sample.csv"
M4_OOLEL_MANIFEST_PATH = SYNTHETIC_DATA_DIR / "m4_oolel_manifest.json"
M5_MAPPINGS_PATH = ARTIFACTS_DIR / "m5_youtube_variant_candidates.csv"
M5_ACCEPTED_MAPPINGS_PATH = ARTIFACTS_DIR / "m5_youtube_variant_mappings.csv"
M5_MAPPING_REVIEW_PATH = ARTIFACTS_DIR / "m5_mapping_review_sample.csv"
M5_GENERATION_REVIEW_PATH = SYNTHETIC_DATA_DIR / "m5_generation_review_sample.csv"
M5_YOUTUBE_PATH = SYNTHETIC_DATA_DIR / "m5_youtube_mined_pairs.csv"
M5_YOUTUBE_MANIFEST_PATH = SYNTHETIC_DATA_DIR / "m5_youtube_mined_manifest.json"
M6_LINGUISTIC_PATH = SYNTHETIC_DATA_DIR / "m6_gold_linguistic_pairs.csv"
M6_LINGUISTIC_MANIFEST_PATH = SYNTHETIC_DATA_DIR / "m6_gold_linguistic_manifest.json"
M6_PROFILE_PATH = ARTIFACTS_DIR / "m6_gold_linguistic_profile.json"
M6_GOLD_MAPPINGS_PATH = ARTIFACTS_DIR / "m6_gold_linguistic_mappings.csv"
M6_TRANSFORMATIONS_PATH = ARTIFACTS_DIR / "m6_gold_linguistic_transformations.csv"
M6_REVIEW_PATH = SYNTHETIC_DATA_DIR / "m6_gold_linguistic_review_sample.csv"
GOLD_TRAIN_PATH = GOLD_SPLITS_DIR / "gold_train.csv"
GOLD_DEV_PATH = GOLD_SPLITS_DIR / "gold_dev.csv"
GOLD_TEST_PATH = GOLD_SPLITS_DIR / "gold_test.csv"
GOLD_SPLIT_MANIFEST_PATH = GOLD_SPLITS_DIR / "split_manifest.json"

# M7 downstream normalization benchmark
M7_RUNS_DIR = CHECKPOINTS_DIR / "m7_seq2seq"
M7_RESULTS_DIR = PROJECT_ROOT / "results" / "m7_seq2seq"
M7_SUMMARY_PATH = M7_RESULTS_DIR / "benchmark_summary.csv"

# Reverse-pipeline evaluation files
REVERSE_TEST_INPUT_PATH = DATA_DIR / "test_exported_data.csv"
REVERSE_TEST_OUTPUT_PATH = DATA_DIR / "test_exported_data_recovered.csv"

# Scraper progress/input files
VIDEO_LINKS_PATH = SCRAPING_DIR / "all_video_links.txt"
SCRAPED_LINKS_PATH = SCRAPING_DIR / "already_scraped_links.txt"

# Optional character-level training input expected by train_lstm_seq2seq.py
SEQ2SEQ_DATA_PATH = PROJECT_ROOT / "data.csv"


def ensure_output_directories() -> None:
    """Create directories used by scripts that append or generate outputs."""
    for directory in (
        DATA_DIR,
        YOUTUBE_DATA_DIR,
        ANNOTATIONS_DIR,
        ARTIFACTS_DIR,
        CHECKPOINTS_DIR,
        SYNTHETIC_DATA_DIR,
        EXTERNAL_DATA_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
