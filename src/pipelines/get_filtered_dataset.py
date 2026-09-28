"""Join checkpoint labels to their original comments and retain positive rows."""

import warnings

import pandas as pd

from pathlib import Path
import sys
external_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(external_dir))

from config import CHECKPOINT_DATA_PATH, CLEAN_COMMENTS_PATH, FILTERED_COMMENTS_PATH

warnings.filterwarnings("ignore")


def _coerce_boolean_labels(labels: pd.Series) -> pd.Series:
    """Convert persisted bool/string labels without treating 'False' as truthy."""
    if pd.api.types.is_bool_dtype(labels):
        return labels

    normalized = labels.astype(str).str.strip().str.casefold()
    mapping = {"true": True, "1": True, "false": False, "0": False}
    invalid = sorted(set(normalized) - set(mapping))
    if invalid:
        raise ValueError(f"Unsupported checkpoint labels: {invalid}")
    return normalized.map(mapping).astype(bool)


def build_filtered_dataset(
    clean_comments: pd.DataFrame, checkpoint_data: pd.DataFrame
) -> pd.DataFrame:
    """Return classified-positive comments joined by the original row index."""
    required_clean = {"clean_comment"}
    required_checkpoint = {"index", "is_informally_code_switched"}
    missing_clean = required_clean - set(clean_comments.columns)
    missing_checkpoint = required_checkpoint - set(checkpoint_data.columns)
    if missing_clean:
        raise ValueError(f"Clean dataset is missing columns: {sorted(missing_clean)}")
    if missing_checkpoint:
        raise ValueError(
            f"Checkpoint dataset is missing columns: {sorted(missing_checkpoint)}"
        )

    checkpoints = checkpoint_data.copy()
    checkpoints["index"] = pd.to_numeric(checkpoints["index"], errors="raise").astype(int)
    checkpoints["is_informally_code_switched"] = _coerce_boolean_labels(
        checkpoints["is_informally_code_switched"]
    )
    checkpoints = checkpoints.drop_duplicates("index", keep="last")

    comments = clean_comments.reset_index(names="source_index")
    merged = comments.merge(
        checkpoints.rename(columns={"index": "source_index"}),
        on="source_index",
        how="inner",
        validate="one_to_one",
    )
    filtered = merged.loc[merged["is_informally_code_switched"]].copy()

    output_columns = ["source_index", "clean_comment"]
    if "video_url" in filtered.columns:
        output_columns.append("video_url")
    output_columns.append("is_informally_code_switched")
    return filtered[output_columns].sort_values("source_index").reset_index(drop=True)


def main() -> None:
    clean_comments = pd.read_csv(CLEAN_COMMENTS_PATH)
    checkpoint_data = pd.read_csv(CHECKPOINT_DATA_PATH)
    filtered = build_filtered_dataset(clean_comments, checkpoint_data)

    FILTERED_COMMENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(FILTERED_COMMENTS_PATH, index=False, encoding="utf-8")

    classified = checkpoint_data["index"].nunique()
    percentage = 100 * len(filtered) / classified if classified else 0.0
    print(f"Classified comments: {classified:,}")
    print(f"Positive comments: {len(filtered):,} ({percentage:.1f}%)")
    print(f"Saved: {FILTERED_COMMENTS_PATH}")


if __name__ == "__main__":
    main()
