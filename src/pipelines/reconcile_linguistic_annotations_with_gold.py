"""Remove linguistic rows whose occurrence no longer matches gold train.

Dry-run is the default. Source files must be archived before using ``--apply``.
The application can then recreate safe automatic rows and expose any genuinely
changed token pairs for review.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.annotation.linguistic_annotation_core import build_occurrences
from src.config import GOLD_TRAIN_PATH, LINGUISTIC_ANNOTATIONS_PATH


def invalid_annotation_mask(
    gold_train: pd.DataFrame, annotations: pd.DataFrame
) -> pd.Series:
    occurrences = build_occurrences(gold_train).set_index("occurrence_id")
    invalid = pd.Series(False, index=annotations.index, dtype=bool)
    for index, row in annotations.iterrows():
        occurrence_id = str(row["occurrence_id"])
        if occurrence_id not in occurrences.index:
            invalid.loc[index] = True
            continue
        current = occurrences.loc[occurrence_id]
        invalid.loc[index] = (
            str(row["informal_token"]) != str(current["informal_token"])
            or str(row["formal_token"]) != str(current["formal_token"])
        )
    return invalid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-train", type=Path, default=GOLD_TRAIN_PATH)
    parser.add_argument("--annotations", type=Path, default=LINGUISTIC_ANNOTATIONS_PATH)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    gold_train = pd.read_csv(
        args.gold_train, keep_default_na=False, dtype={"source_index": "string"}
    )
    annotations = pd.read_csv(
        args.annotations, keep_default_na=False, dtype={"source_index": "string"}
    )
    invalid = invalid_annotation_mask(gold_train, annotations)
    removed = annotations.loc[invalid]
    print(
        f"Invalid rows: {len(removed)}; kept rows: {len(annotations) - len(removed)}"
    )
    if not removed.empty:
        print(
            removed[["occurrence_id", "informal_token", "formal_token"]]
            .to_string(index=False)
        )
    if not args.apply:
        print("Dry run only; no files were changed.")
        return

    temporary = args.annotations.with_suffix(".gold-reconcile.tmp.csv")
    annotations.loc[~invalid].to_csv(temporary, index=False, encoding="utf-8")
    temporary.replace(args.annotations)
    print(f"Removed {len(removed)} invalid linguistic rows.")


if __name__ == "__main__":
    main()
