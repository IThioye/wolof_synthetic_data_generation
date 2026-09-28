"""Dataset preparation and leakage guards for the M7 seq2seq benchmark."""

from __future__ import annotations

import json
import random
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import pandas as pd

from src.config import (
    GOLD_DEV_PATH,
    GOLD_SPLIT_MANIFEST_PATH,
    GOLD_TEST_PATH,
    GOLD_TRAIN_PATH,
    M0_IDENTITY_PATH,
    M0_IDENTITY_MANIFEST_PATH,
    M1_BEQI_RULES_PATH,
    M1_BEQI_RULES_MANIFEST_PATH,
    M2_EFLOMAL_PATH,
    M2_EFLOMAL_MANIFEST_PATH,
    M3_CMDR_PATH,
    M3_CMDR_MANIFEST_PATH,
    M4_OOLEL_PATH,
    M4_OOLEL_MANIFEST_PATH,
    M5_YOUTUBE_PATH,
    M5_YOUTUBE_MANIFEST_PATH,
    M6_LINGUISTIC_PATH,
    M6_LINGUISTIC_MANIFEST_PATH,
)


PAIR_COLUMNS = {"source_id", "method", "informal_wolof", "formal_wolof"}
GOLD_COLUMNS = {
    "source_index",
    "comment",
    "manual_formal_wolof",
    "video_url",
}
METHOD_PATHS: dict[str, Path] = {
    "m0": M0_IDENTITY_PATH,
    "m1": M1_BEQI_RULES_PATH,
    "m2": M2_EFLOMAL_PATH,
    "m3": M3_CMDR_PATH,
    "m4": M4_OOLEL_PATH,
    "m5": M5_YOUTUBE_PATH,
    "m6": M6_LINGUISTIC_PATH,
}
METHOD_MANIFEST_PATHS: dict[str, Path] = {
    "m0": M0_IDENTITY_MANIFEST_PATH,
    "m1": M1_BEQI_RULES_MANIFEST_PATH,
    "m2": M2_EFLOMAL_MANIFEST_PATH,
    "m3": M3_CMDR_MANIFEST_PATH,
    "m4": M4_OOLEL_MANIFEST_PATH,
    "m5": M5_YOUTUBE_MANIFEST_PATH,
    "m6": M6_LINGUISTIC_MANIFEST_PATH,
}


@dataclass(frozen=True)
class LeakageAudit:
    training_rows: int
    held_out_rows: int
    input_overlap: int
    target_overlap: int
    pair_overlap: int

    @property
    def unsafe(self) -> bool:
        return self.input_overlap > 0 or self.pair_overlap > 0

    def as_dict(self) -> dict[str, int | bool]:
        return {
            "training_rows": self.training_rows,
            "held_out_rows": self.held_out_rows,
            "input_overlap": self.input_overlap,
            "target_overlap": self.target_overlap,
            "pair_overlap": self.pair_overlap,
            "unsafe": self.unsafe,
        }


def canonical_text(value: object) -> str:
    text = unicodedata.normalize("NFC", str(value) if value is not None else "")
    return " ".join(text.split()).casefold()


def available_methods(paths: Mapping[str, Path] = METHOD_PATHS) -> dict[str, bool]:
    return {method: Path(path).exists() for method, path in paths.items()}


def load_synthetic_pairs(
    method: str,
    *,
    path: Path | None = None,
    max_rows: int | None = None,
    seed: int = 2026,
    allow_provisional: bool = False,
    manifest_path: Path | None = None,
) -> pd.DataFrame:
    method_key = method.casefold()
    selected_path = Path(path or METHOD_PATHS.get(method_key, ""))
    if not selected_path.is_file():
        raise FileNotFoundError(f"Synthetic dataset for {method_key} is unavailable: {selected_path}")
    selected_manifest = Path(
        manifest_path or METHOD_MANIFEST_PATHS.get(method_key, "")
    )
    if selected_manifest.is_file():
        manifest = json.loads(selected_manifest.read_text(encoding="utf-8"))
        if manifest.get("provisional", False) and not allow_provisional:
            raise ValueError(
                f"{method_key} is provisional and cannot enter final M7 training: "
                f"{manifest.get('leakage_status', selected_manifest)}"
            )
        pending = manifest.get("finalization_required", [])
        if pending and not allow_provisional:
            raise ValueError(
                f"{method_key} has incomplete quality gates and cannot enter final M7 training: "
                + "; ".join(str(item) for item in pending)
            )
    data = pd.read_csv(selected_path)
    missing = PAIR_COLUMNS - set(data.columns)
    if missing:
        raise ValueError(f"{selected_path.name} is missing columns: {sorted(missing)}")
    data = data.copy()
    for column in ("informal_wolof", "formal_wolof"):
        data[column] = data[column].fillna("").astype(str).map(lambda text: " ".join(text.split()))
    if data["informal_wolof"].eq("").any() or data["formal_wolof"].eq("").any():
        raise ValueError(f"{selected_path.name} contains empty training text")
    if data["source_id"].duplicated().any():
        raise ValueError(f"{selected_path.name} contains duplicate source_id values")
    if max_rows is not None:
        if max_rows <= 0:
            raise ValueError("max_rows must be positive")
        data = data.sample(min(max_rows, len(data)), random_state=seed)
    output = data[["source_id", "method", "informal_wolof", "formal_wolof"]].rename(
        columns={"informal_wolof": "source_text", "formal_wolof": "target_text"}
    )
    output["source_type"] = "synthetic"
    return output.reset_index(drop=True)


def load_gold_split(path: Path, split_name: str) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Locked gold {split_name} split is unavailable: {path}")
    data = pd.read_csv(path)
    missing = GOLD_COLUMNS - set(data.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
    data = data.copy()
    data["comment"] = data["comment"].fillna("").astype(str).map(lambda text: " ".join(text.split()))
    data["manual_formal_wolof"] = (
        data["manual_formal_wolof"].fillna("").astype(str).map(lambda text: " ".join(text.split()))
    )
    if data["comment"].eq("").any() or data["manual_formal_wolof"].eq("").any():
        raise ValueError(f"{path.name} contains empty text")
    output = pd.DataFrame(
        {
            "source_id": [f"gold:{split_name}:{value}" for value in data["source_index"]],
            "method": f"gold_{split_name}",
            "source_text": data["comment"],
            "target_text": data["manual_formal_wolof"],
            "source_type": "gold",
            "video_url": data["video_url"].fillna("").astype(str),
        }
    )
    return output.reset_index(drop=True)


def load_locked_gold() -> dict[str, pd.DataFrame]:
    if not GOLD_SPLIT_MANIFEST_PATH.is_file():
        raise FileNotFoundError(
            f"Gold split manifest is not locked yet: {GOLD_SPLIT_MANIFEST_PATH}"
        )
    splits = {
        "train": load_gold_split(GOLD_TRAIN_PATH, "train"),
        "dev": load_gold_split(GOLD_DEV_PATH, "dev"),
        "test": load_gold_split(GOLD_TEST_PATH, "test"),
    }
    validate_gold_disjointness(splits)
    manifest = json.loads(GOLD_SPLIT_MANIFEST_PATH.read_text(encoding="utf-8"))
    missing_manifest_splits = set(splits) - set(manifest.get("splits", {}))
    if missing_manifest_splits:
        raise ValueError(
            "Gold manifest is incomplete: " + ", ".join(sorted(missing_manifest_splits))
        )
    return splits


def validate_gold_disjointness(splits: Mapping[str, pd.DataFrame]) -> None:
    names = list(splits)
    for left_index, left_name in enumerate(names):
        left = splits[left_name]
        if left["source_id"].duplicated().any():
            raise ValueError(f"Gold {left_name} contains duplicate source IDs")
        left_videos = set(left.get("video_url", pd.Series(dtype=str)).dropna()) - {""}
        for right_name in names[left_index + 1 :]:
            right = splits[right_name]
            right_videos = set(right.get("video_url", pd.Series(dtype=str)).dropna()) - {""}
            overlap = left_videos & right_videos
            if overlap:
                raise ValueError(
                    f"Gold video leakage between {left_name} and {right_name}: {sorted(overlap)}"
                )


def assemble_training_data(
    synthetic: pd.DataFrame | None,
    gold_train: pd.DataFrame | None,
    *,
    regime: str,
    seed: int = 2026,
) -> pd.DataFrame:
    regime = regime.casefold()
    if regime == "synthetic_only":
        if synthetic is None:
            raise ValueError("synthetic_only requires a synthetic dataset")
        parts = [synthetic]
    elif regime == "gold_only":
        if gold_train is None:
            raise ValueError("gold_only requires the locked gold training split")
        parts = [gold_train]
    elif regime == "synthetic_plus_gold":
        if synthetic is None or gold_train is None:
            raise ValueError("synthetic_plus_gold requires synthetic and gold training data")
        # Gold is placed first so an exact duplicate retains authentic provenance.
        parts = [gold_train, synthetic]
    else:
        raise ValueError(f"Unknown training regime: {regime}")
    combined = pd.concat(parts, ignore_index=True)
    combined["pair_key"] = [
        (canonical_text(source), canonical_text(target))
        for source, target in zip(combined["source_text"], combined["target_text"])
    ]
    combined = combined.drop_duplicates("pair_key", keep="first").drop(columns="pair_key")
    order = list(range(len(combined)))
    random.Random(seed).shuffle(order)
    return combined.iloc[order].reset_index(drop=True)


def audit_training_leakage(training: pd.DataFrame, held_out: pd.DataFrame) -> LeakageAudit:
    train_inputs = {canonical_text(value) for value in training["source_text"]}
    train_targets = {canonical_text(value) for value in training["target_text"]}
    train_pairs = {
        (canonical_text(source), canonical_text(target))
        for source, target in zip(training["source_text"], training["target_text"])
    }
    held_inputs = {canonical_text(value) for value in held_out["source_text"]}
    held_targets = {canonical_text(value) for value in held_out["target_text"]}
    held_pairs = {
        (canonical_text(source), canonical_text(target))
        for source, target in zip(held_out["source_text"], held_out["target_text"])
    }
    return LeakageAudit(
        training_rows=len(training),
        held_out_rows=len(held_out),
        input_overlap=len(train_inputs & held_inputs),
        target_overlap=len(train_targets & held_targets),
        pair_overlap=len(train_pairs & held_pairs),
    )


def require_no_held_out_leakage(training: pd.DataFrame, held_out: pd.DataFrame) -> LeakageAudit:
    audit = audit_training_leakage(training, held_out)
    if audit.unsafe:
        raise ValueError(f"Unsafe held-out overlap detected: {audit.as_dict()}")
    return audit
