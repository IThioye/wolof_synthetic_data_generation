"""Official InjongoIntent Wolof split handling and Laya conversion."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, Sequence


INJONGO_REVISION = "1f8be590da6699aee3dc23de6f63e801e2352eff"
INJONGO_SOURCE = (
    "https://github.com/masakhane-io/masakhane-nlu/"
    f"tree/{INJONGO_REVISION}/InjongoIntent/wol"
)
INJONGO_RAW_ROOT = (
    "https://raw.githubusercontent.com/masakhane-io/masakhane-nlu/"
    f"{INJONGO_REVISION}/InjongoIntent/wol"
)
OFFICIAL_SPLITS = ("train", "dev", "test")
EXPECTED_SPLIT_SIZES = {"train": 2240, "dev": 320, "test": 638}

INJONGO_INTENTS = (
    "alarm",
    "balance",
    "bill_balance",
    "book_flight",
    "book_hotel",
    "calendar_update",
    "cancel_reservation",
    "car_rental",
    "confirm_reservation",
    "cook_time",
    "exchange_rate",
    "food_last",
    "freeze_account",
    "ingredients_list",
    "interest_rate",
    "international_visa",
    "make_call",
    "meal_suggestion",
    "min_payment",
    "pay_bill",
    "pin_change",
    "play_music",
    "plug_type",
    "recipe",
    "restaurant_reservation",
    "restaurant_reviews",
    "restaurant_suggestion",
    "share_location",
    "shopping_list_update",
    "spending_history",
    "text",
    "time",
    "timezone",
    "transactions",
    "transfer",
    "translate",
    "travel_notification",
    "travel_suggestion",
    "update_playlist",
    "weather",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_official_splits(
    output_dir: str | Path = "data/external/injongo_intent_wol",
    *,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Download the three official files without redistributing examples."""

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for split in OFFICIAL_SPLITS:
        path = root / f"{split}.jsonl"
        if overwrite or not path.is_file():
            url = f"{INJONGO_RAW_ROOT}/{split}.jsonl"
            with urllib.request.urlopen(url, timeout=90) as response:
                payload = response.read()
            path.write_bytes(payload)
        paths[split] = path

    # A local provenance manifest is useful because data/external is ignored.
    manifest = {
        "dataset": "InjongoIntent/wol",
        "repository_revision": INJONGO_REVISION,
        "source": INJONGO_SOURCE,
        "files": {
            split: {
                "path": str(path.as_posix()),
                "sha256": _sha256(path),
                "rows": sum(1 for line in path.open(encoding="utf-8") if line.strip()),
            }
            for split, path in paths.items()
        },
    }
    (root / "download_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return paths


def load_split(path: str | Path, split: str) -> list[dict]:
    """Load one official split and attach invariant provenance fields."""

    if split not in OFFICIAL_SPLITS:
        raise ValueError(f"Unknown official split: {split!r}")
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Run download_official_splits() first."
        )
    rows: list[dict] = []
    seen_row_ids: set[str] = set()
    required = {"example_id", "text", "intent"}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            missing = required.difference(raw)
            if missing:
                raise ValueError(f"{path}:{line_number} missing {sorted(missing)}")
            example_id = str(raw["example_id"])
            # Injongo reuses IDs such as train-00000000 across intent blocks.
            # Preserve the official ID and add a collision-free composite key.
            row_id = f"{split}:{raw['intent']}:{example_id}"
            if row_id in seen_row_ids:
                raise ValueError(f"Duplicate composite row_id {row_id!r} in {path}")
            if not str(raw["text"]).strip():
                raise ValueError(f"Empty text for {example_id!r} in {path}")
            if raw["intent"] not in INJONGO_INTENTS:
                raise ValueError(f"Unexpected intent {raw['intent']!r} in {path}")
            seen_row_ids.add(row_id)
            rows.append(
                {
                    "example_id": example_id,
                    "row_id": row_id,
                    "text": str(raw["text"]),
                    "intent": str(raw["intent"]),
                    "split": split,
                    "source": "InjongoIntent/wol official",
                    "spans": raw.get("spans"),
                    "target": raw.get("target"),
                }
            )
    expected = EXPECTED_SPLIT_SIZES[split]
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} {split} rows, found {len(rows)}")
    return rows


def validate_official_splits(splits: Mapping[str, Sequence[Mapping]]) -> dict:
    """Fail fast on split leakage, label drift, or altered official sizes."""

    missing = set(OFFICIAL_SPLITS).difference(splits)
    if missing:
        raise ValueError(f"Missing official splits: {sorted(missing)}")
    ids = {name: {str(row["row_id"]) for row in splits[name]} for name in OFFICIAL_SPLITS}
    for left_index, left in enumerate(OFFICIAL_SPLITS):
        for right in OFFICIAL_SPLITS[left_index + 1 :]:
            overlap = ids[left].intersection(ids[right])
            if overlap:
                raise ValueError(
                    f"Official split leakage between {left} and {right}: "
                    f"{sorted(overlap)[:5]}"
                )
    labels = {
        split: {str(row["intent"]) for row in splits[split]}
        for split in OFFICIAL_SPLITS
    }
    expected_labels = set(INJONGO_INTENTS)
    for split, observed in labels.items():
        if observed != expected_labels:
            raise ValueError(
                f"Intent inventory mismatch in {split}: "
                f"missing={sorted(expected_labels-observed)}, "
                f"extra={sorted(observed-expected_labels)}"
            )
    return {
        "sizes": {split: len(splits[split]) for split in OFFICIAL_SPLITS},
        "intent_count": len(INJONGO_INTENTS),
        "train_intent_counts": dict(
            sorted(Counter(row["intent"] for row in splits["train"]).items())
        ),
        "id_overlap": False,
    }


def load_official_splits(
    root: str | Path = "data/external/injongo_intent_wol",
    *,
    download: bool = False,
) -> dict[str, list[dict]]:
    root = Path(root)
    if download:
        download_official_splits(root)
    splits = {
        split: load_split(root / f"{split}.jsonl", split)
        for split in OFFICIAL_SPLITS
    }
    validate_official_splits(splits)
    return splits


def make_laya_question(options: Sequence[str] = INJONGO_INTENTS) -> dict:
    """One stable typed-choice question shared by training and inference."""

    options = tuple(options)
    if len(options) != len(set(options)):
        raise ValueError("Laya options must be unique")
    return {
        "id": "intent",
        "t": "choice",
        "ins": "Choose the intent that best describes the Wolof utterance.",
        # Empty descriptions keep the 40-option head compact. The label is
        # still rendered and returned verbatim by Laya.
        "crit": {intent: "" for intent in options},
    }


def make_laya_state(text: str) -> dict[str, str]:
    return {"utterance": str(text)}


def build_laya_item(
    row: Mapping,
    tokenizer,
    config: Mapping,
    options: Sequence[str] = INJONGO_INTENTS,
) -> dict:
    """Convert one Injongo row with the official Laya sequence builder."""

    from laya.common import QTYPES, build_sequence

    options = tuple(options)
    intent = str(row["intent"])
    if intent not in options:
        raise ValueError(f"Gold intent {intent!r} is absent from Laya options")
    question = make_laya_question(options)
    ids, markers, stats = build_sequence(
        tokenizer,
        make_laya_state(str(row["text"])),
        question,
        max_len=int(config.get("max_len", 1024)),
        head_max_len=int(config.get("head_max_len", 256)),
        return_stats=True,
    )
    if len(markers) != len(options):
        raise ValueError(
            f"Laya retained {len(markers)}/{len(options)} intent markers for "
            f"{row['example_id']}; increase max_len/head_max_len"
        )
    if stats["options_distinct"] != len(options):
        raise ValueError(
            f"Only {stats['options_distinct']}/{len(options)} options remain "
            "token-distinct under the configured head budget"
        )
    label = options.index(intent)
    target = [0.0] * len(options)
    target[label] = 1.0
    return {
        "ids": ids,
        "markers": markers,
        "qtype": QTYPES["choice"],
        "target": target,
        "label": label,
        "example_id": str(row["example_id"]),
        "row_id": str(row.get("row_id", row["example_id"])),
        "intent": intent,
        "split": str(row["split"]),
        "option_stats": stats,
    }


def collate_laya_items(items: Sequence[Mapping], pad_id: int):
    """Collate prepared variable-length Laya examples."""

    import torch

    if not items:
        raise ValueError("Cannot collate an empty batch")
    n = len(items)
    sequence_length = max(len(item["ids"]) for item in items)
    option_count = max(len(item["markers"]) for item in items)
    ids = torch.full((n, sequence_length), pad_id, dtype=torch.long)
    attention = torch.zeros((n, sequence_length), dtype=torch.long)
    marker_positions = torch.zeros((n, option_count), dtype=torch.long)
    marker_mask = torch.zeros((n, option_count), dtype=torch.bool)
    target = torch.zeros((n, option_count), dtype=torch.float32)
    for index, item in enumerate(items):
        length = len(item["ids"])
        count = len(item["markers"])
        ids[index, :length] = torch.tensor(item["ids"], dtype=torch.long)
        attention[index, :length] = 1
        marker_positions[index, :count] = torch.tensor(item["markers"], dtype=torch.long)
        marker_mask[index, :count] = True
        target[index, : len(item["target"])] = torch.tensor(
            item["target"], dtype=torch.float32
        )
    return {
        "input_ids": ids,
        "attention_mask": attention,
        "marker_pos": marker_positions,
        "marker_mask": marker_mask,
        "target": target,
        "qtype": torch.tensor([item["qtype"] for item in items]),
        "label": torch.tensor([item["label"] for item in items]),
        "example_id": [str(item["example_id"]) for item in items],
    }


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data/external/injongo_intent_wol")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    download_official_splits(args.output_dir, overwrite=args.overwrite)
    splits = load_official_splits(args.output_dir)
    summary = validate_official_splits(splits)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _main()
