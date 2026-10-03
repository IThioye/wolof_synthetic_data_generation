"""Immutable references and guards for downstream analysis V2."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Mapping


V1_RESULTS_DIR = Path("results/downstream/laya_normalization_v1")
V2_RESULTS_DIR = Path("results/downstream/laya_normalization_v2")
LAYA_CHECKPOINT = Path("checkpoints/downstream/laya_normalization_v1")
TNT_CHECKPOINT = Path(
    "checkpoints/m7_seq2seq/tnt_hybrid_v3_optimization_v1/m1_v3/seed_2026/"
    "gold_adapt_lr_5e5_word_keep_t050/attempt_001/best_model"
)
INJONGO_TEST = Path("data/external/injongo_intent_wol/test.jsonl")

V1_REAL_ACCURACY = 0.8087774294670846
V1_REAL_MACRO_F1 = 0.8117716357020083
V1_LAYA_SHA256 = "9f3aa8e8040e83db2dcb564e0645e73fd19025a0376b0aa09d8a838bac1396ce"
V1_TNT_SHA256 = "78b1cbc9777ac8fbb07b189d277cdeb387bea845e5e340f709ae0eaa2d967780"
INJONGO_TEST_SHA256 = "97956d275da88024942085b56ae665340b3a5b7045c343be94ae24267b04b01e"
DEFAULT_METRIC_TOLERANCE = 1e-9


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_hash(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Missing frozen {label}: {path}")
    observed = sha256_file(path)
    if observed != expected:
        raise RuntimeError(
            f"Frozen {label} hash changed: expected {expected}, observed {observed}"
        )
    return observed


def validate_frozen_artifacts(repository_root: str | Path = ".") -> dict:
    root = Path(repository_root).resolve()
    laya_model = root / LAYA_CHECKPOINT / "model.safetensors"
    tnt_model = root / TNT_CHECKPOINT / "model.safetensors"
    test_path = root / INJONGO_TEST
    laya_hash = _assert_hash(laya_model, V1_LAYA_SHA256, "Laya checkpoint")
    tnt_hash = _assert_hash(tnt_model, V1_TNT_SHA256, "TNT checkpoint")
    test_hash = _assert_hash(test_path, INJONGO_TEST_SHA256, "Injongo test split")
    training_manifest_path = root / LAYA_CHECKPOINT / "training_manifest.json"
    training_manifest = json.loads(training_manifest_path.read_text(encoding="utf-8"))
    if training_manifest.get("test_used") is not False:
        raise RuntimeError("The frozen Laya training manifest does not protect test")
    return {
        "laya_checkpoint": str((root / LAYA_CHECKPOINT).resolve()),
        "laya_model_sha256": laya_hash,
        "tnt_checkpoint": str((root / TNT_CHECKPOINT).resolve()),
        "tnt_model_sha256": tnt_hash,
        "injongo_test": str(test_path.resolve()),
        "injongo_test_sha256": test_hash,
    }


def assert_v1_real_baseline(
    metrics: Mapping[str, float],
    *,
    tolerance: float = DEFAULT_METRIC_TOLERANCE,
) -> None:
    expected = {
        "accuracy": V1_REAL_ACCURACY,
        "macro_f1": V1_REAL_MACRO_F1,
    }
    for name, value in expected.items():
        observed = float(metrics[name])
        if abs(observed - value) > tolerance:
            raise RuntimeError(
                f"REAL baseline drift for {name}: expected {value:.12f}, "
                f"observed {observed:.12f}, tolerance {tolerance}"
            )


def current_git_commit(repository_root: str | Path = ".") -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(repository_root),
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
