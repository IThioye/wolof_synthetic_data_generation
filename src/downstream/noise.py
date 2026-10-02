"""Reproducible synthetic-noise preparation for the downstream benchmark."""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Mapping

from src.generation.benchmark_baselines import informalize_beqi_rules


DEFAULT_NOISE_METHOD = "m1_beqi_inspired_rules"


def apply_m1_noise(text: str) -> tuple[str, dict[str, int]]:
    """Apply the existing deterministic M1 generator to one utterance."""

    noisy, operations = informalize_beqi_rules(str(text))
    if not isinstance(operations, Counter):
        operations = Counter(operations)
    return noisy, dict(sorted((str(key), int(value)) for key, value in operations.items()))


def build_noisy_records(
    records: Iterable[Mapping],
    *,
    seed: int = 2026,
    method: str = DEFAULT_NOISE_METHOD,
) -> list[dict]:
    """Preserve IDs/labels while producing one synthetic-noise condition.

    M1 is deterministic; ``seed`` is nevertheless recorded as part of the
    experimental manifest so the interface can later accommodate stochastic
    existing generators without changing the output schema.
    """

    if method != DEFAULT_NOISE_METHOD:
        raise ValueError(f"Unsupported V1 noise method: {method!r}")
    output: list[dict] = []
    for row in records:
        real_text = str(row["text"])
        noisy_text, operations = apply_m1_noise(real_text)
        output.append(
            {
                "example_id": str(row["example_id"]),
                "row_id": str(row.get("row_id", row["example_id"])),
                "real_text": real_text,
                "noisy_text": noisy_text,
                "intent": str(row["intent"]),
                "split": str(row["split"]),
                "source": str(row.get("source", "InjongoIntent/wol official")),
                "noise_method": method,
                "noise_kind": "synthetic noisy Wolof",
                "noise_operations": operations,
                "seed": int(seed),
                "changed": noisy_text != real_text,
            }
        )
    return output
