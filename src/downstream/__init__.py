"""Downstream evaluations of the Wolof normalization models.

The modules in this package are intentionally separate from the historical
normalization experiments under :mod:`src.modeling`.
"""

from .injongo import (
    INJONGO_INTENTS,
    INJONGO_REVISION,
    build_laya_item,
    load_official_splits,
)

__all__ = [
    "INJONGO_INTENTS",
    "INJONGO_REVISION",
    "build_laya_item",
    "load_official_splits",
]
