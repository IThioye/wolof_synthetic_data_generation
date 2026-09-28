"""Windows-native Wolof-French lexicon induction with NLTK IBM Model 1."""

from __future__ import annotations

import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from nltk.translate import AlignedSent, IBMModel1


def word_tokens(text: str) -> list[str]:
    """Tokenize exactly as the existing Eflomal experiment does."""
    return re.findall(r"\w+(?:[-'’]\w+)*", str(text).lower(), flags=re.UNICODE)


def prepare_bitext(
    dataframe: pd.DataFrame,
    wolof_col: str = "wolof",
    french_col: str = "french",
) -> list[AlignedSent]:
    """Convert sentence pairs to NLTK's French-target/Wolof-source convention."""
    missing = {wolof_col, french_col} - set(dataframe.columns)
    if missing:
        raise ValueError(f"Parallel data is missing columns: {sorted(missing)}")

    bitext = []
    for wolof_text, french_text in dataframe[[wolof_col, french_col]].itertuples(
        index=False, name=None
    ):
        wolof = word_tokens(wolof_text)
        french = word_tokens(french_text)
        if not wolof or not french:
            continue
        # NLTK learns translation from `mots` to `words`. We therefore put
        # Wolof in mots (source) and French in words (target).
        bitext.append(AlignedSent(words=french, mots=wolof))
    if not bitext:
        raise ValueError("No non-empty parallel sentence pairs were found")
    return bitext


def train_ibm1(
    dataframe: pd.DataFrame,
    iterations: int = 5,
    wolof_col: str = "wolof",
    french_col: str = "french",
) -> tuple[IBMModel1, list[AlignedSent]]:
    """Train IBM Model 1 and return both the model and its aligned corpus."""
    if iterations < 1:
        raise ValueError("iterations must be at least 1")
    bitext = prepare_bitext(dataframe, wolof_col, french_col)
    model = IBMModel1(bitext, iterations)
    model.align_all(bitext)
    return model, bitext


def build_hard_alignment_lexicon(
    bitext: list[AlignedSent],
    min_count: int = 3,
    min_prob: float = 0.1,
) -> dict[str, dict[str, float]]:
    """Build the same Wolof -> {French: probability} structure as Eflomal."""
    if min_count < 1:
        raise ValueError("min_count must be at least 1")
    if not 0 <= min_prob <= 1:
        raise ValueError("min_prob must be between 0 and 1")

    cooccurrences: dict[str, Counter] = defaultdict(Counter)
    wolof_totals = Counter()

    for sentence in bitext:
        # AlignedSent stores pairs as (words_index, mots_index), which here is
        # (French index, Wolof index). A NULL source alignment uses None.
        for french_index, wolof_index in sentence.alignment:
            if french_index is None or wolof_index is None:
                continue
            if french_index >= len(sentence.words) or wolof_index >= len(sentence.mots):
                continue
            wolof_word = sentence.mots[wolof_index]
            french_word = sentence.words[french_index]
            cooccurrences[wolof_word][french_word] += 1
            wolof_totals[wolof_word] += 1

    lexicon = {}
    for wolof_word, french_counts in cooccurrences.items():
        total = wolof_totals[wolof_word]
        translations = {
            french_word: count / total
            for french_word, count in french_counts.items()
            if count >= min_count and count / total >= min_prob
        }
        if translations:
            lexicon[wolof_word] = dict(
                sorted(translations.items(), key=lambda item: item[1], reverse=True)
            )
    return lexicon


def train_lexicon(
    dataframe: pd.DataFrame,
    iterations: int = 5,
    min_count: int = 3,
    min_prob: float = 0.1,
    wolof_col: str = "wolof",
    french_col: str = "french",
):
    """Train IBM Model 1 and extract an Eflomal-compatible hard lexicon."""
    model, bitext = train_ibm1(dataframe, iterations, wolof_col, french_col)
    lexicon = build_hard_alignment_lexicon(bitext, min_count, min_prob)
    return lexicon, model, bitext


def save_lexicon(lexicon: dict, path: Path | str) -> Path:
    """Persist a lexicon without changing the active Eflomal artifact."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as handle:
        pickle.dump(lexicon, handle)
    return output_path


def top_translations(lexicon: dict, wolof_word: str, limit: int = 5):
    translations = lexicon.get(wolof_word.casefold(), {})
    return list(translations.items())[:limit]
