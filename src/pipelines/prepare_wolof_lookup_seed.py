"""Convert the Wolof dictionary CSV into the M6 annotation seed lookup."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.config import FORMAL_TOKEN_LOOKUP_SEED_PATH, WOLOF_LOOKUP_SOURCE_PATH


REQUIRED_SOURCE_COLUMNS = {"word", "pos", "translation", "pos full name", "categorie"}


def clean(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return " ".join(str(value).strip().split())


def map_universal_pos_candidates(
    pos: object, full_name: object, category: object
) -> tuple[str, ...]:
    """Map one dictionary sense to every plausible conservative UPOS label."""
    short = clean(pos).casefold()
    full = clean(full_name).casefold()
    broad = clean(category).casefold()
    descriptor = " | ".join((short, full, broad))

    if not descriptor.replace("|", "").strip():
        return ()
    if broad == "." or full == "." or short == ".":
        return ("PUNCT",)
    if "nom propre" in full or short == "np.":
        return ("PROPN",)
    if "verbe auxiliaire" in full or "v.aux" in short or "verbal auxiliaire" in full:
        return ("AUX",)

    candidates: set[str] = set()
    if "locution adverbiale" in full or "adverbe" in full or broad == "adverbe":
        candidates.add("ADV")
    if "locution conjonctive" in full or "conjonction" in descriptor:
        candidates.add("CCONJ")
    if "locution prépositive" in full or "préposition" in descriptor:
        candidates.add("ADP")
    if "locution verbale" in full or "verbe" in descriptor or short.startswith("v."):
        candidates.add("VERB")
    if "adjectif" in descriptor:
        candidates.add("ADJ")
    if "pronom" in descriptor or "substitut nominal" in descriptor:
        candidates.add("PRON")
    if (
        "article" in descriptor
        or "déterminant" in descriptor
        or "art." in short
        or short.startswith("art")
    ):
        candidates.add("DET")
    if "numéral" in descriptor:
        candidates.add("NUM")
    if "interjection" in descriptor or "exclamation" in descriptor:
        candidates.add("INTJ")
    if (
        broad == "nom"
        or full == "nom"
        or full.startswith("nom ")
    ) and "nom propre" not in full:
        candidates.add("NOUN")
    if (
        "particule" in descriptor
        or "marqueur" in descriptor
        or "marque aspectuelle" in descriptor
        or "inaccompli" in descriptor
    ):
        candidates.add("PART")
    if "onomatop" in descriptor:
        candidates.add("INTJ")
    if "idéophone" in descriptor or "idéo." in descriptor:
        candidates.add("X")
    if not candidates and (
        "locution" in descriptor or "syntagme" in descriptor or "expression" in descriptor
    ):
        candidates.add("X")
    return tuple(sorted(candidates))


def map_universal_pos(pos: object, full_name: object, category: object) -> str:
    """Return a single UPOS label only when one interpretation is available."""
    candidates = map_universal_pos_candidates(pos, full_name, category)
    return candidates[0] if len(candidates) == 1 else "UNKNOWN"


def json_values(values: pd.Series) -> str:
    unique = []
    for value in values:
        cleaned = clean(value)
        if cleaned and cleaned not in unique:
            unique.append(cleaned)
    return json.dumps(unique, ensure_ascii=False)


def build_seed_lookup(dictionary: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_SOURCE_COLUMNS - set(dictionary.columns)
    if missing:
        raise ValueError(f"Wolof lookup source is missing columns: {sorted(missing)}")

    data = dictionary.copy()
    for column in REQUIRED_SOURCE_COLUMNS:
        data[column] = data[column].map(clean)
    data = data.loc[data["word"].ne("")].copy()
    data["formal_key"] = data["word"].str.casefold()
    data["mapped_pos_candidates"] = data.apply(
        lambda row: map_universal_pos_candidates(
            row["pos"], row["pos full name"], row["categorie"]
        ),
        axis=1,
    )

    records: list[dict[str, object]] = []
    for _key, group in data.groupby("formal_key", sort=True):
        formal_token = group.iloc[0]["word"]
        candidates = sorted(
            {
                candidate
                for row_candidates in group["mapped_pos_candidates"]
                for candidate in row_candidates
            }
        )
        pos = candidates[0] if len(candidates) == 1 else "UNKNOWN"
        ambiguous = len(candidates) > 1
        records.append(
            {
                "formal_token": formal_token,
                "lemma": formal_token if " " not in formal_token else "",
                "pos": pos,
                "target_language": "wo",
                "entity_type": "UNKNOWN",
                "protected": "no",
                "review_status": "reviewed",
                "source": "lookup_table_wolof.csv",
                "language_candidates_json": json.dumps(["wo"], ensure_ascii=False),
                "language_ambiguous": "false",
                "entry_count": len(group),
                "ambiguous_pos": str(ambiguous).lower(),
                "pos_candidates_json": json.dumps(candidates, ensure_ascii=False),
                "dictionary_pos_json": json_values(group["pos full name"]),
                "dictionary_categories_json": json_values(group["categorie"]),
                "translations_json": json_values(group["translation"]),
                "lemmas_json": json.dumps([formal_token], ensure_ascii=False),
            }
        )
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=WOLOF_LOOKUP_SOURCE_PATH)
    parser.add_argument("--output", type=Path, default=FORMAL_TOKEN_LOOKUP_SEED_PATH)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite seed lookup: {args.output}")
    dictionary = pd.read_csv(args.input, keep_default_na=False)
    seed = build_seed_lookup(dictionary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    seed.to_csv(args.output, index=False, encoding="utf-8-sig")
    ambiguous = int(seed["ambiguous_pos"].eq("true").sum())
    unknown = int(seed["pos"].eq("UNKNOWN").sum())
    print(
        f"Wrote {len(seed):,} unique Wolof lookup entries to {args.output} "
        f"({ambiguous:,} POS-ambiguous; {unknown:,} UNKNOWN)."
    )


if __name__ == "__main__":
    main()
