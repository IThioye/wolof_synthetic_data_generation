"""Build the gold-train Wolof/French lexical prefill used by the M6 app.

Reviewed names take precedence.  Wolof/French surface-form overlaps are kept
as explicit language ambiguities instead of silently choosing either resource.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.config import (
    FORMAL_TOKEN_LOOKUP_SEED_PATH,
    FRENCH_WORDLIST_PATH,
    GOLD_TRAIN_PATH,
    SENEGALESE_SURNAMES_PATH,
    WOLOF_LOOKUP_SOURCE_PATH,
)
from src.pipelines.prepare_wolof_lookup_seed import build_seed_lookup, clean


FRENCH_POS_MAP = {
    "VER": "VERB",
    "NOM": "NOUN",
    "ADJ": "ADJ",
    "ADV": "ADV",
    "ONO": "INTJ",
    "PRE": "ADP",
    "AUX": "AUX",
    "CON": "CCONJ",
    "PRO:per": "PRON",
    "PRO:ind": "PRON",
    "PRO:pos": "PRON",
    "PRO:rel": "PRON",
    "PRO:int": "PRON",
    "PRO:dem": "PRON",
    "ART:def": "DET",
    "ART:ind": "DET",
    "ADJ:num": "NUM",
    "ADJ:ind": "DET",
    "ADJ:pos": "DET",
    "ADJ:int": "DET",
    "ADJ:dem": "DET",
}


def extract_formal_units(gold_train: pd.DataFrame) -> dict[str, str]:
    if "token_corrections_json" not in gold_train.columns:
        raise ValueError("Gold train is missing token_corrections_json")
    units: dict[str, str] = {}
    for payload_text in gold_train["token_corrections_json"]:
        payload = json.loads(payload_text or "[]")
        for token_row in payload:
            informal = clean(token_row.get("token"))
            formal = (
                clean(token_row.get("final_correction"))
                or clean(token_row.get("selected_candidate"))
                or informal
            )
            if formal:
                units.setdefault(formal.casefold(), formal)
    return units


def build_french_seed_lookup(
    lexique: pd.DataFrame,
    formal_units: dict[str, str],
    excluded_keys: set[str],
) -> pd.DataFrame:
    required = {"1_Mot", "4_Lemme", "5_Cgram"}
    missing = required - set(lexique.columns)
    if missing:
        raise ValueError(f"Lexique is missing columns: {sorted(missing)}")

    relevant_keys = set(formal_units) - set(excluded_keys)
    data = lexique.copy()
    for column in required:
        data[column] = data[column].map(clean)
    data["formal_key"] = data["1_Mot"].str.casefold()
    data = data.loc[data["formal_key"].isin(relevant_keys)].copy()
    data["mapped_pos"] = data["5_Cgram"].map(FRENCH_POS_MAP).fillna("UNKNOWN")

    records: list[dict[str, object]] = []
    for key, group in data.groupby("formal_key", sort=True):
        candidates = sorted(set(group["mapped_pos"]) - {"UNKNOWN"})
        pos = candidates[0] if len(candidates) == 1 else "UNKNOWN"
        lemmas = list(dict.fromkeys(value for value in group["4_Lemme"] if value))
        formal_token = formal_units[key]
        lemma = lemmas[0] if len(lemmas) == 1 else formal_token
        records.append(
            {
                "formal_token": formal_token,
                "lemma": lemma,
                "pos": pos,
                "target_language": "fr",
                "entity_type": "UNKNOWN",
                "protected": "yes",
                "review_status": "reviewed",
                "source": "Lexique4.tsv",
                "language_candidates_json": json.dumps(["fr"], ensure_ascii=False),
                "language_ambiguous": "false",
                "entry_count": len(group),
                "ambiguous_pos": str(len(candidates) > 1).lower(),
                "pos_candidates_json": json.dumps(candidates, ensure_ascii=False),
                "dictionary_pos_json": json.dumps(
                    list(dict.fromkeys(group["5_Cgram"])), ensure_ascii=False
                ),
                "dictionary_categories_json": json.dumps([], ensure_ascii=False),
                "translations_json": json.dumps([], ensure_ascii=False),
                "lemmas_json": json.dumps(lemmas, ensure_ascii=False),
            }
        )
    return pd.DataFrame(records)


def build_name_seed_lookup(names: list[str]) -> pd.DataFrame:
    canonical: dict[str, str] = {}
    for value in names:
        name = clean(value)
        if name:
            canonical.setdefault(name.casefold(), name)
    records = []
    for key in sorted(canonical):
        name = canonical[key]
        records.append(
            {
                "formal_token": name,
                "lemma": name,
                "pos": "PROPN",
                "target_language": "unknown",
                "entity_type": "PERSON",
                "protected": "yes",
                "review_status": "reviewed",
                "source": "senegalese_surnames.txt",
                "language_candidates_json": json.dumps(["unknown"], ensure_ascii=False),
                "language_ambiguous": "false",
                "entry_count": 1,
                "ambiguous_pos": "false",
                "pos_candidates_json": json.dumps(["PROPN"], ensure_ascii=False),
                "dictionary_pos_json": json.dumps(["proper person name"], ensure_ascii=False),
                "dictionary_categories_json": json.dumps(["PERSON"], ensure_ascii=False),
                "translations_json": json.dumps([], ensure_ascii=False),
                "lemmas_json": json.dumps([name], ensure_ascii=False),
            }
        )
    return pd.DataFrame(records)


def _json_list(row: pd.Series, column: str) -> list[str]:
    try:
        values = json.loads(clean(row.get(column)) or "[]")
    except (json.JSONDecodeError, TypeError):
        values = []
    return [clean(value) for value in values if clean(value)] if isinstance(values, list) else []


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def merge_language_ambiguous_entries(
    wolof: pd.DataFrame, french: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Separate resource-only entries and combine Wolof/French homographs."""
    if wolof.empty or french.empty:
        return wolof, french, pd.DataFrame()
    wolof_by_key = {
        clean(row["formal_token"]).casefold(): row
        for _, row in wolof.iterrows()
    }
    french_by_key = {
        clean(row["formal_token"]).casefold(): row
        for _, row in french.iterrows()
    }
    overlap = set(wolof_by_key) & set(french_by_key)
    records: list[dict[str, object]] = []
    for key in sorted(overlap):
        wo = wolof_by_key[key]
        fr = french_by_key[key]
        wo_pos = _json_list(wo, "pos_candidates_json")
        fr_pos = _json_list(fr, "pos_candidates_json")
        candidates = sorted(set(wo_pos) | set(fr_pos))
        records.append(
            {
                "formal_token": clean(wo["formal_token"]),
                "lemma": clean(wo["formal_token"]),
                "pos": candidates[0] if len(candidates) == 1 else "UNKNOWN",
                "target_language": "unknown",
                "entity_type": "UNKNOWN",
                "protected": "uncertain",
                "review_status": "uncertain",
                "source": "lookup_table_wolof.csv+Lexique4.tsv",
                "language_candidates_json": json.dumps(["wo", "fr"], ensure_ascii=False),
                "language_ambiguous": "true",
                "entry_count": int(clean(wo.get("entry_count")) or 0)
                + int(clean(fr.get("entry_count")) or 0),
                "ambiguous_pos": str(len(candidates) > 1).lower(),
                "pos_candidates_json": json.dumps(candidates, ensure_ascii=False),
                "pos_candidates_by_language_json": json.dumps(
                    {"wo": wo_pos, "fr": fr_pos}, ensure_ascii=False
                ),
                "dictionary_pos_json": json.dumps(
                    _unique(
                        [f"Wolof: {value}" for value in _json_list(wo, "dictionary_pos_json")]
                        + [f"French: {value}" for value in _json_list(fr, "dictionary_pos_json")]
                    ),
                    ensure_ascii=False,
                ),
                "dictionary_categories_json": clean(wo.get("dictionary_categories_json")) or "[]",
                "translations_json": clean(wo.get("translations_json")) or "[]",
                "lemmas_json": json.dumps(
                    _unique(
                        _json_list(wo, "lemmas_json") + _json_list(fr, "lemmas_json")
                    ),
                    ensure_ascii=False,
                ),
            }
        )
    wolof_only = wolof.loc[
        ~wolof["formal_token"].str.casefold().isin(overlap)
    ].copy()
    french_only = french.loc[
        ~french["formal_token"].str.casefold().isin(overlap)
    ].copy()
    return wolof_only, french_only, pd.DataFrame(records)


def build_multilingual_seed(
    wolof_dictionary: pd.DataFrame,
    lexique: pd.DataFrame,
    gold_train: pd.DataFrame,
    names: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    wolof = build_seed_lookup(wolof_dictionary)
    name_seed = build_name_seed_lookup(names or [])
    name_keys = (
        set(name_seed["formal_token"].str.casefold()) if not name_seed.empty else set()
    )
    # A reviewed Senegalese name inventory is more specific than a generic
    # Wolof or French dictionary surface-form match.
    wolof = wolof.loc[~wolof["formal_token"].str.casefold().isin(name_keys)].copy()
    formal_units = extract_formal_units(gold_train)
    french = build_french_seed_lookup(lexique, formal_units, name_keys)
    wolof, french, ambiguous_language = merge_language_ambiguous_entries(
        wolof, french
    )
    combined = pd.concat(
        [name_seed, ambiguous_language, wolof, french], ignore_index=True, sort=False
    ).fillna("")
    if combined["formal_token"].str.casefold().duplicated().any():
        raise RuntimeError("Combined lookup contains duplicate case-folded headwords")
    stats = {
        "wolof_entries": len(wolof),
        "name_entries": len(name_seed),
        "ambiguous_language_entries": len(ambiguous_language),
        "french_gold_entries": len(french),
        "combined_entries": len(combined),
        "gold_units": len(formal_units),
        "gold_covered": len(set(formal_units) & set(combined["formal_token"].str.casefold())),
    }
    return combined, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wolof", type=Path, default=WOLOF_LOOKUP_SOURCE_PATH)
    parser.add_argument("--french", type=Path, default=FRENCH_WORDLIST_PATH)
    parser.add_argument("--gold-train", type=Path, default=GOLD_TRAIN_PATH)
    parser.add_argument("--names", type=Path, default=SENEGALESE_SURNAMES_PATH)
    parser.add_argument("--output", type=Path, default=FORMAL_TOKEN_LOOKUP_SEED_PATH)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite seed lookup: {args.output}")
    wolof = pd.read_csv(args.wolof, keep_default_na=False)
    lexique = pd.read_csv(
        args.french,
        sep="\t",
        usecols=["1_Mot", "4_Lemme", "5_Cgram"],
        keep_default_na=False,
        low_memory=False,
    )
    gold_train = pd.read_csv(args.gold_train, keep_default_na=False)
    names = args.names.read_text(encoding="utf-8-sig").splitlines()
    combined, stats = build_multilingual_seed(wolof, lexique, gold_train, names)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(
        f"Wrote {stats['combined_entries']:,} lookup entries: "
        f"{stats['name_entries']:,} Senegalese names + "
        f"{stats['ambiguous_language_entries']:,} Wolof/French ambiguities + "
        f"{stats['wolof_entries']:,} Wolof + "
        f"{stats['french_gold_entries']:,} French-only gold forms. "
        f"Gold-train coverage: {stats['gold_covered']}/{stats['gold_units']}."
    )


if __name__ == "__main__":
    main()
