import pandas as pd

from src.annotation.lookup_suggestions import LookupSuggestionIndex


def lookup(words):
    return pd.DataFrame(
        {
            "word": words,
            "pos": ["n." for _ in words],
            "translation": [f"definition {word}" for word in words],
        }
    )


def test_lookup_distance_does_not_prefer_equal_length_candidates():
    index = LookupSuggestionIndex(
        lookup(["abc", "abce"]),
        french_words=[],
        max_distance_ratio=0.26,
        min_candidate_margin=0.06,
    )

    decision = index.suggest("abcd")
    scores = {candidate.word: candidate.score for candidate in decision.candidates}

    assert scores["abc"] == scores["abce"] == 0.25
    assert decision.category == "ambiguous_wolof"
    assert decision.auto == "abcd"


def test_lookup_phonetic_contraction_can_change_token_length():
    index = LookupSuggestionIndex(
        lookup(["bañ", "xale"]),
        french_words=[],
        max_distance_ratio=0.10,
        min_candidate_margin=0.08,
    )

    decision = index.suggest("bagn")

    assert decision.category == "informal_wolof"
    assert decision.auto == "bañ"
    assert decision.candidates[0].score < 0.1
    assert decision.review_required is True


def test_lookup_keeps_exact_french_and_cross_language_ambiguity_for_review():
    index = LookupSuggestionIndex(
        lookup(["la", "xale"]),
        french_words=["la", "bonjour"],
    )

    ambiguous = index.suggest("la")
    french = index.suggest("bonjour")

    assert ambiguous.category == "ambiguous_exact"
    assert ambiguous.auto == "la"
    assert ambiguous.review_required is True
    assert french.category == "french_exact"
    assert french.auto == "bonjour"
    assert french.review_required is True


def test_lookup_rejects_a_near_french_typo_instead_of_forcing_wolof():
    index = LookupSuggestionIndex(
        lookup(["nuyul"]),
        french_words=["bonjour"],
        max_distance_ratio=0.26,
    )

    decision = index.suggest("bonjor")

    assert decision.category == "possible_french"
    assert decision.auto == "bonjor"
    assert decision.review_required is True


def test_lookup_rejects_candidates_above_threshold():
    index = LookupSuggestionIndex(
        lookup(["xale"]),
        french_words=[],
        max_distance_ratio=0.26,
    )

    decision = index.suggest("zzzzz")

    assert decision.category == "unresolved"
    assert decision.auto == "zzzzz"
    assert decision.accepted is False
