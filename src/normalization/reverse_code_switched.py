"""
REVERSE PIPELINE: informal code-switched Wolof/French -> formal Wolof

This is the inverse of the forward pipeline that:
  1. masked entities/numbers
  2. substituted ~20% of Wolof tokens with French translations
     (constrained to words present in the paired French sentence)
  3. informalized remaining Wolof tokens via regex normalization rules

Given ONLY informal code-switched text (no paired French sentence, unseen data),
we:
  1. mask entities/numbers (NER)
  2. classify each remaining token as:
       - formal Wolof          -> keep
       - French (from lexicon substitution) -> revert via reverse_lexicon
       - French (native/unmappable)         -> keep as-is
       - ambiguous cognate                  -> keep (default to Wolof)
       - informal Wolof                     -> reverse normalization rules
                                                + weighted-distance match
                                                against formal Wolof vocab
  3. restore protected spans

Artifacts needed (built once, offline, reused across unseen data):
  - lexicon            : wolof -> {french_word: prob}   (from build_lexicon on training corpus)
  - reverse_lexicon    : french -> {wolof_word: prob}   (derived from lexicon)
  - wolof_vocab_set    : set of formal Wolof words      (your vocab file, one word per line)
  - french_vocab_corpus: set of French words seen in training corpus's french column
  - french_vocab_general: large general-purpose French wordlist (Lexique383 / hunspell / wiktionary)
"""

import re
import math
from collections import defaultdict, Counter


# =========================================================================
# 1. TOKENIZATION (same as forward pipeline)
# =========================================================================

def tokenize(text):
    return re.findall(r"\w+(?:[-'']\w+)*|[^\w\s]", str(text), flags=re.UNICODE)


def word_tokens(text):
    return re.findall(r"\w+(?:[-'']\w+)*", str(text).lower(), flags=re.UNICODE)


# =========================================================================
# 2. ENTITY / NUMBER PROTECTION (same as forward pipeline)
# =========================================================================

def mask_protected(text, entities=None):
    spans = []
    if entities:
        for ent in entities:
            spans.append((ent["start"], ent["end"]))
    for m in re.finditer(r"\b[A-ZÀ-Ý0-9]{2,}\b", text):
        spans.append((m.start(), m.end()))
    for m in re.finditer(r"\b\d+(?:[.,]\d+)?\b", text):
        spans.append((m.start(), m.end()))
    spans = sorted(spans)
    merged = []
    for s, e in spans:
        if not merged or s > merged[-1][1]:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)
    saved = {}
    masked = text
    for i, (s, e) in enumerate(reversed(merged)):
        token = f"__protected_{i}__"
        saved[token] = masked[s:e]
        masked = masked[:s] + token + masked[e:]
    return masked, saved


def restore_protected(text, saved):
    for token, value in saved.items():
        text = text.replace(token, value)
    return text


# =========================================================================
# 3. REVERSE LEXICON (derived from forward `lexicon`)
# =========================================================================

def build_reverse_lexicon(lexicon):
    """
    lexicon: wolof_word -> {french_word: prob}
    returns: french_word -> {wolof_word: prob}
    """
    reverse_lexicon = defaultdict(dict)
    for wo_word, fr_dict in lexicon.items():
        for fr_word, prob in fr_dict.items():
            # keep the highest prob if multiple wolof words map to same french word
            if fr_word not in reverse_lexicon or prob > max(reverse_lexicon[fr_word].values()):
                reverse_lexicon[fr_word][wo_word] = prob
            else:
                reverse_lexicon[fr_word][wo_word] = prob
    return dict(reverse_lexicon)


def best_reverse_translation(fr_word, reverse_lexicon):
    candidates = reverse_lexicon.get(fr_word)
    if not candidates:
        return None
    return max(candidates, key=candidates.get)


# =========================================================================
# 4. EQUIVALENCE-AWARE WEIGHTED EDIT DISTANCE
#    (encodes the informalization rules as low-cost substitutions)
# =========================================================================

# Pairs of (char_or_short_string, char_or_short_string) that the
# normalize_wolof_token rules treat as roughly equivalent.
# Substituting between members of a pair costs less than a generic substitution.
EQUIV_PAIRS = {
    ("a", "à"), ("a", "á"), ("a", "â"), ("a", "ä"),
    ("e", "é"), ("e", "è"), ("e", "ê"), ("e", "ë"),
    ("o", "ó"),
    ("u", "ou"),
    ("n", "ñ"), ("n", "gn"), ("n", "ng"), ("n", "ŋ"),
    ("c", "th"), ("c", "thi"),
    ("j", "dj"), ("j", "di"), ("j", "dio"),
    ("x", "kh"), ("q", "kh"),
    ("a", "aa"), ("e", "ee"), ("i", "ii"), ("o", "oo"),  # vowel doubling collapse
    ("u", "uu"),
}

# normalize pair lookups both directions
_EQUIV_SET = set()
for a, b in EQUIV_PAIRS:
    _EQUIV_SET.add((a, b))
    _EQUIV_SET.add((b, a))


def weighted_distance(a, b, equiv_cost=0.3, sub_cost=1.0, indel_cost=1.0):
    """
    Levenshtein-style edit distance where substitutions between
    characters in EQUIV_PAIRS cost `equiv_cost` instead of `sub_cost`.

    Only single-character vs single-character equivalences are checked
    in the DP cell-by-cell step (multi-char equivalences like 'th'<->'c'
    are partially captured because after one cheap substitution the
    remaining characters align with low extra cost, e.g. 'th'->'c' is
    one equiv-cost sub + one deletion).
    """
    m, n = len(a), len(b)
    if m == 0:
        return n
    if n == 0:
        return m

    dp = [[0.0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i * indel_cost
    for j in range(n + 1):
        dp[0][j] = j * indel_cost

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            ca, cb = a[i - 1], b[j - 1]
            if ca == cb:
                cost = 0.0
            elif (ca, cb) in _EQUIV_SET:
                cost = equiv_cost
            else:
                cost = sub_cost
            dp[i][j] = min(
                dp[i - 1][j] + indel_cost,      # deletion
                dp[i][j - 1] + indel_cost,      # insertion
                dp[i - 1][j - 1] + cost          # substitution / match
            )
    return dp[m][n]


# =========================================================================
# 5. REVERSE NORMALIZATION RULES (candidate generation)
#    Generate plausible "formal" candidates from an informal token
#    by inverting the forward `rules` list. Many forward rules are
#    many-to-one, so we generate MULTIPLE candidates per inversion
#    rather than a single deterministic string.
# =========================================================================

# Each entry: (pattern_on_informal_token, list_of_possible_replacements)
# Applied as alternatives -> branching candidate set.
REVERSE_RULES = [
    # 'ou' may have come from 'u' (most common) or stayed 'ou'
    (r"ou", ["u", "ou"]),
    # 'eu' may have come from 'ë'
    (r"eu", ["ë", "eu"]),
    # trailing 'é' may have come from 'e'
    (r"é\b", ["e", "é"]),
    # trailing 'ne' may have come from 'n'
    (r"ne\b", ["n", "ne"]),
    # 'gn' may have come from 'ñ'
    (r"gn", ["ñ", "gn"]),
    # 'ng' may have come from 'ŋ'
    (r"ng", ["ŋ", "ng"]),
    # 'thi'/'th' may have come from 'c'
    (r"thi", ["ci", "thi"]),
    (r"th", ["c", "th"]),
    # 'kh' may have come from 'x' or 'q'
    (r"kh", ["x", "q", "kh"]),
    # 'dj'/'di'/'dio' may have come from 'j'
    (r"dio", ["ju", "dio"]),
    (r"dj", ["ji", "dj"]),
    (r"di", ["je", "ja", "dë", "di"]),
    # accented vowels possibly simplified from doubled/accented forms
    (r"a", ["a", "à", "â", "aa"]),
    (r"e", ["e", "é", "ée", "ee"]),
    (r"o", ["o", "ó", "oo"]),
    (r"g", ["g", "gg"]),
]


def generate_candidates(token, max_candidates=60):
    """
    Generate a bounded set of candidate "formal-ish" spellings for an
    informal token by applying REVERSE_RULES combinatorially (branching
    on alternatives). Limited to keep the candidate set manageable.
    """
    candidates = {token}

    for pattern, replacements in REVERSE_RULES:
        new_candidates = set()
        for cand in candidates:
            matches = list(re.finditer(pattern, cand))
            if not matches:
                new_candidates.add(cand)
                continue
            # apply replacement only at the first match site per branch
            # (applying at every site combinatorially explodes too fast)
            m = matches[0]
            for repl in replacements:
                new_cand = cand[:m.start()] + repl + cand[m.end():]
                new_candidates.add(new_cand)
            new_candidates.add(cand)  # also keep unchanged version
        candidates = new_candidates
        if len(candidates) > max_candidates:
            # trim to avoid explosion; keep shortest + most diverse-ish
            candidates = set(sorted(candidates, key=len)[:max_candidates])

    return candidates


# =========================================================================
# 6. FORMAL WOLOF VOCAB LOOKUP (with length-bucketed index for speed)
# =========================================================================

class VocabIndex:
    """
    Wraps a formal Wolof vocabulary set with a length-bucket index
    so distance search only scans plausibly-similar-length words.
    """

    def __init__(self, vocab_words):
        self.vocab_set = set(w.strip().lower() for w in vocab_words if w.strip())
        self.by_length = defaultdict(list)
        for w in self.vocab_set:
            self.by_length[len(w)].append(w)

    def __contains__(self, word):
        return word in self.vocab_set

    def candidates_near_length(self, length, window=2):
        out = []
        for l in range(max(1, length - window), length + window + 1):
            out.extend(self.by_length.get(l, []))
        return out

    def find_best_match(self, token, max_ratio=0.34, length_window=2):
        """
        Find the closest vocab word to `token` using weighted_distance,
        restricted to words within `length_window` of token's length.
        Returns (best_word, distance) or (None, None) if nothing close enough.
        max_ratio: max allowed distance as a fraction of token length.
        """
        if token in self.vocab_set:
            return token, 0.0

        pool = self.candidates_near_length(len(token), window=length_window)
        if not pool:
            pool = list(self.vocab_set)  # fallback, slow but rare

        best_word, best_dist = None, math.inf
        for w in pool:
            d = weighted_distance(token, w)
            if d < best_dist:
                best_word, best_dist = w, d

        threshold = max(1.0, len(token) * max_ratio)
        if best_dist <= threshold:
            return best_word, best_dist
        return None, None


def find_formal_form(informal_tok, vocab_index):
    """
    Try reverse-rule candidates first (cheap, precise); if any candidate
    is directly in the vocab, use it. Otherwise fall back to weighted
    distance search over the full vocab for the original token, and
    also for each generated candidate, taking the overall best match.
    """
    if informal_tok in vocab_index:
        return informal_tok, 0.0, "exact"

    candidates = generate_candidates(informal_tok)

    # Prefer rule-generated forms even if original token is in vocab
    direct_hits = [c for c in candidates if c in vocab_index and c != informal_tok]

    if direct_hits:
        best = min(
            direct_hits,
            key=lambda c: (
                weighted_distance(informal_tok, c),
                abs(len(c) - len(informal_tok)),
                c
            )
        )
        return best, 0.0, "rule_exact"

    # Only now keep exact token
    if informal_tok in vocab_index:
        return informal_tok, 0.0, "exact"

    # Then fuzzy search
    best_word, best_dist, best_source = None, math.inf, None

    for cand in candidates:
        w, d = vocab_index.find_best_match(
            cand,
            max_ratio=0.25,
            length_window=1
        )
        if w is not None and (
            d < best_dist or
            (d == best_dist and best_word is not None and abs(len(w) - len(informal_tok)) < abs(len(best_word) - len(informal_tok)))
        ):
            best_word, best_dist, best_source = w, d, cand

    if best_word is not None:
        return best_word, best_dist, f"distance_after_rules(via='{best_source}')"

    return informal_tok, None, "unmatched"


# =========================================================================
# 7. FRENCH DETECTION
# =========================================================================

def classify_token(tok, vocab_index, french_vocab_corpus, french_vocab_general, reverse_lexicon):
    """
    Decide what category a (lowercased, already-tokenized) token belongs to.

    Returns one of:
      "wolof_formal"       - already in formal Wolof vocab, keep as-is
      "french_substituted" - French token traceable to reverse_lexicon, revert
      "french_native"      - French token with no known Wolof source, keep as-is
      "ambiguous_cognate"  - in both Wolof vocab and French vocab, keep (Wolof default)
      "informal_wolof"     - none of the above, needs rule-reversal + distance match
    """
    in_wo = tok in vocab_index
    in_fr_corpus = tok in french_vocab_corpus
    in_fr_general = tok in french_vocab_general
    in_fr = in_fr_corpus or in_fr_general
    in_rev_lex = tok in reverse_lexicon

    if in_wo and not in_fr:
        return "wolof_formal"

    if in_fr and not in_wo:
        if in_rev_lex:
            return "french_substituted"
        return "french_native"

    if in_wo and in_fr:
        return "ambiguous_cognate"

    return "informal_wolof"


# =========================================================================
# 8. FULL REVERSE PIPELINE FOR ONE SENTENCE
# =========================================================================

def reverse_process_sentence(text, entities, vocab_index,
                               french_vocab_corpus, french_vocab_general,
                               reverse_lexicon, debug=False):
    """
    text     : informal code-switched sentence (string)
    entities : list of NER entity dicts with "start"/"end" (or None / [])
    vocab_index           : VocabIndex over formal Wolof vocab
    french_vocab_corpus   : set of French words seen in your training corpus
    french_vocab_general  : set of general French wordlist words
    reverse_lexicon       : french_word -> {wolof_word: prob}
    debug    : if True, return per-token classification info as well
    """
    if not isinstance(text, str) or not text.strip():
        return text if not debug else (text, [])

    # Step 1: protect entities / numbers / acronyms
    masked, saved = mask_protected(text, entities)

    # Step 2: tokenize (lowercased word tokens, same regime as forward pipeline)
    tokens = word_tokens(masked)

    output_tokens = []
    debug_info = []

    for tok in tokens:
        if re.fullmatch(r"__protected_\d+__", tok, flags=re.IGNORECASE):
            output_tokens.append(tok)
            if debug:
                debug_info.append((tok, "protected", tok, None))
            continue

        category = classify_token(
            tok, vocab_index, french_vocab_corpus, french_vocab_general, reverse_lexicon
        )

        if category == "wolof_formal":
            out_tok = tok

        elif category == "french_substituted":
            wo = best_reverse_translation(tok, reverse_lexicon)
            out_tok = wo if wo else tok

        elif category == "french_native":
            out_tok = tok  # leave genuine French code-switch as-is

        elif category == "ambiguous_cognate":
            out_tok = tok  # default: treat as Wolof, keep as-is

        else:  # informal_wolof
            best_word, dist, source = find_formal_form(tok, vocab_index)
            out_tok = best_word

        output_tokens.append(out_tok)
        if debug:
            debug_info.append((tok, category, out_tok, None))

    result = " ".join(output_tokens)

    # Step 3: restore protected spans
    result = restore_protected(result, saved)
    result = re.sub(r"\s+", " ", result).strip()

    if debug:
        return result, debug_info
    return result


# =========================================================================
# 9. BATCH DRIVER
# =========================================================================

def load_wordlist(path):   
    lex = pd.read_csv(path, sep="\t")
    return set(lex["1_Mot"].str.lower().dropna())
     


def build_french_vocab_from_corpus(french_series):
    vocab = set()
    for sent in french_series:
        vocab.update(word_tokens(str(sent)))
    return vocab


if __name__ == "__main__":
    import logging
    import pickle
    import warnings
    warnings.filterwarnings("ignore")

    import pandas as pd
    import torch
    from tqdm.auto import tqdm
    from transformers import pipeline as hf_pipeline

    from config import (
        FRENCH_WORDLIST_PATH,
        LEXICON_PATH,
        REVERSE_TEST_INPUT_PATH as INPUT_CSV_PATH,
        REVERSE_TEST_OUTPUT_PATH as OUTPUT_CSV_PATH,
        TRAIN_PARQUET_PATH,
        WOLOF_VOCAB_PATH,
    )

    # ================================================================
    # CONFIG: fixed variables (filesystem paths come from config.py)
    # ================================================================
    INPUT_TEXT_COLUMN = "comment"
    OUTPUT_COLUMN = "recovered_formal_wolof"

    NER_MODEL = "davlan/xlm-roberta-large-ner-hrl"
    NER_BATCH_SIZE = 32

    LOG_LEVEL = logging.INFO

    # ================================================================
    # LOGGING
    # ================================================================
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    logger.info("Starting reverse pipeline")

    # ================================================================
    # LOAD ARTIFACTS
    # ================================================================
    logger.info("Loading training corpus from %s", TRAIN_PARQUET_PATH)
    train_df = pd.read_parquet(TRAIN_PARQUET_PATH)

    logger.info("Loading lexicon from %s", LEXICON_PATH)
    with open(LEXICON_PATH, "rb") as f:
        lexicon = pickle.load(f)

    reverse_lexicon = build_reverse_lexicon(lexicon)
    logger.info("Built reverse lexicon with %d French entries", len(reverse_lexicon))

    logger.info("Loading formal Wolof vocab from %s", WOLOF_VOCAB_PATH)
    with open(WOLOF_VOCAB_PATH, encoding="utf-8") as f:
        vocab_index = VocabIndex(f.read().splitlines())


    logger.info("Building French corpus vocab")
    french_vocab_corpus = build_french_vocab_from_corpus(
        tqdm(train_df["french"], desc="Building French corpus vocab")
    )

    logger.info("Loading general French wordlist from %s", FRENCH_WORDLIST_PATH)
    french_vocab_general = load_wordlist(FRENCH_WORDLIST_PATH)

    # ================================================================
    # LOAD NER MODEL
    # ================================================================
    device = 0 if torch.cuda.is_available() else -1
    logger.info("Loading NER model: %s | device=%s", NER_MODEL, device)

    ner_pipeline = hf_pipeline(
        "token-classification",
        model=NER_MODEL,
        aggregation_strategy="simple",
        device=device,
    )

    # ================================================================
    # LOAD INPUT DATA
    # ================================================================
    logger.info("Loading input data from %s", INPUT_CSV_PATH)
    new_df = pd.read_csv(INPUT_CSV_PATH,sep = ';')

    if INPUT_TEXT_COLUMN not in new_df.columns:
        raise ValueError(
            f"Input CSV must contain column '{INPUT_TEXT_COLUMN}'. "
            f"Found columns: {list(new_df.columns)}"
        )

    texts = [
        str(text) if pd.notna(text) else ""
        for text in new_df[INPUT_TEXT_COLUMN]
    ]

    logger.info("Loaded %d rows", len(texts))

    # ================================================================
    # NER ENTITY EXTRACTION
    # ================================================================
    logger.info("Extracting entities")

    all_entities = []
    for entities in tqdm(
        ner_pipeline(texts, batch_size=NER_BATCH_SIZE),
        total=len(texts),
        desc="NER entity extraction",
    ):
        all_entities.append(entities)

    # ================================================================
    # REVERSE CODE-SWITCHING
    # ================================================================
    logger.info("Recovering formal Wolof")

    results = []
    for text, entities in tqdm(
        zip(texts, all_entities),
        total=len(texts),
        desc="Reverse processing",
    ):
        out = reverse_process_sentence(
            text=text,
            entities=entities,
            vocab_index=vocab_index,
            french_vocab_corpus=french_vocab_corpus,
            french_vocab_general=french_vocab_general,
            reverse_lexicon=reverse_lexicon,
        )
        results.append(out)

    new_df[OUTPUT_COLUMN] = results

    # ================================================================
    # SAVE OUTPUT
    # ================================================================
    logger.info("Saving recovered data to %s", OUTPUT_CSV_PATH)
    OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    new_df.to_csv(OUTPUT_CSV_PATH, index=False)

    logger.info("Done")
    print(new_df[[INPUT_TEXT_COLUMN, OUTPUT_COLUMN]].head(10).to_string())
