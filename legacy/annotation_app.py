import csv
import json
import pickle

import pandas as pd
import streamlit as st

from config import (
    CHECKPOINT_DATA_PATH as CHECKPOINT_PATH,
    CLEAN_COMMENTS_PATH,
    FILTERED_COMMENTS_PATH as DEFAULT_COMMENTS_PATH,
    FRENCH_WORDLIST_PATH,
    GOLD_ANNOTATIONS_PATH as GOLD_OUTPUT_PATH,
    LEXICON_PATH,
    TOKEN_CORRECTIONS_PATH as TOKEN_OUTPUT_PATH,
    TRAIN_PARQUET_PATH,
    WOLOF_VOCAB_PATH,
)

from src.normalization.reverse_code_switched import (
    VocabIndex,
    best_reverse_translation,
    build_french_vocab_from_corpus,
    build_reverse_lexicon,
    classify_token,
    find_formal_form,
    generate_candidates,
    mask_protected,
    weighted_distance,
    word_tokens,
)

GOLD_COLUMNS = [
    "source_index",
    "comment",
    "auto_suggestion",
    "manual_formal_wolof",
    "status",
    "video_url",
    "token_corrections_json",
]

TOKEN_COLUMNS = [
    "source_index",
    "informal_token",
    "formal_token",
    "category",
    "comment",
]


@st.cache_data(show_spinner=False)
def load_comments():
    if CHECKPOINT_PATH.exists() and CLEAN_COMMENTS_PATH.exists():
        clean_df = pd.read_csv(CLEAN_COMMENTS_PATH).reset_index()
        checkpoint_df = pd.read_csv(CHECKPOINT_PATH)
        merged = clean_df.merge(checkpoint_df, on="index", how="inner")
        merged = merged.loc[merged["is_informally_code_switched"] == True].copy()
        merged = merged.rename(columns={"index": "source_index"})
        return merged[["source_index", "clean_comment", "video_url"]]

    df = pd.read_csv(DEFAULT_COMMENTS_PATH).reset_index()
    df = df.rename(columns={"index": "source_index"})
    if "video_url" not in df.columns:
        df["video_url"] = ""
    return df[["source_index", "clean_comment", "video_url"]]


def load_existing_annotations():
    if not GOLD_OUTPUT_PATH.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(GOLD_OUTPUT_PATH)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    if "source_index" not in df.columns:
        return pd.DataFrame()
    return df


def load_learned_corrections():
    """Return previously saved corrections, with the newest equivalent first."""
    if not TOKEN_OUTPUT_PATH.exists():
        return {}
    try:
        corrections = pd.read_csv(TOKEN_OUTPUT_PATH)
    except (pd.errors.EmptyDataError, UnicodeDecodeError):
        return {}
    if not {"informal_token", "formal_token"}.issubset(corrections.columns):
        return {}

    learned = {}
    # Walking backwards makes the most recently annotated equivalent the default.
    for row in corrections.iloc[::-1].itertuples(index=False):
        informal = str(row.informal_token).strip()
        formal = str(row.formal_token).strip()
        if not informal or not formal or informal.lower() == "nan" or formal.lower() == "nan":
            continue
        equivalents = learned.setdefault(informal.casefold(), [])
        if formal not in equivalents:
            equivalents.append(formal)
    return learned


def next_unannotated_position(comments_df, annotated_ids, start_pos=0):
    if len(comments_df) == 0:
        return None

    for pos in range(start_pos, len(comments_df)):
        if int(comments_df.iloc[pos]["source_index"]) not in annotated_ids:
            return pos

    for pos in range(0, min(start_pos, len(comments_df))):
        if int(comments_df.iloc[pos]["source_index"]) not in annotated_ids:
            return pos

    return None


def next_batch_positions(comments_df, annotated_ids, start_pos, batch_size):
    """Collect up to `batch_size` distinct unannotated row positions, wrapping around."""
    n = len(comments_df)
    if n == 0:
        return []

    positions = []
    seen = set()
    pos = start_pos % n
    while len(positions) < batch_size and len(seen) < n:
        if pos not in seen:
            seen.add(pos)
            src_idx = int(comments_df.iloc[pos]["source_index"])
            if src_idx not in annotated_ids:
                positions.append(pos)
        pos = (pos + 1) % n
    return positions


@st.cache_resource(show_spinner=True)
def load_resources():
    with open(LEXICON_PATH, "rb") as f:
        lexicon = pickle.load(f)
    reverse_lexicon = build_reverse_lexicon(lexicon)

    with open(WOLOF_VOCAB_PATH, encoding="utf-8") as f:
        vocab_index = VocabIndex(f.read().splitlines())

    train_df = pd.read_parquet(TRAIN_PARQUET_PATH)
    french_vocab_corpus = build_french_vocab_from_corpus(train_df["french"])

    french_vocab_general = set()
    if FRENCH_WORDLIST_PATH.exists():
        lexique = pd.read_csv(FRENCH_WORDLIST_PATH, sep="\t")
        french_vocab_general = set(lexique["1_Mot"].str.lower().dropna())

    return vocab_index, french_vocab_corpus, french_vocab_general, reverse_lexicon


def nearest_vocab_candidates(token, vocab_index, limit=6):
    scored = {}
    generated = generate_candidates(token, max_candidates=40)
    generated.add(token)

    for candidate in generated:
        if candidate in vocab_index:
            scored[candidate] = min(scored.get(candidate, 999.0), weighted_distance(token, candidate))

        match, dist = vocab_index.find_best_match(candidate, max_ratio=0.30, length_window=1)
        if match is not None:
            score = dist + 0.05 * abs(len(match) - len(token))
            scored[match] = min(scored.get(match, 999.0), score)

    ranked = sorted(scored.items(), key=lambda item: (item[1], abs(len(item[0]) - len(token)), item[0]))
    return [word for word, _ in ranked[:limit]]


def token_suggestions(
    token,
    vocab_index,
    french_vocab_corpus,
    french_vocab_general,
    reverse_lexicon,
    learned_corrections=None,
):
    category = classify_token(token, vocab_index, french_vocab_corpus, french_vocab_general, reverse_lexicon)
    remembered = (learned_corrections or {}).get(token.casefold(), [])
    suggestions = [*remembered, token]

    if category == "french_substituted":
        translations = reverse_lexicon.get(token, {})
        suggestions.extend(
            word for word, _ in sorted(translations.items(), key=lambda item: item[1], reverse=True)
        )
        best = best_reverse_translation(token, reverse_lexicon)
        if best:
            suggestions.insert(1, best)

    elif category == "informal_wolof":
        best, _dist, _source = find_formal_form(token, vocab_index)
        if best and best != token:
            suggestions.append(best)
        suggestions.extend(nearest_vocab_candidates(token, vocab_index))

    elif category in {"wolof_formal", "ambiguous_cognate"}:
        suggestions.extend(nearest_vocab_candidates(token, vocab_index, limit=3))

    deduped = []
    for suggestion in suggestions:
        if suggestion and suggestion not in deduped:
            deduped.append(suggestion)
    return category, deduped[:8]


@st.cache_data(show_spinner=False)
def cached_token_suggestions(
    token,
    _vocab_index,
    _french_vocab_corpus,
    _french_vocab_general,
    _reverse_lexicon,
):
    """Cache expensive vocabulary-distance searches across different comments."""
    return token_suggestions(
        token,
        _vocab_index,
        _french_vocab_corpus,
        _french_vocab_general,
        _reverse_lexicon,
    )


@st.cache_data(show_spinner=False)
def analyze_comment(
    text,
    _vocab_index,
    _french_vocab_corpus,
    _french_vocab_general,
    _reverse_lexicon,
):
    """
    Analyze one comment and cache its stable, model-generated candidates.

    Streamlit reruns the script after widget changes. Candidate generation is the
    expensive part, so the resource arguments are prefixed with `_` to exclude
    them from Streamlit's hash and cache by comment text. User corrections are
    applied separately so saving one correction does not invalidate this cache.
    """
    masked, _saved = mask_protected(text, entities=[])
    tokens = word_tokens(masked)
    rows = []
    auto_tokens = []

    for token in tokens:
        category, suggestions = cached_token_suggestions(
            token,
            _vocab_index,
            _french_vocab_corpus,
            _french_vocab_general,
            _reverse_lexicon,
        )
        auto = (
            suggestions[1]
            if len(suggestions) > 1 and category in {"informal_wolof", "french_substituted"}
            else suggestions[0]
        )
        auto_tokens.append(auto)
        rows.append(
            {
                "token": token,
                "category": category,
                "suggestions": suggestions,
                "auto": auto,
            }
        )

    return " ".join(auto_tokens), rows


def apply_learned_corrections(token_rows, learned_corrections):
    """Overlay saved equivalents without rerunning expensive candidate generation."""
    updated_rows = []
    auto_tokens = []

    for row in token_rows:
        updated = dict(row)
        remembered = learned_corrections.get(row["token"].casefold(), [])
        if remembered:
            suggestions = []
            for suggestion in [*remembered, *row["suggestions"]]:
                if suggestion and suggestion not in suggestions:
                    suggestions.append(suggestion)
            updated["suggestions"] = suggestions[:8]
            updated["auto"] = remembered[0]

        auto_tokens.append(updated["auto"])
        updated_rows.append(updated)

    return " ".join(auto_tokens), updated_rows


def append_dict(path, row, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def save_annotation(source_index, comment, video_url, status, auto_sentence, manual_sentence, token_rows):
    token_payload = [
        {
            "token": row["token"],
            "category": row["category"],
            "selected_candidate": row.get("selected", ""),
            "final_correction": row.get("manual", "").strip(),
        }
        for row in token_rows
    ]

    append_dict(
        GOLD_OUTPUT_PATH,
        {
            "source_index": source_index,
            "comment": comment,
            "auto_suggestion": auto_sentence,
            "manual_formal_wolof": manual_sentence,
            "status": status,
            "video_url": video_url,
            "token_corrections_json": json.dumps(token_payload, ensure_ascii=False),
        },
        GOLD_COLUMNS,
    )

    if status != "keep":
        return

    for row in token_rows:
        correction = row.get("manual", "").strip()
        if not correction or correction == row["token"]:
            continue
        append_dict(
            TOKEN_OUTPUT_PATH,
            {
                "source_index": source_index,
                "informal_token": row["token"],
                "formal_token": correction,
                "category": row["category"],
                "comment": comment,
            },
            TOKEN_COLUMNS,
        )


def ensure_session_annotation_state():
    """Load annotation state from disk once, then keep it in session_state.

    Avoids re-reading the (ever-growing) gold_annotations.csv on every rerun.
    """
    if "learned_corrections" not in st.session_state:
        st.session_state.learned_corrections = load_learned_corrections()
    if "annotated_ids" in st.session_state:
        return

    existing = load_existing_annotations()
    if not existing.empty:
        existing = existing.copy()
        existing["source_index"] = existing["source_index"].astype(int)
        st.session_state.annotated_ids = set(existing["source_index"])
        st.session_state.latest_by_id = (
            existing.drop_duplicates("source_index", keep="last")
            .set_index("source_index")
            .to_dict("index")
        )
    else:
        st.session_state.annotated_ids = set()
        st.session_state.latest_by_id = {}


def record_saved(source_index, status, token_rows):
    st.session_state.annotated_ids.add(source_index)
    st.session_state.latest_by_id[source_index] = {"status": status}
    if status != "keep":
        return

    for row in token_rows:
        token = row["token"].strip()
        correction = row.get("manual", "").strip()
        if not correction or correction == token:
            continue
        equivalents = st.session_state.learned_corrections.setdefault(token.casefold(), [])
        if correction in equivalents:
            equivalents.remove(correction)
        equivalents.insert(0, correction)


def render_annotation_form(source_index, comment, video_url, auto_sentence, token_rows, form_key):
    """Render one comment's annotation UI inside an st.form.

    Everything (candidate picks, custom overrides, status, manual sentence)
    is batched client-side and only triggers a rerun on submit, instead of on
    every individual widget change.
    """
    with st.form(key=form_key, border=True):
        st.write(comment)
        if video_url and video_url != "nan":
            st.caption(video_url)

        st.caption("Reverse pipeline suggestion")
        st.write(auto_sentence)

        status = st.radio(
            "Annotation decision",
            ["keep", "discard_false_positive", "discard_uninteresting", "skip_uncertain"],
            horizontal=True,
            key=f"status_{form_key}",
        )

        manual_sentence = st.text_area(
            "Manual formal Wolof sentence",
            value=auto_sentence,
            height=80,
            key=f"manual_sentence_{form_key}",
        )

        st.caption(
            "Pick a suggested candidate, keep the original token, or type an override to the right."
        )

        edited_rows = []
        for i, row in enumerate(token_rows):
            cols = st.columns([1.1, 1.2, 3.0, 2.2])
            cols[0].markdown(f"`{row['token']}`")
            cols[1].caption(row["category"])
            selected = cols[2].selectbox(
                "Candidate",
                options=row["suggestions"],
                index=row["suggestions"].index(row["auto"]) if row["auto"] in row["suggestions"] else 0,
                key=f"candidate_{form_key}_{i}",
                label_visibility="collapsed",
            )
            custom = cols[3].text_input(
                "Override",
                value="",
                key=f"custom_{form_key}_{i}",
                placeholder="or type override",
                label_visibility="collapsed",
            )
            manual = custom.strip() or selected

            edited = dict(row)
            edited["selected"] = selected
            edited["manual"] = manual
            edited_rows.append(edited)

        submitted = st.form_submit_button("Save Annotation", type="primary")

    if submitted:
        save_annotation(
            source_index=source_index,
            comment=comment,
            video_url=video_url,
            status=status,
            auto_sentence=auto_sentence,
            manual_sentence=manual_sentence,
            token_rows=edited_rows,
        )
        record_saved(source_index, status, edited_rows)
        return True

    return False


st.set_page_config(page_title="Wolof Gold Annotation", layout="wide")
st.title("Wolof Gold Annotation")

comments_df = load_comments()
ensure_session_annotation_state()
annotated_ids = st.session_state.annotated_ids
latest_by_id = st.session_state.latest_by_id

resources = load_resources()
vocab_index, french_vocab_corpus, french_vocab_general, reverse_lexicon = resources

max_pos = len(comments_df) - 1

mode = st.sidebar.radio("Mode", ["Batch annotate", "Browse / edit single"])

remaining = len(comments_df) - len(annotated_ids)
st.sidebar.write(f"{len(comments_df):,} candidate comments")
st.sidebar.write(f"{len(annotated_ids):,} saved annotations")
st.sidebar.write(f"{max(remaining, 0):,} remaining")


if mode == "Batch annotate":
    batch_size = st.sidebar.slider("Rows per batch", min_value=1, max_value=10, value=5)

    if "batch_start" not in st.session_state:
        st.session_state.batch_start = 0

    if st.sidebar.button("Skip this batch"):
        st.session_state.batch_start = (st.session_state.batch_start + batch_size) % max(len(comments_df), 1)
        st.rerun()

    positions = next_batch_positions(comments_df, annotated_ids, st.session_state.batch_start, batch_size)

    if not positions:
        st.info("No unannotated comments left.")
    else:
        for pos in positions:
            current = comments_df.iloc[pos]
            source_index = int(current["source_index"])
            comment = str(current["clean_comment"])
            video_url = str(current.get("video_url", ""))

            _base_sentence, token_rows = analyze_comment(
                comment,
                vocab_index,
                french_vocab_corpus,
                french_vocab_general,
                reverse_lexicon,
            )
            auto_sentence, token_rows = apply_learned_corrections(
                token_rows, st.session_state.learned_corrections
            )

            st.markdown(f"##### Comment #{source_index}")
            saved = render_annotation_form(
                source_index=source_index,
                comment=comment,
                video_url=video_url,
                auto_sentence=auto_sentence,
                token_rows=token_rows,
                form_key=f"batch_{source_index}",
            )
            if saved:
                # Stop rendering the rest of the old batch and refill immediately.
                st.rerun()

else:
    if "row_pos" not in st.session_state:
        first_unannotated = next_unannotated_position(comments_df, annotated_ids, start_pos=0)
        st.session_state.row_pos = first_unannotated if first_unannotated is not None else 0

    if "pending_row_pos" in st.session_state:
        st.session_state.row_pos = int(st.session_state.pending_row_pos)
        st.session_state.row_pos_input = int(st.session_state.pending_row_pos)
        del st.session_state.pending_row_pos

    if st.sidebar.button("Resume Next Unannotated"):
        next_pos = next_unannotated_position(comments_df, annotated_ids, st.session_state.row_pos)
        if next_pos is not None:
            st.session_state.pending_row_pos = next_pos
            st.rerun()
        st.sidebar.info("No unannotated comments left.")

    st.sidebar.number_input(
        "Comment index",
        min_value=0,
        max_value=max_pos,
        value=int(st.session_state.row_pos),
        key="row_pos_input",
    )
    st.session_state.row_pos = int(st.session_state.row_pos_input)

    current = comments_df.iloc[st.session_state.row_pos]
    source_index = int(current["source_index"])
    comment = str(current["clean_comment"])
    video_url = str(current.get("video_url", ""))
    existing_row = latest_by_id.get(source_index)

    if existing_row:
        st.info(
            "This comment already has a saved annotation "
            f"with status `{existing_row.get('status', '')}`. "
            "Saving again will append a new row (kept as history)."
        )

    _base_sentence, token_rows = analyze_comment(
        comment,
        vocab_index,
        french_vocab_corpus,
        french_vocab_general,
        reverse_lexicon,
    )
    auto_sentence, token_rows = apply_learned_corrections(
        token_rows, st.session_state.learned_corrections
    )

    saved = render_annotation_form(
        source_index=source_index,
        comment=comment,
        video_url=video_url,
        auto_sentence=auto_sentence,
        token_rows=token_rows,
        form_key=f"single_{source_index}",
    )

    if saved:
        st.success(f"Saved annotation for source index {source_index}.")
        next_pos = next_unannotated_position(
            comments_df,
            st.session_state.annotated_ids,
            start_pos=min(st.session_state.row_pos + 1, max_pos),
        )
        if next_pos is not None:
            st.session_state.pending_row_pos = next_pos
            st.rerun()
        st.info("No unannotated comments left.")

st.caption(f"Outputs: {GOLD_OUTPUT_PATH} and {TOKEN_OUTPUT_PATH}")
