#!/usr/bin/env python3
"""
Wolof/French classifier using Ollama (gemma3:12b).

For each sentence in clean_comment, adds a boolean column `is_wolof`:
  True  = sentence is informal Wolof, code-switched Wolof/French, or both
  False = sentence is mostly or entirely French (not useful for Wolof NLP)

Strategy:
- Assign a temporary integer ID per row (0-based, matching DataFrame index)
- Send batches as a JSON array to the model
- Model returns a JSON array of {id, label} objects
- Validate every ID is accounted for; retry failed batches individually
- Checkpoint progress every N batches so the script is resumable
"""

import argparse
import logging
import json
import re
import shutil
import subprocess
import time

import pandas as pd
import requests
from tqdm import tqdm

from pathlib import Path
import sys
external_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(external_dir))

from config import (
    CHECKPOINTS_DIR,
    CLASSIFIED_COMMENTS_PATH,
    CLASSIFIER_LOG_PATH,
    CLEAN_COMMENTS_PATH,
)

# ── Configuration ──────────────────────────────────────────────────────────────
INPUT_CSV      = CLEAN_COMMENTS_PATH
OUTPUT_CSV     = CLASSIFIED_COMMENTS_PATH
CHECKPOINT_DIR = CHECKPOINTS_DIR
TEXT_COL       = "clean_comment"
GROUP_COL      = "video_url"
LABEL_COL      = "is_informal_code_switched"  # new column for the boolean label

OLLAMA_URL     = "http://localhost:11434/api/generate"
MODEL          = "gemma3:12b"

BATCH_SIZE     = 50          # sentences per LLM call
MAX_RETRIES    = 3           # retries for a whole batch
RETRY_DELAY    = 2           # seconds between retries
CHECKPOINT_EVERY = 5       # save progress every N batches
TIMEOUT        = 300         # seconds per HTTP request

# ── Logging ────────────────────────────────────────────────────────────────────
CLASSIFIER_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(CLASSIFIER_LOG_PATH),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ── Prompt ─────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a language classifier. You will receive a JSON array of objects, each with an "id" (integer) and a "text" (string).

Your task: for each text, decide if it is useful Wolof content.
- Return true  if the text is: informal Wolof, Wolof/French code-switching, or a mix of both.
- Return false if the text is: mostly or entirely French (i.e., it would NOT be useful for a Wolof language dataset).

Rules:
- Respond ONLY with a valid JSON array. No explanation, no markdown, no extra text.
- Each element must be exactly: {"id": <integer>, "label": <true|false>}
- You must return exactly one object per input id, in any order.
- Do not skip any id. Do not add extra ids.

Example input:
[{"id": 0, "text": "Bonjour, comment allez-vous?"}, {"id": 1, "text": "Maa ngi dem école bi"}]

Example output:
[{"id": 0, "label": false}, {"id": 1, "label": true}]"""


def build_prompt(batch: list[dict]) -> str:
    """Build the user prompt for a batch."""
    return SYSTEM_PROMPT + "\n\nInput:\n" + json.dumps(batch, ensure_ascii=False)


def parse_response(text: str, expected_ids: set[int]) -> dict[int, bool] | None:
    """
    Try to extract a valid list of {id, label} from the model response.
    Returns a dict {id: bool} or None if parsing/validation fails.
    """
    # Strip markdown code fences if present
    text = re.sub(r"```(?:json)?", "", text).strip()

    # Find the first [...] block
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        log.warning("No JSON array found in response.")
        return None

    try:
        data = json.loads(match.group())
    except json.JSONDecodeError as e:
        log.warning(f"JSON decode error: {e}")
        return None

    if not isinstance(data, list):
        log.warning("Parsed JSON is not a list.")
        return None

    result = {}
    for item in data:
        if not isinstance(item, dict):
            log.warning(f"Item is not a dict: {item}")
            return None
        if "id" not in item or "label" not in item:
            log.warning(f"Missing id or label in item: {item}")
            return None
        if not isinstance(item["id"], int):
            log.warning(f"id is not int: {item['id']}")
            return None
        if not isinstance(item["label"], bool):
            log.warning(f"label is not bool: {item['label']}")
            return None
        result[item["id"]] = item["label"]

    if set(result.keys()) != expected_ids:
        missing = expected_ids - set(result.keys())
        extra   = set(result.keys()) - expected_ids
        log.warning(f"ID mismatch. Missing: {missing}, Extra: {extra}")
        return None

    return result


def call_ollama(prompt: str) -> str | None:
    """Send a prompt to Ollama and return the raw response text."""
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0,       # deterministic
            "num_predict": 4096,    # enough for 50 JSON objects
        },
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=TIMEOUT)
        resp.raise_for_status()
        return resp.json().get("response", "")
    except requests.exceptions.RequestException as e:
        log.error(f"Ollama request failed: {e}")
        return None


def classify_batch(batch_rows: list[tuple[int, str]]) -> dict[int, bool] | None:
    """
    Classify a batch of (id, text) pairs.
    Returns {id: bool} or None if all retries exhausted.
    """
    batch_input = [{"id": idx, "text": text} for idx, text in batch_rows]
    expected_ids = {idx for idx, _ in batch_rows}
    prompt = build_prompt(batch_input)

    for attempt in range(1, MAX_RETRIES + 1):
        raw = call_ollama(prompt)
        if raw is None:
            log.warning(f"Attempt {attempt}/{MAX_RETRIES}: no response from Ollama.")
        else:
            result = parse_response(raw, expected_ids)
            if result is not None:
                return result
            log.warning(f"Attempt {attempt}/{MAX_RETRIES}: invalid response.")

        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY)

    return None


def classify_one_by_one(batch_rows: list[tuple[int, str]]) -> dict[int, bool]:
    """
    Fallback: classify each sentence individually when batch fails.
    Uses a simpler single-sentence prompt.
    """
    results = {}
    for idx, text in batch_rows:
        single_input = json.dumps([{"id": idx, "text": text}], ensure_ascii=False)
        prompt = SYSTEM_PROMPT + "\n\nInput:\n" + single_input
        label = None
        for attempt in range(1, MAX_RETRIES + 1):
            raw = call_ollama(prompt)
            if raw:
                parsed = parse_response(raw, {idx})
                if parsed is not None:
                    label = parsed[idx]
                    break
            time.sleep(RETRY_DELAY)

        if label is None:
            log.error(f"Row {idx}: failed after all retries. Defaulting to True.")
            label = True  # conservative default: keep the sentence

        results[idx] = label
    return results


def load_checkpoint() -> dict[int, bool]:
    """Load all saved checkpoint files and merge into one dict."""
    labels: dict[int, bool] = {}
    if not CHECKPOINT_DIR.exists():
        return labels
    for f in sorted(CHECKPOINT_DIR.glob("batch_*.json")):
        with open(f) as fh:
            chunk = json.load(fh)
            labels.update({int(k): v for k, v in chunk.items()})
    log.info(f"Loaded {len(labels)} labels from checkpoints.")
    return labels


def next_checkpoint_id() -> int:
    """Return an unused checkpoint number so a resumed run preserves history."""
    existing_ids = []
    for path in CHECKPOINT_DIR.glob("batch_*.json"):
        try:
            existing_ids.append(int(path.stem.rsplit("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return max(existing_ids, default=-1) + 1


def save_checkpoint(batch_idx: int, results: dict[int, bool]):
    """Save a batch result to a checkpoint file."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = CHECKPOINT_DIR / f"batch_{batch_idx:07d}.json"
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint: {path}")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({str(k): v for k, v in results.items()}, fh, ensure_ascii=False)


def rows_in_diverse_video_order(
    dataframe: pd.DataFrame, already_done: set[int]
) -> list[tuple[int, str]]:
    """Interleave pending rows across videos instead of exhausting one video first."""
    pending = dataframe.loc[~dataframe.index.isin(already_done)].copy()
    if pending.empty:
        return []

    pending["_source_index"] = pending.index.astype(int)
    if GROUP_COL in pending.columns:
        missing_group = pending[GROUP_COL].isna() | pending[GROUP_COL].eq("")
        pending["_group"] = pending[GROUP_COL].astype(str)
        pending.loc[missing_group, "_group"] = pending.loc[
            missing_group, "_source_index"
        ].map(lambda value: f"missing-video-{value}")
        pending["_group_order"] = pd.factorize(pending["_group"], sort=False)[0]
        pending["_within_group"] = pending.groupby("_group", sort=False).cumcount()
        pending = pending.sort_values(
            ["_within_group", "_group_order", "_source_index"], kind="stable"
        )

    rows = []
    for row in pending.to_dict("records"):
        text = row.get(TEXT_COL, "")
        rows.append(
            (row["_source_index"], str(text) if pd.notna(text) else "")
        )
    return rows


def wait_for_ollama(max_wait: int = 30):
    """
    Check that Ollama is reachable and the required model is pulled.
    Exits if it isn't ready after max_wait seconds.
    """
    base_url = OLLAMA_URL.rsplit("/api/", 1)[0]
    tags_url = f"{base_url}/api/tags"
    
    log.info(f"Checking Ollama status and model availability...")
    deadline = time.time() + max_wait
    
    while time.time() < deadline:
        try:
            # 1. Check if Ollama service is responsive
            r = requests.get(tags_url, timeout=5)
            if r.status_code == 200:
                models_data = r.json()
                # 2. Verify if the target model is pulled
                available_models = [m["name"] for m in models_data.get("models", [])]
                
                # Ollama model names can sometimes match with or without the ':latest' tag
                if MODEL in available_models or f"{MODEL}:latest" in available_models:
                    log.info(f"Ollama is up and model '{MODEL}' is ready!")
                    return
                else:
                    log.error(
                        f"\n{'='*60}\n"
                        f"Ollama is running, but model '{MODEL}' is NOT pulled.\n"
                        f"Please run this in your terminal:\n"
                        f"    ollama pull {MODEL}\n"
                        f"{'='*60}"
                    )
                    raise SystemExit(1)
        except requests.exceptions.RequestException:
            # Try to start it automatically if 'ollama' executable exists on the system
            if shutil.which("ollama"):
                try:
                    # Launches ollama serve as a background process redirecting output
                    subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    log.info("Attempted to auto-start Ollama background service...")
                except Exception as e:
                    log.debug(f"Failed to auto-start subprocess: {e}")

        log.info("Waiting for Ollama service to respond... (retry in 5 s)")
        time.sleep(5)
        
    log.error(
        f"\n{'='*60}\n"
        "Ollama is NOT running or not reachable.\n"
        "Start it with:   ollama serve\n"
        f"{'='*60}"
    )
    raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Classify cleaned comments with resumable Ollama checkpoints."
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        help="Classify at most this many pending rows in the current run.",
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    if args.max_rows is not None and args.max_rows <= 0:
        parser.error("--max-rows must be positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")

    wait_for_ollama()
    log.info(f"Loading {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)

    if TEXT_COL not in df.columns:
        raise ValueError(f"Column '{TEXT_COL}' not found. Available: {list(df.columns)}")

    log.info(f"Loaded {len(df):,} rows.")

    # Load existing checkpoints (supports resuming)
    all_labels = load_checkpoint()
    already_done = set(all_labels.keys())

    # Build list of (row_index, text) still to classify
    rows_to_classify = rows_in_diverse_video_order(df, already_done)
    if args.max_rows is not None:
        rows_to_classify = rows_to_classify[: args.max_rows]

    log.info(f"Rows to classify: {len(rows_to_classify):,} (already done: {len(already_done):,})")

    # Split into batches
    batches = [
        rows_to_classify[i : i + args.batch_size]
        for i in range(0, len(rows_to_classify), args.batch_size)
    ]

    log.info(f"Batches: {len(batches):,} × up to {args.batch_size} rows each.")

    pending_save: dict[int, bool] = {}
    checkpoint_id = next_checkpoint_id()

    for batch_num, batch in enumerate(tqdm(batches, desc="Classifying")):
        result = classify_batch(batch)

        if result is None:
            log.warning(f"Batch {batch_num} failed. Falling back to one-by-one.")
            result = classify_one_by_one(batch)

        all_labels.update(result)
        pending_save.update(result)

        # Checkpoint periodically
        if (batch_num + 1) % CHECKPOINT_EVERY == 0 or batch_num == len(batches) - 1:
            save_checkpoint(checkpoint_id, pending_save)
            checkpoint_id += 1
            pending_save = {}
            log.info(f"Checkpoint saved at batch {batch_num + 1}.")

    # Save any remaining pending labels
    if pending_save:
        save_checkpoint(checkpoint_id, pending_save)

    log.info("Classification complete. Writing output...")

    df[LABEL_COL] = pd.Series(
        df.index.map(all_labels), index=df.index, dtype="boolean"
    )

    df.to_csv(OUTPUT_CSV, index=False)
    log.info(f"Saved to {OUTPUT_CSV}")

    # Summary
    n_labeled = int(df[LABEL_COL].notna().sum())
    n_true = int(df[LABEL_COL].sum(skipna=True))
    n_false = n_labeled - n_true
    n_pending = len(df) - n_labeled
    denominator = n_labeled or 1
    log.info(f"is_wolof=True  (kept):    {n_true:,} ({100*n_true/denominator:.1f}%)")
    log.info(f"is_wolof=False (filtered): {n_false:,} ({100*n_false/denominator:.1f}%)")
    log.info(f"Unclassified rows:          {n_pending:,}")


if __name__ == "__main__":
    main()
