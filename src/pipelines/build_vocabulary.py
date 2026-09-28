import pandas as pd
import os
import re
from pathlib import Path
import sys

external_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(external_dir))

from config import (
    DICTIONARY_TRANSLATIONS_PATH,
    TRAIN_PARQUET_PATH,
    WOLOF_TRANSLATIONS_PATH,
    WOLOF_VOCAB_PATH,
)


OUTPUT_PATH = WOLOF_VOCAB_PATH

def load_dataframe(file_path):
    """Invokes the correct pandas read function based on the file extension."""
    # Extract the extension and convert to lowercase
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    # Mapping extensions to their respective pandas read functions
    reader_mapping = {
        ".csv": pd.read_csv,
        ".json": pd.read_json,
        ".parquet": pd.read_parquet,
        ".txt": pd.read_csv,  # Typically read as CSV (often requires sep='\t')
    }
    # Check if the extension is supported
    if ext in reader_mapping:
        try:
            return reader_mapping[ext](file_path)
        except Exception as e:
            raise RuntimeError(f"Error reading file '{file_path}' with {reader_mapping[ext].__name__}: {e}")
    else:
        raise ValueError(f"Unsupported file extension: '{ext}'. \nSupported extensions are: {', '.join(reader_mapping.keys())}")

def find_vocab_column(df, possible_names):
    """Finds the vocab column that matches any of the possible names."""
    for col in df.columns:
        if  col in possible_names:
            return col
    return None


def extract_words_from_text(text_series):
    """Splits sentences into individual, clean words."""
    words_set = set()

    for text in text_series.dropna().astype(str):
        # 1. Convert to lowercase
        text = text.lower()

        # 2. Use Regex to find all words (ignores punctuation like periods, commas, etc.)
        # \b\w+\b matches any alphanumeric word boundary
        words = re.findall(r"\b\w+\b", text)

        # 3. Add the discovered words to our set
        words_set.update(words)

    return words_set

def build_vocabulary_file(dataframes, output_path=OUTPUT_PATH):
    output_path = Path(output_path)
    # Using a set automatically ensures all words are unique
    vocabulary = set()

    # Keywords to look for in column names
    target_keywords = ["wolof"]

    for i, df in enumerate(dataframes):
        # Find the column name for the current dataframe
        col_name = find_vocab_column(df, target_keywords)

        if col_name:
            print(f"DataFrame {i}: Found vocabulary in column '{col_name}'")

            # Drop missing values, convert to string, strip whitespace
            words = extract_words_from_text(df[col_name])

            # Add words to our master set
            vocabulary.update(words)
        else:
            print(
                f"⚠️ Warning: DataFrame {i} skipped. No matching vocabulary column found."
            )

    # Remove empty strings if any exist
    vocabulary.discard("")
    vocabulary = {word for word in vocabulary if not word.isdigit()}

    # Sort the vocabulary alphabetically and write to a txt file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for word in sorted(vocabulary):
            f.write(f"{word}\n")

    print(
        f"\n Success! Saved {len(vocabulary)} unique words to '{output_path}'"
    )



if __name__ == "__main__":
    # Building vocabulary from dataset files
    dataset_paths = [
        TRAIN_PARQUET_PATH,
        WOLOF_TRANSLATIONS_PATH,
        DICTIONARY_TRANSLATIONS_PATH,
    ]

    list_of_dfs = [load_dataframe(file_path) for file_path in dataset_paths]

    # Run the function
    build_vocabulary_file(list_of_dfs)
