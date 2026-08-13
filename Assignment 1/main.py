"""
NLP Assignment 1 – Custom Preprocessing Pipeline (AI357)

Course: AI357 – Natural Language Processing
Institute: SVNIT Surat

Outputs untagged space-separated tokens and exports stats to a JSON file.
"""

import json
import os
import re
from typing import List
import pandas as pd


# =============================================================================
# 1. REGEX PATTERNS
# =============================================================================

# URLs, emails, dates, and numbers
URL_PATTERN = r'(?:https?://|www\.)\S+'
EMAIL_PATTERN = r'[\w.\-+]+@[\w\-]+\.[\w.\-]+'
DATE_PATTERN = r'[\d\u0AE6-\u0AEF]{1,4}[/\-.][\d\u0AE6-\u0AEF]{1,2}[/\-.][\d\u0AE6-\u0AEF]{1,4}'
NUMBER_PATTERN = r'[\d\u0AE6-\u0AEF]+(?:,[\d\u0AE6-\u0AEF]+)*(?:\.[\d\u0AE6-\u0AEF]+)?'

# Abbreviations
ABBREVIATIONS = r'(?:[A-Za-z]\.){2,}|(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|eg|ie)\.|(?:તા|ડૉ|પ્રો|સ્વ|જિ|લિ)\.'

# Word scripts
LATIN_WORD_PATTERN = r"[A-Za-z]+(?:'[A-Za-z]+)*"
DEVANAGARI_WORD_PATTERN = r'[\u0900-\u0963\u0966-\u097F]+'
GUJARATI_WORD_PATTERN = r'[\u0A80-\u0AFF]+'
GENERAL_INDIC_PATTERN = r'[\u0980-\u0D7F]+'

# Punctuations
INDIC_RANGES = r'\u0900-\u0963\u0966-\u097F\u0A80-\u0AFF\u0980-\u0D7F'
PUNCT_PATTERN = r'[^\sA-Za-z0-9' + INDIC_RANGES + r']'

# Masking pattern for internal dots
PROTECT_REGEX = re.compile(
    f'{URL_PATTERN}|{EMAIL_PATTERN}|{DATE_PATTERN}|{NUMBER_PATTERN}|{ABBREVIATIONS}'
)

# Sentence boundaries
SENTENCE_BOUNDARY_REGEX = re.compile(
    rf'(?<=[.!?\u0964\u0965])["\')\]]?\s+(?=[A-Z"\'' + INDIC_RANGES + r']|$)'
)

# Master token extraction pattern
TOKEN_REGEX = re.compile(
    rf"""
    {URL_PATTERN}
    |{EMAIL_PATTERN}
    |{DATE_PATTERN}
    |{NUMBER_PATTERN}
    |{LATIN_WORD_PATTERN}
    |{DEVANAGARI_WORD_PATTERN}
    |{GUJARATI_WORD_PATTERN}
    |{GENERAL_INDIC_PATTERN}
    |{PUNCT_PATTERN}
    """,
    re.VERBOSE,
)


# =============================================================================
# 2. TOKENIZATION FUNCTIONS
# =============================================================================

def sentence_tokenize(text: str) -> List[str]:
    """Splits text into sentences using masking."""
    protected_spans = []

    def _mask(match: re.Match) -> str:
        protected_spans.append(match.group(0))
        return f"\uE000{len(protected_spans) - 1}\uE001"

    masked_text = PROTECT_REGEX.sub(_mask, text)
    raw_sentences = SENTENCE_BOUNDARY_REGEX.split(masked_text)

    def _unmask(sentence: str) -> str:
        return re.sub(
            r'\uE000(\d+)\uE001',
            lambda m: protected_spans[int(m.group(1))],
            sentence,
        )

    sentences = [_unmask(s).strip() for s in raw_sentences]
    return [s for s in sentences if s]


def word_tokenize(sentence: str) -> List[str]:
    """Extracts plain tokens from a sentence."""
    return [match.group() for match in TOKEN_REGEX.finditer(sentence)]


# =============================================================================
# 3. PIPELINE & STATISTICS
# =============================================================================

def compute_corpus_stats(sentences: List[str], tokens: List[str], total_chars: int, json_path: str):
    """Calculates corpus statistics and saves them to a JSON file."""
    total_sentences = len(sentences)
    total_words = len(tokens)
    unique_tokens = len(set(tokens))

    avg_sentence_len = total_words / total_sentences if total_sentences > 0 else 0
    avg_word_len = sum(len(w) for w in tokens) / total_words if total_words > 0 else 0
    ttr = unique_tokens / total_words if total_words > 0 else 0

    # Build statistics dictionary
    stats = {
        "total_sentences": total_sentences,
        "total_words": total_words,
        "total_characters": total_chars,
        "avg_sentence_length": round(avg_sentence_len, 2),
        "avg_word_length": round(avg_word_len, 2),
        "type_token_ratio": round(ttr, 6),
    }

    # Save statistics as JSON file
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4)

    print("\n================ CORPUS STATISTICS ================")
    print(f"Total Sentences:           {total_sentences:,}")
    print(f"Total Words (Tokens):      {total_words:,}")
    print(f"Total Characters:          {total_chars:,}")
    print(f"Average Sentence Length:   {avg_sentence_len:.2f} words/sentence")
    print(f"Average Word Length:       {avg_word_len:.2f} characters/word")
    print(f"Type/Token Ratio (TTR):     {ttr:.6f}")
    print("===================================================\n")
    print(f"Saved stats JSON to:          {json_path}")


def process_corpus(input_filepath: str, output_basepath: str):
    """Processes input text file and exports .txt, .parquet, and .json outputs."""
    if not os.path.exists(input_filepath):
        print(f"Error: Input file '{input_filepath}' not found.")
        return

    print(f"Processing corpus from: {input_filepath}...")

    with open(input_filepath, "r", encoding="utf-8") as f:
        raw_paragraphs = [line.strip() for line in f if line.strip()]

    tokenized_sentences = []
    total_raw_characters = 0
    all_tokens = []

    for paragraph in raw_paragraphs:
        total_raw_characters += len(paragraph)
        
        # Tokenize paragraph into sentences
        sentences = sentence_tokenize(paragraph)
        
        for sentence in sentences:
            # Tokenize sentence into raw tokens
            tokens = word_tokenize(sentence)
            if tokens:
                all_tokens.extend(tokens)
                space_separated_sentence = " ".join(tokens)
                tokenized_sentences.append(space_separated_sentence)

    # Ensure output folder exists
    output_dir = os.path.dirname(output_basepath)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    txt_path = f"{output_basepath}.txt"
    parquet_path = f"{output_basepath}.parquet"
    json_path = f"{output_basepath}_stats.json"

    # Save text file (1 space-delimited sentence per line)
    with open(txt_path, "w", encoding="utf-8") as f:
        for sentence in tokenized_sentences:
            f.write(sentence + "\n")
    print(f"Saved text file to:          {txt_path}")

    # Save parquet file
    df = pd.DataFrame({"tokenized_sentence": tokenized_sentences})
    df.to_parquet(parquet_path, engine="pyarrow", compression="snappy")
    print(f"Saved parquet file to:       {parquet_path}")

    # Compute and save statistics in JSON format
    compute_corpus_stats(
        sentences=tokenized_sentences,
        tokens=all_tokens,
        total_chars=total_raw_characters,
        json_path=json_path
    )


# =============================================================================
# 4. EXECUTION
# =============================================================================

if __name__ == "__main__":
    INPUT_FILE = "raw_data/indiccorp_raw.txt"
    OUTPUT_BASE = "output/indiccorp_tokenized"

    process_corpus(INPUT_FILE, OUTPUT_BASE)