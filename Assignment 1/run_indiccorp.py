"""
ONE-SHOT pipeline for IndicCorpV2 (Gujarati) -- download, tokenize, save,
compute stats, all in a single run. (OSCAR-2301 is intentionally left out
of this script -- run download_data.py / preprocess.py for that later,
or extend this file the same way once you're ready.)

Usage:
    python -X utf8 -u run_indiccorp.py

Requires:
    pip install datasets huggingface_hub pandas pyarrow

Output (all under output/):
    indiccorp_tokenized.txt            plain tokenized sentences (assignment's
                                        required format: one sentence per line,
                                        tokens space-separated)
    indiccorp_tokenized_tagged.txt     same sentences, but each token tagged
                                        with its type, e.g. word/WORD ./PUNCT
                                        -- this is the "which word is which
                                        token" view
    indiccorp_tokenized.parquet        compressed columnar version of the
                                        plain tokenized data
    indiccorp_stats.json               the six corpus statistics
"""

import os
from datasets import load_dataset
import pandas as pd

from tokenizer import tokenize_paragraph_with_types
from stats import compute_stats, save_stats, print_stats

INDICCORP_SPLIT = "guj_Gujr"   # lang_Script code confirmed from HF Data Studio
N_DOCS = 20000                  # documents to pull; bump up for the full corpus
RAW_DIR = "raw_data"
OUT_DIR = "output"
BATCH_SIZE = 5000               # rows buffered before flushing to the parquet batch list

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

RAW_PATH = os.path.join(RAW_DIR, "indiccorp_raw.txt")
TXT_OUT_PATH = os.path.join(OUT_DIR, "indiccorp_tokenized.txt")
TAGGED_OUT_PATH = os.path.join(OUT_DIR, "indiccorp_tokenized_tagged.txt")
PARQUET_OUT_PATH = os.path.join(OUT_DIR, "indiccorp_tokenized.parquet")
STATS_OUT_PATH = os.path.join(OUT_DIR, "indiccorp_stats.json")


# ---------------------------------------------------------------------------
# STEP 1: download (streamed, so it doesn't try to pull the whole dataset)
# ---------------------------------------------------------------------------
def download_indiccorp():
    if os.path.exists(RAW_PATH):
        print(f"{RAW_PATH} already exists, skipping download.")
        return

    print(f"Streaming ai4bharat/IndicCorpV2 ({INDICCORP_SPLIT}) ...")
    ds = load_dataset("ai4bharat/IndicCorpV2", split=INDICCORP_SPLIT, streaming=True)

    count = 0
    with open(RAW_PATH, "w", encoding="utf-8") as f:
        for row in ds:
            text = row.get("text", "")
            if not text or not text.strip():
                continue
            f.write(text.strip().replace("\n", " ") + "\n\n")
            count += 1
            if count >= N_DOCS:
                break
            if count % 2000 == 0:
                print(f"  ...{count} documents written")

    print(f"Done: {count} documents -> {RAW_PATH}")


# ---------------------------------------------------------------------------
# STEP 2: read paragraphs (blank-line separated) from the raw file
# ---------------------------------------------------------------------------
def read_paragraphs(path):
    with open(path, "r", encoding="utf-8") as f:
        buf = []
        for line in f:
            if line.strip() == "":
                if buf:
                    yield " ".join(buf)
                    buf = []
            else:
                buf.append(line.strip())
        if buf:
            yield " ".join(buf)


# ---------------------------------------------------------------------------
# STEP 3: tokenize + write both output formats + build the parquet + stats
# ---------------------------------------------------------------------------
def tokenize_and_save():
    all_tokenized_sentences = []   # list[list[str]], for stats
    parquet_rows = []
    parquet_batches = []

    with open(TXT_OUT_PATH, "w", encoding="utf-8") as plain_out, \
         open(TAGGED_OUT_PATH, "w", encoding="utf-8") as tagged_out:

        for paragraph in read_paragraphs(RAW_PATH):
            for sent_typed in tokenize_paragraph_with_types(paragraph):
                if not sent_typed:
                    continue

                tokens = [tok for tok, _ in sent_typed]
                plain_line = " ".join(tokens)
                tagged_line = " ".join(f"{tok}/{typ}" for tok, typ in sent_typed)

                plain_out.write(plain_line + "\n")
                tagged_out.write(tagged_line + "\n")

                all_tokenized_sentences.append(tokens)
                parquet_rows.append(plain_line)

                if len(parquet_rows) >= BATCH_SIZE:
                    parquet_batches.append(pd.DataFrame({"tokenized_sentence": parquet_rows}))
                    parquet_rows = []

        if parquet_rows:
            parquet_batches.append(pd.DataFrame({"tokenized_sentence": parquet_rows}))

    full_df = (pd.concat(parquet_batches, ignore_index=True)
               if parquet_batches else pd.DataFrame({"tokenized_sentence": []}))
    full_df.to_parquet(PARQUET_OUT_PATH, engine="pyarrow", compression="snappy", index=False)

    stats = compute_stats(all_tokenized_sentences)
    save_stats(stats, STATS_OUT_PATH)
    print_stats(stats, "IndicCorpV2 (Gujarati)")

    print(f"\nplain tokenized  -> {TXT_OUT_PATH}")
    print(f"tagged tokenized -> {TAGGED_OUT_PATH}")
    print(f"parquet          -> {PARQUET_OUT_PATH}")
    print(f"stats            -> {STATS_OUT_PATH}")


if __name__ == "__main__":
    download_indiccorp()
    tokenize_and_save()
