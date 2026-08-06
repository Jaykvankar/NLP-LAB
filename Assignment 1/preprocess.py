"""
Tasks 1b, 1c, 1d, and 2: tokenize raw corpus text, save tokenized output
(as .txt and as compressed .parquet), and compute corpus statistics.

Usage:
    python preprocess.py raw_data/indiccorp_raw.txt output/indiccorp
    python preprocess.py raw_data/oscar_raw.txt      output/oscar

Output produced (for out_prefix="output/indiccorp"):
    output/indiccorp_tokenized.txt       one tokenized sentence per line
    output/indiccorp_tokenized.parquet   same data, compressed columnar format
    output/indiccorp_stats.json          corpus statistics
"""

import os
import sys
import pandas as pd

from tokenizer import tokenize_paragraph
from stats import compute_stats, save_stats, print_stats

BATCH_SIZE = 5000  # sentences buffered before flushing to parquet, keeps memory flat


def read_paragraphs(path):
    """Yield paragraphs from a raw text file (paragraphs separated by blank lines)."""
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


def process_corpus(raw_path, out_prefix, corpus_name):
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)

    txt_out_path = out_prefix + "_tokenized.txt"
    parquet_out_path = out_prefix + "_tokenized.parquet"
    stats_out_path = out_prefix + "_stats.json"

    all_tokenized_sentences = []   # kept for the final stats computation
    parquet_batches = []
    batch_rows = []

    with open(txt_out_path, "w", encoding="utf-8") as txt_out:
        for paragraph in read_paragraphs(raw_path):
            for sent_tokens in tokenize_paragraph(paragraph):
                if not sent_tokens:
                    continue
                # Task instructions: combine tokenized words by spaces,
                # one tokenized sentence per line.
                line = " ".join(sent_tokens)
                txt_out.write(line + "\n")

                all_tokenized_sentences.append(sent_tokens)
                batch_rows.append(line)

                if len(batch_rows) >= BATCH_SIZE:
                    parquet_batches.append(pd.DataFrame({"tokenized_sentence": batch_rows}))
                    batch_rows = []

        if batch_rows:
            parquet_batches.append(pd.DataFrame({"tokenized_sentence": batch_rows}))

    # --- save as parquet, compressed (Task: "use compression ... parquet") ---
    if parquet_batches:
        full_df = pd.concat(parquet_batches, ignore_index=True)
    else:
        full_df = pd.DataFrame({"tokenized_sentence": []})
    full_df.to_parquet(parquet_out_path, engine="pyarrow", compression="snappy", index=False)

    # --- corpus statistics ---
    stats = compute_stats(all_tokenized_sentences)
    save_stats(stats, stats_out_path)
    print_stats(stats, corpus_name)

    raw_size = os.path.getsize(txt_out_path) / (1024 * 1024)
    parquet_size = os.path.getsize(parquet_out_path) / (1024 * 1024)
    print(f"{'txt size (MB)':28s}: {raw_size:.2f}")
    print(f"{'parquet size (MB)':28s}: {parquet_size:.2f}  "
          f"({(1 - parquet_size / raw_size) * 100:.1f}% smaller)" if raw_size else "")

    return stats


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python preprocess.py <raw_text_path> <output_prefix>")
        sys.exit(1)

    raw_path, out_prefix = sys.argv[1], sys.argv[2]
    name = os.path.basename(out_prefix)
    process_corpus(raw_path, out_prefix, corpus_name=name)
