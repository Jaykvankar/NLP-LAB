"""
Task 1a: download raw text.

    Dataset 1: ai4bharat/IndicCorpV2   (language = Gujarati)
    Dataset 2: oscar-corpus/OSCAR-2301 (language = Gujarati)

Run this on a machine with real internet access (your laptop / Colab /
Kaggle) -- both datasets are large, so we STREAM them instead of doing a
full download, and cap the number of documents pulled (`N_DOCS`) so a
"complete data extraction" doesn't take hours or blow your disk. Bump
N_DOCS up if your assignment wants the truly full corpus.

Requires:
    pip install datasets huggingface_hub

IMPORTANT -- the two datasets use DIFFERENT language-code conventions:
    - IndicCorpV2 splits are named "<ISO639-3>_<Script>", e.g. "guj_Gujr"
      (confirmed from the Data Studio view: guj_Gujr, asm_Beng, ben_Beng...).
      It has NO config/subset -- you select the language via `split=`.
    - OSCAR-2301 configs use plain ISO639-1 codes, e.g. "gu" for Gujarati.
      Verify this on https://huggingface.co/datasets/oscar-corpus/OSCAR-2301
      (Data Studio / "Subsets and Splits") the same way you did for
      IndicCorpV2 before running -- OSCAR's list can differ.

OSCAR-2301 is a *gated* dataset -- you must:
    1. Log in at https://huggingface.co/datasets/oscar-corpus/OSCAR-2301
       and accept the terms once.
    2. Run `huggingface-cli login` (or set env var HF_TOKEN) before
       running this script.
"""

import os
from datasets import load_dataset

INDICCORP_SPLIT = "guj_Gujr"   # IndicCorpV2: lang_Script code, no config needed
OSCAR_CONFIG = "gu"            # OSCAR-2301: plain ISO639-1 code -- verify on HF
N_DOCS = 20000                  # how many documents to pull from each streamed corpus
RAW_DIR = "raw_data"

os.makedirs(RAW_DIR, exist_ok=True)


def dump_streaming_dataset(dataset_name, out_path, n_docs, split="train",
                            config=None, text_field="text"):
    label = f"{dataset_name} ({config or split})"
    print(f"Streaming {label} ...")
    if config:
        ds = load_dataset(dataset_name, config, split=split, streaming=True)
    else:
        ds = load_dataset(dataset_name, split=split, streaming=True)

    count = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for row in ds:
            text = row.get(text_field, "")
            if not text or not text.strip():
                continue
            f.write(text.strip().replace("\n", " ") + "\n\n")  # blank line = paragraph sep
            count += 1
            if count >= n_docs:
                break
            if count % 2000 == 0:
                print(f"  ...{count} documents written")

    print(f"Done: {count} documents -> {out_path}")


if __name__ == "__main__":
    # --- IndicCorpV2 --- (no config; language selected via split name)
    # dump_streaming_dataset(
    #     dataset_name="ai4bharat/IndicCorpV2",
    #     split=INDICCORP_SPLIT,
    #     out_path=os.path.join(RAW_DIR, "indiccorp_raw.txt"),
    #     n_docs=N_DOCS,
    # )

    # --- OSCAR-2301 --- (language selected via config, split is "train")
    dump_streaming_dataset(
        dataset_name="oscar-corpus/OSCAR-2301",
        config=OSCAR_CONFIG,
        split="train",
        out_path=os.path.join(RAW_DIR, "oscar_raw.txt"),
        n_docs=N_DOCS,
        text_field="text",
    )

