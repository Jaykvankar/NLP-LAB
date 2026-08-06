"""
Runs the full assignment pipeline for both corpora:
    1. IndicCorpV2 (Hindi)
    2. OSCAR-2301  (Hindi)

Assumes download_data.py has already been run so that
raw_data/indiccorp_raw.txt and raw_data/oscar_raw.txt exist.
"""

from preprocess import process_corpus

if __name__ == "__main__":
    process_corpus(
        raw_path="raw_data/indiccorp_raw.txt",
        out_prefix="output/indiccorp",
        corpus_name="IndicCorpV2 (Hindi)",
    )

    process_corpus(
        raw_path="raw_data/oscar_raw.txt",
        out_prefix="output/oscar",
        corpus_name="OSCAR-2301 (Hindi)",
    )
