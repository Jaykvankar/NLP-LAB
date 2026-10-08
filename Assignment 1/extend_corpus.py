"""Extend the existing Gujarati corpus to a target tokenized sentence count.

Stream genuine IndicCorpV2 text and reject already-present tokenized sentences.
Regenerate and verify all outputs before replacing the existing corpus files.
"""
import argparse
import hashlib
import io
import json
import shutil
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq

from main import process_corpus, sentence_tokenize, word_tokenize

ROOT = Path(__file__).resolve().parent
SOURCE = "https://huggingface.co/datasets/ai4bharat/IndicCorpV2/resolve/2d7285e6ce14fdb3fb2449c9f89427b9f582ac3f/data/gu.txt"


def fingerprint(text):
    return hashlib.sha256(text.encode("utf-8")).digest()


def main(target):
    raw = ROOT / "raw_data/indiccorp_raw.txt"
    output = ROOT / "output"
    tokenized = output / "indiccorp_tokenized.txt"
    seen = set()
    total = 0
    with tokenized.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                total += 1
                seen.add(fingerprint(" ".join(line.split())))
    print(f"Existing sentences: {total:,}; target: {target:,}", flush=True)
    if total >= target:
        print("The corpus already meets the requested size.", flush=True)
        return
    needed = target - total
    stage = output / "extension_staging"
    stage.mkdir(exist_ok=True)
    extra = stage / "additional_sentences.txt"
    added = 0
    # Start near the end of the already-downloaded prefix to avoid rereading it.
    # Exact alignment is unnecessary: discard the partial first line, then deduplicate.
    request = urllib.request.Request(SOURCE, headers={
        "User-Agent": "NLP-LAB-corpus-extension", "Range": f"bytes={raw.stat().st_size}-"})
    with urllib.request.urlopen(request, timeout=120) as response:
        print(f"Source HTTP status: {response.status}", flush=True)
        if response.status == 206:
            response.readline()
        with io.TextIOWrapper(response, encoding="utf-8") as stream, extra.open("w", encoding="utf-8") as dest:
            for scanned, paragraph in enumerate(stream, 1):
                if scanned % 2000 == 0:
                    print(f"Scanned {scanned:,} source lines; added {added:,} sentences", flush=True)
                for sentence in sentence_tokenize(paragraph.strip()):
                    words = word_tokenize(sentence)
                    if not words:
                        continue
                    digest = fingerprint(" ".join(words))
                    if digest in seen:
                        continue
                    seen.add(digest)
                    dest.write(sentence + "\n")
                    added += 1
                    if added % 5000 == 0:
                        print(f"Collected {added:,}/{needed:,} new sentences", flush=True)
                    if added == needed:
                        break
                if added == needed:
                    break
    if added != needed:
        raise RuntimeError(f"Source exhausted after adding {added} of {needed} sentences")
    del seen
    combined = stage / "indiccorp_raw.txt"
    shutil.copyfile(raw, combined)
    with combined.open("ab") as dest, extra.open("rb") as source:
        dest.write(b"\n")
        shutil.copyfileobj(source, dest)
    print("Regenerating Assignment 1 text, Parquet, and statistics...", flush=True)
    base = stage / "indiccorp_tokenized"
    process_corpus(str(combined), str(base))
    with base.with_suffix(".txt").open(encoding="utf-8") as source:
        actual = sum(bool(line.strip()) for line in source)
    stats = json.loads((stage / "indiccorp_tokenized_stats.json").read_text(encoding="utf-8"))
    assert actual == target == stats["total_sentences"]
    assert pq.ParquetFile(base.with_suffix(".parquet")).metadata.num_rows == target
    combined.replace(raw)
    for name in ("indiccorp_tokenized.txt", "indiccorp_tokenized.parquet", "indiccorp_tokenized_stats.json"):
        (stage / name).replace(output / name)
    provenance = {"source": SOURCE, "original_sentences": total,
                  "added_sentences": added, "total_sentences": actual,
                  "duplicate_check": "SHA-256 of normalized tokenized sentences; existing and new sentences checked"}
    (output / "corpus_extension.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(f"Verified and saved {actual:,} sentences in TXT, Parquet, and statistics.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=int, default=1_000_000)
    main(parser.parse_args().target)
