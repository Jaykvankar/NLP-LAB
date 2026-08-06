"""
Corpus statistics for NLP Assignment 1.

Takes tokenized sentences (list[list[str]]) and computes:
    i.   Total number of sentences
    ii.  Total number of words
    iii. Total number of characters
    iv.  Average Sentence Length (avg words / sentence)
    v.   Average Word Length (avg chars / word)
    vi.  Type/Token Ratio (unique tokens / total tokens)
"""

import json


def compute_stats(tokenized_sentences):
    total_sentences = len(tokenized_sentences)
    total_words = sum(len(sent) for sent in tokenized_sentences)

    # "characters" = characters across all word tokens (punctuation tokens
    # count too, since they are still tokens produced by the tokenizer;
    # this matches "total number of characters" in the tokenized corpus).
    total_chars = sum(len(tok) for sent in tokenized_sentences for tok in sent)

    avg_sentence_length = total_words / total_sentences if total_sentences else 0.0
    avg_word_length = total_chars / total_words if total_words else 0.0

    all_tokens = [tok.lower() for sent in tokenized_sentences for tok in sent]
    unique_tokens = set(all_tokens)
    ttr = len(unique_tokens) / total_words if total_words else 0.0

    return {
        "total_sentences": total_sentences,
        "total_words": total_words,
        "total_characters": total_chars,
        "average_sentence_length": round(avg_sentence_length, 4),
        "average_word_length": round(avg_word_length, 4),
        "type_token_ratio": round(ttr, 6),
        "unique_tokens": len(unique_tokens),
    }


def save_stats(stats: dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)


def print_stats(stats: dict, corpus_name: str = "Corpus"):
    print(f"\n----- {corpus_name} statistics -----")
    for k, v in stats.items():
        print(f"{k:28s}: {v}")
