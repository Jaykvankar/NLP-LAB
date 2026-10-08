"""Assignment 4 Q2: apply Laplace smoothing to four n-gram models."""
import math
from collections import Counter
from itertools import islice
from pathlib import Path
from sys import intern

CORPUS = Path(__file__).resolve().parent.parent / "Assignment 1/output/indiccorp_tokenized.txt"
NAMES = ["Unigram", "Bigram", "Trigram", "Quadrigram"]


# Read the corpus one sentence at a time as a list of words.
def sentences():
    with CORPUS.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                yield line.split()


# Split the corpus into training, development, and test sets and build the vocabulary.
def prepare_data():
    total = sum(1 for _ in sentences())
    if total < 1_000_000:
        raise ValueError("The assignment requires at least 1,000,000 sentences.")
    train_size = total - 2000
    frequencies = Counter()
    development, test = [], []
    for index, words in enumerate(sentences()):
        if index < train_size:
            frequencies.update(words)
        elif index < train_size + 1000:
            development.append(words)
        else:
            test.append(words)
    # Keep every word, including words that occur only once.
    vocab = set(frequencies)
    # Fix the prediction vocabulary for both evaluation sets; counts stay training-only.
    vocab.update(word for words in development + test for word in words)
    vocab.add("</s>")
    vocab.discard("<s>")  # The start marker is context only.
    print(f"Training: {train_size}, Development: {len(development)}, Test: {len(test)}", flush=True)
    return train_size, vocab, development, test


# Generate n-word tuples with sentence markers, preserving the original words.
def ngrams(words, n, vocab):
    # intern shares repeated word strings to save memory on the large corpus.
    words = [intern(word) for word in words]
    words = ["<s>"] * (n - 1) + words + ["</s>"]
    for i in range(len(words) - n + 1):
        yield tuple(words[i:i + n])


# Count each n-gram and its preceding context in the training data.
def train(n, vocab, train_size):
    counts, contexts = Counter(), Counter()
    for words in islice(sentences(), train_size):
        for gram in ngrams(words, n, vocab):
            counts[gram] += 1
            contexts[gram[:-1]] += 1
    return counts, contexts


# Calculate perplexity with Laplace smoothing on the given data.
def perplexity(data, n, vocab, counts, contexts):
    log_probability = 0.0
    total = 0
    for words in data:
        for gram in ngrams(words, n, vocab):
            numerator = counts[gram] + 1
            denominator = contexts[gram[:-1]] + len(vocab)
            if numerator == 0 or denominator == 0:
                return math.inf
            log_probability += math.log(numerator / denominator)
            total += 1
    return math.exp(-log_probability / total)


# Train all four models and print their development and test perplexities.
def main():
    train_size, vocab, development, test = prepare_data()
    print("Laplace smoothing (add one)")
    print(f"{'Model':<12} {'Development perplexity':>24} {'Test perplexity':>20}")
    for n, name in enumerate(NAMES, start=1):
        counts, contexts = train(n, vocab, train_size)
        dev_score = perplexity(development, n, vocab, counts, contexts)
        test_score = perplexity(test, n, vocab, counts, contexts)
        print(f"{name:<12} {dev_score:>24.4f} {test_score:>20.4f}", flush=True)
        del counts, contexts  # Release this model before training the next one.


if __name__ == "__main__":
    main()
