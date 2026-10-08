"""Assignment 6: Kneser-Ney smoothing for four n-gram models."""
import json
import math
from collections import Counter, defaultdict
from itertools import islice
from pathlib import Path
from sys import intern

CORPUS = Path(__file__).resolve().parent.parent / "Assignment 1/output/indiccorp_tokenized.txt"
OUTPUT = Path(__file__).resolve().parent / "output/kneser_ney_results.json"
NAMES = ["Unigram", "Bigram", "Trigram", "Quadrigram"]
DISCOUNT = 0.75
UNIFORM_WEIGHT = 1e-8


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


# Count this order and distinct left contexts for the next lower order.
def train(n, vocab, train_size, data):
    counts = Counter()
    for words in islice(sentences(), train_size):
        counts.update(ngrams(words, n, vocab))
    wanted = {gram[:-1] for words in data for gram in ngrams(words, n, vocab)}
    lower_wanted = ({gram[:-1] for words in data for gram in ngrams(words, n - 1, vocab)}
                    if n > 1 else set())
    rows, lower_rows = defaultdict(Counter), defaultdict(Counter)
    for gram, count in counts.items():
        if gram[:-1] in wanted:
            rows[gram[:-1]][gram[-1]] = count
        if n > 1 and gram[1:-1] in lower_wanted:
            # A distinct n-gram contributes once, regardless of its frequency.
            lower_rows[gram[1:-1]][gram[-1]] += 1
    contexts = {history: sum(row.values()) for history, row in rows.items()}
    lower_contexts = {history: sum(row.values()) for history, row in lower_rows.items()}
    return (rows, contexts), (lower_rows, lower_contexts)


# Interpolate discounted counts with lower-order continuation probabilities.
def kneser_ney_probability(n, vocab, models, continuation):
    def probability(word, history):
        order = len(history) + 1
        rows, contexts = models[order] if order == n else continuation[order]
        row = rows.get(history, {})
        total = contexts.get(history, 0)
        if order == 1:
            # Continuation unigrams for n >= 2, ordinary frequencies for n = 1.
            # Reserve a tiny uniform mass for unseen words in the fixed vocabulary.
            return ((1 - UNIFORM_WEIGHT) * row.get(word, 0) / total
                    + UNIFORM_WEIGHT / len(vocab))
        lower_probability = probability(word, history[1:])
        if not total:
            return lower_probability
        observed = max(row.get(word, 0) - DISCOUNT, 0) / total
        backoff_weight = DISCOUNT * len(row) / total
        return observed + backoff_weight * lower_probability

    return probability


# Calculate perplexity, counting the sentence-end marker as a predicted word.
def evaluate(data, n, vocab, probability):
    log_probability = 0.0
    total, zero_events = 0, 0
    for words in data:
        for gram in ngrams(words, n, vocab):
            value = probability(gram[-1], gram[:-1])
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid model score: {value}")
            total += 1
            if value == 0:
                zero_events += 1
            else:
                log_probability += math.log(value)
    if total == 0:
        raise ValueError("Cannot evaluate an empty dataset.")
    exponent = math.inf if zero_events else -log_probability / total
    score = math.exp(exponent) if exponent < 709 else math.inf
    return {
        "predicted_tokens": total,
        "zero_probability_events": zero_events,
        # JSON has no infinity literal, so preserve it as a readable string.
        "perplexity": score if math.isfinite(score) else "Infinity",
    }


def perplexity(data, n, vocab, probability):
    return float(evaluate(data, n, vocab, probability)["perplexity"])


# Save this script's results in the same structure as Assignment 5.
def save_results(train_size, vocab, development, test, models):
    total = train_size + len(development) + len(test)
    results = {
        "smoothing": "Kneser-Ney",
        "discount": DISCOUNT,
        "unigram_uniform_weight": UNIFORM_WEIGHT,
        "formula": "P(w|h) = max(C(h,w)-D,0)/C(h) + D*N_1+(h,*)/C(h) * P_lower(w|suffix(h))",
        "total_sentences": total,
        "required_sentences": 1_000_000,
        "requirement_met": total >= 1_000_000,
        "shortfall": max(0, 1_000_000 - total),
        "training_sentences": train_size,
        "development_sentences": len(development),
        "test_sentences": len(test),
        "split_method": "Original order: training, development, test",
        "prediction_vocabulary_size": len(vocab),
        "models": models,
        "vocabulary_policy": "Original words from training, development, and test; counts from training only; sentence end included.",
        "infinity_encoding": "Infinite results are stored as the string Infinity.",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as target:
        json.dump(results, target, indent=2, ensure_ascii=False, allow_nan=False)
        target.write("\n")
    print(f"Results saved to {OUTPUT}", flush=True)


# Train all four models and print this technique's own perplexities.
def main():
    train_size, vocab, development, test = prepare_data()
    results = {}
    models, continuation = [None], [None] * 4
    print("Kneser-Ney smoothing (discount = 0.75)")
    print(f"{'Model':<12} {'Development perplexity':>24} {'Test perplexity':>20}")
    for n, name in enumerate(NAMES, start=1):
        current, lower = train(n, vocab, train_size, development + test)
        models.append(current)
        if n > 1:
            continuation[n - 1] = lower
        probability = kneser_ney_probability(n, vocab, models, continuation)
        dev_result = evaluate(development, n, vocab, probability)
        test_result = evaluate(test, n, vocab, probability)
        results[name.lower()] = {
            "development": dev_result,
            "test": test_result,
        }
        dev_score = float(dev_result["perplexity"])
        test_score = float(test_result["perplexity"])
        print(f"{name:<12} {dev_score:>24.4f} {test_score:>20.4f}", flush=True)

    save_results(train_size, vocab, development, test, results)


if __name__ == "__main__":
    main()
