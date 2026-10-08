"""Assignment 6: Katz backoff smoothing for four n-gram models."""
import json
import math
from collections import Counter, defaultdict
from itertools import islice
from pathlib import Path
from sys import intern

CORPUS = Path(__file__).resolve().parent.parent / "Assignment 1/output/indiccorp_tokenized.txt"
OUTPUT = Path(__file__).resolve().parent / "output/katz_backoff_results.json"
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


# Count one complete order for Good-Turing's frequency-of-frequencies.
def train(n, vocab, train_size, data):
    counts = Counter()
    for words in islice(sentences(), train_size):
        counts.update(ngrams(words, n, vocab))
    frequencies = Counter(counts.values())
    wanted = {gram[:-1] for words in data for gram in ngrams(words, n, vocab)}
    rows = defaultdict(Counter)
    for gram, count in counts.items():
        if gram[:-1] in wanted:
            rows[gram[:-1]][gram[-1]] = count
    # All followers of each retained context are kept; unused contexts are released.
    return rows, frequencies


def adjusted_counts(frequencies):
    # Fit log(Z_r) = intercept + slope * log(r), smoothing missing N_(r+1).
    counts = sorted(frequencies)
    x, y = [], []
    for index, count in enumerate(counts):
        previous = counts[index - 1] if index else 0
        following = counts[index + 1] if index + 1 < len(counts) else 2 * count - previous
        x.append(math.log(count))
        y.append(math.log(2 * frequencies[count] / (following - previous)))
    mean_x, mean_y = sum(x) / len(x), sum(y) / len(y)
    variance = sum((value - mean_x) ** 2 for value in x)
    slope = (sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y)) / variance
             if variance else -2.0)
    # Degenerate/small samples may not produce a valid discounting slope.
    if slope >= -1:
        slope = -2.0
    adjusted, use_smoothed = {}, False
    for count in counts:
        smoothed = (count + 1) * ((count + 1) / count) ** slope
        current, following = frequencies[count], frequencies.get(count + 1, 0)
        if not following:
            use_smoothed = True
        if not use_smoothed:
            empirical = (count + 1) * following / current
            deviation = (count + 1) * math.sqrt(following / current ** 2 * (1 + following / current))
            if abs(empirical - smoothed) <= 1.96 * deviation:
                use_smoothed = True
        adjusted[count] = smoothed if use_smoothed else empirical
    return adjusted


# Normalize the Good-Turing joint estimates within each context.
def good_turing_probability(n, vocab, rows, frequencies):
    vocab_size = len(vocab)
    adjusted = adjusted_counts(frequencies)
    tokens = sum(count * number for count, number in frequencies.items())
    types = sum(frequencies.values())
    possible = (vocab_size + 1) ** (n - 1) * vocab_size
    unseen_types = possible - types
    singleton_count = frequencies[1]
    missing_mass = singleton_count / tokens
    # Avoid degenerate estimates when there are no singletons or only singletons.
    if singleton_count == 0:
        missing_mass = 1 / (tokens + 1)
    elif singleton_count == tokens:
        missing_mass = tokens / (tokens + 1)
    if unseen_types == 0:
        missing_mass = 0.0
    normalizer = sum(number * adjusted[count] for count, number in frequencies.items())
    seen = {count: (1 - missing_mass) * value / normalizer
            for count, value in adjusted.items()}
    unseen = missing_mass / unseen_types if unseen_types else 0.0
    normalizers = {}

    def probability(word, history):
        row = rows.get(history)
        if not row:
            return 1.0 / vocab_size
        if history not in normalizers:
            normalizers[history] = (math.fsum(seen[count] for count in row.values())
                                    + (vocab_size - len(row)) * unseen)
        count = row.get(word, 0)
        return (seen[count] if count else unseen) / normalizers[history]

    return probability


# Back off only for unseen continuations, distributing the discounted mass.
def katz_probability(models, unigram, vocab_size):
    parameters = {}

    def probability(word, history):
        if not history:
            return unigram(word, ())
        order = len(history) + 1
        rows, contexts, discounted = models[order]
        row = rows.get(history)
        if not row:
            return probability(word, history[1:])
        if history not in parameters:
            total = contexts[history]
            leftover = math.fsum(count - discounted[count] for count in row.values()) / total
            lower_unseen = 1 - math.fsum(probability(item, history[1:]) for item in row)
            if len(row) == vocab_size or lower_unseen <= 0:
                denominator = math.fsum(discounted[count] for count in row.values())
                parameters[history] = (denominator, 0.0)
            else:
                parameters[history] = (total, leftover / lower_unseen)
        denominator, alpha = parameters[history]
        count = row.get(word, 0)
        if count:
            return discounted[count] / denominator
        return alpha * probability(word, history[1:])

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
        "smoothing": "Katz Backoff",
        "discount_method": "Smoothed Good-Turing at all counts",
        "formula": "P(w|h) = C*(h,w)/C(h) if seen; otherwise alpha(h) * P(w|suffix(h))",
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


# Keep lower-order models for backoff and print this technique's own results.
def main():
    train_size, vocab, development, test = prepare_data()
    results = {}
    models = [None]
    print("Katz backoff smoothing")
    print(f"{'Model':<12} {'Development perplexity':>24} {'Test perplexity':>20}")
    for n, name in enumerate(NAMES, start=1):
        rows, frequencies = train(n, vocab, train_size, development + test)
        contexts = {history: sum(row.values()) for history, row in rows.items()}
        if n == 1:
            unigram = good_turing_probability(1, vocab, rows, frequencies)
            discounts = {}
        else:
            adjusted = adjusted_counts(frequencies)
            # Smoothed Good-Turing discounts at all counts, without a hard cutoff.
            discounts = {count: min(value, count * (1 - 1e-12))
                         for count, value in adjusted.items()}
        models.append((rows, contexts, discounts))
        probability = katz_probability(models, unigram, len(vocab))
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
