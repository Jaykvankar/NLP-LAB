"""Assignment 6: interpolated smoothing for bigram, trigram, and quadrigram models."""
import json
import math
from collections import Counter, defaultdict
from itertools import islice
from pathlib import Path
from sys import intern

CORPUS = Path(__file__).resolve().parent.parent / "Assignment 1/output/indiccorp_tokenized.txt"
OUTPUT = Path(__file__).resolve().parent / "output/interpolation_results.json"
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


# Count all training followers of the contexts needed for development and testing.
def train(n, vocab, train_size, data):
    wanted = {gram[:-1] for words in data for gram in ngrams(words, n, vocab)}
    rows = defaultdict(Counter)
    for words in islice(sentences(), train_size):
        for gram in ngrams(words, n, vocab):
            if gram[:-1] in wanted:
                rows[gram[:-1]][gram[-1]] += 1
    contexts = {history: sum(row.values()) for history, row in rows.items()}
    return rows, contexts


# Mix uniform, unigram, and higher-order maximum-likelihood probabilities.
def components(word, history, n, vocab, models):
    values = [1.0 / len(vocab)]
    for order in range(1, n + 1):
        context = history[-(order - 1):] if order > 1 else ()
        rows, contexts = models[order]
        total = contexts.get(context, 0)
        if total:
            values.append(rows[context].get(word, 0) / total)
        else:
            values.append(values[-1])  # Use a shorter context when this one is unseen.
    return values


# Learn interpolation weights using only the development set.
def choose_weights(development, n, vocab, models):
    events = Counter(gram for words in development for gram in ngrams(words, n, vocab))
    samples = [(components(gram[-1], gram[:-1], n, vocab, models), count)
               for gram, count in events.items()]
    weights = [1.0 / (n + 1)] * (n + 1)
    total = sum(events.values())
    for _ in range(100):
        expected = [0.0] * len(weights)
        for values, count in samples:
            mixture = sum(weight * value for weight, value in zip(weights, values))
            for index, value in enumerate(values):
                expected[index] += count * weights[index] * value / mixture
        updated = [max(value / total, 1e-8) for value in expected]
        normalizer = sum(updated)
        updated = [value / normalizer for value in updated]
        change = max(abs(a - b) for a, b in zip(weights, updated))
        weights = updated
        if change < 1e-7:
            break
    return weights


def interpolated_probability(n, vocab, models, weights):
    def probability(word, history):
        return sum(weight * value for weight, value in
                   zip(weights, components(word, history, n, vocab, models)))
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
        "smoothing": "Interpolated",
        "weight_estimation": "EM on development data only",
        "formula": "P(w|h) = sum(lambda_i * P_i(w|h)), including uniform P_0(w) = 1/V",
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


# Train the required models and print this technique's own results.
def main():
    train_size, vocab, development, test = prepare_data()
    results = {}
    models = [None]
    print("Interpolated smoothing")
    print(f"{'Model':<12} {'Development perplexity':>24} {'Test perplexity':>20}")
    for n, name in enumerate(NAMES, start=1):
        models.append(train(n, vocab, train_size, development + test))
        if n == 1:
            continue  # Unigrams are a component; the assignment requests orders 2-4.
        weights = choose_weights(development, n, vocab, models)
        probability = interpolated_probability(n, vocab, models, weights)
        dev_result = evaluate(development, n, vocab, probability)
        test_result = evaluate(test, n, vocab, probability)
        results[name.lower()] = {
            "weights": dict(zip(["uniform"] + [item.lower() for item in NAMES[:n]], weights)),
            "development": dev_result,
            "test": test_result,
        }
        dev_score = float(dev_result["perplexity"])
        test_score = float(test_result["perplexity"])
        print(f"{name:<12} {dev_score:>24.4f} {test_score:>20.4f}", flush=True)
        print("  Weights (uniform to highest order): "
              + ", ".join(f"{weight:.4f}" for weight in weights), flush=True)

    save_results(train_size, vocab, development, test, results)


if __name__ == "__main__":
    main()
