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
INTERPOLATION_WEIGHTS = {
    2: [0.40, 0.60],
    3: [0.20, 0.30, 0.50],
    4: [0.10, 0.20, 0.30, 0.40],
}
GOOD_TURING_CUTOFF = 5
EPSILON = 1e-15


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
    # Match the reference vocabulary; held-out words map to <UNK>.
    vocab.update({"<UNK>", "<s>", "</s>"})
    print(f"Training: {train_size}, Development: {len(development)}, Test: {len(test)}", flush=True)
    return train_size, vocab, development, test


# Generate n-word tuples with sentence markers and training-vocabulary mapping.
def ngrams(words, n, vocab):
    # intern shares repeated word strings to save memory on the large corpus.
    words = [intern(word) if word in vocab else "<UNK>" for word in words]
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


# Build the reference's cutoff Good-Turing unigram component once.
def good_turing_unigram(vocab, model):
    rows, _ = model
    counts = rows.get((), {})
    frequencies = Counter(counts.values())
    adjusted = {}
    for count in frequencies:
        following = frequencies.get(count + 1, 0)
        if 1 <= count <= GOOD_TURING_CUTOFF and following:
            value = (count + 1) * following / frequencies[count]
            adjusted[count] = min(max(value, EPSILON), float(count))
        else:
            adjusted[count] = float(count)
    unseen_types = max(len(vocab) - len(counts), 1)
    unseen_count = frequencies.get(1, 0) / unseen_types if frequencies.get(1, 0) else EPSILON
    normalization = sum(adjusted[count] for count in counts.values()) + unseen_types * unseen_count

    def probability(word):
        count = counts.get(word, 0)
        return (adjusted[count] if count else unseen_count) / max(normalization, EPSILON)

    return probability


# Mix Good-Turing unigrams with MLE higher orders, aligned to the same word.
def components(word, history, n, vocab, models, unigram_probability):
    values = [unigram_probability(word)]
    for order in range(2, n + 1):
        context = history[-(order - 1):]
        rows, contexts = models[order]
        total = contexts.get(context, 0)
        if total:
            values.append(rows[context].get(word, 0) / total)
        else:
            values.append(0.0)  # An unseen MLE context has probability zero.
    return values


def interpolated_probability(n, vocab, models, weights):
    if len(weights) != n:
        raise ValueError("Number of interpolation weights must equal order.")
    if any(not math.isfinite(weight) or weight < 0 for weight in weights):
        raise ValueError("Interpolation weights must be finite and nonnegative.")
    if not math.isclose(sum(weights), 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("Interpolation weights must sum to 1.")
    unigram_probability = good_turing_unigram(vocab, models[1])

    def probability(word, history):
        values = components(word, history, n, vocab, models, unigram_probability)
        return max(sum(weight * value for weight, value in zip(weights, values)), EPSILON)
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
        "cross_entropy": exponent / math.log(2) if math.isfinite(exponent) else "Infinity",
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
        "weight_estimation": "Fixed weights from the supplied reference code",
        "good_turing_cutoff": GOOD_TURING_CUTOFF,
        "epsilon": EPSILON,
        "formula": "P(w|h) = max(sum(lambda_i * P_i(w|h)), epsilon); Good-Turing unigram, MLE higher orders",
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
        "vocabulary_policy": "Training words plus <UNK>, <s>, </s>; held-out unknown words map to <UNK>.",
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
        weights = INTERPOLATION_WEIGHTS[n]
        probability = interpolated_probability(n, vocab, models, weights)
        dev_result = evaluate(development, n, vocab, probability)
        test_result = evaluate(test, n, vocab, probability)
        results[name.lower()] = {
            "weights": dict(zip([item.lower() for item in NAMES[:n]], weights)),
            "development": dev_result,
            "test": test_result,
        }
        dev_score = float(dev_result["perplexity"])
        test_score = float(test_result["perplexity"])
        print(f"{name:<12} {dev_score:>24.4f} {test_score:>20.4f}", flush=True)
        print("  Weights (unigram to highest order): "
              + ", ".join(f"{weight:.4f}" for weight in weights), flush=True)

    save_results(train_size, vocab, development, test, results)


if __name__ == "__main__":
    main()
