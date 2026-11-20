"""Hindi POS tagging with Naive Bayes and three HMM decoders.

Run: python pos_tagger.py
All probabilities are estimated from the supplied training split only.
"""

from __future__ import annotations

import math
import csv
from collections import Counter, defaultdict
from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "Hindi-POS-Data"
OUT = ROOT / "output"
START, END, UNK = "<START>", "<END>", "<UNK>"
METHODS = ("Naive Bayes", "Forward Viterbi", "Backward Viterbi", "Forward-Backward")
FEATURE_NAMES = ("length", "word", "previous_word", "next_word", "previous_tag")


def read_corpus(path: Path):
    sentences, words, tags = [], [], []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            if words:
                sentences.append((words, tags))
                words, tags = [], []
            continue
        parts = line.rsplit("\t", 1)
        if len(parts) != 2:
            raise ValueError(f"{path}:{line_number}: expected word<TAB>tag")
        word, tag = parts
        words.append(word)
        tags.append(tag)
    if words:
        sentences.append((words, tags))
    return sentences


class NaiveBayes:
    def __init__(self, train, tagset):
        self.tagset = tagset
        self.tag_counts = Counter()
        self.counts = [defaultdict(Counter) for _ in range(5)]
        self.vocabulary = [set() for _ in range(5)]
        for words, tags in train:
            for i, (word, tag) in enumerate(zip(words, tags)):
                values = self.features(words, i, tags[i - 1] if i else START)
                self.tag_counts[tag] += 1
                for j, value in enumerate(values):
                    self.counts[j][tag][value] += 1
                    self.vocabulary[j].add(value)
        total = sum(self.tag_counts.values())
        self.priors = {t: math.log((self.tag_counts[t] + 1) / (total + len(tagset))) for t in tagset}
        self.denominators = [
            {t: math.log(self.tag_counts[t] + len(vocab) + 1) for t in tagset}
            for vocab in self.vocabulary
        ]

    @staticmethod
    def features(words, i, previous_tag):
        return (
            "MORE" if len(words[i]) > 4 else "LESS",
            words[i],
            words[i - 1] if i else START,
            words[i + 1] if i + 1 < len(words) else END,
            previous_tag,
        )

    def predict(self, words):
        result = []
        previous = START
        for i in range(len(words)):
            values = self.features(words, i, previous)
            scores = []
            for tag in self.tagset:
                score = self.priors[tag]
                for j, value in enumerate(values):
                    score += math.log(self.counts[j][tag][value] + 1) - self.denominators[j][tag]
                scores.append(score)
            previous = self.tagset[max(range(len(scores)), key=scores.__getitem__)]
            result.append(previous)
        return result


class HMM:
    def __init__(self, train, tagset):
        self.tagset = tagset
        self.size = len(tagset)
        indices = {tag: i for i, tag in enumerate(tagset)}
        word_counts = Counter(word for words, _ in train for word in words)
        emission_counts = [Counter() for _ in tagset]
        tag_counts = [0] * self.size
        forward_counts = [[0] * self.size for _ in tagset]
        backward_counts = [[0] * self.size for _ in tagset]
        starts, ends = [0] * self.size, [0] * self.size
        for words, tags in train:
            ids = [indices[tag] for tag in tags]
            starts[ids[0]] += 1
            ends[ids[-1]] += 1
            for i, (word, tag_id) in enumerate(zip(words, ids)):
                emission_counts[tag_id][word if word_counts[word] > 1 else UNK] += 1
                tag_counts[tag_id] += 1
                if i:
                    forward_counts[ids[i - 1]][tag_id] += 1
                    backward_counts[tag_id][ids[i - 1]] += 1
        vocabulary = set().union(*(set(c) for c in emission_counts))
        self.vocabulary = vocabulary
        self.emission_counts = emission_counts
        self.emission_denoms = [tag_counts[i] + len(vocabulary) + 1 for i in range(self.size)]
        n_sentences = len(train)
        self.start = [math.log((starts[i] + 1) / (n_sentences + self.size)) for i in range(self.size)]
        self.reverse_start = [math.log((ends[i] + 1) / (n_sentences + self.size)) for i in range(self.size)]
        self.transition = [
            [(forward_counts[i][j] + 1) / (tag_counts[i] + self.size + 1) for j in range(self.size)]
            for i in range(self.size)
        ]
        self.reverse_transition = [
            [(backward_counts[i][j] + 1) / (tag_counts[i] + self.size + 1) for j in range(self.size)]
            for i in range(self.size)
        ]
        self.end = [math.log((ends[i] + 1) / (tag_counts[i] + self.size + 1)) for i in range(self.size)]
        self.reverse_end = [math.log((starts[i] + 1) / (tag_counts[i] + self.size + 1)) for i in range(self.size)]
        self.log_transition = [[math.log(x) for x in row] for row in self.transition]
        self.log_reverse_transition = [[math.log(x) for x in row] for row in self.reverse_transition]

    def emissions(self, words):
        result = []
        for word in words:
            key = word if word in self.vocabulary else UNK
            result.append([
                math.log((self.emission_counts[i][key] + 1) / self.emission_denoms[i])
                for i in range(self.size)
            ])
        return result

    def viterbi(self, emissions, reverse=False):
        if reverse:
            emissions = emissions[::-1]
            initial, transitions, terminal = self.reverse_start, self.log_reverse_transition, self.reverse_end
        else:
            initial, transitions, terminal = self.start, self.log_transition, self.end
        previous = [initial[i] + emissions[0][i] for i in range(self.size)]
        backpointers = []
        for emission in emissions[1:]:
            scores, pointers = [], []
            for current in range(self.size):
                best = max(range(self.size), key=lambda prior: previous[prior] + transitions[prior][current])
                scores.append(previous[best] + transitions[best][current] + emission[current])
                pointers.append(best)
            previous = scores
            backpointers.append(pointers)
        state = max(range(self.size), key=lambda i: previous[i] + terminal[i])
        path = [state]
        for pointers in reversed(backpointers):
            state = pointers[state]
            path.append(state)
        path.reverse()
        return path[::-1] if reverse else path

    def forward_backward(self, emissions):
        """Posterior decoding using forward and backward probability recurrences."""
        n = self.size
        scaled_emissions = []
        for row in emissions:
            peak = max(row)
            scaled_emissions.append([math.exp(value - peak) for value in row])
        alpha = []
        first = [math.exp(self.start[i]) * scaled_emissions[0][i] for i in range(n)]
        norm = sum(first)
        alpha.append([x / norm for x in first])
        for emission in scaled_emissions[1:]:
            previous = alpha[-1]
            row = [emission[j] * sum(previous[i] * self.transition[i][j] for i in range(n)) for j in range(n)]
            norm = sum(row)
            alpha.append([x / norm for x in row])
        beta = [[math.exp(x) for x in self.end]]
        norm = max(beta[0])
        beta[0] = [x / norm for x in beta[0]]
        for position in range(len(emissions) - 2, -1, -1):
            next_values = [scaled_emissions[position + 1][j] * beta[-1][j] for j in range(n)]
            row = [sum(self.transition[i][j] * next_values[j] for j in range(n)) for i in range(n)]
            norm = max(row)
            beta.append([x / norm for x in row])
        beta.reverse()
        return [max(range(n), key=lambda i: a[i] * b[i]) for a, b in zip(alpha, beta)]


def evaluate(gold, predicted, tagset):
    gold_count, pred_count, correct = Counter(), Counter(), Counter()
    total = right = 0
    for true_tags, estimated_tags in zip(gold, predicted):
        if len(true_tags) != len(estimated_tags):
            raise ValueError("Prediction length differs from reference")
        for true, estimate in zip(true_tags, estimated_tags):
            gold_count[true] += 1
            pred_count[estimate] += 1
            correct[true] += true == estimate
            total += 1
            right += true == estimate
    per_tag = {}
    for tag in tagset:
        p = correct[tag] / pred_count[tag] if pred_count[tag] else 0.0
        r = correct[tag] / gold_count[tag] if gold_count[tag] else 0.0
        per_tag[tag] = 2 * p * r / (p + r) if p + r else 0.0
    return right / total, sum(per_tag.values()) / len(tagset), per_tag, gold_count


def write_probabilities(nb, hmm, tagset):
    """Print compact probabilities and save every learned probability to TSV."""
    path = OUT / "probabilities.tsv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(("model", "probability_type", "condition", "outcome", "probability"))

        for tag in tagset:
            writer.writerow(("Naive Bayes", "tag prior", "", tag, f"{math.exp(nb.priors[tag]):.12g}"))
        for feature_index, feature_name in enumerate(FEATURE_NAMES):
            for tag in tagset:
                denominator = math.exp(nb.denominators[feature_index][tag])
                writer.writerow(("Naive Bayes", f"{feature_name} | tag", tag, "<UNSEEN>", f"{1 / denominator:.12g}"))
                for value, count in sorted(nb.counts[feature_index][tag].items()):
                    writer.writerow(("Naive Bayes", f"{feature_name} | tag", tag, value, f"{(count + 1) / denominator:.12g}"))

        for i, tag in enumerate(tagset):
            writer.writerow(("HMM", "start tag", START, tag, f"{math.exp(hmm.start[i]):.12g}"))
            writer.writerow(("HMM", "end after tag", tag, END, f"{math.exp(hmm.end[i]):.12g}"))
            writer.writerow(("HMM", "reverse start tag", END, tag, f"{math.exp(hmm.reverse_start[i]):.12g}"))
            writer.writerow(("HMM", "reverse end after tag", tag, START, f"{math.exp(hmm.reverse_end[i]):.12g}"))
            for j, next_tag in enumerate(tagset):
                writer.writerow(("HMM", "next tag | current tag", tag, next_tag, f"{hmm.transition[i][j]:.12g}"))
                writer.writerow(("HMM", "previous tag | current tag", tag, next_tag, f"{hmm.reverse_transition[i][j]:.12g}"))
            writer.writerow(("HMM", "word | tag", tag, "<UNSEEN>", f"{1 / hmm.emission_denoms[i]:.12g}"))
            for word, count in sorted(hmm.emission_counts[i].items()):
                writer.writerow(("HMM", "word | tag", tag, word, f"{(count + 1) / hmm.emission_denoms[i]:.12g}"))

    print("Calculated tag probabilities (from training data):")
    print("Tag\tNB prior\tHMM start\tHMM end")
    for i, tag in enumerate(tagset):
        print(f"{tag}\t{math.exp(nb.priors[tag]):.6f}\t{math.exp(hmm.start[i]):.6f}\t{math.exp(hmm.end[i]):.6f}")
    print(f"All calculated model probabilities: {path}")


def make_chart(split, results, tagset):
    width, left, top, row_height = 1060, 118, 75, 28
    plot_width = 780
    height = top + row_height * len(tagset) + 90
    colors = ("#2155a5", "#e27831", "#2e9a62", "#a34094")
    pieces = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
              '<rect width="100%" height="100%" fill="white"/>',
              f'<text x="{left}" y="28" font-family="Arial" font-size="19" font-weight="bold">{escape(split.title())} F1 by POS tag</text>']
    for tick in range(6):
        x = left + plot_width * tick / 5
        pieces.append(f'<line x1="{x}" y1="{top-10}" x2="{x}" y2="{top+row_height*len(tagset)}" stroke="#e1e5e9"/>')
        pieces.append(f'<text x="{x}" y="{top+row_height*len(tagset)+17}" text-anchor="middle" font-family="Arial" font-size="11">{tick/5:.1f}</text>')
    for index, tag in enumerate(tagset):
        y = top + index * row_height
        pieces.append(f'<text x="{left-9}" y="{y+16}" text-anchor="end" font-family="Arial" font-size="11">{escape(tag)}</text>')
        for j, method in enumerate(METHODS):
            value = results[method][2][tag]
            pieces.append(f'<rect x="{left}" y="{y+j*5.5}" width="{plot_width*value:.2f}" height="4" fill="{colors[j]}"/>')
    legend_y = top + row_height * len(tagset) + 45
    for j, method in enumerate(METHODS):
        x = left + j * 230
        pieces.append(f'<rect x="{x}" y="{legend_y}" width="12" height="12" fill="{colors[j]}"/>')
        pieces.append(f'<text x="{x+18}" y="{legend_y+11}" font-family="Arial" font-size="12">{escape(method)}</text>')
    pieces.append('</svg>')
    (OUT / f"{split}_f1_by_tag.svg").write_text("\n".join(pieces), encoding="utf-8")


def write_report(all_results, tagset, datasets):
    lines = ["# Lab Assignment 7: Hindi POS tagging", "", "## Method", "",
             "- Training uses only `hindi-train-pos.txt`. Dev and test are evaluated independently; neither is used to estimate probabilities.",
             "- Naive Bayes uses the five specified features: binary word length (>4), current word, previous word, next word, and previous tag. During inference, the previous tag is the model's preceding prediction. All categorical likelihoods and tag priors use add-one smoothing.",
             "- The HMM uses tag bigrams, sentence start/end probabilities, and word emission probabilities. Words seen only once in training are grouped as `<UNK>` so unseen words have an estimated emission distribution. All HMM counts use add-one smoothing.",
             "- Forward Viterbi finds the highest-probability sequence left to right. Backward Viterbi uses reverse tag transitions and decodes right to left. Forward-Backward computes forward/backward probabilities and chooses the highest-posterior tag at each position (posterior decoding).",
             "- F1 is calculated independently for every tag; a tag with no predictions has F1 = 0. Macro F1 averages all training tags. Accuracy is correct tokens divided by all tokens.", ""]
    for split in ("dev", "test"):
        results = all_results[split]
        sentences = len(datasets[split])
        tokens = sum(len(words) for words, _ in datasets[split])
        lines += [f"## {split.title()} ({sentences:,} sentences, {tokens:,} tokens)", "", "| Algorithm | Accuracy | Macro F1 |", "|---|---:|---:|"]
        for method in METHODS:
            accuracy, macro, _, _ = results[method]
            lines.append(f"| {method} | {accuracy:.4f} | {macro:.4f} |")
        lines += ["", "| Tag | Support | " + " | ".join(METHODS) + " |", "|---|---:|" + "---:|" * len(METHODS)]
        support = results[METHODS[0]][3]
        for tag in tagset:
            lines.append("| " + tag + " | " + str(support[tag]) + " | " + " | ".join(f"{results[m][2][tag]:.4f}" for m in METHODS) + " |")
        best = max(METHODS, key=lambda m: results[m][1])
        lines += ["", f"Highest macro F1: **{best}** ({results[best][1]:.4f}).", "", f"![{split.title()} F1 comparison]({split}_f1_by_tag.svg)", ""]
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    datasets = {split: read_corpus(DATA / f"hindi-{split}-pos.txt") for split in ("train", "dev", "test")}
    tagset = sorted({tag for _, tags in datasets["train"] for tag in tags})
    nb = NaiveBayes(datasets["train"], tagset)
    hmm = HMM(datasets["train"], tagset)
    OUT.mkdir(exist_ok=True)
    write_probabilities(nb, hmm, tagset)
    all_results = {}
    for split in ("dev", "test"):
        print(f"Evaluating {split}...", flush=True)
        gold = [tags for _, tags in datasets[split]]
        predictions = {method: [] for method in METHODS}
        for words, _ in datasets[split]:
            emissions = hmm.emissions(words)
            predictions["Naive Bayes"].append(nb.predict(words))
            predictions["Forward Viterbi"].append([tagset[i] for i in hmm.viterbi(emissions)])
            predictions["Backward Viterbi"].append([tagset[i] for i in hmm.viterbi(emissions, reverse=True)])
            predictions["Forward-Backward"].append([tagset[i] for i in hmm.forward_backward(emissions)])
        all_results[split] = {method: evaluate(gold, predictions[method], tagset) for method in METHODS}
        make_chart(split, all_results[split], tagset)
    write_report(all_results, tagset, datasets)
    print(f"Results written to {OUT}")


if __name__ == "__main__":
    main()
