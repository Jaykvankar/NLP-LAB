"""Noisy-channel spell checker.

    correct = argmax_c  P(error | c) P(c)

P(c) comes from count_unigram.txt. P(error | c) comes from the single edit that
turns c into the error, looked up in count_1edit.txt. In that file the left
side of X|Y is the correct fragment and the right side is what was typed:

    e|i     i was typed where e was correct          (substitution)
    re|r    e was deleted after r                    (deletion)
    t|te    e was inserted after t                   (insertion)
    er|re   er was swapped                           (transposition)

Candidates are dictionary words at Damerau-Levenshtein distance 1. The error
itself is not a candidate: every item in spell-errors.txt is a mistake.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"
LETTERS = "abcdefghijklmnopqrstuvwxyz'"


BOUNDARY = ">"  # start-of-word marker used in count_1edit.txt


def edit_key(correct: str, error: str) -> str | None:
    """The count_1edit key for the one edit taking correct to error.

    Returns None when the pair is not a single insertion, deletion,
    substitution, or adjacent transposition. A deletion or insertion at the
    start of the word is conditioned on the boundary marker '>'.
    """
    if correct == error:
        return None
    n, m = len(correct), len(error)
    if n == m:
        diffs = [i for i in range(n) if correct[i] != error[i]]
        if len(diffs) == 1:
            i = diffs[0]
            return f"{correct[i]}|{error[i]}"
        if (len(diffs) == 2 and diffs[1] == diffs[0] + 1
                and correct[diffs[0]] == error[diffs[1]]
                and correct[diffs[1]] == error[diffs[0]]):
            i = diffs[0]
            return f"{correct[i:i + 2]}|{error[i:i + 2]}"
        return None
    if n == m + 1:
        for i in range(n):
            if i == m or correct[i] != error[i]:
                if correct[i + 1:] != error[i:]:
                    return None
                prev = correct[i - 1] if i else BOUNDARY
                return f"{prev}{correct[i]}|{prev}"
        return None
    if m == n + 1:
        for i in range(m):
            if i == n or correct[i] != error[i]:
                if correct[i:] != error[i + 1:]:
                    return None
                prev = correct[i - 1] if i else BOUNDARY
                return f"{prev}|{prev}{error[i]}"
        return None
    return None


def edits1(word: str, alphabet: str) -> set[str]:
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [l + r[1:] for l, r in splits if r]
    transposes = [l + r[1] + r[0] + r[2:] for l, r in splits if len(r) > 1]
    replaces = [l + c + r[1:] for l, r in splits if r for c in alphabet]
    inserts = [l + c + r for l, r in splits for c in alphabet]
    out = set(deletes + transposes + replaces + inserts)
    out.discard(word)
    return out


def load_unigram(path: Path) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    with path.open(encoding="latin-1") as f:
        for line in f:
            word, count = line.rstrip("\n").split("\t")
            counts[word.lower()] += int(count)
    return dict(counts)


def load_edits(path: Path) -> dict[str, int]:
    edits = {}
    with path.open(encoding="latin-1") as f:
        for line in f:
            key, count = line.rstrip("\n").split("\t")
            edits[key] = edits.get(key, 0) + int(count)
    return edits


def parse_errors(path: Path) -> list[tuple[str, str, int]]:
    """(correct, error, times). A trailing *N repeats that error N times."""
    rows = []
    with path.open(encoding="latin-1") as f:
        for line in f:
            if ":" not in line:
                continue
            correct, rest = line.strip().split(":", 1)
            correct = correct.strip().lower()
            for piece in rest.split(","):
                piece = piece.strip().lower()
                if not piece:
                    continue
                if "*" in piece:
                    word, times = piece.rsplit("*", 1)
                    rows.append((correct, word.strip(), int(times)))
                else:
                    rows.append((correct, piece, 1))
    return rows


class SpellChecker:
    def __init__(self, unigram: dict[str, int], edits: dict[str, int]):
        self.unigram = unigram
        self.total = sum(unigram.values())
        self.log_prior = {w: math.log(c) - math.log(self.total) for w, c in unigram.items()}
        self.edits = edits
        # Character unigrams and bigrams, weighted by word frequency, with '>'
        # marking the start of a word. These are the denominators of the channel:
        # P(edit) = (count of that edit + 1) / (count of the correct letters + 1).
        char_uni: Counter = Counter()
        char_bi: Counter = Counter()
        for word, count in unigram.items():
            chars = BOUNDARY + word
            for ch in chars:
                char_uni[ch] += count
            for a, b in zip(chars, chars[1:]):
                char_bi[a + b] += count
        self.char_uni = char_uni
        self.char_bi = char_bi

    def channel(self, correct: str, error: str) -> float | None:
        key = edit_key(correct, error)
        if key is None:
            return None
        left, _right = key.split("|", 1)
        context = self.char_bi.get(left, 0) if len(left) == 2 else self.char_uni.get(left, 0)
        return math.log(self.edits.get(key, 0) + 1) - math.log(context + 1)

    def correct(self, error: str) -> tuple[str | None, float]:
        alphabet = "".join(sorted(set(LETTERS + error)))
        best_word, best_score = None, -1e300
        freq_word, freq_score = None, -1e300
        for cand in edits1(error, alphabet):
            prior = self.log_prior.get(cand)
            if prior is None:
                continue
            if prior > freq_score or (prior == freq_score and (freq_word is None or cand < freq_word)):
                freq_word, freq_score = cand, prior
            chan = self.channel(cand, error)
            if chan is None:
                continue
            score = prior + chan
            if score > best_score or (score == best_score and (best_word is None or cand < best_word)):
                best_word, best_score = cand, score
        return best_word, best_score, freq_word

    def evaluate(self, rows: list[tuple[str, str, int]]) -> dict:
        weighted_hit = weighted_total = 0
        types_hit = types_total = 0
        within1 = within1_hit = freq_hit = 0
        missing_gold = 0
        mistakes = []
        for correct, error, times in rows:
            if correct not in self.unigram:
                missing_gold += times
            guess, _, freq_word = self.correct(error)
            ok = guess == correct
            weighted_total += times
            weighted_hit += times * ok
            types_total += 1
            types_hit += ok
            key = edit_key(correct, error)
            if key is not None:
                within1 += times
                within1_hit += times * ok
                freq_hit += times * (freq_word == correct)
            if not ok and len(mistakes) < 25:
                mistakes.append({
                    "error": error, "gold": correct, "predicted": guess, "count": times,
                    "edit": key,
                })
        return {
            "error_types": types_total,
            "weighted_instances": weighted_total,
            "accuracy_weighted": weighted_hit / weighted_total,
            "accuracy_unweighted": types_hit / types_total,
            "within_1_edit_instances": within1,
            "accuracy_within_1_edit": within1_hit / within1 if within1 else None,
            "frequency_baseline_within_1_edit": freq_hit / within1 if within1 else None,
            "gold_not_in_unigram_instances": missing_gold,
            "sample_mistakes": mistakes,
        }


def demo_channel() -> None:
    samples = [
        ("raining", "rainning"),
        ("raining", "raning"),
        ("yellow", "yello"),
        ("aggression", "agression"),
        ("four", "fore"),
    ]
    print("Edit keys (correct -> error):")
    for c, e in samples:
        print(f"  {c} -> {e}: {edit_key(c, e)}")


def main() -> None:
    demo_channel()
    print("Loading counts...", flush=True)
    unigram = load_unigram(DATA / "count_unigram.txt")
    edits = load_edits(DATA / "count_1edit.txt")
    rows = parse_errors(DATA / "spell-errors.txt")
    print(f"  {len(unigram):,} words, {len(edits):,} edit types, "
          f"{len(rows):,} error types", flush=True)
    checker = SpellChecker(unigram, edits)
    print("Evaluating...", flush=True)
    result = checker.evaluate(rows)
    shown = {k: v for k, v in result.items() if k != "sample_mistakes"}
    print(json.dumps(shown, indent=2))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "spell_results.json").write_text(json.dumps(result, indent=2) + "\n")

    lines = ["error -> gold (predicted)", "-" * 40]
    for row in rows[:15]:
        correct, error, times = row
        guess, _, _ = checker.correct(error)
        mark = "ok" if guess == correct else "WRONG"
        lines.append(f"{error} -> {correct} ({guess}) [{mark}] x{times}")
    (OUT / "spell_examples.txt").write_text("\n".join(lines) + "\n")
    print(f"Wrote {OUT / 'spell_results.json'}")


if __name__ == "__main__":
    main()
