"""Programs for the five NLP mid-semester questions (Assignment 8, Q1).

Each section prints the full working, not just the final number.
"""

from __future__ import annotations

import math
from collections import Counter
from fractions import Fraction
from pathlib import Path

OUT = Path(__file__).resolve().parent / "outputs"


def h(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


# ---------------------------------------------------------------- Q1 FST
# Generator: it reads a noun stem, then +N, then +SG or +PL, and writes the
# surface form. Plural is "es" after s/x/z/ch/sh and "s" otherwise.

CLASS_OF = {
    "s": "S", "x": "X", "z": "Z", "c": "C", "h": "H",
}
ES_CLASSES = {"S", "X", "Z", "CH", "SH"}
STEM_STATES = ["S", "X", "Z", "C", "H", "CH", "SH", "O"]


def next_class(state: str, letter: str) -> str:
    if letter == "h" and state == "C":
        return "CH"
    if letter == "h" and state == "S":
        return "SH"
    if letter == "h":
        return "H"
    return CLASS_OF.get(letter, "O")


def class_of_stem(stem: str) -> str:
    state = "0"
    for ch in stem:
        state = next_class(state, ch)
    return state


def generate(stem: str, number: str) -> str:
    """Run the FST. number is 'SG' or 'PL'."""
    if not stem or any(c < "a" or c > "z" for c in stem):
        raise ValueError(f"stem must be lowercase a-z, got {stem!r}")
    state = class_of_stem(stem)
    if number == "SG":
        return stem
    if number == "PL":
        return stem + ("es" if state in ES_CLASSES else "s")
    raise ValueError(number)


def analyses(surface: str) -> list[str]:
    """Every (stem, number) pair whose FST output is this surface."""
    found = []
    for n in range(1, len(surface) + 1):
        stem = surface[:n]
        for number in ("SG", "PL"):
            if generate(stem, number) == surface:
                found.append(f"{stem}+N+{number}")
    return found


def question1() -> None:
    h("Q1  Morphological FST for English noun number")
    print("""
The transducer generates a surface noun from its lexical form.
Input:  the stem letters, then the tag +N, then +SG or +PL.
Output: the stem letters, then nothing for singular, or s/es for plural.

Ending classes remembered while the stem is read
  S X Z C H CH SH O
  CH is reached only by h after C, SH only by h after S.
  Any other letter goes to O, except s, x, z, c, h as above.

Letter arcs (every stem state, including the start state 0)
  input s -> S    input x -> X    input z -> Z    input c -> C
  input h -> CH if the state is C, SH if the state is S, else H
  any other letter L -> O
  output on a letter arc is the letter itself.

Feature arcs
  From class K, +N / ε  ->  N_K
  From N_K,  +SG / ε -> accept          (singular copies the stem)
  From N_K,  +PL / es -> accept   if K is S, X, Z, CH or SH
  From N_K,  +PL / s  -> accept   if K is C, H or O
""")
    print(f"{'From':<8}{'Input':<8}{'To':<8}Output")
    print("-" * 36)
    sources = ["0"] + STEM_STATES
    for src in sources:
        for letter, dest in (("s", "S"), ("x", "X"), ("z", "Z"), ("c", "C")):
            print(f"{src:<8}{letter:<8}{dest:<8}{letter}")
        for src_h, dest in (("C", "CH"), ("S", "SH")):
            if src == src_h:
                print(f"{src:<8}{'h':<8}{dest:<8}h")
        if src not in ("C", "S"):
            print(f"{src:<8}{'h':<8}{'H':<8}h")
        print(f"{src:<8}{'other':<8}{'O':<8}the letter")
    print()
    for k in STEM_STATES:
        print(f"{k:<8}{'+N':<8}{'N_' + k:<8}ε")
    for k in STEM_STATES:
        plural = "es" if k in ES_CLASSES else "s"
        print(f"{'N_' + k:<8}{'+SG':<8}{'accept':<8}ε")
        print(f"{'N_' + k:<8}{'+PL':<8}{'accept':<8}{plural}")

    print("\nGenerated forms")
    for stem in ("cat", "dog", "bus", "fox", "buzz", "box", "church", "dish", "picnic"):
        print(f"  {stem}+N+SG -> {generate(stem, 'SG'):<10}  {stem}+N+PL -> {generate(stem, 'PL')}")
    print("\nAnalyses of a surface string (no lexicon, so more than one is possible)")
    for surface in ("cats", "bus", "buses", "foxes", "foxs", "churches"):
        print(f"  {surface:<10} {', '.join(analyses(surface))}")
    print("\nfoxs is a legal singular (stem foxs) and not the plural of fox,")
    print("because a stem ending in x must take es, not s.")


# ---------------------------------------------------------------- Q2
SENTS = [
    "low lower low lower lowest".split(),
    "low lowest low lowest".split(),
    "low new".split(),
    "new new newer newer".split(),
    "new".split(),
]


def merges(corpus: Counter, steps: int, mode: str) -> list[tuple[str, str]]:
    """mode 'wordpiece' scores a pair by count(ab)/(count(a)*count(b));
    mode 'bpe' scores it by count(ab). Ties break toward the higher count,
    then the lexicographically smaller pair."""
    vocab = Counter()
    for word, n in corpus.items():
        vocab[tuple(word)] += n
    done = []
    for step in range(1, steps + 1):
        pair_count: Counter = Counter()
        sym_count: Counter = Counter()
        for symbols, n in vocab.items():
            for s in symbols:
                sym_count[s] += n
            for a, b in zip(symbols, symbols[1:]):
                pair_count[(a, b)] += n
        def score(pair):
            c = pair_count[pair]
            if mode == "bpe":
                return (c, c)
            a, b = pair
            return (c / (sym_count[a] * sym_count[b]), c)
        # Higher score, then higher count, then the alphabetically smaller pair.
        ranked = sorted(pair_count, key=lambda p: (-score(p)[0], -score(p)[1], "".join(p)))
        best = ranked[0]
        a, b = best
        sc = score(best)
        merged = a + b
        print(f"  merge {step}: {a!r} + {b!r} -> {merged!r}   "
              f"count={pair_count[best]}  score={sc[0]:.6f}")
        updated: Counter = Counter()
        for symbols, n in vocab.items():
            out = []
            i = 0
            while i < len(symbols):
                if i + 1 < len(symbols) and symbols[i] == a and symbols[i + 1] == b:
                    out.append(merged)
                    i += 2
                else:
                    out.append(symbols[i])
                    i += 1
            updated[tuple(out)] += n
        vocab = updated
        done.append((a, b))
        print("    vocabulary:", " ".join(
            f"{''.join(s)}/{n}" for s, n in sorted(vocab.items(), key=lambda kv: -kv[1])))
    return done


def apply_merges(word: str, pairs: list[tuple[str, str]]) -> list[str]:
    symbols = list(word)
    for a, b in pairs:
        out = []
        i = 0
        while i < len(symbols):
            if i + 1 < len(symbols) and symbols[i] == a and symbols[i + 1] == b:
                out.append(a + b)
                i += 2
            else:
                out.append(symbols[i])
                i += 1
        symbols = out
    return symbols


def question2() -> None:
    h("Q2  WordPiece, 2 merges, tokenization of 'newest'")
    corpus = Counter(w for s in SENTS for w in s)
    print("Word counts:", dict(corpus))
    print("\nWordPiece (pair score = count(a,b) / (count(a) * count(b)))")
    wp = merges(corpus, 2, "wordpiece")
    tokens = apply_merges("newest", wp)
    print("Tokenization of newest:", " ".join(tokens))
    print("\nFor comparison, BPE (pair score = raw count). Same tie-break.")
    bpe = merges(corpus, 2, "bpe")
    print("BPE tokenization of newest:", " ".join(apply_merges("newest", bpe)))
    print("\nThe question asks for WordPiece, so newest ->", " ".join(tokens))


# ---------------------------------------------------------------- Q3
DOCS = [
    ["cat", "dog", "animal"],
    ["cat", "cat", "pet"],
    ["dog", "pet", "animal"],
    ["cat", "dog", "pet"],
]


def question3() -> None:
    h("Q3  TF-IDF and the closest sentences")
    vocab = sorted({w for d in DOCS for w in d})
    df = {w: sum(w in d for d in DOCS) for w in vocab}
    n = len(DOCS)
    idf = {w: math.log10(n / df[w]) for w in vocab}
    print(f"N = {n} documents. Unsmoothed IDF = log10(N / df).")
    print(f"{'word':<10}{'df':>6}{'idf':>12}")
    for w in vocab:
        print(f"{w:<10}{df[w]:6d}{idf[w]:12.6f}")
    vectors = []
    print("\nRaw TF, then TF-IDF = TF * IDF")
    for i, d in enumerate(DOCS, 1):
        tf = Counter(d)
        vec = [tf[w] * idf[w] for w in vocab]
        vectors.append(vec)
        print(f"  d{i} TF   ", {w: tf[w] for w in vocab})
        print(f"  d{i} tfidf", " ".join(f"{w}={v:.4f}" for w, v in zip(vocab, vec)))
    print("\nCosine similarity")
    best = (0, 1, -1.0)
    for i in range(n):
        for j in range(i + 1, n):
            dot = sum(a * b for a, b in zip(vectors[i], vectors[j]))
            ni = math.sqrt(sum(a * a for a in vectors[i]))
            nj = math.sqrt(sum(b * b for b in vectors[j]))
            cos = dot / (ni * nj)
            print(f"  d{i+1} · d{j+1} = {cos:.6f}")
            if cos > best[2]:
                best = (i + 1, j + 1, cos)
    print(f"\nClosest sentences: d{best[0]} and d{best[1]} "
          f"({ ' '.join(DOCS[best[0]-1]) !r} and { ' '.join(DOCS[best[1]-1]) !r}), "
          f"cosine {best[2]:.6f}")


# ---------------------------------------------------------------- Q4
def question4() -> None:
    h("Q4  Naive Bayes spam filter, email E8 = (free=0, win=0, offer=0)")
    # rows: free, win, offer, spam?
    data = [
        (1, 1, 1, True),
        (1, 1, 0, True),
        (1, 0, 1, True),
        (0, 1, 1, True),
        (0, 0, 1, False),
        (0, 1, 0, False),
        (1, 0, 0, False),
    ]
    features = ("free", "win", "offer")
    for label, flag in (("Spam", True), ("NotSpam", False)):
        rows = [r for r in data if r[3] is flag]
        print(f"\n{label}: {len(rows)} of {len(data)} emails, "
              f"P({label}) = {len(rows)}/{len(data)}")
        for j, name in enumerate(features):
            ones = sum(r[j] for r in rows)
            zeros = len(rows) - ones
            print(f"  P({name}=1 | {label}) = {ones}/{len(rows)}    "
                  f"P({name}=0 | {label}) = {zeros}/{len(rows)}")
    print("\nE8 has every feature 0. No count above is zero, so MLE needs no smoothing.")
    p_spam = Fraction(4, 7) * Fraction(1, 4) * Fraction(1, 4) * Fraction(1, 4)
    p_not = Fraction(3, 7) * Fraction(2, 3) * Fraction(2, 3) * Fraction(2, 3)
    print(f"  P(E8 | Spam) P(Spam)       = (4/7)*(1/4)*(1/4)*(1/4) = {p_spam}")
    print(f"  P(E8 | NotSpam) P(NotSpam) = (3/7)*(2/3)*(2/3)*(2/3) = {p_not}")
    print(f"  ratio NotSpam / Spam = {p_not / p_spam}")
    pred = "NotSpam" if p_not > p_spam else "Spam"
    print(f"\nPrediction for E8: {pred}")


# ---------------------------------------------------------------- Q5
def question5() -> None:
    h("Q5  Kneser-Ney, d = 0.5, no sentence markers")
    sents = [
        "I like NLP".split(),
        "I like AI".split(),
        "You like NLP".split(),
        "You love AI".split(),
        "We love NLP".split(),
    ]
    uni = Counter()
    bi = Counter()
    for s in sents:
        uni.update(s)
        bi.update(zip(s, s[1:]))
    print("Unigrams:", dict(uni))
    print("Bigrams: ", dict(bi))
    followers = {}
    predecessors = {}
    for (h_, w), c in bi.items():
        followers.setdefault(h_, set()).add(w)
        predecessors.setdefault(w, set()).add(h_)
    types = len(bi)
    print(f"Distinct bigram types = {types}")
    d = Fraction(1, 2)

    def p_cont(word: str) -> Fraction:
        return Fraction(len(predecessors[word]), types)

    def pkn(word: str, hist: str) -> Fraction:
        c_h = sum(c for (h_, w), c in bi.items() if h_ == hist)
        c_hw = bi[(hist, word)]
        lam = d * len(followers[hist]) / c_h
        first = max(c_hw - d, 0) / c_h
        print(f"\nP( {word} | {hist} )")
        print(f"  c({hist}) = {c_h}, c({hist},{word}) = {c_hw}, "
              f"followers({hist}) = {sorted(followers[hist])}")
        print(f"  λ({hist}) = d * |followers| / c(h) = {lam}")
        print(f"  max(c-d, 0) / c(h) = {first}")
        print(f"  histories before {word}: {sorted(predecessors[word])}, "
              f"P_cont = {len(predecessors[word])}/{types} = {p_cont(word)}")
        total = first + lam * p_cont(word)
        print(f"  P_KN = {first} + {lam} * {p_cont(word)} = {total}")
        return total

    pkn("AI", "love")
    pkn("NLP", "like")


def main() -> None:
    import sys
    OUT.mkdir(parents=True, exist_ok=True)
    log = OUT / "midsem_solutions.txt"
    sink = open(log, "w", encoding="utf-8")
    sys.stdout = sink
    try:
        question1()
        question2()
        question3()
        question4()
        question5()
    finally:
        sys.stdout = sys.__stdout__
        sink.close()
    print(log.read_text())
    print(f"\nSaved {log}")


if __name__ == "__main__":
    main()
