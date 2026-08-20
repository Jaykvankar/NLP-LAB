def load_nouns(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return set(line.strip().lower() for line in f if line.strip())


def generate(word, nouns):
    word = word.lower()

    # -------------------------
    # Rule 1: Singular
    # -------------------------
    if word in nouns:
        return f"{word}+N+SG"

    # -------------------------
    # Rule 2: S addition
    # -------------------------
    if word.endswith("s"):
        root = word[:-1]

        if root in nouns:
            # S-addition is NOT allowed for these endings
            if root.endswith(("s", "z", "x", "ch", "sh")):
                pass
            elif root.endswith("y") and len(root) >= 2:
                # consonant + y needs Y replacement
                if root[-2] not in "aeiou":
                    pass
                else:
                    return f"{root}+N+PL"
            else:
                return f"{root}+N+PL"

    # -------------------------
    # Rule 3: E insertion
    # -------------------------
    if word.endswith("es"):
        root = word[:-2]

        if root in nouns:
            if root.endswith(("s", "z", "x", "ch", "sh")):
                return f"{root}+N+PL"

    # -------------------------
    # Rule 4: Y replacement
    # -------------------------
    if word.endswith("ies"):
        root = word[:-3] + "y"

        if root in nouns:
            if len(root) >= 2 and root[-2] not in "aeiou":
                return f"{root}+N+PL"

    # -------------------------
    # Nothing matched
    # -------------------------
    return "Invalid Word"


nouns = load_nouns("brown_nouns.txt")

words = [
    "fox",
    "foxes",
    "foxs",
    "bag",
    "bags",
    "watch",
    "watches",
    "try",
    "tries"
]

for word in words:
    print(word, "=", generate(word, nouns))