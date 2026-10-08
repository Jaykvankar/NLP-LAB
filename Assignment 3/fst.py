"""Notebook FST rules and conceptual transition table, with a diagram."""
from pathlib import Path
import json

file_path = Path(__file__).resolve().with_name("brown_nouns.txt")
with open(file_path, "r", encoding="utf-8") as f:
    words = [line.strip().lower() for line in f if line.strip()]
words = list(dict.fromkeys(words))
noun_numbers = json.loads(
    file_path.with_name("brown_noun_numbers.json").read_text(encoding="utf-8")
)

transition_table = [
    ("q0", "a-z", "same", "q1"),
    ("q1", "s", "same", "q_es"),
    ("q1", "z", "same", "q_es"),
    ("q1", "x", "same", "q_es"),
    ("q1", "ch", "same", "q_es"),
    ("q1", "sh", "same", "q_es"),
    ("q1", "consonant+y", "y->i", "q_y"),
    ("q1", "other", "same", "q_s"),
    ("q_es", "end", "epsilon+es", "qf"),
    ("q_y", "end", "epsilon+es", "qf"),
    ("q_s", "end", "epsilon+s", "qf")
]


def generate_plural(word):
    if not word.isalpha() or not word.islower():
        return None, "Invalid"

    # E INSERTION
    if word.endswith("ch"):
        return word + "es", "E insertion"
    if word.endswith("sh"):
        return word + "es", "E insertion"
    if word.endswith("x"):
        return word + "es", "E insertion"
    if word.endswith("z"):
        return word + "es", "E insertion"
    if word.endswith("s"):
        return word + "es", "E insertion"

    # Y REPLACEMENT
    if word.endswith("y") and len(word) > 1:
        previous = word[-2]
        if previous not in "aeiou":
            return word[:-1] + "ies", "Y replacement"

    # S ADDITION
    return word + "s", "S addition"


def fst(word):
    """Apply spelling rules without requiring either form in the corpus."""
    numbers = noun_numbers.get(word, [])
    if numbers == ["PL"]:
        return word, "Already plural (Brown tag); no suffix added"
    if "PL" in numbers and "SG" in numbers:
        return word, "Singular or plural in Brown; context needed"
    plural, rule = generate_plural(word)
    if plural is None:
        return "Invalid Word", "Invalid"
    return plural, rule


def format_analysis(word):
    """Show corpus number tags before applying singular-to-plural rules."""
    form, rule = fst(word)
    numbers = noun_numbers.get(word, [])
    if "PL" in numbers:
        annotation = "SG/PL" if "SG" in numbers else "PL"
        return f"{word} = {word}+N+{annotation}\nRule: {rule}"
    if form == "Invalid Word":
        return f"{word}\nUnsupported input for spelling rules"
    if not numbers:
        return (f"{word}: number unavailable; assuming singular\n"
                f"Generated form: {form}\nRule: {rule}")
    return (f"{word} = {word}+N+SG\n"
            f"{form} = {word}+N+PL\nRule: {rule}")


def print_transition_table():
    print("FST TRANSITION TABLE")
    print("State | Input | Output | Next State")
    for state, inp, output, next_state in transition_table:
        print(state, "|", inp, "|", output, "|", next_state)
    print()


def save_transition_table(filename):
    with open(filename, "w", encoding="utf-8") as f:
        f.write("NOTEBOOK FST - CONCEPTUAL TRANSITION TABLE\n")
        f.write("State | Input | Output | Next State\n")
        for row in transition_table:
            f.write(" | ".join(row) + "\n")


def generate_diagram():
    """Draw the notebook table, grouping its five E-insertion edges."""
    import json
    import subprocess

    lines = ["digraph FST {", "rankdir=LR;",
             'graph [label="Notebook FST - conceptual rule diagram", labelloc=t, fontname="Arial", fontsize=18, ranksep=0.8];',
             'node [shape=circle, fontname="Arial", fontsize=15];',
             'edge [fontname="Arial", fontsize=12];',
             'entry [shape=point];', 'qf [shape=doublecircle];',
             'entry -> q0;']
    grouped = {}
    for state, inp, output, next_state in transition_table:
        grouped.setdefault((state, next_state, output), []).append(inp)
    for (state, next_state, output), inputs in grouped.items():
        label = ", ".join(inputs) + " : " + output
        lines.append(f"{state} -> {next_state} [label={json.dumps(label)}];")
    lines.append("}")
    filename = Path(__file__).resolve().with_name("brown_fst_diagram.gv")
    filename.write_text("\n".join(lines), encoding="utf-8")
    try:
        subprocess.run(["dot", "-Tpng", str(filename), "-o", str(filename.with_suffix(".png"))], check=True)
        print(f"FST image: {filename.with_suffix('.png')}")
    except FileNotFoundError:
        print("Install Graphviz and add dot to PATH to generate the image.")


if __name__ == "__main__":
    print_transition_table()
    generate_diagram()
