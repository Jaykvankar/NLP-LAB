"""Question 1: accept one or more lowercase English letters."""
from string import ascii_lowercase


def dfa(word):
    state = "start"
    for letter in word:
        if state == "reject" or letter not in ascii_lowercase:
            state = "reject"
        else:
            state = "accept"
    return "Accepted" if state == "accept" else "Not Accepted"


def generate_diagram():
    """Use the installed Graphviz program to render the DFA."""
    from pathlib import Path
    import subprocess

    filename = Path(__file__).resolve().with_name("english_dfa.gv")
    filename.write_text('''digraph DFA {
        rankdir=LR;
        node [shape=circle];
        entry [shape=point];
        accept [shape=doublecircle];
        entry -> start;
        start -> accept [label="a-z"];
        accept -> accept [label="a-z"];
        start -> reject [label="other"];
        accept -> reject [label="other"];
        reject -> reject [label="any character"];
    }''', encoding="utf-8")
    try:
        subprocess.run(["dot", "-Tpng", str(filename), "-o", str(filename) + ".png"], check=True)
        print(f"DFA image: {filename}.png")
    except FileNotFoundError:
        print(f"Install Graphviz and add dot to PATH to render {filename.name}.")


if __name__ == "__main__":
    examples = ["cat", "dog", "a", "zebra", "dog1", "1dog",
                "DogHouse", "Dog_house", " cats", ""]
    for word in examples:
        print(f"{word!r} -> {dfa(word)}")

    generate_diagram()
