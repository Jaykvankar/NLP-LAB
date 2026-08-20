from automathon import DFA
import os

# Lowercase letters
letters = set("abcdefghijklmnopqrstuvwxyz")

# States
states = {"q0", "q1"}

# Transitions
transitions = {
    "q0": {letter: "q1" for letter in letters},
    "q1": {letter: "q1" for letter in letters}
}

# Create DFA
dfa = DFA(
    states,
    letters,
    transitions,
    "q0",
    {"q1"}
)

# List of test words from the image
test_words = [
    # Accepted examples
    "cat", "dog", "a", "zebra",
    # Not Accepted examples
    "dog1", "1dog", "DogHouse", "Dog_house", " cats"
]

# Run each test word through the DFA
for word in test_words:
    if dfa.accept(word):
        print(f"'{word}': Accepted")
    else:
        print(f"'{word}': Not Accepted")

# Create large DFA image
dfa.view(
    "english_dfa",
    node_attr={"fontsize": "30"},
    edge_attr={"fontsize": "25"}
)

# Delete the .gv file
if os.path.exists("english_dfa.gv"):
    os.remove("english_dfa.gv")