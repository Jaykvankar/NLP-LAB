"""Notebook examples and full Brown corpus processing, using local paths."""
from pathlib import Path
from fst import format_analysis, words, print_transition_table, save_transition_table, generate_diagram


def main():
    print("Total nouns:", len(words))
    print("First 20 nouns:")
    for word in words[:20]:
        print(word)
    print()
    print_transition_table()

    # TEST EXAMPLES
    print("FST EXAMPLES")
    print("------------")
    examples = ["fox", "watch", "dish", "try", "fly", "bag", "dog", "child", "funds", "date"]
    for word in examples:
        print(format_analysis(word))
        print()

    # PROCESS BROWN CORPUS
    print("BROWN CORPUS FST OUTPUT")
    print("-----------------------")
    results = [format_analysis(word) for word in words]

    for result in results[:50]:
        print(result)
        print()

    # SAVE ALL RESULTS
    folder = Path(__file__).resolve().parent
    output_file = folder / "fst_output.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("FST MORPHOLOGICAL ANALYSIS\n")
        f.write("==========================\n\n")
        for result in results:
            f.write(result)
            f.write("\n\n")
    print("Output saved successfully:")
    print(output_file)

    save_transition_table(folder / "fst_transition_table.txt")
    generate_diagram()


if __name__ == "__main__":
    main()
