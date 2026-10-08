"""Preserve noun number tags from NLTK's original Brown corpus archive.

Source: https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/brown.zip
Run this script once after placing brown.zip beside it.
"""
import json
from pathlib import Path
from zipfile import ZipFile


def main():
    folder = Path(__file__).resolve().parent
    vocabulary = set((folder / "brown_nouns.txt").read_text(encoding="utf-8").split())
    numbers = {}
    tags = {"nn": "SG", "np": "SG", "nr": "SG",
            "nns": "PL", "nps": "PL", "nrs": "PL"}
    with ZipFile(folder / "brown.zip") as archive:
        for name in archive.namelist():
            basename = name.rsplit("/", 1)[-1]
            if len(basename) != 4 or not basename.startswith("c"):
                continue
            for token in archive.read(name).decode("utf-8").split():
                if "/" not in token:
                    continue
                word, tag = token.rsplit("/", 1)
                word = word.lower()
                # Remove headline/title markers, but keep possessives distinct.
                number = tags.get(tag.split("-")[0])
                if word in vocabulary and number:
                    numbers.setdefault(word, set()).add(number)
    output = {word: sorted(values) for word, values in sorted(numbers.items())}
    (folder / "brown_noun_numbers.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print("Saved noun number tags for", len(output), "entries")


if __name__ == "__main__":
    main()
