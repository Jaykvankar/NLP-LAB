import json
import os
import re
import glob
from typing import List
import pandas as pd


# =====================================
# INPUT / OUTPUT FOLDERS
# =====================================

INPUT_FOLDER = "raw_data"
OUTPUT_FOLDER = "tokenized_output"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# =====================================
# REGEX PATTERNS & TAG MAPPINGS
# =====================================

# URLs, emails, dates, and numbers
URL_PATTERN = r'(?:https?://|www\.)\S+'
EMAIL_PATTERN = r'[\w.\-+]+@[\w\-]+\.[\w.\-]+'
DATE_PATTERN = r'[\d\u0AE6-\u0AEF]{1,4}[/\-.][\d\u0AE6-\u0AEF]{1,2}[/\-.][\d\u0AE6-\u0AEF]{1,4}'
NUMBER_PATTERN = r'[\d\u0AE6-\u0AEF]+(?:,[\d\u0AE6-\u0AEF]+)*(?:\.[\d\u0AE6-\u0AEF]+)?'

# Abbreviations
ABBREVIATIONS = r'(?:[A-Za-z]\.){2,}|(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|eg|ie)\.|(?:તા|ડૉ|પ્રો|સ્વ|જિ|લિ)\.'

# Word scripts
LATIN_WORD_PATTERN = r"[A-Za-z]+(?:'[A-Za-z]+)*"
DEVANAGARI_WORD_PATTERN = r'[\u0900-\u0963\u0966-\u097F]+'
GUJARATI_WORD_PATTERN = r'[\u0A80-\u0AFF]+'
GENERAL_INDIC_PATTERN = r'[\u0980-\u0D7F]+'

# Punctuations
INDIC_RANGES = r'\u0900-\u0963\u0966-\u097F\u0A80-\u0AFF\u0980-\u0D7F'
PUNCT_PATTERN = r'[^\sA-Za-z0-9' + INDIC_RANGES + r']'

# Masking pattern for internal dots
PROTECT_REGEX = re.compile(
    f'{URL_PATTERN}|{EMAIL_PATTERN}|{DATE_PATTERN}|{NUMBER_PATTERN}|{ABBREVIATIONS}'
)

# Sentence boundaries
SENTENCE_BOUNDARY_REGEX = re.compile(
    rf'(?<=[.!?\u0964\u0965])["\')\]]?\s+(?=[A-Z"\'' + INDIC_RANGES + r']|$)'
)

# Named token regex
TOKEN_REGEX = re.compile(
    rf"""
    (?P<URL>{URL_PATTERN})
    |(?P<EMAIL>{EMAIL_PATTERN})
    |(?P<DATE>{DATE_PATTERN})
    |(?P<NUMBER>{NUMBER_PATTERN})
    |(?P<WORD>{LATIN_WORD_PATTERN})
    |(?P<WORD_DEV>{DEVANAGARI_WORD_PATTERN})
    |(?P<WORD_GUJ>{GUJARATI_WORD_PATTERN})
    |(?P<WORD_INDIC>{GENERAL_INDIC_PATTERN})
    |(?P<PUNCT>{PUNCT_PATTERN})
    """,
    re.VERBOSE,
)

# Mapping match groups to tags
GROUP_TO_TAG = {
    "URL": "URL",
    "EMAIL": "EMAIL",
    "DATE": "DATE",
    "NUMBER": "NUMBER",
    "WORD": "WORD",
    "WORD_DEV": "WORD",
    "WORD_GUJ": "WORD",
    "WORD_INDIC": "WORD",
    "PUNCT": "PUNCT",
}


# =====================================
# TOKENIZATION FUNCTIONS
# =====================================

def sentence_tokenize(text: str) -> List[str]:
    """Splits text into sentences using masking."""

    text = text.replace("\n", " ")

    protected_spans = []

    def _mask(match: re.Match) -> str:
        protected_spans.append(match.group(0))
        return f"\uE000{len(protected_spans) - 1}\uE001"

    masked_text = PROTECT_REGEX.sub(_mask, text)

    raw_sentences = SENTENCE_BOUNDARY_REGEX.split(masked_text)

    def _unmask(sentence: str) -> str:
        return re.sub(
            r'\uE000(\d+)\uE001',
            lambda m: protected_spans[int(m.group(1))],
            sentence,
        )

    sentences = [_unmask(s).strip() for s in raw_sentences]

    return [s for s in sentences if s]


def tagged_word_tokenize(sentence: str) -> List[str]:
    """Extracts tokens and returns formatted token/TAG strings."""

    tagged_tokens = []

    for match in TOKEN_REGEX.finditer(sentence):
        token_str = match.group()
        tag = GROUP_TO_TAG[match.lastgroup]
        tagged_tokens.append(f"{token_str}/{tag}")

    return tagged_tokens


# =====================================
# STATISTICS VARIABLES
# =====================================

total_sentences = 0
total_words = 0
total_characters = 0

unique_tokens = set()

tokenized_sentences = []
all_tokens = []


# =====================================
# PROCESS FILES
# =====================================

files = glob.glob(os.path.join(INPUT_FOLDER, "*.txt"))

if len(files) == 0:
    print("No txt files found inside raw_data folder.")
    exit()


for file in files:

    print(f"Processing: {file}")

    with open(file, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    total_raw_characters = len(text)

    sentences = sentence_tokenize(text)

    output_file = os.path.join(
        OUTPUT_FOLDER,
        os.path.basename(file)
    )

    # Save tokenized text file
    with open(output_file, "w", encoding="utf-8") as out:

        for sentence in sentences:

            tokens = tagged_word_tokenize(sentence)

            if len(tokens) == 0:
                continue

            total_sentences += 1
            total_words += len(tokens)

            for token in tokens:
                token_without_tag = token.rsplit("/", 1)[0]

                total_characters += len(token_without_tag)
                unique_tokens.add(token_without_tag)
                all_tokens.append(token)

            tokenized_sentence = " ".join(tokens)

            tokenized_sentences.append(tokenized_sentence)

            out.write(tokenized_sentence)
            out.write("\n")

    print(f"Saved tokenized file: {output_file}")


# =====================================
# CORPUS STATISTICS
# =====================================

if total_sentences > 0:
    avg_sentence_length = total_words / total_sentences
else:
    avg_sentence_length = 0

if total_words > 0:
    avg_word_length = total_characters / total_words
else:
    avg_word_length = 0

if total_words > 0:
    ttr = len(unique_tokens) / total_words
else:
    ttr = 0


# =====================================
# PRINT RESULTS
# =====================================

print("=" * 45)
print("Corpus Statistics")
print("=" * 45)

print("Total Sentences :", total_sentences)
print("Total Words     :", total_words)
print("Total Characters:", total_characters)
print("Average Sentence Length :", round(avg_sentence_length, 2))
print("Average Word Length     :", round(avg_word_length, 2))
print("Unique Tokens :", len(unique_tokens))
print("Type Token Ratio (TTR):", round(ttr, 4))


# =====================================
# SAVE STATISTICS
# =====================================

stats_file = os.path.join(
    OUTPUT_FOLDER,
    "corpus_statistics.txt"
)

with open(stats_file, "w", encoding="utf-8") as f:

    f.write("Corpus Statistics\n")
    f.write("=" * 30 + "\n\n")

    f.write(f"Total Sentences : {total_sentences}\n")
    f.write(f"Total Words : {total_words}\n")
    f.write(f"Total Characters : {total_characters}\n")
    f.write(f"Average Sentence Length : {avg_sentence_length:.2f}\n")
    f.write(f"Average Word Length : {avg_word_length:.2f}\n")
    f.write(f"Unique Tokens : {len(unique_tokens)}\n")
    f.write(f"Type Token Ratio (TTR) : {ttr:.4f}\n")

print("\nTokenized files saved inside:", OUTPUT_FOLDER)
print("Statistics saved as corpus_statistics.txt")


# =====================================
# SAVE PARQUET FILE
# =====================================

parquet_file = os.path.join(
    OUTPUT_FOLDER,
    "tokenized_corpus.parquet"
)

df = pd.DataFrame({
    "tokenized_sentence": tokenized_sentences
})

df.to_parquet(
    parquet_file,
    engine="pyarrow",
    compression="snappy"
)

print("Parquet file saved as:", parquet_file)


# =====================================
# SAVE JSON STATISTICS
# =====================================

json_file = os.path.join(
    OUTPUT_FOLDER,
    "corpus_statistics.json"
)

stats = {
    "total_sentences": total_sentences,
    "total_words": total_words,
    "total_characters": total_characters,
    "avg_sentence_length": round(avg_sentence_length, 2),
    "avg_word_length": round(avg_word_length, 2),
    "unique_tokens": len(unique_tokens),
    "type_token_ratio": round(ttr, 6)
}

with open(json_file, "w", encoding="utf-8") as f:
    json.dump(stats, f, indent=4, ensure_ascii=False)

print("JSON statistics saved as:", json_file)


# =====================================
# SHOW SAMPLE TOKENS
# =====================================

print("\nSample Tokens:\n")

with open(files[0], "r", encoding="utf-8", errors="ignore") as f:
    sample = f.read()

sample_sentences = sentence_tokenize(sample)

for sentence in sample_sentences[:5]:
    print(tagged_word_tokenize(sentence))
