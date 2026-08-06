"""
Custom sentence and word tokenizer for NLP Assignment 1 (AI357).

No nltk / spacy / regex-based-library shortcuts are used for the actual
splitting logic -- everything here is hand-written regex + string logic,
since that is the point of the assignment.

Handles:
    - URLs              (http://..., https://..., www...)
    - Email ids          (user@domain.tld)
    - Dates               (dd/mm/yyyy, dd-mm-yyyy, yyyy-mm-dd, dd.mm.yyyy)
    - Numbers with decimals and thousands separators (3.14, 1,23,456.78)
    - Punctuation as individual tokens
    - Devanagari script words (Hindi / Marathi / Sanskrit / etc.)
    - Gujarati script words
    - Latin script words
"""

import re

# ---------------------------------------------------------------------------
# 1. WORD-LEVEL TOKEN PATTERNS
#    Order matters: regex alternation is greedy left-to-right, so more
#    specific patterns (url, email, date) must come BEFORE the generic
#    number/word patterns, otherwise e.g. an email would get shredded by
#    the word pattern first.
# ---------------------------------------------------------------------------

URL_PATTERN = r'(?:https?://|www\.)[^\s]+'
EMAIL_PATTERN = r'[\w.\-]+@[\w\-]+\.[\w.\-]+'
# dd/mm/yyyy , dd-mm-yyyy , yyyy-mm-dd , dd.mm.yyyy  (also 2-digit years)
DATE_PATTERN = r'\d{1,4}[/\-.]\d{1,2}[/\-.]\d{1,4}'
# Indian-style (1,23,456.78) and western-style (123,456.78) numbers + plain decimals
NUMBER_PATTERN = r'\d+(?:,\d+)*(?:\.\d+)?'
# Latin words, allow internal apostrophes (don't, India's)
LATIN_WORD_PATTERN = r"[A-Za-z]+(?:'[A-Za-z]+)*"
# Devanagari block (Hindi, Marathi, Sanskrit, Nepali, Bodo, Maithili, Konkani, etc.)
# NOTE: excludes U+0964 (।) and U+0965 (॥) -- those are sentence-final
# punctuation (danda), not part of the word, so they must NOT be swallowed here.
DEVANAGARI_WORD_PATTERN = r'[\u0900-\u0963\u0966-\u097F]+'
# Gujarati block (U+0A80-U+0AFF). Gujarati doesn't have its own danda -- it
# borrows the Devanagari one (। / ॥), which lives outside this block, so no
# equivalent exclusion is needed here.
GUJARATI_WORD_PATTERN = r'[\u0A80-\u0AFF]+'
INDIC_RANGES = r'\u0900-\u0963\u0966-\u097F\u0A80-\u0AFF'  # used in punct/boundary lookahead
# Any other single non-space, non-word character -> punctuation / symbol token
PUNCT_PATTERN = r'[^\sA-Za-z0-9' + INDIC_RANGES + r']'

TOKEN_REGEX = re.compile(
    '|'.join([
        URL_PATTERN,
        EMAIL_PATTERN,
        DATE_PATTERN,
        NUMBER_PATTERN,
        LATIN_WORD_PATTERN,
        DEVANAGARI_WORD_PATTERN,
        GUJARATI_WORD_PATTERN,
        PUNCT_PATTERN,
    ])
)

# Patterns that must be "protected" before sentence splitting, because they
# can legitimately contain '.', '?', '!' that do NOT mark a sentence boundary
# (e.g. "www.iitb.ac.in", "3.14", "a.b@c.com").
PROTECT_REGEX = re.compile(
    '|'.join([URL_PATTERN, EMAIL_PATTERN, DATE_PATTERN, NUMBER_PATTERN])
)


def word_tokenize(text: str):
    """Tokenize a single sentence/string into a list of word-level tokens."""
    return TOKEN_REGEX.findall(text)


# ---------------------------------------------------------------------------
# 2. SENTENCE TOKENIZER
#    Strategy: mask out substrings that contain "fake" sentence-ending
#    punctuation (urls/emails/dates/decimal numbers), split on real sentence
#    boundaries, then unmask.
# ---------------------------------------------------------------------------

# A sentence boundary = one or more of . ! ? or the Devanagari danda (।)/
# double-danda (॥) followed by whitespace and then an uppercase Latin
# letter, a Devanagari/Gujarati letter, a quote, or end of string.
SENTENCE_BOUNDARY_REGEX = re.compile(
    r'(?<=[.!?\u0964\u0965])["\')\]]?\s+(?=[A-Z"\'' + INDIC_RANGES + r']|$)'
)


def sentence_tokenize(text: str):
    """Split a paragraph into a list of sentences (as raw strings)."""
    protected_spans = []

    def _mask(match):
        protected_spans.append(match.group(0))
        return f"\uE000{len(protected_spans) - 1}\uE001"  # private-use placeholder

    masked = PROTECT_REGEX.sub(_mask, text)

    raw_sentences = SENTENCE_BOUNDARY_REGEX.split(masked)

    def _unmask(s):
        return re.sub(
            r'\uE000(\d+)\uE001',
            lambda m: protected_spans[int(m.group(1))],
            s,
        )

    sentences = [_unmask(s).strip() for s in raw_sentences]
    return [s for s in sentences if s]


def tokenize_paragraph(paragraph: str):
    """Full pipeline: paragraph -> list[list[str]] (sentences of tokens)."""
    sentences = sentence_tokenize(paragraph)
    return [word_tokenize(s) for s in sentences]


if __name__ == "__main__":
    sample = (
        "Visit https://huggingface.co/datasets for more info. "
        "Contact us at support@nit-surat.ac.in or call today! "
        "The event is on 15/08/2026 and the budget is 1,23,456.78 rupees. "
        "Is this working correctly? નમસ્તે, આ એક ગુજરાતી વાક્ય છે। "
        "આ બીજું વાક્ય છે, જેમાં સંખ્યા 3.14 પણ છે।"
    )
    for i, sent in enumerate(tokenize_paragraph(sample), 1):
        print(f"S{i}:", sent)
