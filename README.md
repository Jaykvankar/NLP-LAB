# NLP-LAB

Lab assignments for **AI357 Natural Language Processing** at SVNIT Surat (B.Tech, Semester 5, Monsoon 2026).

## Assignments

| # | Folder | Topic |
|---|---|---|
| 1 | `Assignment 1` | Regex-based tokenizer and corpus preprocessing |
| 2 | `Assignment 2` | Text segmentation using word probabilities |
| 3 | `Assignment 3` | DFA and FST for English noun plurals (Brown corpus) |
| 4 | `Assignment 4` | N-gram language models and Laplace smoothing |
| 5 | `Assignment 5` | Add-K smoothing |
| 6 | `Assignment 6` | Good-Turing, Katz backoff, Kneser-Ney, interpolation, stupid backoff |
| 7 | `Lab-Assignment-7` | Hindi POS tagging: Naive Bayes and HMM |
| 8 | `Lab-Assignment-8` | Midsem solutions and noisy-channel spell checker |
| 9 | `Lab-Assignment-9` | Hindi POS tagging: MEMM and CRF |
| 10 | `Lab-Assignment-10` | Word embeddings: Word2Vec, GloVe, FastText |

---

### 1. Tokenizer and preprocessing
- Sentence and word tokenizers built with regular expressions only (no NLTK or spaCy).
- Handles URLs, emails, dates, numbers, abbreviations, and English, Hindi and Gujarati script.
- Cleans IndicCorpV2 and OSCAR-2301 text and reports corpus statistics: type-token ratio, average sentence length and average word length.

### 2. Text segmentation
- Splits text written without spaces into words.
- Uses unigram word probabilities (in log space) and chooses the most probable split.

### 3. Finite automata
- **DFA** that accepts lowercase English words.
- **FST** that generates plural forms of nouns from the Brown corpus: `fox → foxes`, `try → tries`, `dog → dogs`.
- Prints the transition table and draws the automaton with Graphviz.

### 4. N-gram language models
- Unigram, bigram, trigram and 4-gram models trained on 1 million sentences (998k train / 1k dev / 1k test).
- **Q1:** unsmoothed models. Unseen n-grams get probability 0, so perplexity becomes infinite.
- **Q2:** Laplace (add-one) smoothing removes the zero probabilities.

### 5. Add-K smoothing
- The same four models with Add-K smoothing (k = 0.3), compared by perplexity.

### 6. Advanced smoothing
- The same models with Good-Turing, Katz backoff, Kneser-Ney, linear interpolation and stupid backoff, compared by perplexity on the dev and test sets.

### 7. Hindi POS tagging: Naive Bayes and HMM
- **Naive Bayes** with five features: word length (`MORE`/`LESS`), current word, previous word, next word and previous tag.
- **HMM** with three decoders: forward Viterbi, backward Viterbi and forward-backward.
- **Result (test):** forward-backward is best, with 92.3% accuracy and 0.866 macro F1. Naive Bayes gets 0.760 macro F1.

### 8. Midsem and spell checker
- Programs for all five midsem questions: noun FST, WordPiece merges, TF-IDF with cosine similarity, Naive Bayes spam filter, and Kneser-Ney smoothing.
- **Spell checker** using the noisy channel model: `correct word = argmax P(typo | word) × P(word)`.
  - Candidates are dictionary words one edit away from the typo.
  - The edit probability comes from `count_1edit.txt` and the word frequency from `count_unigram.txt`.
  - Accuracy: **66.88%**.

### 9. Hindi POS tagging: MEMM and CRF
- **MEMM:** a logistic regression classifier for each word, using the previous tag as a feature.
- **CRF:** `sklearn-crfsuite`, scoring the whole tag sequence at once.
- **Features:** prefixes and suffixes of length 1–4, previous/current/next word, and word length.
- **Result (test macro F1):** CRF 0.929 > MEMM 0.927 > HMM 0.861 > Naive Bayes 0.760.

### 10. Word embeddings
- Amazon reviews (10k train / 1k test, stratified sampling) are lowercased, cleaned, tokenized and stopword-filtered. `not`, `no` and `nor` are kept because they flip sentiment.
- **Analogies** on `questions-words.txt`: FastText 87.4%, Word2Vec 73.6%, GloVe 72.0%.
- **Sentiment classifier:** a neural network with one hidden layer, trained on (a) the mean of the word vectors and (b) a TF-IDF-weighted mean.
- **Result (macro F1):** FastText 0.835 ≈ Word2Vec 0.834 > GloVe 0.809. The plain mean beats TF-IDF weighting for all three embeddings.

---

## How to run
Requires Python 3.11.

```bash
pip install numpy pandas scikit-learn nltk matplotlib sklearn-crfsuite gensim pyarrow
```

- **Python scripts:** run from inside their folder, e.g. `cd Lab-Assignment-7` and then `python pos_tagger.py`.
- **Notebooks (`.ipynb`):** open in Jupyter or VS Code and run all cells.
- **Assignments 4–6** read the tokenized corpus produced by Assignment 1, so run Assignment 1 first.

## Data (not in this repository)
Datasets and large generated files are listed in `.gitignore` and are not pushed. Before running an assignment, put its data in the folder shown:

| Assignment | Data | Where to put it |
|---|---|---|
| 1, 4–6 | IndicCorpV2 / OSCAR-2301 text | `Assignment 1/raw_data/` |
| 7, 9 | Hindi POS train/dev/test files | `Hindi-POS-Data/` inside the assignment folder |
| 8 | Spell-correction files | `Lab-Assignment-8/Spell-Correction/` |
| 10 | Amazon reviews and the three embedding models | Run `python download_data.py` (about 6.7 GB) |
