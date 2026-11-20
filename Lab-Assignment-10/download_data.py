"""Download the data used by Assignment_10_Solutions.ipynb into raw_data/.

Sources (mirrors of the resources named in the assignment PDF):
  * Amazon reviews: Hugging Face `fancyzhx/amazon_polarity`. This is the same
    Zhang et al. (2015) Amazon polarity corpus as Kaggle `kritanjalijain/amazon-reviews`
    (3.6M train / 400k test, labels negative/positive, title + review text).
    Kaggle's copy needs an API token, while the HF copy downloads directly.
  * Word2Vec: Hugging Face `fse/word2vec-google-news-300` (the link in the PDF).
  * GloVe: Hugging Face `fse/glove-wiki-gigaword-300` (GloVe 6B, 300d, the
    vectors in the Kaggle GloVe dataset) in gensim KeyedVectors format.
  * FastText: Hugging Face `fse/fasttext-wiki-news-subwords-300`, which is
    wiki-news-300d-1M-subword.vec from fasttext.cc saved in gensim format.
  * Analogies: questions-words.txt from github.com/nicholas-leonard/word2vec.

The gensim-format (.model + .vectors.npy) files can be memory-mapped, so even
the 3.6 GB Word2Vec matrix does not have to fit in RAM.

Usage:  python download_data.py
"""
from pathlib import Path
import shutil
import urllib.request

ROOT = Path(__file__).resolve().parent / "raw_data"
HF = "https://huggingface.co"

FILES = {
    "amazon/questions-words.txt":
        "https://raw.githubusercontent.com/nicholas-leonard/word2vec/master/questions-words.txt",
    "amazon/test-00000-of-00001.parquet":
        f"{HF}/datasets/fancyzhx/amazon_polarity/resolve/main/amazon_polarity/test-00000-of-00001.parquet",
}
for i in range(4):
    name = f"train-0000{i}-of-00004.parquet"
    FILES[f"amazon/{name}"] = f"{HF}/datasets/fancyzhx/amazon_polarity/resolve/main/amazon_polarity/{name}"
for model in ["glove-wiki-gigaword-300", "fasttext-wiki-news-subwords-300", "word2vec-google-news-300"]:
    for suffix in [".model", ".model.vectors.npy"]:
        FILES[f"embeddings/{model}{suffix}"] = f"{HF}/fse/{model}/resolve/main/{model}{suffix}"


def main():
    for rel, url in FILES.items():
        dest = ROOT / rel
        if dest.exists() and dest.stat().st_size > 0:
            print(f"exists  {rel}")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        print(f"fetch   {rel}")
        with urllib.request.urlopen(url) as response, open(tmp, "wb") as out:
            shutil.copyfileobj(response, out, length=1 << 20)
        tmp.rename(dest)


if __name__ == "__main__":
    main()
