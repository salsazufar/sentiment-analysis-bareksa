"""
Text preprocessing utilities for Bareksa review sentiment analysis.

The original notebook performs several steps:
- Lowercasing (casefolding)
- Removing non-alphabetic characters
- Tokenization
- Stopword removal (NLTK + Sastrawi)
- Stemming (Sastrawi)

Here we re-implement a compact but compatible pipeline
to be reused both for training scripts and inference.
"""

from __future__ import annotations

import re
from typing import Iterable, List

import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


_NON_ALPHA_RE = re.compile(r"[^a-zA-Z\s]")


def _ensure_nltk_resources() -> None:
    """
    Ensure required NLTK resources are available.

    This is safe to call multiple times.
    """

    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")

    # Newer NLTK versions also require `punkt_tab` for sentence tokenization.
    # Without it, `nltk.word_tokenize` may fail with "Resource punkt_tab not found".
    try:
        nltk.data.find("tokenizers/punkt_tab/english")
    except LookupError:
        nltk.download("punkt_tab")

    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")


def _get_indonesian_stopwords() -> set[str]:
    """
    Combine NLTK and Sastrawi-style Indonesian stopwords.

    We keep this simple to avoid coupling too much to the notebook implementation,
    but still capture the main idea.
    """

    from nltk.corpus import stopwords

    _ensure_nltk_resources()
    # NLTK has Indonesian stopwords under language code "indonesian"
    stop_id = set(stopwords.words("indonesian"))
    # We could extend this with custom/slang stopwords if needed.
    return stop_id


_STEMMER = StemmerFactory().create_stemmer()
_STOPWORDS_ID = _get_indonesian_stopwords()


def clean_text(text: str) -> str:
    """
    Basic cleaning: lowercase, remove non-alphabetic characters, normalize spaces.
    """

    if not isinstance(text, str):
        text = str(text)

    text = text.lower()
    text = _NON_ALPHA_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_and_stem_id(text: str) -> List[str]:
    """
    Tokenize Indonesian text, remove stopwords, and apply stemming.
    """

    _ensure_nltk_resources()
    tokens = nltk.word_tokenize(text)
    processed: List[str] = []
    for tok in tokens:
        tok = tok.strip()
        if not tok:
            continue
        if tok in _STOPWORDS_ID:
            continue
        stemmed = _STEMMER.stem(tok)
        if stemmed:
            processed.append(stemmed)
    return processed


def preprocess_text(text: str) -> str:
    """
    Full preprocessing pipeline for a single review.

    Returns a space-joined string of processed tokens, suitable for
    feeding into a bag-of-words / TF-IDF model.
    """

    cleaned = clean_text(text)
    tokens = tokenize_and_stem_id(cleaned)
    return " ".join(tokens)


def preprocess_corpus(texts: Iterable[str]) -> List[str]:
    """
    Apply `preprocess_text` to an iterable of texts.
    """

    return [preprocess_text(t) for t in texts]


def preprocess_text_lstm(text: str) -> str:
    """
    Preprocessing khusus untuk LSTM model.
    
    Sesuai dengan preprocessing di notebook untuk training LSTM:
    - Cleaning (remove @, #, RT, URLs, numbers, punctuation)
    - Casefolding (lowercase)
    - Tokenization
    - Stopword removal (Indonesian + English)
    - Join back to sentence
    
    Catatan: Tidak menggunakan stemming karena LSTM dapat mempelajari konteks
    dari kata-kata asli.
    
    Returns:
        Preprocessed text matching training pipeline
    """
    import re
    import string
    
    if not isinstance(text, str):
        text = str(text)
    
    # Step 1: Cleaning (matching notebook's cleaningText function)
    text = re.sub(r'@[A-Za-z0-9]+', '', text)
    text = re.sub(r'#[A-Za-z0-9]+', '', text)
    text = re.sub(r'RT[\s]+', '', text)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'[0-9]+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.replace('\n', ' ')
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip(' ')
    
    # Step 2: Casefolding
    text = text.lower()
    
    # Step 3: Tokenization
    _ensure_nltk_resources()
    tokens = nltk.word_tokenize(text)
    
    # Step 4: Stopword removal (matching notebook's filteringText)
    # Combine Indonesian and English stopwords + custom stopwords
    stopwords_id = _get_indonesian_stopwords()
    try:
        from nltk.corpus import stopwords
        stopwords_en = set(stopwords.words('english'))
        stopwords_id.update(stopwords_en)
    except LookupError:
        pass
    
    # Add custom stopwords from notebook
    custom_stopwords = {'iya', 'yaa', 'gak', 'nya', 'na', 'sih', 'ku', 'di', 'ga', 'ya', 
                        'gaa', 'loh', 'kah', 'woi', 'woii', 'woy'}
    stopwords_id.update(custom_stopwords)
    
    # Filter stopwords
    filtered_tokens = [tok for tok in tokens if tok not in stopwords_id]
    
    # Step 5: Join back to sentence
    return ' '.join(filtered_tokens)


def preprocess_corpus_lstm(texts: Iterable[str]) -> List[str]:
    """
    Apply `preprocess_text_lstm` to an iterable of texts.
    """
    
    return [preprocess_text_lstm(t) for t in texts]

