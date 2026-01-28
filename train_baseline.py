"""
Train a simple TF-IDF + Logistic Regression baseline model for Bareksa reviews.

This script:
- Loads `ulasan_aplikasi.csv`
- Derives sentiment labels from the numeric score (1–5)
- Applies the preprocessing pipeline from `app.preprocess`
- Trains a TF-IDF vectorizer + Logistic Regression classifier
- Saves artifacts under `app/artifacts/`
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from app.preprocess import preprocess_corpus


ROOT_DIR = Path(__file__).resolve().parent
CSV_PATH = ROOT_DIR / "ulasan_aplikasi.csv"
ARTIFACTS_DIR = ROOT_DIR / "app" / "artifacts"
VECTORIZER_PATH = ARTIFACTS_DIR / "tfidf_vectorizer.pkl"
MODEL_PATH = ARTIFACTS_DIR / "logreg_sentiment.pkl"


def derive_sentiment(score: float) -> str:
    """
    Map numeric rating (1–5) to sentiment label.
    Mirrors the common convention used in many app review analyses.
    """

    if score >= 4:
        return "positif"
    if score == 3:
        return "netral"
    return "negatif"


def main() -> None:
    if not CSV_PATH.exists():
        raise SystemExit(f"Data file not found: {CSV_PATH}")

    print(f"Loading data from {CSV_PATH} ...")
    df = pd.read_csv(CSV_PATH)

    if "content" not in df.columns or "score" not in df.columns:
        raise SystemExit("CSV must contain 'content' and 'score' columns.")

    df = df[["content", "score"]].dropna()
    df["sentiment"] = df["score"].apply(derive_sentiment)

    print("Preprocessing texts ...")
    texts = df["content"].astype(str).tolist()
    labels = df["sentiment"].astype(str).tolist()

    texts_proc = preprocess_corpus(texts)

    X_train, X_test, y_train, y_test = train_test_split(
        texts_proc,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    print("Vectorizing with TF-IDF ...")
    vectorizer = TfidfVectorizer(
        max_features=2000,
        ngram_range=(1, 2),
        min_df=2,
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print("Training Logistic Regression ...")
    clf = LogisticRegression(max_iter=1000, n_jobs=-1)
    clf.fit(X_train_tfidf, y_train)

    print("Evaluating ...")
    y_pred = clf.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with VECTORIZER_PATH.open("wb") as f:
        pickle.dump(vectorizer, f)
    with MODEL_PATH.open("wb") as f:
        pickle.dump(clf, f)

    print("Saved artifacts:")
    print(" -", VECTORIZER_PATH)
    print(" -", MODEL_PATH)


if __name__ == "__main__":
    main()

