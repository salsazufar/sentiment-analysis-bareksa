"""
Script untuk menyimpan tokenizer dan label_encoder dari training LSTM.

Script ini mengasumsikan bahwa model LSTM sudah dilatih dan disimpan sebagai
best_lstm_model.keras. Script ini akan:
1. Memuat data yang sama dengan training
2. Membuat tokenizer dan label_encoder dengan preprocessing yang sama
3. Menyimpan tokenizer dan label_encoder ke app/artifacts/

Catatan: Script ini harus dijalankan setelah model LSTM dilatih di notebook,
dengan menggunakan preprocessing yang sama (preprocess_text_lstm).
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer

from app.preprocess import preprocess_corpus_lstm


ROOT_DIR = Path(__file__).resolve().parent
CSV_PATH = ROOT_DIR / "ulasan_aplikasi.csv"
ARTIFACTS_DIR = ROOT_DIR / "app" / "artifacts"
TOKENIZER_PATH = ARTIFACTS_DIR / "tokenizer.pkl"
LABEL_ENCODER_PATH = ARTIFACTS_DIR / "label_encoder.pkl"

# Hyperparameters yang sama dengan training
MAX_WORDS = 10000
MAX_LEN = 100


def derive_sentiment(score: float) -> str:
    """
    Map numeric rating (1–5) to sentiment label.
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

    print("Preprocessing texts for LSTM (cleaning only, no stemming) ...")
    texts = df["content"].astype(str).tolist()
    labels = df["sentiment"].astype(str).tolist()

    # Preprocessing khusus untuk LSTM (hanya cleaning)
    texts_proc = preprocess_corpus_lstm(texts)

    print("Creating tokenizer ...")
    tokenizer = Tokenizer(num_words=MAX_WORDS)
    tokenizer.fit_on_texts(texts_proc)

    print("Creating label encoder ...")
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)

    print(f"Vocabulary size: {len(tokenizer.word_index)}")
    print(f"Number of classes: {len(label_encoder.classes_)}")
    print(f"Classes: {label_encoder.classes_}")

    # Save artifacts
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    
    with TOKENIZER_PATH.open("wb") as f:
        pickle.dump(tokenizer, f)
    print(f"Saved tokenizer to: {TOKENIZER_PATH}")

    with LABEL_ENCODER_PATH.open("wb") as f:
        pickle.dump(label_encoder, f)
    print(f"Saved label encoder to: {LABEL_ENCODER_PATH}")

    print("\nArtifacts saved successfully!")
    print("You can now use the LSTM model in the app.")


if __name__ == "__main__":
    main()
