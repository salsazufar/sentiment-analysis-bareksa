"""
Script untuk melatih model LSTM untuk analisis sentimen.

Script ini menggunakan preprocessing yang sama dengan yang digunakan di inference,
memastikan konsistensi antara training dan inference.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import (
    Bidirectional,
    Dense,
    Dropout,
    Embedding,
    LSTM,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from app.preprocess import preprocess_corpus_lstm

# Paths
ROOT_DIR = Path(__file__).resolve().parent
CSV_PATH = ROOT_DIR / "ulasan_aplikasi.csv"
ARTIFACTS_DIR = ROOT_DIR / "app" / "artifacts"
MODEL_PATH = ROOT_DIR / "best_lstm_model.keras"
TOKENIZER_PATH = ARTIFACTS_DIR / "tokenizer.pkl"
LABEL_ENCODER_PATH = ARTIFACTS_DIR / "label_encoder.pkl"

# Hyperparameters (sama dengan notebook)
MAX_WORDS = 10000
MAX_LEN = 100
EMBEDDING_DIM = 200
LSTM_UNITS = 128
BATCH_SIZE = 32
EPOCHS = 20
VALIDATION_SPLIT = 0.2
RANDOM_STATE = 42


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
    print("=" * 70)
    print("TRAINING LSTM MODEL FOR SENTIMENT ANALYSIS")
    print("=" * 70)

    # 1. Load data
    if not CSV_PATH.exists():
        raise SystemExit(f"Data file not found: {CSV_PATH}")

    print(f"\n1. Loading data from {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)

    if "content" not in df.columns or "score" not in df.columns:
        raise SystemExit("CSV must contain 'content' and 'score' columns.")

    df = df[["content", "score"]].dropna()
    df["sentiment"] = df["score"].apply(derive_sentiment)

    print(f"   Loaded {len(df)} reviews")
    print(f"   Sentiment distribution:")
    print(df["sentiment"].value_counts().to_string())

    # 2. Preprocess texts
    print("\n2. Preprocessing texts for LSTM...")
    texts = df["content"].astype(str).tolist()
    labels = df["sentiment"].astype(str).tolist()

    # Preprocessing yang sama dengan inference
    texts_proc = preprocess_corpus_lstm(texts)
    print(f"   Preprocessed {len(texts_proc)} texts")

    # 3. Split data
    print("\n3. Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        texts_proc,
        labels,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=labels,
    )
    print(f"   Train: {len(X_train)} samples")
    print(f"   Test: {len(X_test)} samples")

    # 4. Create tokenizer
    print("\n4. Creating tokenizer...")
    tokenizer = Tokenizer(num_words=MAX_WORDS)
    tokenizer.fit_on_texts(X_train)
    print(f"   Vocabulary size: {len(tokenizer.word_index)}")

    # 5. Convert texts to sequences
    print("\n5. Converting texts to sequences...")
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)

    # 6. Pad sequences
    print("\n6. Padding sequences...")
    X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LEN)
    X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_LEN)
    print(f"   Padded shape: {X_train_pad.shape}")

    # 7. Encode labels
    print("\n7. Encoding labels...")
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    num_classes = len(label_encoder.classes_)
    print(f"   Number of classes: {num_classes}")
    print(f"   Classes: {label_encoder.classes_}")

    # Convert to categorical
    y_train_cat = keras.utils.to_categorical(y_train_encoded, num_classes)
    y_test_cat = keras.utils.to_categorical(y_test_encoded, num_classes)

    # 8. Build model
    print("\n8. Building model...")
    model = Sequential(
        [
            # Layer Embedding
            Embedding(MAX_WORDS, EMBEDDING_DIM),
            Dropout(0.3),
            # Bidirectional LSTM layers
            Bidirectional(LSTM(LSTM_UNITS, return_sequences=True)),
            Dropout(0.3),
            Bidirectional(LSTM(LSTM_UNITS // 2)),
            Dropout(0.3),
            # Dense layers
            Dense(128, activation="relu"),
            Dropout(0.3),
            Dense(64, activation="relu"),
            Dense(num_classes, activation="softmax"),
        ]
    )

    optimizer = Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    print("   Model architecture:")
    model.summary()

    # 9. Define callbacks
    print("\n9. Setting up callbacks...")
    early_stopping = EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    )

    model_checkpoint = ModelCheckpoint(
        str(MODEL_PATH),
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    )

    # 10. Train model
    print("\n10. Training model...")
    print("=" * 70)
    history = model.fit(
        X_train_pad,
        y_train_cat,
        validation_split=VALIDATION_SPLIT,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping, model_checkpoint],
        verbose=1,
    )

    # 11. Evaluate model
    print("\n11. Evaluating model...")
    test_loss, test_accuracy = model.evaluate(X_test_pad, y_test_cat, verbose=0)
    print(f"   Test accuracy: {test_accuracy:.4f}")
    print(f"   Test loss: {test_loss:.4f}")

    # 12. Save artifacts
    print("\n12. Saving artifacts...")
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    with TOKENIZER_PATH.open("wb") as f:
        pickle.dump(tokenizer, f)
    print(f"   ✓ Saved tokenizer to: {TOKENIZER_PATH}")

    with LABEL_ENCODER_PATH.open("wb") as f:
        pickle.dump(label_encoder, f)
    print(f"   ✓ Saved label encoder to: {LABEL_ENCODER_PATH}")

    print(f"   ✓ Model saved to: {MODEL_PATH}")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\nModel accuracy: {test_accuracy:.4f}")
    print("\nYou can now use the model in the API:")
    print("  uv run uvicorn main:app --reload")


if __name__ == "__main__":
    main()