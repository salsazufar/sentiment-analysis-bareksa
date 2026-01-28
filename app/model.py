"""
Model loading utilities for Bareksa review sentiment analysis.

Menggunakan model LSTM (BiLSTM) yang dilatih di notebook.
Model ini memberikan akurasi lebih tinggi dibanding baseline klasik.

Arsitektur:
- Artefak disimpan di folder `app/artifacts/`:
  - `lstm_model.keras` (atau `best_lstm_model.keras` di root)
  - `tokenizer.pkl`
  - `label_encoder.pkl`
- Artefak tersebut dapat dihasilkan dengan skrip training terpisah.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
MODEL_PATH = ROOT_DIR / "best_lstm_model.keras"  # Model di root directory
TOKENIZER_PATH = ARTIFACTS_DIR / "tokenizer.pkl"
LABEL_ENCODER_PATH = ARTIFACTS_DIR / "label_encoder.pkl"

# Hyperparameters yang sama dengan training
MAX_WORDS = 10000
MAX_LEN = 100


class SentimentModel:
    """
    Lazy-loading wrapper around the LSTM model.

    Usage:
        model = SentimentModel()
        label, score = model.predict_label("ulasan...")
    """

    def __init__(
        self,
        model_path: Path | None = None,
        tokenizer_path: Path | None = None,
        label_encoder_path: Path | None = None,
    ) -> None:
        self.model_path = model_path or MODEL_PATH
        self.tokenizer_path = tokenizer_path or TOKENIZER_PATH
        self.label_encoder_path = label_encoder_path or LABEL_ENCODER_PATH
        self._model = None
        self._tokenizer = None
        self._label_encoder = None

        self._load()

    def _load(self) -> None:
        """
        Load model, tokenizer, and label encoder from disk, raising a helpful error if missing.
        """
        # Lazy import TensorFlow to avoid DLL errors on import
        try:
            import tensorflow as tf  # noqa: PLC0415
        except ImportError as e:
            raise RuntimeError(
                "Failed to import TensorFlow. "
                "Please ensure TensorFlow is properly installed. "
                "On Windows, you may need to install Visual C++ Redistributables (2015-2022)."
            ) from e

        if not self.model_path.exists():
            raise RuntimeError(
                "LSTM model not found. "
                f"Expected: {self.model_path}\n\n"
                "Please ensure the model file exists. "
                "It should be generated from the notebook training."
            )

        if not self.tokenizer_path.exists() or not self.label_encoder_path.exists():
            raise RuntimeError(
                "Model artifacts not found. "
                f"Expected:\n- {self.tokenizer_path}\n- {self.label_encoder_path}\n\n"
                "Please run the training script (e.g. train_lstm.py) to "
                "generate these files before using the API."
            )

        # Load Keras model
        self._model = tf.keras.models.load_model(self.model_path)

        # Load tokenizer
        with self.tokenizer_path.open("rb") as f:
            self._tokenizer = pickle.load(f)

        # Load label encoder
        with self.label_encoder_path.open("rb") as f:
            self._label_encoder = pickle.load(f)

    @property
    def model(self):
        return self._model

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def label_encoder(self):
        return self._label_encoder

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        Predict class probabilities for a list of preprocessed texts.
        Texts should be cleaned but not stemmed (LSTM works with raw cleaned text).
        """
        # Lazy import to avoid loading TensorFlow on module import
        from tensorflow.keras.preprocessing.sequence import pad_sequences  # noqa: PLC0415

        # Convert texts to sequences
        sequences = self._tokenizer.texts_to_sequences(texts)
        
        # Debug: Check if sequences are empty
        for i, seq in enumerate(sequences):
            if len(seq) == 0:
                import warnings
                warnings.warn(
                    f"Warning: Text {i} resulted in empty sequence after tokenization. "
                    "This may indicate a preprocessing mismatch with training data."
                )
        
        # Pad sequences
        padded = pad_sequences(sequences, maxlen=MAX_LEN)
        # Predict
        probs = self._model.predict(padded, verbose=0)
        return probs

    def predict_label(self, text: str) -> Tuple[str, float]:
        """
        Predict label and confidence score for a single preprocessed text.

        Returns:
            (predicted_label, probability_of_that_label)
        """

        probs = self.predict_proba([text])[0]
        class_index = int(np.argmax(probs))
        label = str(self._label_encoder.inverse_transform([class_index])[0])
        score = float(probs[class_index])
        return label, score

