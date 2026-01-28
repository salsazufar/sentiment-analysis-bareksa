"""
High-level prediction function used by the FastAPI app and other clients.
"""

from __future__ import annotations

from dataclasses import dataclass

from .model import SentimentModel
from .preprocess import preprocess_text_lstm


_MODEL: SentimentModel | None = None


def _get_model() -> SentimentModel:
    """Lazy-load the model only when needed."""
    global _MODEL
    if _MODEL is None:
        _MODEL = SentimentModel()
    return _MODEL


@dataclass
class SentimentPrediction:
    label: str
    score: float
    clean_text: str


def predict_sentiment(text: str) -> dict:
    """
    Run full pipeline: preprocessing + model inference.

    Menggunakan preprocessing khusus untuk LSTM (cleaning + stopword removal, tanpa stemming).

    Returns a dict that is JSON-serializable and friendly for APIs:
        {
            "label": "...",
            "score": 0.95,
            "clean_text": "...",
            "probabilities": {
                "positif": 0.85,
                "netral": 0.10,
                "negatif": 0.05
            }
        }
    """

    clean = preprocess_text_lstm(text)
    model = _get_model()
    
    # Get probabilities for all classes
    probs = model.predict_proba([clean])[0]
    all_labels = model.label_encoder.inverse_transform(range(len(probs)))
    
    # Create probabilities dictionary
    probabilities = {
        str(label): float(prob) 
        for label, prob in zip(all_labels, probs)
    }
    
    # Get predicted label and score
    label, score = model.predict_label(clean)
    prediction = SentimentPrediction(label=label, score=score, clean_text=clean)
    
    return {
        "label": prediction.label,
        "score": prediction.score,
        "clean_text": prediction.clean_text,
        "probabilities": probabilities,
    }

