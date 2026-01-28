"""
Simple smoke tests for the preprocessing pipeline, prediction helper, and API.

Run with:
    uvicorn main:app --reload
    python test_api.py
"""

from __future__ import annotations

import json
from typing import Any

import requests

from app.preprocess import preprocess_text
from app.predict import predict_sentiment


def test_preprocess() -> None:
    text = "Aplikasi Bareksa sangat membantu saya berinvestasi!"
    processed = preprocess_text(text)
    assert isinstance(processed, str)
    assert processed != ""
    print("preprocess_text OK ->", processed[:80], "...")


def test_predict() -> None:
    text = "Aplikasinya bagus dan mudah digunakan."
    result = predict_sentiment(text)
    assert "label" in result and "score" in result and "clean_text" in result
    print("predict_sentiment OK ->", result)


def test_api() -> None:
    url = "http://127.0.0.1:8000/predict"
    payload: dict[str, Any] = {
        "text": "Sering error saat login dan penarikan dana terlalu lama.",
    }
    resp = requests.post(url, json=payload, timeout=10)
    print("API status:", resp.status_code)
    print("API body:", resp.text)


if __name__ == "__main__":
    test_preprocess()
    try:
        test_predict()
    except RuntimeError as exc:
        print(
            "predict_sentiment failed (likely missing artifacts). "
            "Run train_baseline.py first.\n",
            exc,
        )

    try:
        test_api()
    except Exception as exc:  # noqa: BLE001
        print(
            "API test failed. Make sure the server is running with "
            "`uvicorn main:app --reload`.\n",
            exc,
        )

