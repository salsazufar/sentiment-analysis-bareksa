"""
FastAPI application exposing sentiment analysis for Bareksa reviews.
"""

from __future__ import annotations

from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from app.predict import predict_sentiment
from app.hf_artifacts import ensure_model_artifacts


templates = Jinja2Templates(directory="templates")


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    label: str
    score: float
    clean_text: str
    probabilities: Dict[str, float]


app = FastAPI(
    title="Bareksa Review Sentiment API",
    description="API untuk analisis sentimen ulasan aplikasi Bareksa.",
    version="0.1.0",
)


@app.on_event("startup")
def _startup_download_artifacts() -> None:
    """
    On Hugging Face Spaces, artifacts may not be committed to the repo.
    Download them from the Hugging Face Hub if missing.
    """

    ensure_model_artifacts()


@app.get("/", response_class=HTMLResponse, tags=["ui"])
def index(request: Request) -> Any:
    """
    Render the main web UI.
    """

    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health", tags=["health"])
def health() -> Dict[str, str]:
    return {"status": "ok", "message": "Bareksa Sentiment API is running"}


@app.post("/predict", response_model=PredictResponse, tags=["prediction"])
def predict(request: PredictRequest) -> Any:
    """
    Predict sentiment for a single review text.
    """

    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Field 'text' tidak boleh kosong.")

    try:
        result = predict_sentiment(text)
    except RuntimeError as exc:
        # Surface helpful error if model artifacts are missing.
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return PredictResponse(**result)

