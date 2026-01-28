# Bareksa App Review Sentiment Analysis (NLP)

NLP project for classifying Google Play reviews of the Bareksa app into sentiments (positive, neutral, negative). Includes data scraping, preprocessing (ID/EN), a classic ML baseline (TF‑IDF + Logistic Regression), a BiLSTM model (TensorFlow/Keras), and a **FastAPI** backend with **web UI** for predictions.

## Features

- **Google Play scraper** to collect reviews
- **Preprocessing**: casefolding, tokenization, slang handling, stopword removal (NLTK + Sastrawi), stemming (for baseline)
- **Baseline**: TF‑IDF (2,000 features) + Logistic Regression — see `train_baseline.py`
- **Deep learning**: BiLSTM (10k vocab, 200‑dim embeddings, BiLSTM 128→64, dropout) — see `train_lstm.py`
- **API & UI**: FastAPI with `/predict`, `/health`, and a web page to try predictions

## Data

- **Raw**: 7,905 reviews
- **File**: `ulasan_aplikasi.csv` (columns `content`, `score`)

## Models and Results (test accuracy)

- **Logistic Regression (TF‑IDF)**: ~0.95 (via `train_baseline.py`)
- **BiLSTM**: ~0.964 — saved as `best_lstm_model.keras`

*Other baselines (NB, RF, XGB) and detailed experiments are in `analisis_sentimen_bareksa.ipynb`.*

## Quick Start

```bash
# 1) Create env and install dependencies
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate   # Linux/macOS
pip install -r requirements.txt

# 2) (Optional) Scrape latest reviews
python scrapping.py

# 3) Train the LSTM model (required before using the API)
python train_lstm.py

# 4) Generate tokenizer & label encoder artifacts for the API
python save_lstm_artifacts.py

# 5) Run API + Web UI
uvicorn main:app --reload
```

Open **http://127.0.0.1:8000** for the UI, or call `POST /predict` with body `{"text": "your review..."}`.

## API

| Method | Endpoint   | Description                          |
|--------|------------|--------------------------------------|
| GET    | `/`        | Web UI for text input & prediction   |
| GET    | `/health`  | Health check                         |
| POST   | `/predict` | Sentiment prediction; body: `{"text": "..."}` |

`/predict` response: `{"label": "positif"|"netral"|"negatif", "score": float, "clean_text": str, "probabilities": {...}}`.

## Repository Structure

| Path | Description |
|------|-------------|
| `main.py` | FastAPI app (routes, `/predict`, serves UI) |
| `app/` | Prediction & preprocessing modules |
| `app/model.py` | Loads LSTM, tokenizer, label encoder |
| `app/predict.py` | Prediction pipeline (preprocess + inference) |
| `app/preprocess.py` | Cleaning, tokenization, stopwords, stemming |
| `app/artifacts/` | `tokenizer.pkl`, `label_encoder.pkl` (from `save_lstm_artifacts.py` or `train_lstm.py`) |
| `templates/index.html` | Web page for trying predictions |
| `train_lstm.py` | Train BiLSTM & save model + artifacts |
| `train_baseline.py` | Train TF‑IDF + Logistic Regression |
| `save_lstm_artifacts.py` | Export tokenizer & label encoder (e.g. when LSTM was trained in the notebook) |
| `scrapping.py` | Google Play review scraper → CSV |
| `ulasan_aplikasi.csv` | Review data |
| `analisis_sentimen_bareksa.ipynb` | Exploration, preprocessing, model evaluation |
| `best_lstm_model.keras` | Trained BiLSTM model |
| `test_api.py` | Smoke tests for preprocessing, `predict_sentiment`, and API |
| `debug_model.py` | Debug LSTM predictions (load model, sample texts) |
| `requirements.txt` | Python dependencies |

## Requirements

- Python 3.9+
- See `requirements.txt` (TensorFlow, FastAPI, uvicorn, NLTK, Sastrawi, scikit-learn, etc.)

## Running Tests

```bash
# Ensure the API is running in another terminal: uvicorn main:app --reload
python test_api.py
```
