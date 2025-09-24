# Bareksa App Review Sentiment Analysis (NLP)

NLP project for classifying Google Play reviews of the Bareksa app into sentiments. It includes data scraping, preprocessing (ID/EN), classic ML baselines (TF‑IDF), and a BiLSTM model (TensorFlow/Keras).

## Features
- Google Play scraper to collect reviews 
- Text cleaning: casefolding, tokenization, slang handling, stopword removal (NLTK + Sastrawi), stemming
- Baselines with TF‑IDF (2,000 features): Naive Bayes, Logistic Regression, Random Forest, XGBoost
- Deep learning model: BiLSTM (10k vocab, 200‑dim embeddings, BiLSTM 128→64, dropout)

## Data
- Raw collected: 7,905 reviews 
- CSV: `ulasan_aplikasi.csv`

## Models and Results (test accuracy)
- Naive Bayes: 0.91
- Logistic Regression: 0.95
- Random Forest: 0.97
- XGBoost: 0.97
- BiLSTM: 0.964 (saved as `best_lstm_model.keras`)

## Quick Start
```bash
# 1) Create env and install deps
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt

# 2) (Optional) Scrape latest reviews
python scrapping.py

# 3) Explore/train/evaluate
# Open analisis_sentimen_bareksa.ipynb and run cells
```

## Repository Structure
- `scrapping.py` — Google Play scraper (to CSV/DataFrame)
- `ulasan_aplikasi.csv` — Collected reviews (raw/processed columns)
- `analisis_sentimen_bareksa.ipynb` — Preprocessing, modeling, evaluation
- `best_lstm_model.keras` — Saved best LSTM weights
- `requirements.txt` — Python dependencies

## Requirements
- Python 3.9+
- See `requirements.txt` 




