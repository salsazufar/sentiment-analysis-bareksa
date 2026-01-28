FROM python:3.10-slim

# Hugging Face Spaces expects the app to listen on 7860
ENV PORT=7860

# Make logs unbuffered and reduce pip noise
ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps (kept minimal; wheels cover most Python deps)
# `libgomp1` is required by scikit-learn at runtime (used by the pickled label encoder).
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libgomp1 \
  && rm -rf /var/lib/apt/lists/*

# Install Python deps first (better layer caching)
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

# Pre-download NLTK data during build (avoid runtime downloads on Spaces)
ENV NLTK_DATA=/usr/local/share/nltk_data
RUN python -c "import nltk; nltk.download('punkt', download_dir='${NLTK_DATA}'); nltk.download('punkt_tab', download_dir='${NLTK_DATA}'); nltk.download('stopwords', download_dir='${NLTK_DATA}')"

# Copy the app code
COPY . /app

# Run FastAPI
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-7860}"]
