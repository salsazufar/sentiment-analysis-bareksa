"""
Application package for Bareksa review sentiment analysis.

This package exposes high-level helpers for:
- Text preprocessing
- Loading the trained sentiment model
- Running sentiment predictions for a single review text
"""

from .predict import predict_sentiment  # noqa: F401

