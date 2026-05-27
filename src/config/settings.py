"""Project configuration settings."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODEL_DIR = PROJECT_ROOT / "models"

RAW_DATA_PATH = RAW_DATA_DIR / "house_price_raw.csv"
MERGED_DATA_PATH = PROCESSED_DATA_DIR / "anjuke_merged_standardized.csv"
CLEANED_DATA_PATH = PROCESSED_DATA_DIR / "anjuke_cleaned.csv"
TRAINED_MODEL_PATH = MODEL_DIR / "house_price_model.pkl"
TRAINING_METRICS_PATH = MODEL_DIR / "training_metrics.json"
FEATURE_COLUMNS_PATH = MODEL_DIR / "feature_columns.json"
