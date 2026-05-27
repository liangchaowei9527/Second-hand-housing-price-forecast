"""Feature engineering utilities."""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportGeneralTypeIssues=false

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config.settings import CLEANED_DATA_PATH
from src.utils.helpers import extract_build_year_from_text, parse_floor_features

TARGET_COLUMN = "total_price_wan"
NUMERIC_FEATURES = ["area_sqm", "build_year", "house_age", "floor_total", "area_log"]
CATEGORICAL_FEATURES = ["region", "orientation", "decoration", "floor_category"]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def _calculate_house_age(value: Any, current_year: int) -> object:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return pd.NA
    if not math.isfinite(number) or number <= 0:
        return pd.NA
    return current_year - int(number)


def _extract_floor_category(value: object) -> object:
    if isinstance(value, dict):
        return value.get("floor_category")
    return pd.NA


def _extract_floor_total(value: object) -> object:
    if isinstance(value, dict):
        return value.get("floor_total")
    return pd.NA


def load_cleaned_data(data_path: str | Path = CLEANED_DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(Path(data_path))


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    feature_frame = df.copy()

    for column in ["area_sqm", "build_year", TARGET_COLUMN]:
        if column in feature_frame.columns:
            feature_frame[column] = pd.to_numeric(feature_frame[column], errors="coerce")

    current_year = pd.Timestamp.now().year
    if "build_year" in feature_frame.columns:
        if "floor" in feature_frame.columns:
            inferred_years = feature_frame["floor"].map(extract_build_year_from_text)
            feature_frame["build_year"] = feature_frame["build_year"].fillna(pd.to_numeric(inferred_years, errors="coerce"))
        feature_frame["house_age"] = [_calculate_house_age(value, current_year) for value in feature_frame["build_year"].tolist()]
    else:
        feature_frame["build_year"] = pd.NA
        feature_frame["house_age"] = pd.NA

    feature_frame["area_log"] = np.log1p(feature_frame["area_sqm"])

    if "floor" in feature_frame.columns:
        parsed_floor = feature_frame["floor"].apply(parse_floor_features)
        feature_frame["floor_category"] = [_extract_floor_category(value) for value in parsed_floor.tolist()]
        feature_frame["floor_total"] = pd.to_numeric([
            _extract_floor_total(value) for value in parsed_floor.tolist()
        ], errors="coerce")
    else:
        feature_frame["floor_category"] = pd.NA
        feature_frame["floor_total"] = pd.NA

    for column in CATEGORICAL_FEATURES:
        if column not in feature_frame.columns:
            feature_frame[column] = pd.NA

    for column in NUMERIC_FEATURES:
        if column in feature_frame.columns:
            feature_frame[column] = pd.to_numeric(feature_frame[column], errors="coerce")

    feature_frame = feature_frame.replace({pd.NA: np.nan})

    return feature_frame


def get_model_feature_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    feature_frame = build_feature_frame(df)
    if TARGET_COLUMN not in feature_frame.columns:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")

    model_features = feature_frame[MODEL_FEATURES].copy()
    target = feature_frame[TARGET_COLUMN].astype(float)
    return model_features, target


def build_features() -> pd.DataFrame:
    """Backward-compatible wrapper that returns the engineered feature frame."""

    return build_feature_frame(load_cleaned_data())
