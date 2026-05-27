"""Model inference utilities."""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportGeneralTypeIssues=false

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Any

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.config.settings import CLEANED_DATA_PATH, TRAINED_MODEL_PATH
from src.features.build_features import MODEL_FEATURES, build_feature_frame


def load_model_bundle(model_path: str | Path = TRAINED_MODEL_PATH) -> dict[str, Any]:
    with Path(model_path).open("rb") as handle:
        return pickle.load(handle)


def prepare_prediction_frame(input_data: dict[str, Any] | pd.DataFrame) -> pd.DataFrame:
    if isinstance(input_data, pd.DataFrame):
        source_frame = input_data.copy()
    else:
        source_frame = pd.DataFrame([input_data])
    feature_frame = build_feature_frame(source_frame)
    return feature_frame[MODEL_FEATURES].copy()


def predict_price(input_data: dict[str, Any] | pd.DataFrame, model_path: str | Path = TRAINED_MODEL_PATH) -> pd.Series:
    bundle = load_model_bundle(model_path)
    pipeline = bundle["pipeline"]
    features = prepare_prediction_frame(input_data)
    predictions = pipeline.predict(features)
    return pd.Series(predictions, name="predicted_total_price_wan")


def predict() -> None:
    """Backward-compatible wrapper for interactive callers."""

    sample_path = CLEANED_DATA_PATH
    if not Path(sample_path).exists():
        raise FileNotFoundError(f"Missing data file: {sample_path}")
    sample = pd.read_csv(sample_path).head(1)
    result = predict_price(sample)
    print(result.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a sample house price prediction.")
    parser.add_argument("--model-path", default=str(TRAINED_MODEL_PATH), help="模型文件路径")
    parser.add_argument("--data-path", default=str(CLEANED_DATA_PATH), help="输入数据路径")
    args = parser.parse_args()

    sample = pd.read_csv(args.data_path).head(1)
    result = predict_price(sample, args.model_path)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
