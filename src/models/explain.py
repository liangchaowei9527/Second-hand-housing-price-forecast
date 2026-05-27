"""Model explanation utilities."""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportGeneralTypeIssues=false

from __future__ import annotations

import argparse
import pickle
import sys
import warnings
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.visualization.font_utils import configure_chinese_font

configure_chinese_font()
import matplotlib.pyplot as plt
import pandas as pd

try:
    import shap
except ImportError:  # pragma: no cover - optional dependency fallback
    shap = None

from src.config.settings import CLEANED_DATA_PATH, FIGURES_DIR, TRAINED_MODEL_PATH
from src.features.build_features import MODEL_FEATURES, build_feature_frame, load_cleaned_data


def load_model_bundle(model_path: str | Path = TRAINED_MODEL_PATH) -> dict[str, Any]:
    with Path(model_path).open("rb") as handle:
        return pickle.load(handle)


def compute_shap_summary(
    data_path: str | Path = CLEANED_DATA_PATH,
    model_path: str | Path = TRAINED_MODEL_PATH,
    figures_dir: str | Path = FIGURES_DIR,
    sample_size: int = 200,
) -> dict[str, Any]:
    if shap is None:
        raise ImportError("shap is required for model explanation.")

    bundle = load_model_bundle(model_path)
    pipeline = bundle["pipeline"]
    data = load_cleaned_data(data_path)
    feature_frame = build_feature_frame(data)
    sample = feature_frame[MODEL_FEATURES].dropna(how="all").sample(min(sample_size, len(feature_frame)), random_state=42)

    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    transformed = preprocessor.transform(sample)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    feature_names = preprocessor.get_feature_names_out()
    background_size = min(len(transformed), 40)
    background = transformed[:background_size]
    explainer = shap.Explainer(lambda data: model.predict(data), background, feature_names=feature_names)
    shap_values = explainer(transformed)

    figures_path = Path(figures_dir)
    figures_path.mkdir(parents=True, exist_ok=True)
    summary_path = figures_path / "shap_summary.png"
    bar_path = figures_path / "shap_bar.png"

    plt.figure(figsize=(10, 6))
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The NumPy global RNG was seeded by calling `np.random.seed`.*",
            category=FutureWarning,
        )
        shap.summary_plot(shap_values.values, transformed, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(summary_path, dpi=160, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 6))
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The NumPy global RNG was seeded by calling `np.random.seed`.*",
            category=FutureWarning,
        )
        shap.summary_plot(shap_values.values, transformed, feature_names=feature_names, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(bar_path, dpi=160, bbox_inches="tight")
    plt.close()

    mean_abs = pd.Series(abs(shap_values.values).mean(axis=0), index=feature_names).sort_values(ascending=False)
    return {
        "summary_path": summary_path,
        "bar_path": bar_path,
        "top_features": mean_abs.head(10),
    }


def explain_model() -> None:
    result = compute_shap_summary()
    print(result["top_features"].to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SHAP explanations for the trained model.")
    parser.add_argument("--data-path", default=str(CLEANED_DATA_PATH), help="清洗后数据路径")
    parser.add_argument("--model-path", default=str(TRAINED_MODEL_PATH), help="模型文件路径")
    parser.add_argument("--figures-dir", default=str(FIGURES_DIR), help="图表输出目录")
    args = parser.parse_args()

    result = compute_shap_summary(args.data_path, args.model_path, args.figures_dir)
    print(result["top_features"].to_string())


if __name__ == "__main__":
    main()
