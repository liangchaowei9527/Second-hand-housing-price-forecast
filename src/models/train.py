"""Model training utilities."""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportGeneralTypeIssues=false

from __future__ import annotations

import argparse
import json
import pickle
import sys
import warnings
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
from src.visualization.font_utils import configure_chinese_font

from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.visualization.font_utils import configure_chinese_font

configure_chinese_font()
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import shap
except ImportError:  # pragma: no cover - optional dependency fallback
    shap = None

from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.config.settings import CLEANED_DATA_PATH, FIGURES_DIR, MODEL_DIR, FEATURE_COLUMNS_PATH, TRAINED_MODEL_PATH, TRAINING_METRICS_PATH
from src.features.build_features import CATEGORICAL_FEATURES, MODEL_FEATURES, NUMERIC_FEATURES, TARGET_COLUMN, build_feature_frame, load_cleaned_data

try:
    from xgboost import XGBRegressor
except ImportError:  # pragma: no cover - optional dependency fallback
    XGBRegressor = None


def _make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover - older scikit-learn compatibility
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _build_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_transformer = Pipeline(steps=numeric_steps)
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", _make_one_hot_encoder()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, NUMERIC_FEATURES),
            ("categorical", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def _build_baseline_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", _build_preprocessor(scale_numeric=True)),
            ("model", Ridge(alpha=1.0, random_state=42)),
        ]
    )


def _build_tree_estimator() -> Any:
    if XGBRegressor is not None:
        return XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.0,
            reg_lambda=1.0,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=4,
        )
    return RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=4)


def _build_tree_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", _build_preprocessor(scale_numeric=False)),
            ("model", _build_tree_estimator()),
        ]
    )


def _evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _save_json(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _save_metrics_comparison(metrics: dict[str, dict[str, float]], output_path: Path) -> None:
    comparison = pd.DataFrame(metrics).T.reset_index().rename(columns={"index": "model"})
    melt = comparison.melt(id_vars="model", value_vars=["mae", "rmse"], var_name="metric", value_name="value")

    fig, ax = plt.subplots(figsize=(8, 5))
    for metric_name, metric_frame in melt.groupby("metric"):
        ax.bar(metric_frame["model"] + " - " + metric_name.upper(), metric_frame["value"])
    ax.set_title("Model Metric Comparison")
    ax.set_ylabel("Value")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _save_feature_importance(pipeline: Pipeline, output_path: Path) -> None:
    preprocessor: ColumnTransformer = pipeline.named_steps["preprocessor"]
    model: Any = pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return

    feature_names = preprocessor.get_feature_names_out()
    importance = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False).head(20)
    if importance.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    importance.sort_values().plot(kind="barh", ax=ax, color="#2563EB")
    ax.set_title("Top Feature Importance")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _save_shap_plots(pipeline: Pipeline, x_train: pd.DataFrame, figures_dir: Path) -> dict[str, Path]:
    if shap is None:
        return {}

    sample_size = min(len(x_train), 120)
    if sample_size <= 0:
        return {}

    sample = x_train.sample(sample_size, random_state=42)
    preprocessor: ColumnTransformer = pipeline.named_steps["preprocessor"]
    model: Any = pipeline.named_steps["model"]
    transformed = preprocessor.transform(sample)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    feature_names = preprocessor.get_feature_names_out()
    background_size = min(len(transformed), 40)
    background = transformed[:background_size]
    explainer = shap.Explainer(lambda data: model.predict(data), background, feature_names=feature_names)
    shap_values = explainer(transformed)

    figures_dir.mkdir(parents=True, exist_ok=True)
    summary_path = figures_dir / "shap_summary.png"
    bar_path = figures_dir / "shap_bar.png"

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

    return {"summary": summary_path, "bar": bar_path}


def train_model(
    data_path: str | Path = CLEANED_DATA_PATH,
    model_path: str | Path = TRAINED_MODEL_PATH,
    metrics_path: str | Path = TRAINING_METRICS_PATH,
    figures_dir: str | Path = FIGURES_DIR,
) -> dict[str, Any]:
    data = load_cleaned_data(data_path)
    feature_frame = build_feature_frame(data)
    model_features = feature_frame[MODEL_FEATURES].copy()
    target = pd.to_numeric(feature_frame[TARGET_COLUMN], errors="coerce")

    valid_mask = target.notna()
    X = model_features.loc[valid_mask].reset_index(drop=True)
    y = target.loc[valid_mask].reset_index(drop=True)

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    baseline_pipeline = _build_baseline_pipeline()
    baseline_pipeline.fit(x_train, y_train)
    baseline_pred = baseline_pipeline.predict(x_test)
    baseline_metrics = _evaluate_predictions(y_test, baseline_pred)

    tree_pipeline = _build_tree_pipeline()
    tree_pipeline.fit(x_train, y_train)
    tree_pred = tree_pipeline.predict(x_test)
    tree_metrics = _evaluate_predictions(y_test, tree_pred)

    cv = KFold(n_splits=3, shuffle=True, random_state=42)
    cv_scores = cross_val_score(tree_pipeline, X, y, cv=cv, scoring="neg_mean_absolute_error", n_jobs=1)

    metrics = {
        "baseline_ridge": baseline_metrics,
        "tree_model": tree_metrics,
        "tree_model_cv_mae": {
            "mean": float((-cv_scores).mean()),
            "std": float((-cv_scores).std()),
        },
    }

    model_dir = Path(model_path).parent
    model_dir.mkdir(parents=True, exist_ok=True)
    figures_path = Path(figures_dir)
    figures_path.mkdir(parents=True, exist_ok=True)

    bundle = {
        "pipeline": tree_pipeline,
        "feature_columns": MODEL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "target_column": TARGET_COLUMN,
        "metrics": metrics,
    }

    with Path(model_path).open("wb") as handle:
        pickle.dump(bundle, handle)

    _save_json({"feature_columns": MODEL_FEATURES}, Path(FEATURE_COLUMNS_PATH))
    _save_json(metrics, Path(metrics_path))
    _save_metrics_comparison({"baseline_ridge": baseline_metrics, "tree_model": tree_metrics}, figures_path / "model_metrics_comparison.png")
    _save_feature_importance(tree_pipeline, figures_path / "feature_importance.png")
    shap_paths = _save_shap_plots(tree_pipeline, x_train, figures_path)

    result = {
        "model_path": Path(model_path),
        "metrics_path": Path(metrics_path),
        "metrics": metrics,
        "shap_paths": shap_paths,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train house price prediction models.")
    parser.add_argument("--data-path", default=str(CLEANED_DATA_PATH), help="清洗后数据路径")
    parser.add_argument("--model-path", default=str(TRAINED_MODEL_PATH), help="模型输出路径")
    parser.add_argument("--metrics-path", default=str(TRAINING_METRICS_PATH), help="评估指标输出路径")
    parser.add_argument("--figures-dir", default=str(FIGURES_DIR), help="图表输出目录")
    args = parser.parse_args()

    result = train_model(args.data_path, args.model_path, args.metrics_path, args.figures_dir)
    print(json.dumps({"model_path": str(result["model_path"]), "metrics_path": str(result["metrics_path"]), "metrics": result["metrics"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
