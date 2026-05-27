"""Visualization helpers for the cleaned Anjuke dataset."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from src.visualization.font_utils import configure_chinese_font

configure_chinese_font()
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.utils.helpers import parse_floor_features


def _save_placeholder(ax: plt.Axes, title: str, message: str, x_label: str | None = None, y_label: str | None = None) -> None:
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=12)
    ax.set_title(title)
    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)
    ax.set_axis_off()


def _save_category_mean_plot(
    df: pd.DataFrame,
    category_column: str,
    target_column: str,
    output_path: Path,
    title: str,
    top_n: int | None = 10,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    sample = df.dropna(subset=[category_column, target_column])
    if sample.empty:
        _save_placeholder(ax, title, "No valid data for this plot")
    else:
        category_mean = sample.groupby(category_column, dropna=True)[target_column].mean().sort_values(ascending=False)
        if top_n is not None:
            category_mean = category_mean.head(top_n)
        if category_mean.empty:
            _save_placeholder(ax, title, "No valid data for this plot")
        else:
            category_mean.sort_values().plot(kind="barh", ax=ax, color="#7A57FF")
            ax.set_title(title)
            ax.set_xlabel(f"Average {target_column.replace('_', ' ').title()}")
            ax.set_ylabel(category_column.replace("_", " ").title())
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _save_scatter_plot(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    output_path: Path,
    title: str,
    color: str,
    add_regression: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    sample = df[[x_column, y_column]].dropna()
    if sample.empty:
        _save_placeholder(ax, title, "No valid data for this plot")
    else:
        sns.scatterplot(data=sample, x=x_column, y=y_column, ax=ax, alpha=0.45, color=color)
        if add_regression and len(sample) > 1:
            sns.regplot(data=sample, x=x_column, y=y_column, ax=ax, scatter=False, color="#1F2937", line_kws={"linewidth": 1.5})
        ax.set_title(title)
        ax.set_xlabel(x_column.replace("_", " ").title())
        ax.set_ylabel(y_column.replace("_", " ").title())
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def generate_basic_eda(df: pd.DataFrame, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")
    configure_chinese_font()

    # 1. 缺失值概览
    missing_ratio = df.isna().mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    missing_nonzero = missing_ratio[missing_ratio > 0]
    if missing_nonzero.empty:
        ax.text(0.5, 0.5, "No missing values", ha="center", va="center", fontsize=12)
        ax.set_axis_off()
    else:
        missing_nonzero.plot(kind="bar", ax=ax, color="#2F6BFF")
        ax.set_title("Missing Value Ratio by Column")
        ax.set_ylabel("Ratio")
        ax.set_xlabel("Column")
    fig.tight_layout()
    fig.savefig(figures_dir / "missing_ratio.png", dpi=160)
    plt.close(fig)

    # 2. 总价分布
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df["total_price_wan"].dropna(), bins=30, kde=True, ax=ax, color="#FF8A3D")
    ax.set_title("Total Price Distribution (Wan)")
    ax.set_xlabel("Total Price (Wan)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(figures_dir / "total_price_distribution.png", dpi=160)
    plt.close(fig)

    # 3. 面积与总价关系
    _save_scatter_plot(df, "area_sqm", "total_price_wan", figures_dir / "area_vs_price.png", "Area vs Total Price", "#00A389", True)

    # 4. 区域均价 Top 10
    if "region" in df.columns:
        _save_category_mean_plot(
            df,
            "region",
            "total_price_wan",
            figures_dir / "region_mean_price_top10.png",
            "Top 10 Regions by Average Total Price",
        )

    # 5. 楼层类型均价
    if "floor" in df.columns:
        floor_categories = df["floor"].apply(lambda value: parse_floor_features(value)["floor_category"])
        floor_df = df.copy()
        floor_df["floor_category"] = floor_categories
        _save_category_mean_plot(
            floor_df,
            "floor_category",
            "total_price_wan",
            figures_dir / "floor_category_mean_price.png",
            "Average Total Price by Floor Category",
            top_n=None,
        )

    # 6. 朝向均价
    if "orientation" in df.columns:
        _save_category_mean_plot(
            df,
            "orientation",
            "total_price_wan",
            figures_dir / "orientation_mean_price.png",
            "Average Total Price by Orientation",
            top_n=10,
        )

    # 7. 装修均价
    if "decoration" in df.columns:
        _save_category_mean_plot(
            df,
            "decoration",
            "total_price_wan",
            figures_dir / "decoration_mean_price.png",
            "Average Total Price by Decoration",
            top_n=10,
        )

    # 8. 房龄与总价关系
    if "house_age" in df.columns:
        _save_scatter_plot(
            df,
            "house_age",
            "total_price_wan",
            figures_dir / "house_age_vs_price.png",
            "House Age vs Total Price",
            "#F97316",
        )

    # 9. 建成年份与总价关系
    if "build_year" in df.columns:
        _save_scatter_plot(
            df,
            "build_year",
            "total_price_wan",
            figures_dir / "build_year_vs_price.png",
            "Build Year vs Total Price",
            "#2563EB",
        )


def render_charts() -> None:
    raise NotImplementedError("Use generate_basic_eda(df, figures_dir) instead.")
