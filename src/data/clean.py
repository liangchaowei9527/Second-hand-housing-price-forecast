# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportUnknownLambdaType=false
"""Data cleaning utilities for the standardized Anjuke house-price dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.utils.helpers import extract_build_year_from_text, normalize_text, normalize_unit_price
from src.visualization.charts import generate_basic_eda


EXPECTED_COLUMNS = [
    "community",
    "region",
    "area_sqm",
    "floor",
    "orientation",
    "decoration",
    "build_year",
    "total_price_wan",
    "unit_price_yuan_per_sqm",
    "source_file",
    "source",
]


def find_standardized_files(input_dir: Path) -> List[Path]:
    return sorted(input_dir.glob("anjuke_ingested_*.csv"))


def select_latest_files_by_source(files: Iterable[Path]) -> List[Path]:
    latest_files: dict[str, Path] = {}
    for file_path in sorted(files, key=lambda path: path.stat().st_mtime, reverse=True):
        try:
            preview = pd.read_csv(file_path, nrows=1)
            source_file = str(preview.iloc[0]["source_file"])
        except Exception:
            source_file = file_path.name
        if source_file not in latest_files:
            latest_files[source_file] = file_path
    return list(latest_files.values())


def load_and_merge_standardized_data(files: Iterable[Path]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    selected_files = select_latest_files_by_source(files)
    for file_path in selected_files:
        frame = pd.read_csv(file_path)
        for column in EXPECTED_COLUMNS:
            if column not in frame.columns:
                frame[column] = pd.NA
        frames.append(frame[EXPECTED_COLUMNS])

    if not frames:
        raise SystemExit("未找到任何标准化后的 CSV 文件，请先运行 ingest 脚本。")

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates().reset_index(drop=True)
    return merged


def normalize_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    text_columns = ["community", "region", "floor", "orientation", "decoration", "source_file", "source"]
    normalized = df.copy()
    for column in text_columns:
        if column in normalized.columns:
            normalized[column] = normalized[column].map(normalize_text)
    return normalized


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = normalize_text_columns(df)

    numeric_columns = ["area_sqm", "build_year", "total_price_wan", "unit_price_yuan_per_sqm"]
    for column in numeric_columns:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    if "floor" in cleaned.columns and "build_year" in cleaned.columns:
        inferred_years = cleaned["floor"].map(extract_build_year_from_text)
        cleaned["build_year"] = cleaned["build_year"].fillna(pd.to_numeric(inferred_years, errors="coerce"))

    cleaned = cleaned[cleaned["total_price_wan"].notna() & (cleaned["total_price_wan"] > 0)]
    cleaned = cleaned[cleaned["area_sqm"].notna() & (cleaned["area_sqm"] > 0)]

    if "unit_price_yuan_per_sqm" in cleaned.columns:
        cleaned["unit_price_yuan_per_sqm"] = normalize_unit_price(
            cleaned["unit_price_yuan_per_sqm"], cleaned["area_sqm"], cleaned["total_price_wan"]
        )

    current_year = pd.Timestamp.now().year
    cleaned["house_age"] = cleaned["build_year"].apply(
        lambda value: current_year - value if pd.notna(value) and value > 0 else pd.NA
    )
    cleaned["price_per_sqm_wan"] = (cleaned["total_price_wan"] / cleaned["area_sqm"]).round(4)

    cleaned = cleaned.sort_values(["region", "community", "total_price_wan"], na_position="last")
    cleaned = cleaned.drop_duplicates(
        subset=[column for column in ["community", "region", "area_sqm", "total_price_wan", "source_file"] if column in cleaned.columns]
    )
    cleaned = cleaned.reset_index(drop=True)
    return cleaned


def save_outputs(merged: pd.DataFrame, cleaned: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    merged_path = output_dir / "anjuke_merged_standardized.csv"
    cleaned_path = output_dir / "anjuke_cleaned.csv"

    merged.to_csv(merged_path, index=False, encoding="utf-8-sig")
    cleaned.to_csv(cleaned_path, index=False, encoding="utf-8-sig")
    return merged_path, cleaned_path


def clean(input_dir: str = "data/raw", output_dir: str = "data/processed", figures_dir: str = "reports/figures") -> tuple[Path, Path]:
    input_path = Path(input_dir)
    standardized_files = find_standardized_files(input_path)
    merged = load_and_merge_standardized_data(standardized_files)
    cleaned = clean_data(merged)

    output_path = Path(output_dir)
    merged_path, cleaned_path = save_outputs(merged, cleaned, output_path)
    generate_basic_eda(cleaned, Path(figures_dir))
    return merged_path, cleaned_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge standardized Anjuke data, clean it, and generate basic EDA.")
    parser.add_argument("--input-dir", default="data/raw", help="包含标准化 CSV 的目录")
    parser.add_argument("--output-dir", default="data/processed", help="清洗后数据输出目录")
    parser.add_argument("--figures-dir", default="reports/figures", help="EDA 图表输出目录")
    args = parser.parse_args()

    merged_path, cleaned_path = clean(args.input_dir, args.output_dir, args.figures_dir)
    print(f"Merged standardized data saved to: {merged_path}")
    print(f"Cleaned data saved to: {cleaned_path}")


if __name__ == "__main__":
    main()
