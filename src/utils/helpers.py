"""General helper utilities."""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportGeneralTypeIssues=false

from __future__ import annotations

import re
from typing import Any

import pandas as pd

YEAR_BUILD_PATTERN = re.compile(r"(?P<year>(?:19|20)\d{2})年建造")
FLOOR_TOTAL_PATTERN = re.compile(r"共\s*(?P<total>\d+)\s*层")


def normalize_text(value: Any) -> str | None:
    if value is None or value is pd.NA:
        return None
    text = str(value).strip()
    return None if text == "" else text


def extract_build_year_from_text(value: Any) -> int | None:
    text = normalize_text(value)
    if text is None:
        return None
    match = YEAR_BUILD_PATTERN.search(str(text))
    if not match:
        return None
    return int(match.group("year"))


def parse_floor_features(value: Any) -> dict[str, Any]:
    text = normalize_text(value)
    features: dict[str, Any] = {
        "floor_category": None,
        "floor_total": None,
        "floor_year_hint": None,
    }

    if text is None:
        return features

    floor_text = str(text)
    year_hint = extract_build_year_from_text(floor_text)
    if year_hint is not None:
        features["floor_category"] = "year_building"
        features["floor_year_hint"] = year_hint
        return features

    if "低层" in floor_text:
        features["floor_category"] = "low"
    elif "中层" in floor_text:
        features["floor_category"] = "middle"
    elif "高层" in floor_text:
        features["floor_category"] = "high"
    elif "全层" in floor_text or floor_text.startswith("共"):
        features["floor_category"] = "whole"
    elif "地下" in floor_text:
        features["floor_category"] = "underground"
    else:
        features["floor_category"] = "other"

    total_match = FLOOR_TOTAL_PATTERN.search(floor_text)
    if total_match:
        features["floor_total"] = int(total_match.group("total"))
    return features


def normalize_unit_price(unit_price: Any, area_sqm: Any, total_price_wan: Any) -> Any:
    unit = pd.to_numeric(unit_price, errors="coerce")
    area = pd.to_numeric(area_sqm, errors="coerce")
    total = pd.to_numeric(total_price_wan, errors="coerce")

    if isinstance(unit, pd.Series):
        inflated_mask = unit > 1_000_000
        unit.loc[inflated_mask] = unit.loc[inflated_mask] / 10000
        unit = unit.where(unit.between(1000, 500000))
        computed = total * 10000 / area
        return unit.fillna(computed).round().astype("Int64")

    if pd.isna(unit):
        computed = total * 10000 / area
        if pd.isna(computed):
            return pd.NA
        return int(round(float(computed)))

    if unit > 1_000_000:
        unit = unit / 10000
    if unit < 1000 or unit > 500000:
        computed = total * 10000 / area
        if pd.isna(computed):
            return pd.NA
        return int(round(float(computed)))
    return int(round(float(unit)))
