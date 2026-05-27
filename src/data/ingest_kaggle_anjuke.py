"""
将 Kaggle 上的 Anjuke 二手房数据标准化到项目统一 schema 并输出 CSV/JSON。

用法示例:
python src/data/ingest_kaggle_anjuke.py --input data/raw/second-hand-house-prices-from-the-anjuke-website.csv

脚本说明（中文注释符合仓库约定）：
- 支持单个 CSV 文件或目录（会处理目录下所有 .csv 文件）
- 尝试智能匹配常见字段名并做类型/单位转换
- 输出到 data/raw/anjuke_ingested_YYYYMMDD_HHMMSS.csv/.json
"""
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
from __future__ import annotations

import argparse
import datetime
import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


def find_column(columns: List[str], candidates: List[str]) -> Optional[str]:
    """按候选名称列表尝试在 columns 中匹配，返回第一个匹配（忽略大小写、下划线/空格/点差异）。"""
    norm = {re.sub(r"[\s_.-]","", c.lower()): c for c in columns}
    for cand in candidates:
        key = re.sub(r"[\s_.-]","", cand.lower())
        if key in norm:
            return norm[key]
    return None


def parse_numeric(s: object) -> Optional[float]:
    """将字符串、数字或带单位的文本尽量解析为浮点数。"""
    if s is None:
        return None
    if isinstance(s, float) and pd.isna(s):
        return None
    if isinstance(s, (int, float)):
        return float(s)
    s = str(s).strip()
    if s == "":
        return None
    # 去掉逗号、单位等
    s = s.replace(",", "")
    m = re.search(r"([0-9]+\.?[0-9]*)", s)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def detect_price_unit(series: pd.Series) -> str:
    """
    简单启发式判断总价单位：如果绝大多数数值>10000，则可能是 元（人民币），否则可能是 万。
    返回 'wan' 或 'yuan'
    """
    nums = pd.to_numeric(series, errors="coerce").dropna()
    if len(nums) == 0:
        return "wan"
    large = sum(1 for value in nums if value > 10000)
    if large / len(nums) > 0.6:
        return "yuan"
    return "wan"


def extract_text_series(df: pd.DataFrame, column_name: Optional[str]) -> pd.Series:
    if column_name is None:
        return pd.Series([pd.NA] * len(df), index=df.index, dtype="string")
    # 使用显式循环生成结果，避免 pandas .str 链式调用造成的类型检查问题
    s = df[column_name].astype("string")
    res: List[object] = []
    is_missing = s.isna().tolist()
    for x, missing in zip(s.tolist(), is_missing):
        if missing:
            res.append(pd.NA)
            continue
        txt = str(x).strip()
        res.append(pd.NA if txt == "" else txt)
    return pd.Series(res, index=df.index, dtype="string")


def extract_numeric_series(df: pd.DataFrame, column_name: Optional[str]) -> pd.Series:
    if column_name is None:
        return pd.Series([float("nan")] * len(df), index=df.index, dtype="float64")
    # 使用 parse_numeric 对每个单元格进行安全解析，避免对 StringMethods 的类型错误
    parsed = df[column_name].apply(parse_numeric)
    return pd.to_numeric(parsed, errors="coerce").astype("float64")


def ingest_file(path: Path, out_dir: Path) -> Path:
    df = pd.read_csv(path, low_memory=False)
    cols = list(df.columns)

    # 常见字段候选名（可扩展）
    col_map_candidates: Dict[str, List[str]] = {
        "community": ["community", "community_name", "xiaoqu", "resblock_name", "小区", "小区名"],
        "region": ["region", "district", "area", "cityarea", "county", "region_name", "区", "街道"],
        "area_sqm": ["area", "buildarea", "size", "area_sqm", "面积", "总面积", "总面 积"],
        "floor": ["floor", "storey", "楼层", "楼层位置", "楼层分布"],
        "orientation": ["orientation", "toward", "dir", "朝向"],
        "decoration": ["decoration", "renovation", "装饰", "装修"],
        "build_year": ["year", "build_year", "built_year", "construction_year", "建造年代", "建成年代", "建造年份", "建造年 份"],
        "total_price": ["total_price", "price", "transaction_price", "总价", "totalprice", "价格"],
        "unit_price": ["unit_price", "unitprice", "price_per_sqm", "每平米单价", "单价"]
    }

    found = {k: find_column(cols, v) for k, v in col_map_candidates.items()}

    out_df = pd.DataFrame(index=df.index)
    out_df["community"] = extract_text_series(df, found["community"])
    out_df["region"] = extract_text_series(df, found["region"])
    out_df["area_sqm"] = extract_numeric_series(df, found["area_sqm"])
    out_df["floor"] = extract_text_series(df, found["floor"])
    out_df["orientation"] = extract_text_series(df, found["orientation"])
    out_df["decoration"] = extract_text_series(df, found["decoration"])

    build_year_series = extract_numeric_series(df, found["build_year"])
    out_df["build_year"] = build_year_series.where(build_year_series > 0).round().astype("Int64")

    total_price_series = extract_numeric_series(df, found["total_price"])
    total_price_unit = detect_price_unit(total_price_series)
    if total_price_unit == "yuan":
        out_df["total_price_wan"] = (total_price_series / 10000.0).round(4)
    else:
        out_df["total_price_wan"] = total_price_series.round(4)

    unit_price_series = extract_numeric_series(df, found["unit_price"])
    if found["unit_price"] is not None:
        out_df["unit_price_yuan_per_sqm"] = unit_price_series.where(unit_price_series > 10000, unit_price_series * 10000).round().astype("Int64")
    else:
        out_df["unit_price_yuan_per_sqm"] = pd.Series([pd.NA] * len(df), index=df.index, dtype="Int64")

    missing_unit_price = out_df["unit_price_yuan_per_sqm"].isna() & out_df["area_sqm"].notna() & out_df["total_price_wan"].notna()
    out_df.loc[missing_unit_price, "unit_price_yuan_per_sqm"] = (
        out_df.loc[missing_unit_price, "total_price_wan"] * 10000 / out_df.loc[missing_unit_price, "area_sqm"]
    ).round().astype("Int64")

    out_df["source_file"] = str(path.name)
    out_df["source"] = "anjuke_kaggle"

    out_df = out_df.reset_index(drop=True)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = out_dir / f"anjuke_ingested_{ts}.csv"
    out_json = out_dir / f"anjuke_ingested_{ts}.json"
    out_df.to_csv(out_csv, index=False)
    out_df.to_json(out_json, orient="records", force_ascii=False, indent=2)
    return out_csv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="输入文件或目录（CSV 或包含 CSV 的 ZIP）")
    p.add_argument("--output-dir", default="data/raw", help="输出目录")
    args = p.parse_args()

    inp = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files: List[Path] = []
    if inp.is_dir():
        files = [path for path in inp.glob("*.csv") if not path.name.startswith("anjuke_ingested_")]
    elif inp.is_file():
        # 处理 zip 中的 csv（若是 kaggle 下载的 zip）
        if inp.suffix.lower() == ".zip":
            # 解压到临时目录
            import zipfile

            with zipfile.ZipFile(inp, "r") as z:
                tmpdir = out_dir / ("_unzipped_" + inp.stem)
                tmpdir.mkdir(exist_ok=True)
                z.extractall(tmpdir)
                files = list(tmpdir.glob("*.csv"))
        else:
            files = [inp]
    else:
        raise SystemExit(f"找不到输入文件或目录: {inp}")

    if not files:
        raise SystemExit("未找到任何 CSV 文件可处理")

    results: List[Path] = []
    for f in files:
        print(f"Processing {f}...")
        out = ingest_file(f, out_dir)
        print(f"Wrote standardized CSV: {out}")
        results.append(out)


if __name__ == "__main__":
    main()
