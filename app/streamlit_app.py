"""Streamlit application entry point."""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportMissingTypeStubs=false, reportUnnecessaryIsInstance=false

from __future__ import annotations

from contextlib import contextmanager
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
from plotly.graph_objs import Figure
import streamlit as st

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config.settings import CLEANED_DATA_PATH, FIGURES_DIR, TRAINED_MODEL_PATH, TRAINING_METRICS_PATH
from src.features.build_features import MODEL_FEATURES, build_feature_frame
from src.models.predict import predict_price
from src.visualization.font_utils import configure_chinese_font


st.set_page_config(
    page_title="二手房价格分析与预测平台",
    layout="wide",
    initial_sidebar_state="expanded",
)

configure_chinese_font()


CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=ZCOOL+XiaoWei&family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {
        --bg-0: #ffffff;
        --bg-1: #f7f9fb;
        --ink-1: #0f172a;
        --ink-2: #334155;
        --ink-3: #6b7280;
        --line-1: rgba(15, 23, 42, 0.08);
        --line-2: rgba(15, 23, 42, 0.06);
        --brand-1: #1f6feb;
        --brand-2: #0f4cd4;
        --muted: #f1f5f9;
    }

    .stApp {
        background: linear-gradient(180deg, var(--bg-1) 0%, var(--bg-0) 100%);
    }

    .stApp, .stMarkdown, .stDataFrame, .stMetric {
        font-family: 'Manrope', 'Microsoft YaHei', sans-serif;
    }

    [data-testid="stSidebar"] {
        display: none;
    }

    h1, h2, h3, .hero-title {
        font-family: 'ZCOOL XiaoWei', 'Microsoft YaHei', serif !important;
        letter-spacing: 0.02em;
    }

    .hero {
        position: relative;
        overflow: hidden;
        padding: 1.45rem 1.4rem;
        border-radius: 0.9rem;
        background:
            linear-gradient(180deg, rgba(31,111,235,0.10), rgba(255,255,255,0.92)),
            linear-gradient(135deg, rgba(31,111,235,0.04), rgba(15,76,212,0.02));
        color: var(--ink-1);
        border: 1px solid var(--line-1);
        box-shadow: 0 14px 36px rgba(15, 23, 42, 0.08);
        margin-bottom: 1.15rem;
        animation: hero-enter 540ms ease-out, hero-float 7s ease-in-out 540ms infinite;
    }

    .hero::after {
        content: "";
        position: absolute;
        inset: auto -8% -32% auto;
        width: 16rem;
        height: 16rem;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(31,111,235,0.12) 0%, rgba(31,111,235,0.02) 48%, transparent 70%);
        pointer-events: none;
    }

    .top-nav-wrap {
        background: transparent;
        border-bottom: 1px solid var(--line-2);
        padding: 0.35rem 0.3rem 0.2rem;
        margin-bottom: 0.75rem;
    }

    .top-nav-head {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 0.8rem;
        margin-bottom: 0.4rem;
        font-size: 0.82rem;
        color: var(--ink-3);
    }

    .top-nav-title {
        color: var(--ink-2);
        font-weight: 700;
        letter-spacing: 0.01em;
    }

    div[data-testid="stSegmentedControl"] {
        margin-top: 0.1rem;
    }

    /* keep segmented controls minimal if present */
    div[data-testid="stSegmentedControl"] [role="radiogroup"] {
        background: transparent;
        padding: 0.08rem;
    }

    div[data-testid="stSegmentedControl"] label {
        padding: 0.28rem 0.6rem !important;
        min-height: 36px;
        font-weight: 600 !important;
        color: var(--ink-2);
    }

    div[data-testid="stSegmentedControl"] label[data-selected="true"] {
        background: transparent !important;
        color: var(--brand-1) !important;
        border-bottom: 2px solid var(--brand-1) !important;
        box-shadow: none;
    }

    /* Streamlit tabs 样式统一，避免主题或外部样式覆盖成红色 */
    div[data-testid="stTabs"] [role="tablist"] {
        background: transparent;
        padding: 0;
        margin-top: 0.35rem;
    }

    div[data-testid="stTabs"] [role="tab"] {
        color: var(--ink-2) !important;
        background: transparent !important;
        border: none !important;
        padding: 8px 14px !important;
        margin-right: 6px !important;
        font-weight: 600 !important;
        border-radius: 4px !important;
    }

    div[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
        color: var(--brand-1) !important;
        border-bottom: 2px solid var(--brand-1) !important;
        background: transparent !important;
        box-shadow: none !important;
    }

    .hero h1, .hero p {
        margin: 0;
    }

    .hero-meta {
        margin-top: 0.55rem;
        font-size: 0.9rem;
        opacity: 0.92;
    }

    .section-card {
        background: white;
        border: 1px solid var(--line-2);
        border-radius: 0.6rem;
        padding: 0.9rem 0.95rem;
    }

    .card-shell {
        margin-bottom: 1rem;
    }

    .metric-card {
        background: white;
        border-radius: 0.6rem;
        padding: 0.8rem;
        border: 1px solid var(--line-2);
        min-height: 118px;
    }

    .small-label {
        color: var(--ink-3);
        font-size: 0.88rem;
        margin-bottom: 0.25rem;
    }

    .big-value {
        font-size: 1.68rem;
        font-weight: 800;
        color: var(--ink-1);
    }

    .hint {
        color: var(--ink-2);
        font-size: 0.9rem;
    }

    .tiny-pill {
        display: inline-block;
        background: transparent;
        color: var(--ink-3);
        padding: 0.12rem 0.46rem;
        border-radius: 99px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-bottom: 0.35rem;
        border: 1px solid var(--muted);
    }

    .insight-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.8rem;
    }

    .insight-item {
        background: white;
        border: 1px solid var(--line-2);
        border-radius: 0.6rem;
        padding: 0.6rem 0.65rem;
    }

    .insight-item .k {
        color: var(--ink-3);
        font-size: 0.82rem;
    }

    .insight-item .v {
        margin-top: 0.2rem;
        color: var(--ink-1);
        font-weight: 800;
    }

    @keyframes fade-up {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes hero-enter {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes hero-float {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-3px); }
    }

    @media (max-width: 900px) {
        .hero { padding: 1rem 0.8rem; border-radius: 0.6rem; }
        .big-value { font-size: 1.2rem; }
        .insight-grid { grid-template-columns: 1fr; }
    }
</style>
"""


@st.cache_data(show_spinner=False)
def load_cleaned_data() -> pd.DataFrame:
    if not CLEANED_DATA_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(CLEANED_DATA_PATH)


@st.cache_resource(show_spinner=False)
def load_model_bundle() -> dict[str, Any] | None:
    if not TRAINED_MODEL_PATH.exists():
        return None
    with TRAINED_MODEL_PATH.open("rb") as handle:
        return pickle.load(handle)


@st.cache_data(show_spinner=False)
def load_metrics() -> dict[str, Any]:
    if not TRAINING_METRICS_PATH.exists():
        return {}
    import json

    with TRAINING_METRICS_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def safe_round(value: Any, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):,.{digits}f}"


def format_int(value: Any) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{int(round(float(value))):,}"


def apply_style() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def build_header(title: str, subtitle: str) -> None:
    now_label = datetime.now().strftime("%Y-%m-%d")
    st.markdown(
        f"""
        <div class=\"hero\">
            <div class=\"tiny-pill\">城市住宅视角</div>
            <h1>{title}</h1>
            <p>{subtitle}</p>
            <div class=\"hero-meta\">更新日期：{now_label} · 数据集：Anjuke 二手房公开样本</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@contextmanager
def card_container(tag: str | None = None):
    with st.container(border=True):
        st.markdown('<div class="card-shell">', unsafe_allow_html=True)
        if tag:
            st.markdown(f'<span class="tiny-pill">{tag}</span>', unsafe_allow_html=True)
        yield
        st.markdown('</div>', unsafe_allow_html=True)


def apply_chart_theme(fig: Figure, *, mode: str = "bar") -> Figure:
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Manrope, Microsoft YaHei, sans-serif", color="#334155"),
        colorway=["#1f6feb", "#4f8ff7", "#7fb0ff", "#a8c8ff"],
        margin=dict(l=0, r=0, t=24, b=0),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(15, 23, 42, 0.06)", zeroline=False, title_standoff=10)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(15, 23, 42, 0.06)", zeroline=False, title_standoff=10)
    if mode == "hist":
        fig.update_traces(marker_color="#1f6feb", marker_line_width=0)
    elif mode == "bar":
        fig.update_traces(marker_color="#1f6feb")
    elif mode == "scatter":
        fig.update_traces(marker=dict(color="#1f6feb", line=dict(width=0)))
    return fig


def flatten_metrics(metrics: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for name, payload in metrics.items():
        if isinstance(payload, dict):
            row = {"name": str(name)}
            for key in ("mae", "rmse", "r2", "mean", "std", "folds"):
                if key in payload:
                    row[key] = safe_round(payload[key]) if key != "folds" else str(payload[key])
            rows.append(row)
        else:
            rows.append({"name": str(name), "value": str(payload)})
    return rows


def render_top_navigation() -> str:
    data_state = CLEANED_DATA_PATH.name if CLEANED_DATA_PATH.exists() else "数据缺失"
    model_state = TRAINED_MODEL_PATH.name if TRAINED_MODEL_PATH.exists() else "模型缺失"
    st.markdown(
        f"""
        <div class=\"top-nav-wrap\">
            <div class=\"top-nav-head\">
                <span class=\"top-nav-title\">导航</span>
                <span style=\"color:var(--ink-3)\">数据: {data_state} · 模型: {model_state}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    # 使用更传统的选项卡导航，直接在选项卡内渲染对应页面
    tabs = st.tabs(["首页概览", "数据分析", "地图热力图", "价格预测", "模型解释"])
    with tabs[0]:
        render_overview(load_cleaned_data())
    with tabs[1]:
        render_analysis(load_cleaned_data())
    with tabs[2]:
        render_heatmap(load_cleaned_data())
    with tabs[3]:
        render_prediction(load_cleaned_data())
    with tabs[4]:
        render_explanation()

    return "tabs-rendered"


def render_metric_cards(metrics: list[tuple[str, str, str]]) -> None:
    columns = st.columns(len(metrics))
    for column, (label, value, hint) in zip(columns, metrics):
        with column:
            st.markdown(
                f"""
                <div class=\"metric-card\">
                    <div class=\"small-label\">{label}</div>
                    <div class=\"big-value\">{value}</div>
                    <div class=\"hint\">{hint}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _column_or_default(df: pd.DataFrame, column: str, default: Any = None) -> pd.Series:
    if column in df.columns:
        return df[column]
    return pd.Series([default] * len(df), index=df.index)


def _unique_choices(series: pd.Series, fallback: list[str]) -> list[str]:
    values = [str(value) for value in series.dropna().astype(str).unique().tolist() if str(value).strip()]
    values = sorted(dict.fromkeys(values))
    return values or fallback


def _filter_dataset(df: pd.DataFrame, region: str, orientation: str, decoration: str) -> pd.DataFrame:
    filtered = df.copy()
    if region != "全部":
        filtered = filtered[filtered["region"].astype(str) == region]
    if orientation != "全部":
        filtered = filtered[filtered["orientation"].astype(str) == orientation]
    if decoration != "全部":
        filtered = filtered[filtered["decoration"].astype(str) == decoration]
    return filtered


def render_overview(df: pd.DataFrame) -> None:
    build_header(
        "二手房价格影响因素分析与预测平台",
        "样本、区域、价格、模型概览。",
    )

    if df.empty:
        st.warning("未找到清洗后数据，请先完成阶段二的数据处理。")
        return

    regions = _column_or_default(df, "region")
    metrics = [
        ("样本量", format_int(len(df)), "清洗后可建模样本数"),
        ("区域数", format_int(regions.nunique(dropna=True)), "覆盖的行政区/板块数量"),
        ("平均总价(万)", safe_round(df["total_price_wan"].mean()), "当前样本的价格中枢"),
        ("平均单价(元/㎡)", format_int(df["unit_price_yuan_per_sqm"].mean()), "反映单位面积价格水平"),
        ("平均房龄", safe_round(df["house_age"].mean()), "样本整体的成熟度"),
    ]
    render_metric_cards(metrics)

    st.write("")
    left, right = st.columns([1.2, 1])
    with left:
        with card_container("市场分布"):
            st.subheader("价格分布")
            st.caption("价格带分布")
            fig = px.histogram(df, x="total_price_wan", nbins=35, marginal="box")
            fig.update_layout(height=380, xaxis_title="总价(万)", yaxis_title="数量")
            st.plotly_chart(apply_chart_theme(fig, mode="hist"), width="stretch")
    with right:
        with card_container("区域层级"):
            st.subheader("区域均价 Top 10")
            st.caption("均价最高的区域")
            top_region = (
                df.dropna(subset=["region", "total_price_wan"])
                .groupby("region", as_index=False)["total_price_wan"]
                .mean()
                .sort_values("total_price_wan", ascending=False)
                .head(10)
            )
            if top_region.empty:
                st.info("区域数据不足，暂无法展示 Top 10。")
            else:
                fig = px.bar(top_region, x="total_price_wan", y="region", orientation="h")
                fig.update_layout(height=380, xaxis_title="平均总价(万)", yaxis_title="区域")
                st.plotly_chart(apply_chart_theme(fig, mode="bar"), width="stretch")

    with card_container("洞察"):
        st.subheader("核心结论速览")
        st.markdown(
            """
            <div class="insight-grid">
                <div class="insight-item"><div class="k">面积与总价</div><div class="v">正相关</div></div>
                <div class="insight-item"><div class="k">区域差异</div><div class="v">明显</div></div>
                <div class="insight-item"><div class="k">模型表现</div><div class="v">树模型更稳</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_analysis(df: pd.DataFrame) -> None:
    build_header("数据分析", "分布、关联、明细。")

    if df.empty:
        st.warning("未找到清洗后数据。")
        return

    region_choices = _unique_choices(df["region"], ["全部"])
    orientation_choices = _unique_choices(df["orientation"], ["全部"])
    decoration_choices = _unique_choices(df["decoration"], ["全部"])

    with card_container("交互筛选"):
        st.caption("筛选条件")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            region = st.selectbox("区域", ["全部"] + region_choices, index=0)
        with c2:
            orientation = st.selectbox("朝向", ["全部"] + orientation_choices, index=0)
        with c3:
            decoration = st.selectbox("装修", ["全部"] + decoration_choices, index=0)
        with c4:
            max_rows = st.number_input("预览行数", min_value=5, max_value=100, value=20, step=5)
        area_min = float(pd.to_numeric(df["area_sqm"], errors="coerce").quantile(0.05))
        area_max = float(pd.to_numeric(df["area_sqm"], errors="coerce").quantile(0.95))
        if not np.isfinite(area_min):
            area_min = 20.0
        if not np.isfinite(area_max):
            area_max = 250.0
        area_min_int = int(max(20, round(area_min)))
        area_max_int = int(max(area_min_int + 1, round(area_max)))
        area_range_max = int(max(200, area_max_int + 10))
        area_range = st.slider(
            "面积范围(㎡)",
            min_value=20,
            max_value=area_range_max,
            value=(area_min_int, min(max(80, area_max_int), area_range_max)),
            step=1,
        )

    filtered = _filter_dataset(df, region, orientation, decoration)
    filtered = filtered[(filtered["area_sqm"] >= area_range[0]) & (filtered["area_sqm"] <= area_range[1])]
    st.caption(f"筛选后样本量：{len(filtered):,} 条")

    top_left, top_right = st.columns([1.15, 0.85])
    with top_left:
        with card_container("关系图"):
            st.subheader("面积与总价")
            st.caption("面积-总价")
            fig = px.scatter(filtered, x="area_sqm", y="total_price_wan", opacity=0.58)
            fig.update_layout(height=380, xaxis_title="面积(㎡)", yaxis_title="总价(万)")
            st.plotly_chart(apply_chart_theme(fig, mode="scatter"), width="stretch")
    with top_right:
        with card_container("结构图"):
            st.subheader("房龄分布")
            st.caption("房龄分布")
            age_frame = filtered.dropna(subset=["house_age"])
            if age_frame.empty:
                st.info("当前筛选条件下没有可用房龄数据。")
            else:
                fig = px.histogram(age_frame, x="house_age", nbins=25)
                fig.update_layout(height=380, xaxis_title="房龄", yaxis_title="数量")
                st.plotly_chart(apply_chart_theme(fig, mode="hist"), width="stretch")

    lower_left, lower_right = st.columns([1, 1])
    with lower_left:
        with card_container("分层"):
            st.subheader("楼层类别均价")
            st.caption("楼层均价")
            if "floor_category" in filtered.columns:
                floor_summary = (
                    filtered.dropna(subset=["floor_category", "total_price_wan"])
                    .groupby("floor_category", as_index=False)["total_price_wan"]
                    .mean()
                    .sort_values("total_price_wan", ascending=False)
                )
                if floor_summary.empty:
                    st.info("没有足够数据计算楼层类别均价。")
                else:
                    fig = px.bar(floor_summary, x="floor_category", y="total_price_wan")
                    fig.update_layout(height=320, xaxis_title="楼层类别", yaxis_title="平均总价(万)")
                    st.plotly_chart(apply_chart_theme(fig, mode="bar"), width="stretch")
    with lower_right:
        with card_container("明细"):
            st.subheader("筛选结果预览")
            st.caption("明细表")
            preview_columns = [
                column
                for column in [
                    "community",
                    "region",
                    "area_sqm",
                    "floor",
                    "orientation",
                    "decoration",
                    "build_year",
                    "total_price_wan",
                    "unit_price_yuan_per_sqm",
                ]
                if column in filtered.columns
            ]
            st.dataframe(filtered[preview_columns].head(int(max_rows)), width="stretch", hide_index=True)


def render_heatmap(df: pd.DataFrame) -> None:
    build_header("地图热力图", "区级价格热度。")

    if df.empty:
        st.warning("未找到清洗后数据。")
        return

    region_price = (
        df.dropna(subset=["region", "total_price_wan"])
        .groupby("region", as_index=False)
        .agg(avg_price=("total_price_wan", "mean"), samples=("total_price_wan", "size"), avg_area=("area_sqm", "mean"))
        .sort_values("avg_price", ascending=False)
    )
    if region_price.empty:
        st.info("没有足够的区域数据生成热力图。")
        return

    with card_container("热力层"):
        st.subheader("区级价格热力排名")
        st.caption("区级热度")
        fig = px.scatter(
            region_price,
            x="samples",
            y="avg_price",
            size="avg_area",
            hover_name="region",
        )
        fig.update_layout(height=440, xaxis_title="样本量", yaxis_title="平均总价(万)")
        st.plotly_chart(apply_chart_theme(fig, mode="scatter"), width="stretch")

    st.write("")
    with card_container("排行表"):
        st.subheader("区域热力表")
        st.caption("区域排名")
        heatmap_frame = region_price.head(20).set_index("region")[["avg_price", "samples", "avg_area"]]
        st.dataframe(heatmap_frame.style.background_gradient(cmap="Blues"), width="stretch")


def _prediction_inputs(bundle: dict[str, Any] | None, df: pd.DataFrame) -> pd.DataFrame:
    if bundle is None:
        return pd.DataFrame()

    defaults = build_feature_frame(df) if not df.empty else pd.DataFrame(columns=MODEL_FEATURES)

    def median_or(default_series: pd.Series, fallback: float) -> float:
        value = pd.to_numeric(default_series, errors="coerce").median()
        return float(value) if pd.notna(value) else fallback

    region_choices = _unique_choices(defaults.get("region", pd.Series(dtype="object")), ["未知"])
    orientation_choices = _unique_choices(defaults.get("orientation", pd.Series(dtype="object")), ["未知"])
    decoration_choices = _unique_choices(defaults.get("decoration", pd.Series(dtype="object")), ["未知"])
    floor_choices = _unique_choices(defaults.get("floor_category", pd.Series(dtype="object")), ["other"])

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            area_sqm = st.number_input("建筑面积(㎡)", min_value=20.0, max_value=1000.0, value=median_or(defaults.get("area_sqm", pd.Series(dtype="float64")), 90.0), step=1.0)
            build_year = st.number_input("建成年份", min_value=1950, max_value=2035, value=int(round(median_or(defaults.get("build_year", pd.Series(dtype="float64")), 2012))), step=1)
        with col2:
            floor_total = st.number_input("总楼层", min_value=1, max_value=100, value=int(round(median_or(defaults.get("floor_total", pd.Series(dtype="float64")), 18))), step=1)
            region = st.selectbox("区域", region_choices, index=0)
        with col3:
            orientation = st.selectbox("朝向", orientation_choices, index=0)
            decoration = st.selectbox("装修", decoration_choices, index=0)

        floor_category = st.selectbox("楼层类别", floor_choices, index=0)
        submitted = st.form_submit_button("预测总价")

    if not submitted:
        return pd.DataFrame()

    payload = pd.DataFrame(
        [
            {
                "area_sqm": area_sqm,
                "build_year": build_year,
                "house_age": max(pd.Timestamp.now().year - build_year, 0),
                "floor_total": floor_total,
                "area_log": float(np.log1p(area_sqm)),
                "region": region,
                "orientation": orientation,
                "decoration": decoration,
                "floor_category": floor_category,
            }
        ]
    )
    return payload


def render_prediction(df: pd.DataFrame) -> None:
    build_header("价格预测", "输入核心特征，输出总价预测。")
    bundle = load_model_bundle()
    if bundle is None:
        st.warning("未找到训练好的模型文件，请先完成阶段三。")
        return

    with card_container("输入"):
        st.subheader("预测输入")
        st.caption("核心字段")
        prediction_frame = _prediction_inputs(bundle, df)

    if prediction_frame.empty:
        st.info("填写表单后点击预测按钮。")
        return

    prediction = predict_price(prediction_frame, model_path=TRAINED_MODEL_PATH)
    predicted_price = float(prediction.iloc[0])

    st.write("")
    scenario_delta = st.slider("市场情景修正(%)", min_value=-15, max_value=15, value=0, step=1, help="用于模拟市场波动，不会改动原模型。")
    adjusted_price = predicted_price * (1 + scenario_delta / 100)
    left, right = st.columns([0.9, 1.1])
    with left:
        with card_container("结果"):
            st.subheader("预测结果")
            st.caption("预测值")
            st.metric("预测总价(万)", f"{predicted_price:,.2f}")
            st.metric("情景后总价(万)", f"{adjusted_price:,.2f}", f"{scenario_delta:+d}%")
    with right:
        with card_container("输入快照"):
            st.subheader("输入摘要")
            st.caption("已输入特征")
            st.dataframe(prediction_frame, width="stretch", hide_index=True)


def render_explanation() -> None:
    build_header("模型解释", "模型指标与特征。")
    metrics = load_metrics()

    with card_container("评估"):
        st.subheader("模型指标")
        st.caption("指标总览")
        if metrics:
            comparison = st.columns(3)
            with comparison[0]:
                st.metric("基线 MAE", safe_round(metrics.get("baseline_ridge", {}).get("mae")))
            with comparison[1]:
                st.metric("主模型 MAE", safe_round(metrics.get("tree_model", {}).get("mae")))
            with comparison[2]:
                cv_info = metrics.get("tree_model_cv_mae", {})
                st.metric("交叉验证 MAE", safe_round(cv_info.get("mean")))

            metric_rows = flatten_metrics(metrics)
            if metric_rows:
                st.dataframe(pd.DataFrame(metric_rows), width="stretch", hide_index=True)
            with st.expander("查看原始指标 JSON", expanded=False):
                import json

                st.code(json.dumps(metrics, ensure_ascii=False, indent=2), language="json")
        else:
            st.info("未找到评估指标文件。")

    fig_left, fig_right = st.columns(2)
    for column, image_name, title in [
        (fig_left, "shap_summary.png", "SHAP Summary"),
        (fig_right, "shap_bar.png", "SHAP Bar"),
    ]:
        with column:
            with card_container("SHAP"):
                st.subheader(title)
                st.caption("特征影响")
                image_path = FIGURES_DIR / image_name
                if image_path.exists():
                    st.image(str(image_path), width="stretch")
                else:
                    st.info(f"未找到 {image_name}。")

    with card_container("特征"):
        st.subheader("特征说明")
        st.caption("输入字段")
        st.write(
            "当前模型主要依赖面积、建成年份、房龄、总楼层、对数面积以及区域、朝向、装修、楼层类别等特征。"
        )
        st.write(f"模型输入字段：{', '.join(MODEL_FEATURES)}")


def main() -> None:
    apply_style()
    # render_top_navigation 现在在选项卡内部直接渲染页面
    render_top_navigation()


if __name__ == "__main__":
    main()
