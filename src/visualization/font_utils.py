"""Matplotlib font helpers for Chinese chart rendering."""

from __future__ import annotations

import matplotlib
from matplotlib import font_manager

CHINESE_FONT_CANDIDATES: tuple[str, ...] = (
    "Microsoft YaHei",
    "Noto Sans SC",
    "SimHei",
    "SimSun",
    "Arial Unicode MS",
)

def configure_chinese_font() -> str:
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    selected_fonts = [font_name for font_name in CHINESE_FONT_CANDIDATES if font_name in available_fonts]
    if not selected_fonts:
        selected_fonts = ["DejaVu Sans"]

    matplotlib.rcParams["font.family"] = [selected_fonts[0]]
    matplotlib.rcParams["font.sans-serif"] = selected_fonts
    matplotlib.rcParams["axes.unicode_minus"] = False
    return selected_fonts[0]