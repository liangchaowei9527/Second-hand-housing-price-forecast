# Anjuke (Kaggle) 数据接入说明

本文件说明如何在本项目中使用 Kaggle 上的 "House Prices from 2024 Anjuke Website" 数据集（以下简称 `Anjuke`）。

准备工作
- 请在你的系统中安装 `kaggle` 命令行工具，并配置 API token（`kaggle.json`）。

下载数据
```powershell
kaggle datasets download -d b2eeze/second-hand-house-prices-from-the-anjuke-website --unzip -p data/raw
```

标准化（运行项目提供的 ingest 脚本）
```powershell
python src/data/ingest_kaggle_anjuke.py --input data/raw/second-hand-house-prices-from-the-anjuke-website.csv --output-dir data/raw
```

输出
- 标准化后的 CSV/JSON 将保存在 `data/raw/anjuke_ingested_YYYYMMDD_HHMMSS.csv` 和 `.json`。

统一 schema 说明（脚本会尽量匹配并生成下列字段）：

- `community`：小区名
- `region`：行政区/街道
- `area_sqm`：建筑面积（平方米）
- `floor`：楼层描述
- `orientation`：朝向
- `decoration`：装修情况
- `build_year`：建成年份（整数）
- `total_price_wan`：总价（单位：万元，float）
- `unit_price_yuan_per_sqm`：单价（单位：元/平方米，int）
- `source_file`：源 CSV 文件名
- `source`：'anjuke_kaggle'

如果你希望我把标准化后的文件进一步合并入主数据集（例如与现有 Lianjia 数据合并并去重、标准化字段顺序），回复 "合并" 并确认你想要的合并策略（保留最新/保留来源优先/按小区均价对齐）。
