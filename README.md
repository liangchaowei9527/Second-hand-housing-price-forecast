# 二手房价格影响因素分析与预测系统

## 项目简介
本项目基于 Kaggle 上的 Anjuke 二手房公开数据，构建了一个可本地运行的 Streamlit 多页面平台，覆盖：
- 数据接入与标准化
- 数据清洗与探索分析
- 房价预测建模
- SHAP 可解释性分析
- 可视化交互展示

## 技术栈
- Python 3.10
- Streamlit
- Pandas / NumPy
- Scikit-learn
- XGBoost / LightGBM
- SHAP
- Plotly / Matplotlib / Seaborn

## 目录结构
```text
二手房价预测/
├── app/
│   └── streamlit_app.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── models/
│   ├── house_price_model.pkl
│   ├── feature_columns.json
│   └── training_metrics.json
├── reports/
│   ├── data_cleaning_eda_summary.md
│   ├── model_report.md
│   ├── project_description.md
│   ├── deployment_guide.md
│   └── figures/
├── assets/
│   └── screenshots/
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── visualization/
│   └── config/
├── requirements.txt
└── plan.md
```

## 快速开始

### 1. 安装依赖
```powershell
py -3.10 -m pip install -r requirements.txt
```

### 2. 数据接入（Kaggle）
下载并解压数据：
```powershell
kaggle datasets download -d b2eeze/second-hand-house-prices-from-the-anjuke-website --unzip -p data/raw
```

标准化字段：
```powershell
py -3.10 src/data/ingest_kaggle_anjuke.py --input data/raw/second-hand-house-prices-from-the-anjuke-website.csv --output-dir data/raw
```

### 3. 数据清洗与 EDA
```powershell
py -3.10 src/data/clean.py --input-dir data/raw --output-dir data/processed --figures-dir reports/figures
```

### 4. 模型训练与解释
训练模型并输出指标：
```powershell
py -3.10 src/models/train.py --data-path data/processed/anjuke_cleaned.csv --model-path models/house_price_model.pkl --metrics-path models/training_metrics.json --figures-dir reports/figures
```

生成 SHAP 图：
```powershell
py -3.10 src/models/explain.py --data-path data/processed/anjuke_cleaned.csv --model-path models/house_price_model.pkl --figures-dir reports/figures
```

### 5. 启动平台
```powershell
py -3.10 -m streamlit run app/streamlit_app.py
```

## 平台页面
- 首页概览：核心统计指标与趋势概览
- 数据分析：分布、关系、筛选明细
- 地图热力图：区级价格热度与样本分布
- 价格预测：输入特征并输出总价预测
- 模型解释：指标对比与 SHAP 解释

## 项目产物
- 清洗数据：data/processed/anjuke_cleaned.csv
- 训练模型：models/house_price_model.pkl
- 模型指标：models/training_metrics.json
- SHAP 图：reports/figures/shap_summary.png、reports/figures/shap_bar.png
- 页面截图：assets/screenshots/01_home.png ~ 05_explanation.png

## 文档索引
- 项目计划：plan.md
- 数据清洗与 EDA：reports/data_cleaning_eda_summary.md
- 模型报告：reports/model_report.md
- 阶段三问题修复：reports/stage3_issues_and_fixes.md
- 项目说明：reports/project_description.md
- 部署说明：reports/deployment_guide.md
- 发布清单：reports/github_release_checklist.md

## 部署
请参考部署文档：reports/deployment_guide.md

## 许可与声明
本项目数据来源于公开数据集，仅用于学习、课程设计与作品集展示。
