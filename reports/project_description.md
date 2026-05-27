# 项目说明文档

## 1. 项目定位
本项目是一个基于 Anjuke 二手房公开数据的端到端实战作品，覆盖数据接入、清洗分析、建模预测、可解释性分析与 Streamlit 应用展示。

## 2. 目标与范围
- 预测目标：总价（万元）
- 辅助指标：单价（元/平方米）
- 地图粒度：区级热度分布
- 技术栈：Python + Streamlit + XGBoost/LightGBM + SHAP

## 3. 功能概览
- 首页概览：展示样本量、均价、中位价、房龄等核心指标
- 数据分析：筛选 + 分布/关系图 + 明细预览
- 地图热力图：按区域展示价格热度和样本量
- 价格预测：输入房源特征，输出预测总价并支持情景修正
- 模型解释：展示训练指标与 SHAP 图

## 4. 数据说明
- 原始数据目录：`data/raw/`
- 标准化中间结果：`data/raw/anjuke_ingested_*.csv`
- 清洗后建模数据：`data/processed/anjuke_cleaned.csv`

统一字段示例：
- `region`：区域
- `area_sqm`：建筑面积（平方米）
- `build_year`：建成年份
- `total_price_wan`：总价（万元）
- `unit_price_yuan_per_sqm`：单价（元/平方米）
- `house_age`：房龄

## 5. 模型产物
- 模型文件：`models/house_price_model.pkl`
- 指标文件：`models/training_metrics.json`
- 特征列：`models/feature_columns.json`
- 图表产物：`reports/figures/`

## 6. 当前完成状态
- 阶段一到阶段四：已完成
- 阶段五：已启动（文档与发布材料完善中）

## 7. 面向 GitHub 发布的核心亮点
- 完整数据链路（采集/标准化 -> 清洗 -> 建模 -> 展示）
- 同时提供预测与可解释性结果
- 可直接本地运行和演示
