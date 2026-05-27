# 房价预测模型报告

## 1. 训练目标

以 `total_price_wan` 作为预测目标，使用清洗后的 Anjuke 二手房数据训练房价回归模型。

## 2. 特征设计

用于训练的主要特征包括：

- 数值特征：`area_sqm`、`build_year`、`house_age`、`floor_total`、`area_log`
- 类别特征：`region`、`orientation`、`decoration`、`floor_category`

## 3. 模型方案

- 基线模型：Ridge 回归
- 主模型：XGBoost 回归
- 评估方式：80/20 切分验证 + 3 折交叉验证 MAE

## 4. 训练结果

本次训练得到的测试集指标如下：

- Baseline Ridge
  - MAE: 203.93
  - RMSE: 499.68
  - R2: 0.6298
- XGBoost
  - MAE: 148.81
  - RMSE: 403.45
  - R2: 0.7587
- XGBoost 3 折交叉验证 MAE
  - Mean: 149.52
  - Std: 0.51

## 5. 产物输出

- 模型文件：`models/house_price_model.pkl`
- 指标文件：`models/training_metrics.json`
- 特征列配置：`models/feature_columns.json`
- 特征重要性图：`reports/figures/feature_importance.png`
- SHAP 图：`reports/figures/shap_summary.png`、`reports/figures/shap_bar.png`

## 6. 说明

当前环境下，直接使用 `shap.TreeExplainer(xgboost)` 会遇到版本兼容问题，因此解释图采用了模型无关的 SHAP 方式生成，确保可稳定输出。