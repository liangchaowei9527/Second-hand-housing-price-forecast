# v1.0.0 二手房价格分析与预测平台首发

## 版本标题
v1.0.0 二手房价格分析与预测平台首发

## 版本亮点
- 完成从数据接入到可视化展示的端到端流程。
- 提供五页 Streamlit 交互平台，覆盖概览、分析、热力、预测、解释。
- 集成 XGBoost 主模型与 Ridge 基线对比，效果稳定提升。
- 提供 SHAP 可解释性图表，支持特征影响解读。
- 配套本地部署与云端部署说明，便于复现与展示。

## 主要更新项
- 新增并完善五个业务页面：
  - 首页概览
  - 数据分析
  - 地图热力图
  - 价格预测
  - 模型解释
- 完成阶段三模型链路产物落盘：
  - 模型文件：models/house_price_model.pkl
  - 指标文件：models/training_metrics.json
  - 特征列配置：models/feature_columns.json
  - 解释图与评估图：reports/figures/
- 补齐阶段四展示材料：
  - 页面截图：assets/screenshots/01_home.png ~ 05_explanation.png
- 启动并推进阶段五文档化：
  - 重写项目总览与运行说明：README.md
  - 新增项目说明文档：reports/project_description.md
  - 新增部署说明文档：reports/deployment_guide.md
  - 新增发布检查清单：reports/github_release_checklist.md

## 模型效果摘要
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

## 已知问题
- 当前环境下，SHAP 与部分 xgboost 版本在 TreeExplainer 路径存在兼容性问题。
- 当前版本已采用模型无关 SHAP 解释方式保障稳定输出，但解释耗时可能略高于树模型专用解释器。
- 部分中文图表在特定系统字体环境可能出现 glyph warning，不影响训练与预测结果。
- 热力图当前为区级价格热度分布，暂未包含小区级经纬度精细可视化。

## 升级建议
- 如用于线上演示，建议保留模型文件与示例数据，确保开箱可运行。
- 如用于长期维护，建议后续补充自动化测试与 CI 校验流程。

## 致谢
感谢公开数据集与开源生态支持。本项目仅用于学习、课程设计与作品集展示。
