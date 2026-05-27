# GitHub 发布清单

## 1. 发布前代码检查
- 确认 Streamlit 应用可在本地启动。
- 确认五个页面可正常切换。
- 确认模型文件与指标文件存在。
- 确认截图目录包含 5 张页面截图。

## 2. 文档检查
- README 已更新为当前 Anjuke 方案。
- 项目说明文档已完善。
- 部署说明已覆盖本地与云端方式。

## 3. 建议提交内容
- `README.md`
- `app/streamlit_app.py`
- `reports/project_description.md`
- `reports/deployment_guide.md`
- `reports/github_release_checklist.md`
- `assets/screenshots/`
- `models/`（如仓库策略允许提交模型产物）

## 4. 建议标签
- `v1.0.0`: 首个可演示版本（含五页应用、模型预测、SHAP 解释）

## 5. Release 文案模板
标题：v1.0.0 二手房价格分析与预测平台首发

要点：
- 完成从数据接入到可视化展示的端到端流程
- 提供五页 Streamlit 交互平台
- 集成模型预测与 SHAP 可解释分析
- 附带部署说明与演示截图
