# 部署说明

## 1. 本地部署（推荐）

### 1.1 环境要求
- Python 3.10
- Windows / macOS / Linux

### 1.2 安装依赖
```powershell
py -3.10 -m pip install -r requirements.txt
```

### 1.3 启动应用
```powershell
py -3.10 -m streamlit run app/streamlit_app.py
```

默认访问地址：
- `http://localhost:8501`

## 2. Streamlit Community Cloud 部署

### 2.1 仓库准备
确保仓库中至少包含：
- `app/streamlit_app.py`
- `requirements.txt`
- `models/house_price_model.pkl`
- `models/training_metrics.json`
- `data/processed/anjuke_cleaned.csv`（若在线端需直接展示现有数据）

### 2.2 部署步骤
1. 将项目推送到 GitHub。
2. 打开 Streamlit Community Cloud 并使用 GitHub 账号登录。
3. 选择仓库与分支。
4. Main file path 填写：`app/streamlit_app.py`。
5. 点击 Deploy。

### 2.3 常见问题
- 启动失败：检查 `requirements.txt` 是否包含 `streamlit`、`xgboost`、`lightgbm`、`shap`。
- 内存不足：减少加载数据规模或移除不必要图表。
- 文件缺失：确认模型与数据文件已提交到仓库。

## 3. 其他公网部署方式

### 3.1 Docker（可选）
可在后续补充 Dockerfile，将应用容器化后部署到云主机。

### 3.2 云服务器（可选）
在云服务器安装 Python 3.10 后，按“本地部署”步骤运行，并通过反向代理暴露端口。

## 4. 发布前检查清单
- 应用可正常启动并切换五个页面。
- 模型预测可返回结果。
- SHAP 图可正常显示。
- README 已包含运行方式与目录说明。
- 截图目录已包含页面示例图。
