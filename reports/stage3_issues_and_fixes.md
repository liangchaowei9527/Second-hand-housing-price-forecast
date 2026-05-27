# 第三阶段问题与修复记录

## 1. 阶段目标

第三阶段的目标是完成二手房价格预测的特征工程、模型训练、模型评估和可解释性输出，形成一条可直接复用的建模链路。

## 2. 遇到的主要问题

### 2.1 特征工程文件还是空壳

最开始的 `src/features/build_features.py`、`src/models/train.py`、`src/models/predict.py`、`src/models/explain.py` 都只有占位函数，没有真实逻辑，导致后续无法直接进入训练。

### 2.2 特征表里保留了 `pd.NA`

训练阶段最先遇到的问题是，特征工程输出里还保留了 pandas 的可空标记 `pd.NA`，`SimpleImputer` 在处理时把列视为 object，触发了 sklearn 的类型错误。

### 2.3 `build_year` 缺失时，房龄计算会崩溃

在房龄派生特征里，部分记录的 `build_year` 为空或者来自楼层文本推断失败，导致 `float(NaN)` / `int(NaN)` 这类转换报错。

### 2.4 SHAP 和当前 xgboost 版本存在兼容问题

使用 `shap.TreeExplainer(xgboost)` 时，当前环境下会遇到 `base_score` 类型转换错误，导致解释阶段无法稳定执行。

### 2.5 训练后生成中文图表时出现字体 warning

模型训练和 EDA 图表里有中文区域名、装修名、楼层名，当前环境默认字体没有覆盖这些字形，所以会出现中文 glyph missing 的 warning。

### 2.6 SHAP 图生成时出现 future warning

SHAP 在绘图过程中触发了 NumPy 全局随机数相关的 future warning，提示以后版本会改变 RNG 行为。

## 3. 处理思路

### 3.1 先打通最小可用链路

我先把空壳模块补成完整链路，优先确保能完成特征工程、训练、预测、解释四步闭环，再处理图表 warning 这类体验问题。

### 3.2 统一特征工程口径

把房价建模所需的共用逻辑集中到 `src/features/build_features.py` 和 `src/utils/helpers.py`，统一处理：

- 房龄派生
- 楼层类别拆分
- 建年文本回填
- 类别和数值字段规范化

### 3.3 为训练前清洗补上兼容性转换

在进入 sklearn 前，把特征表中的 pandas 空值统一转成 `np.nan`，避免 imputer 把整列误判为 object。

### 3.4 SHAP 改为模型无关解释方式

由于 `TreeExplainer(xgboost)` 版本兼容性不稳定，我改成 `shap.Explainer(lambda data: model.predict(data), background)` 的方式生成解释图，保证结果可以稳定输出。

## 4. 最终结果

第三阶段已经完成并验证通过，当前产物包括：

- 模型文件：`models/house_price_model.pkl`
- 指标文件：`models/training_metrics.json`
- 特征列配置：`models/feature_columns.json`
- 模型对比图：`reports/figures/model_metrics_comparison.png`
- 特征重要性图：`reports/figures/feature_importance.png`
- SHAP 图：`reports/figures/shap_summary.png`、`reports/figures/shap_bar.png`

训练结果上，XGBoost 明显优于 Ridge 基线模型，说明特征工程和模型选择都起到了正向作用。

## 5. 关于 warning 的判断

### 5.1 中文字体 warning

这个问题不难修，通常有两种处理方式：

- 给 matplotlib / SHAP 显式指定支持中文的字体
- 在生成图之前统一设置字体回退策略

它主要影响图表渲染效果，不影响模型训练结果。

### 5.2 SHAP future warning

这个 warning 也不难处理，但要看你想要的是：

- 只是不显示 warning
- 还是顺便把相关调用改成新版推荐写法

它主要是兼容性提示，不是当前功能错误。

## 6. 后续建议

如果你后面确认要继续，我建议优先顺手修中文字体 warning，因为它会直接影响 EDA 和解释图的观感。SHAP future warning 可以放到第二步一起处理，不影响现在的功能交付。