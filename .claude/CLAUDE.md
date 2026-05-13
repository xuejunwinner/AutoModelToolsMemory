# AutoModelBuilderTools - 单机版模型自动化工具包

## 项目概述
支持 LightGBM / XGBoost / LR 自动化建模、模型归因分析、报告生成、BeamSearch 特征筛选的单机工具包。
主要用于信贷风控模型的训练、评估、归因分析和特征筛选。

## 核心文件

| 文件 | 说明 |
|------|------|
| `AutoModelBuilderTools.py` | 主模块，包含全部 5 个类 |
| `run_attribution_demo.py` | 归因报告演示运行脚本 |
| `test_workspace/test_all.py` | 全功能测试（67项） |
| `test_workspace/test_comprehensive.py` | 详细测试（含耗时统计） |
| `test_workspace/generate_test_data.py` | 测试数据生成 |

## 架构

### FeatureAnalysisToolkit — 特征分析工具箱
- `calculate_woe()` — WOE 计算
- `calculate_iv_detail()` — IV 计算含分箱明细（替代原 calculate_single_iv / calculate_batch_iv）
- `fit_woe_transformer()` / `apply_woe_transform()` — WOE 转换
- `calculate_psi_detail(dropna=False)` — PSI 计算含分箱明细（替代原 calculate_psi / calculate_psi_simple）
- `calculate_auc_ks(metrics='auc,ks')` — AUC/KS 合并计算（替代原 calculate_single_auc / calculate_ks）
- `eda_analysis()` — EDA 全流程
- `feature_analysis_report()` — Excel 多 Sheet 特征分析报告

### AutoModelBuilder — 自动化建模
- `load_data()` / `train()` / `save_model()` / `load_model()`
- `get_feature_importance()` / `shap_analysis()` / `predict()`
- `hyperparameter_tuning()` — 支持 bayesian / grid 搜索

### ModelAttributionAnalyzer — 模型归因分析
- `analyze_distribution_shift()` → (df_stat, psi_detail_df, auc_summary) — PSI/IV/单特征AUC分析
- `permutation_importance()` → (df_abl, auc_base) — 特征消融
- `full_attribution()` → (df_stat, df_abl, summary) — 完整归因流程，输出 Excel 报告
- `_write_attribution_excel()` — 归因报告 Excel 格式化输出

### ModelReportGenerator — 模型报告生成器
- 依赖外部 reportbuilderv4 库
- `generate_model_summary()` / `generate_model_stability()` / `generate_correlation()`

### BeamSearchFeatureSelector — BeamSearch 特征筛选
- `load_data_csv()` / `train_xgb()` / `beam_search()`
- `feature_importances_frame()`

## 归因报告 Excel 格式

### 输出文件
`attribution_report.xlsx`，4 个 Sheet：

| Sheet | 内容 |
|-------|------|
| 汇总 | 性能概览 + AUC/KS变化 + Top-5特征 + IV/PSI分级说明 |
| 归因明细 | 按分组排列（特征标识/分布偏移/预测力变化/缺失率对比/特征消融），组间竖线分割 |
| 特征消融 | Permutation Importance 结果，按影响降序 |
| PSI明细 | 分箱级 PSI 明细 |

### 样式规范
- 白色背景，隐藏网格线
- 大标题：深蓝底 `#2F5496` + 白字
- 分区标题：浅蓝底 `#D6E4F0`
- 表头：深蓝底 + 白字
- 条件色：绿 `#C6EFCE` / 黄 `#FFEB9C` / 红 `#FFC7CE`
- 归因明细分组间深蓝粗竖线分割
- PSI/IV_drop/abl_delta_mean 条件色标注
- IV 分级：<0.02 无预测力 / 0.02~0.1 弱 / 0.1~0.3 中 / ≥0.3 强
- PSI 分级：<0.1 Stable / 0.1~0.25 Slightly Unstable / ≥0.25 Unstable

## 2026-05-12 优化记录

### API 合并精简
- `calculate_single_iv` + `calculate_batch_iv` → `calculate_iv_detail`（返回 iv_summary + iv_detail）
- `calculate_psi` + `calculate_psi_simple` → `calculate_psi_detail`（返回 psi + detail，dropna=True 替代简版）
- `calculate_single_auc` + `calculate_ks` → `calculate_auc_ks`（metrics 参数选择）
- 移除 `ModelAttributionAnalyzer` 上的委托方法（calc_psi / calc_single_iv / calc_single_auc）
- 移除 `BeamSearchFeatureSelector.predict_evals`（直接用 FeatureAnalysisToolkit.calculate_auc_ks）

### 归因分析输出优化
- `permutation_importance` 移除 output_dir 参数和文件输出逻辑（数据由 full_attribution 统一输出）
- `train()` 添加 `return self.model`（原无返回值）
- `full_attribution` 输出从 CSV 改为 Excel 多 Sheet 报告
- Excel 报告格式：白色背景、无网格线、分组竖线分割、条件色标注、IV/PSI 分级说明

### 测试修复
- test_all.py / test_comprehensive.py 全部适配新 API
- 67/67 测试全部通过

## 2026-05-12 第二轮优化

### 多模型支持
- `__init__` 新增 `model_type` 参数（'auto'/'xgboost'/'lightgbm'/'sklearn'）
- 自动检测：通过模型对象类型判断（`_detect_model_type`）
- `_predict` 统一预测入口：XGB 用 DMatrix、LGBM 用 `.predict`、sklearn 用 `.predict_proba`

### CSI 计算
- `analyze_distribution_shift` 新增 CSI（Characteristic Stability Index）列
- CSI 与 PSI 共用分箱逻辑，但 CSI 强调样本维度偏移
- `csi_label` 使用 PSI 分级标准

### PSI 重复计算消除
- `_distribution_shift_worker` 改为返回 `psi_detail`（分箱明细）
- `analyze_distribution_shift` 从 worker 结果中提取明细，不再二次调用 `calculate_psi_detail`

### 特征消融并行化
- `permutation_importance` 支持 `n_workers > 1` 时特征级并行
- 新增 `_permutation_worker` 模块级函数（多进程安全）
- 单进程模式用 `_predict` 替代硬编码 `xgb.DMatrix`，兼容多模型

### 自动归因结论
- `_generate_conclusion()` 根据阈值自动生成文字结论
- AUC Drop > 0.02 → 显著下降；> 0.005 → 轻微下降
- Score PSI >= 0.25 → 显著偏移；>= 0.1 → 轻微偏移
- 自动识别 Top-PSI/Top-IV-drop/Top-消融/缺失飙升特征
- 结论写入 summary['conclusion']，Excel 汇总页展示

### 模型分分布对比 + Bad Rate 表
- `_build_score_distribution()`: 基准期 vs 当前期模型分等频分箱分布对比
- `_build_bad_rate_table()`: 按模型分分组 Bad Rate 对比（跨期并排）
- 新增 Excel 页签「模型分分布」「BadRate对比」（6页签报告）

### 缺失率告警
- 新增 `miss_alert` 列：缺失率变化 > 5% → '↑缺失飙升'；< -5% → '↓缺失恢复'
- 归因明细汇总行增加缺失飙升计数

### Excel 报告重构
- `_write_attribution_excel` 从 420+ 行拆分为公共函数 + 各页签逻辑
- 样式常量集中为 `S` dict，提取 `_write_title`/`_write_subtitle`/`_write_df_table`
- 条件格式改用 `CellIsRule`（批量规则，性能优于逐单元格）
- 归因明细分组增加「CSI」和「缺失告警」列

### 报告页签结构（6页签）
| Sheet | 内容 |
|-------|------|
| 汇总 | 性能概览 + AUC/KS变化 + 归因结论 + Top特征 + IV/PSI分级说明 |
| 归因明细 | 分组展示（标识/CSI/预测力/缺失率+告警/消融），CellIsRule条件色 |
| 特征消融 | Permutation Importance，CellIsRule条件色 |
| PSI明细 | 分箱级 PSI 明细 |
| 模型分分布 | 基准期 vs 当前期模型分分布对比 |
| BadRate对比 | 按模型分分组 Bad Rate 对比，CellIsRule条件色 |

## 依赖
```bash
# 归因报告演示
cd D:\05.BaseBuilding\P.AutoModelToolsMemory
python run_attribution_demo.py

# 全功能测试
cd D:\05.BaseBuilding\P.AutoModelToolsMemory\test_workspace
python test_all.py
```
