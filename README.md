# AutoModelBuilderTools

自动化建模工具，支持特征EDA分析、模型训练、超参数调优、模型评估和部署。

## 功能特点

1. **特征EDA分析**：
   - 计算特征覆盖率
   - 计算IV和WOE值
   - 进行WOE转换
   - 计算PSI值

2. **自动化建模**：
   - 支持LightGBM (lgb)
   - 支持XGBoost (xgb)
   - 支持Logistic Regression (lr)
   - 支持读取CSV和LibSVM格式文件

3. **超参数调优**：
   - 贝叶斯优化 (bayesian)
   - 网格搜索 (grid)

4. **模型管理**：
   - 模型保存
   - 模型加载

5. **模型评估**：
   - 特征重要性分析
   - SHAP分析 (使用pred_contrib=True)
   - 模型预测

## 安装依赖

```bash
pip install pandas numpy lightgbm xgboost scikit-learn bayesian-optimization shap toad
```

## 使用方法

### 命令行使用

```bash
python AutoModelBuilderTools.py --train_file <训练文件路径> --target_col <目标列名> [其他参数]
```

### 参数说明

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| --train_file | str | 是 | - | 训练文件路径 |
| --test_file | str | 否 | None | 测试文件路径 |
| --target_col | str | 是 | - | 目标列名 |
| --model_type | str | 否 | lgb | 模型类型，可选值：lgb, xgb, lr |
| --file_format | str | 否 | csv | 文件格式，可选值：csv, libsvm |
| --tuning_method | str | 否 | bayesian | 调优方法，可选值：bayesian, grid |
| --n_iter | int | 否 | 50 | 贝叶斯优化迭代次数 |
| --output_dir | str | 否 | model_output | 输出目录 |
| --model_path | str | 否 | model.pkl | 模型保存路径 |

### 示例

```bash
# 使用LightGBM模型，贝叶斯调优
python AutoModelBuilderTools.py --train_file data/train.csv --test_file data/test.csv --target_col target --model_type lgb --tuning_method bayesian

# 使用XGBoost模型，网格搜索
python AutoModelBuilderTools.py --train_file data/train.libsvm --target_col target --model_type xgb --file_format libsvm --tuning_method grid

# 使用Logistic Regression模型
python AutoModelBuilderTools.py --train_file data/train.csv --target_col target --model_type lr
```

### 输出结果

运行完成后，会在指定的输出目录中生成以下文件：

- `eda/`：EDA分析结果
  - `coverage.csv`：特征覆盖率
  - `iv_woe.csv`：IV和WOE值
  - `features_woe.csv`：WOE转换后的特征
  - `psi.csv`：PSI值

- `shap/`：SHAP分析结果
  - `shap_values.csv`：SHAP值
  - `shap_summary.csv`：SHAP值汇总

- `feature_importance.csv`：特征重要性
- `model.pkl`：训练好的模型
- `test_predictions.csv`：测试集预测结果（如果提供了测试文件）

## 代码结构

- `AutoModelBuilder`类：核心建模工具
  - `load_data()`：加载数据
  - `eda_analysis()`：特征EDA分析
  - `hyperparameter_tuning()`：超参数调优
  - `train()`：训练模型
  - `save_model()`：保存模型
  - `load_model()`：加载模型
  - `get_feature_importance()`：获取特征重要性
  - `shap_analysis()`：SHAP分析
  - `predict()`：模型预测

- `main()`函数：命令行入口

## 注意事项

1. 确保数据中没有缺失值，或者在使用前进行适当的缺失值处理
2. 对于大规模数据，贝叶斯优化可能会比较耗时，可以适当调整`n_iter`参数
3. SHAP分析对于大规模数据可能会消耗较多内存，请根据实际情况使用
4. 对于Logistic Regression模型，会自动进行特征标准化