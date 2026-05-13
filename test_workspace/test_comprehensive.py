# -*- coding: utf-8 -*-
"""
AutoModelBuilderTools 全自动化测试脚本 v3
覆盖: 所有CLI模式 / 类方法API / 边界场景 / 数据一致性
"""
import sys
import os
import traceback
import time
import warnings
import shutil
import subprocess
warnings.filterwarnings("ignore")

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from AutoModelBuilderTools import (
    FeatureAnalysisToolkit, AutoModelBuilder,
    ModelAttributionAnalyzer, ModelReportGenerator, BeamSearchFeatureSelector
)
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb

PASS = "PASS"
FAIL = "FAIL"
results = []

def record(name, status, msg="", elapsed=None):
    results.append((name, status, msg, elapsed))
    elapsed_str = f" ({elapsed:.1f}s)" if elapsed is not None else ""
    print(f"  [{status}] {name}" + (f" -- {msg}" if msg else "") + elapsed_str)

# 检测 xgboost 版本以兼容 API
xgb_version = xgb.__version__
xgb_major = int(xgb_version.split('.')[0])

def xgb_predict(booster, dmatrix, ntree_limit=0):
    """兼容 xgboost 1.x/2.x/3.x 的 predict 封装"""
    if ntree_limit > 0:
        if xgb_major >= 2:
            return booster.predict(dmatrix, iteration_range=(0, ntree_limit))
        else:
            return booster.predict(dmatrix, ntree_limit=ntree_limit)
    return booster.predict(dmatrix)


# ============================================================
# 准备阶段：生成测试数据 & 训练模型
# ============================================================
print("=" * 70)
print("准备阶段: 生成测试数据 & 训练模型")
print("=" * 70)

for d in ['test_output_v2']:
    if os.path.exists(d):
        shutil.rmtree(d)
os.makedirs('test_output_v2', exist_ok=True)

np.random.seed(42)
target = 'dpd30_term1'
# 只有数值特征 + target，避免 string 列问题
feature_cols = [f'feature_{i:02d}' for i in range(15)]

# 主数据集（CSV，逗号分隔）
n = 5000
data = {'dpd30_term1': np.random.choice([0, 1], n, p=[0.95, 0.05])}
for i in range(15):
    col = f'feature_{i:02d}'
    if i < 3:
        data[col] = np.random.randn(n) * (i + 1)
    elif i < 8:
        data[col] = np.random.uniform(0, 100, n)
    else:
        data[col] = np.random.choice([0, 1, 2, 3], n).astype(float)
for col in [f'feature_{i:02d}' for i in [2, 5, 9, 13]]:
    mask = np.random.random(n) < 0.05
    arr = np.array(data[col], dtype=float)
    arr[mask] = np.nan
    data[col] = arr

df = pd.DataFrame(data)
df.to_csv('test_output_v2/test_data.csv', index=False)

# 带额外列的数据（用于 attribution / model_evaluation）
df_full = df.copy()
df_full.insert(0, 'appl_no', [f'APP{i:06d}' for i in range(n)])
df_full.insert(1, 'draw_month', np.random.choice(['202511', '202512', '202601', '202602'], n))
df_full.insert(2, 'flag_cg_yz', np.random.choice(['常规户MOB6+', '优质户MOB6+'], n))
df_full.insert(3, 'cust_status3', np.random.choice(['结清365-', '结清365+'], n))
df_full['dpd30_term3'] = np.random.choice([0, 1], n, p=[0.93, 0.07])
df_full.to_csv('test_output_v2/test_data_full.tsv', sep='\t', index=False)

# OOT 数据（用于 beamsearch，逗号分隔 CSV）
n_oot = 2000
oot_data = {'dpd30_term1': np.random.choice([0, 1], n_oot, p=[0.95, 0.05])}
for i in range(15):
    col = f'feature_{i:02d}'
    if i < 3:
        oot_data[col] = np.random.randn(n_oot) * (i + 1)
    elif i < 8:
        oot_data[col] = np.random.uniform(0, 100, n_oot)
    else:
        oot_data[col] = np.random.choice([0, 1, 2, 3], n_oot).astype(float)
oot1 = pd.DataFrame(oot_data)
oot1.to_csv('test_output_v2/test_oot1.csv', index=False)
oot2 = pd.DataFrame(oot_data)
oot2['dpd30_term1'] = np.random.choice([0, 1], n_oot, p=[0.95, 0.05])
oot2.to_csv('test_output_v2/test_oot2.csv', index=False)

print(f"  测试数据已生成: 主{n}, OOT1={n_oot}, OOT2={n_oot}")

# 训练 xgb Booster（用于 scoring / attribution）
features = feature_cols
df_clean = df[df[target].isin([0, 1])].copy()
df_clean[features] = df_clean[features].fillna(-999.0)
dtrain = xgb.DMatrix(df_clean[features].values, label=df_clean[target].values,
                      feature_names=features, missing=-999.0)
xgb_booster = xgb.train(
    {'objective': 'binary:logistic', 'eval_metric': 'auc', 'tree_method': 'hist',
     'max_depth': 4, 'seed': 42, 'verbosity': 0, 'missing': -999.0},
    dtrain, num_boost_round=80)
joblib.dump(xgb_booster, 'test_output_v2/test_booster.pkl')
joblib.dump(features, 'test_output_v2/test_features.pkl')
xgb_booster.save_model('test_output_v2/test_booster.json')

# 打分并保存带分数据
df_scored = df_clean.copy()
dm_all = xgb.DMatrix(df_scored[features].values, feature_names=features, missing=-999.0)
df_scored['model_score'] = xgb_booster.predict(dm_all)
df_scored.to_csv('test_output_v2/test_scored_data.csv', index=False)

# 评估数据（带模型分+时间+客群）
valid = df_full[target].isin([0, 1])
df_eval = df_full[valid].copy()
df_eval[features] = df_eval[features].fillna(-999.0)
dm_eval = xgb.DMatrix(df_eval[features].values, feature_names=features, missing=-999.0)
df_eval['model_score'] = xgb_booster.predict(dm_eval)
df_eval['model_score_v2'] = df_eval['model_score'] * 0.95 + np.random.randn(len(df_eval)) * 0.01
df_eval.to_csv('test_output_v2/test_eval_data.tsv', sep='\t', index=False)

print("  模型训练完成，打分数据已保存")
print("  准备阶段完成\n")


# ============================================================
print("=" * 70)
print("T1. FeatureAnalysisToolkit 单元测试")
print("=" * 70)

# T1.1 calculate_woe
try:
    t0 = time.time()
    woe_result = FeatureAnalysisToolkit.calculate_woe(df['feature_00'].fillna(-999), df[target])
    assert 'woe' in woe_result and 'iv' in woe_result
    assert isinstance(woe_result['iv'], float)
    assert len(woe_result['woe']) > 0
    record("T1.1 calculate_woe", PASS, f"iv={woe_result['iv']:.4f}, woe_keys={len(woe_result['woe'])}", time.time()-t0)
except Exception as e:
    record("T1.1 calculate_woe", FAIL, str(e))

# T1.2 calculate_woe 全0/全1目标
try:
    y_all_bad = pd.Series([1, 1, 1, 1])
    feat_vals = pd.Series([1, 2, 3, 4])
    woe_zero = FeatureAnalysisToolkit.calculate_woe(feat_vals, y_all_bad)
    assert woe_zero['iv'] == 0.0
    record("T1.2 calculate_woe_edge", PASS, "全0/全1目标返回iv=0")
except Exception as e:
    record("T1.2 calculate_woe_edge", FAIL, str(e))

# T1.3 calculate_single_iv → calculate_iv_detail
try:
    t0 = time.time()
    iv_summ, _ = FeatureAnalysisToolkit.calculate_iv_detail(df[['feature_00', target]], target)
    iv_val = iv_summ['iv'].iloc[0]
    assert isinstance(iv_val, float)
    record("T1.3 calculate_single_iv", PASS, f"iv={iv_val:.4f}", time.time()-t0)
except Exception as e:
    record("T1.3 calculate_single_iv", FAIL, str(e))

# T1.4 calculate_single_iv 常数列
try:
    const_df = pd.DataFrame({'feature_const': [42.0]*100, 'dpd30_term1': np.random.choice([0,1], 100)})
    iv_summ_c, _ = FeatureAnalysisToolkit.calculate_iv_detail(const_df, 'dpd30_term1')
    iv_const = iv_summ_c['iv'].iloc[0] if len(iv_summ_c) > 0 else 0.0
    assert iv_const == 0.0
    record("T1.4 calculate_single_iv_const", PASS, f"iv={iv_const}")
except Exception as e:
    record("T1.4 calculate_single_iv_const", FAIL, str(e))

# T1.5 calculate_single_iv 空DF
try:
    empty_df = pd.DataFrame({'f': pd.Series(dtype=float), 't': pd.Series(dtype=float)})
    iv_summ_e, _ = FeatureAnalysisToolkit.calculate_iv_detail(empty_df, 't')
    iv_empty = iv_summ_e['iv'].iloc[0] if len(iv_summ_e) > 0 else 0.0
    assert iv_empty == 0.0
    record("T1.5 calculate_single_iv_empty", PASS, f"iv={iv_empty}")
except Exception as e:
    record("T1.5 calculate_single_iv_empty", FAIL, str(e))

# T1.6 calculate_batch_iv → calculate_iv_detail
try:
    t0 = time.time()
    iv_summ_b, _ = FeatureAnalysisToolkit.calculate_iv_detail(df[features + [target]], target)
    assert isinstance(iv_summ_b, pd.DataFrame)
    assert 'feature' in iv_summ_b.columns and 'iv' in iv_summ_b.columns
    assert len(iv_summ_b) == len(features)
    record("T1.6 calculate_batch_iv", PASS, f"{len(iv_summ_b)} features, top_iv={iv_summ_b.iloc[0]['iv']:.4f}", time.time()-t0)
except Exception as e:
    record("T1.6 calculate_batch_iv", FAIL, str(e))

# T1.7 WOE转换
try:
    t0 = time.time()
    woe_dict = FeatureAnalysisToolkit.fit_woe_transformer(df[features].fillna(-999), df[target])
    X_woe = FeatureAnalysisToolkit.apply_woe_transform(df[features].fillna(-999), woe_dict)
    assert X_woe.shape == df[features].shape
    assert not X_woe.isnull().any().any()
    record("T1.7 woe_transform", PASS, f"shape={X_woe.shape}", time.time()-t0)
except Exception as e:
    record("T1.7 woe_transform", FAIL, str(e))

# T1.8 calculate_psi → calculate_psi_detail
try:
    t0 = time.time()
    df_old = df_full[df_full['draw_month'] == '202511']
    df_new = df_full[df_full['draw_month'] == '202601']
    psi_val = FeatureAnalysisToolkit.calculate_psi_detail(df_old['feature_00'], df_new['feature_00'])['psi']
    assert isinstance(psi_val, float) or np.isnan(psi_val)
    record("T1.8 calculate_psi", PASS, f"psi={psi_val}", time.time()-t0)
except Exception as e:
    record("T1.8 calculate_psi", FAIL, str(e))

# T1.9 calculate_psi 空序列
try:
    psi_empty = FeatureAnalysisToolkit.calculate_psi_detail(pd.Series([], dtype=float), pd.Series([], dtype=float))['psi']
    assert psi_empty == 0.0
    record("T1.9 calculate_psi_empty", PASS, f"psi={psi_empty}")
except Exception as e:
    record("T1.9 calculate_psi_empty", FAIL, str(e))

# T1.10 calculate_psi 全缺失
try:
    s_nan = pd.Series([np.nan, np.nan, np.nan])
    s_nan2 = pd.Series([np.nan, np.nan])
    psi_nan = FeatureAnalysisToolkit.calculate_psi_detail(s_nan, s_nan2)['psi']
    record("T1.10 calculate_psi_all_nan", PASS, f"psi={psi_nan}")
except Exception as e:
    record("T1.10 calculate_psi_all_nan", FAIL, str(e))

# T1.11 calculate_psi_simple → calculate_psi_detail(dropna=True)
try:
    t0 = time.time()
    psi_simple = FeatureAnalysisToolkit.calculate_psi_detail(df_old['feature_00'], df_new['feature_00'], dropna=True)['psi']
    assert isinstance(psi_simple, float)
    record("T1.11 calculate_psi_simple", PASS, f"psi={psi_simple:.4f}", time.time()-t0)
except Exception as e:
    record("T1.11 calculate_psi_simple", FAIL, str(e))

# T1.12 calculate_single_auc → calculate_auc_ks(metrics='auc')
try:
    t0 = time.time()
    valid_mask = df[target].isin([0, 1])
    auc_val = FeatureAnalysisToolkit.calculate_auc_ks(
        df.loc[valid_mask, target].values, df.loc[valid_mask, 'feature_00'].values, metrics='auc')['auc']
    record("T1.12 calculate_single_auc", PASS, f"auc={auc_val}", time.time()-t0)
except Exception as e:
    record("T1.12 calculate_single_auc", FAIL, str(e))

# T1.13 calculate_single_auc 全NaN特征
try:
    nan_vals = np.array([np.nan, np.nan, np.nan, np.nan, np.nan])
    y_vals = np.array([0, 1, 0, 1, 0])
    auc_nan = FeatureAnalysisToolkit.calculate_auc_ks(y_vals, nan_vals, metrics='auc')['auc']
    record("T1.13 calculate_single_auc_nan_feat", PASS, f"auc={auc_nan}")
except Exception as e:
    record("T1.13 calculate_single_auc_nan_feat", FAIL, str(e))

# T1.14 calculate_ks → calculate_auc_ks(metrics='ks')
try:
    t0 = time.time()
    ks_val = FeatureAnalysisToolkit.calculate_auc_ks(
        df.loc[valid_mask, target].values, df.loc[valid_mask, 'feature_00'].values, metrics='ks')['ks']
    assert isinstance(ks_val, float)
    record("T1.14 calculate_ks", PASS, f"ks={ks_val}", time.time()-t0)
except Exception as e:
    record("T1.14 calculate_ks", FAIL, str(e))

# T1.15 calculate_auc_ks
try:
    t0 = time.time()
    res = FeatureAnalysisToolkit.calculate_auc_ks(
        df.loc[valid_mask, target].values, df.loc[valid_mask, 'feature_00'].values)
    assert 'auc' in res and 'ks' in res
    record("T1.15 calculate_auc_ks", PASS, f"auc={res['auc']}, ks={res['ks']}", time.time()-t0)
except Exception as e:
    record("T1.15 calculate_auc_ks", FAIL, str(e))

# T1.16 calculate_auc_ks 单类
try:
    y_one = np.array([0, 0, 0, 0, 0])
    s_one = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    res = FeatureAnalysisToolkit.calculate_auc_ks(y_one, s_one)
    assert np.isnan(res['auc'])
    record("T1.16 calculate_auc_ks_single_class", PASS, "返回NaN（预期）")
except Exception as e:
    record("T1.16 calculate_auc_ks_single_class", FAIL, str(e))

# T1.17 eda_analysis
try:
    t0 = time.time()
    eda_res = FeatureAnalysisToolkit.eda_analysis(
        df.loc[valid_mask, features], df.loc[valid_mask, target],
        output_dir='test_output_v2/eda')
    assert 'coverage' in eda_res and 'iv_woe' in eda_res and 'psi' in eda_res
    assert os.path.exists('test_output_v2/eda/coverage.csv')
    assert os.path.exists('test_output_v2/eda/iv_woe.csv')
    assert os.path.exists('test_output_v2/eda/psi.csv')
    assert os.path.exists('test_output_v2/eda/features_woe.csv')
    record("T1.17 eda_analysis", PASS, "3 outputs, files verified", time.time()-t0)
except Exception as e:
    record("T1.17 eda_analysis", FAIL, str(e))


# ============================================================
print("\n" + "=" * 70)
print("T2. AutoModelBuilder API 测试")
print("=" * 70)

# T2.1 数据加载（逗号分隔CSV，只有数值列）
try:
    t0 = time.time()
    builder = AutoModelBuilder(model_type='xgb', random_state=42)
    X, y = builder.load_data('test_output_v2/test_data.csv', target)
    assert X.shape[0] > 0 and len(y) > 0
    assert builder.features is not None and builder.target is not None
    record("T2.1 load_data", PASS, f"X={X.shape}, y={y.shape}", time.time()-t0)
except Exception as e:
    record("T2.1 load_data", FAIL, str(e))

# T2.2 不支持的文件格式
try:
    builder_bad = AutoModelBuilder(model_type='xgb')
    builder_bad.load_data('test.parquet', 'target', file_format='parquet')
    record("T2.2 load_data_unsupported_format", FAIL, "应该抛出异常")
except ValueError:
    record("T2.2 load_data_unsupported_format", PASS, "正确抛出ValueError")
except Exception as e:
    record("T2.2 load_data_unsupported_format", FAIL, str(e))

# T2.3 XGB训练
try:
    t0 = time.time()
    params = {
        'learning_rate': 0.1, 'n_estimators': 30, 'max_depth': 3,
        'random_state': 42, 'verbosity': 0, 'n_jobs': 1
    }
    model = builder.train(X, y, params=params)
    assert model is not None
    record("T2.3 train_xgb", PASS, f"model={type(model).__name__}", time.time()-t0)
except Exception as e:
    record("T2.3 train_xgb", FAIL, str(e))

# T2.4 LGB训练
try:
    t0 = time.time()
    builder_lgb = AutoModelBuilder(model_type='lgb', random_state=42)
    X_lgb, y_lgb = builder_lgb.load_data('test_output_v2/test_data.csv', target)
    model_lgb = builder_lgb.train(X_lgb, y_lgb, params={'n_estimators': 30, 'verbosity': -1, 'random_state': 42})
    preds_lgb = builder_lgb.predict(X_lgb.iloc[:50])
    assert len(preds_lgb) == 50
    record("T2.4 train_lgb", PASS, f"preds_len={len(preds_lgb)}", time.time()-t0)
except Exception as e:
    record("T2.4 train_lgb", FAIL, str(e))

# T2.5 LR训练（需填充NaN）
try:
    t0 = time.time()
    builder_lr = AutoModelBuilder(model_type='lr', random_state=42)
    X_lr, y_lr = builder_lr.load_data('test_output_v2/test_data.csv', target)
    X_lr_filled = X_lr.fillna(0)
    model_lr = builder_lr.train(X_lr_filled, y_lr, params={'C': 1.0, 'max_iter': 100})
    preds_lr = builder_lr.predict(X_lr_filled.iloc[:50])
    assert len(preds_lr) == 50
    record("T2.5 train_lr", PASS, f"preds_len={len(preds_lr)}, scaler={builder_lr.scaler is not None}", time.time()-t0)
except Exception as e:
    record("T2.5 train_lr", FAIL, str(e))

# T2.6 模型保存/加载
try:
    t0 = time.time()
    builder.save_model('test_output_v2/test_model.pkl')
    builder2 = AutoModelBuilder(model_type='xgb')
    builder2.load_model('test_output_v2/test_model.pkl')
    assert builder2.model is not None
    assert builder2.model_type == 'xgb'
    preds2 = builder2.predict(X.iloc[:20])
    assert len(preds2) == 20
    record("T2.6 save_load_model", PASS, "roundtrip OK", time.time()-t0)
except Exception as e:
    record("T2.6 save_load_model", FAIL, str(e))

# T2.7 特征重要性
try:
    t0 = time.time()
    imp = builder.get_feature_importance()
    assert 'feature' in imp.columns and 'importance' in imp.columns
    assert len(imp) == len(builder.features)
    imp_file = builder.get_feature_importance(output_file='test_output_v2/feature_importance.csv')
    assert os.path.exists('test_output_v2/feature_importance.csv')
    record("T2.7 get_feature_importance", PASS, f"top={imp.iloc[0]['feature']}, imp={imp.iloc[0]['importance']}", time.time()-t0)
except Exception as e:
    record("T2.7 get_feature_importance", FAIL, str(e))

# T2.8 特征重要性 - 未训练模型
try:
    builder_none = AutoModelBuilder()
    builder_none.get_feature_importance()
    record("T2.8 get_importance_no_model", FAIL, "应该抛出异常")
except ValueError:
    record("T2.8 get_importance_no_model", PASS, "正确抛出ValueError")
except Exception as e:
    record("T2.8 get_importance_no_model", FAIL, str(e))

# T2.9 预测
try:
    t0 = time.time()
    preds = builder.predict(X.iloc[:100])
    assert len(preds) == 100
    assert all(0 <= p <= 1 for p in preds)
    record("T2.9 predict", PASS, f"mean={np.mean(preds):.4f}, range=[{np.min(preds):.4f}, {np.max(preds):.4f}]", time.time()-t0)
except Exception as e:
    record("T2.9 predict", FAIL, str(e))

# T2.10 预测 - 未训练模型
try:
    builder_none = AutoModelBuilder()
    builder_none.predict(X.iloc[:5])
    record("T2.10 predict_no_model", FAIL, "应该抛出异常")
except ValueError:
    record("T2.10 predict_no_model", PASS, "正确抛出ValueError")
except Exception as e:
    record("T2.10 predict_no_model", FAIL, str(e))

# T2.11 SHAP分析
try:
    t0 = time.time()
    shap_vals = builder.shap_analysis(X.iloc[:50].fillna(0), output_dir='test_output_v2/shap')
    assert os.path.exists('test_output_v2/shap/shap_values.csv')
    assert os.path.exists('test_output_v2/shap/shap_summary.csv')
    record("T2.11 shap_analysis", PASS, "files verified", time.time()-t0)
except Exception as e:
    record("T2.11 shap_analysis", FAIL, str(e))

# T2.12 LR特征重要性
try:
    imp_lr = builder_lr.get_feature_importance()
    assert 'feature' in imp_lr.columns and 'importance' in imp_lr.columns
    record("T2.12 lr_feature_importance", PASS, f"top={imp_lr.iloc[0]['feature']}")
except Exception as e:
    record("T2.12 lr_feature_importance", FAIL, str(e))


# ============================================================
print("\n" + "=" * 70)
print("T3. Scoring 打分模式测试")
print("=" * 70)

# T3.1 pkl模型打分
try:
    t0 = time.time()
    booster = joblib.load('test_output_v2/test_booster.pkl')
    feat_list = joblib.load('test_output_v2/test_features.pkl')

    df_score = pd.read_csv('test_output_v2/test_data.csv')
    df_score[feat_list] = df_score[feat_list].fillna(-999.0)
    dm = xgb.DMatrix(df_score[feat_list].values, feature_names=feat_list, missing=-999.0)
    scores = booster.predict(dm)
    assert len(scores) == len(df_score)
    assert all(0 <= s <= 1 for s in scores)
    record("T3.1 scoring_pkl_model", PASS, f"scores={len(scores)}, mean={np.mean(scores):.4f}", time.time()-t0)
except Exception as e:
    record("T3.1 scoring_pkl_model", FAIL, str(e))

# T3.2 json模型打分
try:
    t0 = time.time()
    booster_json = xgb.Booster()
    booster_json.load_model('test_output_v2/test_booster.json')
    dm2 = xgb.DMatrix(df_score[feat_list].values, feature_names=feat_list, missing=-999.0)
    scores_json = booster_json.predict(dm2)
    assert len(scores_json) == len(df_score)
    np.testing.assert_array_almost_equal(scores, scores_json, decimal=6)
    record("T3.2 scoring_json_model", PASS, "结果与pkl模型一致", time.time()-t0)
except Exception as e:
    record("T3.2 scoring_json_model", FAIL, str(e))

# T3.3 chunk分批打分
try:
    t0 = time.time()
    from sklearn.metrics import roc_auc_score
    output_path = 'test_output_v2/scored_chunk.csv'
    y_all, s_all = [], []
    for ci, chunk in enumerate(pd.read_csv('test_output_v2/test_data.csv', chunksize=1000)):
        chunk[feat_list] = chunk[feat_list].fillna(-999.0)
        dm_c = xgb.DMatrix(chunk[feat_list].values, feature_names=feat_list, missing=-999.0)
        sc = booster.predict(dm_c)
        chunk['model_score'] = sc
        if target in chunk.columns:
            vm = chunk[target].isin([0, 1])
            if vm.sum() > 0:
                y_all.append(chunk.loc[vm, target].values)
                s_all.append(chunk.loc[vm, 'model_score'].values)
        chunk.to_csv(output_path, index=False, mode='w' if ci == 0 else 'a', header=(ci == 0))
        del dm_c
    y_concat = np.concatenate(y_all)
    s_concat = np.concatenate(s_all)
    auc_final = roc_auc_score(y_concat, s_concat)
    record("T3.3 scoring_chunk_mode", PASS, f"auc={auc_final:.4f}, samples={len(y_concat)}", time.time()-t0)
except Exception as e:
    record("T3.3 scoring_chunk_mode", FAIL, str(e))

# T3.4 ntree_limit限制打分（兼容版本）
try:
    t0 = time.time()
    dm_full = xgb.DMatrix(df_score[feat_list].values, feature_names=feat_list, missing=-999.0)
    scores_full = booster.predict(dm_full)
    scores_half = xgb_predict(booster, dm_full, ntree_limit=40)
    assert not np.allclose(scores_full, scores_half)
    record("T3.4 scoring_ntree_limit", PASS, f"full_mean={np.mean(scores_full):.4f}, half_mean={np.mean(scores_half):.4f}", time.time()-t0)
except Exception as e:
    record("T3.4 scoring_ntree_limit", FAIL, str(e))


# ============================================================
print("\n" + "=" * 70)
print("T4. Model Evaluation 模型分评估测试")
print("=" * 70)

eval_df = pd.read_csv('test_output_v2/test_eval_data.tsv', sep='\t')

# T4.1 基本AUC/KS评估
try:
    t0 = time.time()
    res = FeatureAnalysisToolkit.calculate_auc_ks(eval_df[target].values, eval_df['model_score'].values)
    assert res['auc'] > 0.5
    assert res['ks'] > 0
    record("T4.1 model_eval_basic", PASS, f"auc={res['auc']}, ks={res['ks']}", time.time()-t0)
except Exception as e:
    record("T4.1 model_eval_basic", FAIL, str(e))

# T4.2 多模型分对比
try:
    t0 = time.time()
    res1 = FeatureAnalysisToolkit.calculate_auc_ks(eval_df[target].values, eval_df['model_score'].values)
    res2 = FeatureAnalysisToolkit.calculate_auc_ks(eval_df[target].values, eval_df['model_score_v2'].values)
    record("T4.2 multi_score_compare", PASS, f"score1_auc={res1['auc']}, score2_auc={res2['auc']}", time.time()-t0)
except Exception as e:
    record("T4.2 multi_score_compare", FAIL, str(e))

# T4.3 按时间切片评估
try:
    t0 = time.time()
    eval_results = []
    for month in sorted(eval_df['draw_month'].unique()):
        sub = eval_df[eval_df['draw_month'] == month]
        if len(sub) > 0 and sub[target].nunique() >= 2:
            r = FeatureAnalysisToolkit.calculate_auc_ks(sub[target].values, sub['model_score'].values)
            eval_results.append({'month': month, 'auc': r['auc'], 'ks': r['ks'], 'count': len(sub)})
    pd.DataFrame(eval_results).to_csv('test_output_v2/time_slice_eval.csv', index=False)
    record("T4.3 time_slice_eval", PASS, f"{len(eval_results)} slices", time.time()-t0)
except Exception as e:
    record("T4.3 time_slice_eval", FAIL, str(e))

# T4.4 按客群分组评估
try:
    t0 = time.time()
    group_results = []
    for flag in eval_df['flag_cg_yz'].unique():
        sub = eval_df[eval_df['flag_cg_yz'] == flag]
        if len(sub) > 0 and sub[target].nunique() >= 2:
            r = FeatureAnalysisToolkit.calculate_auc_ks(sub[target].values, sub['model_score'].values)
            group_results.append({'group': flag, 'auc': r['auc'], 'ks': r['ks'], 'count': len(sub)})
    record("T4.4 group_eval", PASS, f"{len(group_results)} groups", time.time()-t0)
except Exception as e:
    record("T4.4 group_eval", FAIL, str(e))

# T4.5 Lift计算
try:
    t0 = time.time()
    y_rate = eval_df[target].mean()
    eval_sorted = eval_df.sort_values('model_score')
    eval_sorted['group'] = pd.qcut(eval_sorted['model_score'], q=10, duplicates='drop')
    lift_df = eval_sorted.groupby('group').agg(
        sample_count=(target, 'count'), y_count=(target, 'sum')).reset_index()
    lift_df['y_rate'] = lift_df['y_count'] / lift_df['sample_count']
    lift_df['lift'] = lift_df['y_rate'] / y_rate if y_rate > 0 else 0
    record("T4.5 lift_calculation", PASS, f"lift_range=[{lift_df['lift'].min():.2f}, {lift_df['lift'].max():.2f}]", time.time()-t0)
except Exception as e:
    record("T4.5 lift_calculation", FAIL, str(e))

# T4.6 交叉分组
try:
    t0 = time.time()
    cross_results = []
    for month in eval_df['draw_month'].unique():
        for flag in eval_df['flag_cg_yz'].unique():
            sub = eval_df[(eval_df['draw_month'] == month) & (eval_df['flag_cg_yz'] == flag)]
            if len(sub) > 10 and sub[target].nunique() >= 2:
                r = FeatureAnalysisToolkit.calculate_auc_ks(sub[target].values, sub['model_score'].values)
                cross_results.append({'month': month, 'group': flag, 'auc': r['auc'], 'ks': r['ks'], 'n': len(sub)})
    record("T4.6 cross_group_eval", PASS, f"{len(cross_results)} segments", time.time()-t0)
except Exception as e:
    record("T4.6 cross_group_eval", FAIL, str(e))


# ============================================================
print("\n" + "=" * 70)
print("T5. Attribution 归因分析测试")
print("=" * 70)

info_vars = ['appl_no', 'draw_month', 'flag_cg_yz', 'cust_status3', 'dpd30_term1', 'dpd30_term3']

# T5.1 初始化
try:
    t0 = time.time()
    analyzer = ModelAttributionAnalyzer(
        'test_output_v2/test_booster.pkl', 'test_output_v2/test_features.pkl', missing_value=-999.0)
    assert len(analyzer.features) == len(features)
    record("T5.1 attribution_init", PASS, f"features={len(analyzer.features)}", time.time()-t0)
except Exception as e:
    record("T5.1 attribution_init", FAIL, str(e))

# T5.2 委托方法 → FeatureAnalysisToolkit
try:
    psi_val = FeatureAnalysisToolkit.calculate_psi_detail(df_old['feature_00'], df_new['feature_00'])['psi']
    iv_summ, _ = FeatureAnalysisToolkit.calculate_iv_detail(df[['feature_00', target]], target)
    iv_val = iv_summ['iv'].iloc[0]
    auc_val = FeatureAnalysisToolkit.calculate_auc_ks(df.loc[valid_mask, target].values, df.loc[valid_mask, 'feature_00'].values, metrics='auc')['auc']
    record("T5.2 attribution_delegates", PASS, f"psi={psi_val}, iv={iv_val:.4f}, auc={auc_val}")
except Exception as e:
    record("T5.2 attribution_delegates", FAIL, str(e))

# T5.3 分布偏移分析
try:
    t0 = time.time()
    df_stat, psi_detail_df, auc_summary = analyzer.analyze_distribution_shift(
        df_full, 'draw_month', target, '202511', '202601',
        info_vars=info_vars, output_dir='test_output_v2')
    assert len(df_stat) > 0
    assert 'psi' in df_stat.columns and 'iv_old' in df_stat.columns
    assert 'auc_old' in auc_summary and 'auc_new' in auc_summary
    assert os.path.exists('test_output_v2/distribution_shift.csv')
    record("T5.3 distribution_shift", PASS, f"auc_drop={auc_summary['auc_drop']:.4f}, {len(df_stat)} features", time.time()-t0)
except Exception as e:
    record("T5.3 distribution_shift", FAIL, str(e))

# T5.4 特征消融（单进程）
try:
    t0 = time.time()
    df_abl, auc_base = analyzer.permutation_importance(
        df_full, 'draw_month', target, '202601',
        info_vars=info_vars, n_workers=1)
    assert len(df_abl) > 0
    assert 'abl_auc_mean' in df_abl.columns and 'abl_delta_mean' in df_abl.columns
    record("T5.4 permutation_importance", PASS, f"base_auc={auc_base:.4f}, {len(df_abl)} features", time.time()-t0)
except Exception as e:
    record("T5.4 permutation_importance", FAIL, str(e))

# T5.5 完整归因流程
try:
    t0 = time.time()
    df_stat2, df_abl2, summary = analyzer.full_attribution(
        df_full, 'draw_month', target, '202511', '202601',
        info_vars=info_vars, n_workers=1, output_dir='test_output_v2/attribution')
    assert 'auc_drop' in summary
    assert 'top_psi_features' in summary
    assert 'top_iv_drop_features' in summary
    assert 'top_abl_features' in summary
    assert os.path.exists('test_output_v2/attribution/attribution_merged.csv')
    assert os.path.exists('test_output_v2/attribution/attribution_summary.csv')
    record("T5.5 full_attribution", PASS, f"auc_drop={summary['auc_drop']:.4f}", time.time()-t0)
except Exception as e:
    record("T5.5 full_attribution", FAIL, str(e))

# T5.6 特征文件格式错误
try:
    joblib.dump("not_a_list", 'test_output_v2/bad_features.pkl')
    analyzer_bad = ModelAttributionAnalyzer(
        'test_output_v2/test_booster.pkl', 'test_output_v2/bad_features.pkl')
    record("T5.6 bad_features_format", FAIL, "应该抛出异常")
except ValueError as e:
    record("T5.6 bad_features_format", PASS, f"正确抛出ValueError")
except Exception as e:
    record("T5.6 bad_features_format", FAIL, str(e))


# ============================================================
print("\n" + "=" * 70)
print("T6. BeamSearch 特征筛选测试")
print("=" * 70)

# T6.1 数据加载
try:
    t0 = time.time()
    selector = BeamSearchFeatureSelector()
    train_df, oot_list, all_feats = selector.load_data_csv(
        'test_output_v2/test_oot1.csv', ['test_output_v2/test_oot2.csv'], 'dpd30_term1')
    assert len(train_df) > 0
    assert len(oot_list) == 1
    assert len(all_feats) > 0
    record("T6.1 beamsearch_load_data", PASS, f"train={len(train_df)}, oot={len(oot_list[0])}, feats={len(all_feats)}", time.time()-t0)
except Exception as e:
    record("T6.1 beamsearch_load_data", FAIL, str(e))

# T6.2 predict_evals → FeatureAnalysisToolkit.calculate_auc_ks
try:
    t0 = time.time()
    y_pred = np.random.rand(100)
    y_true = np.random.choice([0, 1], 100)
    eval_res = FeatureAnalysisToolkit.calculate_auc_ks(y_true, y_pred)
    assert 'auc' in eval_res and 'ks' in eval_res
    record("T6.2 predict_evals", PASS, f"auc={eval_res['auc']}, ks={eval_res['ks']}", time.time()-t0)
except Exception as e:
    record("T6.2 predict_evals", FAIL, str(e))

# T6.3 feature_importances_frame
try:
    t0 = time.time()
    data_bs = [train_df] + oot_list
    initial_feats = all_feats[:3]
    bst, imp, aucs = selector.train_xgb(data_bs, initial_feats, 'dpd30_term1',
                                         num_boost_round=30, early_stopping_rounds=5)
    assert 'feature' in imp.columns and 'imp_gain' in imp.columns
    record("T6.3 feature_importances_frame", PASS, f"imp_df shape={imp.shape}", time.time()-t0)
except Exception as e:
    record("T6.3 feature_importances_frame", FAIL, str(e))

# T6.4 Beam Search 单进程
try:
    t0 = time.time()
    candidate_feats = all_feats[3:8]
    final_model, final_imp, best_feats, best_aucs = selector.beam_search(
        data=data_bs, initial_features=initial_feats,
        candidate_features=candidate_feats, target='dpd30_term1',
        weights_list=[1.0], beam_width=2, patience=2,
        max_workers=1, num_boost_round=30, early_stopping_rounds=5,
        use_memmap=False)
    assert len(best_feats) >= len(initial_feats)
    record("T6.4 beamsearch_single_process", PASS, f"best_feats={len(best_feats)}, aucs={[round(a, 4) for a in best_aucs]}", time.time()-t0)
except Exception as e:
    record("T6.4 beamsearch_single_process", FAIL, str(e))

# T6.5 Beam Search 多进程 memmap
try:
    t0 = time.time()
    final_model2, final_imp2, best_feats2, best_aucs2 = selector.beam_search(
        data=data_bs, initial_features=initial_feats,
        candidate_features=candidate_feats, target='dpd30_term1',
        weights_list=[1.0], beam_width=2, patience=2,
        max_workers=2, num_boost_round=30, early_stopping_rounds=5,
        use_memmap=True, memmap_dir='test_output_v2/beamsearch_memmap')
    record("T6.5 beamsearch_memmap_mp", PASS, f"best_feats={len(best_feats2)}, aucs={[round(a, 4) for a in best_aucs2]}", time.time()-t0)
except Exception as e:
    record("T6.5 beamsearch_memmap_mp", FAIL, str(e))


# ============================================================
print("\n" + "=" * 70)
print("T7. ModelReportGenerator 测试（不依赖外部库）")
print("=" * 70)

# T7.1 load_data
try:
    t0 = time.time()
    df_loaded, df_targets = ModelReportGenerator.load_data(
        file_path='test_output_v2/test_data_full.tsv', separator='\t',
        targets=['dpd30_term1', 'dpd30_term3'])
    assert len(df_loaded) > 0
    assert 'dpd30_term1' in df_targets
    assert 'dpd30_term3' in df_targets
    record("T7.1 report_load_data", PASS, f"df={len(df_loaded)}, targets={len(df_targets)}", time.time()-t0)
except Exception as e:
    record("T7.1 report_load_data", FAIL, str(e))

# T7.2 load_data with column_mapping
try:
    col_map = {'appl_no': 'application_id'}
    df_mapped, _ = ModelReportGenerator.load_data(
        file_path='test_output_v2/test_data_full.tsv', separator='\t',
        column_mapping=col_map, targets=['dpd30_term1'])
    assert 'application_id' in df_mapped.columns
    record("T7.2 report_load_data_mapping", PASS, "列名映射成功")
except Exception as e:
    record("T7.2 report_load_data_mapping", FAIL, str(e))

# T7.3 create_mask
try:
    month_col = 'draw_month'
    assert month_col in df_loaded.columns, f"Missing column {month_col}"
    # draw_month values are int64, not strings
    first_month = df_loaded[month_col].iloc[0]
    mask = ModelReportGenerator.create_mask(df_loaded, {month_col: first_month})
    assert mask.sum() > 0, f"mask sum is 0, unique values: {df_loaded[month_col].unique()}"
    mask2 = ModelReportGenerator.create_mask(df_loaded, {month_col: [first_month, df_loaded[month_col].unique()[1]]})
    assert mask2.sum() >= mask.sum()
    record("T7.3 create_mask", PASS, f"single={mask.sum()}, multi={mask2.sum()}")
except Exception as e:
    record("T7.3 create_mask", FAIL, str(e))

# T7.4 create_groups（嵌套结构）
try:
    groups_config = {
        'df_summary_all': {
            'all': {'conditions': {}},
        }
    }
    groups = ModelReportGenerator.create_groups(df_loaded, groups_config, is_nested=True)
    assert 'df_summary_all' in groups
    assert 'all' in groups['df_summary_all']
    record("T7.4 create_groups_nested", PASS, f"keys={list(groups.keys())}")
except Exception as e:
    record("T7.4 create_groups_nested", FAIL, str(e))

# T7.5 create_groups（平铺结构）
try:
    flat_config = {
        'all': {'conditions': {}},
        'month_202511': {'conditions': {'draw_month': '202511'}},
    }
    flat_groups = ModelReportGenerator.create_groups(df_loaded, flat_config, is_nested=False)
    assert 'all' in flat_groups and 'month_202511' in flat_groups
    record("T7.5 create_groups_flat", PASS, f"groups={list(flat_groups.keys())}")
except Exception as e:
    record("T7.5 create_groups_flat", FAIL, str(e))

# T7.6 create_groups 缺少conditions
try:
    bad_config = {'g1': {'no_conditions_key': {}}}
    ModelReportGenerator.create_groups(df_loaded, bad_config, is_nested=True)
    record("T7.6 create_groups_no_conditions", FAIL, "应该抛出异常")
except KeyError:
    record("T7.6 create_groups_no_conditions", PASS, "正确抛出KeyError")
except Exception as e:
    record("T7.6 create_groups_no_conditions", FAIL, str(e))

# T7.7 load_data with score_list
try:
    df_sl, _ = ModelReportGenerator.load_data(
        file_path='test_output_v2/test_scored_data.csv', separator=',',
        score_list=['model_score'], fill_na_value=-999.0,
        targets=['dpd30_term1'])
    assert 'model_score' in df_sl.columns
    record("T7.7 report_load_data_score", PASS, f"score_min={df_sl['model_score'].min():.4f}")
except Exception as e:
    record("T7.7 report_load_data_score", FAIL, str(e))


# ============================================================
print("\n" + "=" * 70)
print("T8. CLI 命令行模式测试")
print("=" * 70)

PYTHON = sys.executable
SCRIPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'AutoModelBuilderTools.py')

# T8.1 CLI help
try:
    t0 = time.time()
    result = subprocess.run([PYTHON, SCRIPT, '--help'], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0
    assert 'train' in result.stdout and 'scoring' in result.stdout
    record("T8.1 cli_help", PASS, "exit_code=0", time.time()-t0)
except Exception as e:
    record("T8.1 cli_help", FAIL, str(e))

# T8.2 CLI feature_analysis
try:
    t0 = time.time()
    cmd = [
        PYTHON, SCRIPT, 'feature_analysis',
        '--data_path', os.path.abspath('test_output_v2/test_data.csv'),
        '--data_sep', ',',
        '--target_col', target,
        '--compute_iv', '--compute_auc', '--compute_ks', '--compute_coverage',
        '--output_dir', 'test_output_v2/cli_feature_analysis'
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, f"stderr: {result.stderr[:500]}"
    assert os.path.exists('test_output_v2/cli_feature_analysis/feature_analysis_summary.csv')
    record("T8.2 cli_feature_analysis", PASS, "exit=0, summary file exists", time.time()-t0)
except Exception as e:
    record("T8.2 cli_feature_analysis", FAIL, str(e)[:300])

# T8.3 CLI scoring
try:
    t0 = time.time()
    scored_output = os.path.abspath('test_output_v2/cli_scored.csv')
    cmd = [
        PYTHON, SCRIPT, 'scoring',
        '--model_path', os.path.abspath('test_output_v2/test_booster.pkl'),
        '--features_path', os.path.abspath('test_output_v2/test_features.pkl'),
        '--data_path', os.path.abspath('test_output_v2/test_data.csv'),
        '--data_sep', ',',
        '--target_col', target,
        '--missing_value', '-999.0',
        '--score_name', 'model_score',
        '--chunk_size', '2000',
        '--output_path', scored_output
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, f"stderr: {result.stderr[:500]}"
    assert os.path.exists(scored_output)
    df_scored_cli = pd.read_csv(scored_output, sep='\t')
    assert 'model_score' in df_scored_cli.columns
    record("T8.3 cli_scoring", PASS, f"exit=0, rows={len(df_scored_cli)}", time.time()-t0)
except Exception as e:
    record("T8.3 cli_scoring", FAIL, str(e)[:300])

# T8.4 CLI model_evaluation
try:
    t0 = time.time()
    cmd = [
        PYTHON, SCRIPT, 'model_evaluation',
        '--data_path', os.path.abspath('test_output_v2/test_eval_data.tsv'),
        '--data_sep', '\t',
        '--target_col', target,
        '--score_cols', 'model_score,model_score_v2',
        '--score_names', 'XGB模型,V2模型',
        '--time_col', 'draw_month',
        '--group_cols', 'flag_cg_yz',
        '--metrics', 'auc,ks',
        '--include_lift',
        '--output_dir', 'test_output_v2/cli_model_eval'
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, f"stderr: {result.stderr[:500]}"
    assert os.path.exists('test_output_v2/cli_model_eval/model_evaluation_result.csv')
    eval_res_df = pd.read_csv('test_output_v2/cli_model_eval/model_evaluation_result.csv')
    assert len(eval_res_df) > 0
    record("T8.4 cli_model_evaluation", PASS, f"exit=0, rows={len(eval_res_df)}", time.time()-t0)
except Exception as e:
    record("T8.4 cli_model_evaluation", FAIL, str(e)[:300])

# T8.5 CLI train (XGB)
try:
    t0 = time.time()
    cmd = [
        PYTHON, SCRIPT, 'train',
        '--train_file', os.path.abspath('test_output_v2/test_data.csv'),
        '--target_col', target,
        '--model_type', 'xgb',
        '--file_format', 'csv',
        '--tuning_method', 'grid',
        '--n_iter', '5',
        '--output_dir', 'test_output_v2/cli_train'
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode == 0:
        assert os.path.exists('test_output_v2/cli_train/model.pkl')
        record("T8.5 cli_train_xgb", PASS, "exit=0, model saved", time.time()-t0)
    else:
        record("T8.5 cli_train_xgb", FAIL, f"exit={result.returncode}, stderr: {result.stderr[:300]}", time.time()-t0)
except Exception as e:
    record("T8.5 cli_train_xgb", FAIL, str(e)[:300])

# T8.6 CLI attribution
try:
    t0 = time.time()
    cmd = [
        PYTHON, SCRIPT, 'attribution',
        '--model_path', os.path.abspath('test_output_v2/test_booster.pkl'),
        '--features_path', os.path.abspath('test_output_v2/test_features.pkl'),
        '--data_path', os.path.abspath('test_output_v2/test_data_full.tsv'),
        '--data_sep', '\t',
        '--target_col', target,
        '--time_col', 'draw_month',
        '--baseline_month', '202511',
        '--current_month', '202601',
        '--info_vars', 'appl_no,draw_month,flag_cg_yz,cust_status3,dpd30_term1,dpd30_term3',
        '--n_workers', '1',
        '--output_dir', 'test_output_v2/cli_attribution'
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, f"stderr: {result.stderr[:500]}"
    assert os.path.exists('test_output_v2/cli_attribution/attribution_summary.csv')
    record("T8.6 cli_attribution", PASS, "exit=0", time.time()-t0)
except Exception as e:
    record("T8.6 cli_attribution", FAIL, str(e)[:300])

# T8.7 CLI scoring (json model)
try:
    t0 = time.time()
    scored_json_output = os.path.abspath('test_output_v2/cli_scored_json.csv')
    cmd = [
        PYTHON, SCRIPT, 'scoring',
        '--model_path', os.path.abspath('test_output_v2/test_booster.json'),
        '--features_path', os.path.abspath('test_output_v2/test_features.pkl'),
        '--data_path', os.path.abspath('test_output_v2/test_data.csv'),
        '--data_sep', ',',
        '--target_col', target,
        '--missing_value', '-999.0',
        '--score_name', 'json_score',
        '--chunk_size', '0',
        '--output_path', scored_json_output
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, f"stderr: {result.stderr[:500]}"
    assert os.path.exists(scored_json_output)
    record("T8.7 cli_scoring_json_model", PASS, "exit=0", time.time()-t0)
except Exception as e:
    record("T8.7 cli_scoring_json_model", FAIL, str(e)[:300])

# T8.8 CLI feature_analysis with PSI
try:
    t0 = time.time()
    cmd = [
        PYTHON, SCRIPT, 'feature_analysis',
        '--data_path', os.path.abspath('test_output_v2/test_data_full.tsv'),
        '--data_sep', '\t',
        '--target_col', target,
        '--exclude_vars', 'appl_no,draw_month,flag_cg_yz,cust_status3,dpd30_term3',
        '--psi_expected_col', 'draw_month',
        '--psi_expected_val', '202511',
        '--psi_actual_val', '202601,202602',
        '--compute_iv',
        '--output_dir', 'test_output_v2/cli_feature_analysis_psi'
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, f"stderr: {result.stderr[:500]}"
    assert os.path.exists('test_output_v2/cli_feature_analysis_psi/psi.csv')
    record("T8.8 cli_feature_analysis_psi", PASS, "exit=0, psi.csv exists", time.time()-t0)
except Exception as e:
    record("T8.8 cli_feature_analysis_psi", FAIL, str(e)[:300])


# ============================================================
print("\n" + "=" * 70)
print("T9. 数据一致性 & 回归测试")
print("=" * 70)

# T9.1 模型保存加载一致性
try:
    t0 = time.time()
    preds_before = builder.predict(X.iloc[:100])
    builder.save_model('test_output_v2/consistency_model.pkl')
    builder_reload = AutoModelBuilder()
    builder_reload.load_model('test_output_v2/consistency_model.pkl')
    preds_after = builder_reload.predict(X.iloc[:100])
    np.testing.assert_array_almost_equal(preds_before, preds_after, decimal=10)
    record("T9.1 save_load_consistency", PASS, "预测结果完全一致", time.time()-t0)
except Exception as e:
    record("T9.1 save_load_consistency", FAIL, str(e))

# T9.2 AUC/KS结果可复现
try:
    t0 = time.time()
    y_test = df.loc[valid_mask, target].values[:500]
    s_test = df.loc[valid_mask, 'feature_00'].values[:500]
    res1 = FeatureAnalysisToolkit.calculate_auc_ks(y_test, s_test)
    res2 = FeatureAnalysisToolkit.calculate_auc_ks(y_test, s_test)
    assert res1['auc'] == res2['auc'] and res1['ks'] == res2['ks']
    record("T9.2 auc_ks_reproducibility", PASS, f"auc={res1['auc']}, ks={res1['ks']}", time.time()-t0)
except Exception as e:
    record("T9.2 auc_ks_reproducibility", FAIL, str(e))

# T9.3 WOE转换无NaN
try:
    woe_dict = FeatureAnalysisToolkit.fit_woe_transformer(
        df[features].fillna(-999).iloc[:500], df[target].iloc[:500])
    X_woe = FeatureAnalysisToolkit.apply_woe_transform(
        df[features].fillna(-999).iloc[:500], woe_dict)
    nan_count = X_woe.isnull().sum().sum()
    assert nan_count == 0
    record("T9.3 woe_no_nan", PASS, "0 NaN after transform")
except Exception as e:
    record("T9.3 woe_no_nan", FAIL, str(e))

# T9.4 PSI对称性
try:
    s1 = df_old['feature_00'].dropna().iloc[:100]
    s2 = df_new['feature_00'].dropna().iloc[:100]
    psi_forward = FeatureAnalysisToolkit.calculate_psi_detail(s1, s2)['psi']
    psi_backward = FeatureAnalysisToolkit.calculate_psi_detail(s2, s1)['psi']
    assert abs(psi_forward - psi_backward) < 0.5
    record("T9.4 psi_symmetry", PASS, f"forward={psi_forward}, backward={psi_backward}")
except Exception as e:
    record("T9.4 psi_symmetry", FAIL, str(e))

# T9.5 LR模型predict输出在[0,1]
try:
    preds_lr = builder_lr.predict(X_lr_filled.iloc[:200])
    assert all(0 <= p <= 1 for p in preds_lr)
    record("T9.5 lr_predict_range", PASS, f"range=[{min(preds_lr):.4f}, {max(preds_lr):.4f}]")
except Exception as e:
    record("T9.5 lr_predict_range", FAIL, str(e))


# ============================================================
print("\n" + "=" * 70)
print("测试结果汇总")
print("=" * 70)

pass_count = sum(1 for _, t, _, _ in results if t == PASS)
fail_count = sum(1 for _, t, _, _ in results if t == FAIL)
total_time = sum(e for _, _, _, e in results if e is not None)

print(f"\n  通过: {pass_count}/{len(results)}")
print(f"  失败: {fail_count}/{len(results)}")
print(f"  总耗时: {total_time:.1f}s")

if fail_count > 0:
    print("\n  失败详情:")
    for name, tag, msg, _ in results:
        if tag == FAIL:
            print(f"    [FAIL] {name}: {msg[:200]}")

summary_df = pd.DataFrame(results, columns=['test_name', 'status', 'message', 'elapsed_sec'])
summary_df.to_csv('test_output_v2/test_summary.csv', index=False)

print(f"\n{'=' * 70}")
print(f"全自动化测试完成！{'所有测试通过' if fail_count == 0 else f'存在 {fail_count} 个失败项，请检查'}")
print(f"汇总报告: test_output_v2/test_summary.csv")
print(f"{'=' * 70}")
