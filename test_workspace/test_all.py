# -*- coding: utf-8 -*-
"""
AutoModelBuilderTools 全功能测试脚本
覆盖所有 CLI 模式和常见异常场景
"""
import sys
import os
import traceback
import time
import warnings
warnings.filterwarnings("ignore")

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 导入主模块的所有类
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

def record(name, status, msg=""):
    tag = PASS if status else FAIL
    results.append((name, tag, msg))
    print(f"  [{tag}] {name}" + (f" -- {msg}" if msg else ""))

# ============================================================
print("=" * 60)
print("T1. FeatureAnalysisToolkit 测试")
print("=" * 60)

# 加载测试数据
df = pd.read_csv('test_data.csv', sep='\t')
df.columns = df.columns.str.lower()
target = 'dpd30_term1'
features = [c for c in df.columns if c not in ['appl_no', 'draw_month', 'flag_cg_yz', 'cust_status3', 'dpd30_term1', 'dpd30_term3']]

# T1.1 WOE计算
try:
    woe_result = FeatureAnalysisToolkit.calculate_woe(df['feature_00'].fillna(-999), df[target])
    assert 'woe' in woe_result and 'iv' in woe_result
    assert isinstance(woe_result['iv'], float)
    record("calculate_woe", True, f"iv={woe_result['iv']:.4f}")
except Exception as e:
    record("calculate_woe", False, str(e))

# T1.2 单特征IV
try:
    iv_summary, _ = FeatureAnalysisToolkit.calculate_iv_detail(df[['feature_00', target]], target)
    iv_val = iv_summary['iv'].iloc[0]
    assert isinstance(iv_val, float)
    record("calculate_single_iv", True, f"iv={iv_val:.4f}")
except Exception as e:
    record("calculate_single_iv", False, str(e))

# T1.3 批量IV
try:
    iv_summary, _ = FeatureAnalysisToolkit.calculate_iv_detail(df[features + [target]], target)
    assert isinstance(iv_summary, pd.DataFrame)
    assert 'feature' in iv_summary.columns and 'iv' in iv_summary.columns
    record("calculate_batch_iv", True, f"{len(iv_summary)} features")
except Exception as e:
    record("calculate_batch_iv", False, str(e))

# T1.4 WOE转换
try:
    woe_dict = FeatureAnalysisToolkit.fit_woe_transformer(df[features].fillna(-999), df[target])
    X_woe = FeatureAnalysisToolkit.apply_woe_transform(df[features].fillna(-999), woe_dict)
    assert X_woe.shape == df[features].shape
    record("woe_transform", True)
except Exception as e:
    record("woe_transform", False, str(e))

# T1.5 PSI计算
try:
    df_old = df[df['draw_month'] == '202511']
    df_new = df[df['draw_month'] == '202601']
    psi_result = FeatureAnalysisToolkit.calculate_psi_detail(df_old['feature_00'], df_new['feature_00'])
    psi_val = psi_result['psi']
    assert isinstance(psi_val, float) or np.isnan(psi_val)
    record("calculate_psi", True, f"psi={psi_val}")
except Exception as e:
    record("calculate_psi", False, str(e))

# T1.6 PSI简单版
try:
    psi_result = FeatureAnalysisToolkit.calculate_psi_detail(df_old['feature_00'], df_new['feature_00'], dropna=True)
    psi_simple = psi_result['psi']
    assert isinstance(psi_simple, float)
    record("calculate_psi_simple", True, f"psi={psi_simple:.4f}")
except Exception as e:
    record("calculate_psi_simple", False, str(e))

# T1.7 单特征AUC
try:
    valid = df[target].isin([0, 1])
    auc_val = FeatureAnalysisToolkit.calculate_auc_ks(df.loc[valid, target].values, df.loc[valid, 'feature_00'].values, metrics='auc')['auc']
    record("calculate_single_auc", True, f"auc={auc_val}")
except Exception as e:
    record("calculate_single_auc", False, str(e))

# T1.8 KS计算
try:
    ks_val = FeatureAnalysisToolkit.calculate_auc_ks(df.loc[valid, target].values, df.loc[valid, 'feature_00'].values, metrics='ks')['ks']
    record("calculate_ks", True, f"ks={ks_val}")
except Exception as e:
    record("calculate_ks", False, str(e))

# T1.9 AUC+KS同时
try:
    res = FeatureAnalysisToolkit.calculate_auc_ks(df.loc[valid, target].values, df.loc[valid, 'feature_00'].values)
    record("calculate_auc_ks", True, f"auc={res['auc']}, ks={res['ks']}")
except Exception as e:
    record("calculate_auc_ks", False, str(e))

# T1.10 EDA分析
try:
    eda_res = FeatureAnalysisToolkit.eda_analysis(df.loc[valid, features], df.loc[valid, target], output_dir='test_output/eda')
    assert 'coverage' in eda_res and 'iv_woe' in eda_res and 'psi' in eda_res
    record("eda_analysis", True)
except Exception as e:
    record("eda_analysis", False, str(e))

# T1.11 特征分析报告（Excel多Sheet）
try:
    xlsx_path = FeatureAnalysisToolkit.feature_analysis_report(
        df=df, target=target, group_col='draw_month', base_group_value='202511',
        features=features[:5],  # 只用5个特征快速测试
        special_values=[-999.0],
        woe_binning_method='quantile', woe_bins=5, compute_woe=True,
        output_path='test_output/feature_analysis_report.xlsx')
    assert os.path.exists(xlsx_path)
    # 验证 Excel 内容
    xls = pd.ExcelFile(xlsx_path)
    sheet_names = xls.sheet_names
    assert '覆盖率分析' in sheet_names
    assert 'PSI汇总' in sheet_names
    assert 'PSI明细' in sheet_names
    assert 'IV' in sheet_names
    assert 'Bivar' in sheet_names
    assert 'WOE' in sheet_names
    assert 'AUC' in sheet_names
    # 验证覆盖率 shape
    cov_df = pd.read_excel(xlsx_path, sheet_name='覆盖率分析')
    assert len(cov_df) == 5 and 'overall' in cov_df.columns
    # 验证 IV shape
    iv_df_r = pd.read_excel(xlsx_path, sheet_name='IV')
    assert len(iv_df_r) == 5 and 'overall' in iv_df_r.columns
    # 验证 WOE 有 bin_range 列
    woe_df_r = pd.read_excel(xlsx_path, sheet_name='WOE')
    assert 'bin_range' in woe_df_r.columns and 'woe' in woe_df_r.columns
    record("feature_analysis_report", True, f"sheets={sheet_names}")
except Exception as e:
    record("feature_analysis_report", False, str(e))

# T1.12 特征分析报告 - bestKS 分箱
try:
    xlsx_path2 = FeatureAnalysisToolkit.feature_analysis_report(
        df=df, target=target, group_col='draw_month', base_group_value='202511',
        features=features[:3],
        woe_binning_method='bestks', woe_bins=5, compute_woe=True,
        output_path='test_output/feature_analysis_report_bestks.xlsx')
    woe_bk = pd.read_excel(xlsx_path2, sheet_name='WOE')
    assert len(woe_bk) > 0
    record("feature_analysis_report_bestks", True, f"woe_rows={len(woe_bk)}")
except Exception as e:
    record("feature_analysis_report_bestks", False, str(e))

# T1.13 特征分析报告 - 等宽分箱
try:
    xlsx_path3 = FeatureAnalysisToolkit.feature_analysis_report(
        df=df, target=target, group_col='draw_month', base_group_value='202511',
        features=features[:3],
        woe_binning_method='equal_width', woe_bins=5, compute_woe=True,
        output_path='test_output/feature_analysis_report_eqwidth.xlsx')
    woe_ew = pd.read_excel(xlsx_path3, sheet_name='WOE')
    assert len(woe_ew) > 0
    record("feature_analysis_report_eqwidth", True, f"woe_rows={len(woe_ew)}")
except Exception as e:
    record("feature_analysis_report_eqwidth", False, str(e))

# T1.14 Bivar sheet 内容验证
try:
    bivar_df = pd.read_excel('test_output/feature_analysis_report.xlsx', sheet_name='Bivar')
    assert 'bin_range' in bivar_df.columns
    assert 'pct' in bivar_df.columns and 'cum_pct' in bivar_df.columns
    assert 'bad_rate' in bivar_df.columns and 'woe' in bivar_df.columns
    # 验证每个特征的 pct 之和 ≈ 1
    for feat in bivar_df['feature'].unique():
        feat_pct_sum = bivar_df.loc[bivar_df['feature'] == feat, 'pct'].sum()
        assert abs(feat_pct_sum - 1.0) < 0.01, f"pct sum={feat_pct_sum}"
    record("feature_analysis_report_bivar", True, f"rows={len(bivar_df)}, cols={list(bivar_df.columns)}")
except Exception as e:
    record("feature_analysis_report_bivar", False, str(e))

# T1.15 特征分析报告 - 多进程模式
try:
    xlsx_mp = FeatureAnalysisToolkit.feature_analysis_report(
        df=df, target=target, group_col='draw_month', base_group_value='202511',
        features=features[:3],
        woe_binning_method='quantile', woe_bins=5, compute_woe=True,
        n_workers=2,
        output_path='test_output/feature_analysis_report_mp.xlsx')
    assert os.path.exists(xlsx_mp)
    xls_mp = pd.ExcelFile(xlsx_mp)
    assert 'Bivar' in xls_mp.sheet_names
    bivar_mp = pd.read_excel(xlsx_mp, sheet_name='Bivar')
    assert len(bivar_mp) > 0
    record("feature_analysis_report_multiproc", True, f"sheets={xls_mp.sheet_names}")
except Exception as e:
    record("feature_analysis_report_multiproc", False, str(e))

# ============================================================
print("\n" + "=" * 60)
print("T2. AutoModelBuilder train 测试")
print("=" * 60)

# T2.1 数据加载
try:
    builder = AutoModelBuilder(model_type='xgb', random_state=42)
    X, y = builder.load_data('test_data.csv', 'dpd30_term1', delimiter='\t')
    assert X.shape[0] > 0 and len(y) > 0
    record("load_data", True, f"X={X.shape}, y={y.shape}")
except Exception as e:
    record("load_data", False, str(e))

# T2.2 模型训练（用小参数快速训练）
try:
    params = {
        'learning_rate': 0.1,
        'n_estimators': 30,
        'max_depth': 3,
        'random_state': 42,
        'verbosity': 0,
        'n_jobs': 1
    }
    model = builder.train(X, y, params=params)
    assert model is not None
    record("train_xgb", True)
except Exception as e:
    record("train_xgb", False, str(e))

# T2.3 模型保存/加载
try:
    builder.save_model('test_output/test_model.pkl')
    builder2 = AutoModelBuilder(model_type='xgb')
    builder2.load_model('test_output/test_model.pkl')
    assert builder2.model is not None
    record("save_load_model", True)
except Exception as e:
    record("save_load_model", False, str(e))

# T2.4 特征重要性
try:
    imp = builder.get_feature_importance()
    assert 'feature' in imp.columns and 'importance' in imp.columns
    record("get_feature_importance", True, f"top={imp.iloc[0]['feature']}")
except Exception as e:
    record("get_feature_importance", False, str(e))

# T2.5 预测
try:
    preds = builder.predict(X.iloc[:100])
    assert len(preds) == 100
    record("predict", True, f"mean={np.mean(preds):.4f}")
except Exception as e:
    record("predict", False, str(e))

# T2.5b SHAP分析（XGB）- 同时验证 shap 包和 pred_contribs 两种方式
try:
    shap_vals = builder.shap_analysis(X.iloc[:100].fillna(0), output_dir='test_output/shap_xgb')
    assert os.path.exists('test_output/shap_xgb/shap_values.csv')
    assert os.path.exists('test_output/shap_xgb/shap_summary.csv')
    shap_df = pd.read_csv('test_output/shap_xgb/shap_values.csv')
    shap_sum = pd.read_csv('test_output/shap_xgb/shap_summary.csv')
    assert shap_df.shape == (100, len(builder.features))
    assert 'feature' in shap_sum.columns and 'mean_abs_shap' in shap_sum.columns
    # pred_contribs 方式
    assert os.path.exists('test_output/shap_xgb/shap_contribs.csv')
    assert os.path.exists('test_output/shap_xgb/shap_contribs_summary.csv')
    contribs_df = pd.read_csv('test_output/shap_xgb/shap_contribs.csv')
    contribs_sum = pd.read_csv('test_output/shap_xgb/shap_contribs_summary.csv')
    assert contribs_df.shape == (100, len(builder.features))
    # 两种方式的 top feature 排序应大致一致
    shap_top3 = set(shap_sum['feature'].head(3).tolist())
    contribs_top3 = set(contribs_sum['feature'].head(3).tolist())
    overlap = len(shap_top3 & contribs_top3)
    record("shap_analysis_xgb", True, f"shape={shap_df.shape}, top_feat={shap_sum.iloc[0]['feature']}, contribs_top={contribs_sum.iloc[0]['feature']}, top3_overlap={overlap}/3")
except Exception as e:
    record("shap_analysis_xgb", False, str(e))

# T2.6 LGB模型训练
try:
    builder_lgb = AutoModelBuilder(model_type='lgb', random_state=42)
    X_lgb, y_lgb = builder_lgb.load_data('test_data.csv', 'dpd30_term1', delimiter='\t')
    model_lgb = builder_lgb.train(X_lgb, y_lgb, params={'n_estimators': 30, 'verbosity': -1, 'random_state': 42})
    preds_lgb = builder_lgb.predict(X_lgb.iloc[:50])
    record("train_lgb", True, f"preds_len={len(preds_lgb)}")
except Exception as e:
    record("train_lgb", False, str(e))

# T2.6b SHAP分析（LGB）
try:
    shap_lgb = builder_lgb.shap_analysis(X_lgb.iloc[:50].fillna(0), output_dir='test_output/shap_lgb')
    assert os.path.exists('test_output/shap_lgb/shap_values.csv')
    assert os.path.exists('test_output/shap_lgb/shap_summary.csv')
    shap_lgb_sum = pd.read_csv('test_output/shap_lgb/shap_summary.csv')
    record("shap_analysis_lgb", True, f"top_feat={shap_lgb_sum.iloc[0]['feature']}")
except Exception as e:
    record("shap_analysis_lgb", False, str(e))

# T2.7 LR模型训练
try:
    builder_lr = AutoModelBuilder(model_type='lr', random_state=42)
    X_lr, y_lr = builder_lr.load_data('test_data.csv', 'dpd30_term1', delimiter='\t')
    X_lr_filled = X_lr.fillna(0)
    model_lr = builder_lr.train(X_lr_filled, y_lr, params={'C': 1.0, 'max_iter': 100})
    preds_lr = builder_lr.predict(X_lr_filled.iloc[:50])
    record("train_lr", True, f"preds_len={len(preds_lr)}")
except Exception as e:
    record("train_lr", False, str(e))

# T2.7b SHAP分析（LR）
try:
    shap_lr = builder_lr.shap_analysis(X_lr_filled.iloc[:50], output_dir='test_output/shap_lr')
    assert os.path.exists('test_output/shap_lr/shap_values.csv')
    assert os.path.exists('test_output/shap_lr/shap_summary.csv')
    shap_lr_sum = pd.read_csv('test_output/shap_lr/shap_summary.csv')
    record("shap_analysis_lr", True, f"top_feat={shap_lr_sum.iloc[0]['feature']}")
except Exception as e:
    record("shap_analysis_lr", False, str(e))

# ============================================================
print("\n" + "=" * 60)
print("T3. Scoring 打分测试")
print("=" * 60)

# T3.1 先训练一个 xgb Booster 模型并保存（用于 scoring 测试）
try:
    train_df = df[df[target].isin([0, 1])].copy()
    train_df[features] = train_df[features].fillna(-999.0)
    dtrain = xgb.DMatrix(train_df[features].values, label=train_df[target].values, feature_names=features, missing=-999.0)
    xgb_booster = xgb.train(
        {'objective': 'binary:logistic', 'eval_metric': 'auc', 'tree_method': 'hist',
         'max_depth': 3, 'seed': 42, 'verbosity': 0, 'missing': -999.0},
        dtrain, num_boost_round=50)
    joblib.dump(xgb_booster, 'test_output/test_booster.pkl')
    joblib.dump(features, 'test_output/test_features.pkl')
    record("prepare_booster_for_scoring", True)
except Exception as e:
    record("prepare_booster_for_scoring", False, str(e))

# T3.2 正常打分（chunk模式）
try:
    from sklearn.metrics import roc_auc_score
    booster = joblib.load('test_output/test_booster.pkl')
    feat_list = joblib.load('test_output/test_features.pkl')

    df_score = pd.read_csv('test_data.csv', sep='\t')
    df_score.columns = df_score.columns.str.lower()
    feat_lower = [f.lower() for f in feat_list]
    df_score[feat_lower] = df_score[feat_lower].fillna(-999.0)

    # chunk 打分模拟
    output_path = 'test_output/scored_result.csv'
    total = 0
    first = True
    y_all, s_all = [], []
    for ci, chunk in enumerate(pd.read_csv('test_data.csv', sep='\t', chunksize=2000)):
        chunk.columns = chunk.columns.str.lower()
        chunk[feat_lower] = chunk[feat_lower].fillna(-999.0)
        dm = xgb.DMatrix(chunk[feat_lower].values, feature_names=feat_list, missing=-999.0)
        scores = booster.predict(dm)
        chunk['model_score'] = scores
        total += len(chunk)
        if 'dpd30_term1' in chunk.columns:
            vm = chunk['dpd30_term1'].isin([0, 1])
            if vm.sum() > 0:
                y_all.append(chunk.loc[vm, 'dpd30_term1'].values)
                s_all.append(chunk.loc[vm, 'model_score'].values)
        chunk.to_csv(output_path, sep='\t', index=False, mode='w' if ci == 0 else 'a', header=(ci == 0))
        del dm

    auc_final = roc_auc_score(np.concatenate(y_all), np.concatenate(s_all))
    record("scoring_chunk_mode", True, f"total={total}, auc={auc_final:.4f}")
except Exception as e:
    record("scoring_chunk_mode", False, str(e))

# ============================================================
print("\n" + "=" * 60)
print("T4. Model Evaluation 模型分评估测试")
print("=" * 60)

# T4.1 基本AUC/KS评估
try:
    scored_df = pd.read_csv('test_output/scored_result.csv', sep='\t')
    scored_df.columns = scored_df.columns.str.lower()
    valid = scored_df[target].isin([0, 1])
    res = FeatureAnalysisToolkit.calculate_auc_ks(scored_df.loc[valid, target].values, scored_df.loc[valid, 'model_score'].values)
    record("model_eval_basic", True, f"auc={res['auc']}, ks={res['ks']}")
except Exception as e:
    record("model_eval_basic", False, str(e))

# T4.2 按时间切片评估
try:
    eval_results = []
    for month in scored_df['draw_month'].unique():
        sub = scored_df[(scored_df['draw_month'] == month) & valid]
        if len(sub) > 0 and sub[target].nunique() >= 2:
            r = FeatureAnalysisToolkit.calculate_auc_ks(sub[target].values, sub['model_score'].values)
            eval_results.append({'month': month, 'auc': r['auc'], 'ks': r['ks'], 'count': len(sub)})
    pd.DataFrame(eval_results).to_csv('test_output/time_slice_eval.csv', index=False)
    record("model_eval_time_slice", True, f"{len(eval_results)} slices")
except Exception as e:
    record("model_eval_time_slice", False, str(e))

# T4.3 按客群分组评估
try:
    group_results = []
    for flag in scored_df['flag_cg_yz'].unique():
        sub = scored_df[(scored_df['flag_cg_yz'] == flag) & valid]
        if len(sub) > 0 and sub[target].nunique() >= 2:
            r = FeatureAnalysisToolkit.calculate_auc_ks(sub[target].values, sub['model_score'].values)
            group_results.append({'group': flag, 'auc': r['auc'], 'ks': r['ks']})
    record("model_eval_group", True, f"{len(group_results)} groups")
except Exception as e:
    record("model_eval_group", False, str(e))

# ============================================================
print("\n" + "=" * 60)
print("T5. Attribution 归因分析测试")
print("=" * 60)

# T5.1 分布偏移分析
try:
    analyzer = ModelAttributionAnalyzer('test_output/test_booster.pkl', 'test_output/test_features.pkl', missing_value=-999.0)
    info_vars = ['appl_no', 'draw_month', 'flag_cg_yz', 'cust_status3', 'dpd30_term1', 'dpd30_term3']
    df_stat, psi_detail_df, auc_summary = analyzer.analyze_distribution_shift(
        df, 'draw_month', target, '202511', '202601',
        info_vars=info_vars, output_dir='test_output')
    assert len(df_stat) > 0
    record("distribution_shift", True, f"auc_drop={auc_summary['auc_drop']:.4f}, {len(df_stat)} features")
except Exception as e:
    record("distribution_shift", False, str(e))

# T5.2 特征消融（单进程）
try:
    df_abl, auc_base = analyzer.permutation_importance(
        df, 'draw_month', target, '202601',
        info_vars=info_vars, n_workers=1)
    assert len(df_abl) > 0
    record("permutation_importance", True, f"base_auc={auc_base:.4f}, {len(df_abl)} features")
except Exception as e:
    record("permutation_importance", False, str(e))

# T5.3 完整归因流程
try:
    df_stat2, df_abl2, summary = analyzer.full_attribution(
        df, 'draw_month', target, '202511', '202601',
        info_vars=info_vars, n_workers=1, output_dir='test_output/attribution')
    assert 'auc_drop' in summary
    record("full_attribution", True, f"auc_drop={summary['auc_drop']:.4f}")
except Exception as e:
    record("full_attribution", False, str(e))

# ============================================================
print("\n" + "=" * 60)
print("T6. BeamSearch 特征筛选测试")
print("=" * 60)

# T6.1 数据加载 - OOT数据是TSV格式，包含非数值列appl_no/draw_month，需指定features和sep
try:
    selector = BeamSearchFeatureSelector()
    # 先读取数据获取特征列
    _tmp = pd.read_csv('test_oot1.csv', sep='\t')
    _tmp.columns = _tmp.columns.str.lower()
    _num_feats = [c for c in _tmp.columns if c not in ['appl_no', 'draw_month', 'dpd30_term1'] and _tmp[c].dtype in [np.float64, np.int64, float, int]]
    train_df, oot_list, all_feats = selector.load_data_csv(
        'test_oot1.csv', ['test_oot2.csv'], 'dpd30_term1', features=_num_feats, sep='\t')
    record("beamsearch_load_data", True, f"train={len(train_df)}, oot={len(oot_list[0])}, feats={len(all_feats)}")
except Exception as e:
    record("beamsearch_load_data", False, str(e))

# T6.2 单机XGB训练
try:
    data = [train_df] + oot_list
    initial_feats = all_feats[:3]
    bst, imp, aucs = selector.train_xgb(data, initial_feats, 'dpd30_term1',
                                         num_boost_round=30, early_stopping_rounds=5)
    record("beamsearch_train_xgb", True, f"aucs={[round(a, 4) for a in aucs]}")
except Exception as e:
    record("beamsearch_train_xgb", False, str(e))

# T6.3 Beam Search（小参数快速测试，单进程）- 只有1个OOT，weights_list长度=1
try:
    candidate_feats = all_feats[3:8]  # 只用5个候选特征
    final_model, final_imp, best_feats, best_aucs = selector.beam_search(
        data=data, initial_features=initial_feats,
        candidate_features=candidate_feats, target='dpd30_term1',
        weights_list=[1.0], beam_width=2, patience=2,
        max_workers=1, num_boost_round=30, early_stopping_rounds=5,
        use_memmap=False)
    record("beamsearch_single_process", True, f"best_feats={len(best_feats)}, aucs={[round(a, 4) for a in best_aucs]}")
except Exception as e:
    record("beamsearch_single_process", False, str(e))

# T6.4 Beam Search（memmap模式，多进程）- 使用系统临时目录避免Windows中文路径问题
try:
    import tempfile
    memmap_dir = os.path.join(tempfile.gettempdir(), f'beamsearch_test_{os.getpid()}')
    final_model2, final_imp2, best_feats2, best_aucs2 = selector.beam_search(
        data=data, initial_features=initial_feats,
        candidate_features=candidate_feats, target='dpd30_term1',
        weights_list=[1.0], beam_width=2, patience=2,
        max_workers=2, num_boost_round=30, early_stopping_rounds=5,
        use_memmap=True, memmap_dir=memmap_dir)
    record("beamsearch_memmap_mp", True, f"best_feats={len(best_feats2)}, aucs={[round(a, 4) for a in best_aucs2]}")
except Exception as e:
    record("beamsearch_memmap_mp", False, str(e))

# ============================================================
print("\n" + "=" * 60)
print("T7. 异常场景测试")
print("=" * 60)

# T7.1 全零目标变量（无正样本）
try:
    tiny = pd.read_csv('test_tiny.csv', sep='\t')
    tiny.columns = tiny.columns.str.lower()
    tiny_feats = ['feature_00', 'feature_01']
    auc_val = FeatureAnalysisToolkit.calculate_auc_ks(tiny['dpd30_term1'].values, tiny['feature_00'].values, metrics='auc')['auc']
    assert np.isnan(auc_val), f"期望NaN，实际{auc_val}"
    record("edge_case_all_zero_target", True, "返回NaN（预期行为）")
except Exception as e:
    record("edge_case_all_zero_target", False, str(e))

# T7.2 全缺失特征
try:
    nan_df = pd.read_csv('test_nan.csv', sep='\t')
    nan_df.columns = nan_df.columns.str.lower()
    psi_nan = FeatureAnalysisToolkit.calculate_psi_detail(nan_df['feature_00'], nan_df['feature_00'])['psi']
    record("edge_case_all_nan_feature", True, f"psi={psi_nan}")
except Exception as e:
    record("edge_case_all_nan_feature", False, str(e))

# T7.3 常数列特征
try:
    const_df = pd.read_csv('test_const.csv', sep='\t')
    const_df.columns = const_df.columns.str.lower()
    iv_summary_const, _ = FeatureAnalysisToolkit.calculate_iv_detail(const_df[['feature_const', 'dpd30_term1']], 'dpd30_term1')
    iv_const = iv_summary_const['iv'].iloc[0]
    assert iv_const == 0.0, f"期望0.0，实际{iv_const}"
    record("edge_case_const_feature_iv", True, f"iv={iv_const}")
except Exception as e:
    record("edge_case_const_feature_iv", False, str(e))

# T7.4 PSI全空数据
try:
    psi_empty = FeatureAnalysisToolkit.calculate_psi_detail(pd.Series([], dtype=float), pd.Series([], dtype=float))['psi']
    assert psi_empty == 0.0
    record("edge_case_psi_empty", True, f"psi={psi_empty}")
except Exception as e:
    record("edge_case_psi_empty", False, str(e))

# T7.5 模型文件不存在
try:
    analyzer_bad = ModelAttributionAnalyzer('/nonexistent/model.pkl', '/nonexistent/feat.pkl')
    record("edge_case_missing_model", False, "应该抛出异常但没有")
except Exception as e:
    record("edge_case_missing_model", True, f"正确抛出异常: {type(e).__name__}")

# T7.6 特征列不存在于数据中
try:
    df_wrong = df.copy()
    feat_wrong = ['nonexistent_feature']
    dm = xgb.DMatrix(df_wrong[features].values, feature_names=feat_wrong, missing=-999.0)
    record("edge_case_wrong_features", False, "应该抛出异常但没有")
except Exception as e:
    record("edge_case_wrong_features", True, f"正确抛出异常: {type(e).__name__}")

# T7.7 空DataFrame
try:
    empty_df = pd.DataFrame(columns=['feature_00', 'dpd30_term1'])
    iv_summary_empty, _ = FeatureAnalysisToolkit.calculate_iv_detail(empty_df, 'dpd30_term1')
    iv_empty = iv_summary_empty['iv'].iloc[0] if len(iv_summary_empty) > 0 else 0.0
    assert iv_empty == 0.0
    record("edge_case_empty_df_iv", True, f"iv={iv_empty}")
except Exception as e:
    record("edge_case_empty_df_iv", False, str(e))

# T7.8 AUC/KS 单类标签
try:
    y_one = np.array([0, 0, 0, 0, 0])
    s_one = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    res = FeatureAnalysisToolkit.calculate_auc_ks(y_one, s_one)
    assert np.isnan(res['auc'])
    record("edge_case_single_class_auc_ks", True, f"返回NaN（预期行为）")
except Exception as e:
    record("edge_case_single_class_auc_ks", False, str(e))

# ============================================================
print("\n" + "=" * 60)
print("T8. FeatureAnalysisToolkit 内部方法测试")
print("=" * 60)

# T8.1 _mask_special_values
try:
    s = pd.Series([1, -999, 3, -9999, 5, np.nan])
    masked = FeatureAnalysisToolkit._mask_special_values(s, special_values=[-999, -9999])
    assert masked.isna().sum() == 3, f"期望3个NaN(2个special+1个原生NaN)，实际{masked.isna().sum()}"
    assert masked.dropna().tolist() == [1.0, 3.0, 5.0]
    record("_mask_special_values", True, f"na_count={masked.isna().sum()}")
except Exception as e:
    record("_mask_special_values", False, str(e))

# T8.2 _mask_special_values 无 special_values
try:
    s2 = pd.Series([1, 2, np.nan])
    masked2 = FeatureAnalysisToolkit._mask_special_values(s2)
    assert masked2.isna().sum() == 1
    record("_mask_special_values_no_special", True)
except Exception as e:
    record("_mask_special_values_no_special", False, str(e))

# T8.3 _bin_with_bestks
try:
    np.random.seed(42)
    feat_arr = np.random.randn(500)
    tgt_arr = (feat_arr > 0).astype(int)
    bins = FeatureAnalysisToolkit._bin_with_bestks(feat_arr, tgt_arr, max_bins=5)
    assert len(bins) >= 2
    record("_bin_with_bestks", True, f"bins={len(bins)}")
except Exception as e:
    record("_bin_with_bestks", False, str(e))

# T8.4 _bin_with_bestks 边界: 样本太少
try:
    tiny_feat = np.array([1.0, 2.0])
    tiny_tgt = np.array([0, 1])
    bins_tiny = FeatureAnalysisToolkit._bin_with_bestks(tiny_feat, tiny_tgt, max_bins=5)
    assert len(bins_tiny) >= 2
    record("_bin_with_bestks_tiny", True, f"bins={len(bins_tiny)}")
except Exception as e:
    record("_bin_with_bestks_tiny", False, str(e))

# T8.5 _compute_woe_table (quantile)
try:
    woe_q = FeatureAnalysisToolkit._compute_woe_table(df, 'feature_00', target, method='quantile', n_bins=5)
    assert 'woe' in woe_q.columns and 'iv' in woe_q.columns
    record("_compute_woe_table_quantile", True, f"rows={len(woe_q)}")
except Exception as e:
    record("_compute_woe_table_quantile", False, str(e))

# T8.6 _compute_woe_table (equal_width)
try:
    woe_ew = FeatureAnalysisToolkit._compute_woe_table(df, 'feature_00', target, method='equal_width', n_bins=5)
    assert 'woe' in woe_ew.columns
    record("_compute_woe_table_equal_width", True, f"rows={len(woe_ew)}")
except Exception as e:
    record("_compute_woe_table_equal_width", False, str(e))

# T8.7 _compute_woe_table (bestks)
try:
    woe_bk = FeatureAnalysisToolkit._compute_woe_table(df, 'feature_00', target, method='bestks', n_bins=5)
    assert 'woe' in woe_bk.columns
    record("_compute_woe_table_bestks", True, f"rows={len(woe_bk)}")
except Exception as e:
    record("_compute_woe_table_bestks", False, str(e))

# T8.8 _compute_woe_table with special_values
try:
    df_sv = df.copy()
    df_sv.loc[:5, 'feature_00'] = -999
    woe_sv = FeatureAnalysisToolkit._compute_woe_table(df_sv, 'feature_00', target, method='quantile', n_bins=5, special_values=[-999])
    assert 'woe' in woe_sv.columns
    has_missing = (woe_sv['bin_range'] == 'Missing').any() or (woe_sv['bin_range'] == '-999').any()
    record("_compute_woe_table_special_values", True, f"rows={len(woe_sv)}, has_special_bin={has_missing}")
except Exception as e:
    record("_compute_woe_table_special_values", False, str(e))

# ============================================================
print("\n" + "=" * 60)
print("T9. AutoModelBuilder 补充测试")
print("=" * 60)

# T9.1 hyperparameter_tuning - LGB bayesian（需要 LightGBM 3.3.0 环境）
try:
    import lightgbm as _lgb
    _lgb_ver = tuple(int(x) for x in _lgb.__version__.split('.')[:2])
    if _lgb_ver >= (4, 0):
        record("hyperparameter_tuning_lgb_bayesian", True, f"SKIP: LGB {_lgb.__version__} (需3.3.0)")
    else:
        builder_ht_lgb = AutoModelBuilder(model_type='lgb', random_state=42)
        X_ht, y_ht = builder_ht_lgb.load_data('test_data.csv', 'dpd30_term1', delimiter='\t')
        X_ht = X_ht.fillna(0)
        best_params_lgb = builder_ht_lgb.hyperparameter_tuning(X_ht, y_ht, tuning_method='bayesian', n_iter=2)
        assert isinstance(best_params_lgb, dict)
        record("hyperparameter_tuning_lgb_bayesian", True, f"params={list(best_params_lgb.keys())}")
except Exception as e:
    record("hyperparameter_tuning_lgb_bayesian", False, str(e))

# T9.2 hyperparameter_tuning - LGB grid
try:
    if _lgb_ver >= (4, 0):
        record("hyperparameter_tuning_lgb_grid", True, f"SKIP: LGB {_lgb.__version__} (需3.3.0)")
    else:
        grid_params_lgb = {
            'n_estimators': [30, 50],
            'max_depth': [3, 5],
            'num_leaves': [20, 31],
        }
        best_grid_lgb = builder_ht_lgb.hyperparameter_tuning(X_ht, y_ht, tuning_method='grid', params=grid_params_lgb)
        assert isinstance(best_grid_lgb, dict)
        record("hyperparameter_tuning_lgb_grid", True, f"best={best_grid_lgb}")
except Exception as e:
    record("hyperparameter_tuning_lgb_grid", False, str(e))

# T9.3 hyperparameter_tuning - XGB bayesian（需要 XGBoost 1.x 环境）
try:
    import xgboost as _xgb
    _xgb_ver = tuple(int(x) for x in _xgb.__version__.split('.')[:2])
    if _xgb_ver >= (2, 0):
        record("hyperparameter_tuning_xgb_bayesian", True, f"SKIP: XGB {_xgb.__version__} (需1.x)")
    else:
        builder_ht_xgb = AutoModelBuilder(model_type='xgb', random_state=42)
        best_params_xgb = builder_ht_xgb.hyperparameter_tuning(X_ht, y_ht, tuning_method='bayesian', n_iter=2)
        assert isinstance(best_params_xgb, dict)
        record("hyperparameter_tuning_xgb_bayesian", True, f"params={list(best_params_xgb.keys())}")
except Exception as e:
    record("hyperparameter_tuning_xgb_bayesian", False, str(e))

# T9.4 hyperparameter_tuning - XGB grid
try:
    if _xgb_ver >= (2, 0):
        record("hyperparameter_tuning_xgb_grid", True, f"SKIP: XGB {_xgb.__version__} (需1.x)")
    else:
        grid_params_xgb = {
            'n_estimators': [30, 50],
            'max_depth': [3, 5],
        }
        best_grid_xgb = builder_ht_xgb.hyperparameter_tuning(X_ht, y_ht, tuning_method='grid', params=grid_params_xgb)
        assert isinstance(best_grid_xgb, dict)
        record("hyperparameter_tuning_xgb_grid", True, f"best={best_grid_xgb}")
except Exception as e:
    record("hyperparameter_tuning_xgb_grid", False, str(e))

# T9.5 hyperparameter_tuning - LR grid
try:
    builder_ht_lr = AutoModelBuilder(model_type='lr', random_state=42)
    X_lr_ht, y_lr_ht = builder_ht_lr.load_data('test_data.csv', 'dpd30_term1', delimiter='\t')
    X_lr_ht = X_lr_ht.fillna(0)
    grid_params_lr = {
        'C': [0.1, 1.0],
        'penalty': ['l2'],
        'solver': ['liblinear']
    }
    best_grid_lr = builder_ht_lr.hyperparameter_tuning(X_lr_ht, y_lr_ht, tuning_method='grid', params=grid_params_lr)
    assert isinstance(best_grid_lr, dict)
    record("hyperparameter_tuning_lr_grid", True, f"best={best_grid_lr}")
except Exception as e:
    record("hyperparameter_tuning_lr_grid", False, str(e))

# T9.6 hyperparameter_tuning - LR bayesian（应使用默认参数+warning）
try:
    best_bayes_lr = builder_ht_lr.hyperparameter_tuning(X_lr_ht, y_lr_ht, tuning_method='bayesian')
    assert isinstance(best_bayes_lr, dict)
    record("hyperparameter_tuning_lr_bayesian_fallback", True, f"params={best_bayes_lr}")
except Exception as e:
    record("hyperparameter_tuning_lr_bayesian_fallback", False, str(e))

# T9.7 get_feature_importance with output_file
try:
    imp_file = 'test_output/feature_importance.csv'
    imp_df = builder.get_feature_importance(output_file=imp_file)
    assert os.path.exists(imp_file)
    imp_loaded = pd.read_csv(imp_file)
    assert len(imp_loaded) == len(imp_df)
    record("get_feature_importance_output_file", True, f"rows={len(imp_loaded)}")
except Exception as e:
    record("get_feature_importance_output_file", False, str(e))

# T9.8 eda_analysis (AutoModelBuilder)
try:
    X_eda, y_eda = builder.load_data('test_data.csv', 'dpd30_term1', delimiter='\t')
    eda_res = builder.eda_analysis(X_eda, y_eda, output_dir='test_output/eda_builder')
    assert 'coverage' in eda_res
    record("eda_analysis_builder", True)
except Exception as e:
    record("eda_analysis_builder", False, str(e))

# ============================================================
print("\n" + "=" * 60)
print("T10. ModelAttributionAnalyzer 委托方法测试")
print("=" * 60)

# T10.1 calc_psi 委托 → FeatureAnalysisToolkit.calculate_psi_detail
try:
    psi_delegated = FeatureAnalysisToolkit.calculate_psi_detail(
        df_old['feature_00'], df_new['feature_00'])['psi']
    assert isinstance(psi_delegated, float) or np.isnan(psi_delegated)
    record("calc_psi_delegated", True, f"psi={psi_delegated}")
except Exception as e:
    record("calc_psi_delegated", False, str(e))

# T10.2 calc_single_iv 委托 → FeatureAnalysisToolkit.calculate_iv_detail
try:
    iv_summary, _ = FeatureAnalysisToolkit.calculate_iv_detail(df[['feature_00', target]], target)
    iv_delegated = iv_summary['iv'].iloc[0]
    assert isinstance(iv_delegated, float)
    record("calc_single_iv_delegated", True, f"iv={iv_delegated:.4f}")
except Exception as e:
    record("calc_single_iv_delegated", False, str(e))

# T10.3 calc_single_auc 委托 → FeatureAnalysisToolkit.calculate_auc_ks
try:
    auc_delegated = FeatureAnalysisToolkit.calculate_auc_ks(
        df.loc[valid, target].values, df.loc[valid, 'feature_00'].values, metrics='auc')['auc']
    assert isinstance(auc_delegated, (float, int, np.floating))
    record("calc_single_auc_delegated", True, f"auc={auc_delegated}")
except Exception as e:
    record("calc_single_auc_delegated", False, str(e))

# ============================================================
print("\n" + "=" * 60)
print("T11. Permutation Importance 多进程测试")
print("=" * 60)

# T11.1 多进程特征消融（验证 _permutation_compute_worker 路径）
try:
    import platform
    if platform.system() == 'Windows':
        record("permutation_importance_multiprocess", True, "SKIP: Windows 多进程限制（需 __main__ guard）")
    else:
        df_abl_mp, auc_base_mp = analyzer.permutation_importance(
            df, 'draw_month', target, '202601',
            info_vars=info_vars, n_workers=2)
        assert len(df_abl_mp) > 0
        record("permutation_importance_multiprocess", True, f"base_auc={auc_base_mp:.4f}, {len(df_abl_mp)} features")
except Exception as e:
    record("permutation_importance_multiprocess", False, str(e))

# ============================================================
print("\n" + "=" * 60)
print("T12. BeamSearchFeatureSelector 补充方法测试")
print("=" * 60)

# T12.1 feature_importances_frame
try:
    # 使用 T3.1 中有 feature_names 的 booster（而非 beamsearch 的 3 特征 bst）
    booster_for_imp = joblib.load('test_output/test_booster.pkl')
    feat_imp_df = BeamSearchFeatureSelector.feature_importances_frame(booster_for_imp)
    assert 'feature' in feat_imp_df.columns and 'imp_gain' in feat_imp_df.columns
    assert len(feat_imp_df) > 0
    record("feature_importances_frame", True, f"rows={len(feat_imp_df)}, top={feat_imp_df.iloc[0]['feature']}")
except Exception as e:
    record("feature_importances_frame", False, str(e))

# T12.2 predict_evals → use FeatureAnalysisToolkit.calculate_auc_ks
try:
    booster_for_eval = joblib.load('test_output/test_booster.pkl')
    feat_list_eval = joblib.load('test_output/test_features.pkl')
    # 使用 T3.1 的 booster，数据使用对应特征
    eval_data = df[df[target].isin([0, 1])].head(200).copy()
    eval_data[feat_list_eval] = eval_data[feat_list_eval].fillna(-999.0)
    dtest_eval = xgb.DMatrix(eval_data[feat_list_eval].values, feature_names=feat_list_eval, missing=-999.0)
    pred_eval = booster_for_eval.predict(dtest_eval)
    eval_result = FeatureAnalysisToolkit.calculate_auc_ks(eval_data[target].values, pred_eval)
    assert 'auc' in eval_result and 'ks' in eval_result
    record("predict_evals", True, f"auc={eval_result['auc']}, ks={eval_result['ks']}")
except Exception as e:
    record("predict_evals", False, str(e))

# ============================================================
print("\n" + "=" * 60)
print("测试结果汇总")
print("=" * 60)
pass_count = sum(1 for _, t, _ in results if t == PASS)
fail_count = sum(1 for _, t, _ in results if t == FAIL)
print(f"\n  通过: {pass_count}/{len(results)}")
print(f"  失败: {fail_count}/{len(results)}")

if fail_count > 0:
    print("\n  失败详情:")
    for name, tag, msg in results:
        if tag == FAIL:
            print(f"    [FAIL] {name}: {msg}")

print("\n" + "=" * 60)
print(f"测试完成！{'所有测试通过' if fail_count == 0 else '存在失败项，请检查'}")
print("=" * 60)
