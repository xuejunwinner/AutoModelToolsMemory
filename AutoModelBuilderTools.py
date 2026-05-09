# -*- coding:utf8 -*-
"""
AutoModelBuilderTools - 单机版模型自动化工具包
支持 LightGBM / XGBoost / LR 自动化建模、模型归因分析、报告生成、BeamSearch特征筛选

类结构说明:
  FeatureAnalysisToolkit         - 特征分析工具箱（集中管理IV/KS/PSI/WOE/AUC/EDA）
    .calculate_woe() / .calculate_single_iv() / .calculate_batch_iv()              - WOE/IV计算
    .fit_woe_transformer() / .apply_woe_transform()                                - WOE转换
    .calculate_psi() / .calculate_psi_simple()                                     - PSI计算（含空值/简版）
    .calculate_single_auc() / .calculate_ks() / .calculate_auc_ks()                - AUC/KS计算
    .eda_analysis()                                                                 - EDA全流程

  AutoModelBuilder               - 自动化建模（LGB/XGB/LR）
    .load_data() / .eda_analysis() / .hyperparameter_tuning() / .train()
    .save_model() / .load_model() / .get_feature_importance() / .shap_analysis() / .predict()

  ModelAttributionAnalyzer       - 模型性能异动归因分析
    .calc_psi() / .calc_single_iv() / .calc_single_auc()                          - 委托FeatureAnalysisToolkit
    .analyze_distribution_shift()                                                  - PSI/IV/单特征AUC分析
    .permutation_importance()                                                      - 特征消融（支持多进程）
    .full_attribution()                                                            - 完整归因流程

  ModelReportGenerator           - 模型报告生成器（依赖reportbuilderv4）
    .load_data() / .create_mask() / .create_groups()                               - 数据加载与客群构建
    .generate_model_summary()                                                      - Model Summary报告
    .generate_model_stability()                                                    - Model Stability报告
    .generate_correlation()                                                        - 模型分相关性报告
    .save()                                                                        - 保存报告

  BeamSearchFeatureSelector      - 单机版BeamSearch特征筛选
    .feature_importances_frame()                                                    - XGBoost特征重要性
    .load_data_csv() / .load_data_spark_hive()                                     - 数据加载
    .train_xgb()                                                                   - XGBoost训练
    .beam_search()                                                                 - Beam Search特征筛选

CLI运行模式:
  python AutoModelBuilderTools.py train              -- 自动化建模
  python AutoModelBuilderTools.py attribution        -- 模型性能异动归因
  python AutoModelBuilderTools.py report             -- 模型报告生成
  python AutoModelBuilderTools.py beamsearch         -- BeamSearch特征筛选
  python AutoModelBuilderTools.py feature_analysis   -- 特征分析（IV/PSI/KS/AUC/EDA）
  python AutoModelBuilderTools.py scoring            -- 模型批量打分
  python AutoModelBuilderTools.py model_evaluation   -- 模型分评估（AUC/KS/Lift）
"""

# 标准库
import argparse
import gc
import json
import logging
import multiprocessing
import operator
import os
import re
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import reduce

# 第三方库
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from bayes_opt import BayesianOptimization
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ==============================
# 特征分析工具箱（集中管理 IV/KS/PSI/WOE/EDA/AUC 等指标）
# ==============================
class FeatureAnalysisToolkit:
    """
    特征分析工具箱
    集中管理所有特征级指标计算：WOE/IV、PSI、KS、单特征AUC、覆盖率、EDA
    以及通用工具方法：目录创建、CSV加载、参数解析等
    所有核心方法均为 @staticmethod，可直接类名调用，无需实例化
    """

    # ---- 通用工具方法 ----
    @staticmethod
    def ensure_dir(path):
        """确保目录存在，不存在则创建"""
        os.makedirs(path, exist_ok=True)

    @staticmethod
    def parse_csv_list(value, lower=False):
        """解析逗号分隔的字符串参数为列表"""
        if not value:
            return []
        items = [v.strip() for v in value.split(',') if v.strip()]
        return [v.lower() for v in items] if lower else items

    @staticmethod
    def load_json_config(path):
        """加载 JSON 配置文件，含异常处理"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"配置文件不存在: {path}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"配置文件JSON格式错误: {e}")
            return None

    @staticmethod
    def load_csv(file_path, sep=',', encoding='utf-8', lower_cols=True):
        """通用 CSV 加载，统一列名小写"""
        df = pd.read_csv(file_path, sep=sep, encoding=encoding)
        if lower_cols:
            df.columns = df.columns.str.lower()
        return df

    @staticmethod
    def coerce_time_value(dtype, *values):
        """将时间列参数自动转换为 DataFrame 列的 dtype 类型"""
        if not hasattr(dtype, 'type'):
            return values if len(values) > 1 else values[0]
        result = []
        for v in values:
            try:
                result.append(dtype.type(v) if not isinstance(v, dtype.type) else v)
            except (ValueError, TypeError):
                result.append(v)
        return tuple(result) if len(result) > 1 else result[0]

    # ---- WOE/IV ----
    @staticmethod
    def calculate_woe(feature, target):
        """
        计算单个特征的WOE字典
        :return: {'woe': {val: woe_val, ...}, 'iv': float}
        """
        df = pd.DataFrame({'feature': feature, 'target': target})
        grouped = df.groupby('feature').agg({'target': ['count', 'sum']})
        grouped.columns = ['total', 'bad']
        grouped['good'] = grouped['total'] - grouped['bad']

        total_bad = grouped['bad'].sum()
        total_good = grouped['good'].sum()
        if total_bad == 0 or total_good == 0:
            return {'woe': {k: 0.0 for k in grouped.index}, 'iv': 0.0}

        grouped['bad_rate'] = grouped['bad'] / total_bad
        grouped['good_rate'] = grouped['good'] / total_good
        grouped['woe'] = np.log(grouped['good_rate'] / grouped['bad_rate'])
        grouped['iv'] = (grouped['good_rate'] - grouped['bad_rate']) * grouped['woe']

        grouped['woe'] = grouped['woe'].replace([np.inf, -np.inf], 0).fillna(0)
        grouped['iv'] = grouped['iv'].replace([np.inf, -np.inf], 0).fillna(0)

        return {
            'woe': grouped['woe'].to_dict(),
            'iv': float(grouped['iv'].sum())
        }

    @staticmethod
    def calculate_single_iv(df, feat, target):
        """
        计算单特征IV（自动分箱，空值单独处理）
        :param df: DataFrame
        :param feat: 特征列名
        :param target: 目标列名
        :return: IV值(float)
        """
        try:
            d = df[[feat, target]].copy().dropna()
            if len(d) == 0:
                return 0.0
            n_unique = d[feat].nunique()
            if n_unique <= 1:
                return 0.0
            elif n_unique < 10:
                d['bin'] = d[feat]
            else:
                d['bin'] = pd.qcut(d[feat], q=10, duplicates='drop')

            g = d.groupby('bin', observed=False)[target].agg(count='count', bad='sum')
            g['good'] = g['count'] - g['bad']
            g = g.astype(float)
            g['good'] = g['good'].replace(0, 0.5)
            g['bad'] = g['bad'].replace(0, 0.5)
            total_good, total_bad = g['good'].sum(), g['bad'].sum()
            if total_good == 0 or total_bad == 0:
                return 0.0

            g['woe'] = np.log((g['good'] / total_good) / (g['bad'] / total_bad))
            g['iv'] = ((g['good'] / total_good) - (g['bad'] / total_bad)) * g['woe']
            iv_sum = g['iv'].sum()
            return 0.0 if not np.isfinite(iv_sum) else round(iv_sum, 4)
        except Exception as e:
            logger.warning(f"IV计算异常 feature={feat}: {e}")
            return 0.0

    @staticmethod
    def calculate_batch_iv(data, target_col):
        """
        批量计算所有特征的IV值
        :return: DataFrame[feature, iv]，按IV降序排列
        """
        features = [col for col in data.columns if col != target_col]
        iv_results = [{'feature': f, 'iv': FeatureAnalysisToolkit.calculate_single_iv(data, f, target_col)}
                      for f in features]
        return pd.DataFrame(iv_results).sort_values('iv', ascending=False)

    # ---- WOE 转换 ----
    @staticmethod
    def fit_woe_transformer(X, y):
        """
        拟合WOE转换器，返回woe_dict
        :return: {col: {'woe': {val: woe}, 'iv': float}}
        """
        woe_dict = {}
        for col in X.columns:
            woe_dict[col] = FeatureAnalysisToolkit.calculate_woe(X[col], y)
        return woe_dict

    @staticmethod
    def apply_woe_transform(X, woe_dict):
        """应用WOE转换"""
        X_woe = X.copy()
        cols_to_transform = [c for c in X.columns if c in woe_dict]
        for col in cols_to_transform:
            X_woe[col] = X[col].map(woe_dict[col]['woe']).fillna(0)
        return X_woe

    # ---- PSI ----
    @staticmethod
    def calculate_psi(expected, actual, bins=10):
        """
        计算PSI（Population Stability Index），空值单独一组
        :param expected: 期望分布（基准期）
        :param actual: 实际分布（对比期）
        :param bins: 分箱数
        :return: PSI值(float)
        """
        try:
            if len(expected) == 0 or len(actual) == 0:
                return 0.0

            exp_null = expected.isnull().sum()
            act_null = actual.isnull().sum()
            exp_total, act_total = len(expected), len(actual)
            exp_non_null = expected.dropna()
            act_non_null = actual.dropna()

            if len(exp_non_null) == 0:
                p_exp, p_act = 1.0, max(act_null / act_total, 1e-10) if act_total > 0 else 1.0
                return round((p_act - p_exp) * np.log(p_act / p_exp), 4)
            if len(act_non_null) == 0:
                p_exp = max(exp_null / exp_total, 1e-10)
                return round((1.0 - p_exp) * np.log(1.0 / p_exp), 4)

            if exp_non_null.nunique() <= 1:
                mode_val = exp_non_null.iloc[0]
                p_act = max((act_non_null == mode_val).sum() / len(act_non_null), 1e-10)
                psi_non_null = (p_act - 1.0) * np.log(p_act)
                p_exp_null = max(exp_null / exp_total, 1e-10)
                p_act_null = max(act_null / act_total, 1e-10)
                psi_null = (p_act_null - p_exp_null) * np.log(p_act_null / p_exp_null)
                return round(psi_non_null + psi_null, 4)

            breaks = np.quantile(exp_non_null, np.linspace(0, 1, bins + 1))
            breaks = np.unique(breaks)
            if len(breaks) < 2:
                return 0.0

            e_pct = pd.cut(exp_non_null, bins=breaks, include_lowest=True).value_counts(normalize=True).sort_index()
            a_pct = pd.cut(act_non_null, bins=breaks, include_lowest=True).value_counts(normalize=True).sort_index()
            a_pct = a_pct.reindex(e_pct.index, fill_value=0.0)
            e_pct = e_pct.reindex(a_pct.index, fill_value=0.0)

            psi_val = 0.0
            e_arr = e_pct.values.copy()
            a_arr = a_pct.values.copy()
            nonzero = ~((e_arr == 0) & (a_arr == 0))
            e_arr = np.maximum(e_arr, 1e-10)
            a_arr = np.maximum(a_arr, 1e-10)
            psi_val = float(np.sum((a_arr[nonzero] - e_arr[nonzero]) * np.log(a_arr[nonzero] / e_arr[nonzero])))

            psi_null = 0.0
            if exp_total > 0 and act_total > 0:
                p_exp = max(exp_null / exp_total, 1e-10)
                p_act = max(act_null / act_total, 1e-10)
                if p_exp > 0 or p_act > 0:
                    psi_null = (p_act - p_exp) * np.log(p_act / p_exp)

            total_psi = psi_val + psi_null
            return 0.0 if not np.isfinite(total_psi) else round(total_psi, 4)
        except Exception as e:
            logger.warning(f"PSI计算异常: {e}")
            return np.nan

    @staticmethod
    def calculate_psi_simple(expected, actual, bins=10):
        """
        简版PSI计算（不含空值单独分组，用于EDA场景）
        """
        expected = expected.dropna()
        actual = actual.dropna()
        if len(expected) == 0 or len(actual) == 0:
            return 0.0
        try:
            expected_bins = pd.cut(expected, bins=bins, retbins=True, duplicates='drop')
            actual_bins = pd.cut(actual, bins=expected_bins[1], duplicates='drop')

            expected_freq = expected_bins[0].value_counts(normalize=True).sort_index()
            actual_freq = actual_bins.value_counts(normalize=True).sort_index()

            psi_df = pd.DataFrame({'expected': expected_freq, 'actual': actual_freq}).fillna(0)
            psi_df['psi'] = (psi_df['actual'] - psi_df['expected']) * np.log(psi_df['actual'] / psi_df['expected'])
            psi_df['psi'] = psi_df['psi'].replace([np.inf, -np.inf], 0).fillna(0)
            return round(psi_df['psi'].sum(), 4)
        except Exception as e:
            logger.warning(f"PSI_simple计算异常: {e}")
            return 0.0

    # ---- 单特征 AUC/KS ----
    @staticmethod
    def calculate_single_auc(y, feat_vals):
        """计算单特征AUC"""
        try:
            mask = ~np.isnan(feat_vals)
            return round(roc_auc_score(y[mask], feat_vals[mask]), 4)
        except Exception as e:
            logger.warning(f"AUC计算异常: {e}")
            return np.nan

    @staticmethod
    def calculate_ks(y_true, y_pred, pos_label=1):
        """计算KS值"""
        if len(y_true) == 0:
            return np.nan
        n_classes = y_true.nunique() if hasattr(y_true, 'nunique') else len(set(y_true))
        if n_classes < 2:
            return np.nan
        fpr, tpr, _ = roc_curve(y_true=y_true, y_score=y_pred, pos_label=pos_label)
        return round(float(np.max(tpr - fpr)), 4)

    @staticmethod
    def calculate_auc_ks(y_true, y_pred, pos_label=1):
        """同时计算AUC和KS"""
        try:
            auc = round(roc_auc_score(y_true=y_true, y_score=y_pred), 4)
            fpr, tpr, _ = roc_curve(y_true=y_true, y_score=y_pred, pos_label=pos_label)
            ks = round(float(np.max(tpr - fpr)), 4)
            return {'auc': auc, 'ks': ks}
        except Exception as e:
            logger.warning(f"AUC/KS计算异常: {e}")
            return {'auc': np.nan, 'ks': np.nan}

    # ---- EDA ----
    @staticmethod
    def eda_analysis(X, y, output_dir='eda_results', random_state=42):
        """
        特征EDA分析（覆盖率、IV、WOE转换、PSI）
        :param X: 特征DataFrame
        :param y: 目标Series
        :param output_dir: 输出目录
        :param random_state: 随机种子
        :return: {'coverage': df, 'iv_woe': df, 'psi': df}
        """
        logger.info("Performing EDA analysis")
        FeatureAnalysisToolkit.ensure_dir(output_dir)

        features = list(X.columns)
        target_col = '__target__'

        # 覆盖率（无需复制整个 DataFrame）
        coverage = (1 - X.isnull().sum() / len(X)).to_dict()
        coverage_df = pd.DataFrame.from_dict(coverage, orient='index', columns=['coverage'])
        coverage_df.to_csv(os.path.join(output_dir, 'coverage.csv'))
        logger.info("Coverage analysis completed")

        # IV
        data_for_iv = X.copy()
        data_for_iv[target_col] = y
        iv_df = FeatureAnalysisToolkit.calculate_batch_iv(data_for_iv, target_col)
        iv_df.to_csv(os.path.join(output_dir, 'iv_woe.csv'), index=False)
        logger.info("IV analysis completed")

        # WOE转换
        woe_dict = FeatureAnalysisToolkit.fit_woe_transformer(X, y)
        X_woe = FeatureAnalysisToolkit.apply_woe_transform(X, woe_dict)
        X_woe.to_csv(os.path.join(output_dir, 'features_woe.csv'))
        logger.info("WOE transformation completed")

        # PSI
        X_train, X_test, _, _ = train_test_split(X, y, test_size=0.3, random_state=random_state)
        psi_result = {col: FeatureAnalysisToolkit.calculate_psi_simple(X_train[col], X_test[col]) for col in features}
        psi_df = pd.DataFrame.from_dict(psi_result, orient='index', columns=['psi'])
        psi_df.to_csv(os.path.join(output_dir, 'psi.csv'))
        logger.info("PSI analysis completed")

        return {'coverage': coverage_df, 'iv_woe': iv_df, 'psi': psi_df}

    # ---- 特征分析报告 ----
    @staticmethod
    def _mask_special_values(series, special_values=None):
        """将用户指定的特殊值替换为 NaN"""
        s = series.copy()
        if hasattr(s, 'cat'):
            s = s.astype(s.dtype.categories.dtype if hasattr(s.dtype, 'categories') else float)
        if special_values:
            replace_map = {sv: np.nan for sv in special_values}
            s = s.replace(replace_map)
        return s

    @staticmethod
    def _bin_with_bestks(feature, target, max_bins=10):
        """
        BestKS 分箱：使用 DecisionTree 迭代寻找最优分裂点，保证 bad_rate 单调
        :return: list of split points（含左右边界）
        """
        d = pd.DataFrame({'feat': feature, 'target': target}).dropna()
        if len(d) < max_bins * 2 or d['target'].nunique() < 2:
            _, bins = pd.qcut(d['feat'], q=min(max_bins, len(d) // 2), retbins=True, duplicates='drop')
            return bins.tolist()

        feat_min = d['feat'].min() - 1e-6
        feat_max = d['feat'].max() + 1e-6
        current_bins = [(feat_min, feat_max)]

        # 预提取 numpy 数组加速
        feat_arr = d['feat'].values
        tgt_arr = d['target'].values

        for _ in range(max_bins - 1):
            best_score = -1
            best_threshold = None
            best_idx = None

            for idx, (low, high) in enumerate(current_bins):
                mask = (feat_arr > low) & (feat_arr <= high)
                sub_feat = feat_arr[mask]
                sub_tgt = tgt_arr[mask]
                if len(sub_feat) < 10 or len(np.unique(sub_tgt)) < 2:
                    continue
                dt = DecisionTreeClassifier(max_depth=1, min_samples_leaf=max(5, int(len(sub_feat) * 0.05)),
                                            criterion='gini')
                dt.fit(sub_feat.reshape(-1, 1), sub_tgt)
                if dt.tree_.node_count < 3:
                    continue
                threshold = dt.tree_.threshold[0]
                if threshold <= low or threshold >= high:
                    continue
                left_n = (sub_feat <= threshold).sum()
                right_n = len(sub_feat) - left_n
                if left_n < 5 or right_n < 5:
                    continue
                n = len(sub_feat)
                gini_parent = dt.tree_.impurity[0]
                gini_left = dt.tree_.impurity[1]
                gini_right = dt.tree_.impurity[2]
                score = n * gini_parent - left_n * gini_left - right_n * gini_right
                if score > best_score:
                    best_score = score
                    best_threshold = threshold
                    best_idx = idx

            if best_threshold is None:
                break
            low, high = current_bins[best_idx]
            current_bins[best_idx] = (low, best_threshold)
            current_bins.insert(best_idx + 1, (best_threshold, high))

        # 合并非单调相邻 bin（使用 numpy 向量化计算 bad_rate）
        def _bad_rates(bins):
            rates = []
            for low, high in bins:
                m = (feat_arr > low) & (feat_arr <= high)
                rates.append(tgt_arr[m].mean() if m.sum() > 0 else 0)
            return rates

        for _ in range(len(current_bins) - 1):
            rates = _bad_rates(current_bins)
            is_mono = (all(rates[i] <= rates[i + 1] + 1e-8 for i in range(len(rates) - 1)) or
                       all(rates[i + 1] <= rates[i] + 1e-8 for i in range(len(rates) - 1)))
            if is_mono or len(current_bins) <= 2:
                break
            min_diff = float('inf')
            merge_idx = 0
            for i in range(len(rates) - 1):
                diff = abs(rates[i + 1] - rates[i])
                if diff < min_diff:
                    min_diff = diff
                    merge_idx = i
            low, high = current_bins[merge_idx]
            _, high2 = current_bins[merge_idx + 1]
            current_bins[merge_idx] = (low, high2)
            del current_bins[merge_idx + 1]

        cut_points = sorted(set([b[0] for b in current_bins] + [b[1] for b in current_bins]))
        return cut_points

    @staticmethod
    def _bin_stats_one_group(series, target_arr, breakpoints):
        """
        按指定 breakpoints 对单个分组分箱统计
        :param series: 特征列（已 mask 特殊值的 Series）
        :param target_arr: 目标列（ndarray）
        :param breakpoints: 分箱边界
        :return: (bins_list, total_good, total_bad)
        """
        null_mask = series.isna().values
        vals = series.values[~null_mask]
        tgt = np.asarray(target_arr)[~null_mask].astype(float)

        bins = []
        if len(vals) > 0 and breakpoints is not None and len(breakpoints) >= 2:
            binned = pd.cut(vals, bins=breakpoints, include_lowest=True)
            df_t = pd.DataFrame({'bin': binned, 'target': tgt})
            grouped = df_t.groupby('bin', observed=False)['target'].agg(total='count', bad='sum')
            for idx, row in grouped.iterrows():
                bins.append({
                    'bin_range': str(idx),
                    'total': int(row['total']),
                    'good': int(row['total'] - row['bad']),
                    'bad': int(row['bad'])
                })

        # Missing 组
        null_count = int(null_mask.sum())
        if null_count > 0:
            null_tgt = np.asarray(target_arr)[null_mask].astype(float)
            null_bad = int(null_tgt.sum())
            bins.append({
                'bin_range': 'Missing',
                'total': null_count,
                'good': null_count - null_bad,
                'bad': null_bad
            })

        total_good = sum(b['good'] for b in bins)
        total_bad = sum(b['bad'] for b in bins)
        return bins, total_good, total_bad

    @staticmethod
    def _compute_woe_table(df, feat, target, method='quantile', n_bins=10, special_values=None):
        """
        计算 WOE 明细表（独立调用接口，保留兼容）
        :return: DataFrame[feature, bin_range, total_count, good_count, bad_count, bad_rate, woe, iv]
        """
        d = df[[feat, target]].copy()
        d[feat] = FeatureAnalysisToolkit._mask_special_values(d[feat], special_values)
        null_mask = d[feat].isna()
        d_non_null = d[~null_mask].copy()
        d_null = d[null_mask].copy()
        rows = []

        if len(d_non_null) == 0:
            return pd.DataFrame()

        # 分箱
        feat_vals = d_non_null[feat]
        if method == 'quantile':
            if feat_vals.nunique() <= n_bins:
                d_non_null['bin'] = feat_vals.astype(str)
            else:
                d_non_null['bin'] = pd.qcut(feat_vals, q=n_bins, duplicates='drop')
        elif method == 'equal_width':
            d_non_null['bin'] = pd.cut(feat_vals, bins=n_bins, duplicates='drop')
        elif method == 'bestks':
            cut_points = FeatureAnalysisToolkit._bin_with_bestks(feat_vals, d_non_null[target], max_bins=n_bins)
            if len(cut_points) >= 2:
                d_non_null['bin'] = pd.cut(feat_vals, bins=cut_points, include_lowest=True)
            else:
                d_non_null['bin'] = feat_vals.astype(str)

        grouped = d_non_null.groupby('bin', observed=False)[target].agg(
            total_count='count', bad_count='sum')
        grouped['good_count'] = grouped['total_count'] - grouped['bad_count']
        total_good = grouped['good_count'].sum()
        total_bad = grouped['bad_count'].sum()
        if total_good == 0 or total_bad == 0:
            return pd.DataFrame()

        for idx, row in grouped.iterrows():
            gc = max(row['good_count'], 0.5)
            bc = max(row['bad_count'], 0.5)
            woe = np.log((gc / total_good) / (bc / total_bad))
            iv = ((gc / total_good) - (bc / total_bad)) * woe
            rows.append({
                'feature': feat, 'bin_range': str(idx),
                'total_count': int(row['total_count']),
                'good_count': int(row['good_count']),
                'bad_count': int(row['bad_count']),
                'bad_rate': round(row['bad_count'] / row['total_count'], 4),
                'woe': round(woe, 4), 'iv': round(iv, 4)
            })

        # Missing 组
        if len(d_null) > 0:
            n_bad = d_null[target].sum()
            n_good = len(d_null) - n_bad
            gc = max(n_good, 0.5)
            bc = max(n_bad, 0.5)
            woe = np.log((gc / total_good) / (bc / total_bad))
            iv = ((gc / total_good) - (bc / total_bad)) * woe
            rows.append({
                'feature': feat, 'bin_range': 'Missing',
                'total_count': len(d_null),
                'good_count': int(n_good), 'bad_count': int(n_bad),
                'bad_rate': round(n_bad / len(d_null), 4),
                'woe': round(woe, 4), 'iv': round(iv, 4)
            })

        # 汇总行
        total_iv = sum(r['iv'] for r in rows)
        total_cnt = sum(r['total_count'] for r in rows)
        total_bad_cnt = sum(r['bad_count'] for r in rows)
        rows.append({
            'feature': feat, 'bin_range': 'TOTAL',
            'total_count': total_cnt, 'good_count': total_cnt - total_bad_cnt,
            'bad_count': total_bad_cnt, 'bad_rate': round(total_bad_cnt / total_cnt, 4),
            'woe': '-', 'iv': round(total_iv, 4)
        })
        return pd.DataFrame(rows)

    @staticmethod
    def feature_analysis_report(
        df, target, group_col, base_group_value,
        features=None, exclude_vars=None, special_values=None,
        woe_binning_method='quantile', woe_bins=10, compute_woe=False,
        n_workers=1, output_path='feature_analysis_report.xlsx'
    ):
        """
        一键特征分析报告，输出到 Excel（多 Sheet）
        PSI/IV/WOE/Bivar 共享基准期分箱，支持多进程并行。
        :param df: 数据 DataFrame
        :param target: 目标列名
        :param group_col: 分组列名（如 draw_month）
        :param base_group_value: PSI 基准期取值，分箱边界以此期数据为准
        :param features: 特征列表（None 则自动检测数值列）
        :param exclude_vars: 排除列名列表
        :param special_values: 特殊值列表（视同缺失，如 [-999, -9999]）
        :param woe_binning_method: WOE 分箱方式（独立调用时生效，报告模式统一用基准期等频分箱）
        :param woe_bins: 分箱数
        :param compute_woe: 是否计算 WOE 明细
        :param n_workers: 并行进程数（1=单进程）
        :param output_path: 输出 Excel 路径
        :return: output_path
        """
        logger.info("========== 特征分析报告 ==========")
        try:
            import openpyxl
        except ImportError:
            raise ImportError("需要 openpyxl 库来输出 Excel，请运行: pip install openpyxl")

        # 特征列解析
        if features is None:
            exclude = set(exclude_vars or [])
            exclude.update([target, group_col])
            features = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]
        logger.info(f"特征数: {len(features)}, 分组列: {group_col}, 基准期: {base_group_value}, 进程数: {n_workers}")

        # base_group_value 类型对齐
        col_dtype = df[group_col].dtype
        base_group_value = FeatureAnalysisToolkit.coerce_time_value(col_dtype, base_group_value)

        group_values = sorted(df[group_col].unique().tolist())
        compare_values = [gv for gv in group_values if str(gv) != str(base_group_value)]
        n_bins = woe_bins

        # 预处理：一次性 mask 特殊值 + 按分组预切片
        logger.info("预处理数据...")
        y_all = df[target].values
        group_indices = {gv: df[group_col] == gv for gv in group_values}
        y_groups = {gv: df.loc[group_indices[gv], target].values for gv in group_values}

        # 构建 worker 参数列表
        worker_args = []
        for feat in features:
            masked = FeatureAnalysisToolkit._mask_special_values(df[feat], special_values)
            groups_data = {gv: masked[group_indices[gv]] for gv in group_values}
            worker_args.append((
                feat, masked, y_all, group_values, compare_values, base_group_value,
                y_groups, groups_data, n_bins, compute_woe
            ))

        # 并行 / 串行执行
        use_mp = n_workers > 1 and len(features) > 1
        if use_mp:
            logger.info(f"多进程计算（{n_workers} workers）...")
            with multiprocessing.Pool(processes=n_workers) as pool:
                results = list(tqdm(pool.imap_unordered(_feature_analysis_one_feat, worker_args),
                                   total=len(worker_args), desc="特征分析"))
        else:
            logger.info("计算覆盖率/PSI/IV/Bivar/WOE/AUC...")
            results = [_feature_analysis_one_feat(a) for a in tqdm(worker_args, desc="特征分析")]

        # 按特征原始顺序排列
        feat_order = {f: i for i, f in enumerate(features)}
        results.sort(key=lambda r: feat_order.get(r['coverage']['feature'], 0))

        # 汇总
        coverage_df = pd.DataFrame([r['coverage'] for r in results])
        psi_summary_df = pd.DataFrame([r['psi_summary'] for r in results])
        psi_detail_list = [r['psi_detail'] for r in results if len(r['psi_detail']) > 0]
        psi_detail_df = pd.concat(psi_detail_list, ignore_index=True) if psi_detail_list else pd.DataFrame()
        iv_df = pd.DataFrame([r['iv'] for r in results])
        bivar_list = [r['bivar'] for r in results if len(r['bivar']) > 0]
        bivar_df = pd.concat(bivar_list, ignore_index=True) if bivar_list else pd.DataFrame()
        auc_df = pd.DataFrame([r['auc'] for r in results])
        woe_list = [r['woe'] for r in results if len(r['woe']) > 0]
        woe_df = pd.concat(woe_list, ignore_index=True) if woe_list else pd.DataFrame()

        # ==== 写入 Excel ====
        logger.info(f"写入Excel: {output_path}")
        output_dir = os.path.dirname(output_path)
        if output_dir:
            FeatureAnalysisToolkit.ensure_dir(output_dir)
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            coverage_df.to_excel(writer, sheet_name='覆盖率分析', index=False)
            psi_summary_df.to_excel(writer, sheet_name='PSI汇总', index=False)
            if len(psi_detail_df) > 0:
                psi_detail_df.to_excel(writer, sheet_name='PSI明细', index=False)
            iv_df.to_excel(writer, sheet_name='IV', index=False)
            if len(bivar_df) > 0:
                bivar_df.to_excel(writer, sheet_name='Bivar', index=False)
            if compute_woe and len(woe_df) > 0:
                woe_df.to_excel(writer, sheet_name='WOE', index=False)
            auc_df.to_excel(writer, sheet_name='AUC', index=False)

        logger.info(f"特征分析报告已保存: {output_path}")
        return output_path


def _feature_analysis_one_feat(args):
    """
    单特征分析 worker（模块级函数，支持 multiprocessing pickle）
    一次分箱同时输出覆盖率/PSI汇总+明细/IV/Bivar/WOE/AUC
    """
    (feat, feat_series, y_all, group_values, compare_values, base_group_value,
     y_groups, groups_data, n_bins, compute_woe) = args

    coverage_row = {'feature': feat}
    psi_summary_row = {'feature': feat}
    iv_row = {'feature': feat}
    auc_row = {'feature': feat}
    psi_detail_dfs = []
    bivar_rows = []
    woe_df = pd.DataFrame()

    # --- 覆盖率 ---
    coverage_row['overall'] = round(1 - feat_series.isna().sum() / len(feat_series), 4)
    for gv in group_values:
        sub = groups_data[gv]
        coverage_row[str(gv)] = round(1 - sub.isna().sum() / len(sub), 4)

    # --- 确定分箱边界（基于基准期等频分箱）---
    base_series = groups_data.get(base_group_value, feat_series)
    base_non_null = base_series.dropna()
    breakpoints = None
    if len(base_non_null) >= 2 and base_non_null.nunique() >= 2:
        quantiles = np.quantile(base_non_null, np.linspace(0, 1, n_bins + 1))
        breakpoints = sorted(set(quantiles))
        if len(breakpoints) < 2:
            breakpoints = None

    # --- 对所有分组 + overall 一次分箱统计 ---
    all_bin_stats = {}
    for gv in group_values:
        bins, tg, tb = FeatureAnalysisToolkit._bin_stats_one_group(
            groups_data[gv], y_groups[gv], breakpoints)
        all_bin_stats[str(gv)] = (bins, tg, tb)
    bins_overall, tg_o, tb_o = FeatureAnalysisToolkit._bin_stats_one_group(
        feat_series, y_all, breakpoints)

    # --- PSI 汇总 + 明细 ---
    base_bins, base_tg, base_tb = all_bin_stats[str(base_group_value)]
    base_total = base_tg + base_tb
    for gv in compare_values:
        comp_bins, comp_tg, comp_tb = all_bin_stats[str(gv)]
        comp_total = comp_tg + comp_tb
        if base_total == 0 or comp_total == 0:
            psi_summary_row[str(gv)] = np.nan
            continue
        comp_bin_map = {b['bin_range']: b for b in comp_bins}
        detail_rows = []
        for b_bin in base_bins:
            e_count = b_bin['total']
            c_bin = comp_bin_map.get(b_bin['bin_range'])
            a_count = c_bin['total'] if c_bin else 0
            e_pct = max(e_count / base_total, 1e-10)
            a_pct = max(a_count / comp_total, 1e-10)
            psi_c = (a_pct - e_pct) * np.log(a_pct / e_pct)
            detail_rows.append({
                'feature': feat, 'base_period': str(base_group_value),
                'compare_period': str(gv), 'bin_range': b_bin['bin_range'],
                'expected_count': e_count, 'actual_count': a_count,
                'expected_pct': round(e_pct, 6), 'actual_pct': round(a_pct, 6),
                'psi_contribution': round(psi_c, 6)
            })
        total_psi = sum(r['psi_contribution'] for r in detail_rows)
        label = 'Stable' if total_psi < 0.1 else ('Slightly unstable' if total_psi < 0.25 else 'Unstable')
        detail_rows.append({
            'feature': feat, 'base_period': str(base_group_value),
            'compare_period': str(gv), 'bin_range': 'TOTAL',
            'expected_count': base_total, 'actual_count': comp_total,
            'expected_pct': '-', 'actual_pct': '-',
            'psi_contribution': round(total_psi, 4), 'label': label
        })
        psi_detail_dfs.append(pd.DataFrame(detail_rows))
        psi_summary_row[str(gv)] = round(total_psi, 4)
    psi_summary_row[str(base_group_value)] = '-'

    # --- IV ---
    for lbl, (bins, tg, tb) in [('overall', (bins_overall, tg_o, tb_o))] + \
            [(str(gv), all_bin_stats[str(gv)]) for gv in group_values]:
        if tg == 0 or tb == 0:
            iv_row[lbl] = 0.0
            continue
        feat_iv = 0.0
        for b in bins:
            gc = max(b['good'], 0.5)
            bc = max(b['bad'], 0.5)
            woe = np.log((gc / tg) / (bc / tb))
            feat_iv += ((gc / tg) - (bc / tb)) * woe
        iv_row[lbl] = round(feat_iv, 4)

    # --- Bivar：每个 bin 的占比 + bad_rate ---
    overall_total = tg_o + tb_o
    for b in bins_overall:
        pct = b['total'] / overall_total if overall_total > 0 else 0
        bad_rate = b['bad'] / b['total'] if b['total'] > 0 else 0
        cum_pct = 0  # 会在后面累加
        bivar_rows.append({
            'feature': feat, 'bin_range': b['bin_range'],
            'total_count': b['total'], 'good_count': b['good'], 'bad_count': b['bad'],
            'pct': round(pct, 4), 'cum_pct': 0,
            'bad_rate': round(bad_rate, 4), 'woe': 0
        })
    # 累加 cum_pct 和 woe
    if tg_o > 0 and tb_o > 0:
        cum = 0.0
        for b, bv in zip(bins_overall, bivar_rows):
            gc = max(b['good'], 0.5)
            bc = max(b['bad'], 0.5)
            woe = round(np.log((gc / tg_o) / (bc / tb_o)), 4)
            cum += bv['pct']
            bv['cum_pct'] = round(cum, 4)
            bv['woe'] = woe

    # --- AUC ---
    auc_row['overall'] = FeatureAnalysisToolkit.calculate_single_auc(y_all, feat_series.values)
    for gv in group_values:
        auc_row[str(gv)] = FeatureAnalysisToolkit.calculate_single_auc(y_groups[gv], groups_data[gv].values)

    # --- WOE 明细（可选）---
    if compute_woe and tg_o > 0 and tb_o > 0:
        woe_rows = []
        for b in bins_overall:
            gc = max(b['good'], 0.5)
            bc = max(b['bad'], 0.5)
            woe = np.log((gc / max(tg_o, 0.5)) / (bc / max(tb_o, 0.5)))
            iv = ((gc / max(tg_o, 0.5)) - (bc / max(tb_o, 0.5))) * woe
            woe_rows.append({
                'feature': feat, 'bin_range': b['bin_range'],
                'total_count': b['total'], 'good_count': b['good'], 'bad_count': b['bad'],
                'bad_rate': round(b['bad'] / b['total'], 4) if b['total'] > 0 else 0,
                'woe': round(woe, 4), 'iv': round(iv, 4)
            })
        total_iv = sum(r['iv'] for r in woe_rows)
        total_cnt = sum(r['total_count'] for r in woe_rows)
        total_bad_cnt = sum(r['bad_count'] for r in woe_rows)
        woe_rows.append({
            'feature': feat, 'bin_range': 'TOTAL',
            'total_count': total_cnt, 'good_count': total_cnt - total_bad_cnt,
            'bad_count': total_bad_cnt,
            'bad_rate': round(total_bad_cnt / total_cnt, 4) if total_cnt > 0 else 0,
            'woe': '-', 'iv': round(total_iv, 4)
        })
        woe_df = pd.DataFrame(woe_rows)

    return {
        'coverage': coverage_row,
        'psi_summary': psi_summary_row,
        'psi_detail': pd.concat(psi_detail_dfs, ignore_index=True) if psi_detail_dfs else pd.DataFrame(),
        'iv': iv_row,
        'bivar': pd.DataFrame(bivar_rows) if bivar_rows else pd.DataFrame(),
        'auc': auc_row,
        'woe': woe_df
    }


class AutoModelBuilder:
    def __init__(self, model_type='lgb', random_state=42):
        """
        初始化自动化建模工具
        :param model_type: 模型类型，支持'lgb', 'xgb', 'lr'
        :param random_state: 随机种子
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.scaler = None
        self.features = None
        self.target = None

    @staticmethod
    def load_xgb_booster(model_path):
        """
        加载 XGBoost 模型并返回 (booster, best_iteration)
        支持 .pkl（joblib 序列化）和 .json（XGBoost 原生格式）
        """
        if model_path.endswith('.pkl'):
            model = joblib.load(model_path)
            if hasattr(model, 'nativeBooster'):
                booster = model.nativeBooster
            elif isinstance(model, xgb.Booster):
                booster = model
            elif hasattr(model, 'get_booster'):
                booster = model.get_booster()
            else:
                booster = model
            best_iter = getattr(booster, 'best_iteration', 0)
            return booster, best_iter
        elif model_path.endswith('.json'):
            booster = xgb.Booster()
            booster.load_model(model_path)
            return booster, 0
        raise ValueError(f"不支持的模型格式: {model_path}")

    def load_data(self, file_path, target_col, file_format='csv', delimiter=',', encoding='utf-8'):
        """
        加载数据
        :param file_path: 文件路径
        :param target_col: 目标列名
        :param file_format: 文件格式，支持'csv', 'libsvm'
        :param delimiter: CSV文件分隔符
        :param encoding: 文件编码
        :return: 特征和目标变量
        """
        logger.info(f"Loading data from {file_path} with format {file_format}")

        if file_format == 'csv':
            data = FeatureAnalysisToolkit.load_csv(file_path, sep=delimiter, encoding=encoding)
        elif file_format == 'libsvm':
            from sklearn.datasets import load_svmlight_file
            X, y = load_svmlight_file(file_path)
            data = pd.DataFrame(X.toarray())
            data[target_col] = y
        else:
            raise ValueError("Unsupported file format. Only 'csv' and 'libsvm' are supported.")

        self.target = target_col
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        self.features = [col for col in numeric_cols if col != target_col]
        
        logger.info(f"Data loaded successfully. Shape: {data.shape}")
        logger.info(f"Features: {self.features}")
        logger.info(f"Target column: {self.target}")
        
        return data[self.features], data[target_col]
    
    def eda_analysis(self, X, y, output_dir='eda_results'):
        """
        特征EDA分析（委托给FeatureAnalysisToolkit）
        """
        return FeatureAnalysisToolkit.eda_analysis(X, y, output_dir=output_dir, random_state=self.random_state)
    
    def hyperparameter_tuning(self, X, y, tuning_method='bayesian', params=None, n_iter=50):
        """
        模型超参数调优
        :param X: 特征数据
        :param y: 目标数据
        :param tuning_method: 调优方法，支持'bayesian', 'grid'
        :param params: 调参范围
        :param n_iter: 贝叶斯优化迭代次数
        :return: 最佳参数
        """
        logger.info(f"Performing hyperparameter tuning with {tuning_method}")
        
        if self.model_type == 'lgb':
            if params is None:
                params = {
                    'learning_rate': (0.01, 0.1),
                    'n_estimators': (100, 1000),
                    'max_depth': (3, 10),
                    'num_leaves': (20, 100),
                    'min_data_in_leaf': (10, 100),
                    'feature_fraction': (0.6, 1.0),
                    'bagging_fraction': (0.6, 1.0),
                    'bagging_freq': (1, 10)
                }
            
            if tuning_method == 'bayesian':
                from lightgbm import early_stopping as lgb_early_stopping, log_evaluation
                rs = self.random_state
                X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.3, random_state=rs)
                def lgb_eval(learning_rate, n_estimators, max_depth, num_leaves,
                             min_data_in_leaf, feature_fraction, bagging_fraction, bagging_freq):
                    model = lgb.LGBMClassifier(
                        learning_rate=learning_rate,
                        n_estimators=int(n_estimators),
                        max_depth=int(max_depth),
                        num_leaves=int(num_leaves),
                        min_data_in_leaf=int(min_data_in_leaf),
                        feature_fraction=feature_fraction,
                        bagging_fraction=bagging_fraction,
                        bagging_freq=int(bagging_freq),
                        random_state=rs
                    )
                    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                              callbacks=[lgb_early_stopping(50, verbose=False), log_evaluation(period=0)])
                    return model.best_score_['valid_0']['auc']
                
                bo = BayesianOptimization(lgb_eval, params, random_state=self.random_state)
                bo.maximize(init_points=10, n_iter=n_iter)
                best_params = bo.max['params']
                best_params['n_estimators'] = int(best_params['n_estimators'])
                best_params['max_depth'] = int(best_params['max_depth'])
                best_params['num_leaves'] = int(best_params['num_leaves'])
                best_params['min_data_in_leaf'] = int(best_params['min_data_in_leaf'])
                best_params['bagging_freq'] = int(best_params['bagging_freq'])
            
            elif tuning_method == 'grid':
                model = lgb.LGBMClassifier(random_state=self.random_state)
                grid_params = {
                    'learning_rate': [0.01, 0.05, 0.1],
                    'n_estimators': [300, 500, 700],
                    'max_depth': [5, 7, 9],
                    'num_leaves': [31, 63, 127]
                }
                grid = GridSearchCV(model, grid_params, cv=3, scoring='roc_auc', n_jobs=-1)
                grid.fit(X, y)
                best_params = grid.best_params_

        elif self.model_type == 'xgb':
            if params is None:
                params = {
                    'learning_rate': (0.01, 0.1),
                    'n_estimators': (100, 1000),
                    'max_depth': (3, 10),
                    'min_child_weight': (1, 10),
                    'subsample': (0.6, 1.0),
                    'colsample_bytree': (0.6, 1.0),
                    'gamma': (0, 1)
                }
            
            if tuning_method == 'bayesian':
                rs = self.random_state
                X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.3, random_state=rs)
                def xgb_eval(learning_rate, n_estimators, max_depth, min_child_weight,
                             subsample, colsample_bytree, gamma):
                    model = xgb.XGBClassifier(
                        learning_rate=learning_rate,
                        n_estimators=int(n_estimators),
                        max_depth=int(max_depth),
                        min_child_weight=min_child_weight,
                        subsample=subsample,
                        colsample_bytree=colsample_bytree,
                        gamma=gamma,
                        random_state=rs,
                        early_stopping_rounds=50
                    )
                    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
                    return model.best_score_['validation_0']['auc']
                
                bo = BayesianOptimization(xgb_eval, params, random_state=self.random_state)
                bo.maximize(init_points=10, n_iter=n_iter)
                best_params = bo.max['params']
                best_params['n_estimators'] = int(best_params['n_estimators'])
                best_params['max_depth'] = int(best_params['max_depth'])
            
            elif tuning_method == 'grid':
                model = xgb.XGBClassifier(random_state=self.random_state)
                grid_params = {
                    'learning_rate': [0.01, 0.05, 0.1],
                    'n_estimators': [300, 500, 700],
                    'max_depth': [5, 7, 9],
                    'min_child_weight': [1, 3, 5]
                }
                grid = GridSearchCV(model, grid_params, cv=3, scoring='roc_auc', n_jobs=-1)
                grid.fit(X, y)
                best_params = grid.best_params_
        
        elif self.model_type == 'lr':
            if params is None:
                params = {
                    'C': [0.001, 0.01, 0.1, 1.0, 10.0],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear']
                }

            if tuning_method != 'grid':
                logger.warning("LR 暂不支持贝叶斯优化，自动回退到 grid search")
            model = LogisticRegression(random_state=self.random_state)
            grid = GridSearchCV(model, params, cv=3, scoring='roc_auc', n_jobs=-1)
            grid.fit(X, y)
            best_params = grid.best_params_
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")

        logger.info(f"Hyperparameter tuning completed. Best params: {best_params}")
        return best_params
    
    def _prepare_features(self, X):
        """预处理特征数据：LR路径做fillna+scale，其他路径直接返回"""
        if self.scaler and self.model_type == 'lr':
            X_filled = X.fillna(0) if hasattr(X, 'fillna') else X
            return self.scaler.transform(X_filled)
        return X

    def train(self, X, y, params=None, eval_set=None, early_stopping_rounds=None):
        """
        训练模型
        :param X: 特征数据
        :param y: 目标数据
        :param params: 模型参数
        :param eval_set: 验证集 [(X_val, y_val), ...]，用于监控过拟合
        :param early_stopping_rounds: 早停轮数
        :return: 训练好的模型
        """
        logger.info(f"Training {self.model_type} model")

        if self.model_type == 'lgb':
            if params is None:
                params = {
                    'learning_rate': 0.05,
                    'n_estimators': 500,
                    'max_depth': 7,
                    'num_leaves': 63,
                    'random_state': self.random_state
                }
            self.model = lgb.LGBMClassifier(**params)
            fit_kwargs = {}
            if eval_set:
                fit_kwargs['eval_set'] = eval_set
            if early_stopping_rounds:
                from lightgbm import early_stopping as lgb_early_stopping, log_evaluation  # noqa: F811
                fit_kwargs['callbacks'] = [
                    lgb_early_stopping(early_stopping_rounds, verbose=False),
                    log_evaluation(period=0)
                ]
            self.model.fit(X, y, **fit_kwargs)

        elif self.model_type == 'xgb':
            if params is None:
                params = {
                    'learning_rate': 0.05,
                    'n_estimators': 500,
                    'max_depth': 7,
                    'random_state': self.random_state
                }
            self.model = xgb.XGBClassifier(**params)
            fit_kwargs = {}
            if eval_set:
                fit_kwargs['eval_set'] = eval_set
            if early_stopping_rounds:
                fit_kwargs['early_stopping_rounds'] = early_stopping_rounds
            self.model.fit(X, y, verbose=False, **fit_kwargs)

        elif self.model_type == 'lr':
            self.scaler = StandardScaler()
            X_filled = X.fillna(0) if hasattr(X, 'fillna') else X
            X_scaled = self.scaler.fit_transform(X_filled)
            if params is None:
                params = {
                    'C': 1.0,
                    'penalty': 'l2',
                    'solver': 'liblinear',
                    'random_state': self.random_state
                }
            self.model = LogisticRegression(**params)
            self.model.fit(X_scaled, y)

        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
    
    def save_model(self, model_path):
        """
        保存模型
        :param model_path: 模型保存路径
        """
        logger.info(f"Saving model to {model_path}")

        model_dir = os.path.dirname(model_path)
        if model_dir:
            FeatureAnalysisToolkit.ensure_dir(model_dir)

        # 保存模型和相关信息
        model_info = {
            'model': self.model,
            'model_type': self.model_type,
            'features': self.features,
            'target': self.target,
            'scaler': self.scaler
        }

        joblib.dump(model_info, model_path)
        logger.info("Model saved successfully")

    def load_model(self, model_path):
        """
        加载模型
        :param model_path: 模型加载路径
        :return: 加载的模型
        """
        logger.info(f"Loading model from {model_path}")

        model_info = joblib.load(model_path)

        self.model = model_info['model']
        self.model_type = model_info['model_type']
        self.features = model_info['features']
        self.target = model_info['target']
        self.scaler = model_info['scaler']

        logger.info("Model loaded successfully")
        return self.model
    
    def get_feature_importance(self, output_file=None):
        """
        获取特征重要性
        :param output_file: 输出文件路径
        :return: 特征重要性
        """
        logger.info("Getting feature importance")
        
        if self.model is None:
            raise ValueError("Model is not trained or loaded")
        
        if self.model_type in ['lgb', 'xgb']:
            importance = self.model.feature_importances_
            feature_importance = pd.DataFrame({
                'feature': self.features,
                'importance': importance
            }).sort_values('importance', ascending=False)
        elif self.model_type == 'lr':
            importance = np.abs(self.model.coef_[0])
            feature_importance = pd.DataFrame({
                'feature': self.features,
                'importance': importance
            }).sort_values('importance', ascending=False)
        
        if output_file:
            # 创建目录
            output_dir = os.path.dirname(output_file)
            if output_dir:
                FeatureAnalysisToolkit.ensure_dir(output_dir)
            feature_importance.to_csv(output_file, index=False)
            logger.info(f"Feature importance saved to {output_file}")
        
        return feature_importance
    
    def shap_analysis(self, X, output_dir='shap_results', use_pred_contribs=True):
        """
        SHAP分析
        :param X: 特征数据
        :param output_dir: 结果输出目录
        :param use_pred_contribs: XGB是否额外使用pred_contribs计算SHAP
        :return: SHAP值
        """
        logger.info("Performing SHAP analysis")

        if self.model is None:
            raise ValueError("Model is not trained or loaded")

        # 创建输出目录
        FeatureAnalysisToolkit.ensure_dir(output_dir)

        # 准备数据
        X_processed = self._prepare_features(X)

        # SHAP分析
        if self.model_type in ['lgb', 'xgb']:
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer(X_processed)
        elif self.model_type == 'lr':
            explainer = shap.LinearExplainer(self.model, X_processed)
            shap_values = explainer(X_processed)

        # 保存SHAP值
        sv = shap_values.values
        shap_values_df = pd.DataFrame(sv, columns=self.features)
        shap_values_df.to_csv(os.path.join(output_dir, 'shap_values.csv'), index=False)

        # 汇总SHAP值
        shap_summary = pd.DataFrame({
            'feature': self.features,
            'mean_abs_shap': np.abs(sv).mean(axis=0)
        }).sort_values('mean_abs_shap', ascending=False)
        shap_summary.to_csv(os.path.join(output_dir, 'shap_summary.csv'), index=False)

        # XGB 额外使用 pred_contribs 方式计算 SHAP
        if self.model_type == 'xgb' and use_pred_contribs:
            self._shap_contribs(X_processed, output_dir)

        logger.info("SHAP analysis completed")
        return shap_values

    def _shap_contribs(self, X, output_dir):
        """
        XGB 原生 pred_contribs 方式计算 SHAP 值（无需 shap 包）
        :param X: 特征数据
        :param output_dir: 输出目录
        """
        logger.info("Computing SHAP via xgb.Booster.predict(pred_contribs=True)")
        try:
            booster = self.model.get_booster()
            X_processed = self._prepare_features(X)
            missing = 0.0
            dm = xgb.DMatrix(X_processed, feature_names=self.features, missing=missing)

            # pred_contribs 返回 shape=(n_samples, n_features+1)，最后一列是 bias
            contribs = booster.predict(dm, pred_contribs=True)
            feat_contribs = contribs[:, :-1]  # 去掉 bias 列
            bias = contribs[0, -1]

            # 保存
            contribs_df = pd.DataFrame(feat_contribs, columns=self.features)
            contribs_df.to_csv(os.path.join(output_dir, 'shap_contribs.csv'), index=False)

            contribs_summary = pd.DataFrame({
                'feature': self.features,
                'mean_abs_shap': np.abs(feat_contribs).mean(axis=0)
            }).sort_values('mean_abs_shap', ascending=False)
            contribs_summary.to_csv(os.path.join(output_dir, 'shap_contribs_summary.csv'), index=False)

            logger.info(f"pred_contribs SHAP 完成, bias={bias:.6f}")
        except Exception as e:
            logger.warning(f"pred_contribs 计算失败: {e}")
    
    def predict(self, X):
        """
        模型预测
        :param X: 特征数据
        :return: 预测结果
        """
        logger.info("Making predictions")
        
        if self.model is None:
            raise ValueError("Model is not trained or loaded")
        
        # 准备数据
        X_processed = self._prepare_features(X)

        # 预测概率
        if hasattr(self.model, 'predict_proba'):
            predictions = self.model.predict_proba(X_processed)[:, 1]
        else:
            predictions = self.model.predict(X_processed)
        
        return predictions

# ==============================
# 模型性能异动归因分析器
# ==============================
class ModelAttributionAnalyzer:
    """
    模型性能异动归因分析
    定位模型AUC下降的原因：特征分布偏移(PSI)、特征预测力下降(IV/AUC)、特征重要性(Permutation Importance)
    """

    def __init__(self, model_path, features_path, missing_value=-999.0):
        """
        :param model_path: XGBoost模型文件路径(.pkl)
        :param features_path: 模型特征列表文件路径(.pkl)
        :param missing_value: 缺失值填充值
        """
        self.missing_value = missing_value
        logger.info("加载模型和特征列表...")
        self.model = joblib.load(model_path)
        self.features = joblib.load(features_path)
        if not isinstance(self.features, list):
            raise ValueError(f"特征文件格式错误，期望list，实际为{type(self.features)}")
        logger.info(f"模型加载完成，特征数: {len(self.features)}")

    def analyze_distribution_shift(self, df, time_col, target_col, baseline_month, current_month,
                                   info_vars=None, output_path=None):
        """
        特征分布偏移分析（PSI + IV + 单特征AUC）
        :param df: 完整数据（含所有月份）
        :param time_col: 时间列名
        :param target_col: 目标变量列名
        :param baseline_month: 基准月份值
        :param current_month: 当前月份值
        :param info_vars: 排除的非特征列
        :param output_path: 结果输出CSV路径
        :return: DataFrame[feature, psi, iv_old, iv_new, iv_drop, s_auc_old, s_auc_new, s_auc_drop]
        """
        logger.info(f"=== 特征分布偏移分析 === 基准月={baseline_month}, 当前月={current_month}")

        if info_vars is None:
            info_vars = []
        feature_cols = [c for c in df.columns if c not in info_vars and c != target_col and c != time_col]

        # 自动转换时间列类型以匹配 baseline/current_month 参数
        baseline_month, current_month = FeatureAnalysisToolkit.coerce_time_value(
            df[time_col].dtype, baseline_month, current_month)

        df_old = df[df[time_col] == baseline_month].copy()
        df_new = df[df[time_col] == current_month].copy()
        logger.info(f"基准月样本: {len(df_old)}, 当前月样本: {len(df_new)}")

        X_old = df_old[feature_cols].fillna(self.missing_value)
        X_new = df_new[feature_cols].fillna(self.missing_value)
        y_old = df_old[target_col]
        y_new = df_new[target_col]

        # 整体AUC对比
        dm_old = xgb.DMatrix(X_old[self.features], missing=self.missing_value)
        dm_new = xgb.DMatrix(X_new[self.features], missing=self.missing_value)
        auc_old = roc_auc_score(y_old, self.model.predict(dm_old))
        auc_new = roc_auc_score(y_new, self.model.predict(dm_new))
        auc_drop = auc_old - auc_new
        logger.info(f"基准月 AUC: {auc_old}, 当前月 AUC: {auc_new}, AUC下降: {auc_drop}")

        stat_list = []
        total = len(feature_cols)
        for i, f in enumerate(tqdm(feature_cols, desc="分布偏移分析")):
            psi = FeatureAnalysisToolkit.calculate_psi(X_new[f], X_old[f])
            iv_old = FeatureAnalysisToolkit.calculate_single_iv(df_old, f, target_col)
            iv_new = FeatureAnalysisToolkit.calculate_single_iv(df_new, f, target_col)
            sa_old = FeatureAnalysisToolkit.calculate_single_auc(y_old, X_old[f].values)
            sa_new = FeatureAnalysisToolkit.calculate_single_auc(y_new, X_new[f].values)
            stat_list.append({
                "feature": f,
                "psi": psi,
                "iv_old": iv_old,
                "iv_new": iv_new,
                "iv_drop": round(iv_old - iv_new, 4),
                "s_auc_old": sa_old,
                "s_auc_new": sa_new,
                "s_auc_drop": round(sa_old - sa_new, 4)
            })

        df_stat = pd.DataFrame(stat_list)
        if output_path:
            df_stat.to_csv(output_path, index=False)
            logger.info(f"分布偏移结果已保存: {output_path}")

        return df_stat, {"auc_old": auc_old, "auc_new": auc_new, "auc_drop": auc_drop}

    def permutation_importance(self, df, time_col, target_col, current_month,
                               info_vars=None, n_workers=1, output_path=None):
        """
        特征消融（Permutation Importance）
        :param df: 完整数据
        :param time_col: 时间列名
        :param target_col: 目标变量列名
        :param current_month: 分析月份
        :param info_vars: 排除列
        :param n_workers: 并行进程数（1=单进程）
        :param output_path: 结果输出CSV路径
        :return: DataFrame[feature, abl_auc, abl_delta]
        """
        logger.info(f"=== 特征消融分析 === 月份={current_month}, 并行={n_workers}")

        if info_vars is None:
            info_vars = []
        current_month = FeatureAnalysisToolkit.coerce_time_value(df[time_col].dtype, current_month)
        df_new = df[df[time_col] == current_month].copy()
        df_new[self.features] = df_new[self.features].fillna(self.missing_value)
        df_new[self.features] = df_new[self.features].replace([np.inf, -np.inf], self.missing_value)

        X_np = df_new[self.features].values.astype(np.float32)
        y_arr = df_new[target_col].values

        # baseline AUC
        dtest_base = xgb.DMatrix(X_np, feature_names=self.features, missing=self.missing_value)
        auc_base = roc_auc_score(y_arr, self.model.predict(dtest_base))
        logger.info(f"Baseline AUC: {auc_base:.4f}")

        abl_list = []

        if n_workers <= 1:
            # 单进程模式
            for i, f in enumerate(self.features):
                original_vals = X_np[:, i].copy()
                X_np[:, i] = np.random.permutation(original_vals)
                try:
                    dtest_abl = xgb.DMatrix(X_np, feature_names=self.features, missing=self.missing_value, nthread=1)
                    pred_abl = self.model.predict(dtest_abl)
                    new_auc = roc_auc_score(y_arr, pred_abl)
                    abl_list.append({
                        "feature": f,
                        "abl_auc": round(new_auc, 4),
                        "abl_delta": round(new_auc - auc_base, 4)
                    })
                    if (i + 1) % 50 == 0:
                        logger.info(f"进度: {i+1}/{len(self.features)}")
                finally:
                    X_np[:, i] = original_vals
        else:
            # 多进程模式
            model_tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
            model_tmp.close()
            self.model.save_model(model_tmp.name)
            model_path_tmp = model_tmp.name

            args_list = [(i, f, X_np, y_arr, model_path_tmp, auc_base, self.missing_value)
                         for i, f in enumerate(self.features)]

            results = [None] * len(self.features)
            try:
                with ProcessPoolExecutor(max_workers=n_workers) as executor:
                    futures = {executor.submit(_permutation_compute_worker, arg): idx
                               for idx, arg in enumerate(args_list)}
                    completed = 0
                    for future in as_completed(futures):
                        result = future.result()
                        idx = futures[future]
                        results[idx] = result
                        completed += 1
                        if completed % 50 == 0:
                            logger.info(f"完成: {completed}/{len(self.features)}")
            finally:
                os.unlink(model_path_tmp)
            abl_list = [r for r in results if r is not None and 'error' not in r]

        df_abl = pd.DataFrame(abl_list)
        if output_path:
            df_abl.to_csv(output_path, index=False)
            logger.info(f"消融结果已保存: {output_path}")
        return df_abl, auc_base

    def full_attribution(self, df, time_col, target_col, baseline_month, current_month,
                         info_vars=None, n_workers=1, output_dir=None):
        """
        完整归因分析流程：分布偏移 + 特征消融
        :return: (df_stat, df_abl, summary_dict)
        """
        if output_dir:
            FeatureAnalysisToolkit.ensure_dir(output_dir)

        shift_path = os.path.join(output_dir, 'distribution_shift.csv') if output_dir else None
        abl_path = os.path.join(output_dir, 'permutation_importance.csv') if output_dir else None

        df_stat, auc_summary = self.analyze_distribution_shift(
            df, time_col, target_col, baseline_month, current_month, info_vars, shift_path)

        df_abl, auc_base = self.permutation_importance(
            df, time_col, target_col, current_month, info_vars, n_workers, abl_path)

        # 合并汇总
        df_merged = pd.merge(df_stat, df_abl[['feature', 'abl_auc', 'abl_delta']], on='feature', how='left')
        summary = {
            **auc_summary,
            "baseline_month": baseline_month,
            "current_month": current_month,
            "total_features_analyzed": len(df_stat),
            "model_features_count": len(self.features),
            "top_psi_features": df_stat.nlargest(10, 'psi')['feature'].tolist(),
            "top_iv_drop_features": df_stat.nlargest(10, 'iv_drop')['feature'].tolist(),
            "top_abl_features": (df_abl.nsmallest(10, 'abl_delta')['feature'].tolist()
                                  if 'abl_delta' in df_abl.columns else [])
        }
        if output_dir:
            df_merged.to_csv(os.path.join(output_dir, 'attribution_merged.csv'), index=False)
            pd.DataFrame([summary]).to_csv(os.path.join(output_dir, 'attribution_summary.csv'), index=False)
            logger.info(f"完整归因结果已保存到: {output_dir}")

        logger.info("=" * 60)
        logger.info("归因分析汇总:")
        logger.info(f"  AUC下降: {auc_summary['auc_drop']}")
        logger.info(f"  Top-PSI特征(分布偏移): {summary['top_psi_features'][:5]}")
        logger.info(f"  Top-IV下降特征(预测力下降): {summary['top_iv_drop_features'][:5]}")
        logger.info(f"  Top消融特征(对模型影响最大): {summary['top_abl_features'][:5]}")
        logger.info("=" * 60)

        return df_stat, df_abl, summary


# ==============================
# 模型报告生成器
# ==============================
class ModelReportGenerator:
    """
    模型报告生成器
    支持 Model Summary、Model Stability、模型分相关性 等报告模块
    依赖外部 reportbuilderv4 库
    """

    def __init__(self, workbook_name, reportbuilder_path=None, ui='future', is_return=False):
        """
        :param workbook_name: 报告名称
        :param reportbuilder_path: reportbuilderv4 库路径
        :param ui: 报告样式
        :param is_return: 是否返回报告对象
        """
        if reportbuilder_path:
            sys.path.append(reportbuilder_path)
        try:
            from reportbuilderv4 import ModelReport
            self.report = ModelReport(workbookname=workbook_name, ui=ui, is_return=is_return)
            logger.info(f"报告构建器初始化成功: {workbook_name}")
        except ImportError as e:
            raise ImportError(f"导入 reportbuilderv4 失败: {e}, 请检查路径: {reportbuilder_path}")
        self.workbook_name = workbook_name

    @staticmethod
    def load_data(file_path, separator='\t', column_mapping=None, fill_na_value=-999.0,
                  targets=None, score_list=None):
        """
        加载和预处理数据
        :return: (df, df_targets_dict)
        """
        logger.info(f"加载数据: {file_path}")
        df = pd.read_csv(file_path, sep=separator)
        df.columns = df.columns.str.lower()
        logger.info(f"数据量: {len(df):,} 行, {len(df.columns)} 列")

        if column_mapping:
            df.rename(columns=column_mapping, inplace=True)

        if score_list:
            df[score_list] = df[score_list].apply(pd.to_numeric, errors='coerce')
            df[score_list] = df[score_list].fillna(fill_na_value)

        df_targets = {}
        if targets:
            for target in targets:
                df[target] = df[target].astype(float)
                df_valid = df[df[target].isin([0, 1])]
                df_targets[target] = df_valid
                logger.info(f"目标变量 {target} 有效样本: {len(df_valid):,}")

        return df, df_targets

    @staticmethod
    def create_mask(df, conditions):
        """根据条件字典创建过滤掩码"""
        mask = pd.Series(True, index=df.index)
        for field, value in conditions.items():
            if isinstance(value, list):
                mask &= df[field].isin(value)
            else:
                mask &= (df[field] == value)
        return mask

    @classmethod
    def create_groups(cls, df, groups_config, is_nested=True):
        """
        通用客群生成
        :param is_nested: True=嵌套结构 {dict_name: {group_name: {conditions: ...}}}, False=平铺结构
        """
        result = {}
        if is_nested:
            for dict_name, groups in groups_config.items():
                result[dict_name] = {}
                for group_name, config in groups.items():
                    if 'conditions' not in config:
                        raise KeyError(f"组 '{group_name}' 缺少 'conditions' 键")
                    mask = cls.create_mask(df, config['conditions'])
                    result[dict_name][group_name] = df[mask]
                    logger.info(f"  {dict_name} - {group_name}: {len(df[mask]):,} 条")
        else:
            for group_name, config in groups_config.items():
                mask = cls.create_mask(df, config['conditions'])
                result[group_name] = df[mask]
                logger.info(f"客群 {group_name}: {len(df[mask]):,} 条")
        return result

    def generate_model_summary(self, customer_groups, target, benchmark, score_list, n_groups=20):
        """生成 Model Summary 报告"""
        logger.info(f"生成 Model Summary: target={target}")
        for dict_name, groups in customer_groups.items():
            data_dict = {name: data for name, data in groups.items() if len(data) > 0}
            if data_dict:
                try:
                    self.report.model_summary(
                        data_dict, target, benchmark, None,
                        [col for col in score_list if col != benchmark],
                        True, n_groups, None,
                        f'MS_{target}_{dict_name.replace("df_summary_", "")}')
                    logger.info(f"Model Summary 完成: {target} - {dict_name}")
                except Exception as e:
                    logger.error(f"Model Summary 失败 {target} - {dict_name}: {e}")

    def generate_model_stability(self, stability_groups, target, score_list, segvar='draw_month', n_groups=20):
        """生成 Model Stability 报告"""
        logger.info(f"生成 Model Stability: target={target}")
        for group_name, group_data in stability_groups.items():
            if len(group_data) > 0:
                try:
                    self.report.model_stability(
                        {group_name: group_data}, segvar, score_list, target,
                        [group_name], None, f'稳定性_{target}_{group_name}', n_groups)
                    logger.info(f"Model Stability 完成: {target} - {group_name}")
                except Exception as e:
                    logger.error(f"Model Stability 失败 {target} - {group_name}: {e}")

    def generate_correlation(self, df, target, score_list, correlation_months,
                             time_col='draw_month', filter_conditions=None, threshold=0.9):
        """生成模型分相关性报告"""
        logger.info(f"生成模型分相关性: target={target}")
        try:
            mask = pd.Series(True, index=df.index)
            if correlation_months:
                mask &= df[time_col].isin(correlation_months)
            if filter_conditions:
                for field, value in filter_conditions.items():
                    if isinstance(value, list):
                        mask &= df[field].isin(value)
                    else:
                        mask &= (df[field] == value)

            # 所有模型分 > 0
            mask_scores = reduce(operator.and_, [df[col] > 0 for col in score_list])
            df_corr = df[mask & mask_scores]
            logger.info(f"相关性数据量: {len(df_corr):,}")

            self.report.feature_corr(
                {'data_corr': df_corr}, score_list, None, threshold, False,
                f'模型分相关性_{target}')
            logger.info(f"模型分相关性完成: {target}")
        except Exception as e:
            logger.error(f"模型分相关性失败 {target}: {e}")

    def save(self):
        """保存报告"""
        try:
            self.report.report_save()
            logger.info(f"报告保存成功: {self.workbook_name}")
        except Exception as e:
            logger.error(f"保存报告失败: {e}")
            raise


def _permutation_compute_worker(args):
    """Permutation Importance 多进程 worker（模块级函数，支持 pickle 序列化）"""
    idx, feat_name, X_np, y_arr, m_path, base_auc, miss_val = args
    original = X_np[:, idx].copy()
    m = None
    dtest = None
    try:
        X_np[:, idx] = np.random.permutation(original)
        m = xgb.Booster()
        m.load_model(m_path)
        dtest = xgb.DMatrix(X_np, missing=miss_val, nthread=1)
        pred = m.predict(dtest)
        new_auc = roc_auc_score(y_arr, pred)
        return {"feature": feat_name, "abl_auc": round(new_auc, 4),
                "abl_delta": round(new_auc - base_auc, 4), "index": idx}
    except Exception as e:
        return {"feature": feat_name, "error": str(e), "index": idx}
    finally:
        X_np[:, idx] = original
        for obj in [m, dtest]:
            if obj is not None:
                del obj
        gc.collect()


def _beamsearch_eval_mp(args):
    """BeamSearch 多进程 worker（模块级函数，支持 pickle 序列化）"""
    selected_set, feat, config = args
    try_features = list(selected_set) + [feat]
    try:
        feat_indices = [config['feature_to_idx'][f] for f in try_features]

        if config['use_memmap']:
            x_local = []
            for shm_name, shape, dtype in config['x_shm_meta']:
                mmap = np.memmap(shm_name, dtype=dtype, mode='r', shape=shape)
                x_local.append(mmap[:, feat_indices])
            y_local = []
            for shm_name, shape, dtype in config['y_shm_meta']:
                mmap = np.memmap(shm_name, dtype=dtype, mode='r', shape=shape)
                y_local.append(mmap)
        else:
            x_local = [x[:, feat_indices] for x in config['x_arrays']]
            y_local = list(config['y_arrays'])

        dtrain = xgb.DMatrix(x_local[0], label=y_local[0], missing=config['missing_value'])
        evals = [(dtrain, 'train')]
        for i in range(1, len(x_local)):
            doot = xgb.DMatrix(x_local[i], label=y_local[i], missing=config['missing_value'])
            evals.append((doot, f'oot_{i}'))

        booster = xgb.train(params=config['params'], dtrain=dtrain, num_boost_round=config['num_boost_round'],
                            evals=evals, early_stopping_rounds=config['early_stopping_rounds'], verbose_eval=False)
        best_iter = booster.best_iteration

        oot_aucs = []
        for i in range(1, len(x_local)):
            doot = xgb.DMatrix(x_local[i], label=y_local[i], missing=config['missing_value'])
            pred = booster.predict(doot, iteration_range=(0, best_iter + 1))
            oot_aucs.append(roc_auc_score(y_local[i], pred))

        combined_auc = np.dot(config['weights_list'], oot_aucs)
        del booster, dtrain
        gc.collect()
        return (try_features, *oot_aucs, combined_auc)
    except Exception as e:
        gc.collect()
        return None


# ==============================
# 单机版 BeamSearch 特征筛选
# ==============================
class BeamSearchFeatureSelector:
    """
    单机版 BeamSearch 特征筛选工具
    支持 CSV/Spark Hive 数据源、多 OOT 评估、加权 Beam Search、贝叶斯超参优化
    """

    def __init__(self, params=None, missing_value=-999.0):
        """
        :param params: XGBoost 参数字典
        :param missing_value: 缺失值填充
        """
        self.missing_value = missing_value
        self.params = params or {
            'booster': 'gbtree',
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'tree_method': 'hist',
            'max_bin': 512,
            'learning_rate': 0.05,
            'max_depth': 4,
            'min_child_weight': 80,
            'gamma': 5,
            'missing': missing_value,
            'subsample': 0.9,
            'colsample_bytree': 0.9,
            'scale_pos_weight': 1,
            'seed': 27,
            'n_jobs': -1,
            'verbosity': 0
        }

    @staticmethod
    def feature_importances_frame(clf):
        """获取特征重要性 DataFrame"""
        gain = clf.get_score(importance_type='gain')
        weight = clf.get_score(importance_type='weight')
        df_gain = pd.DataFrame(list(gain.items()), columns=['feature', 'imp_gain'])
        df_weight = pd.DataFrame(list(weight.items()), columns=['feature', 'imp_weight'])
        df = pd.merge(df_gain, df_weight, on='feature', how='outer').fillna(0)
        return df.sort_values(by='imp_gain', ascending=False)

    def load_data_csv(self, train_path, oot_paths, target, features=None, sep=','):
        """从 CSV 加载数据"""
        logger.info(f"加载训练数据: {train_path}")
        df_train = FeatureAnalysisToolkit.load_csv(train_path, sep=sep)
        if features is None:
            features = [c for c in df_train.columns if c != target]
        df_train = df_train[df_train[target].isin([0, 1])].copy()
        df_train[features] = df_train[features].fillna(self.missing_value)

        oot_list = []
        for i, path in enumerate(oot_paths):
            logger.info(f"加载 OOT{i+1}: {path}")
            df_oot = FeatureAnalysisToolkit.load_csv(path, sep=sep)
            df_oot = df_oot[df_oot[target].isin([0, 1])].copy()
            df_oot[features] = df_oot[features].fillna(self.missing_value)
            oot_list.append(df_oot)

        return df_train, oot_list, features

    def load_data_spark_hive(self, spark, train_table, oot_tables, target,
                             features=None, exclude_vars=None, where_clause_train=None,
                             where_clause_oot=None):
        """从 Spark Hive 表加载数据"""

        def _validate_table_name(name):
            if not re.match(r'^[a-zA-Z0-9_.]+$', name):
                raise ValueError(f"非法表名: {name}")

        _SQL_INJECTION_PATTERNS = re.compile(r';|--|/\*|\*/|xp_|exec\s', re.IGNORECASE)

        def read_table(table_name, where_clause):
            _validate_table_name(table_name)
            if where_clause and _SQL_INJECTION_PATTERNS.search(where_clause):
                raise ValueError(f"where_clause 包含潜在危险内容: {where_clause[:100]}")
            query = f"SELECT * FROM {table_name}"
            if where_clause:
                query += f" WHERE {where_clause}"
            return spark.sql(query).toPandas()

        logger.info(f"读取训练表: {train_table}")
        df_train = read_table(train_table, where_clause_train)
        df_train.columns = df_train.columns.str.lower()
        _exclude = exclude_vars.split(",") if isinstance(exclude_vars, str) else (exclude_vars or [])
        exclude_lower = [x.lower() for x in _exclude]
        if features is None:
            features = [c for c in df_train.columns if c.lower() != target.lower() and c not in exclude_lower]
        df_train = df_train[df_train[target].isin([0, 1])].copy()
        df_train[features] = df_train[features].fillna(self.missing_value)

        oot_list = []
        for i, table in enumerate(oot_tables):
            logger.info(f"读取 OOT{i+1}: {table}")
            df_oot = read_table(table, where_clause_oot)
            df_oot.columns = df_oot.columns.str.lower()
            df_oot = df_oot[df_oot[target].isin([0, 1])].copy()
            df_oot[features] = df_oot[features].fillna(self.missing_value)
            oot_list.append(df_oot)

        return df_train, oot_list, features

    def train_xgb(self, data, features, target, num_boost_round=1000, early_stopping_rounds=20,
                  verbose_eval=False, sample_weights=None):
        """
        训练 XGBoost 模型
        :param data: [train_df, oot1_df, oot2_df, ...]
        :return: (booster, importance_df, oot_auc_list)
        """
        train_df = data[0]
        oot_dfs = data[1:]

        dtrain = xgb.DMatrix(
            train_df[features].values, label=train_df[target].values,
            weight=sample_weights, feature_names=features, missing=self.missing_value)

        evals = [(dtrain, 'train')]
        doot_list = []
        for i, oot_df in enumerate(oot_dfs):
            doot = xgb.DMatrix(oot_df[features].values, label=oot_df[target].values,
                               feature_names=features, missing=self.missing_value)
            evals.append((doot, f'oot_{i}'))
            doot_list.append(doot)

        logger.info(f"训练 XGBoost: {len(features)} 个特征...")
        booster = xgb.train(
            params=self.params, dtrain=dtrain, num_boost_round=num_boost_round,
            evals=evals, early_stopping_rounds=early_stopping_rounds, verbose_eval=verbose_eval)

        best_iter = booster.best_iteration
        logger.info(f"最优迭代: {best_iter}")

        imp_df = self.feature_importances_frame(booster)

        oot_aucs = []
        for i, (oot_df, doot) in enumerate(zip(oot_dfs, doot_list)):
            pred = booster.predict(doot, iteration_range=(0, best_iter + 1))
            auc = roc_auc_score(oot_df[target], pred)
            oot_aucs.append(auc)
            logger.info(f"  OOT{i+1} AUC: {auc:.4f}")

        return booster, imp_df, oot_aucs

    def beam_search(self, data, initial_features, candidate_features, target,
                    weights_list, beam_width=3, patience=3, max_workers=4,
                    num_boost_round=1000, early_stopping_rounds=20,
                    use_memmap=True, memmap_dir='./beamsearch_memmap'):
        """
        Beam Search 特征筛选
        :param data: [train_df, oot1_df, oot2_df, ...]
        :param initial_features: 初始特征列表
        :param candidate_features: 候选特征列表
        :param target: 目标列名
        :param weights_list: 各OOT的AUC权重
        :param beam_width: beam宽度
        :param patience: 容忍轮数
        :param max_workers: 并行进程数
        :param use_memmap: 是否使用memmap共享内存（大数据量时推荐）
        :param memmap_dir: memmap临时目录
        :return: (final_model, final_importance, best_features, best_auc_list)
        """
        n_oot = len(data) - 1
        if len(weights_list) != n_oot:
            raise ValueError(f"weights_list长度({len(weights_list)})与OOT数据集数({n_oot})不匹配")
        best_combined = -1.0
        best_features = []
        best_auc_list = []
        no_improve = 0
        beam_list = [tuple(initial_features)]
        history_best = []

        logger.info(f"========== Beam Search 初始化 ==========")
        logger.info(f"初始特征: {initial_features}")
        logger.info(f"候选特征数: {len(candidate_features)}, 权重: {weights_list}, memmap: {use_memmap}")

        # 预提取所有特征列到 numpy 数组（避免反复 DataFrame 切片）
        all_features = sorted(set(initial_features) | set(candidate_features))
        x_arrays = []
        y_arrays = []
        for df in data:
            x_arrays.append(df[all_features].values.astype(np.float32))
            y_arrays.append(df[target].values.astype(np.float32))
        feature_to_idx = {f: i for i, f in enumerate(all_features)}

        if use_memmap:
            # 将数据写入 memmap 文件，子进程通过磁盘映射只读访问，避免内存复制
            if memmap_dir is None:
                memmap_dir = os.path.join(tempfile.gettempdir(), f'beamsearch_memmap_{os.getpid()}')
            FeatureAnalysisToolkit.ensure_dir(memmap_dir)

            x_shm_meta = []
            y_shm_meta = []
            for idx, (x_arr, y_arr) in enumerate(zip(x_arrays, y_arrays)):
                # X
                x_path = os.path.join(memmap_dir, f'x_{idx}.dat')
                x_mmap = np.memmap(x_path, dtype=x_arr.dtype, mode='w+', shape=x_arr.shape)
                x_mmap[:] = x_arr[:]
                x_mmap.flush()
                x_shm_meta.append((x_path, x_arr.shape, x_arr.dtype))
                # Y
                y_path = os.path.join(memmap_dir, f'y_{idx}.dat')
                y_mmap = np.memmap(y_path, dtype=y_arr.dtype, mode='w+', shape=y_arr.shape)
                y_mmap[:] = y_arr[:]
                y_mmap.flush()
                y_shm_meta.append((y_path, y_arr.shape, y_arr.dtype))

            data_size_mb = sum(x.nbytes + y.nbytes for x, y in zip(x_arrays, y_arrays)) / (1024 ** 2)
            logger.info(f"memmap 数据已写入: {memmap_dir}, 总大小: {data_size_mb:.1f}MB")

        # 模型参数序列化（子进程需要）
        params = self.params.copy()
        missing_value = self.missing_value

        def _train_and_eval(feature_indices):
            """单进程内：构建DMatrix、训练、评估"""
            feat_x_arrays = []
            for x_arr in x_arrays:
                feat_x_arrays.append(x_arr[:, feature_indices])
            dtrain = xgb.DMatrix(feat_x_arrays[0], label=y_arrays[0],
                                 feature_names=None, missing=missing_value)
            evals = [(dtrain, 'train')]
            doot_list = []
            for i in range(1, len(feat_x_arrays)):
                doot = xgb.DMatrix(feat_x_arrays[i], label=y_arrays[i],
                                   feature_names=None, missing=missing_value)
                evals.append((doot, f'oot_{i}'))
                doot_list.append(doot)

            booster = xgb.train(params=params, dtrain=dtrain, num_boost_round=num_boost_round,
                                evals=evals, early_stopping_rounds=early_stopping_rounds, verbose_eval=False)
            best_iter = booster.best_iteration
            oot_aucs = []
            for i, doot in enumerate(doot_list):
                pred = booster.predict(doot, iteration_range=(0, best_iter + 1))
                oot_aucs.append(roc_auc_score(y_arrays[i + 1], pred))
            return oot_aucs

        def _eval_combination(args):
            """单进程模式：直接在主进程评估"""
            selected_set, feat = args
            try_features = list(selected_set) + [feat]
            try:
                feat_indices = [feature_to_idx[f] for f in try_features]
                oot_auc_list = _train_and_eval(feat_indices)
                combined_auc = np.dot(weights_list, oot_auc_list)
                return (try_features, *oot_auc_list, combined_auc)
            except Exception as e:
                logger.warning(f"评估失败 {try_features}: {e}")
                return None

        # 多进程配置
        mp_config = {
            'params': params, 'missing_value': missing_value, 'weights_list': weights_list,
            'feature_to_idx': feature_to_idx, 'y_arrays': y_arrays,
            'num_boost_round': num_boost_round, 'early_stopping_rounds': early_stopping_rounds
        }

        if use_memmap:
            mp_config['x_shm_meta'] = x_shm_meta
            mp_config['y_shm_meta'] = y_shm_meta
            mp_config['use_memmap'] = True
        else:
            mp_config['x_arrays'] = x_arrays
            mp_config['use_memmap'] = False

        # 选择并行策略
        use_multiprocess = max_workers > 1
        pbar_main = tqdm(total=patience, desc="整体进度")
        iter_cnt = 1

        try:
          while no_improve < patience:
            logger.info(f"---------- 迭代 {iter_cnt} ----------")
            logger.info(f"当前 beam: {[list(b) for b in beam_list]}")

            # 生成候选
            candidates = []
            seen = set()
            for base in beam_list:
                for f in candidate_features:
                    if f in base:
                        continue
                    new_combo = tuple(sorted(list(base) + [f]))
                    if new_combo in seen:
                        continue
                    seen.add(new_combo)
                    candidates.append((base, f))

            logger.info(f"生成 {len(candidates)} 个候选组合")

            # 评估
            results = []
            if use_multiprocess:
                mp_args = [(base, f, mp_config) for base, f in candidates]
                with multiprocessing.Pool(processes=max_workers) as pool:
                    with tqdm(total=len(mp_args), desc=f"迭代{iter_cnt}评估") as pbar:
                        for res in pool.imap_unordered(_beamsearch_eval_mp, mp_args):
                            if res is not None:
                                results.append(res)
                            pbar.update(1)
            else:
                with tqdm(total=len(candidates), desc=f"迭代{iter_cnt}评估") as pbar:
                    for res in map(_eval_combination, candidates):
                        if res is not None:
                            results.append(res)
                        pbar.update(1)

            if not results:
                logger.warning("无有效候选，提前终止")
                break

            results.sort(key=lambda x: x[-1], reverse=True)
            top_k = results[:beam_width]
            beam_list = [tuple(r[0]) for r in top_k]
            current_best = top_k[0]

            if current_best[-1] > best_combined + 1e-4:
                best_combined = current_best[-1]
                best_features = list(current_best[0])
                best_auc_list = list(current_best[1:-1])
                history_best.append({
                    'features': best_features.copy(),
                    'aucs': best_auc_list.copy(),
                    'combined_auc': best_combined
                })
                no_improve = 0
                logger.info(f"新最优: {best_features}")
                logger.info(f"  加权AUC: {best_combined:.4f} | 各OOT AUC: {[round(a, 4) for a in best_auc_list]}")
            else:
                no_improve += 1
                pbar_main.update(1)
                logger.info(f"未改进 (连续 {no_improve}/{patience})")

            iter_cnt += 1
            gc.collect()

        finally:
            pbar_main.close()

            # 清理 memmap 文件
            if use_memmap:
                import shutil
                try:
                    shutil.rmtree(memmap_dir)
                    logger.info(f"memmap 临时目录已清理: {memmap_dir}")
                except Exception:
                    pass

        logger.info(f"========== 搜索结束 ==========")
        if not best_features:
            logger.warning("Beam Search 未找到优于初始特征集的组合，使用初始特征")
            best_features = list(initial_features)
            best_auc_list = []
        logger.info(f"最优特征 ({len(best_features)}): {sorted(best_features)}")
        logger.info(f"最终加权AUC: {best_combined:.4f}")

        # 用最优特征重训最终模型
        final_model, final_imp, final_aucs = self.train_xgb(
            data, list(best_features), target,
            num_boost_round=num_boost_round, early_stopping_rounds=early_stopping_rounds,
            verbose_eval=False)

        return final_model, final_imp, best_features, final_aucs


def _build_arg_parser():
    """构建命令行参数解析器"""
    parser = argparse.ArgumentParser(description='Auto Model Builder Tools')
    subparsers = parser.add_subparsers(dest='mode', help='运行模式')

    # ========== train ==========
    p_train = subparsers.add_parser('train', help='自动化建模')
    p_train.add_argument('--train_file', type=str, required=True, help='训练数据路径')
    p_train.add_argument('--test_file', type=str, help='测试数据路径')
    p_train.add_argument('--target_col', type=str, required=True, help='目标列名')
    p_train.add_argument('--model_type', type=str, default='lgb', choices=['lgb', 'xgb', 'lr'], help='模型类型')
    p_train.add_argument('--file_format', type=str, default='csv', choices=['csv', 'libsvm'], help='文件格式')
    p_train.add_argument('--tuning_method', type=str, default='bayesian', choices=['bayesian', 'grid'], help='调参方法')
    p_train.add_argument('--n_iter', type=int, default=50, help='贝叶斯优化迭代次数')
    p_train.add_argument('--output_dir', type=str, default='model_output', help='输出目录')
    p_train.add_argument('--model_path', type=str, default=None, help='模型保存路径（默认: output_dir/model.pkl）')

    # ========== attribution ==========
    p_attr = subparsers.add_parser('attribution', help='模型性能异动归因')
    p_attr.add_argument('--model_path', type=str, required=True, help='XGBoost模型文件路径(.pkl)')
    p_attr.add_argument('--features_path', type=str, required=True, help='特征列表文件路径(.pkl)')
    p_attr.add_argument('--data_path', type=str, required=True, help='分析数据文件路径')
    p_attr.add_argument('--data_sep', type=str, default='\t', help='数据分隔符')
    p_attr.add_argument('--target_col', type=str, required=True, help='目标变量列名')
    p_attr.add_argument('--time_col', type=str, required=True, help='时间列名')
    p_attr.add_argument('--baseline_month', type=str, required=True, help='基准月份')
    p_attr.add_argument('--current_month', type=str, required=True, help='当前月份')
    p_attr.add_argument('--info_vars', type=str, default='', help='排除列名，逗号分隔')
    p_attr.add_argument('--n_workers', type=int, default=1, help='并行进程数(1=单进程)')
    p_attr.add_argument('--output_dir', type=str, default='attribution_output', help='输出目录')

    # ========== report ==========
    p_rpt = subparsers.add_parser('report', help='模型报告生成')
    p_rpt.add_argument('--data_path', type=str, required=True, help='评估数据文件路径')
    p_rpt.add_argument('--data_sep', type=str, default='\t', help='数据分隔符')
    p_rpt.add_argument('--workbook_name', type=str, required=True, help='报告名称')
    p_rpt.add_argument('--reportbuilder_path', type=str, default=None, help='reportbuilderv4库路径')
    p_rpt.add_argument('--targets', type=str, required=True, help='目标变量列名，逗号分隔')
    p_rpt.add_argument('--score_list', type=str, required=True, help='模型分列名，逗号分隔')
    p_rpt.add_argument('--benchmark', type=str, required=True, help='基准模型分列名')
    p_rpt.add_argument('--column_mapping', type=str, default=None, help='列名映射，格式: old1:new1,old2:new2')
    p_rpt.add_argument('--stability_groups_json', type=str, default=None, help='稳定性客群配置JSON文件路径')
    p_rpt.add_argument('--customer_groups_json', type=str, default=None, help='Model Summary客群配置JSON文件路径')
    p_rpt.add_argument('--correlation_months', type=str, default=None, help='相关性计算月份，逗号分隔')
    p_rpt.add_argument('--stability_segvar', type=str, default='draw_month', help='稳定性分析时间变量')
    p_rpt.add_argument('--output_dir', type=str, default='report_output', help='输出目录')

    # ========== beamsearch ==========
    p_bs = subparsers.add_parser('beamsearch', help='BeamSearch特征筛选')
    p_bs.add_argument('--train_path', type=str, required=True, help='训练数据路径(CSV)')
    p_bs.add_argument('--oot_paths', type=str, required=True, help='OOT数据路径，逗号分隔')
    p_bs.add_argument('--target', type=str, required=True, help='目标列名')
    p_bs.add_argument('--initial_features', type=str, required=True, help='初始特征，逗号分隔')
    p_bs.add_argument('--weights', type=str, default='0.5,0.5', help='各OOT的AUC权重，逗号分隔')
    p_bs.add_argument('--beam_width', type=int, default=3, help='beam宽度')
    p_bs.add_argument('--patience', type=int, default=3, help='容忍轮数')
    p_bs.add_argument('--max_workers', type=int, default=4, help='并行线程数')
    p_bs.add_argument('--num_boost_round', type=int, default=1000, help='最大迭代轮数')
    p_bs.add_argument('--early_stopping_rounds', type=int, default=20, help='早停轮数')
    p_bs.add_argument('--output_dir', type=str, default='beamsearch_output', help='输出目录')

    # ========== feature_analysis ==========
    p_fa = subparsers.add_parser('feature_analysis', help='特征分析（IV/PSI/KS/EDA/覆盖率）')
    p_fa.add_argument('--data_path', type=str, required=True, help='数据文件路径')
    p_fa.add_argument('--data_sep', type=str, default='\t', help='数据分隔符')
    p_fa.add_argument('--target_col', type=str, required=True, help='目标变量列名')
    p_fa.add_argument('--exclude_vars', type=str, default='', help='排除列名，逗号分隔')
    p_fa.add_argument('--psi_expected_col', type=str, default=None, help='PSI基准列名（如时间列）')
    p_fa.add_argument('--psi_expected_val', type=str, default=None, help='PSI基准值（基准期值）')
    p_fa.add_argument('--psi_actual_val', type=str, default=None, help='PSI对比值（对比期值，逗号分隔）')
    p_fa.add_argument('--psi_bins', type=int, default=10, help='PSI分箱数')
    p_fa.add_argument('--compute_iv', action='store_true', help='是否计算IV')
    p_fa.add_argument('--compute_auc', action='store_true', help='是否计算单特征AUC')
    p_fa.add_argument('--compute_ks', action='store_true', help='是否计算KS')
    p_fa.add_argument('--compute_coverage', action='store_true', help='是否计算覆盖率')
    p_fa.add_argument('--full_eda', action='store_true', help='完整EDA（覆盖率+IV+PSI）')
    p_fa.add_argument('--n_workers', type=int, default=1, help='并行进程数(1=单进程)')
    p_fa.add_argument('--output_dir', type=str, default='feature_analysis_output', help='输出目录')

    # ========== feature_analysis_report ==========
    p_far = subparsers.add_parser('feature_analysis_report', help='特征分析报告（Excel多Sheet：覆盖率/PSI/IV/WOE/AUC）')
    p_far.add_argument('--data_path', type=str, required=True, help='数据文件路径')
    p_far.add_argument('--data_sep', type=str, default='\t', help='数据分隔符')
    p_far.add_argument('--target_col', type=str, required=True, help='目标变量列名')
    p_far.add_argument('--group_col', type=str, required=True, help='分组列名（如draw_month）')
    p_far.add_argument('--base_group_value', type=str, required=True, help='PSI基准期值')
    p_far.add_argument('--features', type=str, default=None, help='特征列名，逗号分隔（不填则自动检测）')
    p_far.add_argument('--exclude_vars', type=str, default='', help='排除列名，逗号分隔')
    p_far.add_argument('--special_values', type=str, default=None, help='特殊值（视同缺失值），逗号分隔，如 -999,-9999')
    p_far.add_argument('--woe_binning_method', type=str, default='quantile',
        choices=['quantile', 'equal_width', 'bestks'], help='WOE分箱方法')
    p_far.add_argument('--woe_bins', type=int, default=10, help='WOE分箱数')
    p_far.add_argument('--compute_woe', action='store_true', help='是否计算WOE明细')
    p_far.add_argument('--n_workers', type=int, default=1, help='并行进程数(1=单进程)')
    p_far.add_argument('--output_path', type=str, default='feature_analysis_report.xlsx', help='输出Excel文件路径')

    # ========== scoring ==========
    p_sc = subparsers.add_parser('scoring', help='模型批量打分')
    p_sc.add_argument('--model_path', type=str, required=True, help='模型文件路径(.pkl或.json)')
    p_sc.add_argument('--features_path', type=str, required=True, help='特征列表文件路径(.pkl)')
    p_sc.add_argument('--data_path', type=str, required=True, help='打分数据文件路径')
    p_sc.add_argument('--data_sep', type=str, default='\t', help='数据分隔符')
    p_sc.add_argument('--target_col', type=str, default=None, help='目标列名（不提供则只打分不评估）')
    p_sc.add_argument('--missing_value', type=float, default=-999.0, help='缺失值填充')
    p_sc.add_argument('--ntree_limit', type=int, default=0, help='使用前N棵树（0=使用最优迭代）')
    p_sc.add_argument('--score_name', type=str, default='model_score', help='打分列名')
    p_sc.add_argument('--chunk_size', type=int, default=500000, help='分批读取行数（0=一次性读取）')
    p_sc.add_argument('--output_path', type=str, default=None, help='输出文件路径')

    # ========== model_evaluation ==========
    p_me = subparsers.add_parser('model_evaluation', help='模型分评估（AUC/KS/Lift）')
    p_me.add_argument('--data_path', type=str, required=True, help='评估数据文件路径')
    p_me.add_argument('--data_sep', type=str, default='\t', help='数据分隔符')
    p_me.add_argument('--target_col', type=str, required=True, help='目标变量列名')
    p_me.add_argument('--score_cols', type=str, required=True, help='模型分列名，逗号分隔')
    p_me.add_argument('--score_names', type=str, default=None, help='模型分中文名，逗号分隔')
    p_me.add_argument('--time_col', type=str, default=None, help='时间列名（按时间切片评估）')
    p_me.add_argument('--group_cols', type=str, default=None, help='客群分组列名，逗号分隔')
    p_me.add_argument('--metrics', type=str, default='auc,ks', help='评估指标，逗号分隔: auc,ks')
    p_me.add_argument('--include_lift', action='store_true', help='是否计算Lift')
    p_me.add_argument('--lift_groups', type=int, default=10, help='Lift分组数')
    p_me.add_argument('--fill_score_value', type=float, default=None, help='模型分缺失填充值')
    p_me.add_argument('--output_dir', type=str, default='model_eval_output', help='输出目录')

    return parser


# ========================
# 各模式的 handler 函数
# ========================

def _handle_train(args):
    """train 模式：自动化建模"""
    start_time = time.time()
    FeatureAnalysisToolkit.ensure_dir(args.output_dir)

    builder = AutoModelBuilder(model_type=args.model_type)
    X_train, y_train = builder.load_data(args.train_file, args.target_col, file_format=args.file_format)

    try:
        builder.eda_analysis(X_train, y_train, output_dir=os.path.join(args.output_dir, 'eda'))
    except Exception as e:
        logger.warning(f"EDA分析失败，跳过: {e}")

    best_params = builder.hyperparameter_tuning(X_train, y_train, tuning_method=args.tuning_method, n_iter=args.n_iter)

    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=builder.random_state)
    builder.train(X_tr, y_tr, params=best_params, eval_set=[(X_val, y_val)], early_stopping_rounds=50)

    # 评估
    train_eval = FeatureAnalysisToolkit.calculate_auc_ks(y_train, builder.predict(X_train))
    logger.info(f"训练集评估: AUC={train_eval['auc']}, KS={train_eval['ks']}")
    if np.isnan(train_eval['auc']):
        logger.warning("训练集 AUC 为 NaN，请检查数据质量和标签分布")

    val_eval = FeatureAnalysisToolkit.calculate_auc_ks(y_val, builder.predict(X_val))
    logger.info(f"验证集评估: AUC={val_eval['auc']}, KS={val_eval['ks']}")

    eval_records = [
        {'dataset': 'train', 'auc': train_eval['auc'], 'ks': train_eval['ks'],
         'sample_count': len(y_train), 'y_count': int(y_train.sum())},
        {'dataset': 'validation', 'auc': val_eval['auc'], 'ks': val_eval['ks'],
         'sample_count': len(y_val), 'y_count': int(y_val.sum())}
    ]

    model_save_path = args.model_path or os.path.join(args.output_dir, 'model.pkl')
    builder.save_model(model_save_path)
    builder.get_feature_importance(output_file=os.path.join(args.output_dir, 'feature_importance.csv'))

    try:
        builder.shap_analysis(X_train, output_dir=os.path.join(args.output_dir, 'shap'),
                              use_pred_contribs=False)
    except Exception as e:
        logger.warning(f"SHAP分析失败，跳过: {e}")

    if args.test_file:
        X_test, y_test = builder.load_data(args.test_file, args.target_col, file_format=args.file_format)
        X_test = X_test[builder.features]
        predictions = builder.predict(X_test)
        test_eval = FeatureAnalysisToolkit.calculate_auc_ks(y_test, predictions)
        logger.info(f"测试集评估: AUC={test_eval['auc']}, KS={test_eval['ks']}")
        eval_records.append({
            'dataset': 'test', 'auc': test_eval['auc'], 'ks': test_eval['ks'],
            'sample_count': len(y_test), 'y_count': int(y_test.sum())})
        test_results = X_test.copy()
        test_results['true_label'] = y_test
        test_results['prediction'] = predictions
        test_results.to_csv(os.path.join(args.output_dir, 'test_predictions.csv'), index=False)

    pd.DataFrame(eval_records).to_csv(os.path.join(args.output_dir, 'model_evaluation.csv'), index=False)

    train_config = {
        'model_type': args.model_type, 'tuning_method': args.tuning_method,
        'best_params': best_params, 'features': builder.features, 'target': builder.target,
        'train_file': args.train_file, 'test_file': args.test_file,
        'train_samples': len(y_train), 'random_state': builder.random_state
    }
    with open(os.path.join(args.output_dir, 'train_config.json'), 'w', encoding='utf-8') as f:
        json.dump(train_config, f, ensure_ascii=False, indent=2, default=str)

    elapsed = time.time() - start_time
    logger.info(f"Train mode completed successfully in {elapsed:.1f}s!")


def _handle_attribution(args):
    """attribution 模式：模型性能异动归因"""
    FeatureAnalysisToolkit.ensure_dir(args.output_dir)

    analyzer = ModelAttributionAnalyzer(
        model_path=args.model_path, features_path=args.features_path)

    df = FeatureAnalysisToolkit.load_csv(args.data_path, sep=args.data_sep)
    info_vars = FeatureAnalysisToolkit.parse_csv_list(args.info_vars)

    analyzer.full_attribution(
        df=df, time_col=args.time_col, target_col=args.target_col,
        baseline_month=args.baseline_month, current_month=args.current_month,
        info_vars=info_vars, n_workers=args.n_workers, output_dir=args.output_dir)

    logger.info("Attribution mode completed successfully!")


def _handle_report(args):
    """report 模式：模型报告生成"""
    FeatureAnalysisToolkit.ensure_dir(args.output_dir)

    targets = FeatureAnalysisToolkit.parse_csv_list(args.targets)
    score_list = FeatureAnalysisToolkit.parse_csv_list(args.score_list)

    column_mapping = None
    if args.column_mapping:
        column_mapping = {}
        for pair in args.column_mapping.split(','):
            if ':' in pair:
                k, v = pair.split(':', 1)
                column_mapping[k.strip()] = v.strip()

    report_gen = ModelReportGenerator(
        workbook_name=args.workbook_name, reportbuilder_path=args.reportbuilder_path)

    df, df_targets = ModelReportGenerator.load_data(
        file_path=args.data_path, separator=args.data_sep,
        column_mapping=column_mapping, score_list=score_list, targets=targets)

    # Model Stability（JSON 加载失败则跳过该模块而非终止整个流程）
    if args.stability_groups_json:
        config = FeatureAnalysisToolkit.load_json_config(args.stability_groups_json)
        if config is not None:
            for target in targets:
                stability_groups = ModelReportGenerator.create_groups(
                    df_targets[target], config, is_nested=False)
                report_gen.generate_model_stability(
                    stability_groups, target, score_list, segvar=args.stability_segvar)

    # Model Summary
    if args.customer_groups_json:
        config = FeatureAnalysisToolkit.load_json_config(args.customer_groups_json)
        if config is not None:
            for target in targets:
                customer_groups = ModelReportGenerator.create_groups(
                    df_targets[target], config, is_nested=True)
                report_gen.generate_model_summary(
                    customer_groups, target, args.benchmark, score_list)

    # 相关性报告
    if args.correlation_months:
        corr_months = FeatureAnalysisToolkit.parse_csv_list(args.correlation_months)
        for target in targets:
            report_gen.generate_correlation(df, target, score_list, corr_months)

    report_gen.save()
    logger.info("Report mode completed successfully!")


def _handle_beamsearch(args):
    """beamsearch 模式：BeamSearch特征筛选"""
    FeatureAnalysisToolkit.ensure_dir(args.output_dir)

    oot_paths = FeatureAnalysisToolkit.parse_csv_list(args.oot_paths)
    weights_list = [float(w.strip()) for w in args.weights.split(',') if w.strip()]
    initial_features = FeatureAnalysisToolkit.parse_csv_list(args.initial_features)

    selector = BeamSearchFeatureSelector()
    df_train, oot_list, all_features = selector.load_data_csv(
        train_path=args.train_path, oot_paths=oot_paths, target=args.target)

    data = [df_train] + oot_list
    candidate_features = [f for f in all_features if f not in initial_features]

    final_model, final_imp, best_features, best_aucs = selector.beam_search(
        data=data, initial_features=initial_features, candidate_features=candidate_features,
        target=args.target, weights_list=weights_list, beam_width=args.beam_width,
        patience=args.patience, max_workers=args.max_workers,
        num_boost_round=args.num_boost_round, early_stopping_rounds=args.early_stopping_rounds)

    joblib.dump(final_model, os.path.join(args.output_dir, 'beamsearch_final_model.pkl'))
    joblib.dump(best_features, os.path.join(args.output_dir, 'beamsearch_best_features.pkl'))
    final_imp.to_csv(os.path.join(args.output_dir, 'beamsearch_feature_importance.csv'), index=False)

    pd.DataFrame([{
        'best_features': str(best_features),
        'best_aucs': str([round(a, 4) for a in best_aucs]),
        'combined_auc': round(np.dot(weights_list, best_aucs), 4),
        'feature_count': len(best_features)
    }]).to_csv(os.path.join(args.output_dir, 'beamsearch_summary.csv'), index=False)

    logger.info(f"BeamSearch mode completed! 最佳特征数: {len(best_features)}")


def _handle_feature_analysis(args):
    """feature_analysis 模式：特征分析（IV/PSI/KS/AUC/覆盖率/EDA）"""
    FeatureAnalysisToolkit.ensure_dir(args.output_dir)

    df = FeatureAnalysisToolkit.load_csv(args.data_path, sep=args.data_sep)
    target = args.target_col.lower()
    exclude = FeatureAnalysisToolkit.parse_csv_list(args.exclude_vars, lower=True)
    features = [c for c in df.columns if c != target and c not in exclude]
    n_workers = args.n_workers
    logger.info(f"数据量: {len(df)}, 特征数: {len(features)}, 并行: {n_workers}")

    results = {}

    # 覆盖率
    if args.compute_coverage or args.full_eda:
        coverage = (1 - df[features].isnull().sum() / len(df))
        results['coverage'] = pd.DataFrame({'feature': coverage.index, 'coverage': coverage.values})
        results['coverage'].to_csv(os.path.join(args.output_dir, 'coverage.csv'), index=False)
        logger.info("覆盖率计算完成")

    # IV（多进程安全：使用模块级 worker 函数）
    if args.compute_iv or args.full_eda:
        if n_workers > 1:
            worker_args = [(feat, df, target) for feat in features]
            with multiprocessing.Pool(processes=n_workers) as pool:
                iv_list = list(tqdm(pool.imap_unordered(_fa_iv_worker, worker_args),
                                    total=len(features), desc="IV计算"))
        else:
            iv_list = [_fa_iv_worker((f, df, target)) for f in tqdm(features, desc="IV计算")]
        results['iv'] = pd.DataFrame(iv_list).sort_values('iv', ascending=False)
        results['iv'].to_csv(os.path.join(args.output_dir, 'iv.csv'), index=False)
        logger.info("IV计算完成")

    # PSI
    if args.psi_expected_col and args.psi_expected_val:
        col_dtype = df[args.psi_expected_col].dtype
        exp_val = FeatureAnalysisToolkit.coerce_time_value(col_dtype, args.psi_expected_val)
        act_vals = FeatureAnalysisToolkit.parse_csv_list(args.psi_actual_val)
        act_vals = [FeatureAnalysisToolkit.coerce_time_value(col_dtype, v) for v in act_vals]
        expected_mask = df[args.psi_expected_col] == exp_val
        psi_records = []
        for act_val in act_vals:
            actual_mask = df[args.psi_expected_col] == act_val
            if n_workers > 1:
                psi_worker_args = [(feat, df.loc[expected_mask, feat], df.loc[actual_mask, feat],
                                    args.psi_bins, exp_val, act_val) for feat in features]
                with multiprocessing.Pool(processes=n_workers) as pool:
                    psi_records.extend(list(tqdm(pool.imap_unordered(_fa_psi_worker, psi_worker_args),
                                                 total=len(features), desc=f"PSI({act_val})")))
            else:
                for feat in features:
                    psi = FeatureAnalysisToolkit.calculate_psi(
                        df.loc[expected_mask, feat], df.loc[actual_mask, feat], bins=args.psi_bins)
                    psi_records.append({
                        'feature': feat, 'psi': psi,
                        'baseline': exp_val, 'compare': act_val,
                        'stability': ('Stable' if pd.notna(psi) and psi < 0.1
                                      else ('Slightly unstable' if pd.notna(psi) and psi < 0.25
                                            else 'Unstable'))
                    })
        results['psi'] = pd.DataFrame(psi_records)
        results['psi'].to_csv(os.path.join(args.output_dir, 'psi.csv'), index=False)
        logger.info(f"PSI计算完成: {len(act_vals)} 个对比期")

    # 单特征AUC
    if args.compute_auc:
        valid_mask = df[target].isin([0, 1])
        y = df.loc[valid_mask, target].values
        if n_workers > 1:
            worker_args = [(feat, df.loc[valid_mask], y) for feat in features]
            with multiprocessing.Pool(processes=n_workers) as pool:
                auc_records = list(tqdm(pool.imap_unordered(_fa_auc_worker, worker_args),
                                        total=len(features), desc="AUC计算"))
        else:
            auc_records = [_fa_auc_worker((f, df.loc[valid_mask], y)) for f in tqdm(features, desc="AUC计算")]
        results['single_auc'] = pd.DataFrame(auc_records).sort_values('auc', ascending=False)
        results['single_auc'].to_csv(os.path.join(args.output_dir, 'single_auc.csv'), index=False)
        logger.info("单特征AUC计算完成")

    # KS
    if args.compute_ks:
        valid_mask = df[target].isin([0, 1])
        y = df.loc[valid_mask, target].values
        if n_workers > 1:
            worker_args = [(feat, df.loc[valid_mask], y) for feat in features]
            with multiprocessing.Pool(processes=n_workers) as pool:
                ks_records = list(tqdm(pool.imap_unordered(_fa_ks_worker, worker_args),
                                       total=len(features), desc="KS计算"))
        else:
            ks_records = [_fa_ks_worker((f, df.loc[valid_mask], y)) for f in tqdm(features, desc="KS计算")]
        results['ks'] = pd.DataFrame(ks_records).sort_values('ks', ascending=False)
        results['ks'].to_csv(os.path.join(args.output_dir, 'ks.csv'), index=False)
        logger.info("KS计算完成")

    # 完整EDA
    if args.full_eda:
        valid_mask = df[target].isin([0, 1])
        eda_result = FeatureAnalysisToolkit.eda_analysis(
            df.loc[valid_mask, features], df.loc[valid_mask, target],
            output_dir=os.path.join(args.output_dir, 'eda'), random_state=42)
        results['eda'] = eda_result

    # 合并汇总表
    merge_keys = ['coverage', 'iv', 'single_auc', 'ks']
    merge_dfs = [results[k] for k in merge_keys if k in results and isinstance(results[k], pd.DataFrame)]
    if merge_dfs:
        merged = reduce(lambda l, r: pd.merge(l, r, on='feature', how='outer'), merge_dfs)
        merged.to_csv(os.path.join(args.output_dir, 'feature_analysis_summary.csv'), index=False)
        logger.info(f"特征分析汇总已保存: {len(merged)} 个特征")

    logger.info("Feature analysis mode completed successfully!")


def _handle_feature_analysis_report(args):
    """feature_analysis_report 模式：特征分析报告（Excel多Sheet）"""
    logger.info("特征分析报告模式")
    df = FeatureAnalysisToolkit.load_csv(args.data_path, sep=args.data_sep)

    feat_list = FeatureAnalysisToolkit.parse_csv_list(args.features, lower=True) if args.features else None
    exclude = FeatureAnalysisToolkit.parse_csv_list(args.exclude_vars, lower=True)
    sv = [float(v.strip()) for v in args.special_values.split(',') if v.strip()] if args.special_values else None

    FeatureAnalysisToolkit.feature_analysis_report(
        df=df, target=args.target_col.lower(), group_col=args.group_col.lower(),
        base_group_value=args.base_group_value,
        features=feat_list, exclude_vars=exclude, special_values=sv,
        woe_binning_method=args.woe_binning_method, woe_bins=args.woe_bins,
        compute_woe=args.compute_woe, n_workers=args.n_workers, output_path=args.output_path)
    logger.info("Feature analysis report mode completed successfully!")


def _handle_scoring(args):
    """scoring 模式：模型批量打分"""
    logger.info(f"加载模型: {args.model_path}")
    booster, best_iter = AutoModelBuilder.load_xgb_booster(args.model_path)

    features = joblib.load(args.features_path)
    if not isinstance(features, list):
        raise ValueError(f"特征文件格式错误，期望list，实际为{type(features)}")
    logger.info(f"特征数: {len(features)}, 最优迭代: {best_iter}")

    ntree_limit = args.ntree_limit if args.ntree_limit > 0 else (best_iter + 1 if best_iter > 0 else 0)
    features_lower = [f.lower() for f in features]
    output_path = args.output_path or args.data_path.replace('.csv', '_scored.csv')
    chunk_size = args.chunk_size
    target = args.target_col.lower() if args.target_col else None

    y_all, score_all = [], []
    logger.info(f"开始分批打分: {args.data_path}, chunk_size={chunk_size}")

    total_scored = 0
    first_chunk = True

    for chunk_idx, chunk in enumerate(pd.read_csv(args.data_path, sep=args.data_sep, chunksize=chunk_size)):
        chunk.columns = chunk.columns.str.lower()

        if first_chunk:
            missing_cols = [f for f in features_lower if f not in chunk.columns]
            if missing_cols:
                raise ValueError(f"数据中缺少特征列: {missing_cols[:20]}")
            first_chunk = False

        chunk[features_lower] = chunk[features_lower].fillna(args.missing_value)
        X = chunk[features_lower].values

        dmatrix = xgb.DMatrix(X, feature_names=features_lower, missing=args.missing_value)
        if ntree_limit > 0:
            scores = booster.predict(dmatrix, iteration_range=(0, ntree_limit))
        else:
            scores = booster.predict(dmatrix)

        chunk[args.score_name] = scores
        total_scored += len(chunk)

        if target and target in chunk.columns:
            valid_mask = chunk[target].isin([0, 1])
            if valid_mask.sum() > 0:
                y_all.append(chunk.loc[valid_mask, target].values)
                score_all.append(chunk.loc[valid_mask, args.score_name].values)

        chunk.to_csv(output_path, sep='\t', index=False, mode='w' if chunk_idx == 0 else 'a', header=(chunk_idx == 0))

        if chunk_idx % 10 == 0:
            logger.info(f"已打分: {total_scored:,} 行")

        del dmatrix, X

    gc.collect()

    if y_all:
        y_all = np.concatenate(y_all)
        score_all = np.concatenate(score_all)
        result = FeatureAnalysisToolkit.calculate_auc_ks(y_all, score_all)
        logger.info(f"整体评估: AUC={result['auc']}, KS={result['ks']}, 有效样本={len(y_all):,}")

    logger.info(f"打分完成: 共 {total_scored:,} 条, 结果已保存: {output_path}")


def _handle_model_evaluation(args):
    """model_evaluation 模式：模型分评估（AUC/KS/Lift）"""
    FeatureAnalysisToolkit.ensure_dir(args.output_dir)

    df = FeatureAnalysisToolkit.load_csv(args.data_path, sep=args.data_sep)

    score_cols = FeatureAnalysisToolkit.parse_csv_list(args.score_cols, lower=True)
    score_names = FeatureAnalysisToolkit.parse_csv_list(args.score_names) if args.score_names else score_cols
    target = args.target_col.lower()
    metrics = FeatureAnalysisToolkit.parse_csv_list(args.metrics, lower=True)
    group_cols = FeatureAnalysisToolkit.parse_csv_list(args.group_cols, lower=True)

    logger.info(f"评估数据: {len(df)} 行, 模型分列: {score_cols}, 目标: {target}")

    all_results = []
    time_values = df[args.time_col].unique().tolist() if args.time_col else [None]

    # 构建客群组合（带上限保护）
    MAX_GROUP_COMBOS = 1000
    if group_cols:
        group_combos = [{}]
        for gcol in group_cols:
            new_combos = []
            for combo in group_combos:
                for gval in df[gcol].unique():
                    new_combos.append({**combo, gcol: gval})
            group_combos = new_combos
            if len(group_combos) > MAX_GROUP_COMBOS:
                logger.warning(f"客群组合数超过 {MAX_GROUP_COMBOS}，截断")
                group_combos = group_combos[:MAX_GROUP_COMBOS]
                break
    else:
        group_combos = [{}]

    for time_val in time_values:
        for group_dict in group_combos:
            mask = pd.Series(True, index=df.index)
            seg_desc_parts = []

            if time_val is not None:
                mask &= (df[args.time_col] == time_val)
                seg_desc_parts.append(f"{args.time_col}={time_val}")

            for gcol, gval in group_dict.items():
                mask &= (df[gcol] == gval)
                seg_desc_parts.append(f"{gcol}={gval}")

            sub_df = df[mask]
            if len(sub_df) == 0:
                continue

            valid = sub_df[target].isin([0, 1])
            for score_col, score_name in zip(score_cols, score_names):
                eval_df = sub_df.loc[valid]
                if args.fill_score_value is not None:
                    eval_df = eval_df.copy()
                    eval_df[score_col] = eval_df[score_col].fillna(args.fill_score_value)
                else:
                    eval_df = eval_df[eval_df[score_col].notna()]

                row = {
                    'segment': ' | '.join(seg_desc_parts) if seg_desc_parts else '全部',
                    'sample_count': len(eval_df),
                    'y_count': int(eval_df[target].sum()),
                    'y_rate': round(eval_df[target].mean(), 4),
                    'score_col': score_col,
                    'score_name': score_name,
                }

                if args.time_col:
                    row[args.time_col] = time_val
                for gcol, gval in group_dict.items():
                    row[gcol] = gval

                if len(eval_df) > 0 and eval_df[target].nunique() >= 2:
                    if 'auc' in metrics:
                        row['auc'] = FeatureAnalysisToolkit.calculate_single_auc(
                            eval_df[target].values, eval_df[score_col].values)
                    if 'ks' in metrics:
                        row['ks'] = FeatureAnalysisToolkit.calculate_ks(
                            eval_df[target].values, eval_df[score_col].values)
                else:
                    row['auc'] = None
                    row['ks'] = None

                all_results.append(row)

                # Lift（从高分到低分排列）
                if args.include_lift and len(eval_df) > 0 and eval_df[target].nunique() >= 2:
                    try:
                        y_rate = eval_df[target].mean()
                        eval_sorted = eval_df.sort_values(score_col, ascending=False).reset_index(drop=True)
                        eval_sorted['group'] = pd.qcut(eval_sorted[score_col], q=args.lift_groups, duplicates='drop')
                        lift_df = eval_sorted.groupby('group', observed=False).agg(
                            sample_count=(target, 'count'),
                            y_count=(target, 'sum')
                        ).reset_index()
                        lift_df['y_rate'] = lift_df['y_count'] / lift_df['sample_count']
                        lift_df['lift'] = lift_df['y_rate'] / y_rate if y_rate > 0 else 0
                        cum_y = lift_df['y_count'].cumsum()
                        cum_n = lift_df['sample_count'].cumsum()
                        lift_df['cumlift'] = (cum_y / cum_n) / y_rate if y_rate > 0 else 0
                        lift_df['score_col'] = score_col
                        lift_df['score_name'] = score_name
                        lift_df['segment'] = row['segment']
                        if args.time_col:
                            lift_df[args.time_col] = time_val
                        for gcol, gval in group_dict.items():
                            lift_df[gcol] = gval
                        safe_seg_name = re.sub(r'[^\w\-.]', '_', row["segment"])
                        lift_path = os.path.join(args.output_dir, f'lift_{score_col}_{safe_seg_name}.csv')
                        lift_df.to_csv(lift_path, index=False)
                    except Exception as e:
                        logger.warning(f"Lift计算失败 {score_col}: {e}")

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(args.output_dir, 'model_evaluation_result.csv'), index=False)
    logger.info(f"模型分评估完成: {len(results_df)} 条评估记录")


# ========================
# feature_analysis 多进程 worker 函数（模块级，支持 pickle）
# ========================

def _fa_psi_worker(args):
    """PSI 计算 worker"""
    feat, expected, actual, bins, baseline, compare = args
    psi = FeatureAnalysisToolkit.calculate_psi(expected, actual, bins=bins)
    return {
        'feature': feat, 'psi': psi,
        'baseline': baseline, 'compare': compare,
        'stability': ('Stable' if pd.notna(psi) and psi < 0.1
                      else ('Slightly unstable' if pd.notna(psi) and psi < 0.25
                            else 'Unstable'))
    }


def _fa_iv_worker(args):
    """IV 计算 worker"""
    feat, df, target = args
    return {'feature': feat, 'iv': FeatureAnalysisToolkit.calculate_single_iv(df, feat, target)}


def _fa_auc_worker(args):
    """单特征 AUC 计算 worker"""
    feat, df_subset, y = args
    feat_vals = df_subset[feat].values
    nan_mask = ~np.isnan(feat_vals)
    if nan_mask.sum() > 0 and y[nan_mask].sum() > 0 and (1 - y[nan_mask]).sum() > 0:
        return {'feature': feat, 'auc': FeatureAnalysisToolkit.calculate_single_auc(y[nan_mask], feat_vals[nan_mask])}
    return {'feature': feat, 'auc': np.nan}


def _fa_ks_worker(args):
    """KS 计算 worker"""
    feat, df_subset, y = args
    feat_vals = df_subset[feat].values
    nan_mask = ~np.isnan(feat_vals)
    if nan_mask.sum() > 0 and y[nan_mask].sum() > 0 and (1 - y[nan_mask]).sum() > 0:
        try:
            return {'feature': feat, 'ks': FeatureAnalysisToolkit.calculate_ks(y[nan_mask], feat_vals[nan_mask])}
        except Exception:
            return {'feature': feat, 'ks': np.nan}
    return {'feature': feat, 'ks': np.nan}


# ========================
# 主入口
# ========================

_MODE_HANDLERS = {
    'train': _handle_train,
    'attribution': _handle_attribution,
    'report': _handle_report,
    'beamsearch': _handle_beamsearch,
    'feature_analysis': _handle_feature_analysis,
    'feature_analysis_report': _handle_feature_analysis_report,
    'scoring': _handle_scoring,
    'model_evaluation': _handle_model_evaluation,
}


def main():
    """主函数：解析参数，分发到对应模式 handler"""
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.mode is None:
        parser.print_help()
        return

    handler = _MODE_HANDLERS.get(args.mode)
    if handler:
        try:
            handler(args)
        except Exception as e:
            logger.error(f"[{args.mode}] 执行失败: {e}", exc_info=True)
            raise
    else:
        parser.print_help()


if __name__ == '__main__':
    main()