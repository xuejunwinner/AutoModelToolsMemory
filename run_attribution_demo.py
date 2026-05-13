# -*- coding: utf-8 -*-
"""
模拟运行 full_attribution 并输出归因报告
使用 test_workspace 中已有的测试数据和模型
"""
import sys
import os
import warnings
import multiprocessing
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_workspace'))

from AutoModelBuilderTools import ModelAttributionAnalyzer, FeatureAnalysisToolkit
import pandas as pd
import numpy as np

def main():
    # 1. 加载测试数据
    print("=" * 60)
    print("加载测试数据...")
    df = pd.read_csv('test_data.csv', sep='\t')
    df.columns = df.columns.str.lower()
    target = 'dpd30_term1'
    info_vars = ['appl_no', 'draw_month', 'flag_cg_yz', 'cust_status3', 'dpd30_term1', 'dpd30_term3']

    print(f"  数据量: {len(df)}")
    print(f"  月份: {sorted(df['draw_month'].unique())}")
    print(f"  目标分布: {df[target].value_counts().to_dict()}")

    # 2. 初始化归因分析器
    print("\n初始化归因分析器...")
    analyzer = ModelAttributionAnalyzer(
        'test_output/test_booster.pkl',
        'test_output/test_features.pkl',
        missing_value=-999.0
    )
    print(f"  模型特征数: {len(analyzer.features)}")
    print(f"  特征列表: {analyzer.features}")

    # 3. 运行完整归因分析并输出报告
    output_dir = 'test_output/attribution_report'
    print(f"\n运行 full_attribution，输出目录: {output_dir}")
    print("-" * 60)

    df_stat, df_abl, summary = analyzer.full_attribution(
        df, 'draw_month', target, '202511', '202601',
        info_vars=info_vars, n_workers=1, n_repeats=5,
        top_n=15, enable_pair_ablation=True, pair_top_n=10,
        current_months=['202512', '202601', '202602'],
        thresholds={'auc_drop_warn': 0.003},
        output_dir=output_dir
    )

# 4. 打印结果
    print("\n" + "=" * 60)
    print("归因分析结果")
    print("=" * 60)

    print(f"\n--- 汇总 ---")
    print(f"  AUC: {summary['auc_old']} -> {summary['auc_new']} (drop={summary['auc_drop']})")
    print(f"  KS:  {summary['ks_old']} -> {summary['ks_new']} (drop={summary['ks_drop']})")
    print(f"  Score PSI: {summary['score_psi']} ({summary['score_stability']})")
    print(f"  分析特征数: {summary['total_features_analyzed']}")

    print(f"\n--- 归因结论 ---")
    print(f"  {summary.get('conclusion', '')}")

    print(f"\n--- 建议动作 ---")
    for action, reason in summary.get('actions', []):
        print(f"  * {action}: {reason}")

    print(f"\n--- 校准检测 ---")
    hl = summary.get('hl_stat', {})
    print(f"  HL chi2 基准={hl.get('hl_chi2_old', 0):.2f} 当前={hl.get('hl_chi2_new', 0):.2f} 偏移={hl.get('calib_shift', 0):.2f}")

    print(f"\n--- Top-扰动 特征 (打乱后AUC上升=对OOT是噪声) ---")
    top_noise = df_abl[df_abl['abl_delta_mean'] > 0].nlargest(5, 'abl_delta_mean')[['feature', 'abl_auc_mean', 'abl_delta_mean', 'abl_label']]
    print(top_noise.to_string(index=False) if not top_noise.empty else "  无扰动特征")

    print(f"\n--- Top-有效 特征 (打乱后AUC下降=有效特征) ---")
    top_valid = df_abl[df_abl['abl_delta_mean'] < 0].nsmallest(5, 'abl_delta_mean')[['feature', 'abl_auc_mean', 'abl_delta_mean', 'abl_label']]
    print(top_valid.to_string(index=False) if not top_valid.empty else "  无有效特征")

    # 5. 列出输出文件
    print(f"\n--- 输出文件 ---")
    if os.path.exists(output_dir):
        for f in sorted(os.listdir(output_dir)):
            fpath = os.path.join(output_dir, f)
            if os.path.isfile(fpath):
                size_kb = os.path.getsize(fpath) / 1024
                print(f"  {f} ({size_kb:.1f} KB)")

    print("\n" + "=" * 60)
    print("归因报告生成完成!")
    print("=" * 60)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
