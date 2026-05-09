# -*- coding: utf-8 -*-
"""生成虚假测试数据，用于 AutoModelBuilderTools 全功能测试"""
import pandas as pd
import numpy as np
import os

np.random.seed(42)
output_dir = os.path.dirname(os.path.abspath(__file__))

# === 1. 主数据集（用于 train / feature_analysis / scoring / model_evaluation / attribution）===
n = 5000
data = {
    'appl_no': [f'APP{i:06d}' for i in range(n)],
    'draw_month': np.random.choice(['202511', '202512', '202601', '202602'], n),
    'flag_cg_yz': np.random.choice(['常规户MOB6+', '优质户MOB6+'], n),
    'cust_status3': np.random.choice(['结清365-', '结清365+'], n),
    'dpd30_term1': np.random.choice([0, 1], n, p=[0.95, 0.05]),
    'dpd30_term3': np.random.choice([0, 1], n, p=[0.93, 0.07]),
}
# 生成特征列
for i in range(15):
    col = f'feature_{i:02d}'
    if i < 3:
        data[col] = np.random.randn(n) * (i + 1)
    elif i < 8:
        data[col] = np.random.uniform(0, 100, n)
    else:
        data[col] = np.random.choice([0, 1, 2, 3], n).astype(float)

# 注入缺失值
for col in [f'feature_{i:02d}' for i in [2, 5, 9, 13]]:
    mask = np.random.random(n) < 0.05
    arr = np.array(data[col], dtype=float)
    arr[mask] = np.nan
    data[col] = arr

df = pd.DataFrame(data)
df.to_csv(os.path.join(output_dir, 'test_data.csv'), sep='\t', index=False)
print(f"主数据集: {df.shape}, 已保存")

# === 2. OOT 数据（用于 beamsearch）===
n_oot = 2000
oot_data = {
    'appl_no': [f'OOT_APP{i:06d}' for i in range(n_oot)],
    'draw_month': np.random.choice(['202601', '202602'], n_oot),
    'dpd30_term1': np.random.choice([0, 1], n_oot, p=[0.95, 0.05]),
}
for i in range(15):
    col = f'feature_{i:02d}'
    if i < 3:
        oot_data[col] = np.random.randn(n_oot) * (i + 1)
    elif i < 8:
        oot_data[col] = np.random.uniform(0, 100, n_oot)
    else:
        oot_data[col] = np.random.choice([0, 1, 2, 3], n_oot).astype(float)

oot1 = pd.DataFrame(oot_data)
oot1.to_csv(os.path.join(output_dir, 'test_oot1.csv'), sep='\t', index=False)
print(f"OOT1 数据集: {oot1.shape}, 已保存")

oot2 = oot1.copy()
oot2['appl_no'] = [f'OOT2_APP{i:06d}' for i in range(n_oot)]
oot2.to_csv(os.path.join(output_dir, 'test_oot2.csv'), sep='\t', index=False)
print(f"OOT2 数据集: {oot2.shape}, 已保存")

# === 3. 小数据集（异常场景测试）===
# 只有几行，用于测试边界情况
tiny = pd.DataFrame({
    'appl_no': ['T001', 'T002', 'T003'],
    'draw_month': ['202601', '202601', '202601'],
    'dpd30_term1': [0, 0, 0],  # 全为0，无正样本
    'feature_00': [1.0, 2.0, 3.0],
    'feature_01': [4.0, 5.0, 6.0],
})
tiny.to_csv(os.path.join(output_dir, 'test_tiny.csv'), sep='\t', index=False)
print(f"微小数据集: {tiny.shape}, 已保存")

# 全缺失数据
nan_data = pd.DataFrame({
    'appl_no': ['N001', 'N002'],
    'draw_month': ['202601', '202601'],
    'dpd30_term1': [0, 1],
    'feature_00': [np.nan, np.nan],
    'feature_01': [np.nan, np.nan],
})
nan_data.to_csv(os.path.join(output_dir, 'test_nan.csv'), sep='\t', index=False)
print(f"全缺失数据集: {nan_data.shape}, 已保存")

# 单值常数列数据
const_data = pd.DataFrame({
    'appl_no': [f'C{i:03d}' for i in range(100)],
    'draw_month': np.random.choice(['202511', '202512'], 100),
    'dpd30_term1': np.random.choice([0, 1], 100, p=[0.9, 0.1]),
    'feature_00': np.random.randn(100),
    'feature_01': np.random.randn(100),
    'feature_const': [42.0] * 100,  # 常数列
})
const_data.to_csv(os.path.join(output_dir, 'test_const.csv'), sep='\t', index=False)
print(f"常数列数据集: {const_data.shape}, 已保存")

print("\n所有测试数据已生成！")
