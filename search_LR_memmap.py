import pandas as pd
import numpy as np
import datetime
from IPython.display import clear_output
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import joblib
import os
import time
import gc
import multiprocessing
import pdb

# param_grid = {
#     'C': [0.01, 1],
#     'max_iter': [100]
# }

# register_vars = ['train_df', 'oot_123d_sleep', 'oot_123d_jt', 'oot_123d_jt_info', 'oot_123d_jt_Ninfo', 'oot_123d_fz', 'oot_123d_fz_info', 'oot_123d_fz_Ninfo']
# weights_oot = [0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1]

# # 结清T1T2搜索
register_vars = ['train_df', 'oot_2509_df', 'oot_2510_df', 'oot_2511_df', 'oot_2512_df']
weights_oot = [0.25, 0.25, 0.25, 0.25]

hard_cond_oot = [None, None, None, None]

def split_data(target, train_span, oot_span, **kwargs):
    df_p_all = df_p[df_p[target].isin([0, 1])]
    get_idx = lambda time_span: df_p_all['recall_date'].ge(time_span[0]) & df_p_all['recall_date'].le(time_span[1])

    train_idx = get_idx(train_span)
    oot_idx = get_idx(oot_span)

    train_df, oot_df = df_p_all[train_idx], df_p_all[oot_idx]

    oot_2509_df = oot_df[oot_df['recall_date'].ge('2025-09-01') & oot_df['recall_date'].le('2025-09-30')]
    oot_2510_df = oot_df[oot_df['recall_date'].ge('2025-10-01') & oot_df['recall_date'].le('2025-10-31')]
    oot_2511_df = oot_df[oot_df['recall_date'].ge('2025-11-01') & oot_df['recall_date'].le('2025-11-30')]
    oot_2512_df = oot_df[oot_df['recall_date'].ge('2025-12-01') & oot_df['recall_date'].le('2025-12-31')]

    for var in register_vars:
        print(f'{var} total samples {len(eval(var))}')
    
    return eval(f'({",".join(register_vars)})')

def print_auc_list(auc_list):
    oot_name_list = register_vars[1:]
    return '  '.join([f'[Auc {name}: {new_auc:.4f}]' for name, new_auc in zip(oot_name_list, auc_list)])

def print_col(feat_idx):
    return f'{[score_all[idx] for idx in feat_idx]}'

sql_template_lr = '''
1 / (1 + exp(-({linearcomb} {intercept})))
'''
stringify = lambda x, i: f'{"+" if x > 0 and i > 0 else ""}{x}'

def print_fusion_SQL(clf, subscores):
    coef = clf.coef_.tolist()[0]
    intercept = clf.intercept_.tolist()[0]
    linearcomb_str = ' '.join([f'{stringify(c, i)} * {s}' for i, (c, s) in enumerate(zip(coef, subscores))])
    return sql_template_lr.format(linearcomb = linearcomb_str, intercept = stringify(intercept, 1))

def train_LR(data, features, params):
    x_arrays, y_arrays = data
    train_x, oot_xs = x_arrays[0], x_arrays[1:]
    train_y, oot_ys = y_arrays[0], y_arrays[1:]

    # clf = LogisticRegression(**params)
    # grid_search = GridSearchCV(
    #     estimator=clf,
    #     param_grid=param_grid,
    #     cv=2,
    #     scoring='roc_auc',
    #     n_jobs=1,
    #     verbose=1
    # )
    # grid_search.fit(train_x[:, features], train_y)

    clf = LogisticRegression(**params, penalty='l1').fit(train_x[:, features], train_y)
    
    print('fit finished')
    oot_auc_list = [roc_auc_score(oot_y, clf.predict_proba(oot_x[:, features])[:, 1]) for oot_x, oot_y in zip(oot_xs, oot_ys)]
    
    return clf, {}, oot_auc_list

print_lock = multiprocessing.Lock()
def process(args):
    x_arrays_shm, y_arrays_shm, features, params, weights_list = args

    x_arrays = []
    for shm_name, shape, dtype in x_arrays_shm:
        x_arrays.append(np.memmap(shm_name, dtype=dtype, mode='r', shape=shape))

    y_arrays = []
    for shm_name, shape, dtype in y_arrays_shm:
        y_arrays.append(np.memmap(shm_name, dtype=dtype, mode='r', shape=shape))

    with print_lock:
        print(f'Process-{multiprocessing.current_process().pid} trying features {features}')

    bst, bst_params, oot_auc_list = train_LR((x_arrays, y_arrays), features, params)

    # 计算综合指标（带权重）
    combined_auc = sum([w * cur for w, cur in zip(weights_list, oot_auc_list)])
    with print_lock:
        print(f'Process-{multiprocessing.current_process().pid}, features {print_col(features)}, current combined auc: {combined_auc:.4f}\nall auc list: {print_auc_list(oot_auc_list)}')

    return (features, bst, bst_params, *oot_auc_list, combined_auc)

# 硬条件过滤
def hard_filter_wrapper(hard_cond):
    def hard_filter(candidate):
        auc_list = candidate[3:-1]
        return all([(True if thres is None else auc >= thres) for auc, thres in zip(auc_list, hard_cond)])
    return hard_filter

# 可解释性过滤
def coef_filter_wrapper():
    def coef_filter(candidate):
        clf = candidate[1]
        return all([c > 0 for c in clf.coef_.tolist()[0]])
    return coef_filter

def trainNselect(data, feature_col, params, 
                 beam_width = 5,    # 搜索宽度
                 patience = 3,      # 最大允许无改进次数
                 weights_list = weights_oot,    # 权重设置（从低到高）
                 hard_cond = hard_cond_oot
    ):
    best_combined = 0  # 综合最优值
    # best_auc_list = [0] * len(oot_df_list)
    # best_features = []       # 记录历史最佳组合

    best_candidate_per_level = []

    no_improve_count = 0    

    # selected = []
    candidate_features = feature_col

    # 创建共享内存
    x_arrays_shm = []
    y_arrays_shm = []
    x_arrays = []
    y_arrays = []

    if not os.path.exists(memmap_dir):
        os.makedirs(memmap_dir)

    for idx, df in enumerate(data): # 默认第0个为train_df
        arr_x = df[score_all].values
        name_x = os.path.join(memmap_dir, f'x_{idx}.dat')
        # 这里 w+ 即overwrite；没有追加模式哈
        mmap = np.memmap(name_x, dtype=arr_x.dtype, mode='w+', shape=arr_x.shape)
        mmap[:] = arr_x[:]
        mmap.flush()

        x_arrays.append(arr_x)
        x_arrays_shm.append((name_x, arr_x.shape, arr_x.dtype))  # 存储元数据

        arr_y = df[target].values
        name_y = os.path.join(memmap_dir, f'y_{idx}.dat')
        mmap_y = np.memmap(name_y, dtype=arr_y.dtype, mode='w+', shape=arr_y.shape)
        mmap_y[:] = arr_y[:]
        mmap_y.flush()

        y_arrays.append(arr_y)
        y_arrays_shm.append((name_y, arr_y.shape, arr_y.dtype))  # 存储元数据
    
    beam_list = [set()]
    iter_cnt = 1

    while len(best_candidate_per_level) < len(candidate_features) and no_improve_count < patience:        
        # candidates = []
        features_to_search = []
        
        history = set()
        
        for selected_set in beam_list:
            # 并行评估候选特征
            for feat in candidate_features:
                if feat in selected_set:
                    continue
                    
                try_features = list(selected_set) + [feat]
                
                cur_feature_index = tuple(sorted(try_features))
                if cur_feature_index in history:
                    continue
                history.add(cur_feature_index)

                features_to_search.append(try_features)

        with multiprocessing.Pool(processes = 4) as pool:
            candidates = pool.map(process, [(x_arrays_shm, y_arrays_shm, try_features, params, weights_list) for try_features in features_to_search])

        print(f'Beam search iterations {iter_cnt}')
        clear_output()

        # 优先综合指标，其余指标按权重排序（倒排）
        candidates.sort(key=lambda x: x[3:][::-1], reverse = True) 
        filter_func = hard_filter_wrapper(hard_cond)
        candidate_hard_T = [cand for cand in candidates if filter_func(cand)]
        candidate_hard_F = [cand for cand in candidates if not filter_func(cand)]
        candidates = candidate_hard_T + candidate_hard_F

        coef_filter = coef_filter_wrapper()
        candidates = [cand for cand in candidates if coef_filter(cand)]

        improve_flag = False
        best_candidate_per_level.append(candidates[0])

        for candidate in candidates[:beam_width]:
            # 选择最优候选
            feat_list, combined_auc = candidate[0], candidate[-1]
            # auc_list = candidate[1:-1]

            if combined_auc > best_combined + 0.0001:
                best_combined = combined_auc
                # best_features = feat_list
                # best_auc_list = auc_list

                improve_flag = True

        print_result(f'====================== best {beam_width} for iteration {iter_cnt} ===================')
        for cand in candidates[:beam_width]:
            print_result(f"feature set with {len(cand[0])} features: {print_col(cand[0])}")
            print_result(f"combined auc: {cand[-1]:.4f}, with all AUC: {print_auc_list(cand[3:-1])}")
            # print_result(f'params: {cand[2]}')
            print_result(print_fusion_SQL(cand[1], [score_all[idx] for idx in cand[0]]))
            print_result('--------------------------')

        beam_list = [set(candidate[0]) for candidate in candidates[:beam_width]]
        if improve_flag:
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        iter_cnt += 1
        gc.collect()

    # 最终选择历史最佳组合
    # selected = best_features

    return best_candidate_per_level
    # return train_LR((x_arrays, y_arrays), selected, params)

def print_result_wrapper(file_name = ''):
    def print_result(msg):
        print(msg)
        if file_name:
            with open(file_name, 'a') as f:
                print(msg, file = f)
    return print_result

model_name = 'drawbcard6t'
output_dir = './output_model'
memmap_dir = './numpy_memmap_all/numpy_memmap'
current_date = '20251104'
data_csv_path = 'jxh_model_draw_6t_train_beamsearch.csv'
LR_model_template = 'LR_level{}_fusion.pkl'
LR_output_name = 'LR_fusion_result_0309.txt'

# 移除两通版
# LR_output_name = 'LR_fusion_result_0108_jqt1t2_ncr.txt'

target_config = [
   {
       'target': 'mob6_30_ever',
       'identifier': 'fusion6t30+ever',
       'train_span': ['2025-08-01', '2025-08-31'],
       'oot_span': ['2025-09-01', '2025-12-31'],
        # 排除子分列
        'exc_cols': ['appl_no', 'recall_date', 'mob6_30_ever'],
        'params': {}
   },
]

base_param = {'random_state': 42}

if __name__ == '__main__':
    df_p = pd.read_csv(data_csv_path, sep = '\t')

    df_p['recall_date'] = pd.to_datetime(df_p['recall_date'], errors='coerce').map(lambda x:str(x)[0:10])

    all_candidate = []

    for conf in target_config:
        target: str = conf["target"]
        params = conf['params']

        exc_cols = conf['exc_cols']
        score_all = [col for col in df_p.columns if not col in exc_cols]

        # 移除两通过滤
        # score_all = [col for col in score_all if (('ncr' in col or 'distill' in col) and not col == 'fusion3yddistillScore')]

        print(f'searching among {len(score_all)} subscores')
        print(score_all)

        df_p[score_all] = df_p[score_all].astype(float).fillna(-999.0)

        print_result = print_result_wrapper(os.path.join(output_dir, conf['identifier'], LR_output_name))

        if not os.path.exists(os.path.join(output_dir, conf["identifier"])):
            os.makedirs(os.path.join(output_dir, conf["identifier"]))

        print_result(f'===== searching for config {conf["identifier"]} =====')

        selected = score_all

        print(f'Selecing target {target}')
        data = split_data(**conf)
        train_df = data[0]
        oot_df_list = data[1:]

        print(f'Training target {target}')

        # cur_params = {k: eval(v) for k, v in param_grid}
        cand = trainNselect(data, list(range(len(score_all))), base_param)
        all_candidate.append(cand)
        # joblib.dump(clf, os.path.join(output_dir, f"{model_name}_LR_model_{target}_{current_date}.pkl"))

        gc.collect()

    # 回写结果，输出模型
    for conf_cand, conf in zip(all_candidate, target_config):
        print(f'===== best level candidate for config {conf["identifier"]} =====')
        target = conf["target"]
        data = split_data(**conf)
        train_df = data[0]

        oot_df_list = data[1:]

        for idx, cand in enumerate(conf_cand):
            output_model_path = os.path.join(output_dir, conf["identifier"], LR_model_template.format(idx + 1))

            feat_cols = [score_all[idx] for idx in cand[0]]
            clf = LogisticRegression(**base_param, **cand[1])
            clf.fit(train_df[feat_cols], train_df[target])

            joblib.dump(clf, output_model_path)

            print_result(f"best level feature set with {len(cand[0])} features: {print_col(cand[0])}")
            print_result(f"best level combined auc: {cand[-1]:.4f}, with all AUC: {print_auc_list(cand[3:-1])}")

