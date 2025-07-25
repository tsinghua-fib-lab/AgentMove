import glob
import math
import os
import re
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from scipy.sparse.linalg import eigsh
import scipy.sparse as sp

def fit_delimiter(string='', length=80, delimiter="="):
    result_len = length - len(string)
    half_len = math.floor(result_len / 2)
    result = delimiter * half_len + string + delimiter * half_len
    return result


def init_torch_seeds(seed=0):
    torch.manual_seed(seed)
    if seed == 0:  # slower, more reproducible
        cudnn.benchmark, cudnn.deterministic = False, True
    else:  # faster, less reproducible
        cudnn.benchmark, cudnn.deterministic = True, False


def zipdir(path, ziph, include_format):
    for root, dirs, files in os.walk(path):
        for file in files:
            if os.path.splitext(file)[-1] in include_format:
                filename = os.path.join(root, file)
                arcname = os.path.relpath(os.path.join(root, file), os.path.join(path, '..'))
                ziph.write(filename, arcname)


def increment_path(path, exist_ok=True, sep=''):
    # Increment path, i.e. runs/exp --> runs/exp{sep}0, runs/exp{sep}1 etc.
    path = Path(path)  # os-agnostic
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        return f"{path}{sep}{n}"  # update path


def get_normalized_features(X):
    # X.shape=(num_nodes, num_features)
    means = np.mean(X, axis=0)  # mean of features, shape:(num_features,)
    X = X - means.reshape((1, -1))
    stds = np.std(X, axis=0)  # std of features, shape:(num_features,)
    X = X / stds.reshape((1, -1))
    return X, means, stds


# def calculate_laplacian_matrix(adj_mat, mat_type):
#     n_vertex = adj_mat.shape[0]

#     # row sum
#     deg_mat_row = np.asmatrix(np.diag(np.sum(adj_mat, axis=1)))
#     # column sum
#     # deg_mat_col = np.asmatrix(np.diag(np.sum(adj_mat, axis=0)))
#     deg_mat = deg_mat_row

#     adj_mat = np.asmatrix(adj_mat)
#     id_mat = np.asmatrix(np.identity(n_vertex))

#     if mat_type == 'com_lap_mat':
#         # Combinatorial
#         com_lap_mat = deg_mat - adj_mat
#         return com_lap_mat
#     elif mat_type == 'wid_rw_normd_lap_mat':
#         # For ChebConv
#         rw_lap_mat = np.matmul(np.linalg.matrix_power(deg_mat, -1), adj_mat)
#         rw_normd_lap_mat = id_mat - rw_lap_mat
#         lambda_max_rw = eigsh(rw_lap_mat, k=1, which='LM', return_eigenvectors=False)[0]
#         wid_rw_normd_lap_mat = 2 * rw_normd_lap_mat / lambda_max_rw - id_mat
#         return wid_rw_normd_lap_mat
#     elif mat_type == 'hat_rw_normd_lap_mat':
#         # For GCNConv
#         wid_deg_mat = deg_mat + id_mat
#         wid_adj_mat = adj_mat + id_mat
#         hat_rw_normd_lap_mat = np.matmul(np.linalg.matrix_power(wid_deg_mat, -1), wid_adj_mat)
#         return hat_rw_normd_lap_mat
#     else:
#         raise ValueError(f'ERROR: {mat_type} is unknown.')

def calculate_laplacian_matrix(adj_mat, mat_type):
    # 转为稀疏矩阵表示
    adj_mat = sp.csr_matrix(adj_mat)
    n_vertex = adj_mat.shape[0]

    # 计算度矩阵（行和）
    deg_mat = sp.diags(np.array(adj_mat.sum(axis=1)).flatten())

    # 单位矩阵
    id_mat = sp.eye(n_vertex)

    if mat_type == 'com_lap_mat':
        # Combinatorial Laplacian
        com_lap_mat = deg_mat - adj_mat
        return com_lap_mat.toarray()  # 如果需要返回密集矩阵
    elif mat_type == 'wid_rw_normd_lap_mat':
        # Widely normalized Laplacian (for ChebConv)
        rw_lap_mat = sp.linalg.inv(deg_mat).dot(adj_mat)  # 使用稀疏矩阵求逆
        rw_normd_lap_mat = id_mat - rw_lap_mat
        lambda_max_rw = eigsh(rw_lap_mat, k=1, which='LM', return_eigenvectors=False)[0]
        wid_rw_normd_lap_mat = 2 * rw_normd_lap_mat / lambda_max_rw - id_mat
        return wid_rw_normd_lap_mat.toarray()
    elif mat_type == 'hat_rw_normd_lap_mat':
        # Hat normalized Laplacian (for GCNConv)
        wid_deg_mat = deg_mat + id_mat
        wid_adj_mat = adj_mat + id_mat
        hat_rw_normd_lap_mat = sp.linalg.inv(wid_deg_mat).dot(wid_adj_mat)
        return hat_rw_normd_lap_mat.toarray()
    else:
        raise ValueError(f'ERROR: {mat_type} is unknown.')

def maksed_mse_loss(input, target, mask_value=-1):
    mask = target == mask_value
    if len(input.shape) < 2:
        input = input.unsqueeze(0)
    out = (input[~mask] - target[~mask]) ** 2
    loss = out.mean()
    return loss

def dcg_at_k(scores, k):
    """Calculate DCG for the top k scoring items."""
    order = np.argsort(scores)[::-1]
    scores = np.array(scores)[order][:k]
    gains = 2 ** scores - 1  # 这里假设分数已经是相关性得分
    discounts = np.log2(np.arange(2, k + 2))
    return np.sum(gains / discounts)

def ndcg_at_k(scores, k):
    """Calculate NDCG for the top k scoring items."""
    best_scores = sorted(scores, reverse=True)[:k]
    ideal_dcg = dcg_at_k(best_scores, k)
    actual_dcg = dcg_at_k(scores, k)
    if ideal_dcg == 0:
        return 0
    return actual_dcg / ideal_dcg


def top_k_acc(y_true_seq, y_pred_seq, k):
    hit = 0
    # Convert to binary relevance (nonzero is relevant).
    for y_true, y_pred in zip(y_true_seq, y_pred_seq):
        top_k_rec = y_pred.argsort()[-k:][::-1]
        idx = np.where(top_k_rec == y_true)[0]
        if len(idx) != 0:
            hit += 1
    return hit / len(y_true_seq)


def mAP_metric(y_true_seq, y_pred_seq, k):
    # AP: area under PR curve
    # But in next POI rec, the number of positive sample is always 1. Precision is not well defined.
    # Take def of mAP from Personalized Long- and Short-term Preference Learning for Next POI Recommendation
    rlt = 0
    for y_true, y_pred in zip(y_true_seq, y_pred_seq):
        rec_list = y_pred.argsort()[-k:][::-1]
        r_idx = np.where(rec_list == y_true)[0]
        if len(r_idx) != 0:
            rlt += 1 / (r_idx[0] + 1)
    return rlt / len(y_true_seq)


def MRR_metric(y_true_seq, y_pred_seq):
    """Mean Reciprocal Rank: Reciprocal of the rank of the first relevant item """
    rlt = 0
    for y_true, y_pred in zip(y_true_seq, y_pred_seq):
        rec_list = y_pred.argsort()[-len(y_pred):][::-1]
        r_idx = np.where(rec_list == y_true)[0][0]
        rlt += 1 / (r_idx + 1)
    return rlt / len(y_true_seq)


def top_k_acc_last_timestep(y_true_seq, y_pred_seq, k):
    """ next poi metrics """
    y_true = y_true_seq[-1]
    y_pred = y_pred_seq[-1]
    top_k_rec = y_pred.argsort()[-k:][::-1]
    idx = np.where(top_k_rec == y_true)[0]
    if len(idx) != 0:
        return 1
    else:
        return 0


def mAP_metric_last_timestep(y_true_seq, y_pred_seq, k):
    """ next poi metrics """
    # AP: area under PR curve
    # But in next POI rec, the number of positive sample is always 1. Precision is not well defined.
    # Take def of mAP from Personalized Long- and Short-term Preference Learning for Next POI Recommendation
    y_true = y_true_seq[-1]
    y_pred = y_pred_seq[-1]
    rec_list = y_pred.argsort()[-k:][::-1]
    r_idx = np.where(rec_list == y_true)[0]
    if len(r_idx) != 0:
        return 1 / (r_idx[0] + 1)
    else:
        return 0


def MRR_metric_last_timestep(y_true_seq, y_pred_seq):
    """ next poi metrics """
    # Mean Reciprocal Rank: Reciprocal of the rank of the first relevant item
    y_true = y_true_seq[-1]
    y_pred = y_pred_seq[-1]
    rec_list = y_pred.argsort()[-len(y_pred):][::-1]
    r_idx = np.where(rec_list == y_true)[0]
    if r_idx.size == 0:
        return 0  # or return np.nan, or handle it in a different way depending on your logic
    
    return 1 / (r_idx[0] + 1)

def ndcg_k_last_timestep(y_true_seq, y_pred_scores, k):
    """Calculate NDCG for the last timestep predictions based on scores."""
    y_true = y_true_seq[-1]
    y_pred_scores = y_pred_scores[-1]  # 假设这是分数数组而不是ID

    # 创建相关性分数（1 如果是正确的 POI，否则为 0）
    relevance_scores = [1 if i == y_true else 0 for i in range(len(y_pred_scores))]

    # 计算 NDCG@k
    ndcg_score = ndcg_at_k(relevance_scores, k)
    return ndcg_score

def array_round(x, k=4):
    # For a list of float values, keep k decimals of each element
    return list(np.around(np.array(x), k))
