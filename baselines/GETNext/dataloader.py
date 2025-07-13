import numpy as np
import pandas as pd


# def load_graph_adj_mtx(path):
#     """A.shape: (num_node, num_node), edge from row_index to col_index with weight"""
#     A = np.loadtxt(path, delimiter=',')
#     return A

def load_graph_adj_mtx(path):
    """Load adjacency matrix from binary (.npy) file. A.shape: (num_node, num_node)"""
    A = np.load(path)  # 加载二进制文件
    return A


# def load_graph_node_features(path, feature1='checkin_cnt', feature2='poi_catid_code',
#                              feature3='latitude', feature4='longitude'):
#     """X.shape: (num_node, 4), four features: checkin cnt, poi cat, latitude, longitude"""
#     df = pd.read_csv(path)
#     rlt_df = df[[feature1, feature2, feature3, feature4]]
#     X = rlt_df.to_numpy()

#     return X
def load_graph_node_features(path, feature_indices=(1, 2, 5, 6)):
    """
    Load specific features from node binary (.npy) file.
    feature_indices: A tuple indicating the indices of the features in the .npy file
                     (e.g., (1, 2, 5, 6) for checkin_cnt, poi_catid_code, latitude, longitude)
    """
    nodes_array = np.load(path, allow_pickle=True)  # 加载节点信息二进制文件
    # 提取指定的列
    X = nodes_array[:, feature_indices].astype(float)
    return X

