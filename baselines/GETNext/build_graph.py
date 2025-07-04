""" Build the user-agnostic global trajectory flow map from the sequence data """
import os
import pickle

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm
from param_parser import parameter_parser

def build_global_POI_checkin_graph(df, exclude_user=None):
    G = nx.DiGraph()
    users = list(set(df['user_id'].to_list()))
    if exclude_user in users: users.remove(exclude_user)
    loop = tqdm(users)
    for user_id in loop:
        user_df = df[df['user_id'] == user_id]

        # Add node (POI)
        for i, row in user_df.iterrows():
            node = row['POI_id']
            if node not in G.nodes():
                G.add_node(row['POI_id'],
                           checkin_cnt=1,
                           poi_catid=row['POI_catid'],
                           poi_catid_code=row['POI_catid_code'],
                           poi_catname=row['POI_catname'],
                           latitude=row['latitude'],
                           longitude=row['longitude'])
            else:
                G.nodes[node]['checkin_cnt'] += 1

        # Add edges (Check-in seq)
        previous_poi_id = 0
        previous_traj_id = 0
        for i, row in user_df.iterrows():
            poi_id = row['POI_id']
            traj_id = row['trajectory_id']
            # No edge for the begin of the seq or different traj
            if (previous_poi_id == 0) or (previous_traj_id != traj_id):
                previous_poi_id = poi_id
                previous_traj_id = traj_id
                continue

            # Add edges
            if G.has_edge(previous_poi_id, poi_id):
                G.edges[previous_poi_id, poi_id]['weight'] += 1
            else:  # Add new edge
                G.add_edge(previous_poi_id, poi_id, weight=1)
            previous_traj_id = traj_id
            previous_poi_id = poi_id

    return G


def save_graph_to_csv(G, dst_dir):
    # Save graph to an adj matrix file and a nodes file
    # Adj matrix file: edge from row_idx to col_idx with weight; Rows and columns are ordered according to nodes file.
    # Nodes file: node_name/poi_id, node features (category, location); Same node order with adj matrix.

    # Save adj matrix
    nodelist = list(G.nodes())
    A = nx.adjacency_matrix(G, nodelist=nodelist)
    # np.save(os.path.join(dst_dir, 'adj_mtx.npy'), A.todense())
    np.save(os.path.join(dst_dir, 'graph_A.npy'), A.todense(), allow_pickle=True)

        # 保存节点信息为二进制文件 (.npy)
    nodes_data = []
    for each in G.nodes.data():
        node_name = each[0]
        node_data = each[1]
        nodes_data.append([
            node_name,
            node_data.get('checkin_cnt', 0),
            node_data.get('poi_catid', ''),
            node_data.get('poi_catid_code', ''),
            node_data.get('poi_catname', ''),
            node_data.get('latitude', 0.0),
            node_data.get('longitude', 0.0)
        ])

    # 转换为numpy数组后保存
    nodes_array = np.array(nodes_data, dtype=object)  # 使用dtype=object以支持字符串和数值
    np.save(os.path.join(dst_dir, 'graph_X.npy'), nodes_array)


def save_graph_to_pickle(G, dst_dir):
    pickle.dump(G, open(os.path.join(dst_dir, 'graph.pkl'), 'wb'))


def save_graph_edgelist(G, dst_dir):
    # 确保目标目录存在
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    # 节点ID到索引的映射
    nodelist = list(G.nodes())
    node_id2idx = {k: v for v, k in enumerate(nodelist)}

    # 保存节点ID映射为二进制文件 (.npy)
    node_id2idx_array = np.array(list(node_id2idx.items()), dtype=object)  # 使用 dtype=object 支持字符串和整数
    np.save(os.path.join(dst_dir, 'graph_node_id2idx.npy'), node_id2idx_array)

    # 保存边列表为二进制文件 (.npy)
    edge_data = []
    for src_node, dst_node, weight in G.edges(data='weight'):
        edge_data.append([node_id2idx[src_node], node_id2idx[dst_node], weight])

    # 将边列表转换为 numpy 数组并保存
    edge_array = np.array(edge_data, dtype=object)  # 使用 dtype=object 支持浮点数和整数
    np.save(os.path.join(dst_dir, 'graph_edge.npy'), edge_array)

def load_graph_adj_mtx(path):
    """A.shape: (num_node, num_node), edge from row_index to col_index with weight"""
    A = np.loadtxt(path, delimiter=',')
    return A


def load_graph_node_features(path, feature1='checkin_cnt', feature2='poi_catid_code',
                             feature3='latitude', feature4='longitude'):
    """X.shape: (num_node, 4), four features: checkin cnt, poi cat, latitude, longitude"""
    df = pd.read_csv(path)
    rlt_df = df[[feature1, feature2, feature3, feature4]]
    X = rlt_df.to_numpy()

    return X


def print_graph_statisics(G):
    print(f"Num of nodes: {G.number_of_nodes()}")
    print(f"Num of edges: {G.number_of_edges()}")

    # Node degrees (mean and percentiles)
    node_degrees = [each[1] for each in G.degree]
    print(f"Node degree (mean): {np.mean(node_degrees):.2f}")
    for i in range(0, 101, 20):
        print(f"Node degree ({i} percentile): {np.percentile(node_degrees, i)}")

    # Edge weights (mean and percentiles)
    edge_weights = []
    for n, nbrs in G.adj.items():
        for nbr, attr in nbrs.items():
            weight = attr['weight']
            edge_weights.append(weight)
    print(f"Edge frequency (mean): {np.mean(edge_weights):.2f}")
    for i in range(0, 101, 20):
        print(f"Edge frequency ({i} percentile): {np.percentile(edge_weights, i)}")


if __name__ == '__main__':
    args = parameter_parser()
    dst_dir = f'dataset/{str(args.train_sample)}/{args.dataset_name}'
    

    # Build POI checkin trajectory graph
    train_df = pd.read_csv(os.path.join(dst_dir, 'train.csv'))
    print('Build global POI checkin graph -----------------------------------')
    G = build_global_POI_checkin_graph(train_df)

    # Save graph to disk
    save_graph_to_pickle(G, dst_dir=dst_dir)
    save_graph_to_csv(G, dst_dir=dst_dir)
    save_graph_edgelist(G, dst_dir=dst_dir)
