"""
created by weiyx15 @ 2019.1.4
Cora dataset interface
"""

import random
import numpy as np
from config import get_config
from utils.construct_hypergraph import edge_to_hyperedge
import pickle as pkl
import networkx as nx
import scipy.sparse as sp


def parse_index_file(filename):
    """
    Copied from gcn
    Parse index file.
    """
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))  # 行求和
    r_inv = np.power(rowsum, -1).flatten()  # 倒数
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)  # 归一化特征
    return features


def load_citation_data(cfg):
    """
    Copied from gcn
    citeseer/cora/pubmed with gcn split
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("{}/ind.{}.{}".format(cfg['citation_root'], cfg['activate_dataset'], names[i]), 'rb') as f:
            objects.append(pkl.load(f, encoding='latin1'))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("{}/ind.{}.test.index".format(cfg['citation_root'], cfg['activate_dataset']))  # 测试集索引
    test_idx_range = np.sort(test_idx_reorder)

    if cfg['activate_dataset'] == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()  # 构建节点特征; train + test
    features[test_idx_reorder, :] = features[test_idx_range, :]  # 测试集节点对应到特征;
    features = preprocess_features(features)  # 特征预处理; 归一化特征;
    features = features.todense()

    # G = nx.from_dict_of_lists(graph)  # 构建图;
    # edge_list = G.adjacency_list()  # 源码：networkx = 1.x; G.adjacency_list()
    edge_list = [list(graph[i]) for i in graph]  # networkx = 2.x 使用这个，如果是1.x可以使用上面两行的源码;
    degree = [0] * len(edge_list)
    if cfg['add_self_loop']:  # 添加自连接边;
        for i in range(len(edge_list)):
            edge_list[i].append(i)
            degree[i] = len(edge_list[i])  # 节点degree
    max_deg = max(degree)
    mean_deg = sum(degree) / len(degree)
    print(f'max degree: {max_deg}, mean degree:{mean_deg}')

    labels = np.vstack((ally, ty))  # label拼接
    labels[test_idx_reorder, :] = labels[test_idx_range, :]  # 将test对应到label     # one-hot labels
    n_sample = labels.shape[0]  # 节点数量
    n_category = labels.shape[1]  # label类别数量
    lbls = np.zeros((n_sample,))  # label
    if cfg['activate_dataset'] == 'citeseer':
        n_category += 1                                         # one-hot labels all zero: new category
        for i in range(n_sample):
            try:
                lbls[i] = np.where(labels[i]==1)[0]                     # numerical labels
            except ValueError:                              # labels[i] all zeros
                lbls[i] = n_category + 1                        # new category
    else:
        for i in range(n_sample):
            lbls[i] = np.where(labels[i]==1)[0]  # 对应到label，数值label，非one-hot               # numerical labels

    idx_test = test_idx_range.tolist()  # 测试集索引
    idx_train = list(range(len(y)))  # 训练集索引
    idx_val = list(range(len(y), len(y) + 500))  # 验证集索引
    
    return features, lbls, idx_train, idx_val, idx_test, n_category, edge_list, edge_list


if __name__ == '__main__':
    cfg = get_config('config/config.yaml')
    load_citation_data(cfg)
