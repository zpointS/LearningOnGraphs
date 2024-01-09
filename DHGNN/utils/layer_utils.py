import torch
from torch import nn
import pandas as pd


def cos_dis(X):
        """
        cosine distance
        :param X: (N, d)
        :return: (N, N)
        """
        X = nn.functional.normalize(X)  # 特征归一化
        XT = X.transpose(0, 1)
        return torch.matmul(X, XT)  # X和XT内积; 节点之间的距离


def sample_ids(ids, k):
    """
    sample `k` indexes from ids, must sample the centroid node itself
    :param ids: indexes sampled from
    :param k: number of samples
    :return: sampled indexes
    """
    df = pd.DataFrame(ids)
    sampled_ids = df.sample(k - 1, replace=True).values
    sampled_ids = sampled_ids.flatten().tolist()
    sampled_ids.append(ids[-1])  # must sample the centroid node itself; 必须要有节点本身
    return sampled_ids


def sample_ids_v2(ids, k):
    """
    purely sample `k` indexes from ids
    :param ids: indexes sampled from
    :param k: number of samples
    :return: sampled indexes
    """
    df = pd.DataFrame(ids)  # 最近的簇中节点集合
    sampled_ids = df.sample(k, replace=True).values  # 可重复采样
    sampled_ids = sampled_ids.flatten().tolist()
    return sampled_ids