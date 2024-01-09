import math
import copy
import torch
import time
from torch import nn
from torch.nn.parameter import Parameter
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from utils.layer_utils import sample_ids, sample_ids_v2, cos_dis


class Transform(nn.Module):
    """
    A Vertex Transformation module
    Permutation invariant transformation: (N, k, d) -> (N, k, d)
    """
    def __init__(self, dim_in, k):
        """
        :param dim_in: input feature dimension
        :param k: k neighbors
        """
        super().__init__()

        self.convKK = nn.Conv1d(k, k * k, dim_in, groups=k)
        self.activation = nn.Softmax(dim=-1)
        self.dp = nn.Dropout()

    def forward(self, region_feats):
        """
        :param region_feats: (N, k, d)
        :return: (N, k, d)
        """
        N, k, _ = region_feats.size()  # (N, k, d)
        conved = self.convKK(region_feats)  # [140, 4096, 1]
        multiplier = conved.view(N, k, k)  # [140, 64, 64]
        multiplier = self.activation(multiplier)  # 最后一个维度softmax
        transformed_feats = torch.matmul(multiplier, region_feats)  # 邻居节点加权求和
        return transformed_feats


class VertexConv(nn.Module):
    """
    A Vertex Convolution layer
    Transform (N, k, d) feature to (N, d) feature by transform matrix and 1-D convolution
    """
    def __init__(self, dim_in, k):
        """
        :param dim_in: input feature dimension
        :param k: k neighbors
        """
        super().__init__()

        self.trans = Transform(dim_in, k)                   # (N, k, d) -> (N, k, d)
        self.convK1 = nn.Conv1d(k, 1, 1)                    # (N, k, d) -> (N, 1, d)

    def forward(self, region_feats):
        """
        :param region_feats: (N, k, d) 邻居特征
        :return: (N, d)
        """
        transformed_feats = self.trans(region_feats)  # 邻居特征加权求和;
        pooled_feats = self.convK1(transformed_feats)  # (N, 1, d)
        pooled_feats = pooled_feats.squeeze(1)
        return pooled_feats


class GraphConvolution(nn.Module):
    """
    A GCN layer
    """
    def __init__(self, **kwargs):
        """
        :param kwargs:
        # dim_in,
        # dim_out,
        # dropout_rate=0.5,
        # activation
        """
        super().__init__()

        self.dim_in = kwargs['dim_in']
        self.dim_out = kwargs['dim_out']
        self.fc = nn.Linear(self.dim_in, self.dim_out, bias=kwargs['has_bias'])
        self.dropout = nn.Dropout(p=0.5)
        self.activation = kwargs['activation']

    def _region_aggregate(self, feats, edge_dict):
        N = feats.size()[0]  # 节点数量
        pooled_feats = torch.stack([torch.mean(feats[edge_dict[i]], dim=0) for i in range(N)])  # 遍历, 节点及其邻居求均值

        return pooled_feats

    def forward(self, ids, feats, edge_dict, G, ite):
        """
        :param ids: compatible with `MultiClusterConvolution`
        :param feats:
        :param edge_dict:
        :return:
        """
        x = feats  # (N, d)
        x = self.dropout(self.activation(self.fc(x)))  # 1433 -> 256
        x = self._region_aggregate(x, edge_dict)  # mean pooling
        return x


class EdgeConv(nn.Module):
    """
    A Hyperedge Convolution layer
    Using self-attention to aggregate hyperedges
    """
    def __init__(self, dim_ft, hidden):
        """
        :param dim_ft: feature dimension
        :param hidden: number of hidden layer neurons
        """
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(dim_ft, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, ft):
        """
        use self attention coefficient to compute weighted average on dim=-2
        :param ft (N, t, d)
        :return: y (N, d)
        """
        scores = []
        n_edges = ft.size(1)  #  超边的种类
        for i in range(n_edges):
            scores.append(self.fc(ft[:, i]))  # 全连接层
        scores = torch.softmax(torch.stack(scores, 1), 1)  # 多个类别超边进行加权求和;
        
        return (scores * ft).sum(1)  # 超边进行加权求和


class DHGLayer(GraphConvolution):
    """
    A Dynamic Hypergraph Convolution Layer
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)  # GraphConvolution

        self.ks = kwargs['structured_neighbor'] # number of sampled nodes in graph adjacency
        self.n_cluster = kwargs['n_cluster']  # cluster数量，400;          # number of clusters
        self.n_center = kwargs['n_center']                # a node has #n_center adjacent clusters
        self.kn = kwargs['nearest_neighbor']  # KNN采样数量     # number of the 'k' in k-NN
        self.kc = kwargs['cluster_neighbor']  # Kmeans中采样数量;    # number of sampled nodes in a adjacent k-means cluster
        self.wu_knn=kwargs['wu_knn']
        self.wu_kmeans=kwargs['wu_kmeans']
        self.wu_struct=kwargs['wu_struct']
        self.vc_sn = VertexConv(self.dim_in, self.ks+self.kn)    # structured trans
        self.vc_s = VertexConv(self.dim_in, self.ks)    # structured trans
        self.vc_n = VertexConv(self.dim_in, self.kn)    # nearest trans
        self.vc_c = VertexConv(self.dim_in, self.kc)   # k-means cluster trans
        self.ec = EdgeConv(self.dim_in, hidden=self.dim_in//4)  # 超边卷积
        self.kmeans = None
        self.structure = None

    def _vertex_conv(self, func, x):
        return func(x)

    def _structure_select(self, ids, feats, edge_dict):
        """
        :param ids: indices selected during train/valid/test, torch.LongTensor
        :param feats:
        :param edge_dict: torch.LongTensor
        :return: mapped graph neighbors
        """
        if self.structure is None:
            _N = feats.size(0)
            idx = torch.LongTensor([sample_ids(edge_dict[i], self.ks) for i in range(_N)])    # 遍历节点
            self.structure = idx  # 节点的邻居节点
        else:
            idx = self.structure

        idx = idx[ids]  # 训练节点的邻居节点
        N = idx.size(0)
        d = feats.size(1)
        region_feats = feats[idx.view(-1)].view(N, self.ks, d)          # (N, ks, d)
        return region_feats

    def _nearest_select(self, ids, feats):
        """
        :param ids: indices selected during train/valid/test, torch.LongTensor
        :param feats:
        :return: mapped nearest neighbors
        """
        dis = cos_dis(feats)  # 节点和节点之间的距离
        _, idx = torch.topk(dis, self.kn, dim=1)  # 所有k个近邻;
        idx = idx[ids]  # 训练集的k个近邻;
        N = len(idx)
        d = feats.size(1)  # 特征维度
        nearest_feature = feats[idx.view(-1)].view(N, self.kn, d)  # 所有邻居节点的特征
        return nearest_feature

    def _cluster_select(self, ids, feats):
        """
        compute k-means centers and cluster labels of each node
        return top #n_cluster nearest cluster transformed features
        :param ids: indices selected during train/valid/test, torch.LongTensor
        :param feats:
        :return: top #n_cluster nearest cluster mapped features
        """
        if self.kmeans is None:
            _N = feats.size(0)
            np_feats = feats.detach().cpu().numpy()
            kmeans = KMeans(n_clusters=self.n_cluster, random_state=0, n_jobs=-1).fit(np_feats)  # Kmeans聚类;
            centers = kmeans.cluster_centers_  # 聚类中心
            dis = euclidean_distances(np_feats, centers)  # 每个距离聚类中心的聚类 (2708, 400)
            _, cluster_center_dict = torch.topk(torch.Tensor(dis), self.n_center, largest=False)  # 距离最近的k个(1个)
            cluster_center_dict = cluster_center_dict.numpy()
            point_labels = kmeans.labels_  # 每个节点属于的类别
            point_in_which_cluster = [np.where(point_labels == i)[0] for i in range(self.n_cluster)]  # 每个类别中包含的节点;
            idx = torch.LongTensor([[sample_ids_v2(point_in_which_cluster[cluster_center_dict[point][i]], self.kc)  # 簇中最近的k个节点
                        for i in range(self.n_center)] for point in range(_N)])    # 遍历每个节点; 遍历k个kmeans超边;
            self.kmeans = idx  # 每个节点对应到最近簇的节点采样; (2708, 1, 64)
        else:
            idx = self.kmeans
        
        idx = idx[ids]  # 训练节点对应的簇邻居节点
        N = idx.size(0)
        d = feats.size(1)
        cluster_feats = feats[idx.view(-1)].view(N, self.n_center, self.kc, d)  # 簇邻居节点的特征; [140, 1, 64, 256]

        return cluster_feats                    # (N, n_center, kc, d)

    def _edge_conv(self, x):
        return self.ec(x)

    def _fc(self, x):
        return self.activation(self.fc(self.dropout(x)))

    def forward(self, ids, feats, edge_dict, G, ite):
        hyperedges = []    
        if ite >= self.wu_kmeans:  # Kmeans簇中采样
            c_feat = self._cluster_select(ids, feats)  # kmeans距离
            for c_idx in range(c_feat.size(1)):  # 选择几个簇类;
                xc = self._vertex_conv(self.vc_c, c_feat[:, c_idx, :, :])  # 节点卷积; 邻居加权求和
                xc  = xc.view(len(ids), 1, feats.size(1))               # (N, 1, d)          
                hyperedges.append(xc)
        if ite >= self.wu_knn:  # KNN方式
            n_feat = self._nearest_select(ids, feats)  # 节点选择K和最近邻, 特征;
            xn = self._vertex_conv(self.vc_n, n_feat)  # 对邻居节点进行权重求和;
            xn  = xn.view(len(ids), 1, feats.size(1))                   # (N, 1, d)
            hyperedges.append(xn)  # 采用KNN方式，对邻居进行加权求和结果;
        if ite >= self.wu_struct:  # 有图结构情况下，会对节点连接到的节点进行采样;
            s_feat = self._structure_select(ids, feats, edge_dict)
            xs = self._vertex_conv(self.vc_s, s_feat)
            xs  = xs.view(len(ids), 1, feats.size(1))                   # (N, 1, d)
            hyperedges.append(xs)
        x = torch.cat(hyperedges, dim=1)
        x = self._edge_conv(x)  # 超边聚合                                       # (N, d)
        x = self._fc(x)  # 全连接输出结果                                           # (N, d')
        return x


class HGNN_conv(nn.Module):
    """
    A HGNN layer
    """
    def __init__(self, **kwargs):
        super(HGNN_conv, self).__init__()

        self.dim_in = kwargs['dim_in']
        self.dim_out = kwargs['dim_out']
        self.fc = nn.Linear(self.dim_in, self.dim_out, bias=kwargs['has_bias'])
        self.dropout = nn.Dropout(p=0.5)
        self.activation = kwargs['activation']


    def forward(self, ids, feats, edge_dict, G, ite):
        x = feats
        x = self.activation(self.fc(x))
        x = G.matmul(x)
        x = self.dropout(x)
        return x
