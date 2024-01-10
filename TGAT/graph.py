import numpy as np
import torch

class NeighborFinder:
    def __init__(self, adj_list, uniform=False):
        """
        Params
        ------
        node_idx_l: List[int]
        node_ts_l: List[int]
        off_set_l: List[int], such that node_idx_l[off_set_l[i]:off_set_l[i + 1]] = adjacent_list[i]
        """ 
        # dst节点,  节点时间,    边的编号,    图中节点对应到的所有连接边的数量
        node_idx_l, node_ts_l, edge_idx_l, off_set_l = self.init_off_set(adj_list)
        self.node_idx_l = node_idx_l
        self.node_ts_l = node_ts_l
        self.edge_idx_l = edge_idx_l
        
        self.off_set_l = off_set_l
        
        self.uniform = uniform
        
    def init_off_set(self, adj_list):
        """
        Params
        ------
        adj_list: List[List[int]]
        
        """
        n_idx_l = []
        n_ts_l = []
        e_idx_l = []
        off_set_l = [0]
        for i in range(len(adj_list)):  # 遍历邻接矩阵 [dst, id, time]
            curr = adj_list[i]  # 当前节点连接的邻居节点; (dst, idx, time)
            curr = sorted(curr, key=lambda x: x[1])  # 按idx排序, 其实也就是time排序
            n_idx_l.extend([x[0] for x in curr])  # 所有的dst
            e_idx_l.extend([x[1] for x in curr])  # idx
            n_ts_l.extend([x[2] for x in curr])  # time
           
            
            off_set_l.append(len(n_idx_l))  # 每个节点，连接到之前的节点总和; 递增
        n_idx_l = np.array(n_idx_l)
        n_ts_l = np.array(n_ts_l)
        e_idx_l = np.array(e_idx_l)
        off_set_l = np.array(off_set_l)

        assert(len(n_idx_l) == len(n_ts_l))
        assert(off_set_l[-1] == len(n_ts_l))
        # dst节点;  time;  边的索引;  图中节点的涉及到的所有边的数量
        return n_idx_l, n_ts_l, e_idx_l, off_set_l
        
    def find_before(self, src_idx, cut_time):
        """
        Params
        ------
        src_idx: int
        cut_time: float
        """
        # NeighborFinder中定义的节点信息
        node_idx_l = self.node_idx_l  # dst节点; src就是self.off_set_l的索引;
        node_ts_l = self.node_ts_l  # time
        edge_idx_l = self.edge_idx_l  # edge_index
        off_set_l = self.off_set_l  # 图中节点连接的信息
        
        neighbors_idx = node_idx_l[off_set_l[src_idx]:off_set_l[src_idx + 1]]  # 这个节点所连接到的邻居节点
        neighbors_ts = node_ts_l[off_set_l[src_idx]:off_set_l[src_idx + 1]]  # 每个节点连接到其他节点的时间
        neighbors_e_idx = edge_idx_l[off_set_l[src_idx]:off_set_l[src_idx + 1]]  # 索引
        
        if len(neighbors_idx) == 0 or len(neighbors_ts) == 0:  # 没有邻居
            return neighbors_idx, neighbors_ts, neighbors_e_idx

        left = 0
        right = len(neighbors_idx) - 1
        
        while left + 1 < right:  # 二分法计算小于当前时间的索引
            mid = (left + right) // 2
            curr_t = neighbors_ts[mid]
            if curr_t < cut_time:
                left = mid
            else:
                right = mid
        # 自己写的代码
        if neighbors_ts[right] < cut_time:  # 如果邻居节点小于当前时间; 是所有时间都小于curr_time的情况
            return neighbors_idx[:right + 1], neighbors_e_idx[:right + 1], neighbors_ts[:right + 1]
        elif neighbors_ts[left] >= cut_time:
            return np.array([]), np.array([]), np.array([])
        else:
            return neighbors_idx[:left + 1], neighbors_e_idx[:left + 1], neighbors_ts[:left + 1]

        '''
        源代码：
            
        if neighbors_ts[right] < cut_time:  # 如果邻居节点小于当前时间;
            return neighbors_idx[:right], neighbors_e_idx[:right], neighbors_ts[:right]
        else:
            return neighbors_idx[:left], neighbors_e_idx[:left], neighbors_ts[:left]
        '''

    def get_temporal_neighbor(self, src_idx_l, cut_time_l, num_neighbors=20):
        """
        Params
        ------
        src_idx_l: List[int]
        cut_time_l: List[float],
        num_neighbors: int
        """
        assert(len(src_idx_l) == len(cut_time_l))
        
        out_ngh_node_batch = np.zeros((len(src_idx_l), num_neighbors)).astype(np.int32)  # 节点和对应到的邻居节点
        out_ngh_t_batch = np.zeros((len(src_idx_l), num_neighbors)).astype(np.float32)  # time和对应邻居节点
        out_ngh_eidx_batch = np.zeros((len(src_idx_l), num_neighbors)).astype(np.int32)  # idx索引和对应邻居节点
        
        for i, (src_idx, cut_time) in enumerate(zip(src_idx_l, cut_time_l)):  # src, time
            ngh_idx, ngh_eidx, ngh_ts = self.find_before(src_idx, cut_time)  # 找到小于时间的邻居节点

            if len(ngh_idx) > 0:
                if self.uniform:
                    sampled_idx = np.random.randint(0, len(ngh_idx), num_neighbors)  # 抽样选取
                    
                    out_ngh_node_batch[i, :] = ngh_idx[sampled_idx]
                    out_ngh_t_batch[i, :] = ngh_ts[sampled_idx]
                    out_ngh_eidx_batch[i, :] = ngh_eidx[sampled_idx]
                    
                    # resort based on time
                    pos = out_ngh_t_batch[i, :].argsort()
                    out_ngh_node_batch[i, :] = out_ngh_node_batch[i, :][pos]
                    out_ngh_t_batch[i, :] = out_ngh_t_batch[i, :][pos]
                    out_ngh_eidx_batch[i, :] = out_ngh_eidx_batch[i, :][pos]
                else:
                    ngh_ts = ngh_ts[-num_neighbors:]  # 直接选择最近时间的20个; !!! 源码ngh_ts[:num_neighbors]
                    ngh_idx = ngh_idx[-num_neighbors:]  # 节点索引
                    ngh_eidx = ngh_eidx[-num_neighbors:]  # 边的索引
                    
                    assert(len(ngh_idx) <= num_neighbors)
                    assert(len(ngh_ts) <= num_neighbors)
                    assert(len(ngh_eidx) <= num_neighbors)
                    # 训练数据，
                    out_ngh_node_batch[i, num_neighbors - len(ngh_idx):] = ngh_idx  # 节点的邻居节点，从后填充，前面补零
                    out_ngh_t_batch[i, num_neighbors - len(ngh_ts):] = ngh_ts
                    out_ngh_eidx_batch[i,  num_neighbors - len(ngh_eidx):] = ngh_eidx
                    
        return out_ngh_node_batch, out_ngh_eidx_batch, out_ngh_t_batch

    def find_k_hop(self, k, src_idx_l, cut_time_l, num_neighbors=20):
        """Sampling the k-hop sub graph
        """
        x, y, z = self.get_temporal_neighbor(src_idx_l, cut_time_l, num_neighbors)
        node_records = [x]
        eidx_records = [y]
        t_records = [z]
        for _ in range(k -1):
            ngn_node_est, ngh_t_est = node_records[-1], t_records[-1] # [N, *([num_neighbors] * (k - 1))]
            orig_shape = ngn_node_est.shape
            ngn_node_est = ngn_node_est.flatten()
            ngn_t_est = ngh_t_est.flatten()
            out_ngh_node_batch, out_ngh_eidx_batch, out_ngh_t_batch = self.get_temporal_neighbor(ngn_node_est, ngn_t_est, num_neighbors)
            out_ngh_node_batch = out_ngh_node_batch.reshape(*orig_shape, num_neighbors) # [N, *([num_neighbors] * k)]
            out_ngh_eidx_batch = out_ngh_eidx_batch.reshape(*orig_shape, num_neighbors)
            out_ngh_t_batch = out_ngh_t_batch.reshape(*orig_shape, num_neighbors)

            node_records.append(out_ngh_node_batch)
            eidx_records.append(out_ngh_eidx_batch)
            t_records.append(out_ngh_t_batch)
        return node_records, eidx_records, t_records

            

