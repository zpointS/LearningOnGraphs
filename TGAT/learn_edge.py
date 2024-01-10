"""Unified interface to all dynamic graph model experiments"""
import math
import logging
import time
import random
import sys
import argparse

import torch
import pandas as pd
import numpy as np
#import numba

from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

from module import TGAN
from graph import NeighborFinder
from utils import EarlyStopMonitor, RandEdgeSampler

### Argument and global variables
parser = argparse.ArgumentParser('Interface for TGAT experiments on link predictions')
parser.add_argument('-d', '--data', type=str, help='data sources to use, try wikipedia or reddit', default='wikipedia')
parser.add_argument('--bs', type=int, default=200, help='batch_size')
parser.add_argument('--prefix', type=str, default='', help='prefix to name the checkpoints')
parser.add_argument('--n_degree', type=int, default=20, help='number of neighbors to sample')
parser.add_argument('--n_head', type=int, default=2, help='number of heads used in attention layer')
parser.add_argument('--n_epoch', type=int, default=50, help='number of epochs')
parser.add_argument('--n_layer', type=int, default=2, help='number of network layers')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--drop_out', type=float, default=0.1, help='dropout probability')
parser.add_argument('--gpu', type=int, default=0, help='idx for the gpu to use')
parser.add_argument('--node_dim', type=int, default=100, help='Dimentions of the node embedding')
parser.add_argument('--time_dim', type=int, default=100, help='Dimentions of the time embedding')
parser.add_argument('--agg_method', type=str, choices=['attn', 'lstm', 'mean'], help='local aggregation method', default='attn')
parser.add_argument('--attn_mode', type=str, choices=['prod', 'map'], default='prod', help='use dot product attention or mapping based')
parser.add_argument('--time', type=str, choices=['time', 'pos', 'empty'], help='how to use time information', default='time')
parser.add_argument('--uniform', action='store_true', help='take uniform sampling from temporal neighbors')

try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)

'''
'''
# args.data = 'example'

BATCH_SIZE = args.bs
NUM_NEIGHBORS = args.n_degree
NUM_NEG = 1
NUM_EPOCH = args.n_epoch
NUM_HEADS = args.n_head
DROP_OUT = args.drop_out
GPU = args.gpu
UNIFORM = args.uniform
# NEW_NODE = args.new_node
USE_TIME = args.time
AGG_METHOD = args.agg_method
ATTN_MODE = args.attn_mode
SEQ_LEN = NUM_NEIGHBORS
DATA = args.data
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr
NODE_DIM = args.node_dim
TIME_DIM = args.time_dim


MODEL_SAVE_PATH = f'./saved_models/{args.prefix}-{args.agg_method}-{args.attn_mode}-{args.data}.pth'
get_checkpoint_path = lambda epoch: f'./saved_checkpoints/{args.prefix}-{args.agg_method}-{args.attn_mode}-{args.data}-{epoch}.pth'

### set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('log/{}.log'.format(str(time.time())))
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.WARN)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info(args)


def eval_one_epoch(hint, tgan, sampler, src, dst, ts, label):
    val_acc, val_ap, val_f1, val_auc = [], [], [], []
    with torch.no_grad():
        tgan = tgan.eval()
        TEST_BATCH_SIZE=30
        num_test_instance = len(src)
        num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)
        for k in range(num_test_batch):
            # percent = 100 * k / num_test_batch
            # if k % int(0.2 * num_test_batch) == 0:
            #     logger.info('{0} progress: {1:10.4f}'.format(hint, percent))
            s_idx = k * TEST_BATCH_SIZE
            e_idx = min(num_test_instance - 1, s_idx + TEST_BATCH_SIZE)
            src_l_cut = src[s_idx:e_idx]
            dst_l_cut = dst[s_idx:e_idx]
            ts_l_cut = ts[s_idx:e_idx]
            # label_l_cut = label[s_idx:e_idx]

            size = len(src_l_cut)
            src_l_fake, dst_l_fake = sampler.sample(size)

            pos_prob, neg_prob = tgan.contrast(src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, NUM_NEIGHBORS)
            
            pred_score = np.concatenate([(pos_prob).cpu().numpy(), (neg_prob).cpu().numpy()])
            pred_label = pred_score > 0.5
            true_label = np.concatenate([np.ones(size), np.zeros(size)])
            
            val_acc.append((pred_label == true_label).mean())
            val_ap.append(average_precision_score(true_label, pred_score))
            # val_f1.append(f1_score(true_label, pred_label))
            val_auc.append(roc_auc_score(true_label, pred_score))
    return np.mean(val_acc), np.mean(val_ap), np.mean(val_f1), np.mean(val_auc)

### Load data and train val test split
g_df = pd.read_csv('./processed/ml_{}.csv'.format(DATA))  # 编号, user, item, time, label, idx
e_feat = np.load('./processed/ml_{}.npy'.format(DATA))  # 边的特征; 多了一位，是后面0边(不存在边)的特征; 数据中点和边索引都是从1开始的
n_feat = np.load('./processed/ml_{}_node.npy'.format(DATA))  # 点的特征; 多了一位，是后面0节点的特征
# 验证和测试集的时间
val_time, test_time = list(np.quantile(g_df.ts, [0.70, 0.85]))  # numpy==1.16.4

src_l = g_df.u.values  # src
dst_l = g_df.i.values  # dst
e_idx_l = g_df.idx.values
label_l = g_df.label.values
ts_l = g_df.ts.values  # time

max_src_index = src_l.max()
max_idx = max(src_l.max(), dst_l.max())  # 节点最大编号

random.seed(2020)

total_node_set = set(np.unique(np.hstack([g_df.u.values, g_df.i.values])))  # 所有节点集合
num_total_unique_nodes = len(total_node_set)  # 节点数量
# 在验证和测试集中所有节点进行采样，排除在训练集中。选出一些节点，测试这两个节点不在训练集中的，推理任务
mask_node_set = set(random.sample(set(src_l[ts_l > val_time]).union(set(dst_l[ts_l > val_time])), int(0.1 * num_total_unique_nodes)))  # 对满足测试集时间的样本进行采样
mask_src_flag = g_df.u.map(lambda x: x in mask_node_set).values  # 所有数据中，src节点是否在测试采样节点中
mask_dst_flag = g_df.i.map(lambda x: x in mask_node_set).values  # 所有数据中，dst节点是否在测试采样节点中
none_node_flag = (1 - mask_src_flag) * (1 - mask_dst_flag)  # src和dst节点都不在测试采样中
# 训练集节点
valid_train_flag = (ts_l <= val_time) * (none_node_flag > 0)  # 训练集时间 & 不在测试集中的节点
# 选取训练集节点，边，label信息
train_src_l = src_l[valid_train_flag]
train_dst_l = dst_l[valid_train_flag]
train_ts_l = ts_l[valid_train_flag]  # time
train_e_idx_l = e_idx_l[valid_train_flag]  # 索引
train_label_l = label_l[valid_train_flag]  # label

# define the new nodes sets for testing inductiveness of the model
train_node_set = set(train_src_l).union(train_dst_l)  # 训练节点集合; src + dst
assert(len(train_node_set - mask_node_set) == len(train_node_set))
new_node_set = total_node_set - train_node_set  # 不在训练集中的节点

# select validation and test dataset
valid_val_flag = (ts_l <= test_time) * (ts_l > val_time)  # 验证集时间
valid_test_flag = ts_l > test_time  # 测试集时间
# 至少有一个节点不在训练集中的边; 即新节点产生的边, 可以做推理任务
is_new_node_edge = np.array([(a in new_node_set or b in new_node_set) for a, b in zip(src_l, dst_l)])
nn_val_flag = valid_val_flag * is_new_node_edge  # 推理任务中，val涉及到的边
nn_test_flag = valid_test_flag * is_new_node_edge  # 推理任务中，test涉及到的边

# 满足时间关系; 验证集
val_src_l = src_l[valid_val_flag]
val_dst_l = dst_l[valid_val_flag]
val_ts_l = ts_l[valid_val_flag]
val_e_idx_l = e_idx_l[valid_val_flag]
val_label_l = label_l[valid_val_flag]
# 满足时间关系; 测试集
test_src_l = src_l[valid_test_flag]
test_dst_l = dst_l[valid_test_flag]
test_ts_l = ts_l[valid_test_flag]
test_e_idx_l = e_idx_l[valid_test_flag]
test_label_l = label_l[valid_test_flag]
# 产生新边; 验证集
nn_val_src_l = src_l[nn_val_flag]
nn_val_dst_l = dst_l[nn_val_flag]
nn_val_ts_l = ts_l[nn_val_flag]
nn_val_e_idx_l = e_idx_l[nn_val_flag]
nn_val_label_l = label_l[nn_val_flag]
# 产生新边; 测试集
nn_test_src_l = src_l[nn_test_flag]
nn_test_dst_l = dst_l[nn_test_flag]
nn_test_ts_l = ts_l[nn_test_flag]
nn_test_e_idx_l = e_idx_l[nn_test_flag]
nn_test_label_l = label_l[nn_test_flag]

### Initialize the data structure for graph and edge sampling
# build the graph for fast query
# graph only contains the training data (with 10% nodes removal)
adj_list = [[] for _ in range(max_idx + 1)]  # 空的邻接矩阵
for src, dst, eidx, ts in zip(train_src_l, train_dst_l, train_e_idx_l, train_ts_l):  # src, dst, idx, time
    adj_list[src].append((dst, eidx, ts))  # src节点所连接的节点信息
    adj_list[dst].append((src, eidx, ts))  # 无向边
train_ngh_finder = NeighborFinder(adj_list, uniform=UNIFORM)  # 训练集的数据统计

# full graph with all the data for the test and validation purpose; 完整图
full_adj_list = [[] for _ in range(max_idx + 1)]
for src, dst, eidx, ts in zip(src_l, dst_l, e_idx_l, ts_l):  # 所有数据
    full_adj_list[src].append((dst, eidx, ts))
    full_adj_list[dst].append((src, eidx, ts))
full_ngh_finder = NeighborFinder(full_adj_list, uniform=UNIFORM)  # 全图的数据统计
# 随机采样
train_rand_sampler = RandEdgeSampler(train_src_l, train_dst_l)  # 训练集: src, dst
val_rand_sampler = RandEdgeSampler(src_l, dst_l)  # 全量样本
nn_val_rand_sampler = RandEdgeSampler(nn_val_src_l, nn_val_dst_l)  # 验证集新节点产生的边
test_rand_sampler = RandEdgeSampler(src_l, dst_l)  # 全量样本
nn_test_rand_sampler = RandEdgeSampler(nn_test_src_l, nn_test_dst_l)  # 测试集新节点产生的边


### Model initialize
device = torch.device('cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu')
tgan = TGAN(train_ngh_finder, n_feat, e_feat,  # 训练集统计数据，点特征，边特征
            num_layers=NUM_LAYER, use_time=USE_TIME, agg_method=AGG_METHOD, attn_mode=ATTN_MODE,
            seq_len=SEQ_LEN, n_head=NUM_HEADS, drop_out=DROP_OUT, node_dim=NODE_DIM, time_dim=TIME_DIM)
optimizer = torch.optim.Adam(tgan.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.BCELoss()
tgan = tgan.to(device)

num_instance = len(train_src_l)  # 训练集的数量
num_batch = math.ceil(num_instance / BATCH_SIZE)  # batch数量

logger.info('num of training instances: {}'.format(num_instance))
logger.info('num of batches per epoch: {}'.format(num_batch))
idx_list = np.arange(num_instance)  # 对训练集编号
np.random.shuffle(idx_list)

early_stopper = EarlyStopMonitor()  # early stop
for epoch in range(NUM_EPOCH):
    # Training 
    # training use only training graph
    tgan.ngh_finder = train_ngh_finder  # 训练数据
    acc, ap, f1, auc, m_loss = [], [], [], [], []
    np.random.shuffle(idx_list)  # 再次打乱
    logger.info('start {} epoch'.format(epoch))
    for k in range(num_batch):
        # percent = 100 * k / num_batch
        # if k % int(0.2 * num_batch) == 0:
        #     logger.info('progress: {0:10.4f}'.format(percent))
        # 筛选训练集
        s_idx = k * BATCH_SIZE  # 开始索引
        e_idx = min(num_instance - 1, s_idx + BATCH_SIZE)  # end index
        src_l_cut, dst_l_cut = train_src_l[s_idx:e_idx], train_dst_l[s_idx:e_idx]  # 选择训练集，起始和终止节点
        ts_l_cut = train_ts_l[s_idx:e_idx]  # time
        label_l_cut = train_label_l[s_idx:e_idx]  # label
        size = len(src_l_cut)
        src_l_fake, dst_l_fake = train_rand_sampler.sample(size)  # 负采样
        
        with torch.no_grad():
            pos_label = torch.ones(size, dtype=torch.float, device=device)  # label不更新
            neg_label = torch.zeros(size, dtype=torch.float, device=device)
        
        optimizer.zero_grad()
        tgan = tgan.train()
        pos_prob, neg_prob = tgan.contrast(src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, NUM_NEIGHBORS)  # 正负样本
    
        loss = criterion(pos_prob, pos_label)  # loss
        loss += criterion(neg_prob, neg_label)
        
        loss.backward()
        optimizer.step()
        # get training results
        with torch.no_grad():
            tgan = tgan.eval()
            pred_score = np.concatenate([(pos_prob).cpu().detach().numpy(), (neg_prob).cpu().detach().numpy()])
            pred_label = pred_score > 0.5
            true_label = np.concatenate([np.ones(size), np.zeros(size)])
            acc.append((pred_label == true_label).mean())
            ap.append(average_precision_score(true_label, pred_score))
            # f1.append(f1_score(true_label, pred_label))
            m_loss.append(loss.item())
            auc.append(roc_auc_score(true_label, pred_score))

    # validation phase use all information
    tgan.ngh_finder = full_ngh_finder
    val_acc, val_ap, val_f1, val_auc = eval_one_epoch('val for old nodes', tgan, val_rand_sampler, val_src_l, 
    val_dst_l, val_ts_l, val_label_l)

    nn_val_acc, nn_val_ap, nn_val_f1, nn_val_auc = eval_one_epoch('val for new nodes', tgan, val_rand_sampler, nn_val_src_l, 
    nn_val_dst_l, nn_val_ts_l, nn_val_label_l)
        
    logger.info('epoch: {}:'.format(epoch))
    logger.info('Epoch mean loss: {}'.format(np.mean(m_loss)))
    logger.info('train acc: {}, val acc: {}, new node val acc: {}'.format(np.mean(acc), val_acc, nn_val_acc))
    logger.info('train auc: {}, val auc: {}, new node val auc: {}'.format(np.mean(auc), val_auc, nn_val_auc))
    logger.info('train ap: {}, val ap: {}, new node val ap: {}'.format(np.mean(ap), val_ap, nn_val_ap))
    # logger.info('train f1: {}, val f1: {}, new node val f1: {}'.format(np.mean(f1), val_f1, nn_val_f1))

    if early_stopper.early_stop_check(val_ap):
        logger.info('No improvment over {} epochs, stop training'.format(early_stopper.max_round))
        logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
        best_model_path = get_checkpoint_path(early_stopper.best_epoch)
        tgan.load_state_dict(torch.load(best_model_path))
        logger.info(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
        tgan.eval()
        break
    else:
        torch.save(tgan.state_dict(), get_checkpoint_path(epoch))


# testing phase use all information
tgan.ngh_finder = full_ngh_finder
test_acc, test_ap, test_f1, test_auc = eval_one_epoch('test for old nodes', tgan, test_rand_sampler, test_src_l, 
test_dst_l, test_ts_l, test_label_l)

nn_test_acc, nn_test_ap, nn_test_f1, nn_test_auc = eval_one_epoch('test for new nodes', tgan, nn_test_rand_sampler, nn_test_src_l, 
nn_test_dst_l, nn_test_ts_l, nn_test_label_l)

logger.info('Test statistics: Old nodes -- acc: {}, auc: {}, ap: {}'.format(test_acc, test_auc, test_ap))
logger.info('Test statistics: New nodes -- acc: {}, auc: {}, ap: {}'.format(nn_test_acc, nn_test_auc, nn_test_ap))

logger.info('Saving TGAN model')
torch.save(tgan.state_dict(), MODEL_SAVE_PATH)
logger.info('TGAN models saved')

 




