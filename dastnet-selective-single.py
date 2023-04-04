# -*- coding: utf-8 -*-
# @Time    : 2023/4/4 14:58
# @Author  : 银尘
# @FileName: dastnet-selective-single.py
# @Software: PyCharm
# @Email   : liwudi@liwudi.fun
# @Info    : why create this file
import argparse
import ast
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PaperCrawlerUtil.common_util import *
from PaperCrawlerUtil.constant import *
from PaperCrawlerUtil.crawler_util import *
from dgl.nn import GATConv
from dtaidistance import dtw
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.utils.tensorboard import SummaryWriter
from model import *
from funcs import *
from params import *
from util import *
from PaperCrawlerUtil.research_util import *
import argparse
import torch
import copy
import time
import os
import numpy as np
import torch.optim as optim

from utils.data import MyDataLoader
from utils.funcs import load_data, load_all_adj, StandardScaler
from utils.funcs import masked_loss
from utils.vec import generate_vector
from model import DASTNet, Domain_classifier_DG
from PaperCrawlerUtil.common_util import *
from PaperCrawlerUtil.research_util import *
import ast

basic_config(logs_style=LOG_STYLE_ALL)
p_bar = process_bar(final_prompt="初始化准备完成", unit="part")
args = params()
p_bar.process(0, 1, 5)
if args.c != "default":
    c = ast.literal_eval(args.c)
    record = ResearchRecord(**c)
    record_id = record.insert(__file__, get_timestamp(), args.__str__())
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
gpu_available = torch.cuda.is_available()
if gpu_available:
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')
dataname = args.dataname
scity = args.scity
scity2 = args.scity2
tcity = args.tcity
datatype = args.datatype
num_epochs = args.num_epochs
num_tuine_epochs = args.num_tuine_epochs
start_time = time.time()
log("Running CrossTReS, from %s and %s to %s, %s %s experiments, with %d days of data, on %s model" % \
    (scity, scity2, tcity, dataname, datatype, args.data_amount, args.model))
p_bar.process(1, 1, 5)
# Load spatio temporal data
# (8784, 21, 20)
# 8784 = 366 * 24
target_data = np.load("../data/%s/%s%s_%s.npy" % (tcity, dataname, tcity, datatype))
# (21, 20) 经纬度分割
lng_target, lat_target = target_data.shape[1], target_data.shape[2]
# numpy.sum()，求和某一维度或者维度为none时，求和所有，减掉一个维度
# 此处，target_data (8784, 21, 20) -> (21, 20)
# 然后，通过对于每个元素判断是否大于0， 转成Bool向量
mask_target = target_data.sum(0) > 0
# reshape （21， 20） -》 （1， 21， 20）
th_mask_target = torch.Tensor(mask_target.reshape(1, lng_target, lat_target)).to(device)
log("%d valid regions in target" % np.sum(mask_target))
# (（21， 20）-> 420, （21， 20）-> 420)
target_emb_label = masked_percentile_label(target_data.sum(0).reshape(-1), mask_target.reshape(-1))



# (8784, 20, 23)
source_data = np.load("../data/%s/%s%s_%s.npy" % (scity, dataname, scity, datatype))
log(source_data.shape)
# (20, 23)
lng_source, lat_source = source_data.shape[1], source_data.shape[2]
mask_source = source_data.sum(0) > 0
# mask -> th_mask = (20, 23) -> (1, 20, 23)
th_mask_source = torch.Tensor(mask_source.reshape(1, lng_source, lat_source)).to(device)
log("%d valid regions in source" % np.sum(mask_source))

source_data2 = np.load("../data/%s/%s%s_%s.npy" % (scity2, dataname, scity2, datatype))
log(source_data2.shape)
lng_source2, lat_source2 = source_data2.shape[1], source_data2.shape[2]
mask_source2 = source_data2.sum(0) > 0
th_mask_source2 = torch.Tensor(mask_source2.reshape(1, lng_source2, lat_source2)).to(device)
log("%d valid regions in source" % np.sum(mask_source2))

p_bar.process(2, 1, 5)
# 按照百分比分配标签
source_emb_label = masked_percentile_label(source_data.sum(0).reshape(-1), mask_source.reshape(-1))

lag = [-6, -5, -4, -3, -2, -1]
source_data, smax, smin = min_max_normalize(source_data)
target_data, max_val, min_val = min_max_normalize(target_data)

source_emb_label2 = masked_percentile_label(source_data2.sum(0).reshape(-1), mask_source2.reshape(-1))
source_data2, smax2, smin2 = min_max_normalize(source_data2)


# [(5898, 6, 20, 23), (5898, 1, 20, 23), (1440, 6, 20, 23), (1440, 1, 20, 23), (1440, 6, 20, 23), (1440, 1, 20, 23)]
# 第一维是数量，第二维是每条数据中的数量
source_train_x, source_train_y, source_val_x, source_val_y, source_test_x, source_test_y = split_x_y(source_data, lag)
source_train_x2, source_train_y2, source_val_x2, source_val_y2, source_test_x2, source_test_y2 = split_x_y(source_data2,
                                                                                                           lag)
# we concatenate all source data
# (8778, 6, 20, 23)
source_x = np.concatenate([source_train_x, source_val_x, source_test_x], axis=0)
# (8778, 1, 20, 23)
source_y = np.concatenate([source_train_y, source_val_y, source_test_y], axis=0)
source_x2 = np.concatenate([source_train_x2, source_val_x2, source_test_x2], axis=0)
source_y2 = np.concatenate([source_train_y2, source_val_y2, source_test_y2], axis=0)
target_train_x, target_train_y, target_val_x, target_val_y, target_test_x, target_test_y = split_x_y(target_data, lag)
p_bar.process(3, 1, 5)


if args.data_amount != 0:
    # 负号表示从倒数方向数，
    # i.e.
    # a = [12, 3, 4, 5, 6, 7, 8]
    # c, d = a[-2:], a[:-2]
    # print(c)
    # print(d)
    # [7, 8]
    # [12, 3, 4, 5, 6]
    target_train_x = target_train_x[-args.data_amount * 24:, :, :, :]
    target_train_y = target_train_y[-args.data_amount * 24:, :, :, :]
log("Source split to: x %s, y %s" % (str(source_x.shape), str(source_y.shape)))
# log("val_x %s, val_y %s" % (str(source_val_x.shape), str(source_val_y.shape)))
# log("test_x %s, test_y %s" % (str(source_test_x.shape), str(source_test_y.shape)))
log("Source2 split to: x %s, y %s" % (str(source_x2.shape), str(source_y2.shape)))
log("Target split to: train_x %s, train_y %s" % (str(target_train_x.shape), str(target_train_y.shape)))
log("val_x %s, val_y %s" % (str(target_val_x.shape), str(target_val_y.shape)))
log("test_x %s, test_y %s" % (str(target_test_x.shape), str(target_test_y.shape)))


# 这些代码 numpy -> Tensor -> TensorDataset -> DataLoader
target_train_dataset = TensorDataset(torch.Tensor(target_train_x), torch.Tensor(target_train_y))
target_val_dataset = TensorDataset(torch.Tensor(target_val_x), torch.Tensor(target_val_y))
target_test_dataset = TensorDataset(torch.Tensor(target_test_x), torch.Tensor(target_test_y))
target_train_loader = DataLoader(target_train_dataset, batch_size=args.batch_size, shuffle=True)
target_val_loader = DataLoader(target_val_dataset, batch_size=args.batch_size)
target_test_loader = DataLoader(target_test_dataset, batch_size=args.batch_size)
source_test_dataset = TensorDataset(torch.Tensor(source_test_x), torch.Tensor(source_test_y))
source_test_loader = DataLoader(source_test_dataset, batch_size=args.batch_size)
source_dataset = TensorDataset(torch.Tensor(source_x), torch.Tensor(source_y))
source_loader = DataLoader(source_dataset, batch_size=args.batch_size, shuffle=True)
source_test_dataset2 = TensorDataset(torch.Tensor(source_test_x2), torch.Tensor(source_test_y2))
source_test_loader2 = DataLoader(source_test_dataset2, batch_size=args.batch_size)
source_dataset2 = TensorDataset(torch.Tensor(source_x2), torch.Tensor(source_y2))
source_loader2 = DataLoader(source_dataset2, batch_size=args.batch_size, shuffle=True)


# Load auxiliary data: poi data
# (20, 23, 14)
source_poi = np.load("../data/%s/%s_poi.npy" % (scity, scity))
source_poi2 = np.load("../data/%s/%s_poi.npy" % (scity2, scity2))
target_poi = np.load("../data/%s/%s_poi.npy" % (tcity, tcity))
# (460, 14)
source_poi = source_poi.reshape(lng_source * lat_source, -1)  # regions * classes
source_poi2 = source_poi2.reshape(lng_source2 * lat_source2, -1)  # regions * classes
target_poi = target_poi.reshape(lng_target * lat_target, -1)  # regions * classes
transform = TfidfTransformer()
# 规范正则化到（0，1）
source_norm_poi = np.array(transform.fit_transform(source_poi).todense())
transform = TfidfTransformer()
# 规范正则化到（0，1）
source_norm_poi2 = np.array(transform.fit_transform(source_poi2).todense())
transform = TfidfTransformer()
target_norm_poi = np.array(transform.fit_transform(target_poi).todense())



# Build graphs
# add_self_loop 增加一个自循环，对角线的值=1
source_prox_adj = add_self_loop(build_prox_graph(lng_source, lat_source))
source_prox_adj2 = add_self_loop(build_prox_graph(lng_source2, lat_source2))
target_prox_adj = add_self_loop(build_prox_graph(lng_target, lat_target))
source_road_adj = add_self_loop(build_road_graph(scity, lng_source, lat_source))
source_road_adj2 = add_self_loop(build_road_graph(scity2, lng_source2, lat_source2))
target_road_adj = add_self_loop(build_road_graph(tcity, lng_target, lat_target))
source_poi_adj, source_poi_cos = build_poi_graph(source_norm_poi, args.topk)
source_poi_adj2, source_poi_cos2 = build_poi_graph(source_norm_poi2, args.topk)
target_poi_adj, target_poi_cos = build_poi_graph(target_norm_poi, args.topk)
source_poi_adj = add_self_loop(source_poi_adj)
source_poi_adj2 = add_self_loop(source_poi_adj2)
target_poi_adj = add_self_loop(target_poi_adj)
source_s_adj, source_d_adj, source_od_adj = build_source_dest_graph(scity, dataname, lng_source, lat_source, args.topk)
source_s_adj2, source_d_adj2, source_od_adj2 = build_source_dest_graph(scity2, dataname, lng_source2, lat_source2,
                                                                       args.topk)
target_s_adj, target_d_adj, target_od_adj = build_source_dest_graph(tcity, dataname, lng_target, lat_target, args.topk)
source_s_adj = add_self_loop(source_s_adj)
source_s_adj2 = add_self_loop(source_s_adj2)
source_t_adj = add_self_loop(source_d_adj)
source_t_adj2 = add_self_loop(source_d_adj2)
source_od_adj = add_self_loop(source_od_adj)
source_od_adj2 = add_self_loop(source_od_adj2)
target_s_adj = add_self_loop(target_s_adj)
target_t_adj = add_self_loop(target_d_adj)
target_od_adj = add_self_loop(target_od_adj)
log("Source graphs: ")
log("prox_adj: %d nodes, %d edges" % (source_prox_adj.shape[0], np.sum(source_prox_adj)))
log("road adj: %d nodes, %d edges" % (source_road_adj.shape[0], np.sum(source_road_adj > 0)))
log("poi_adj, %d nodes, %d edges" % (source_poi_adj.shape[0], np.sum(source_poi_adj > 0)))
log("s_adj, %d nodes, %d edges" % (source_s_adj.shape[0], np.sum(source_s_adj > 0)))
log("d_adj, %d nodes, %d edges" % (source_d_adj.shape[0], np.sum(source_d_adj > 0)))
log()
log("Source2 graphs: ")
log("prox_adj: %d nodes, %d edges" % (source_prox_adj2.shape[0], np.sum(source_prox_adj2)))
log("road adj: %d nodes, %d edges" % (source_road_adj2.shape[0], np.sum(source_road_adj2 > 0)))
log("poi_adj, %d nodes, %d edges" % (source_poi_adj2.shape[0], np.sum(source_poi_adj2 > 0)))
log("s_adj, %d nodes, %d edges" % (source_s_adj2.shape[0], np.sum(source_s_adj2 > 0)))
log("d_adj, %d nodes, %d edges" % (source_d_adj2.shape[0], np.sum(source_d_adj2 > 0)))
log()
log("Target graphs:")
log("prox_adj: %d nodes, %d edges" % (target_prox_adj.shape[0], np.sum(target_prox_adj)))
log("road adj: %d nodes, %d edges" % (target_road_adj.shape[0], np.sum(target_road_adj > 0)))
log("poi_adj, %d nodes, %d edges" % (target_poi_adj.shape[0], np.sum(target_poi_adj > 0)))
log("s_adj, %d nodes, %d edges" % (target_s_adj.shape[0], np.sum(target_s_adj > 0)))
log("d_adj, %d nodes, %d edges" % (target_d_adj.shape[0], np.sum(target_d_adj > 0)))
log()
source_graphs = adjs_to_graphs([source_prox_adj, source_road_adj, source_poi_adj, source_s_adj, source_d_adj])
source_graphs2 = adjs_to_graphs([source_prox_adj2, source_road_adj2, source_poi_adj2, source_s_adj2, source_d_adj2])
target_graphs = adjs_to_graphs([target_prox_adj, target_road_adj, target_poi_adj, target_s_adj, target_d_adj])
for i in range(len(source_graphs)):
    source_graphs[i] = source_graphs[i].to(device)
    source_graphs2[i] = source_graphs2[i].to(device)
    target_graphs[i] = target_graphs[i].to(device)


source_edges, source_edge_labels = graphs_to_edge_labels(source_graphs)
source_edges2, source_edge_labels2 = graphs_to_edge_labels(source_graphs2)
target_edges, target_edge_labels = graphs_to_edge_labels(target_graphs)
p_bar.process(4, 1, 5)





# 评分模型
class Scoring(nn.Module):
    def __init__(self, emb_dim, source_mask, target_mask):
        super().__init__()
        self.emb_dim = emb_dim
        self.score = nn.Sequential(nn.Linear(self.emb_dim, self.emb_dim // 2),
                                   nn.ReLU(inplace=True),
                                   nn.Linear(self.emb_dim // 2, self.emb_dim // 2))
        self.source_mask = source_mask
        self.target_mask = target_mask

    def forward(self, source_emb, target_emb):
        """
        求源城市评分
        注意这里求评分，是source的每一个区域对于目标城市整体
        换句话说，是形参2的每一个区域，对于形参3整体
        :param target_mask:
        :param source_mask:
        :param source_emb:
        :param target_emb:
        :return:
        """
        # target_context = tanh(self.score(target_emb[bool mask]).mean(0))
        # 对于横向的进行求平均 460*64 -> 460*32 -> 207*32 -> 纵向求平均 1*32 代表所有目标城市
        target_context = torch.tanh(
            torch.quantile(
                self.score(target_emb[self.target_mask.view(-1).bool()]),
                torch.Tensor([0.1, 0.25, 0.5, 0.75, 0.9]).to(device), dim=0).mean(0)
        )
        source_trans_emb = self.score(source_emb)
        # 460*32 * 1*32 = 462*32, 这里乘法表示1*32列表去乘460*32的每一行，逐元素
        # i.e.
        # tensor([[2, 2, 2],
        #         [1, 2, 2],
        #         [2, 2, 1]])
        # tensor([[2, 2, 2]])
        # tensor([[4, 4, 4],
        #         [2, 4, 4],
        #         [4, 4, 2]])
        source_score = (source_trans_emb * target_context).sum(1)
        # the following lines modify inner product similarity to cosine similarity
        # target_norm = target_context.pow(2).sum().pow(1/2)
        # source_norm = source_trans_emb.pow(2).sum(1).pow(1/2)
        # source_score /= source_norm
        # source_score /= target_norm
        # log(source_score)
        return F.relu(torch.tanh(source_score))[self.source_mask.view(-1).bool()]




mmd = MMD_loss()





num_gat_layers = 2
in_dim = 14
hidden_dim = 64
emb_dim = 64
num_heads = 2
mmd_w = args.mmd_w
et_w = args.et_w
ma_param = args.ma_coef

mvgat = MVGAT(len(source_graphs), num_gat_layers, in_dim, hidden_dim, emb_dim, num_heads, True).to(device)
fusion = FusionModule(len(source_graphs), emb_dim, 0.8).to(device)
scoring = Scoring(emb_dim, th_mask_source, th_mask_target).to(device)
edge_disc = EdgeTypeDiscriminator(len(source_graphs), emb_dim).to(device)
mmd = MMD_loss()

emb_param_list = list(mvgat.parameters()) + list(fusion.parameters()) + list(edge_disc.parameters())
emb_optimizer = optim.Adam(emb_param_list, lr=args.learning_rate, weight_decay=args.weight_decay)
# 元学习部分
meta_optimizer = optim.Adam(scoring.parameters(), lr=args.outerlr, weight_decay=args.weight_decay)
best_val_rmse = 999
best_test_rmse = 999
best_test_mae = 999
p_bar.process(5, 1, 5)


class DomainClassify(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.dc = nn.Sequential(nn.Linear(self.emb_dim, self.emb_dim // 2),
                                nn.ReLU(inplace=True),
                                nn.Linear(self.emb_dim // 2, self.emb_dim // 2),
                                nn.Linear(self.emb_dim // 2, 2))

    def forward(self, feature):
        res = torch.sigmoid(self.dc(feature))
        return res


def forward_emb(graphs_, in_feat_, od_adj_, poi_cos_):
    """
    1. 图卷积提取图特征 mvgat
    2. 融合多图特征 fusion
    3. 对于多图中的s，d，poi进行预测，并计算损失函数
    :param graphs_:
    :param in_feat_:
    :param od_adj_:
    :param poi_cos_:
    :return:
    """
    # 图注意，注意这里用了小写，指的是forward方法
    views = mvgat(graphs_, torch.Tensor(in_feat_).to(device))
    fused_emb, embs = fusion(views)
    # embs嵌入是5个图，以下找出start，destination， poi图
    s_emb = embs[-2]
    d_emb = embs[-1]
    poi_emb = embs[-3]
    # start和destination相乘求出记录预测s和d
    recons_sd = torch.matmul(s_emb, d_emb.transpose(0, 1))
    # 注意dim维度0和1分别求s和d
    pred_d = torch.log(torch.softmax(recons_sd, dim=1) + 1e-5)
    loss_d = (torch.Tensor(od_adj_).to(device) * pred_d).mean()
    pred_s = torch.log(torch.softmax(recons_sd, dim=0) + 1e-5)
    loss_s = (torch.Tensor(od_adj_).to(device) * pred_s).mean()
    # poi预测求差，loss
    poi_sim = torch.matmul(poi_emb, poi_emb.transpose(0, 1))
    loss_poi = ((poi_sim - torch.Tensor(poi_cos_).to(device)) ** 2).mean()
    loss = -loss_s - loss_d + loss_poi

    return loss, fused_emb, embs


def train_emb_epoch2():
    """
    训练图网络-特征网络，融合网络，边类型分类器
    1. 通过forward_emb融合特征，计算损失，
    2. 抽样边，标签，训练边缘分类器，抽样计算MMD误差
    3. 反向传播计算
    emb_param_list = list(mvgat.parameters()) + list(fusion.parameters()) + list(edge_disc.parameters())
    emb_optimizer = optim.Adam(emb_param_list, lr=args.learning_rate, weight_decay=args.weight_decay)
    训练特征网络 mvgat，fusion，边缘分类器，节点MMD，在训练的同时，对于mvgat和fusion的特征进行指导，特征重新对齐分布
    :return:
    """
    # loss， 460*64， 5*460*64
    loss_source, fused_emb_s, embs_s = forward_emb(source_graphs, source_norm_poi, source_od_adj, source_poi_cos)
    loss_target, fused_emb_t, embs_t = forward_emb(target_graphs, target_norm_poi, target_od_adj, target_poi_cos)

    loss_emb = loss_source + loss_target
    # compute domain adaptation loss
    # 随机抽样128个，计算最大平均误差
    source_ids = np.random.randint(0, np.sum(mask_source), size=(128,))
    source_ids2 = np.random.randint(0, np.sum(mask_source2), size=(128,))
    target_ids = np.random.randint(0, np.sum(mask_target), size=(128,))
    # source1 & target
    mmd_loss = mmd(fused_emb_s[th_mask_source.view(-1).bool()][source_ids, :],
                   fused_emb_t[th_mask_target.view(-1).bool()][target_ids, :])
    mmd_losses = mmd_loss
    # 随机抽样边256
    source_batch_edges = np.random.randint(0, len(source_edges), size=(256,))
    source_batch_edges2 = np.random.randint(0, len(source_edges2), size=(256,))
    target_batch_edges = np.random.randint(0, len(target_edges), size=(256,))
    source_batch_src = torch.Tensor(source_edges[source_batch_edges, 0]).long()
    source_batch_dst = torch.Tensor(source_edges[source_batch_edges, 1]).long()
    source_emb_src = fused_emb_s[source_batch_src, :]
    source_emb_dst = fused_emb_s[source_batch_dst, :]
    source_batch_src2 = torch.Tensor(source_edges2[source_batch_edges2, 0]).long()
    source_batch_dst2 = torch.Tensor(source_edges2[source_batch_edges2, 1]).long()
    target_batch_src = torch.Tensor(target_edges[target_batch_edges, 0]).long()
    target_batch_dst = torch.Tensor(target_edges[target_batch_edges, 1]).long()
    target_emb_src = fused_emb_t[target_batch_src, :]
    target_emb_dst = fused_emb_t[target_batch_dst, :]
    # 源城市目的城市使用同样的边分类器
    pred_source = edge_disc.forward(source_emb_src, source_emb_dst)
    pred_target = edge_disc.forward(target_emb_src, target_emb_dst)
    source_batch_labels = torch.Tensor(source_edge_labels[source_batch_edges]).to(device)
    target_batch_labels = torch.Tensor(target_edge_labels[target_batch_edges]).to(device)
    # -（label*log(sigmod(pred)+0.000001)) + (1-label)*log(1-sigmod+0.000001) sum mean
    loss_et_source = -((source_batch_labels * torch.log(torch.sigmoid(pred_source) + 1e-6)) + (
            1 - source_batch_labels) * torch.log(1 - torch.sigmoid(pred_source) + 1e-6)).sum(1).mean()
    loss_et_target = -((target_batch_labels * torch.log(torch.sigmoid(pred_target) + 1e-6)) + (
            1 - target_batch_labels) * torch.log(1 - torch.sigmoid(pred_target) + 1e-6)).sum(1).mean()
    loss_et = loss_et_source + loss_et_target

    emb_optimizer.zero_grad()
    # 公式11
    loss = loss_emb + mmd_w * mmd_losses + et_w * loss_et
    loss.backward()
    emb_optimizer.step()
    return loss_emb.item(), mmd_losses.item(), loss_et.item()


emb_losses = []
mmd_losses = []
edge_losses = []
pretrain_emb_epoch = 80
# 预训练图数据嵌入，边类型分类，节点对齐 ——> 获得区域特征
for emb_ep in range(pretrain_emb_epoch):
    loss_emb_, loss_mmd_, loss_et_ = train_emb_epoch2()
    emb_losses.append(loss_emb_)
    mmd_losses.append(loss_mmd_)
    edge_losses.append(loss_et_)
log("[%.2fs]Pretrain embeddings for %d epochs, average emb loss %.4f, mmd loss %.4f, edge loss %.4f" % (
    time.time() - start_time, pretrain_emb_epoch, np.mean(emb_losses), np.mean(mmd_losses), np.mean(edge_losses)))
with torch.no_grad():
    views = mvgat(source_graphs, torch.Tensor(source_norm_poi).to(device))
    fused_emb_s, _ = fusion(views)
    views = mvgat(target_graphs, torch.Tensor(target_norm_poi).to(device))
    fused_emb_t, _ = fusion(views)

emb_s = fused_emb_s.cpu().numpy()[mask_source.reshape(-1)]
emb_t = fused_emb_t.cpu().numpy()[mask_target.reshape(-1)]
logreg = LogisticRegression(max_iter=500)
cvscore_s = cross_validate(logreg, emb_s, source_emb_label)['test_score'].mean()
cvscore_t = cross_validate(logreg, emb_t, target_emb_label)['test_score'].mean()
log("[%.2fs]Pretraining embedding, source cvscore %.4f, target cvscore %.4f" % \
    (time.time() - start_time, cvscore_s, cvscore_t))
log()


def net_fix(source, y, weight, mask, fast_weights, bn_vars, net):
    pred_source = net.functional_forward(vec_pems04, vec_pems07, vec_pems08, source, True, fast_weights, bn_vars, bn_training=True, data_set="4")
    label = y.reshape((pred_source.shape[0], -1, pred_source.shape[2]))
    mask = mask.reshape((1, mask.shape[1] * mask.shape[2], 1))
    fast_loss = torch.abs(pred_source - label)[:, mask.view(-1).bool(),:]
    fast_loss = (fast_loss * weight.view((1, -1, 1))).mean(0).sum()
    a = [(i, torch.autograd.grad(fast_loss, fast_weights[i], create_graph=True, allow_unused=True)) for i in fast_weights.keys()]
    grads = {}
    used_fast_weight = OrderedDict()
    for i in a:
        if i[1][0] is not None:
            grads[i[0]] = i[1][0]
            used_fast_weight[i[0]] = fast_weights[i[0]]

    for name, grad in zip(grads.keys(), grads.values()):
        fast_weights[name] = fast_weights[name] - args.innerlr * grad
    return fast_loss, fast_weights, bn_vars


def meta_train_epoch(s_embs, t_embs, net):
    meta_query_losses = []
    for meta_ep in range(args.outeriter):
        fast_losses = []
        fast_weights, bn_vars = get_weights_bn_vars(net)
        source_weights = scoring(s_embs, t_embs, th_mask_source, th_mask_target)
        # inner loop on source, pre-train with weights
        for meta_it in range(args.sinneriter):
            s_x1, s_y1 = batch_sampler((torch.Tensor(source_train_x), torch.Tensor(source_train_y)),
                                       args.batch_size)
            s_x1 = s_x1.reshape((s_x1.shape[0], s_x1.shape[1], s_x1.shape[2] * s_x1.shape[3]))
            s_y1 = s_y1.reshape((s_y1.shape[0], s_y1.shape[1], s_y1.shape[2] * s_y1.shape[3]))
            s_x1 = s_x1.to(device)
            s_y1 = s_y1.to(device)
            fast_loss, fast_weights, bn_vars = net_fix(s_x1, s_y1, source_weights, th_mask_source, fast_weights, bn_vars, net)
            fast_losses.append(fast_loss.item())

        for meta_it in range(args.tinneriter):
            t_x, t_y = batch_sampler((torch.Tensor(target_train_x), torch.Tensor(target_train_y)), args.batch_size)
            t_x = t_x.reshape((t_x.shape[0], t_x.shape[1], t_x.shape[2] * t_x.shape[3]))
            t_y = t_y.reshape((t_y.shape[0], t_y.shape[1], t_y.shape[2] * t_y.shape[3]))

            t_x = t_x.to(device)
            t_y = t_y.to(device)
            pred_source = net.functional_forward(vec_pems04, vec_pems07, vec_pems08, t_x, True, fast_weights, bn_vars, bn_training=True, data_set="8")
            label = t_y.reshape((pred_source.shape[0], -1, pred_source.shape[2]))
            mask = th_mask_target
            mask = mask.reshape((1, mask.shape[1] * mask.shape[2], 1))
            fast_loss = torch.abs(pred_source - label)[:, mask.view(-1).bool(),:]
            fast_loss = fast_loss.mean(0).sum()
            a = [(i, torch.autograd.grad(fast_loss, fast_weights[i], create_graph=True, allow_unused=True)) for i in fast_weights.keys()]
            grads = {}
            used_fast_weight = OrderedDict()
            for i in a:
                if i[1][0] is not None:
                    grads[i[0]] = i[1][0]
                    used_fast_weight[i[0]] = fast_weights[i[0]]

            for name, grad in zip(grads.keys(), grads.values()):
                fast_weights[name] = fast_weights[name] - args.innerlr * grad

        q_losses = []
        target_iter = max(args.sinneriter, args.tinneriter)
        for k in range(3):
            # query loss
            x_q = None
            y_q = None
            temp_mask = None

            x_q, y_q = batch_sampler((torch.Tensor(target_train_x), torch.Tensor(target_train_y)), args.batch_size)
            temp_mask = th_mask_target
            x_q = x_q.reshape((x_q.shape[0], x_q.shape[1], x_q.shape[2] * x_q.shape[3]))
            y_q = y_q.reshape((y_q.shape[0], y_q.shape[1], y_q.shape[2] * y_q.shape[3]))

            x_q = x_q.to(device)
            y_q = y_q.to(device)
            pred_source = net.functional_forward(vec_pems04, vec_pems07, vec_pems08, x_q, True, fast_weights, bn_vars, bn_training=True, data_set="8")
            label = y_q.reshape((pred_source.shape[0], -1, pred_source.shape[2]))
            mask = temp_mask.reshape((1, temp_mask.shape[1] * temp_mask.shape[2], 1))
            fast_loss = torch.abs(pred_source - label)[:, mask.view(-1).bool(),:]
            fast_loss = fast_loss.mean(0).sum()

            q_losses.append(fast_loss)
        q_loss = torch.stack(q_losses).mean()
        weights_mean = source_weights.mean()
        meta_loss = q_loss + weights_mean * args.weight_reg
        meta_optimizer.zero_grad()
        meta_loss.backward(inputs=list(scoring.parameters()), retain_graph=True)
        torch.nn.utils.clip_grad_norm_(scoring.parameters(), max_norm=2)
        meta_optimizer.step()
        meta_query_losses.append(q_loss.item())
    return np.mean(meta_query_losses)


def get_weight(net, type):
    if type == "fine-tune":
        return None
    for emb_ep in range(5):
        loss_emb_, loss_mmd_, loss_et_ = train_emb_epoch2()
        emb_losses.append(loss_emb_)
        mmd_losses.append(loss_mmd_)
        edge_losses.append(loss_et_)

    with torch.no_grad():
        views = mvgat(source_graphs, torch.Tensor(source_norm_poi).to(device))
        fused_emb_s, _ = fusion(views)
        views = mvgat(target_graphs, torch.Tensor(target_norm_poi).to(device))
        fused_emb_t, _ = fusion(views)

    meta_train_epoch(fused_emb_s, fused_emb_t, net)
    with torch.no_grad():
        source_weights = scoring(fused_emb_s, fused_emb_t, th_mask_source, th_mask_target)
    return source_weights


def select_mask(a):
    if a == 420:
        return dcmask
    elif a == 476:
        return chimask
    elif a == 460:
        return nymask


def train(dur, model, optimizer, total_step, start_step, need_road, train_dataloader,val_dataloader, testdl, type, weight):
    t0 = time.time()
    train_mae, val_mae, train_rmse, val_rmse, train_acc = list(), list(), list(), list(), list()
    train_correct = 0

    model.train()
    if type == 'pretrain':
        domain_classifier.train()

    for i, (feat, label) in enumerate(train_dataloader.get_iterator()):
        mask = select_mask(feat.shape[2])
        Reverse = False
        if i > 0:
            if train_acc[-1] > 0.333333:
                Reverse = True
        p = float(i + start_step) / total_step
        constant = 2. / (1. + np.exp(-10 * p)) - 1

        feat = torch.FloatTensor(feat).to(device)
        label = torch.FloatTensor(label).to(device)
        if torch.sum(scaler.inverse_transform(label)) <= 0.001:
            continue

        optimizer.zero_grad()
        if args.models not in ['DCRNN', 'STGCN', 'HA']:
            if type == 'pretrain':
                pred, shared_pems04_feat, shared_pems07_feat, shared_pems08_feat = model(vec_pems04, vec_pems07,
                                                                                         vec_pems08, feat, False,
                                                                                         need_road)
            elif type == 'fine-tune':
                pred = model(vec_pems04, vec_pems07, vec_pems08, feat, False, need_road)

            pred = pred.transpose(1, 2).reshape((-1, feat.size(2)))
            label = label.reshape((-1, label.size(2)))

            if type == 'pretrain':
                pems04_pred = domain_classifier(shared_pems04_feat, constant, Reverse)
                pems08_pred = domain_classifier(shared_pems08_feat, constant, Reverse)

                pems04_label = 0 * torch.ones(pems04_pred.shape[0]).long().to(device)
                pems08_label = 1 * torch.ones(pems08_pred.shape[0]).long().to(device)

                pems04_pred_label = pems04_pred.max(1, keepdim=True)[1]
                pems04_correct = pems04_pred_label.eq(pems04_label.view_as(pems04_pred_label)).sum()

                pems08_pred_label = pems08_pred.max(1, keepdim=True)[1]
                pems08_correct = pems08_pred_label.eq(pems08_label.view_as(pems08_pred_label)).sum()

                pems04_loss = domain_criterion(pems04_pred, pems04_label)

                pems08_loss = domain_criterion(pems08_pred, pems08_label)

                domain_loss = pems04_loss + pems08_loss

        if type == 'pretrain':
            train_correct = pems04_correct + pems08_correct

        mae_train, rmse_train, mape_train = masked_loss(scaler.inverse_transform(pred), scaler.inverse_transform(label),
                                                        maskp=mask, weight=weight)

        if type == 'pretrain':
            loss = mae_train + args.beta * (args.theta * domain_loss)
        elif type == 'fine-tune':
            loss = mae_train

        loss.backward()
        optimizer.step()

        train_mae.append(mae_train.item())
        train_rmse.append(rmse_train.item())

        if type == 'pretrain':
            train_acc.append(train_correct.item() / 855)
        elif type == 'fine-tune':
            train_acc.append(0)

    if type == 'pretrain':
        domain_classifier.eval()
    model.eval()

    for i, (feat, label) in enumerate(val_dataloader.get_iterator()):
        mask = select_mask(feat.shape[2])
        feat = torch.FloatTensor(feat).to(device)
        label = torch.FloatTensor(label).to(device)
        if torch.sum(scaler.inverse_transform(label)) <= 0.001:
            continue
        pred = model(vec_pems04, vec_pems07, vec_pems08, feat, True, need_road)
        pred = pred.transpose(1, 2).reshape((-1, feat.size(2)))
        label = label.reshape((-1, label.size(2)))
        mae_val, rmse_val, mape_val = masked_loss(scaler.inverse_transform(pred), scaler.inverse_transform(label),
                                                  maskp=mask)
        val_mae.append(mae_val.item())
        val_rmse.append(rmse_val.item())

    test_mae, test_rmse, test_mape = test(testdl, type)
    dur.append(time.time() - t0)
    return np.mean(train_mae), np.mean(train_rmse), np.mean(val_mae), np.mean(
        val_rmse), test_mae, test_rmse, test_mape, np.mean(train_acc)


def test(test_dataloader, type):
    if type == 'pretrain':
        domain_classifier.eval()
    model.eval()

    test_mape, test_rmse, test_mae = list(), list(), list()

    for i, (feat, label) in enumerate(test_dataloader.get_iterator()):
        feat = torch.FloatTensor(feat).to(device)
        label = torch.FloatTensor(label).to(device)
        mask = select_mask(feat.shape[2])
        if torch.sum(scaler.inverse_transform(label)) <= 0.001:
            continue

        pred = model(vec_pems04, vec_pems07, vec_pems08, feat, True, args.need_road)
        pred = pred.transpose(1, 2).reshape((-1, feat.size(2)))
        label = label.reshape((-1, label.size(2)))

        mae_test, rmse_test, mape_test = masked_loss(scaler.inverse_transform(pred), scaler.inverse_transform(label),
                                                     maskp=mask)

        test_mae.append(mae_test.item())
        test_rmse.append(rmse_test.item())
        test_mape.append(mape_test.item())

    test_rmse = np.mean(test_rmse)
    test_mae = np.mean(test_mae)
    test_mape = np.mean(test_mape)

    return test_mae, test_rmse, test_mape


def model_train(args, model, optimizer, train_dataloader, val_dataloader, test_dataloader, type):
    dur = []
    epoch = 1
    best = 999999999999999
    acc = list()

    step_per_epoch = train_dataloader.get_num_batch()
    total_step = 200 * step_per_epoch

    while epoch <= args.epoch:
        if type == 'pretrain' and args.need_weight == 1:
            source_weights = get_weight(model, type)
            if epoch == 1:
                source_weights_ma = torch.ones_like(source_weights, device=device, requires_grad=False)
            source_weights_ma = ma_param * source_weights_ma + (1 - ma_param) * source_weights
            log(source_weights_ma.mean())
        else:
            source_weights_ma = None
        start_step = epoch * step_per_epoch
        if type == 'fine-tune' and epoch > 1000:
            args.val = True
        mae_train, rmse_train, mae_val, rmse_val, mae_test, rmse_test, mape_test, train_acc = train(dur, model,
                                                                                                    optimizer,
                                                                                                    total_step,
                                                                                                    start_step,
                                                                                                    args.need_road,
                                                                                                    train_dataloader, val_dataloader, test_dataloader, type, source_weights_ma)
        log(f'Epoch {epoch} | acc_train: {train_acc: .4f} | mae_train: {mae_train: .4f} | rmse_train: {rmse_train: .4f} | mae_val: {mae_val: .4f} | rmse_val: {rmse_val: .4f} | mae_test: {mae_test: .4f} | rmse_test: {rmse_test: .4f} | mape_test: {mape_test: .4f} | Time(s) {dur[-1]: .4f}')
        epoch += 1
        acc.append(train_acc)
        if mae_val <= best:
            if type == 'fine-tune' and mae_val > 0.001:
                best = mae_val
                state = dict([('model', copy.deepcopy(model.state_dict())),
                              ('optim', copy.deepcopy(optimizer.state_dict())),
                              ('domain_classifier', copy.deepcopy(domain_classifier.state_dict()))])
                cnt = 0
            elif type == 'pretrain':
                best = mae_val
                state = dict([('model', copy.deepcopy(model.state_dict())),
                              ('optim', copy.deepcopy(optimizer.state_dict())),
                              ('domain_classifier', copy.deepcopy(domain_classifier.state_dict()))])
                cnt = 0
        else:
            cnt += 1
        if cnt == args.patience or epoch > args.epoch:
            print(f'Stop!!')
            print(f'Avg acc: {np.mean(acc)}')
            break
    print("Optimization Finished!")
    return state


device = torch.device("cuda:" + str(args.device) if torch.cuda.is_available() else "cpu")
print(f'device: {device}')
torch.manual_seed(0)
np.random.seed(0)

if args.labelrate > 100:
    args.labelrate = 100

adj_pems04, adj_pems07, adj_pems08 = load_all_adj(device)
vec_pems04 = vec_pems07 = vec_pems08 = None, None, None

dc = np.load("./data/DC/{}DC_{}.npy".format(args.dataname, args.datatype))
dcmask = dc.sum(0) > 0

chi = np.load("./data/CHI/{}CHI_{}.npy".format(args.dataname, args.datatype))
chimask = chi.sum(0) > 0

ny = np.load("./data/NY/{}NY_{}.npy".format(args.dataname, args.datatype))
nymask = ny.sum(0) > 0

cur_dir = os.getcwd()
if cur_dir[-2:] == 'sh':
    cur_dir = cur_dir[:-2]

pems04_emb_path = os.path.join('{}'.format(cur_dir), 'embeddings', 'node2vec', 'pems04',
                               '{}_vecdim.pkl'.format(args.vec_dim))
pems07_emb_path = os.path.join('{}'.format(cur_dir), 'embeddings', 'node2vec', 'pems07',
                               '{}_vecdim.pkl'.format(args.vec_dim))
pems08_emb_path = os.path.join('{}'.format(cur_dir), 'embeddings', 'node2vec', 'pems08',
                               '{}_vecdim.pkl'.format(args.vec_dim))

for i in [pems04_emb_path, pems07_emb_path, pems08_emb_path]:
    a = i.split("/")
    b = []
    for i in a:
        if "pkl" in i:
            continue
        else:
            b.append(i)
    local_path_generate(folder_name="/".join(b), create_folder_only=True)

if os.path.exists(pems04_emb_path):
    print(f'Loading pems04 embedding...')
    vec_pems04 = torch.load(pems04_emb_path, map_location='cpu')
    vec_pems04 = vec_pems04.to(device)
else:
    print(f'Generating pems04 embedding...')
    args.dataset = '4'
    vec_pems04, _ = generate_vector(adj_pems04.cpu().numpy(), args)
    vec_pems04 = vec_pems04.to(device)
    print(f'Saving pems04 embedding...')
    torch.save(vec_pems04.cpu(), pems04_emb_path)

if os.path.exists(pems07_emb_path):
    print(f'Loading pems07 embedding...')
    vec_pems07 = torch.load(pems07_emb_path, map_location='cpu')
    vec_pems07 = vec_pems07.to(device)
else:
    print(f'Generating pems07 embedding...')
    args.dataset = '7'
    vec_pems07, _ = generate_vector(adj_pems07.cpu().numpy(), args)
    vec_pems07 = vec_pems07.to(device)
    print(f'Saving pems07 embedding...')
    torch.save(vec_pems07.cpu(), pems07_emb_path)

if os.path.exists(pems08_emb_path):
    print(f'Loading pems08 embedding...')
    vec_pems08 = torch.load(pems08_emb_path, map_location='cpu')
    vec_pems08 = vec_pems08.to(device)
else:
    print(f'Generating pems08 embedding...')
    args.dataset = '8'
    vec_pems08, _ = generate_vector(adj_pems08.cpu().numpy(), args)
    vec_pems08 = vec_pems08.to(device)
    print(f'Saving pems08 embedding...')
    torch.save(vec_pems08.cpu(), pems08_emb_path)


print(
    f'Successfully load embeddings, 4: {vec_pems04.shape}, 7: {vec_pems07.shape}, 8: {vec_pems08.shape}')

domain_criterion = torch.nn.NLLLoss()
domain_classifier = Domain_classifier_DG(num_class=2, encode_dim=args.enc_dim)

domain_classifier = domain_classifier.to(device)
state = g = None, None

batch_seen = 0
cur_dir = os.getcwd()
if cur_dir[-2:] == 'sh':
    cur_dir = cur_dir[:-2]
assert args.models in ["DASTNet"]

bak_epoch = args.epoch
bak_val = args.val
bak_test = args.test
type = 'pretrain'
pretrain_model_path = os.path.join('{}'.format(cur_dir), 'pretrained', 'transfer_models',
                                   '{}'.format(args.dataset), '{}_prelen'.format(args.pre_len),
                                   'flow_model4_{}_epoch_{}_{}_{}_{}_{}_{}{}{}{}.pkl'.format(
                                       args.models, args.epoch, args.dataname, args.datatype,
                                       str(args.s1_rate).replace(".", ""),
                                       str(args.s2_rate).replace(".", ""),
                                       str(args.s3_rate).replace(".", ""),
                                       str(args.learning_rate),
                                       str(args.batch_size),
                                       str(args.split_ratio)
                                   )
                                   )

a = pretrain_model_path.split("/")
b = []
for i in a:
    if "pkl" not in i:
        b.append(i)
local_path_generate("/".join(b), create_folder_only=True)


args.dataset = "8"
if os.path.exists(pretrain_model_path):
    print(f'Loading pretrained model at {pretrain_model_path}')
    state = torch.load(pretrain_model_path, map_location='cpu')
else:
    print(f'No existing pretrained model at {pretrain_model_path}')
    args.val = args.test = False
    datasets = ["4"]
    dataset_bak = args.dataset
    labelrate_bak = args.labelrate
    args.labelrate = 100
    dataset_count = 0

    for dataset in [item for item in datasets if item not in [dataset_bak]]:
        dataset_count = dataset_count + 1

        print(
            f'\n\n****************************************************************************************************************')
        print(f'dataset: {dataset}, model: {args.models}, pre_len: {args.pre_len}, labelrate: {args.labelrate}')
        print(
            f'****************************************************************************************************************\n\n')

        if dataset == '4':
            g = vec_pems04
        elif dataset == '7':
            g = vec_pems07
        elif dataset == '8':
            g = vec_pems08

        args.dataset = dataset


        def load_graphdata_channel3(args, feat_dir, time, scaler=None, visualize=False, cut=False):
            data = ny
            data = data.reshape((data.shape[0], data.shape[1] * data.shape[2]))
            print(data.shape)
            if time:
                num_data, num_sensor = data.shape
                data = np.expand_dims(data, axis=-1)
                data = data.tolist()

                for i in range(num_data):
                    time = (i % 288) / 288
                    for j in range(num_sensor):
                        data[i][j].append(time)

                data = np.array(data)

            max_val = np.max(data)
            time_len = data.shape[0]
            seq_len = args.seq_len
            pre_len = args.pre_len
            split_ratio = args.split_ratio
            train_size = int(time_len * split_ratio)
            val_size = int(time_len * (1 - split_ratio) / 3)
            train_data = data[:train_size]
            val_data = data[train_size:train_size + val_size]
            test_data = data[train_size + val_size:time_len]
            print(data.shape)
            if args.labelrate != 100:
                import random
                new_train_size = int(train_size * args.labelrate / 100)
                start = random.randint(0, train_size - new_train_size - 1)
                train_data = train_data[start:start + new_train_size]

            train_X, train_Y, val_X, val_Y, test_X, test_Y = list(), list(), list(), list(), list(), list()

            for i in range(len(train_data) - seq_len - pre_len):
                train_X.append(np.array(train_data[i: i + seq_len]))
                train_Y.append(np.array(train_data[i + seq_len: i + seq_len + pre_len]))
            for i in range(len(val_data) - seq_len - pre_len):
                val_X.append(np.array(val_data[i: i + seq_len]))
                val_Y.append(np.array(val_data[i + seq_len: i + seq_len + pre_len]))
            for i in range(len(test_data) - seq_len - pre_len):
                test_X.append(np.array(test_data[i: i + seq_len]))
                test_Y.append(np.array(test_data[i + seq_len: i + seq_len + pre_len]))

            if visualize:
                test_X = test_X[-288:]
                test_Y = test_Y[-288:]

            if args.labelrate != 0:
                train_X = np.array(train_X)
                train_Y = np.array(train_Y)
            val_X = np.array(val_X)
            val_Y = np.array(val_Y)
            test_X = np.array(test_X)
            test_Y = np.array(test_Y)

            if args.labelrate != 0:
                max_xtrain = np.max(train_X)
                max_ytrain = np.max(train_Y)
            max_xval = np.max(val_X)
            max_yval = np.max(val_Y)
            max_xtest = np.max(test_X)
            max_ytest = np.max(test_Y)

            if args.labelrate != 0:
                min_xtrain = np.min(train_X)
                min_ytrain = np.min(train_Y)
            min_xval = np.min(val_X)
            min_yval = np.min(val_Y)
            min_xtest = np.min(test_X)
            min_ytest = np.min(test_Y)

            if args.labelrate != 0:
                max_speed = max(max_xtrain, max_ytrain, max_xval, max_yval, max_xtest, max_ytest)
                min_speed = min(min_xtrain, min_ytrain, min_xval, min_yval, min_xtest, min_ytest)

                # scaler = StandardScaler(mean=train_X[..., 0].mean(), std=train_X[..., 0].std())
                scaler = StandardScaler(mean=train_X.mean(), std=train_X.std())

                train_X = scaler.transform(train_X)
                train_Y = scaler.transform(train_Y)
            else:
                max_speed = max(max_xval, max_yval, max_xtest, max_ytest)
                min_speed = min(min_xval, min_yval, min_xtest, min_ytest)

            val_X = scaler.transform(val_X)
            val_Y = scaler.transform(val_Y)
            test_X = scaler.transform(test_X)
            test_Y = scaler.transform(test_Y)

            if args.labelrate != 0:
                max_xtrain = np.max(train_X)
                max_ytrain = np.max(train_Y)
            max_xval = np.max(val_X)
            max_yval = np.max(val_Y)
            max_xtest = np.max(test_X)
            max_ytest = np.max(test_Y)

            if args.labelrate != 0:
                min_xtrain = np.min(train_X)
                min_ytrain = np.min(train_Y)
            min_xval = np.min(val_X)
            min_yval = np.min(val_Y)
            min_xtest = np.min(test_X)
            min_ytest = np.min(test_Y)

            if args.labelrate != 0:
                max_speed = max(max_xtrain, max_ytrain, max_xval, max_yval, max_xtest, max_ytest)
                min_speed = min(min_xtrain, min_ytrain, min_xval, min_yval, min_xtest, min_ytest)

            else:
                max_speed = max(max_xval, max_yval, max_xtest, max_ytest)
                min_speed = min(min_xval, min_yval, min_xtest, min_ytest)
            if cut:
                train_X = train_X[-args.data_amount * 24:, :, :]
                train_Y = train_Y[-args.data_amount * 24:, :, :]
            return train_X, train_Y, val_X, val_Y, test_X, test_Y, max_val, scaler


        train_dataloader, val_dataloader, test_dataloader, adj, max_speed, scaler = load_data(args)
        train_X, train_Y, val_X, val_Y, test_X, test_Y, max_speed, scaler = load_graphdata_channel3(args, "", False, scaler,
                                                                                                    visualize=False)
        print([i.shape for i in [train_X, train_Y, val_X, val_Y, test_X, test_Y]])
        train_dataloader = MyDataLoader(torch.FloatTensor(train_X), torch.FloatTensor(train_Y),
                                        batch_size=args.batch_size)
        val_dataloader = MyDataLoader(torch.FloatTensor(val_X), torch.FloatTensor(val_Y), batch_size=args.batch_size)
        test_dataloader = MyDataLoader(torch.FloatTensor(test_X), torch.FloatTensor(test_Y), batch_size=args.batch_size)
        adj = 0
        model = DASTNet(input_dim=args.vec_dim, hidden_dim=args.hidden_dim, encode_dim=args.enc_dim,
                        device=device, batch_size=args.batch_size, etype=args.etype, pre_len=args.pre_len,
                        dataset=args.dataset, ft_dataset=dataset_bak,
                        adj_pems04=adj_pems04, adj_pems07=adj_pems07, adj_pems08=adj_pems08).to(device)
        optimizer = optim.SGD([{'params': model.parameters()},
                               {'params': domain_classifier.parameters()}], lr=args.learning_rate, momentum=0.8)

        if dataset_count != 1:
            model.load_state_dict(state['model'])
            optimizer.load_state_dict(state['optim'])

        state = model_train(args, model, optimizer, train_dataloader, val_dataloader, test_dataloader, type)

    print(f'Saving model to {pretrain_model_path} ...')
    torch.save(state, pretrain_model_path)
    args.dataset = dataset_bak
    args.labelrate = labelrate_bak
    args.val = bak_val
    args.test = bak_test

type = 'fine-tune'
args.epoch = args.fine_epoch

print(f'\n\n*******************************************************************************************')
print(
    f'dataset: {args.dataset}, model: {args.models}, pre_len: {args.pre_len}, labelrate: {args.labelrate}, seed: {args.division_seed}')
print(f'*******************************************************************************************\n\n')

if args.dataset == '4':
    g = vec_pems04
elif args.dataset == '7':
    g = vec_pems07
elif args.dataset == '8':
    g = vec_pems08

train_dataloader, val_dataloader, test_dataloader, adj, max_speed, scaler = load_data(args, cut=True)
model = DASTNet(input_dim=args.vec_dim, hidden_dim=args.hidden_dim, encode_dim=args.enc_dim,
                device=device, batch_size=args.batch_size, etype=args.etype, pre_len=args.pre_len,
                dataset=args.dataset, ft_dataset=args.dataset,
                adj_pems04=adj_pems04, adj_pems07=adj_pems07, adj_pems08=adj_pems08).to(device)
optimizer = optim.SGD([{'params': model.parameters()},
                       {'params': domain_classifier.parameters()}], lr=args.learning_rate, momentum=0.8)
model.load_state_dict(state['model'])
optimizer.load_state_dict(state['optim'])

if args.labelrate != 0:
    test_state = model_train(args, model, optimizer, train_dataloader, val_dataloader, test_dataloader, type)
    model.load_state_dict(test_state['model'])
    optimizer.load_state_dict(test_state['optim'])

test_mae, test_rmse, test_mape = test(test_dataloader, type)
print(f'mae: {test_mae: .4f}, rmse: {test_rmse: .4f}, mape: {test_mape * 100: .4f}\n\n')
if args.c != "default":
    if args.need_remark == 1:
        record.update(record_id, get_timestamp(),
                      "%.4f,%.4f,%.4f" %
                      (test_rmse, test_mae, test_mape * 100),
                      remark="{}C {} {} {} {}".format("2" if args.need_third == 0 else "3", str(args.data_amount),
                                                      args.dataname, args.datatype, args.machine_code))
    else:
        record.update(record_id, get_timestamp(),
                      "%.4f,%.4f, %.4f" %
                      (test_rmse, test_mae, test_mape * 100),
                      remark="{}".format(args.machine_code))
