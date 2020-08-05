import pandas as pd
import numpy as np
import scipy.sparse as sp
import dgl
from dgl import DGLGraph
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score, roc_auc_score
# 其他
import time
import argparse
import tqdm
# 自定义
import Model
import utils
import Data


# data
dl = Data.LoadData('wikipedia')
n_users = dl.n_users
n_items = dl.n_items
u_feats = dl.u_feats
v_feats = dl.v_feats
features_u = dl.features_u
features_v = dl.features_v
features_e = dl.e_feats
t = dl.t
label = dl.label
head, tail = dl.train
head_t, tail_t = dl.test
# graph
g = dgl.bipartite(list(zip(head, tail)), 'user', 'edit', 'item', num_nodes=(n_users, n_items))
g_t = dgl.bipartite(list(zip(head_t, tail_t)), 'user', 'edit', 'item', num_nodes=(n_users, n_items))
# cuda
use_cuda = False

# sampler
sampler = utils.Sample(g, num_negs=1)

# batch, train
n_edges = g.number_of_edges()
batch_size = 1000
num_heads = 2
in_feats_u = features_u.shape[1]
in_feats_v = features_v.shape[1]
in_feats_t = 100
in_feats_e = features_e.shape[1]
in_feats_s = 100
in_feats_m = in_feats_s * 2 + in_feats_t + in_feats_e

# model, loss function, optimizer
model = Model.TGNBasic(in_feats_m, in_feats_u, in_feats_v, in_feats_t,
                       in_feats_e, in_feats_s, num_heads, activation=F.relu, dropout=0.3, use_cuda=False)
loss_func = utils.Unsuper_Cross_Entropy()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
model.train()
# each epoch
si, sj = torch.zeros(n_users, in_feats_s), torch.zeros(n_items, in_feats_s)
zi, zj = torch.empty(n_users, in_feats_s), torch.zeros(n_items, in_feats_s)
for start in tqdm.trange(0, head.shape[0], batch_size):
    end = start + batch_size
    if end > head.shape[0]:
        end = head.shape[0]
    head_b = head[start:end]
    tail_b = tail[start:end]
    # sample
    pos_graph, pos_graph_r, neg_graph, neg_tails = sampler.obtain_Bs(head_b, tail_b)
    # id
    head_id = pos_graph.srcdata[dgl.NID]
    tail_id = pos_graph.dstdata[dgl.NID]
    neg_id = neg_graph.dstdata[dgl.NID]
    # input
    si_b, sj_b, sn_b = si[head_id], sj[tail_id], sj[neg_id]
    vi_b, vj_b, vn_b = features_u[head_id], features_v[tail_id], features_v[neg_id]
    e_b = features_e[start:end]
    t_b = t[start:end]
    # forward
    zi_b, zj_b, zn_b = model.forward(pos_graph, pos_graph_r, neg_graph,
                                     si_b, sj_b, sn_b,
                                     e_b, t_b,
                                     vi_b, vj_b, vn_b)
    si_b2, sj_b2 = model.evolve(pos_graph, pos_graph_r, si_b, sj_b, t_b, e_b)
    # loss / backward
    loss = loss_func(zi_b, zj_b, zn_b, pos_graph, neg_graph, use_cuda)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # overwrite
    si[head_id], sj[tail_id] = si_b2, sj_b2
    zi[head_id], zj[tail_id] = zi_b, zj_b
