import numpy as np
import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import coo_matrix
from sklearn.metrics import average_precision_score, roc_auc_score
import tqdm


# Loss
class Unsuper_Cross_Entropy(nn.Module):
    def forward(self, zi, zj, zn, pos_graph, neg_graph, cuda):
        pos_graph.srcdata['h'] = zi
        pos_graph.dstdata['h'] = zj
        pos_graph.apply_edges(fn.u_dot_v('h', 'h', 'score'))
        pos_score = pos_graph.edata['score']
        neg_graph.srcdata['h'] = zi
        neg_graph.dstdata['h'] = zn
        neg_graph.apply_edges(fn.u_dot_v('h', 'h', 'score'))
        neg_score = neg_graph.edata['score']

        score = torch.cat([pos_score, neg_score])
        label = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).long()
        if cuda:
            label = label.cuda()
        loss = F.binary_cross_entropy_with_logits(score, label.float())
        return loss


# sample
class Sample(object):
    def __init__(self, g, num_negs):
        self.g = g
        self.num_negs = num_negs
        self.weights = self.g.in_degrees().float() ** 0.5

    def obtain_Bs(self, head_b, tail_b):
        n_edges = head_b.shape[0]
        heads, tails = head_b, tail_b
        neg_tails = self.weights.multinomial(self.num_negs * n_edges, replacement=True)
        neg_heads = torch.LongTensor(heads).view(-1, 1).expand(n_edges, self.num_negs).flatten()
        spmat_p = coo_matrix((np.ones(heads.shape[0]), (heads, tails)),
                             shape=(self.g.number_of_nodes('user'), self.g.number_of_nodes('item')))
        spmat_pr = coo_matrix((np.ones(heads.shape[0]), (tails, heads)),
                             shape=(self.g.number_of_nodes('item'), self.g.number_of_nodes('user')))
        spmat_neg = coo_matrix((np.ones(heads.shape[0]), (neg_heads, neg_tails)),
                             shape=(self.g.number_of_nodes('user'), self.g.number_of_nodes('item')))
        pos_graph = dgl.bipartite(spmat_p, 'user', 'edit', 'item')
        pos_graph_r = dgl.bipartite(spmat_pr, 'item', 'edit', 'user')
        neg_graph = dgl.bipartite(spmat_neg, 'user', 'edit', 'item')
        # pos_graph, neg_graph = dgl.compact_graphs([pos_graph, pos_graph_r, neg_graph]) 用了这句会删节点！
        # 可以读取NID
        pos_graph = pos_graph.edge_subgraph({('user', 'edit', 'item'): list(range(pos_graph.number_of_edges()))})
        pos_graph_r = pos_graph_r.edge_subgraph({('item', 'edit', 'user'): list(range(pos_graph_r.number_of_edges()))})
        neg_graph = neg_graph.edge_subgraph({('user', 'edit', 'item'): list(range(neg_graph.number_of_edges()))})
        return pos_graph, pos_graph_r, neg_graph


# 节点分类
class LR_Classification(nn.Module):
    def __init__(self, in_feats, hid_feats, n_layers=2, dropout=0.0):
        super(LR_Classification, self).__init__()
        self.n_layers = n_layers
        self.linear_1 = nn.Linear(in_feats, hid_feats)
        self.linear_2 = nn.Linear(hid_feats, hid_feats)
        self.linear_3 = nn.Linear(hid_feats, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X):
        hid = F.relu(self.linear_1(X))
        hid = self.dropout(hid)
        for layer in range(self.n_layers - 2):
            hid = F.relu(self.linear_2(hid))
            hid = self.dropout(hid)
        hid = self.linear_3(hid)
        hid = self.dropout(hid)
        y_pred = torch.sigmoid(hid)
        return y_pred


class GraphNC(nn.Module):
    def __init__(self, in_feats, hid_feats):
        super(GraphNC, self).__init__()
        self.nc = LR_Classification(in_feats, hid_feats, dropout=0.3)

    def edge_cat(self, edges):
        hi = edges.src['z']
        hj = edges.dst['z']
        raw = torch.cat([hi, hj], dim=1)
        return {'raw': raw}

    def forward(self, zi, zj, pos_graph):
        pos_graph.srcdata['z'] = zi
        pos_graph.dstdata['z'] = zj
        pos_graph.apply_edges(self.edge_cat)
        pos_graph.edata['nc_score'] = self.nc(pos_graph.edata['raw'])
        pos_score = pos_graph.edata['nc_score']
        return pos_score


# evaluation-nc
def node_class(model, nc, sampler, n_train, head_t, tail_t, batch_size_test,
             features_u, features_v, features_e, t, label,
             n_users, n_items, in_feats_s, use_cuda=False, gpu=-1, advanced=False):
    score = np.zeros(head_t.shape[0])
    print('Start Node Classification...')
    model.eval()
    nc.eval()
    with torch.no_grad():
        si, sj = torch.zeros(n_users, in_feats_s), torch.zeros(n_items, in_feats_s)
        zi, zj = torch.zeros(n_users, in_feats_s), torch.zeros(n_items, in_feats_s)
        ## cuda
        if use_cuda:
            si, sj = si.cuda(), sj.cuda()
            zi, zj = zi.cuda(), zj.cuda()
        for start in tqdm.trange(0, head_t.shape[0], batch_size_test):
            end = start + batch_size_test
            if end > head_t.shape[0]:
                end = head_t.shape[0]
            head_b = head_t[start:end]
            tail_b = tail_t[start:end]
            # sample
            pos_graph, pos_graph_r, neg_graph = sampler.obtain_Bs(head_b, tail_b)
            ## cuda
            if use_cuda:
                pos_graph.to(torch.device('cuda:{}'.format(gpu)))
                pos_graph_r.to(torch.device('cuda:{}'.format(gpu)))
                neg_graph.to(torch.device('cuda:{}'.format(gpu)))
            # id
            head_id = pos_graph.srcdata[dgl.NID]
            tail_id = pos_graph.dstdata[dgl.NID]
            neg_id = neg_graph.dstdata[dgl.NID]
            # input
            si_b, sj_b, sn_b = si[head_id], sj[tail_id], sj[neg_id]
            vi_b, vj_b, vn_b = features_u[head_id], features_v[tail_id], features_v[neg_id]
            e_b = features_e[start + n_train: end + n_train]
            t_b = t[start + n_train: end + n_train]
            # forward
            if advanced:
                zi_b, zj_b, zn_b, si_b2, sj_b2 = model.infer(pos_graph, pos_graph_r, neg_graph,
                                                             si_b, sj_b, sn_b,
                                                             e_b, t_b,
                                                             vi_b, vj_b, vn_b)
            else:
                zi_b, zj_b, zn_b = model.forward(pos_graph, pos_graph_r, neg_graph,
                                                 si_b, sj_b, sn_b,
                                                 e_b, t_b,
                                                 vi_b, vj_b, vn_b)
                si_b2, sj_b2 = model.evolve(pos_graph, pos_graph_r, si_b, sj_b, t_b, e_b)
            # output
            si[head_id], sj[tail_id] = si_b2, sj_b2
            zi[head_id], zj[tail_id] = zi_b, zj_b
            # eval
            sc = nc(zi_b, zj_b, pos_graph).view(-1).cpu().numpy()
            score[start: end] = sc
    model.train()
    nc.train()
    auc = roc_auc_score(label[n_train:].cpu().numpy(), score)
    res = {'AUC': auc}
    return res


# 链接预测
class Link_Prediction(nn.Module):
    def __init__(self, in_feats, hid_feats, n_layers=2, dropout=0.0):
        super(Link_Prediction, self).__init__()
        self.n_layers = n_layers
        self.linear_1 = nn.Linear(in_feats, hid_feats)
        self.linear_2 = nn.Linear(hid_feats, hid_feats)
        self.linear_3 = nn.Linear(hid_feats, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X):
        hid = F.relu(self.linear_1(X))
        hid = self.dropout(hid)
        for layer in range(self.n_layers - 2):
            hid = F.relu(self.linear_2(hid))
            hid = self.dropout(hid)
        hid = self.linear_3(hid)
        hid = self.dropout(hid)
        y_pred = torch.sigmoid(hid)
        return y_pred


# graph lp
class GraphLP(nn.Module):
    def __init__(self, in_feats, hid_feats):
        super(GraphLP, self).__init__()
        self.lp = Link_Prediction(in_feats, hid_feats)

    def edge_cat(self, edges):
        hi = edges.src['z']
        hj = edges.dst['z']
        raw = torch.cat([hi, hj], dim=1)
        return {'raw': raw}

    def forward(self, zi, zj, zn, pos_graph, neg_graph):
        pos_graph.srcdata['z'] = zi
        pos_graph.dstdata['z'] = zj
        pos_graph.apply_edges(self.edge_cat)
        pos_graph.edata['lp_score'] = self.lp(pos_graph.edata['raw'])
        pos_score = pos_graph.edata['lp_score']
        neg_graph.srcdata['z'] = zi
        neg_graph.dstdata['z'] = zn
        neg_graph.apply_edges(self.edge_cat)
        neg_graph.edata['lp_score'] = self.lp(pos_graph.edata['raw'])
        neg_score = neg_graph.edata['lp_score']
        # score = torch.cat([pos_score, neg_score])
        return pos_score, neg_score


# evaluation-lp
def link_pre(model, sampler, n_train, head_t, tail_t, batch_size_test,
             features_u, features_v, features_e, t,
             n_users, n_items, in_feats_s, use_cuda=False, gpu=-1, advanced=False):
    val_ap, val_auc = [], []
    print('Start Link Prediction...')
    model.eval()
    with torch.no_grad():
        si, sj = torch.zeros(n_users, in_feats_s), torch.zeros(n_items, in_feats_s)
        zi, zj = torch.zeros(n_users, in_feats_s), torch.zeros(n_items, in_feats_s)
        ## cuda
        if use_cuda:
            si, sj = si.cuda(), sj.cuda()
            zi, zj = zi.cuda(), zj.cuda()
        for start in tqdm.trange(0, head_t.shape[0], batch_size_test):
            end = start + batch_size_test
            if end > head_t.shape[0]:
                end = head_t.shape[0]
            head_b = head_t[start:end]
            tail_b = tail_t[start:end]
            # sample
            pos_graph, pos_graph_r, neg_graph = sampler.obtain_Bs(head_b, tail_b)
            ## cuda
            if use_cuda:
                pos_graph.to(torch.device('cuda:{}'.format(gpu)))
                pos_graph_r.to(torch.device('cuda:{}'.format(gpu)))
                neg_graph.to(torch.device('cuda:{}'.format(gpu)))
            # id
            head_id = pos_graph.srcdata[dgl.NID]
            tail_id = pos_graph.dstdata[dgl.NID]
            neg_id = neg_graph.dstdata[dgl.NID]
            # input
            si_b, sj_b, sn_b = si[head_id], sj[tail_id], sj[neg_id]
            vi_b, vj_b, vn_b = features_u[head_id], features_v[tail_id], features_v[neg_id]
            e_b = features_e[start + n_train: end + n_train]
            t_b = t[start + n_train: end + n_train]
            # forward
            if advanced:
                zi_b, zj_b, zn_b, si_b2, sj_b2 = model.infer(pos_graph, pos_graph_r, neg_graph,
                                                             si_b, sj_b, sn_b,
                                                             e_b, t_b,
                                                             vi_b, vj_b, vn_b)
            else:
                zi_b, zj_b, zn_b = model.forward(pos_graph, pos_graph_r, neg_graph,
                                                 si_b, sj_b, sn_b,
                                                 e_b, t_b,
                                                 vi_b, vj_b, vn_b)
                si_b2, sj_b2 = model.evolve(pos_graph, pos_graph_r, si_b, sj_b, t_b, e_b)
            # output
            si[head_id], sj[tail_id] = si_b2, sj_b2
            zi[head_id], zj[tail_id] = zi_b, zj_b
            # eval
            pos_graph.srcdata['z'] = zi_b
            pos_graph.dstdata['z'] = zj_b
            pos_graph.apply_edges(fn.u_dot_v('z', 'z', 'score'))
            pos_score = pos_graph.edata['score']
            neg_graph.srcdata['z'] = zi_b
            neg_graph.dstdata['z'] = zn_b
            neg_graph.apply_edges(fn.u_dot_v('z', 'z', 'score'))
            neg_score = neg_graph.edata['score']
            # metrics
            score = torch.cat([pos_score, neg_score]).view(-1, 1).cpu().numpy()
            target = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).cpu().numpy()
            val_ap.append(average_precision_score(target, score))
            val_auc.append(roc_auc_score(target, score))
    model.train()
    res = {'AP': np.mean(val_ap), 'AUC': np.mean(val_auc)}
    return res
