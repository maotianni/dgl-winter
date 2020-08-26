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
    def __init__(self, g, num_negs, num_nei):
        self.g = g
        self.num_negs = num_negs
        self.num_nei = num_nei
        self.weights = self.g.in_degrees().float() ** 0.5

    def find_edges(self, heads, tails, neg_tails, start):
        head_nodes = set(np.unique(heads))
        tail_nodes = set(np.unique(tails))
        neg_nodes = set(np.unique(neg_tails.cpu().numpy()))
        potential_heads, potential_tails = self.g.all_edges(order='eid')
        potential_heads, potential_tails = potential_heads[:start], potential_tails[:start]
        # extra edge ids
        extra_v_u_id, extra_u_v_id, extra_neg_id = [], [], []
        count_v_u, count_u_v, count_neg = dict(), dict(), dict()
        for i in range(potential_heads.shape[0]-1, -1, -1):
            if potential_heads[i].item() in head_nodes:
                count_v_u[i] = count_v_u.get(i, 0) + 1
                if count_v_u[i] <= self.num_nei:
                    extra_v_u_id.append(i)
            if potential_tails[i].item() in tail_nodes:
                count_u_v[i] = count_u_v.get(i, 0) + 1
                if count_u_v[i] <= self.num_nei:
                    extra_u_v_id.append(i)
            if potential_tails[i].item() in neg_nodes:
                count_neg[i] = count_neg.get(i, 0) + 1
                if count_neg[i] <= self.num_nei:
                    extra_neg_id.append(i)
        # extra edges
        extra_u_v, extra_v_u = (potential_heads[extra_u_v_id].cpu().numpy(), potential_tails[extra_u_v_id].cpu().numpy()),\
                               (potential_heads[extra_v_u_id].cpu().numpy(), potential_tails[extra_v_u_id].cpu().numpy())
        extra_neg = (potential_heads[extra_neg_id].cpu().numpy(), potential_tails[extra_neg_id].cpu().numpy())
        return extra_v_u_id, extra_u_v_id, extra_neg_id, extra_u_v, extra_v_u, extra_neg

    def obtain_Bs(self, head_b, tail_b, start):
        n_edges = head_b.shape[0]
        heads, tails = head_b, tail_b
        neg_tails = self.weights.multinomial(self.num_negs * n_edges, replacement=True)
        neg_heads = torch.LongTensor(heads).view(-1, 1).expand(n_edges, self.num_negs).flatten()
        # find edges
        # 找到前面的边
        extra_v_u_id, extra_u_v_id, extra_neg_id,\
        extra_u_v, extra_v_u, extra_neg = self.find_edges(heads, tails, neg_tails, start)
        extended_heads = np.concatenate([extra_u_v[0], heads])
        extended_tails = np.concatenate([extra_u_v[1], tails])
        extended_heads_r = np.concatenate([extra_v_u[0], heads])
        extended_tails_r = np.concatenate([extra_v_u[1], tails])
        extended_heads_neg = np.concatenate([extra_neg[0], neg_heads])
        extended_tails_neg = np.concatenate([extra_neg[1], neg_tails])
        # 造图
        spmat_p = coo_matrix((np.ones(extended_heads.shape[0]), (extended_heads, extended_tails)),
                             shape=(self.g.number_of_nodes('user'), self.g.number_of_nodes('item')))
        spmat_pr = coo_matrix((np.ones(extended_heads_r.shape[0]), (extended_tails_r, extended_heads_r)),
                             shape=(self.g.number_of_nodes('item'), self.g.number_of_nodes('user')))
        spmat_neg = coo_matrix((np.ones(extended_heads_neg.shape[0]), (extended_heads_neg, extended_tails_neg)),
                             shape=(self.g.number_of_nodes('user'), self.g.number_of_nodes('item')))
        spmat_p_v = coo_matrix((np.ones(heads.shape[0]), (heads, tails)),
                               shape=(self.g.number_of_nodes('user'), self.g.number_of_nodes('item')))
        spmat_neg_v = coo_matrix((np.ones(heads.shape[0]), (neg_heads, neg_tails)),
                                 shape=(self.g.number_of_nodes('user'), self.g.number_of_nodes('item')))
        pos_graph = dgl.bipartite(spmat_p, 'user', 'edit', 'item')
        pos_graph_r = dgl.bipartite(spmat_pr, 'item', 'edit', 'user')
        neg_graph = dgl.bipartite(spmat_neg, 'user', 'edit', 'item')
        pos_graph_v = dgl.bipartite(spmat_p_v, 'user', 'edit', 'item')
        neg_graph_v = dgl.bipartite(spmat_neg_v, 'user', 'edit', 'item')
        # pos_graph, neg_graph = dgl.compact_graphs([pos_graph, pos_graph_r, neg_graph]) 用了这句会删节点！
        # 可以读取NID
        pos_graph = pos_graph.edge_subgraph({('user', 'edit', 'item'): list(range(pos_graph.number_of_edges()))})
        pos_graph_r = pos_graph_r.edge_subgraph({('item', 'edit', 'user'): list(range(pos_graph_r.number_of_edges()))})
        neg_graph = neg_graph.edge_subgraph({('user', 'edit', 'item'): list(range(neg_graph.number_of_edges()))})
        pos_graph_v = pos_graph_v.edge_subgraph({('user', 'edit', 'item'): list(range(pos_graph_v.number_of_edges()))})
        neg_graph_v = neg_graph_v.edge_subgraph({('user', 'edit', 'item'): list(range(neg_graph_v.number_of_edges()))})
        return pos_graph, pos_graph_r, neg_graph, pos_graph_v, neg_graph_v, extra_v_u_id, extra_u_v_id, extra_neg_id


# 节点分类
class LR_Classification(nn.Module):
    def __init__(self, in_feats, hid_feats_1, hid_feats_2, dropout=0.0):
        super(LR_Classification, self).__init__()
        self.linear_1 = nn.Linear(in_feats, hid_feats_1)
        self.linear_2 = nn.Linear(hid_feats_1, hid_feats_2)
        self.linear_3 = nn.Linear(hid_feats_2, 1)
        self.dropout = nn.Dropout(p=dropout, inplace=True)

    def forward(self, X):
        hid = F.relu(self.linear_1(X))
        hid = self.dropout(hid)
        hid = F.relu(self.linear_2(hid))
        hid = self.dropout(hid)
        hid = self.linear_3(hid)
        y_pred = torch.sigmoid(hid)
        return y_pred


class GraphNC(nn.Module):
    def __init__(self, in_feats, hid_feats_1, hid_feats_2, dropout=0.3):
        super(GraphNC, self).__init__()
        self.nc = LR_Classification(in_feats, hid_feats_1, hid_feats_2, dropout)

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
             n_users, n_items, in_feats_s, out_feats, use_cuda=False, gpu=-1, advanced=False):
    score = np.zeros(head_t.shape[0])
    print('Start Node Classification...')
    #model.eval()
    nc.eval()
    with torch.no_grad():
        si, sj = torch.zeros(n_users, in_feats_s), torch.zeros(n_items, in_feats_s)
        zi, zj = torch.zeros(n_users, out_feats), torch.zeros(n_items, out_feats)
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
            pos_graph, pos_graph_r, neg_graph, \
            pos_graph_v, neg_graph_v, \
            extra_v_u_id, extra_u_v_id, extra_neg_id = sampler.obtain_Bs(head_b, tail_b, start)
            ## cuda
            if use_cuda:
                pos_graph.to(torch.device('cuda:{}'.format(gpu)))
                pos_graph_r.to(torch.device('cuda:{}'.format(gpu)))
                neg_graph.to(torch.device('cuda:{}'.format(gpu)))
                pos_graph_v.to(torch.device('cuda:{}'.format(gpu)))
                neg_graph_v.to(torch.device('cuda:{}'.format(gpu)))
            # id
            head_id = pos_graph.srcdata[dgl.NID]
            tail_id = pos_graph.dstdata[dgl.NID]
            head_id_r = pos_graph_r.srcdata[dgl.NID]
            tail_id_r = pos_graph_r.dstdata[dgl.NID]
            head_id_neg = neg_graph.srcdata[dgl.NID]
            tail_id_neg = neg_graph.dstdata[dgl.NID]
            head_id_out = pos_graph_v.srcdata[dgl.NID]
            tail_id_out = pos_graph_v.dstdata[dgl.NID]
            # input
            si_b, sj_b = si[head_id], sj[tail_id]
            si_b_r, sj_b_r = sj[head_id_r], si[tail_id_r]
            si_b_n, sj_b_n = si[head_id_neg], sj[tail_id_neg]
            vi_b, vj_b = features_u[head_id], features_v[tail_id]
            vi_b_r, vj_b_r = features_v[head_id_r], features_u[tail_id_r]
            vi_b_n, vj_b_n = features_u[head_id_neg], features_v[tail_id_neg]
            e_b = torch.cat([features_e[extra_u_v_id], features_e[start + n_train: end + n_train]], dim=0)
            e_b_r = torch.cat([features_e[extra_v_u_id], features_e[start + n_train: end + n_train]], dim=0)
            e_b_n = torch.cat([features_e[extra_neg_id], features_e[start + n_train: end + n_train]], dim=0)
            t_b = torch.cat([t[extra_u_v_id], t[start + n_train: end + n_train]])
            t_b_r = torch.cat([t[extra_v_u_id], t[start + n_train: end + n_train]])
            t_b_n = torch.cat([t[extra_neg_id], t[start + n_train: end + n_train]])
            # forward
            if advanced:
                zi_b, zj_b, zn_b, si_b2, sj_b2 = model.infer(pos_graph, pos_graph_r, neg_graph,
                                                             si_b, sj_b, si_b_r, sj_b_r, si_b_n, sj_b_n,
                                                             e_b, e_b_r, e_b_n, t_b, t_b_r, t_b_n,
                                                             vi_b, vj_b, vi_b_r, vj_b_r, vi_b_n, vj_b_n)
            else:
                zi_b, zj_b, zn_b = model.forward(pos_graph, pos_graph_r, neg_graph,
                                                 si_b, sj_b, si_b_r, sj_b_r, si_b_n, sj_b_n,
                                                 e_b, e_b_r, e_b_n, t_b, t_b_r, t_b_n,
                                                 vi_b, vj_b, vi_b_r, vj_b_r, vi_b_n, vj_b_n)
                si_b2, sj_b2 = model.evolve(pos_graph, pos_graph_r, si_b, sj_b, si_b_r, sj_b_r, t_b, t_b_r, e_b, e_b_r)
            # output
            si[head_id_out], sj[tail_id_out] = si_b2, sj_b2
            zi[head_id_out], zj[tail_id_out] = zi_b, zj_b
            # eval
            sc = nc(zi_b, zj_b, pos_graph).view(-1).cpu().numpy()
            score[start: end] = sc
    #model.train()
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
             n_users, n_items, in_feats_s, out_feats, inductive=False, new_id=None,
             use_cuda=False, gpu=-1, advanced=False):
    val_ap, val_auc = [], []
    print('Start Link Prediction...')
    model.eval()
    with torch.no_grad():
        si, sj = torch.zeros(n_users, in_feats_s), torch.zeros(n_items, in_feats_s)
        zi, zj = torch.zeros(n_users, out_feats), torch.zeros(n_items, out_feats)
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
            pos_graph, pos_graph_r, neg_graph, \
            pos_graph_v, neg_graph_v, \
            extra_v_u_id, extra_u_v_id, extra_neg_id = sampler.obtain_Bs(head_b, tail_b, start)
            ## cuda
            if use_cuda:
                pos_graph.to(torch.device('cuda:{}'.format(gpu)))
                pos_graph_r.to(torch.device('cuda:{}'.format(gpu)))
                neg_graph.to(torch.device('cuda:{}'.format(gpu)))
                pos_graph_v.to(torch.device('cuda:{}'.format(gpu)))
                neg_graph_v.to(torch.device('cuda:{}'.format(gpu)))
            # id
            head_id = pos_graph.srcdata[dgl.NID]
            tail_id = pos_graph.dstdata[dgl.NID]
            head_id_r = pos_graph_r.srcdata[dgl.NID]
            tail_id_r = pos_graph_r.dstdata[dgl.NID]
            head_id_neg = neg_graph.srcdata[dgl.NID]
            tail_id_neg = neg_graph.dstdata[dgl.NID]
            head_id_out = pos_graph_v.srcdata[dgl.NID]
            tail_id_out = pos_graph_v.dstdata[dgl.NID]
            # input
            si_b, sj_b = si[head_id], sj[tail_id]
            si_b_r, sj_b_r = sj[head_id_r], si[tail_id_r]
            si_b_n, sj_b_n = si[head_id_neg], sj[tail_id_neg]
            vi_b, vj_b = features_u[head_id], features_v[tail_id]
            vi_b_r, vj_b_r = features_v[head_id_r], features_u[tail_id_r]
            vi_b_n, vj_b_n = features_u[head_id_neg], features_v[tail_id_neg]
            e_b = torch.cat([features_e[extra_u_v_id], features_e[start + n_train: end + n_train]], dim=0)
            e_b_r = torch.cat([features_e[extra_v_u_id], features_e[start + n_train: end + n_train]], dim=0)
            e_b_n = torch.cat([features_e[extra_neg_id], features_e[start + n_train: end + n_train]], dim=0)
            t_b = torch.cat([t[extra_u_v_id], t[start + n_train: end + n_train]])
            t_b_r = torch.cat([t[extra_v_u_id], t[start + n_train: end + n_train]])
            t_b_n = torch.cat([t[extra_neg_id], t[start + n_train: end + n_train]])
            # forward
            if advanced:
                zi_b, zj_b, zn_b, si_b2, sj_b2 = model.infer(pos_graph, pos_graph_r, neg_graph,
                                                             si_b, sj_b, si_b_r, sj_b_r, si_b_n, sj_b_n,
                                                             e_b, t_b,
                                                             vi_b, vj_b, vi_b_r, vj_b_r, vi_b_n, vj_b_n)
            else:
                zi_b, zj_b, zn_b = model.forward(pos_graph, pos_graph_r, neg_graph,
                                                 si_b, sj_b, si_b_r, sj_b_r, si_b_n, sj_b_n,
                                                 e_b, e_b_r, e_b_n, t_b, t_b_r, t_b_n,
                                                 vi_b, vj_b, vi_b_r, vj_b_r, vi_b_n, vj_b_n)
                si_b2, sj_b2 = model.evolve(pos_graph, pos_graph_r, si_b, sj_b, si_b_r, sj_b_r, t_b, t_b_r, e_b, e_b_r)
            # output
            si[head_id_out], sj[tail_id_out] = si_b2, sj_b2
            zi[head_id_out], zj[tail_id_out] = zi_b, zj_b
            # eval
            pos_graph_v.srcdata['z'] = zi_b
            pos_graph_v.dstdata['z'] = zj_b
            pos_graph_v.apply_edges(fn.u_dot_v('z', 'z', 'score'))
            pos_score = pos_graph_v.edata['score']
            neg_graph_v.srcdata['z'] = zi_b
            neg_graph_v.dstdata['z'] = zn_b
            neg_graph_v.apply_edges(fn.u_dot_v('z', 'z', 'score'))
            neg_score = neg_graph_v.edata['score']
            # inductive
            if inductive:
                id_tmp = new_id[start:end]
                pos_score = pos_score[np.where(id_tmp == 1)]
                neg_score = neg_score[np.where(id_tmp == 1)]
            # metrics
            score = torch.cat([pos_score, neg_score]).view(-1, 1).cpu().numpy()
            target = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).cpu().numpy()
            if len(pos_score) > 0:
                val_ap.append(average_precision_score(target, score))
                val_auc.append(roc_auc_score(target, score))
    model.train()
    res = {'AP': np.mean(val_ap), 'AUC': np.mean(val_auc)}
    return res
