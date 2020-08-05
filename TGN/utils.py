import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        pos_graph = dgl.bipartite((heads, tails), 'user', 'edit', 'item',
                                  num_nodes=(self.g.number_of_nodes('user'), self.g.number_of_nodes('item')))
        neg_graph = dgl.bipartite((neg_heads, neg_tails), 'user', 'edit', 'item',
                                  num_nodes=(self.g.number_of_nodes('user'), self.g.number_of_nodes('item')))
        pos_graph, neg_graph = dgl.compact_graphs([pos_graph, neg_graph])
        # 可以读取NID
        pos_graph = pos_graph.edge_subgraph({('user', 'edit', 'item'): list(range(pos_graph.number_of_edges()))})
        neg_graph = neg_graph.edge_subgraph({('user', 'edit', 'item'): list(range(neg_graph.number_of_edges()))})
        return pos_graph, neg_graph, neg_tails


class LR_Classification(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(LR_Classification, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.logsoftmax = nn.LogSoftmax()

    def forward(self, X):
        y_pred = self.logsoftmax(self.linear(X))
        return y_pred


class Link_Prediction(nn.Module):
    def __init__(self, in_feats, hid_feats, n_layers=2):
        super(Link_Prediction, self).__init__()
        self.n_layers = n_layers
        self.linear_1 = nn.Linear(in_feats, hid_feats)
        self.linear_2 = nn.Linear(hid_feats, hid_feats)
        self.linear_3 = nn.Linear(hid_feats, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X):
        hid = F.relu(self.linear_1(X))
        for layer in range(self.n_layers - 2):
            hid = F.relu(self.linear_2(hid))
        y_pred = F.sigmoid(self.linear_3(hid))
        return y_pred
