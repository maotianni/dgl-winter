import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl.sampling.neighbor import sample_neighbors
import tqdm

# Loss
class Unsuper_Cross_Entropy(nn.Module):
    def forward(self, out, pos_graph, neg_graph, cuda):
        pos_graph.ndata['h'] = out
        pos_graph.apply_edges(fn.u_dot_v('h', 'h', 'score'))
        pos_score = pos_graph.edata['score']
        neg_graph.ndata['h'] = out
        neg_graph.apply_edges(fn.u_dot_v('h', 'h', 'score'))
        neg_score = neg_graph.edata['score']

        score = torch.cat([pos_score, neg_score])
        label = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).long()
        if cuda:
            label = label.cuda()
        loss = F.binary_cross_entropy_with_logits(score, label.float())
        return loss

class Sample(object):
    def __init__(self, g, fanout, num_negs):
        self.g = g
        self.fanout = fanout
        self.num_negs = num_negs
        self.weights = self.g.in_degrees().float() ** 0.5

    def obtain_Bs(self, ids):
        n_nodes = len(ids)
        n_ids = torch.LongTensor(np.asarray(ids))
        out_degrees = self.g.out_degrees(n_ids)
        heads = torch.cat([torch.ones(out_degrees[i] if out_degrees[i] < 10
                                      else 10).long() * n_ids[i] for i in range(n_nodes)])
        cat = []
        for x in n_ids:
            if self.g.successors(x).shape[0] < 10:
                cat.append(self.g.successors(x))
            else:
                suc = self.g.successors(x)
                wei = self.weights[suc]
                res_id = wei.multinomial(10)
                res = suc[res_id]
                cat.append(res)
        tails = torch.cat(cat)

        n_edges = tails.shape[0]
        neg_tails = self.weights.multinomial(self.num_negs * n_edges, replacement=True)
        neg_heads = heads.view(-1, 1).expand(n_edges, self.num_negs).flatten()
        pos_graph = dgl.graph((heads, tails), num_nodes=self.g.number_of_nodes())
        neg_graph = dgl.graph((neg_heads, neg_tails), num_nodes=self.g.number_of_nodes())
        pos_graph, neg_graph = dgl.compact_graphs([pos_graph, neg_graph])

        ids = pos_graph.ndata[dgl.NID]
        B = []
        for s in self.fanout:
            nf = sample_neighbors(self.g, nodes=ids, fanout=s, replace=True)      # 返回采样后的图，节点不变，边仅保留采样到的
            '''
            _, _, edge_ids = nf.edge_ids(
                torch.cat([heads, tails, neg_heads, neg_tails]),
                torch.cat([tails, heads, neg_tails, neg_heads]),
                return_uv=True)
            nf = dgl.remove_edges(nf, edge_ids)          # 用于计算损失函数的边剔除，前向传播用剩下的边
            '''
            b = dgl.to_block(nf, ids)        # 转为二部图，可以方便读取src和dst节点，将后一层节点作为dst
            ids = b.srcdata[dgl.NID]        # 二部图源节点作为前一层的ids
            B.insert(0, b)                  # 插入到列表最前
        return pos_graph, neg_graph, B

class GraphSAGE(nn.Module):
    def __init__(self, in_feat, hid_feat, out_feat, K, bias=False,
                 aggregator='mean', activation=None, norm=None,
                 dropout=0.0, use_cuda=False):
        super(GraphSAGE, self).__init__()
        self.in_feat = in_feat
        self.hid_feat = hid_feat
        self.out_feat = out_feat
        self.K = K
        self.bias = bias
        self.aggregator = aggregator
        self.activation = activation
        self.norm = norm

        self.weight_in = nn.Parameter(torch.Tensor(2, self.in_feat, self.hid_feat))
        nn.init.xavier_uniform_(self.weight_in, gain=nn.init.calculate_gain('relu'))
        if self.bias:
            self.bias_in_k = nn.Parameter(torch.Tensor(2, self.hid_feat))
            nn.init.zeros_(self.bias_in_k)
        if self.K > 2:
            self.weight_hid = nn.Parameter(torch.Tensor(self.K - 2, 2, self.hid_feat, self.hid_feat))
            nn.init.xavier_uniform_(self.weight_hid, gain=nn.init.calculate_gain('relu'))
            if self.bias:
                self.bias_hid_k = nn.Parameter(torch.Tensor(self.K - 2, 2, self.hid_feat))
                nn.init.zeros_(self.bias_hid_k)
        self.weight_out = nn.Parameter(torch.Tensor(2, self.hid_feat, self.out_feat))
        nn.init.xavier_uniform_(self.weight_out, gain=nn.init.calculate_gain('relu'))
        if self.bias:
            self.bias_out_k = nn.Parameter(torch.Tensor(2, self.out_feat))
            nn.init.zeros_(self.bias_out_k)

        self.dropout = nn.Dropout(dropout)
        self.use_cuda = use_cuda

        if self.aggregator == 'gcn':
            self.weight_gcn_in = nn.Parameter(torch.Tensor(self.in_feat, self.in_feat))
            nn.init.xavier_uniform_(self.weight_gcn_in, gain=nn.init.calculate_gain('relu'))
            self.weight_gcn_hid = nn.Parameter(torch.Tensor(self.hid_feat, self.hid_feat))
            nn.init.xavier_uniform_(self.weight_gcn_hid, gain=nn.init.calculate_gain('relu'))

        if self.aggregator == 'lstm':
            self.lstm_in = nn.LSTM(self.in_feat, self.in_feat, batch_first=True)
            self.lstm_in.reset_parameters()
            self.lstm_hid = nn.LSTM(self.hid_feat, self.hid_feat, batch_first=True)
            self.lstm_hid.reset_parameters()

        if self.aggregator == 'pool':
            self.weight_pool_in = nn.Parameter(torch.Tensor(self.in_feat, self.in_feat))
            nn.init.xavier_uniform_(self.weight_pool_in, gain=nn.init.calculate_gain('relu'))
            self.weight_pool_hid = nn.Parameter(torch.Tensor(self.hid_feat, self.hid_feat))
            nn.init.xavier_uniform_(self.weight_pool_hid, gain=nn.init.calculate_gain('relu'))
            if self.bias:
                self.bias_in = nn.Parameter(torch.Tensor(self.in_feat))
                nn.init.zeros_(self.bias_in)
                self.bias_hid = nn.Parameter(torch.Tensor(self.hid_feat))
                nn.init.zeros_(self.bias_hid)

    def lstm_reducer_in(self, nodes):
        m = nodes.mailbox['m']
        batch_size = m.shape[0]
        h = (m.new_zeros((1, batch_size, self.in_feat)),
             m.new_zeros((1, batch_size, self.in_feat)))
        _, (neigh, _) = self.lstm_in(m, h)
        return {'neigh': neigh.squeeze(0)}

    def lstm_reducer_hid(self, nodes):
        m = nodes.mailbox['m']
        batch_size = m.shape[0]
        h = (m.new_zeros((1, batch_size, self.hid_feat)),
             m.new_zeros((1, batch_size, self.hid_feat)))
        _, (neigh, _) = self.lstm_hid(m, h)
        return {'neigh': neigh.squeeze(0)}

    def agg(self, x, B):
        h = x
        x = self.dropout(x)
        for i in range(self.K):
            if i == 0:
                if self.aggregator == 'pool':
                    x = torch.matmul(x, self.weight_pool_in)
                    if self.bias:
                        x = x + self.bias_in
                if self.aggregator == 'gcn':
                    B[i].srcdata['h'] = torch.matmul(x, self.weight_gcn_in)
                else:
                    B[i].srcdata['h'] = x
                B[i].dstdata['h'] = x[:B[i].number_of_dst_nodes()]
            else:
                if self.aggregator == 'pool':
                    hh = torch.matmul(B[i - 1].dstdata['h'], self.weight_pool_hid)
                    if self.bias:
                        hh = hh + self.bias_hid
                else:
                    hh = B[i - 1].dstdata['h']
                if self.aggregator == 'gcn':
                    B[i].srcdata['h'] = torch.matmul(hh, self.weight_gcn_hid)
                else:
                    B[i].srcdata['h'] = hh
                B[i].dstdata['h'] = hh[:B[i].number_of_dst_nodes()]
            if self.aggregator == 'gcn':
                B[i].update_all(fn.copy_src('h', 'm'), fn.sum('m', 'neigh'))
            elif self.aggregator == 'mean':
                B[i].update_all(fn.copy_src('h', 'm'), fn.mean('m', 'neigh'))
            elif self.aggregator == 'lstm':
                B[i].update_all(fn.copy_src('h', 'm'), self.lstm_reducer_in if i == 0 else self.lstm_reducer_hid)
            else:
                B[i].update_all(fn.copy_src('h', 'm'), fn.max('m', 'neigh'))
            h_neigh = B[i].dstdata['neigh']
            if i == 0:
                h = torch.matmul(B[i].dstdata['h'], self.weight_in[0, :, :]) \
                    + (torch.matmul(h_neigh, self.weight_in[1, :, :]) if self.aggregator != 'gcn' else 0)
                if self.bias:
                    h = h + self.bias_in_k[0, :] + (self.bias_in_k[1, :] if self.aggregator != 'gcn' else 0)
            elif i == self.K - 1:
                h = torch.matmul(B[i].dstdata['h'], self.weight_out[0, :, :])\
                    + (torch.matmul(h_neigh, self.weight_out[1, :, :]) if self.aggregator != 'gcn' else 0)
                if self.bias:
                    h = h + self.bias_out_k[0, :] + (self.bias_out_k[1, :] if self.aggregator != 'gcn' else 0)
            else:
                h = torch.matmul(B[i].dstdata['h'], self.weight_hid[i - 1, 0, :, :])\
                    + (torch.matmul(h_neigh, self.weight_hid[i - 1, 1, :, :]) if self.aggregator != 'gcn' else 0)
                if self.bias:
                    h = h + self.bias_hid_k[0, :] + (self.bias_hid_k[1, :] if self.aggregator != 'gcn' else 0)
            if self.activation and i != self.K - 1:
                h = self.activation(h, inplace=False)
            if i != self.K - 1:
                h = self.dropout(h)
            if self.norm:
                norm = torch.norm(h, dim=1)
                norm = norm + (norm == 0).long()
                h = h / norm.unsqueeze(-1)
            B[i].dstdata['h'] = h
        return h

    def forward(self, x, B):
        z = self.agg(x, B)
        return z

    def full(self, g, batch_size):
        nodes = torch.arange(g.number_of_nodes())
        x = g.ndata['features']
        x = self.dropout(x)
        for i in range(self.K):
            if i != self.K - 1:
                y = torch.zeros(g.number_of_nodes(), self.hid_feat)
            else:
                y = torch.zeros(g.number_of_nodes(), self.out_feat)
            for start in tqdm.trange(0, g.number_of_nodes(), batch_size):
                end = start + batch_size
                batch_nodes = nodes[start:end]
                block = dgl.to_block(dgl.in_subgraph(g, batch_nodes), batch_nodes)
                input_nodes = block.srcdata[dgl.NID]
                h = x[input_nodes]
                if self.use_cuda:
                    h = h.cuda()              # 下一层时使用了上一层的y，y默认为cpu()
                if self.aggregator == 'pool':
                    if i == 0:
                        h = torch.matmul(h, self.weight_pool_in)
                        if self.bias:
                            h = h + self.bias_in
                    else:
                        h = torch.matmul(h, self.weight_pool_hid)
                        if self.bias:
                            h = h + self.bias_hid
                if self.aggregator == 'gcn':
                    if i == 0:
                        block.srcdata['h'] = torch.matmul(h, self.weight_gcn_in)
                    else:
                        block.srcdata['h'] = torch.matmul(h, self.weight_gcn_hid)
                else:
                    block.srcdata['h'] = h
                block.dstdata['h'] = h[:block.number_of_dst_nodes()]
                if self.aggregator == 'gcn':
                    block.update_all(fn.copy_src('h', 'm'), fn.sum('m', 'neigh'))
                elif self.aggregator == 'mean':
                    block.update_all(fn.copy_src('h', 'm'), fn.mean('m', 'neigh'))
                elif self.aggregator == 'lstm':
                    block.update_all(fn.copy_src('h', 'm'), self.lstm_reducer_in if i == 0 else self.lstm_reducer_hid)
                else:
                    block.update_all(fn.copy_src('h', 'm'), fn.max('m', 'neigh'))
                h_neigh = block.dstdata['neigh']
                if i == 0:
                    h = torch.matmul(block.dstdata['h'], self.weight_in[0, :, :]) \
                        + (torch.matmul(h_neigh, self.weight_in[1, :, :]) if self.aggregator != 'gcn' else 0)
                    if self.bias:
                        h = h + self.bias_in_k[0, :] + (self.bias_in_k[1, :] if self.aggregator != 'gcn' else 0)
                elif i == self.K - 1:
                    h = torch.matmul(block.dstdata['h'], self.weight_out[0, :, :]) \
                        + (torch.matmul(h_neigh, self.weight_out[1, :, :]) if self.aggregator != 'gcn' else 0)
                    if self.bias:
                        h = h + self.bias_out_k[0, :] + (self.bias_out_k[1, :] if self.aggregator != 'gcn' else 0)
                else:
                    h = torch.matmul(block.dstdata['h'], self.weight_hid[i - 1, 0, :, :]) \
                        + (torch.matmul(h_neigh, self.weight_hid[i - 1, 1, :, :])  if self.aggregator != 'gcn' else 0)
                    if self.bias:
                        h = h + self.bias_hid_k[0, :] + (self.bias_hid_k[1, :] if self.aggregator != 'gcn' else 0)
                if self.activation and i != self.K - 1:
                    h = self.activation(h, inplace=False)
                if i != self.K - 1:
                    h = self.dropout(h)
                if self.norm:
                    norm = torch.norm(h, dim=1)
                    norm = norm + (norm == 0).long()
                    h = h / norm.unsqueeze(-1)
                y[start:end] = h
            x = y
        if self.use_cuda:
            x = x.cuda()
        g.ndata['z'] = x
        return g

    def infer(self, g, batch_size):
        g = self.full(g, batch_size)
        z = g.ndata['z']
        return z

