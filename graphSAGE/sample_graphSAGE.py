import numpy as np
import torch
import torch.nn as nn
import dgl
import dgl.function as fn
from dgl.sampling.neighbor import sample_neighbors

class Sample(object):
    def __init__(self, g, fanout):
        self.g = g
        self.fanout = fanout

    def obtain_Bs(self, ids):
        ids = torch.LongTensor(np.asarray(ids))
        B = []
        for s in self.fanout:
            nf = sample_neighbors(self.g, nodes=ids, fanout=s, replace=True)      # 返回采样后的图，节点不变，边仅保留采样到的
            b = dgl.to_block(nf, ids)        # 转为二部图，可以方便读取src和dst节点，将后一层节点作为dst
            ids = b.srcdata[dgl.NID]        # 二部图源节点作为前一层的ids
            B.insert(0, b)                  # 插入到列表最前
        return B

class GraphSAGE(nn.Module):
    def __init__(self, in_feat, hid_feat, out_feat, K, bias=False,
                 aggregator='mean', activation=None,
                 dropout=0.0, use_cuda=False):
        super(GraphSAGE, self).__init__()
        self.in_feat = in_feat
        self.hid_feat = hid_feat
        self.out_feat = out_feat
        self.K = K
        self.bias = bias
        self.aggregator = aggregator
        self.activation = activation

        self.weight_in = nn.Parameter(torch.Tensor(self.in_feat * 2, self.hid_feat))
        nn.init.xavier_uniform_(self.weight_in, gain=nn.init.calculate_gain('relu'))
        if self.K > 2:
            self.weight_hid = nn.Parameter(torch.Tensor(self.K - 2, self.hid_feat * 2, self.hid_feat))
            nn.init.xavier_uniform_(self.weight_hid, gain=nn.init.calculate_gain('relu'))
        self.weight_out = nn.Parameter(torch.Tensor(self.hid_feat * 2, self.out_feat))
        nn.init.xavier_uniform_(self.weight_out, gain=nn.init.calculate_gain('relu'))

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

    def gcn_agg(self, x, B):
        h = x
        x = self.dropout(x)
        for i in range(self.K):
            if i == 0:
                B[i].srcdata['h'] = torch.matmul(x, self.weight_gcn_in)
                B[i].dstdata['h'] = x[:B[i].number_of_dst_nodes()]
                B[i].update_all(fn.copy_src('h', 'm'), fn.sum('m', 'neigh'))
                degs = B[i].in_degrees().to(x)
                h_neigh = (B[i].dstdata['neigh'] + B[i].dstdata['h']) / (degs.unsqueeze(-1) + 1)
                h = torch.matmul(torch.cat([B[i].dstdata['h'], h_neigh], dim=1), self.weight_in)
                if self.activation:
                    h = self.activation(h, inplace=False)
                norm = torch.norm(h, dim=1)
                h = h / (norm.unsqueeze(-1) + 0.0001)
                B[i].dstdata['h'] = h
            elif i == self.K - 1:
                hh = B[i - 1].dstdata['h']
                B[i].srcdata['h'] = torch.matmul(hh, self.weight_gcn_hid)
                B[i].dstdata['h'] = hh[:B[i].number_of_dst_nodes()]
                B[i].update_all(fn.copy_src('h', 'm'), fn.sum('m', 'neigh'))
                degs = B[i].in_degrees().to(x)
                h_neigh = (B[i].dstdata['neigh'] + B[i].dstdata['h']) / (degs.unsqueeze(-1) + 1)
                h = torch.matmul(torch.cat([B[i].dstdata['h'], h_neigh], dim=1), self.weight_out)
                norm = torch.norm(h, dim=1)
                h = h / (norm.unsqueeze(-1) + 0.0001)
                B[i].dstdata['h'] = h
            else:
                hh = B[i - 1].dstdata['h']
                B[i].srcdata['h'] = torch.matmul(hh, self.weight_gcn_hid)
                B[i].dstdata['h'] = hh[:B[i].number_of_dst_nodes()]
                B[i].update_all(fn.copy_src('h', 'm'), fn.sum('m', 'neigh'))
                degs = B[i].in_degrees().to(x)
                h_neigh = (B[i].dstdata['neigh'] + B[i].dstdata['h']) / (degs.unsqueeze(-1) + 1)
                h = torch.matmul(torch.cat([B[i].dstdata['h'], h_neigh], dim=1), self.weight_hid[i-1, :, :])
                if self.activation:
                    h = self.activation(h, inplace=False)
                norm = torch.norm(h, dim=1)
                h = h / (norm.unsqueeze(-1) + 0.0001)
                B[i].dstdata['h'] = h
        return h

    def mean_agg(self, x, B):
        h = x
        x = self.dropout(x)
        for i in range(self.K):
            if i == 0:
                B[i].srcdata['h'] = x
                B[i].dstdata['h'] = x[:B[i].number_of_dst_nodes()]
                B[i].update_all(fn.copy_src('h', 'm'), fn.mean('m', 'neigh'))
                h_neigh = B[i].dstdata['neigh']
                h = torch.matmul(torch.cat([B[i].dstdata['h'], h_neigh], dim=1), self.weight_in)
                if self.activation:
                    h = self.activation(h, inplace=False)
                norm = torch.norm(h, dim=1)
                h = h / (norm.unsqueeze(-1) + 0.0001)
                B[i].dstdata['h'] = h
            elif i == self.K - 1:
                hh = B[i-1].dstdata['h']
                B[i].srcdata['h'] = hh
                B[i].dstdata['h'] = hh[:B[i].number_of_dst_nodes()]
                B[i].update_all(fn.copy_src('h', 'm'), fn.mean('m', 'neigh'))
                h_neigh = B[i].dstdata['neigh']
                h = torch.matmul(torch.cat([B[i].dstdata['h'], h_neigh], dim=1), self.weight_out)
                norm = torch.norm(h, dim=1)
                h = h / (norm.unsqueeze(-1) + 0.0001)
                B[i].dstdata['h'] = h
            else:
                hh = B[i - 1].dstdata['h']
                B[i].srcdata['h'] = hh
                B[i].dstdata['h'] = hh[:B[i].number_of_dst_nodes()]
                B[i].update_all(fn.copy_src('h', 'm'), fn.mean('m', 'neigh'))
                h_neigh = B[i].dstdata['neigh']
                h = torch.matmul(torch.cat([B[i].dstdata['h'], h_neigh], dim=1), self.weight_hid[i-1, :, :])
                if self.activation:
                    h = self.activation(h, inplace=False)
                norm = torch.norm(h, dim=1)
                h = h / (norm.unsqueeze(-1) + 0.0001)
                B[i].dstdata['h'] = h
        return h

    def lstm_agg(self, x, B):
        h = x
        x = self.dropout(x)
        for i in range(self.K):
            if i == 0:
                B[i].srcdata['h'] = x
                B[i].dstdata['h'] = x[:B[i].number_of_dst_nodes()]
                B[i].update_all(fn.copy_src('h', 'm'), self.lstm_reducer_in)
                h_neigh = B[i].dstdata['neigh']
                h = torch.matmul(torch.cat([B[i].dstdata['h'], h_neigh], dim=1), self.weight_in)
                if self.activation:
                    h = self.activation(h, inplace=False)
                norm = torch.norm(h, dim=1)
                h = h / (norm.unsqueeze(-1) + 0.0001)
                B[i].dstdata['h'] = h
            elif i == self.K - 1:
                hh = B[i - 1].dstdata['h']
                B[i].srcdata['h'] = hh
                B[i].dstdata['h'] = hh[:B[i].number_of_dst_nodes()]
                B[i].update_all(fn.copy_src('h', 'm'), self.lstm_reducer_hid)
                h_neigh = B[i].dstdata['neigh']
                h = torch.matmul(torch.cat([B[i].dstdata['h'], h_neigh], dim=1), self.weight_out)
                norm = torch.norm(h, dim=1)
                h = h / (norm.unsqueeze(-1) + 0.0001)
                B[i].dstdata['h'] = h
            else:
                hh = B[i - 1].dstdata['h']
                B[i].srcdata['h'] = hh
                B[i].dstdata['h'] = hh[:B[i].number_of_dst_nodes()]
                B[i].update_all(fn.copy_src('h', 'm'), self.lstm_reducer_hid)
                h_neigh = B[i].dstdata['neigh']
                h = torch.matmul(torch.cat([B[i].dstdata['h'], h_neigh], dim=1), self.weight_hid[i-1, :, :])
                if self.activation:
                    h = self.activation(h, inplace=False)
                norm = torch.norm(h, dim=1)
                h = h / (norm.unsqueeze(-1) + 0.0001)
                B[i].dstdata['h'] = h
        return h

    def pool_agg(self, x, B):
        x = self.dropout(x)
        h = torch.matmul(x, self.weight_pool_in)
        if self.bias:
            h = h + self.bias_in
        for i in range(self.K):
            if i == 0:
                B[i].srcdata['h'] = h
                B[i].dstdata['h'] = h[:B[i].number_of_dst_nodes()]
                B[i].update_all(fn.copy_src('h', 'm'), fn.max('m', 'neigh'))
                h_neigh = B[i].dstdata['neigh']
                h = torch.matmul(torch.cat([B[i].dstdata['h'], h_neigh], dim=1), self.weight_in)
                if self.activation:
                    h = self.activation(h, inplace=False)
                norm = torch.norm(h, dim=1)
                h = h / (norm.unsqueeze(-1) + 0.0001)
                B[i].dstdata['h'] = h
            elif i == self.K - 1:
                h = torch.matmul(B[i - 1].dstdata['h'], self.weight_pool_hid)
                if self.bias:
                    h = h + self.bias_hid
                B[i].srcdata['h'] = h
                B[i].dstdata['h'] = h[:B[i].number_of_dst_nodes()]
                B[i].update_all(fn.copy_src('h', 'm'), fn.max('m', 'neigh'))
                h_neigh = B[i].dstdata['neigh']
                h = torch.matmul(torch.cat([B[i].dstdata['h'], h_neigh], dim=1), self.weight_out)
                norm = torch.norm(h, dim=1)
                h = h / (norm.unsqueeze(-1) + 0.0001)
                B[i].dstdata['h'] = h
            else:
                h = torch.matmul(B[i - 1].dstdata['h'], self.weight_pool_hid)
                if self.bias:
                    h = h + self.bias_hid
                B[i].srcdata['h'] = h
                B[i].dstdata['h'] = h[:B[i].number_of_dst_nodes()]
                B[i].update_all(fn.copy_src('h', 'm'), fn.max('m', 'neigh'))
                h_neigh = B[i].dstdata['neigh']
                h = torch.matmul(torch.cat([B[i].srcdata['h'], h_neigh], dim=1), self.weight_hid[i-1, :, :])
                if self.activation:
                    h = self.activation(h, inplace=False)
                norm = torch.norm(h, dim=1)
                h = h / (norm.unsqueeze(-1) + 0.0001)
                B[i].dstdata['h'] = h
        return h

    def forward(self, x, B):
        if self.aggregator == 'gcn':
            z = self.gcn_agg(x, B)
        elif self.aggregator == 'mean':
            z = self.mean_agg(x, B)
        elif self.aggregator == 'lstm':
            z = self.lstm_agg(x, B)
        else:
            z = self.pool_agg(x, B)
        return z

    def gcn_full(self, g):
        x = g.ndata['features']
        x = self.dropout(x)
        g.srcdata['h'] = torch.matmul(x, self.weight_gcn_in)
        g.dstdata['h'] = x
        for i in range(self.K):
            if i == 0:
                g.update_all(fn.copy_src('h', 'm'), fn.sum('m', 'neigh'))
                degs = g.in_degrees().to(x)
                h_neigh = (g.dstdata['neigh'] + g.dstdata['h']) / (degs.unsqueeze(-1) + 1)
                h = torch.matmul(torch.cat([g.srcdata['h'], h_neigh], dim=1), self.weight_in)
                if self.activation:
                    h = self.activation(h, inplace=False)
                norm = torch.norm(h, dim=1)
                h = h / (norm.unsqueeze(-1) + 0.05)
                g.srcdata['h'] = h
                g.dstdata['h'] = h
            elif i == self.K - 1:
                g.srcdata['h'] = torch.matmul(g.srcdata['h'], self.weight_gcn_hid)
                g.update_all(fn.copy_src('h', 'm'), fn.sum('m', 'neigh'))
                degs = g.in_degrees().to(x)
                h_neigh = (g.dstdata['neigh'] + g.dstdata['h']) / (degs.unsqueeze(-1) + 1)
                h = torch.matmul(torch.cat([g.srcdata['h'], h_neigh], dim=1), self.weight_out)

                norm = torch.norm(h, dim=1)
                h = h / (norm.unsqueeze(-1) + 0.05)
                g.ndata['z'] = h
            else:
                g.srcdata['h'] = torch.matmul(g.srcdata['h'], self.weight_gcn_hid)
                g.update_all(fn.copy_src('h', 'm'), fn.sum('m', 'neigh'))
                degs = g.in_degrees().to(x)
                h_neigh = (g.dstdata['neigh'] + g.dstdata['h']) / (degs.unsqueeze(-1) + 1)
                h = torch.matmul(torch.cat([g.srcdata['h'], h_neigh], dim=1), self.weight_hid[i-1, :, :])
                if self.activation:
                    h = self.activation(h, inplace=False)
                norm = torch.norm(h, dim=1)
                h = h / (norm.unsqueeze(-1) + 0.05)
                g.srcdata['h'] = h
                g.dstdata['h'] = h
        return g

    def mean_full(self, g):
        x = g.ndata['features']
        x = self.dropout(x)
        g.srcdata['h'] = x
        for i in range(self.K):
            if i == 0:
                g.update_all(fn.copy_src('h', 'm'), fn.mean('m', 'neigh'))
                h_neigh = g.dstdata['neigh']
                h = torch.matmul(torch.cat([g.srcdata['h'], h_neigh], dim=1), self.weight_in)
                if self.activation:
                    h = self.activation(h, inplace=False)
                norm = torch.norm(h, dim=1)
                h = h / (norm.unsqueeze(-1) + 0.05)
                g.srcdata['h'] = h
            elif i == self.K - 1:
                g.update_all(fn.copy_src('h', 'm'), fn.mean('m', 'neigh'))
                h_neigh = g.dstdata['neigh']
                h = torch.matmul(torch.cat([g.srcdata['h'], h_neigh], dim=1), self.weight_out)

                norm = torch.norm(h, dim=1)
                h = h / (norm.unsqueeze(-1) + 0.05)
                g.ndata['z'] = h
            else:
                g.update_all(fn.copy_src('h', 'm'), fn.mean('m', 'neigh'))
                h_neigh = g.dstdata['neigh']
                h = torch.matmul(torch.cat([g.srcdata['h'], h_neigh], dim=1), self.weight_hid[i-1, :, :])
                if self.activation:
                    h = self.activation(h, inplace=False)
                norm = torch.norm(h, dim=1)
                h = h / (norm.unsqueeze(-1) + 0.05)
                g.srcdata['h'] = h
        return g

    def lstm_full(self, g):
        g.srcdata['h'] = g.ndata['features']
        for i in range(self.K):
            if i == 0:
                g.update_all(fn.copy_src('h', 'm'), self.lstm_reducer_in)
                h_neigh = g.dstdata['neigh']
                h = torch.matmul(torch.cat([g.srcdata['h'], h_neigh], dim=1), self.weight_in)
                if self.activation:
                    h = self.activation(h, inplace=False)
                norm = torch.norm(h, dim=1)
                h = h / (norm.unsqueeze(-1) + 0.05)
                g.srcdata['h'] = h
            elif i == self.K - 1:
                g.update_all(fn.copy_src('h', 'm'), self.lstm_reducer_hid)
                h_neigh = g.dstdata['neigh']
                h = torch.matmul(torch.cat([g.srcdata['h'], h_neigh], dim=1), self.weight_out)

                norm = torch.norm(h, dim=1)
                h = h / (norm.unsqueeze(-1) + 0.05)
                g.ndata['z'] = h
            else:
                g.update_all(fn.copy_src('h', 'm'), self.lstm_reducer_hid)
                h_neigh = g.dstdata['neigh']
                h = torch.matmul(torch.cat([g.srcdata['h'], h_neigh], dim=1), self.weight_hid[i-1, :, :])
                if self.activation:
                    h = self.activation(h, inplace=False)
                norm = torch.norm(h, dim=1)
                h = h / (norm.unsqueeze(-1) + 0.05)
                g.srcdata['h'] = h
        return g

    def pool_full(self, g):
        x = g.ndata['features']
        x = self.dropout(x)
        h = torch.matmul(x, self.weight_pool_in)
        if self.bias:
            h = h + self.bias_in
        g.srcdata['h'] = h
        for i in range(self.K):
            if i == 0:
                g.update_all(fn.copy_src('h', 'm'), fn.max('m', 'neigh'))
                h_neigh = g.dstdata['neigh']
                h = torch.matmul(torch.cat([g.srcdata['h'], h_neigh], dim=1), self.weight_in)
                if self.activation:
                    h = self.activation(h, inplace=False)
                norm = torch.norm(h, dim=1)
                h = h / (norm.unsqueeze(-1) + 0.05)
                g.srcdata['h'] = h
            elif i == self.K - 1:
                h = torch.matmul(g.srcdata['h'], self.weight_pool_hid)
                if self.bias:
                    h = h + self.bias_hid
                g.srcdata['h'] = h
                g.update_all(fn.copy_src('h', 'm'), fn.max('m', 'neigh'))
                h_neigh = g.dstdata['neigh']
                h = torch.matmul(torch.cat([g.srcdata['h'], h_neigh], dim=1), self.weight_out)

                norm = torch.norm(h, dim=1)
                h = h / (norm.unsqueeze(-1) + 0.05)
                g.ndata['z'] = h
            else:
                h = torch.matmul(g.srcdata['h'], self.weight_pool_hid)
                if self.bias:
                    h = h + self.bias_hid
                g.srcdata['h'] = h
                g.update_all(fn.copy_src('h', 'm'), fn.max('m', 'neigh'))
                h_neigh = g.dstdata['neigh']
                h = torch.matmul(torch.cat([g.srcdata['h'], h_neigh], dim=1), self.weight_hid[i-1, :, :])
                if self.activation:
                    h = self.activation(h, inplace=False)
                norm = torch.norm(h, dim=1)
                h = h / (norm.unsqueeze(-1) + 0.05)
                g.srcdata['h'] = h
        return g

    def infer(self, g):
        if self.aggregator == 'gcn':
            g = self.gcn_full(g)
        elif self.aggregator == 'mean':
            g = self.mean_full(g)
        elif self.aggregator == 'lstm':
            g = self.lstm_full(g)
        else:
            g = self.pool_full(g)
        z = g.ndata['z']
        return z