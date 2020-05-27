import torch
import torch.nn as nn
import dgl.function as fn


class FullGraphSAGE(nn.Module):
    def __init__(self, in_feat, hid_feat, out_feat, K, bias=False,
                 aggregator='mean', activation=None,
                 dropout=0.0, use_cuda=False):
        super(FullGraphSAGE, self).__init__()
        self.in_feat = in_feat
        self.hid_feat = hid_feat
        self.out_feat = out_feat
        self.K = K
        self.bias = bias
        self.aggregator = aggregator
        self.activation = activation

        self.weight_in = nn.Parameter(torch.Tensor(self.in_feat*2, self.hid_feat))
        nn.init.xavier_uniform_(self.weight_in, gain=nn.init.calculate_gain('relu'))
        if self.K > 2:
            self.weight_hid = nn.Parameter(torch.Tensor(self.K-2, self.hid_feat*2, self.hid_feat))
            nn.init.xavier_uniform_(self.weight_hid, gain=nn.init.calculate_gain('relu'))
        self.weight_out = nn.Parameter(torch.Tensor(self.hid_feat*2, self.out_feat))
        nn.init.xavier_uniform_(self.weight_out, gain=nn.init.calculate_gain('relu'))

        self.dropout= nn.Dropout(dropout)
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

    def gcn_agg(self, g):
        x = g.ndata['x']
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

    def mean_agg(self, g):
        x = g.ndata['x']
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

    def lstm_agg(self, g):
        g.srcdata['h'] = g.ndata['x']
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

    def pool_agg(self, g):
        x = g.ndata['x']
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

    def forward(self, g):
        if self.aggregator == 'gcn':
            g = self.gcn_agg(g)
        elif self.aggregator == 'mean':
            g = self.mean_agg(g)
        elif self.aggregator == 'lstm':
            g = self.lstm_agg(g)
        else:
            g = self.pool_agg(g)
        z = g.ndata['z']
        return z
