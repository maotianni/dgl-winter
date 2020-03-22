import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GraphCovLayer(nn.Module):
    def __init__(self,
                 in_feat,
                 hid_feat,
                 num_of_ratings,
                 w_sharing=False,
                 activation=None,
                 dropout=0.0):
        super(GraphCovLayer, self).__init__()
        self.in_feat = in_feat
        self.hid_feat = hid_feat
        self.num_of_ratings = num_of_ratings
        self.w_sharing = w_sharing
        # 初始化W_{r}的参数
        self.weight = nn.Parameter(th.Tensor(self.num_of_ratings, self.in_feat, self.hid_feat))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))
        # activation
        self.activation = activation
        # dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, g):
        g = g.local_var()
        u_s = g.edges()[0]
        v_s = g.edges()[1]
        h_u = th.zeros(g.number_of_nodes('user'), self.hid_feat)
        h_v = th.zeros(g.number_of_nodes('item'), self.hid_feat)
        if self.w_sharing:
            for i in range(1, self.num_of_ratings):
                self.weight[i] = self.weight[i] + self.weight[i - 1]
            weight = self.weight
        else:
            weight = self.weight
        # user --> item
        for i in range(g.number_of_nodes('user')):
            neighbour = np.argwhere(u_s == i)[0]
            u_ij = th.zeros(self.out_feat)
            for j in neighbour:
                index_w = g.edges['rate'].data['rate'][j]
                c_ij = len(np.intersect1d(neighbour, np.argwhere(g.edges['rate'].data['rate'] == index_w).T[0]))
                u_ij += th.matmul(g.nodes['item'].data['x'][v_s[j]], weight[index_w]) / c_ij
            h_u[i] = u_ij
            if self.activation:
                h_u[i] = self.activation(h_u[i])
        # item --> user
        for i in range(g.number_of_nodes('item')):
            neighbour = np.argwhere(v_s == i)[0]
            v_ij = th.zeros(self.out_feat)
            for j in neighbour:
                index_w = g.edges['rate'].data['rate'][j]
                c_ij = len(np.intersect1d(neighbour, np.argwhere(g.edges['rate'].data['rate'] == index_w).T[0]))
                v_ij += th.matmul(g.nodes['user'].data['x'][u_s[j]], weight[index_w]) / c_ij
            h_v[i] = v_ij
            if self.activation:
                h_v[i] = self.activation(h_v[i])
        h_u = self.dropout(h_u)
        h_v = self.dropout(h_v)
        g.nodes['user'].data['h'] = h_u
        g.nodes['item'].data['h'] = h_v
        return g


class EmbeddingLayer(nn.Module):
    def __init__(self, in_feat, hid_feat, out_feat,
                 activation=None, side_information=False, bias=False, dropout=0.0):
        super(EmbeddingLayer, self).__init__()
        self.in_feat = in_feat
        self.hid_feat = hid_feat
        self.out_feat = out_feat
        self.activation = activation
        self.side_information = side_information
        self.bias = bias
        self.dropout = nn.Dropout(dropout)
        # side information
        if self.side_information:
            self.weight_1_u = nn.Parameter(th.Tensor(self.in_feat, self.hid_feat))
            nn.init.xavier_uniform_(self.weight_1_u, gain=nn.init.calculate_gain('relu'))
            self.weight_2_u = nn.Parameter(th.Tensor(self.hid_feat, self.out_feat))
            nn.init.xavier_uniform_(self.weight_2_u, gain=nn.init.calculate_gain('relu'))
            self.weight_1_v = nn.Parameter(th.Tensor(self.in_feat, self.hid_feat))
            nn.init.xavier_uniform_(self.weight_1_v, gain=nn.init.calculate_gain('relu'))
            self.weight_2_v = nn.Parameter(th.Tensor(self.hid_feat, self.out_feat))
            nn.init.xavier_uniform_(self.weight_2_v, gain=nn.init.calculate_gain('relu'))
            self.weight = nn.Parameter(th.Tensor(self.hid_feat, self.out_feat))
            nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))
        else:
            self.weight = nn.Parameter(th.Tensor(self.hid_feat, self.out_feat))
            nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))
        # bias
        if self.bias:
            self.f_bias_u = nn.Parameter(th.Tensor(self.hid_feat))
            nn.init.zeros_(self.f_bias_u)
            self.f_bias_v = nn.Parameter(th.Tensor(self.hid_feat))
            nn.init.zeros_(self.f_bias_v)

    def forward(self, g):
        # 遇到变量类型报错，尝试.（*变量类型），以转换到相应的变量类型
        g = g.local_var()
        x_u = g.nodes['user'].data['x']
        x_v = g.nodes['item'].data['x']
        h_u = g.nodes['user'].data['h']
        h_v = g.nodes['item'].data['h']
        if self.side_information:
            f_u = th.matmul(x_u, self.weight_1_u)
            f_v = th.matmul(x_v, self.weight_1_v)
            if self.bias:
                f_u = f_u + self.f_bias_u
                f_v = f_v + self.f_bias_v
            # activation
            if self.activation:
                f_u = self.activation(f_u)
                f_v = self.activation(f_v)
            z_u = th.matmul(h_u, self.weight) + th.matmul(f_u, self.weight_2_u)
            z_v = th.matmul(h_v, self.weight) + th.matmul(f_v, self.weight_2_v)
        else:
            z_u = th.matmul(h_u, self.weight)
            z_v = th.matmul(h_v, self.weight)
        # activation
        if self.activation:
            z_u = self.activation(z_u)
            z_v = self.activation(z_v)
        z_u = self.dropout(z_u)
        z_v = self.dropout(z_v)
        g.nodes['user'].data['z'] = z_u
        g.nodes['item'].data['z'] = z_v
        return g


class BilinearDecoder(nn.Module):
    def __init__(self, out_feat, num_of_ratings, num_of_bases, w_sharing=False):
        super(BilinearDecoder, self).__init__()
        self.out_feat = out_feat
        self.num_of_ratings = num_of_ratings
        self.num_of_bases = num_of_bases
        self.w_sharing = w_sharing
        if self.w_sharing:
            self.P_s = nn.Parameter(th.Tensor(self.num_of_bases, self.out_feat, self.out_feat))
            nn.init.xavier_uniform_(self.P_s, gain=nn.init.calculate_gain('relu'))
            self.a_rs = nn.Parameter(th.Tensor(self.num_of_ratings, self.num_of_bases))
            nn.init.xavier_uniform_(self.a_rs, gain=nn.init.calculate_gain('relu'))
            self.P_s = self.P_s.view(self.num_of_bases, self.out_feat * self.out_feat)
            self.Q_r = th.matmul(self.a_rs, self.P_s).view(num_of_ratings, self.out_feat, self.out_feat)
        else:
            self.Q_r = nn.Parameter(th.Tensor(self.num_of_ratings, self.out_feat, self.out_feat))
            nn.init.xavier_uniform_(self.Q_r, gain=nn.init.calculate_gain('relu'))

    def forward(self, g):
        g = g.local_var()
        z_u = g.nodes['user'].data['z']
        z_v = g.nodes['item'].data['z']
        p = th.zeros(self.num_of_ratings, g.number_of_nodes('user'), g.number_of_nodes('item'))
        Q_r = self.Q_r
        for i in range(self.num_of_ratings):
            temp = th.matmul(z_u, Q_r[i])
            p[i] = th.matmul(temp, th.transpose(z_v, 1, 0))
        P = F.softmax(p, dim=0)
        return P
