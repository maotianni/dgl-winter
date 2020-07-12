import torch as th
import torch.nn as nn
import torch.nn.functional as F


class GraphCovLayer(nn.Module):
    def __init__(self,
                 u_num,
                 v_num,
                 in_feat,
                 hid_feat,
                 num_of_ratings,
                 norm='left',
                 accum='sum',
                 w_sharing=False,
                 activation=None,
                 dropout=0.0,
                 use_cuda=False):
        super(GraphCovLayer, self).__init__()
        self.in_feat = in_feat
        self.hid_feat = hid_feat
        self.num_of_ratings = num_of_ratings
        self.w_sharing = w_sharing
        self.norm = norm
        # stack
        self.accum = accum
        if self.accum == 'stack':
            self.hid_feat = self.hid_feat // self.num_of_ratings
        # 初始化W_{r}的参数
        self.raw_weight = nn.Parameter(th.Tensor(self.num_of_ratings, self.in_feat, self.hid_feat))
        nn.init.xavier_uniform_(self.raw_weight, gain=nn.init.calculate_gain('relu'))
        # activation
        self.activation = activation
        # dropout
        self.dropout = nn.Dropout(dropout)
        # cuda
        self.use_cuda = use_cuda

    def forward(self, x_u, x_v, a_u, a_v, d_u, d_v, h_u=None, h_v=None):
        # 参数矩阵
        if self.w_sharing:
            new_weight = []
            for i in range(0, self.num_of_ratings):
                new_weight.append(th.sum(self.raw_weight[:i+1, :, :], dim=0, keepdim=True))
            weight = th.cat(new_weight)
        else:
            weight = self.raw_weight
        # 运算部分
        h_us = []
        h_vs = []
        for r in range(5):
            h_ur = th.matmul(x_v, weight[r, :, :])
            h_vr = th.matmul(x_u, weight[r, :, :])
            if self.norm == 'symmetric':
                h_ur = th.matmul(d_v, h_ur)
                h_vr = th.matmul(d_u, h_vr)
            h_ur = th.matmul(a_u[r], h_ur)
            h_vr = th.matmul(a_v[r], h_vr)
            if self.norm == 'symmetric':
                h_ur = th.matmul(d_u, h_ur)
                h_vr = th.matmul(d_v, h_vr)
            else:
                h_ur = th.matmul(d_u ** 2, h_ur)
                h_vr = th.matmul(d_v ** 2, h_vr)
            if self.dropout:
                h_ur = self.dropout(h_ur)
                h_vr = self.dropout(h_vr)
            h_us.append(h_ur)
            h_vs.append(h_vr)
        if self.accum == 'stack':
            h_u = th.cat(h_us, dim=1)
            h_v = th.cat(h_vs, dim=1)
        else:
            h_u = sum(h_us)
            h_v = sum(h_vs)
        if self.activation:
            h_u = self.activation(h_u, inplace=False)
            h_v = self.activation(h_v, inplace=False)
        return h_u, h_v


class EmbeddingLayer(nn.Module):
    def __init__(self, in_feat, ins_feat, hid_feat, out_feat,
                 activation=None, side_information=False, bias=False, dropout=0.0, use_cuda=False):
        super(EmbeddingLayer, self).__init__()
        self.in_feat = in_feat
        self.ins_feat = ins_feat
        self.hid_feat = hid_feat
        self.out_feat = out_feat
        self.activation = activation
        self.side_information = side_information
        self.bias = bias
        self.dropout = nn.Dropout(dropout)
        self.use_cuda = use_cuda
        # side information
        if self.side_information:
            self.weight_1_u = nn.Parameter(th.Tensor(self.in_feat, self.ins_feat))
            nn.init.xavier_uniform_(self.weight_1_u, gain=nn.init.calculate_gain('relu'))
            self.weight_2_u = nn.Parameter(th.Tensor(self.ins_feat, self.out_feat))
            nn.init.xavier_uniform_(self.weight_2_u, gain=nn.init.calculate_gain('relu'))
            self.weight_1_v = nn.Parameter(th.Tensor(self.in_feat, self.ins_feat))
            nn.init.xavier_uniform_(self.weight_1_v, gain=nn.init.calculate_gain('relu'))
            self.weight_2_v = nn.Parameter(th.Tensor(self.ins_feat, self.out_feat))
            nn.init.xavier_uniform_(self.weight_2_v, gain=nn.init.calculate_gain('relu'))
            self.weight = nn.Parameter(th.Tensor(self.hid_feat, self.out_feat))
            nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))
        else:
            self.weight = nn.Parameter(th.Tensor(self.hid_feat, self.out_feat))
            nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))
        # bias
        if self.bias:
            self.f_bias_u = nn.Parameter(th.Tensor(self.ins_feat))
            nn.init.zeros_(self.f_bias_u)
            self.f_bias_v = nn.Parameter(th.Tensor(self.ins_feat))
            nn.init.zeros_(self.f_bias_v)

    def forward(self, x_u, x_v, a_u, a_v, d_u, d_v, h_u, h_v):
        # 遇到变量类型报错，尝试.（*变量类型），以转换到相应的变量类型
        if self.use_cuda:
            x_u = x_u.cuda()
            x_v = x_v.cuda()
            h_u = h_u.cuda()
            h_v = h_v.cuda()
        if self.side_information:
            f_u = th.matmul(x_u, self.weight_1_u)
            f_v = th.matmul(x_v, self.weight_1_v)
            if self.bias:
                f_u = f_u + self.f_bias_u
                f_v = f_v + self.f_bias_v
            # activation
            if self.activation:
                f_u = self.activation(f_u, inplace=False)
                f_v = self.activation(f_v, inplace=False)
            z_u = th.matmul(h_u, self.weight) + th.matmul(f_u, self.weight_2_u)
            z_v = th.matmul(h_v, self.weight) + th.matmul(f_v, self.weight_2_v)
        else:
            z_u = th.matmul(h_u, self.weight)
            z_v = th.matmul(h_v, self.weight)
        # activation
        if self.activation:
            z_u = self.activation(z_u, inplace=False)
            z_v = self.activation(z_v, inplace=False)
        z_u = self.dropout(z_u)
        z_v = self.dropout(z_v)
        return z_u, z_v


class BilinearDecoder(nn.Module):
    def __init__(self, u_num, v_num, out_feat, num_of_ratings, num_of_bases, w_sharing=False, use_cuda=False):
        super(BilinearDecoder, self).__init__()
        self.u_num = u_num
        self.v_num = v_num
        self.out_feat = out_feat
        self.num_of_ratings = num_of_ratings
        self.num_of_bases = num_of_bases
        self.w_sharing = w_sharing
        self.use_cuda = use_cuda
        if self.w_sharing:
            self.P_s = nn.Parameter(th.Tensor(self.num_of_bases, self.out_feat * self.out_feat))
            nn.init.xavier_uniform_(self.P_s, gain=nn.init.calculate_gain('relu'))
            self.a_rs = nn.Parameter(th.Tensor(self.num_of_ratings, self.num_of_bases))
            nn.init.xavier_uniform_(self.a_rs, gain=nn.init.calculate_gain('relu'))
        else:
            self.Q_r = nn.Parameter(th.Tensor(self.num_of_ratings, self.out_feat, self.out_feat))
            nn.init.xavier_uniform_(self.Q_r, gain=nn.init.calculate_gain('relu'))

    def calculate_p(self, z_u, z_v):
        p = []
        # 参数矩阵
        if self.w_sharing:
            Q_r = th.matmul(self.a_rs, self.P_s).view(self.num_of_ratings, self.out_feat, self.out_feat)
        else:
            Q_r = self.Q_r
        # 运算部分
        for i in range(self.num_of_ratings):
            temp = th.matmul(z_u, Q_r[i, :, :])
            pi = th.matmul(temp, th.transpose(z_v, 1, 0))
            p.append(pi.view(1, self.u_num, self.v_num))
        P = th.cat(p)
        if self.use_cuda:
            P = P.cuda()
        return P

    def forward(self, x_u, x_v, a_u, a_v, d_u, d_v, h_u, h_v):
        z_u = h_u
        z_v = h_v
        if self.use_cuda:
            z_u = z_u.cuda()
            z_v = z_v.cuda()
        p = self.calculate_p(z_u, z_v)
        P = F.softmax(p, dim=0)
        temp = None
        return P, temp
