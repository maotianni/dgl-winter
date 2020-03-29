import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn


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
        self.u_num = u_num
        self.v_num = v_num
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
        if self.w_sharing:
            new_weight = []
            for i in range(0, self.num_of_ratings):
                new_weight.append(th.sum(self.raw_weight[:i+1, :, :], dim=0, keepdim=True))
            self.weight = th.cat(new_weight)
        else:
            self.weight = self.raw_weight
        # activation
        self.activation = activation
        # dropout
        self.dropout = nn.Dropout(dropout)
        # cuda
        self.use_cuda = use_cuda

    def forward(self, g, h_u=None, h_v=None):
        funcs = {}
        for i in range(5):
            x_u = th.matmul(g.nodes['user'].data['x'], self.weight[i])
            x_v = th.matmul(g.nodes['item'].data['x'], self.weight[i])
            # right norm and dropout
            if self.norm == 'symmetric':
                x_u = x_u * g.nodes['user'].data['cj'].view(self.u_num, 1)
                x_v = x_v * g.nodes['item'].data['cj'].view(self.v_num, 1)
            if self.dropout:
                x_u = self.dropout(x_u)
                x_v = self.dropout(x_v)
            g.nodes['user'].data['h{}'.format(i + 1)] = x_u
            g.nodes['item'].data['h{}'.format(i + 1)] = x_v
            funcs['{}'.format(i + 1)] = (fn.copy_u('h{}'.format(i + 1), 'm'), fn.sum('m', 'h'))
            funcs['rev{}'.format(i + 1)] = (fn.copy_u('h{}'.format(i + 1), 'm'), fn.sum('m', 'h'))
        g.multi_update_all(funcs, self.accum)
        h_u = g.nodes['user'].data['h'].view(self.u_num, -1)
        h_v = g.nodes['item'].data['h'].view(self.v_num, -1)
        # print('h_u:{}'.format(sum(sum(th.isnan(h_u)))))
        # print('h_v:{}'.format(sum(sum(th.isnan(h_v)))))
        # left norm
        if self.norm == 'symmetric':
            h_u = h_u * g.nodes['user'].data['ci'].view(self.u_num, 1)
            h_v = h_v * g.nodes['item'].data['ci'].view(self.v_num, 1)
        else:
            h_u = h_u * g.nodes['user'].data['ci'].view(self.u_num, 1) * g.nodes['user'].data['ci'].view(self.u_num, 1)
            h_v = h_v * g.nodes['item'].data['ci'].view(self.v_num, 1) * g.nodes['item'].data['ci'].view(self.v_num, 1)
        # print('h_u, right norm:{}'.format(sum(sum(th.isnan(h_u)))))
        # print('h_v, right norm:{}'.format(sum(sum(th.isnan(h_v)))))
        if self.activation:
            h_u = self.activation(h_u, inplace=False)
            h_v = self.activation(h_v, inplace=False)
            # print('h_u, activation:{}'.format(sum(sum(th.isnan(h_u)))))
            # print('h_v, activation:{}'.format(sum(sum(th.isnan(h_v)))))
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

    def forward(self, g, h_u, h_v):
        # 遇到变量类型报错，尝试.（*变量类型），以转换到相应的变量类型
        # g = g.local_var()
        x_u = g.nodes['user'].data['x']
        x_v = g.nodes['item'].data['x']
        # h_u = g.nodes['user'].data['h']
        # h_v = g.nodes['item'].data['h']
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
            # print('f_u:{}'.format(sum(sum(th.isnan(f_u)))))
            # print('f_v:{}'.format(sum(sum(th.isnan(f_v)))))
            # activation
            if self.activation:
                f_u = self.activation(f_u, inplace=False)
                f_v = self.activation(f_v, inplace=False)
                # print('f_u:{}'.format(sum(sum(th.isnan(f_u)))))
                # print('f_v:{}'.format(sum(sum(th.isnan(f_v)))))
            z_u = th.matmul(h_u, self.weight) + th.matmul(f_u, self.weight_2_u)
            z_v = th.matmul(h_v, self.weight) + th.matmul(f_v, self.weight_2_v)
            # print('z_u:{}'.format(sum(sum(th.isnan(z_u)))))
            # print('z_v:{}'.format(sum(sum(th.isnan(z_v)))))
        else:
            z_u = th.matmul(h_u, self.weight)
            z_v = th.matmul(h_v, self.weight)
        # activation
        if self.activation:
            z_u = self.activation(z_u, inplace=False)
            z_v = self.activation(z_v, inplace=False)
            # print('z_u:{}'.format(sum(sum(th.isnan(z_u)))))
            # print('z_v:{}'.format(sum(sum(th.isnan(z_v)))))
        z_u = self.dropout(z_u)
        z_v = self.dropout(z_v)
        # print('z_u:{}'.format(sum(sum(th.isnan(z_u)))))
        # print('z_v:{}'.format(sum(sum(th.isnan(z_v)))))
        g.nodes['user'].data['z'] = z_u
        g.nodes['item'].data['z'] = z_v
        return z_u, z_v


class BilinearDecoder(nn.Module):
    def __init__(self, out_feat, num_of_ratings, num_of_bases, w_sharing=False, use_cuda=False):
        super(BilinearDecoder, self).__init__()
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
            self.Q_r = th.matmul(self.a_rs, self.P_s).view(self.num_of_ratings, self.out_feat, self.out_feat)
        else:
            self.Q_r = nn.Parameter(th.Tensor(self.num_of_ratings, self.out_feat, self.out_feat))
            nn.init.xavier_uniform_(self.Q_r, gain=nn.init.calculate_gain('relu'))
        if self.use_cuda:
            self.Q_r = self.Q_r.cuda()

    def calculate_p(self, g, z_u, z_v):
        p = []
        Q_r = self.Q_r
        for i in range(self.num_of_ratings):
            temp = th.matmul(z_u, Q_r[i])
            pi = th.matmul(temp, th.transpose(z_v, 1, 0))
            p.append(pi.view(1, g.number_of_nodes('user'), g.number_of_nodes('item')))
        P = th.cat(p)
        if self.use_cuda:
            P = P.cuda()
        return P

    def forward(self, g, h_u, h_v):
        # g = g.local_var()
        z_u = h_u
        z_v = h_v
        if self.use_cuda:
            z_u = z_u.cuda()
            z_v = z_v.cuda()
        p = self.calculate_p(g, z_u, z_v)
        # print('p:{}'.format(sum(sum(sum(th.isnan(p))))))
        P = F.softmax(p, dim=0)
        # print('P:{}'.format(sum(sum(sum(th.isnan(P))))))
        temp = None
        return P, temp

