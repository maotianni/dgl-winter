import torch as th
import torch.nn as nn

import dgl.function as fn
from dgl.nn.pytorch import utils


class RelGraphConv(nn.Module):
    def __init__(self,
                 in_feat,
                 out_feat,
                 num_rels,
                 regularizer="basis",
                 norm = "n",
                 num_bases=None,
                 bias=True,
                 activation=None,
                 self_loop=False,
                 dropout=0.0):
        super(RelGraphConv, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_rels = num_rels
        self.regularizer = regularizer
        self.norm = norm
        self.num_bases = num_bases
        if self.num_bases is None or self.num_bases > self.num_rels or self.num_bases < 0:
            self.num_bases = self.num_rels
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop

        if regularizer == "basis":
            # add basis weights
            # 先初始化基函数矩阵的参数，再初始化arb的参数。
            # 根据Glorot, X.和Bengio, Y.在“Understanding the difficulty of training deep feedforward neural networks”
            # 中描述的方法，用一个均匀分布生成值，填充输入的张量或变量。
            # 结果张量中的值采样自U(-a, a)，其中a= gain * sqrt( 2/(fan_in + fan_out))* sqrt(3)
            # 其中relu的gain为sqrt(2)
            # xavier基本思想是通过网络层时，输入和输出的方差相同，包括前向传播和后向传播。
            self.weight = nn.Parameter(th.Tensor(self.num_bases, self.in_feat, self.out_feat))
            if self.num_bases < self.num_rels:
                # linear combination coefficients
                self.w_comp = nn.Parameter(th.Tensor(self.num_rels, self.num_bases))
            nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))
            if self.num_bases < self.num_rels:
                nn.init.xavier_uniform_(self.w_comp,
                                        gain=nn.init.calculate_gain('relu'))
            # message func
            self.message_func = self.basis_message_func
        elif regularizer == "bdd":
            if in_feat % num_bases != 0 or out_feat % num_bases != 0:
                raise ValueError('Feature size must be a multiplier of num_bases.')
            # add block diagonal weights
            self.submat_in = in_feat // self.num_bases
            self.submat_out = out_feat // self.num_bases

            # assuming in_feat and out_feat are both divisible by num_bases
            # 初始化，整个wr系数。但参数数量有不同，原来是self.num_bases * self.num_bases * self.submat_in * self.submat_out
            # 现在是self.num_bases * self.submat_in * self.submat_out
            self.weight = nn.Parameter(th.Tensor(
                self.num_rels, self.num_bases * self.submat_in * self.submat_out))
            nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))
            # message func
            self.message_func = self.bdd_message_func
        else:
            self.weight = nn.Parameter(th.Tensor(self.num_rels, self.in_feat, self.out_feat))
            nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))
            # message func
            self.message_func = self.basis_message_func
            # raise ValueError("Regularizer must be either 'basis' or 'bdd'")

        # bias
        if self.bias:
            self.h_bias = nn.Parameter(th.Tensor(out_feat))
            nn.init.zeros_(self.h_bias)

        # weight for self loop
        if self.self_loop:
            self.loop_weight = nn.Parameter(th.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight,
                                    gain=nn.init.calculate_gain('relu'))

        self.dropout = nn.Dropout(dropout)

    # 没有regularization时的消息传递函数
    def without_regular_message_func(self, edges):
        weight = self.weight
        msg = utils.bmm_maybe_select(edges.src['h'], weight, edges.data['type'].long())
        if 'norm' in edges.data:
            if self.norm == "n":
                msg = msg * edges.data['norm']
            elif self.norm == "n2":
                normm = th.pow(edges.data['norm'], 2)
                msg = msg * normm
            elif self.norm == "sqrt":
                normm = th.sqrt(edges.data['norm'])
                msg = msg * normm
            elif self.norm == "clamp":
                normm = th.clamp(edges.data['norm'], min=0.05)
                msg = msg * normm
        return {'msg': msg}

    def basis_message_func(self, edges):
        """Message function for basis regularizer"""
        if self.num_bases < self.num_rels:
            # generate all weights from bases
            # 压缩维度
            weight = self.weight.view(self.num_bases,
                                      self.in_feat * self.out_feat)
            # 矩阵相乘，回归维度，上一步也所维度为了矩阵点乘，下一步再重新返回，得到self.num_rels*self.in_feat*self.out_feat张量
            weight = th.matmul(self.w_comp, weight).view(
                self.num_rels, self.in_feat, self.out_feat)
        else:
            # 接__init__函数中的判断语句，如果base比rel多，则不进行basis.
            weight = self.weight

        # 根据edges.data['type']切割张量weight
        # 根据参数weight和edges.data['type']先选择相应的relation下的矩阵
        # 对于edges.src['h']，在第2个维度上增加一个维度，使原来的n*din变为n*1*din
        # 运用bmm算法，将edges.src['h']与weight相乘，得到n*1*dout维张量，最后去掉一维，得到n*dout维输出
        msg = utils.bmm_maybe_select(edges.src['h'], weight, edges.data['type'].long())
        if 'norm' in edges.data:
            if self.norm == "n":
                msg = msg * edges.data['norm']
            elif self.norm == "n2":
                normm = th.pow(edges.data['norm'], 2)
                msg = msg * normm
            elif self.norm == "sqrt":
                normm = th.sqrt(edges.data['norm'])
                msg = msg * normm
            elif self.norm == "clamp":
                normm = th.clamp(edges.data['norm'], min=0.05)
                msg = msg * normm
        return {'msg': msg}

    def bdd_message_func(self, edges):
        """Message function for block-diagonal-decomposition regularizer"""
        if edges.src['h'].dtype == th.int64 and len(edges.src['h'].shape) == 1:
            raise TypeError('Block decomposition does not allow integer ID feature.')
        # 选择每一个node的weight，将原本的n*self.num_bases * self.submat_in * self.submat_out维度，
        # 变为(n*self.num_bases) * self.submat_in * self.submat_out维度。
        weight = self.weight.index_select(0, edges.data['type'].long()).view(-1, self.submat_in, self.submat_out)
        node = edges.src['h'].view(-1, 1, self.submat_in)
        msg = th.bmm(node, weight).view(-1, self.out_feat)
        if 'norm' in edges.data:
            if self.norm == "n":
                msg = msg * edges.data['norm']
            elif self.norm == "n2":
                normm = th.pow(edges.data['norm'], 2)
                msg = msg * normm
            elif self.norm == "sqrt":
                normm = th.sqrt(edges.data['norm'])
                msg = msg * normm
            elif self.norm == "clamp":
                normm = th.clamp(edges.data['norm'], min=0.05)
                msg = msg * normm
        return {'msg': msg}

    def forward(self, g, x, etypes, norm=None):
        g = g.local_var()
        g.ndata['h'] = x
        g.edata['type'] = etypes
        if norm is not None:
            g.edata['norm'] = norm
        if self.self_loop:
            loop_message = utils.matmul_maybe_select(x, self.loop_weight)
        # message passing
        g.update_all(self.message_func, fn.sum(msg='msg', out='h'))
        # apply bias and activation
        node_repr = g.ndata['h']
        if self.bias:
            node_repr = node_repr + self.h_bias
        if self.self_loop:
            node_repr = node_repr + loop_message
        if self.activation:
            node_repr = self.activation(node_repr)
        node_repr = self.dropout(node_repr)
        return node_repr
