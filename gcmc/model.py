import torch.nn as nn

from layers import GraphCovLayer, EmbeddingLayer, BilinearDecoder


class Template(nn.Module):
    def __init__(self, u_num, v_num, in_dim, ins_dim, hid_dim, out_dim, num_of_ratings, num_of_bases,
                 norm='left', accum='sum', w_sharing=False, activation=None,
                 side_information=False, bias=False, dropout=0.0, use_cuda=False):
        super(Template, self).__init__()
        self.u_num = u_num
        self.v_num = v_num
        self.in_dim = in_dim
        self.ins_dim = ins_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.num_of_ratings = num_of_ratings
        self.num_of_bases = num_of_bases
        self.norm = norm
        self.accum = accum
        self.w_sharing = w_sharing
        self.activation = activation
        self.side_information = side_information
        self.bias = bias
        self.num_of_ratings = num_of_ratings
        self.dropout = dropout
        self.use_cuda = use_cuda
        # 构建模型
        self.build_model()

    def build_model(self):
        self.layers = nn.ModuleList()
        # i2h
        i2h = self.build_GConv_layer()
        if i2h is not None:
            self.layers.append(i2h)
        # h2z
        h2z = self.build_Emb_layer()
        if h2z is not None:
            self.layers.append(h2z)
        # z2o
        z2o  = self.build_Decode_layer()
        if z2o is not None:
            self.layers.append(z2o)

    def build_GConv_layer(self):
        return None

    def build_Emb_layer(self):
        return None

    def build_Decode_layer(self):
        return None

    def forward(self, x_u, x_v, a_u, a_v, d_u, d_v, h_u, h_v):
        for layer in self.layers:
            h_u, h_v = layer(x_u, x_v, a_u, a_v, d_u, d_v, h_u, h_v)
        return h_u, h_v


class EncoderNDecoder(Template):
    def build_GConv_layer(self):
        return GraphCovLayer(self.u_num, self.v_num, self.in_dim, self.hid_dim, self.num_of_ratings,
                             norm=self.norm, accum=self.accum, w_sharing=self.w_sharing, activation=self.activation,
                             dropout=self.dropout, use_cuda=self.use_cuda)

    def build_Emb_layer(self):
        return EmbeddingLayer(self.in_dim, self.ins_dim, self.hid_dim, self.out_dim,
                              activation=self.activation, side_information=self.side_information,
                              bias=self.bias, dropout=self.dropout, use_cuda=self.use_cuda)

    def build_Decode_layer(self):
        return BilinearDecoder(self.u_num, self.v_num, self.out_dim, self.num_of_ratings, self.num_of_bases,
                               w_sharing=self.w_sharing, use_cuda=self.use_cuda)


class GraphConvMatrixCompletion(nn.Module):
    def __init__(self, u_num, v_num, in_dim, ins_dim, hid_dim, out_dim, num_of_ratings, num_bases,
                 norm='left', accum='sum', w_sharing=False, activation=None,
                 side_information=False, bias=False, dropout=0.0, use_cuda=False):
        super(GraphConvMatrixCompletion, self).__init__()
        self.encoderNdecoder = EncoderNDecoder(u_num, v_num, in_dim, ins_dim, hid_dim, out_dim,
                                               num_of_ratings, num_bases, norm=norm,
                                    accum=accum, w_sharing=w_sharing, activation=activation,
                                    side_information=side_information, bias=bias,
                                    dropout=dropout, use_cuda=use_cuda)

    def forward(self, x_u, x_v, a_u, a_v, d_u, d_v, h_u, h_v):
        return self.encoderNdecoder.forward(x_u, x_v, a_u, a_v, d_u, d_v, h_u, h_v)
