from Modules import *


class TGNBasic(nn.Module):
    def __init__(self, in_feats_m, in_feats_u, in_feats_v, in_feats_t,
                 in_feats_e, in_feats_s, num_heads, activation, dropout=0.0, use_cuda=False):
        super(TGNBasic, self).__init__()
        self.in_feats_m = in_feats_m
        self.in_feats_u = in_feats_u
        self.in_feats_v = in_feats_v
        self.in_feats_t = in_feats_t
        self.in_feats_e = in_feats_e
        self.in_feats_s = in_feats_s
        self.num_heads = num_heads
        self.activation = activation
        self.dropout = dropout
        self.use_cuda = use_cuda
        # model
        self.compute_emb = NodeEmbeddingID(self.in_feats_u, self.in_feats_v, self.in_feats_t,
                                           self.in_feats_e, self.in_feats_s, self.num_heads,
                                           self.activation, self.dropout, self.use_cuda)
        self.evolve_memory = Message(self.in_feats_m, self.in_feats_s, self.in_feats_t)

    def forward(self, g, g_r, g_n, si, sj, sn, e, t, vi, vj, vn):
        return self.compute_emb(g, g_r, g_n, si, sj, sn, e, t, vi, vj, vn)

    def evolve_memory(self, g, g_r, si, sj, t, e):
        return self.evolve_memory(g, g_r, si, sj, t, e)
