from Modules import *


class TGNBasic(nn.Module):
    def __init__(self, in_feats_m, in_feats_u, in_feats_v, in_feats_t,
                 in_feats_e, in_feats_s, out_feats, num_heads, activation,
                 method='last', dropout=0.0, use_cuda=False):
        super(TGNBasic, self).__init__()
        self.in_feats_m = in_feats_m
        self.in_feats_u = in_feats_u
        self.in_feats_v = in_feats_v
        self.in_feats_t = in_feats_t
        self.in_feats_e = in_feats_e
        self.in_feats_s = in_feats_s
        self.out_feats = out_feats
        self.num_heads = num_heads
        self.activation = activation
        self.method= method
        self.dropout = dropout
        self.use_cuda = use_cuda
        # model
        self.compute_emb = NodeEmbeddingID(self.in_feats_u, self.in_feats_v, self.in_feats_t,
                                           self.in_feats_e, self.in_feats_s, self.out_feats, self.num_heads,
                                           self.activation, self.dropout, self.use_cuda)
        self.evolve_memory = Message(self.in_feats_m, self.in_feats_s, self.in_feats_t,
                                     self.activation, self.method, self.dropout, self.use_cuda)

    def forward(self, g, g_r, g_n,
                si, sj, si_r, sj_r, si_n, sj_n,
                e, e_r, e_n, t, t_r, t_n,
                vi, vj, vi_r, vj_r, vi_n, vj_n):
        return self.compute_emb(g, g_r, g_n,
                si, sj, si_r, sj_r, si_n, sj_n,
                e, e_r, e_n, t, t_r, t_n,
                vi, vj, vi_r, vj_r, vi_n, vj_n)

    def evolve(self, g, g_r, si, sj, si_r, sj_r, t, t_r, e, e_r):
        return self.evolve_memory(g, g_r, si, sj, si_r, sj_r, t, t_r, e, e_r)


class AdvancedTGN(nn.Module):
    def __init__(self, in_feats_u, in_feats_v, in_feats_m, in_feats_t, in_feats_e, in_feats_s, out_feats,
                 num_heads, activation, method='last', dropout=0.0, use_cuda=False):
        super(AdvancedTGN, self).__init__()
        self.in_feats_u = in_feats_u
        self.in_feats_v = in_feats_v
        self.in_feats_m = in_feats_m
        self.in_feats_t = in_feats_t
        self.in_feats_e = in_feats_e
        self.in_feats_s = in_feats_s
        self.out_feats = out_feats
        self.num_heads = num_heads
        self.activation = activation
        self.method = method
        self.dropout = dropout
        self.use_cuda = use_cuda
        # model
        self.advanced_tgn = MessageNEmbedding(self.in_feats_u, self.in_feats_v, self.in_feats_m, self.in_feats_t,
                                              self.in_feats_e, self.in_feats_s, self.out_feats,
                                              self.num_heads, self.activation,
                                              self.method, self.dropout, self.use_cuda)

    def forward(self, g, g_r, g_n,
                si, sj, si_r, sj_r, si_n, sj_n,
                m_raw, m_raw_r, m_raw_n,
                e, e_r, e_n,
                t, t_r, t_n,
                vi, vj, vi_r, vj_r, vi_n, vj_n):
        return self.advanced_tgn.forward(g, g_r, g_n,
                si, sj, si_r, sj_r, si_n, sj_n,
                m_raw, m_raw_r, m_raw_n,
                e, e_r, e_n,
                t, t_r, t_n,
                vi, vj, vi_r, vj_r, vi_n, vj_n)

    def infer(self, g, g_r, g_n,
              si, sj, si_r, sj_r, si_n, sj_n,
              e, e_r, e_n, t, t_r, t_n,
              vi, vj, vi_r, vj_r, vi_n, vj_n):
        return self.advanced_tgn.infer(g, g_r, g_n,
              si, sj, si_r, sj_r, si_n, sj_n,
              e, e_r, e_n, t, t_r, t_n,
              vi, vj, vi_r, vj_r, vi_n, vj_n)
