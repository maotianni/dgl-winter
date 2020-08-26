import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeEncode(torch.nn.Module):
    def __init__(self, expand_dim):
        super(TimeEncode, self).__init__()
        self.time_dim = expand_dim
        self.basis_freq = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.time_dim))).float(),
                                             requires_grad=False)
        self.phase = torch.nn.Parameter(torch.zeros(self.time_dim).float(), requires_grad=False)

    def forward(self, ts):
        batch_size = ts.size(0)
        ts = ts.view(batch_size, 1)  # [N, 1]
        map_ts = ts * self.basis_freq.view(1, -1)  # [N, time_dim]
        map_ts += self.phase.view(1, self.time_dim)
        # 三角变换
        harmonic = torch.cos(map_ts)
        return harmonic


# Basic TGN (Message)
class Message(nn.Module):
    def __init__(self, in_feats_m, in_feats_s, in_feats_t, activation, method='last', dropout=0.0, use_cuda=False):
        super(Message, self).__init__()
        self.in_feats_m = in_feats_m
        self.in_feats_s = in_feats_s
        self.in_feats_t = in_feats_t
        self.activation = activation
        self.method = method
        self.dropout = nn.Dropout(dropout)
        self.use_cuda = use_cuda
        # Memory
        self.gru = nn.GRU(self.in_feats_m, self.in_feats_s)
        self.time = TimeEncode(self.in_feats_t)

    def id_msg_func(self, edges):
        si = edges.src['memory']
        sj = edges.dst['memory']
        t = edges.data['t']
        time_emb = self.time(t)
        e = edges.data['e']
        out = torch.cat([si, sj, time_emb, e], dim=1)
        return {'mail': out}

    def last_reduce_func(self, nodes):
        msgs = nodes.mailbox['mail'][:, -1, :]
        return {'message': msgs}

    def mean_reduce_func(self, nodes):
        msgs = nodes.mailbox['mail']
        out = torch.mean(msgs, dim=1)
        return {'message': out}

    def forward(self, g, g_r, si, sj, si_r, sj_r, t, t_r, e, e_r):
        g.local_var()
        g_r.local_var()
        g.srcdata['memory'], g_r.srcdata['memory'] = si, si_r
        g.edata['t'], g_r.edata['t'] = t, t_r
        g.edata['e'], g_r.edata['e'] = e, e_r
        g.dstdata['memory'], g_r.dstdata['memory'] = sj, sj_r
        if self.method == 'last':
            g.update_all(message_func=self.id_msg_func, reduce_func=self.last_reduce_func)
            g_r.update_all(message_func=self.id_msg_func, reduce_func=self.last_reduce_func)
        elif self.method == 'mean':
            g.update_all(message_func=self.id_msg_func, reduce_func=self.mean_reduce_func)
            g_r.update_all(message_func=self.id_msg_func, reduce_func=self.mean_reduce_func)
        mj, mi = g.dstdata['message'], g_r.dstdata['message']
        _, si = self.gru(mi.view(1, g_r.number_of_nodes('user'), -1), sj_r.view(1, g_r.number_of_nodes('user'), -1))
        si = si.view(g_r.number_of_nodes('user'), -1)
        _, sj = self.gru(mj.view(1, g.number_of_nodes('item'), -1), sj.view(1, g.number_of_nodes('item'), -1))
        sj = sj.view(g.number_of_nodes('item'), -1)
        if self.dropout:
            si = self.dropout(si)
            sj = self.dropout(sj)
        #if self.activation:
            #si = self.activation(si)
            #sj = self.activation(sj)
        return si, sj


# Basic TGN (Embedding)
class NodeEmbeddingID(nn.Module):
    def __init__(self, in_feats_u, in_feats_v, in_feats_t, in_feats_e, in_feats_s, out_feats,
                 num_heads, activation, dropout=0.0, use_cuda=False):
        super(NodeEmbeddingID, self).__init__()
        self.in_feats_u = in_feats_u
        self.in_feats_v = in_feats_v
        self.in_feats_t = in_feats_t
        self.in_feats_e = in_feats_e
        self.in_feats_s = in_feats_s
        self.out_feats = out_feats
        self.num_heads = num_heads
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.use_cuda = use_cuda
        # Modules
        self.decompose_u = nn.Linear(self.in_feats_u, self.in_feats_s)
        self.decompose_v = nn.Linear(self.in_feats_v, self.in_feats_s)
        self.mlp_1 = nn.Linear(self.in_feats_s + self.in_feats_s + self.in_feats_t, self.in_feats_s)
        self.mlp_2 = nn.Linear(self.in_feats_s, self.out_feats)
        self.time = TimeEncode(self.in_feats_t)
        self.attention = nn.MultiheadAttention(embed_dim=self.in_feats_s + self.in_feats_t,
                                               num_heads=self.num_heads,
                                               kdim=self.in_feats_s + self.in_feats_e + self.in_feats_t,
                                               vdim=self.in_feats_s + self.in_feats_e + self.in_feats_t)

    # Message Passing
    # Message
    def attn_msg_func(self, edges):
        h = edges.src['h']
        e = edges.data['e']
        t0 = edges.dst['t0']
        t = edges.data['t']
        time_emb = self.time(t0 - t)
        out = torch.cat([h, e, time_emb], dim=1)
        return {'mail': out}

    def pass_timestamp(self, edges):
        timestamp = edges.data['t']
        return {'t_mail': timestamp}

    # reduce
    def sum_reduce_func(self, nodes):
        msgs = nodes.mailbox['mail']
        out = torch.sum(msgs, dim=1)
        return {'c': out}

    def reduce_timestamp(self, nodes):
        msgs = nodes.mailbox['t_mail']
        out = torch.max(msgs, dim=1)[0]
        return {'t0': out}

    def forward(self, g, g_r, g_n,
                si, sj, si_r, sj_r, si_n, sj_n,
                e, e_r, e_n, t, t_r, t_n,
                vi, vj, vi_r, vj_r, vi_n, vj_n):
        # decompose
        hi0, hj0 = self.decompose_u(vi), self.decompose_v(vj)
        hi0_r, hj0_r = self.decompose_v(vi_r), self.decompose_u(vj_r)
        hi0_n, hj0_n = self.decompose_u(vi_n), self.decompose_v(vj_n)
        # activation
        if self.activation:
            hi0, hj0 = self.activation(hi0), self.activation(hj0)
            hi0_r, hj0_r = self.activation(hi0_r), self.activation(hj0_r)
            hi0_n, hj0_n = self.activation(hi0_n), self.activation(hj0_n)
        hi, hj = hi0 + si, hj0 + sj
        hi_r, hj_r = hi0_r + si_r, hj0_r + sj_r
        hi_n, hj_n = hi0_n + si_n, hj0_n + sj_n
        # feats
        g.local_var()
        g_r.local_var()
        g_n.local_var()
        g.srcdata['h'], g_r.srcdata['h'], g_n.srcdata['h'] = hi, hi_r, hi_n
        g.edata['t'], g_r.edata['t'], g_n.edata['t'] = t, t_r, t_n
        g.edata['e'], g_r.edata['e'], g_n.edata['e'] = e, e_r, e_n
        # t0
        g.update_all(message_func=self.pass_timestamp, reduce_func=self.reduce_timestamp)
        g_r.update_all(message_func=self.pass_timestamp, reduce_func=self.reduce_timestamp)
        g_n.update_all(message_func=self.pass_timestamp, reduce_func=self.reduce_timestamp)
        # message passing
        g.update_all(message_func=self.attn_msg_func, reduce_func=self.sum_reduce_func)
        g_r.update_all(message_func=self.attn_msg_func, reduce_func=self.sum_reduce_func)
        g_n.update_all(message_func=self.attn_msg_func, reduce_func=self.sum_reduce_func)
        # Attention, j
        cj, ci, cn = g.dstdata['c'], g_r.dstdata['c'], g_n.dstdata['c']
        kj = vj = cj.view(1, -1, self.in_feats_s + self.in_feats_e + self.in_feats_t)
        ki = vi = ci.view(1, -1, self.in_feats_s + self.in_feats_e + self.in_feats_t)
        kn = vn = cn.view(1, -1, self.in_feats_s + self.in_feats_e + self.in_feats_t)
        tj, ti = torch.zeros((g.number_of_nodes('item'), 1)), torch.zeros((g_r.number_of_nodes('user'), 1))
        tn = torch.zeros((g_n.number_of_nodes('item'), 1))
        if self.use_cuda:
            tj, ti, tn = tj.cuda(), ti.cuda(), tn.cuda()
        tj, ti, tn = self.time(tj), self.time(ti), self.time(tn)
        qj, qi, qn = torch.cat([hj, tj], dim=1), torch.cat([hj_r, ti], dim=1), torch.cat([hj_n, tn], dim=1)
        qj = qj.view(1, -1, self.in_feats_s + self.in_feats_t)
        qi = qi.view(1, -1, self.in_feats_s + self.in_feats_t)
        qn = qn.view(1, -1, self.in_feats_s + self.in_feats_t)
        # compute attention
        hj_tmp, _ = self.attention(qj, kj, vj)
        hi_tmp, _ = self.attention(qi, ki, vi)
        hn_tmp, _ = self.attention(qn, kn, vn)
        hj_tmp = hj_tmp.view(-1, self.in_feats_s + self.in_feats_t)
        hi_tmp = hi_tmp.view(-1, self.in_feats_s + self.in_feats_t)
        hn_tmp = hn_tmp.view(-1, self.in_feats_s + self.in_feats_t)
        # compute embedding
        hj, hi, hn = torch.cat([hj, hj_tmp], dim=1), \
                     torch.cat([hj_r, hi_tmp], dim=1), \
                     torch.cat([hj_n, hn_tmp], dim=1)
        if self.dropout:
            hj, hi, hn = self.dropout(hj), self.dropout(hi), self.dropout(hn)
        hj, hi, hn = self.mlp_1(hj), self.mlp_1(hi), self.mlp_1(hn)
        hj, hi, hn = F.relu(hj), F.relu(hi), F.relu(hn)
        hj, hi, hn = self.mlp_2(hj), self.mlp_2(hi), self.mlp_2(hn)
        return hi, hj, hn


# Advanced TGN
class MessageNEmbedding(nn.Module):
    def __init__(self, in_feats_u, in_feats_v, in_feats_m, in_feats_t, in_feats_e, in_feats_s, out_feats,
                 num_heads, activation, method='last', dropout=0.0, use_cuda=False):
        super(MessageNEmbedding, self).__init__()
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
        self.dropout = nn.Dropout(dropout)
        self.use_cuda = use_cuda
        # Memory
        self.gru = nn.GRU(self.in_feats_m, self.in_feats_s)
        self.time = TimeEncode(self.in_feats_t)
        # Modules
        self.decompose_u = nn.Linear(self.in_feats_u, self.in_feats_s)
        self.decompose_v = nn.Linear(self.in_feats_v, self.in_feats_s)
        self.mlp_1 = nn.Linear(self.in_feats_s + self.in_feats_s + self.in_feats_t, self.in_feats_s)
        self.mlp_2 = nn.Linear(self.in_feats_s, self.out_feats)
        self.attention = nn.MultiheadAttention(embed_dim=self.in_feats_s + self.in_feats_t,
                                               num_heads=self.num_heads,
                                               kdim=self.in_feats_s + self.in_feats_e + self.in_feats_t,
                                               vdim=self.in_feats_s + self.in_feats_e + self.in_feats_t)

    # message & reduce
    ## Message
    ### m_raw
    def m_raw_compute(self, edges):
        si = edges.src['memory']
        sj = edges.dst['memory']
        t = edges.data['t']
        time_emb = self.time(t)
        e = edges.data['e']
        out = torch.cat([si, sj, time_emb, e], dim=1)
        return {'m_raw': out}

    ### m_raw_msg
    def m_raw_msg(self, edges):
        out = edges.data['m_raw']
        return {'m_mail': out}

    ### attention
    def attn_msg_func(self, edges):
        h = edges.src['h']
        e = edges.data['e']
        t0 = edges.dst['t0']
        t = edges.data['t']
        time_emb = self.time(t0 - t)
        out = torch.cat([h, e, time_emb], dim=1)
        return {'mail': out}

    ### t0
    def pass_timestamp(self, edges):
        timestamp = edges.data['t']
        return {'t_mail': timestamp}

    ## Reduce
    ### m_bar
    def m_bar_reduce(self, nodes):
        if self.method == 'last':
            msgs = nodes.mailbox['m_mail'][:, -1, :]
        else:
            msgs = torch.mean(nodes.mailbox['m_mail'], dim=1)
        return {'m_bar': msgs}

    ### attention
    def sum_reduce_func(self, nodes):
        msgs = nodes.mailbox['mail']
        out = torch.sum(msgs, dim=1)
        return {'c': out}

    ### time_emb
    def reduce_timestamp(self, nodes):
        msgs = nodes.mailbox['t_mail']
        out = torch.max(msgs, dim=1)[0]
        return {'t0': out}

    def forward(self, g, g_r, g_n,
                si, sj, si_r, sj_r, si_n, sj_n,
                m_raw, m_raw_r, m_raw_n,
                e, e_r, e_n,
                t, t_r, t_n,
                vi, vj, vi_r, vj_r, vi_n, vj_n):
        g.local_var()
        g_r.local_var()
        g_n.local_var()
        # feats
        g.edata['m_raw'], g_r.edata['m_raw'], g_n.edata['m_raw'] = m_raw, m_raw_r, m_raw_n
        g.edata['t'], g_r.edata['t'], g_n.edata['t'] = t, t_r, t_n
        g.edata['e'], g_r.edata['e'], g_n.edata['e'] = e, e_r, e_n
        # memory
        g.update_all(message_func=self.m_raw_msg, reduce_func=self.m_bar_reduce)
        g_r.update_all(message_func=self.m_raw_msg, reduce_func=self.m_bar_reduce)
        g_n.update_all(message_func=self.m_raw_msg, reduce_func=self.m_bar_reduce)
        mj, mi, mn = g.dstdata['m_bar'], g_r.dstdata['m_bar'], g_n.dstdata['m_bar']
        _, sj_r = self.gru(mi.view(1, g_r.number_of_nodes('user'), -1), sj_r.view(1, g_r.number_of_nodes('user'), -1))
        sj_r = sj_r.view(g_r.number_of_nodes('user'), -1)
        _, sj = self.gru(mj.view(1, g.number_of_nodes('item'), -1), sj.view(1, g.number_of_nodes('item'), -1))
        sj = sj.view(g.number_of_nodes('item'), -1)
        _, sj_n = self.gru(mn.view(1, g_n.number_of_nodes('item'), -1), sj_n.view(1, g_n.number_of_nodes('item'), -1))
        sj_n = sj_n.view(g_n.number_of_nodes('item'), -1)
        if self.dropout:
            si, sj = self.dropout(si), self.dropout(sj)
            si_r, sj_r = self.dropout(si_r), self.dropout(sj_r)
            si_n, sj_n = self.dropout(si_n), self.dropout(sj_n)
        # embedding
        ## decompose
        hi0, hj0 = self.decompose_u(vi), self.decompose_v(vj)
        hi0_r, hj0_r = self.decompose_v(vi_r), self.decompose_u(vj_r)
        hi0_n, hj0_n = self.decompose_u(vi_n), self.decompose_v(vj_n)
        ## activation
        if self.activation:
            hi0, hj0 = self.activation(hi0), self.activation(hj0)
            hi0_r, hj0_r = self.activation(hi0_r), self.activation(hj0_r)
            hi0_n, hj0_n = self.activation(hi0_n), self.activation(hj0_n)
        hi, hj = hi0 + si, hj0 + sj
        hi_r, hj_r = hi0_r + si_r, hj0_r + sj_r
        hi_n, hj_n = hi0_n + si_n, hj0_n + sj_n
        ## memory feats
        g.srcdata['memory'], g_r.srcdata['memory'], g_n.srcdata['memory'] = si, si_r, si_n
        g.dstdata['memory'], g_r.dstdata['memory'], g_n.dstdata['memory'] = sj, sj_r, sj_n
        ## other feats
        g.srcdata['h'], g_r.srcdata['h'], g_n.srcdata['h'] = hi, hi_r, hi_n
        ## t0
        g.update_all(message_func=self.pass_timestamp, reduce_func=self.reduce_timestamp)
        g_r.update_all(message_func=self.pass_timestamp, reduce_func=self.reduce_timestamp)
        g_n.update_all(message_func=self.pass_timestamp, reduce_func=self.reduce_timestamp)
        ## message passing
        g.update_all(message_func=self.attn_msg_func, reduce_func=self.sum_reduce_func)
        g_r.update_all(message_func=self.attn_msg_func, reduce_func=self.sum_reduce_func)
        g_n.update_all(message_func=self.attn_msg_func, reduce_func=self.sum_reduce_func)
        ## Attention, j
        cj, ci, cn = g.dstdata['c'], g_r.dstdata['c'], g_n.dstdata['c']
        kj = vj = cj.view(1, -1, self.in_feats_s + self.in_feats_e + self.in_feats_t)
        ki = vi = ci.view(1, -1, self.in_feats_s + self.in_feats_e + self.in_feats_t)
        kn = vn = cn.view(1, -1, self.in_feats_s + self.in_feats_e + self.in_feats_t)
        tj, ti = torch.zeros((g.number_of_nodes('item'), 1)), torch.zeros((g_r.number_of_nodes('user'), 1))
        tn = torch.zeros((g_n.number_of_nodes('item'), 1))
        if self.use_cuda:
            tj, ti, tn = tj.cuda(), ti.cuda(), tn.cuda()
        tj, ti, tn = self.time(tj), self.time(ti), self.time(tn)
        qj, qi, qn = torch.cat([hj, tj], dim=1), torch.cat([hj_r, ti], dim=1), torch.cat([hj_n, tn], dim=1)
        qj = qj.view(1, -1, self.in_feats_s + self.in_feats_t)
        qi = qi.view(1, -1, self.in_feats_s + self.in_feats_t)
        qn = qn.view(1, -1, self.in_feats_s + self.in_feats_t)
        ## compute attention
        hj_tmp, _ = self.attention(qj, kj, vj)
        hi_tmp, _ = self.attention(qi, ki, vi)
        hn_tmp, _ = self.attention(qn, kn, vn)
        hj_tmp = hj_tmp.view(-1, self.in_feats_s + self.in_feats_t)
        hi_tmp = hi_tmp.view(-1, self.in_feats_s + self.in_feats_t)
        hn_tmp = hn_tmp.view(-1, self.in_feats_s + self.in_feats_t)
        ## compute embedding
        hj, hi, hn = torch.cat([hj, hj_tmp], dim=1), \
                     torch.cat([hj_r, hi_tmp], dim=1), \
                     torch.cat([hj_n, hn_tmp], dim=1)
        if self.dropout:
            hj, hi, hn = self.dropout(hj), self.dropout(hi), self.dropout(hn)
        hj, hi, hn = self.mlp_1(hj), self.mlp_1(hi), self.mlp_1(hn)
        hj, hi, hn = F.relu(hj), F.relu(hi), F.relu(hn)
        hj, hi, hn = self.mlp_2(hj), self.mlp_2(hi), self.mlp_2(hn)
        # update message
        g.apply_edges(self.m_raw_compute)
        g_r.apply_edges(self.m_raw_compute)
        m_raw_i = g.edata['m_raw']
        m_raw_j = g_r.edata['m_raw']
        return hi, hj, hn, sj_r, sj, m_raw_i, m_raw_j

    def infer(self, g, g_r, g_n,
              si, sj, si_r, sj_r, si_n, sj_n,
              e, e_r, e_n, t, t_r, t_n,
              vi, vj, vi_r, vj_r, vi_n, vj_n):
        g.local_var()
        g_r.local_var()
        g_n.local_var()
        # feats
        g.edata['t'], g_r.edata['t'], g_n.edata['t'] = t, t_r, t_n
        g.edata['e'], g_r.edata['e'], g_n.edata['e'] = e, e_r, e_n
        g.srcdata['memory'], g_r.srcdata['memory'], g_n.srcdata['memory'] = si, si_r, si_n
        g.dstdata['memory'], g_r.dstdata['memory'], g_n.dstdata['memory'] = sj, sj_r, sj_n
        # update message
        g.apply_edges(self.m_raw_compute)
        g_r.apply_edges(self.m_raw_compute)
        g_n.apply_edges(self.m_raw_compute)
        # memory
        g.update_all(message_func=self.m_raw_msg, reduce_func=self.m_bar_reduce)
        g_r.update_all(message_func=self.m_raw_msg, reduce_func=self.m_bar_reduce)
        g_n.update_all(message_func=self.m_raw_msg, reduce_func=self.m_bar_reduce)
        mj, mi, mn = g.dstdata['m_bar'], g_r.dstdata['m_bar'], g_n.dstdata['m_bar']
        _, sj_r = self.gru(mi.view(1, g_r.number_of_nodes('user'), -1), sj_r.view(1, g_r.number_of_nodes('user'), -1))
        sj_r = sj_r.view(g_r.number_of_nodes('user'), -1)
        _, sj = self.gru(mj.view(1, g.number_of_nodes('item'), -1), sj.view(1, g.number_of_nodes('item'), -1))
        sj = sj.view(g.number_of_nodes('item'), -1)
        _, sj_n = self.gru(mn.view(1, g_n.number_of_nodes('item'), -1), sj_n.view(1, g_n.number_of_nodes('item'), -1))
        sj_n = sj_n.view(g_n.number_of_nodes('item'), -1)
        if self.dropout:
            si, sj = self.dropout(si), self.dropout(sj)
            si_r, sj_r = self.dropout(si_r), self.dropout(sj_r)
            si_n, sj_n = self.dropout(si_n), self.dropout(sj_n)
        # embedding
        ## decompose
        hi0, hj0 = self.decompose_u(vi), self.decompose_v(vj)
        hi0_r, hj0_r = self.decompose_v(vi_r), self.decompose_u(vj_r)
        hi0_n, hj0_n = self.decompose_u(vi_n), self.decompose_v(vj_n)
        ## activation
        if self.activation:
            hi0, hj0 = self.activation(hi0), self.activation(hj0)
            hi0_r, hj0_r = self.activation(hi0_r), self.activation(hj0_r)
            hi0_n, hj0_n = self.activation(hi0_n), self.activation(hj0_n)
        hi, hj = hi0 + si, hj0 + sj
        hi_r, hj_r = hi0_r + si_r, hj0_r + sj_r
        hi_n, hj_n = hi0_n + si_n, hj0_n + sj_n
        g.srcdata['h'], g_r.srcdata['h'], g_n.srcdata['h'] = hi, hi_r, hi_n
        ## t0
        g.update_all(message_func=self.pass_timestamp, reduce_func=self.reduce_timestamp)
        g_r.update_all(message_func=self.pass_timestamp, reduce_func=self.reduce_timestamp)
        g_n.update_all(message_func=self.pass_timestamp, reduce_func=self.reduce_timestamp)
        ## message passing
        g.update_all(message_func=self.attn_msg_func, reduce_func=self.sum_reduce_func)
        g_r.update_all(message_func=self.attn_msg_func, reduce_func=self.sum_reduce_func)
        g_n.update_all(message_func=self.attn_msg_func, reduce_func=self.sum_reduce_func)
        ## Attention, j
        cj, ci, cn = g.dstdata['c'], g_r.dstdata['c'], g_n.dstdata['c']
        kj = vj = cj.view(1, -1, self.in_feats_s + self.in_feats_e + self.in_feats_t)
        ki = vi = ci.view(1, -1, self.in_feats_s + self.in_feats_e + self.in_feats_t)
        kn = vn = cn.view(1, -1, self.in_feats_s + self.in_feats_e + self.in_feats_t)
        tj, ti = torch.zeros((g.number_of_nodes('item'), 1)), torch.zeros((g_r.number_of_nodes('user'), 1))
        tn = torch.zeros((g_n.number_of_nodes('item'), 1))
        if self.use_cuda:
            tj, ti, tn = tj.cuda(), ti.cuda(), tn.cuda()
        tj, ti, tn = self.time(tj), self.time(ti), self.time(tn)
        qj, qi, qn = torch.cat([hj, tj], dim=1), torch.cat([hj_r, ti], dim=1), torch.cat([hj_n, tn], dim=1)
        qj = qj.view(1, -1, self.in_feats_s + self.in_feats_t)
        qi = qi.view(1, -1, self.in_feats_s + self.in_feats_t)
        qn = qn.view(1, -1, self.in_feats_s + self.in_feats_t)
        ## compute attention
        hj_tmp, _ = self.attention(qj, kj, vj)
        hi_tmp, _ = self.attention(qi, ki, vi)
        hn_tmp, _ = self.attention(qn, kn, vn)
        hj_tmp = hj_tmp.view(-1, self.in_feats_s + self.in_feats_t)
        hi_tmp = hi_tmp.view(-1, self.in_feats_s + self.in_feats_t)
        hn_tmp = hn_tmp.view(-1, self.in_feats_s + self.in_feats_t)
        ## compute embedding
        hj, hi, hn = torch.cat([hj, hj_tmp], dim=1), \
                     torch.cat([hj_r, hi_tmp], dim=1), \
                     torch.cat([hj_n, hn_tmp], dim=1)
        if self.dropout:
            hj, hi, hn = self.dropout(hj), self.dropout(hi), self.dropout(hn)
        hj, hi, hn = self.mlp_1(hj), self.mlp_1(hi), self.mlp_1(hn)
        hj, hi, hn = F.relu(hj), F.relu(hi), F.relu(hn)
        hj, hi, hn = self.mlp_2(hj), self.mlp_2(hi), self.mlp_2(hn)
        return hi, hj, hn, sj_r, sj
