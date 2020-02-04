import argparse
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from dgl.contrib.data import load_data

from R_GCN import RelGraphConv
import utils

class BaseRelGCN(nn.Module):
    def __init__(self, num_nodes, h_dim, out_dim, num_rels, num_bases,
                 regularizer='basis', norm='n',
                 num_hidden_layers=1, dropout=0,
                 use_self_loop=False, use_cuda=False):
        super(BaseRelGCN, self).__init__()
        self.num_nodes = num_nodes
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.regularizer = regularizer
        self.norm = norm
        self.num_bases = None if num_bases < 0 else num_bases
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.use_self_loop = use_self_loop
        self.use_cuda = use_cuda

        # 创建RGCN层
        self.build_model()

    def build_model(self):
        self.layers = nn.ModuleList()
        # i2h
        i2h = self.build_input_layer()
        if i2h is not None:
            self.layers.append(i2h)
        # h2h
        for idx in range(self.num_hidden_layers):
            h2h = self.build_hidden_layer(idx)
            self.layers.append(h2h)
        # h2o
        h2o = self.build_output_layer()
        if h2o is not None:
            self.layers.append(h2o)

    def build_input_layer(self):
        return None

    def build_hidden_layer(self, idx):
        raise NotImplementedError

    def build_output_layer(self):
        return None

    def forward(self, g, h, r, norm):
        for layer in self.layers:
            h = layer(g, h, r, norm)
        return h


class EmbeddingLayer(nn.Module):
    def __init__(self, num_nodes, h_dim):
        super(EmbeddingLayer, self).__init__()
        # 参考nn.Embedding源代码注释
        # 根据num_nodes和h_dim生成num_nodes * h_dim张量，同时以标准正态初始化
        # 生成num_nodes个h_dim维的量
        self.embedding = torch.nn.Embedding(num_nodes, h_dim)

    def forward(self, g, h, r, norm):
        # 输入节点，根据节点编号，结合上面的embedding，给与每一点以h_dim维的量
        return self.embedding(h.squeeze())

class RGCN(BaseRelGCN):
    def build_input_layer(self):
        return EmbeddingLayer(self.num_nodes, self.h_dim)

    def build_hidden_layer(self, idx):
        act = F.relu if idx < self.num_hidden_layers - 1 else None
        return RelGraphConv(self.h_dim, self.h_dim, self.num_rels, self.regularizer, self.norm,
                self.num_bases, activation=act, self_loop=True,
                dropout=self.dropout)

class LinkPredict(nn.Module):
    def __init__(self, in_dim, h_dim, num_rels, num_bases=-1, regularizer='basis', norm='n',
                 num_hidden_layers=1, dropout=0, use_cuda=False, reg_param=0):
        super(LinkPredict, self).__init__()
        self.rgcn = RGCN(in_dim, h_dim, h_dim, num_rels * 2, num_bases=num_bases, regularizer=regularizer, norm=norm,
                         num_hidden_layers=num_hidden_layers, dropout=dropout, use_cuda=use_cuda)
        self.reg_param = reg_param
        self.w_relation = nn.Parameter(torch.Tensor(num_rels, h_dim))
        nn.init.xavier_uniform_(self.w_relation,
                                gain=nn.init.calculate_gain('relu'))

    def calc_score(self, embedding, triplets):
        # DistMult
        s = embedding[triplets[:,0]]
        r = self.w_relation[triplets[:,1]]
        o = embedding[triplets[:,2]]
        score = torch.sum(s * r * o, dim=1)
        return score

    def forward(self, g, h, r, norm):
        return self.rgcn.forward(g, h, r, norm)

    def regularization_loss(self, embedding):
        return torch.mean(embedding.pow(2)) + torch.mean(self.w_relation.pow(2))

    def get_loss(self, g, embed, triplets, labels):
        # triplets is a list of data samples (positive and negative)
        # each row in the triplets is a 3-tuple of (source, relation, destination)
        score = self.calc_score(embed, triplets)
        predict_loss = F.binary_cross_entropy_with_logits(score, labels)
        reg_loss = self.regularization_loss(embed)
        return predict_loss + self.reg_param * reg_loss

def node_norm_to_edge_norm(g, node_norm):
    g = g.local_var()
    # convert to edge norm
    g.ndata['norm'] = node_norm
    g.apply_edges(lambda edges : {'norm' : edges.dst['norm']})
    return g.edata['norm']

def main(args):
    # load graph data
    data = load_data(args.dataset)
    num_nodes = data.num_nodes
    train_data = data.train
    valid_data = data.valid
    test_data = data.test
    num_rels = data.num_rels

    # check cuda
    gpu = args.gpu
    use_cuda = gpu >= 0 and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(gpu)

    # create model
    n_hidden = args.n_hidden
    n_bases = args.n_bases
    regularizer = args.regularizer
    norm = args.norm
    n_layers = args.n_layers
    dropout = args.dropout
    regularization = args.regularization
    model = LinkPredict(num_nodes,
                        n_hidden,
                        num_rels,
                        num_bases=n_bases,
                        regularizer=regularizer,
                        norm=norm,
                        num_hidden_layers=n_layers,
                        dropout=dropout,
                        use_cuda=use_cuda,
                        reg_param=regularization)

    # validation and testing triplets
    valid_data = torch.LongTensor(valid_data)
    test_data = torch.LongTensor(test_data)

    # build test graph
    test_graph, test_rel, test_norm = utils.build_test_graph(
        num_nodes, num_rels, train_data)
    test_deg = test_graph.in_degrees(
                range(test_graph.number_of_nodes())).float().view(-1,1)
    test_node_id = torch.arange(0, num_nodes, dtype=torch.long).view(-1, 1)
    test_rel = torch.from_numpy(test_rel)
    test_norm = node_norm_to_edge_norm(test_graph, torch.from_numpy(test_norm).view(-1, 1))

    if use_cuda:
        model.cuda()

    # build adj list and calculate degrees for sampling
    adj_list, degrees = utils.get_adj_and_degrees(num_nodes, train_data)

    # optimizer
    lr = args.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model_state_file = 'model_state.pth'
    forward_time = []
    backward_time = []

    # training loop
    graph_batch_size = args.graph_batch_size
    graph_split_size = args.graph_split_size
    negative_sample = args.negative_sample
    evaluate_every = args.evaluate_every
    edge_sampler = args.edge_sampler
    grad_norm = args.grad_norm
    eval_batch_size = args.eval_batch_size
    n_epochs = args.n_epochs
    print("start training...")

    epoch = 0
    best_mrr = 0
    while True:
        model.train()
        epoch += 1

        # perform edge neighborhood sampling to generate training graph and data
        g, node_id, edge_type, node_norm, data, labels = \
            utils.generate_sampled_graph_and_labels(
                train_data, graph_batch_size, graph_split_size,
                num_rels, adj_list, degrees, negative_sample,
                edge_sampler)
        print("Done edge sampling")

        # set node/edge feature
        node_id = torch.from_numpy(node_id).view(-1, 1).long()
        edge_type = torch.from_numpy(edge_type)
        edge_norm = node_norm_to_edge_norm(g, torch.from_numpy(node_norm).view(-1, 1))
        data, labels = torch.from_numpy(data), torch.from_numpy(labels)
        deg = g.in_degrees(range(g.number_of_nodes())).float().view(-1, 1)
        if use_cuda:
            node_id, deg = node_id.cuda(), deg.cuda()
            edge_type, edge_norm = edge_type.cuda(), edge_norm.cuda()
            data, labels = data.cuda(), labels.cuda()

        t0 = time.time()
        embed = model(g, node_id, edge_type, edge_norm)
        loss = model.get_loss(g, embed, data, labels)
        t1 = time.time()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm) # clip gradients
        optimizer.step()
        t2 = time.time()

        forward_time.append(t1 - t0)
        backward_time.append(t2 - t1)
        print("Epoch {:04d} | Loss {:.4f} | Best MRR {:.4f} | Forward {:.4f}s | Backward {:.4f}s".
                format(epoch, loss.item(), best_mrr, forward_time[-1], backward_time[-1]))

        optimizer.zero_grad()

        # validation
        if epoch % evaluate_every == 0:
            # perform validation on CPU because full graph is too large
            if use_cuda:
                model.cpu()
            model.eval()
            print("start eval")
            embed = model(test_graph, test_node_id, test_rel, test_norm)
            mrr = utils.calc_mrr(embed, model.w_relation, valid_data,
                                    hits=[1, 3, 10], eval_bz=eval_batch_size)
            # save best model
            if mrr < best_mrr:
                if epoch >= n_epochs:
                    break
            else:
                best_mrr = mrr
                torch.save({'state_dict': model.state_dict(), 'epoch': epoch},
                            model_state_file)
            if use_cuda:
                model.cuda()

    print("training done")
    print("Mean forward time: {:4f}s".format(np.mean(forward_time)))
    print("Mean Backward time: {:4f}s".format(np.mean(backward_time)))

    print("\nstart testing:")
    # use best model checkpoint
    checkpoint = torch.load(model_state_file)
    if use_cuda:
        model.cpu() # test on CPU
    model.eval()
    model.load_state_dict(checkpoint['state_dict'])
    print("Using best epoch: {}".format(checkpoint['epoch']))
    embed = model(test_graph, test_node_id, test_rel, test_norm)
    utils.calc_mrr(embed, model.w_relation, test_data,
                    hits=[1, 3, 10], eval_bz=eval_batch_size)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RGCN')
    parser.add_argument("--dropout", type=float, default=0.2,
            help="dropout probability")
    parser.add_argument("--n-hidden", type=int, default=500,
            help="number of hidden units")
    parser.add_argument("--gpu", type=int, default=-1,
            help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
            help="learning rate")
    parser.add_argument("--n-bases", type=int, default=100,
            help="number of weight blocks for each relation")
    parser.add_argument("--regularizer", type=str, default=None,
                        help="basis, bdd or None")
    parser.add_argument("--norm", type=str, default='n',
                        help="n, n2, sqrt or clamp")
    parser.add_argument("--n-layers", type=int, default=2,
            help="number of propagation rounds")
    parser.add_argument("--n-epochs", type=int, default=6000,
            help="number of minimum training epochs")
    parser.add_argument("-d", "--dataset", type=str, required=True,
            help="dataset to use")
    parser.add_argument("--eval-batch-size", type=int, default=500,
            help="batch size when evaluating")
    parser.add_argument("--regularization", type=float, default=0.01,
            help="regularization weight")
    parser.add_argument("--grad-norm", type=float, default=1.0,
            help="norm to clip gradient to")
    parser.add_argument("--graph-batch-size", type=int, default=30000,
            help="number of edges to sample in each iteration")
    parser.add_argument("--graph-split-size", type=float, default=0.5,
            help="portion of edges used as positive sample")
    parser.add_argument("--negative-sample", type=int, default=10,
            help="number of negative samples per positive sample")
    parser.add_argument("--evaluate-every", type=int, default=500,
            help="perform evaluation every n epochs")
    parser.add_argument("--edge-sampler", type=str, default="uniform",
            help="type of edge sampler: 'uniform' or 'neighbor'")

    args = parser.parse_args()
    print(args)
    main(args)
