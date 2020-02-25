from scipy.io import loadmat
from scipy import sparse
from classify import *

import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
from dgl import DGLGraph

from functools import partial
import argparse
import time
import numpy as np


def main(args):
    # load graph data
    data = loadmat('data/ACM3025.mat')
    num_nodes = data['PTP'].shape[0]
    num_rels = 2
    labels = torch.from_numpy(data['label']).long()
    # feats = torch.from_numpy(data['feature']).float()
    feats = torch.arange(num_nodes)
    num_classes = labels.shape[1]
    labels = labels.nonzero()[:, 1]
    train_idx =  torch.from_numpy(data['train_idx']).long().squeeze(0)
    test_idx = torch.from_numpy(data['test_idx']).long().squeeze(0)
    val_idx = torch.from_numpy(data['val_idx']).long().squeeze(0)

    data['PAP'] = sparse.csr_matrix(data['PAP'] - np.eye(num_nodes))
    data['PLP'] = sparse.csr_matrix(data['PLP'] - np.eye(num_nodes))

    # graph
    g = DGLGraph()
    g.add_nodes(num_nodes)
    edge_list_1 = np.argwhere(data['PAP'] == 1)
    edge_list_2 = np.argwhere(data['PLP'] == 1)
    # idx_1 = list(range(0, edge_list_1.shape[0] // 2 * 2, 2))
    # idx_2 = list(range(0, edge_list_2.shape[0] // 220 * 220, 220))
    # edge_list_1 = edge_list_1[idx_1, :]
    # edge_list_2 = edge_list_2[idx_2, :]
    src_1, dst_1 = tuple(zip(*edge_list_1))
    src_2, dst_2 = tuple(zip(*edge_list_2))
    g.add_edges(src_1, dst_1)
    g.add_edges(src_2, dst_2)
    edge_type = torch.from_numpy(np.array([0] * edge_list_1.shape[0] + [1] * edge_list_2.shape[0]))
    edge_norm = None

    # check cuda
    gpu = args.gpu
    use_cuda = gpu >= 0 and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(gpu)
        feats = feats.cuda()
        edge_type = edge_type.cuda()
        labels = labels.cuda()


    # create model
    n_hidden = args.n_hidden
    n_bases = args.n_bases
    n_layers = args.n_layers
    regularizer = args.regularizer
    norm = args.norm
    dropout = args.dropout
    use_self_loop = args.use_self_loop
    model = EntityClassify(len(g),
                            n_hidden,
                            num_classes,
                            num_rels,
                            num_bases=n_bases,
                            regularizer=regularizer,
                            norm=norm,
                            num_hidden_layers=n_layers - 2,
                            dropout=dropout,
                            use_self_loop=use_self_loop,
                            use_cuda=use_cuda)

    if use_cuda:
        model.cuda()

    # optimizer
    lr = args.lr
    l2norm = args.l2norm
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2norm)

    # training loop
    n_epochs = args.n_epochs
    print("start training...")
    forward_time = []
    backward_time = []
    model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        t0 = time.time()
        logits = model(g, feats, edge_type, edge_norm)
        loss = F.cross_entropy(logits[train_idx], labels[train_idx].long())
        t1 = time.time()
        loss.backward()
        optimizer.step()
        t2 = time.time()

        forward_time.append(t1 - t0)
        backward_time.append(t2 - t1)
        print("Epoch {:05d} | Train Forward Time(s) {:.4f} | Backward Time(s) {:.4f}".
                format(epoch, forward_time[-1], backward_time[-1]))
        train_acc = torch.sum(logits[train_idx].argmax(dim=1) == labels[train_idx]).item() / len(train_idx)
        val_loss = F.cross_entropy(logits[val_idx], labels[val_idx].long())
        val_acc = torch.sum(logits[val_idx].argmax(dim=1) == labels[val_idx]).item() / len(val_idx)
        print("Train Accuracy: {:.4f} | Train Loss: {:.4f} | Validation Accuracy: {:.4f} | Validation loss: {:.4f}".
                format(train_acc, loss.item(), val_acc, val_loss.item()))
    print()

    model.eval()
    logits = model.forward(g, feats, edge_type, edge_norm)
    test_loss = F.cross_entropy(logits[test_idx], labels[test_idx].long())
    test_acc = torch.sum(logits[test_idx].argmax(dim=1) == labels[test_idx]).item() / len(test_idx)
    print("Test Accuracy: {:.4f} | Test loss: {:.4f}".format(test_acc, test_loss.item()))
    print()

    print("Mean forward time: {:4f}".format(np.mean(forward_time[len(forward_time) // 4:])))
    print("Mean backward time: {:4f}".format(np.mean(backward_time[len(backward_time) // 4:])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RGCN')
    parser.add_argument("--dropout", type=float, default=0,
            help="dropout probability")
    parser.add_argument("--n-hidden", type=int, default=16,
            help="number of hidden units")
    parser.add_argument("--gpu", type=int, default=-1,
            help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
            help="learning rate")
    parser.add_argument("--n-bases", type=int, default=-1,
            help="number of filter weight matrices, default: -1 [use all]")
    parser.add_argument("--n-layers", type=int, default=2,
            help="number of propagation rounds")
    parser.add_argument("--regularizer", type=str, default=None,
                        help="basis, bdd or None")
    parser.add_argument("--norm", type=str, default='n',
                        help="n, n2, sqrt or clamp")
    parser.add_argument("-e", "--n-epochs", type=int, default=50,
            help="number of training epochs")
    parser.add_argument("--l2norm", type=float, default=0,
            help="l2 norm coef")
    parser.add_argument("--relabel", default=False, action='store_true',
            help="remove untouched nodes and relabel")
    parser.add_argument("--use-self-loop", default=False, action='store_true',
            help="include self feature as a special relation")
    fp = parser.add_mutually_exclusive_group(required=False)
    fp.add_argument('--testing', dest='validation', action='store_false')
    parser.set_defaults(validation=True)

    args = parser.parse_args()
    print(args)
    args.bfs_level = args.n_layers + 1 # pruning used nodes for memory
    main(args)
