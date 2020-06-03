import scipy.sparse as sp
import numpy as np
import dgl
from dgl import DGLGraph
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time
import argparse
from sample_graphSAGE import Sample, GraphSAGE


def main(args):
    # graph
    coo_adj = sp.load_npz("reddit/reddit_graph.npz")
    graph = DGLGraph(coo_adj, readonly=True)
    # features and labels
    reddit_data = np.load("reddit/reddit_data.npz")
    features = reddit_data["feature"]
    labels = reddit_data["label"]
    num_labels = 41
    # tarin/val/test indices
    node_ids = reddit_data["node_ids"]
    node_types = reddit_data["node_types"]
    train_mask = (node_types == 1)
    val_mask = (node_types == 2)
    test_mask = (node_types == 3)
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask
    graph.ndata['feat'] = features
    graph.ndata['label'] = labels
    features = torch.Tensor(features)
    in_feats = features.shape[1]
    labels = torch.LongTensor(labels)
    train_nid = torch.LongTensor(np.where(train_mask==True)[0])
    train_mask = torch.BoolTensor(train_mask)
    val_nid = torch.LongTensor(np.where(val_mask==True)[0])
    val_mask = torch.BoolTensor(val_mask)
    test_nid = torch.LongTensor(np.where(test_mask==True)[0])
    test_mask = torch.BoolTensor(test_mask)

    g = dgl.graph(graph.all_edges())         # 转为HetroGraph
    g.ndata['features'] = features

    gpu = args.gpu
    use_cuda = gpu >= 0 and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(gpu)
        g.to(torch.device('cuda:{}'.format(gpu)))
        labels = labels.cuda()

    fanouts = list(map(int, args.fan_out.split(',')))
    sampler = Sample(g, fanouts)
    # 将数据集打乱顺序，分多个batch，每个batch采样两个B
    batch_size = args.batch_size
    num_workers = args.num_workers
    dataloader = DataLoader(dataset=train_nid.numpy(), batch_size=batch_size, collate_fn=sampler.obtain_Bs,
            shuffle=True, drop_last=False, num_workers=num_workers)

    # 设定模型
    num_hid = args.num_hidden
    ks = args.num_layers
    dropout_r = args.dropout
    agg = args.agg
    bias = args.bias
    model = GraphSAGE(in_feats, num_hid, num_labels, ks, bias=bias,
                      aggregator=agg,activation=F.relu, dropout=dropout_r)
    if use_cuda:
        model.cuda()
    loss_fcn = torch.nn.CrossEntropyLoss()
    # use optimizer
    l_r = args.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=l_r)

    # acc
    def compute_acc(pred, labels):
        acc = (torch.argmax(pred, dim=1) == labels).float().sum() / len(pred)
        return acc
    # eval
    def evaluation(model, g, labels, id, batch_size):
        model.eval()
        with torch.no_grad():
            logits = model.infer(g, batch_size)
            pred = logits[id]
            label = labels[id]
        model.train()
        return compute_acc(pred, label)

    # 训练、验证与测试
    n_epochs = args.num_epochs
    log_every = args.log_every
    eval_every = args.eval_every
    avg = 0
    iter_tput = []
    for epoch in range(n_epochs):
        time_epoch_0 = time.time()
        for step, blocks in enumerate(dataloader):
            time_step = time.time()

            input_nodes = blocks[0].srcdata[dgl.NID]
            seeds = blocks[-1].dstdata[dgl.NID]              # 最后一个block的dst为batch中的节点

            batch_inputs = g.ndata['features'][input_nodes]
            batch_labels = labels[seeds]
            if use_cuda:
                batch_inputs = batch_inputs.cuda()
                batch_labels = batch_labels.cuda()

            batch_pred = model(batch_inputs, blocks)
            loss = loss_fcn(batch_pred, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_tput.append(len(seeds) / (time.time() - time_step))
            if step % log_every == 0:
                acc = compute_acc(batch_pred, batch_labels)
                print(
                    'Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | Speed (samples/sec) {:.4f}'.format(
                        epoch, step, loss.item(), acc.item(), np.mean(iter_tput[3:])))

        time_epoch_1 = time.time()
        print('Epoch Time(s): {:.4f}'.format(time_epoch_1 - time_epoch_0))
        if epoch >= 5:
            avg += time_epoch_1 - time_epoch_0
        if epoch % eval_every == 0 and epoch != 0:
            print('\n')
            print('Eval-ing...')
            time_ev_0 = time.time()
            eval_acc = evaluation(model, g, labels, val_mask, batch_size)
            time_ev_1 = time.time()
            print('Eval Acc {:.4f} | Eval Time(s): {:.4f}'.format(eval_acc, time_ev_1 - time_ev_0))
    print('\n')
    print('Eval-ing...')
    time_ev_0 = time.time()
    eval_acc = evaluation(model, g, labels, val_mask, batch_size)
    time_ev_1 = time.time()
    print('Eval Acc {:.4f} | Eval Time(s): {:.4f}'.format(eval_acc, time_ev_1 - time_ev_0))
    print('\n')
    print('Testing...')
    time_ev_0 = time.time()
    test_acc = evaluation(model, g, labels, test_mask, batch_size)
    time_ev_1 = time.time()
    print('Test Acc {:.4f} | Eval Time(s): {:.4f}'.format(test_acc, time_ev_1 - time_ev_0))
    print('Finish!')

if __name__ == '__main__':
    argparser = argparse.ArgumentParser("multi-gpu training")
    argparser.add_argument('--gpu', type=int, default=0,
        help="GPU device ID. Use -1 for CPU training")
    argparser.add_argument('--num-epochs', type=int, default=20)
    argparser.add_argument('--num-hidden', type=int, default=16)
    argparser.add_argument('--num-layers', type=int, default=2)
    argparser.add_argument('--fan-out', type=str, default='10,25')
    argparser.add_argument('--agg', type=str, default='mean')
    argparser.add_argument('--batch-size', type=int, default=1000)
    argparser.add_argument('--log-every', type=int, default=20)
    argparser.add_argument('--eval-every', type=int, default=5)
    argparser.add_argument('--lr', type=float, default=0.003)
    argparser.add_argument('--dropout', type=float, default=0.5)
    argparser.add_argument('--num-workers', type=int, default=0,
        help="Number of sampling processes. Use 0 for no extra process.")
    argparser.add_argument("--bias", default=True, action='store_false',
                        help="bias")
    args = argparser.parse_args()
    print(args)
    main(args)
