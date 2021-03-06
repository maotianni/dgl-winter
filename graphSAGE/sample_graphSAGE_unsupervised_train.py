import scipy.sparse as sp
import pandas as pd
import numpy as np
import dgl
from dgl import DGLGraph
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
import tqdm
import time
import argparse
from pyinstrument import Profiler
from sample_graphSAGE_unsupervised import Unsuper_Cross_Entropy, Sample, GraphSAGE
from utils import LR_classification, Link_Prediction

def main(args):
    # graph
    coo_adj = sp.load_npz("reddit_self_loop/reddit_self_loop_graph.npz")
    graph = DGLGraph(coo_adj, readonly=True)
    # features and labels
    reddit_data = np.load("reddit_self_loop/reddit_data.npz")
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
    sampler = Sample(g, fanouts, args.num_neg)
    # 将数据集打乱顺序，分多个batch，每个batch采样两个B
    batch_size = args.batch_size
    num_workers = args.num_workers
    train_ids = torch.LongTensor(np.arange(g.number_of_edges()))
    dataloader = DataLoader(dataset=train_ids, batch_size=batch_size, collate_fn=sampler.obtain_Bs,
            shuffle=True, drop_last=False, num_workers=num_workers, pin_memory=True)

    # 设定模型
    num_hid = args.num_hidden
    ks = args.num_layers
    dropout_r = args.dropout
    agg = args.agg
    bias = args.bias
    norm = args.norm
    model = GraphSAGE(in_feats, num_hid, num_labels, ks, bias=bias,
                      aggregator=agg, activation=F.relu, norm=norm, dropout=dropout_r, use_cuda=use_cuda)
    if use_cuda:
        model.cuda()
    loss_fcn = Unsuper_Cross_Entropy()
    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # acc
    def compute_acc(logits, labels, train_nids, val_nids, test_nids, num_labels):
        # 输出标准化
        print('Computing accuracy...')
        logits = (logits - logits.mean(0)) / logits.std(0, unbiased=False)  # unbiased=False结果与numpy相同

        clf = LR_classification(num_labels, num_labels)
        if use_cuda:
            clf.cuda()
        criterion = nn.NLLLoss()
        optimizer = torch.optim.Adam(clf.parameters(), lr=0.003, weight_decay=1e-4)

        for epoch in range(10000):
            y_pred = clf(logits[train_nids])
            loss = criterion(y_pred, labels[train_nids])
            #if epoch % 500 == 0:
                #print('epoch: {} | loss: {}'.format(epoch, loss.item()))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        #print('epoch: {} | loss: {}'.format(epoch, loss.item()))

        clf.eval()
        with torch.no_grad():
            pred = clf.forward(logits)

        f1_micro_eval = (torch.argmax(pred[val_nids], dim=1) == labels[val_nids]).float().sum() / val_nids.shape[0]
        f1_micro_test = (torch.argmax(pred[test_nids], dim=1) == labels[test_nids]).float().sum() / test_nids.shape[0]
        return f1_micro_eval, f1_micro_test

    # eval
    def evaluation(model, g, labels, train_nids, val_nids, test_nids, batch_size, num_labels):
        model.eval()
        with torch.no_grad():
            logits = model.infer(g, batch_size)
        model.train()
        return compute_acc(logits, labels, train_nids, val_nids, test_nids, num_labels)

    # link-prediction
    def link_prediction(model, g, batch_size):
        model.eval()
        with torch.no_grad():
            logits = model.infer(g, batch_size)
        model.train()
        
        # 计算精度...
        precise = 0
        for start in tqdm.trange(0, g.number_of_edges(), batch_size):
            end = start + batch_size
            if end > g.number_of_edges():
                end = g.number_of_edges()
            ed_ids = torch.LongTensor(np.arange(start, end))
            heads, tails = g.find_edges(ed_ids)
            src = g.ndata['z'][heads, :]
            dst = g.ndata['z'][tails, :]
            y_pred = nn.Sigmoid()((src * dst).sum(dim=1))
            prec = (y_pred >= 0.5).float().sum()
            precise = precise + prec
        precision = precise / g.number_of_edges()
        return precision

    # 训练、验证与测试
    n_epochs = args.num_epochs
    log_every = args.log_every
    eval_every = args.eval_every
    iter_pos = []
    iter_neg = []
    iter_d = []
    iter_t = []
    best_eval_acc = 0
    best_test_acc = 0
    best_precision = 0
    for epoch in range(n_epochs):
        time_epoch_0 = time.time()
        time_step = time.time()
        for step, (pos_graph, neg_graph, blocks) in enumerate(dataloader):
            input_nodes = blocks[0].srcdata[dgl.NID]
            batch_inputs = g.ndata['features'][input_nodes]
            if use_cuda:
                batch_inputs = batch_inputs.cuda()
            time_load = time.time()

            batch_pred = model(batch_inputs, blocks)
            loss = loss_fcn(batch_pred, pos_graph, neg_graph, use_cuda)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            time_train = time.time()

            edge_pos = pos_graph.number_of_edges()
            edge_neg = neg_graph.number_of_edges()
            iter_pos.append(edge_pos / (time_train - time_step))
            iter_neg.append(edge_neg / (time_train - time_step))
            iter_d.append(time_load - time_step)
            iter_t.append(time_train - time_load)
            if step % log_every == 0:
                if step == 0:
                    print(
                        'Epoch {:05d} | Step {:05d} | Loss {:.4f} | '
                        'Speed (samples/sec) {:.4f} & {:.4f} | Load Time(sec) {:.4f} | Train Time(sec) {:.4f}'.format(
                            epoch, step, loss.item(), np.mean(iter_pos),
                            np.mean(iter_neg), np.mean(iter_d), np.mean(iter_t)))
                else:
                    print(
                        'Epoch {:05d} | Step {:05d} | Loss {:.4f} | '
                        'Speed (samples/sec) {:.4f} & {:.4f} | Load Time(sec) {:.4f} | Train Time(sec) {:.4f}'.format(
                            epoch, step, loss.item(), np.mean(iter_pos[3:]),
                            np.mean(iter_neg[3:]), np.mean(iter_d[3:]), np.mean(iter_t[3:])))
            time_step = time.time()
            if step % eval_every == 0:
                print('\n')
                print('Eval-ing...')
                time_ev_0 = time.time()
                if args.link:
                    precision = link_prediction(model, g, batch_size)
                    if precision > best_precision:
                        best_precision = precision
                    time_ev_1 = time.time()
                    print('Precision {:.4f} | Eval Time(s): {:.4f}'.format(precision, time_ev_1 - time_ev_0))
                    print('Best Precision {:.4f}'.format(best_precision))
                else:
                    eval_acc, test_acc = evaluation(model, g, labels, train_nid, val_nid, test_nid, batch_size, num_labels)
                    if eval_acc > best_eval_acc:
                        best_eval_acc = eval_acc
                        best_test_acc = test_acc
                    time_ev_1 = time.time()
                    print('Eval Acc {:.4f} | Eval Time(s): {:.4f}'.format(eval_acc, time_ev_1 - time_ev_0))
                    print('Best Eval Acc {:.4f} | Best Test Acc {:.4f}'.format(best_eval_acc, best_test_acc))
                time_step = time.time()
            #if step == 2:
                #break

        time_epoch_1 = time.time()
        print('Epoch Time(s): {:.4f}'.format(time_epoch_1 - time_epoch_0))
        #if epoch == 1:
            #break

    print('\n')
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
    argparser.add_argument('--batch-size', type=int, default=10000)
    argparser.add_argument('--log-every', type=int, default=20)
    argparser.add_argument('--eval-every', type=int, default=1000)
    argparser.add_argument('--lr', type=float, default=0.003)
    argparser.add_argument('--dropout', type=float, default=0.5)
    argparser.add_argument('--num-neg', type=int, default=1)
    argparser.add_argument('--num-workers', type=int, default=0,
        help="Number of sampling processes. Use 0 for no extra process.")
    argparser.add_argument("--bias", default=True, action='store_false',
                        help="bias")
    argparser.add_argument("--norm", default=False, action='store_true',
                           help="norm")
    argparser.add_argument("--link", default=False, action='store_true',
                           help="link prediction")
    args = argparser.parse_args()
    print(args)
    # 检测耗时
    #profiler = Profiler()
    #profiler.start()
    main(args)
    #profiler.stop()
    #print(profiler.output_text(unicode=True, color=True))
