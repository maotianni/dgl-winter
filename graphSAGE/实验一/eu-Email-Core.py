import pandas as pd
import numpy as np
import dgl
from dgl import DGLGraph
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time
import argparse
from pyinstrument import Profiler
from sample_graphSAGE_unsupervised import Unsuper_Cross_Entropy, Sample, GraphSAGE

def main(args):
    df = {'src': [], 'dst': [], 't': []}
    with open('data/email-Eu-core-temporal.txt', 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip('\n').split(' ')
            line = list(map(lambda x: int(x), line))
            df['src'].append(line[0])
            df['dst'].append(line[1])
            df['t'].append(line[2])
    edges = pd.DataFrame(df)
    del df
    nodes = pd.DataFrame({'nodeID': np.arange(0, pd.concat([edges['src'], edges['dst']], axis=0).max()+1).astype('object')})
    nodes = pd.get_dummies(nodes)
    features = nodes.values
    # tarin/val/test indices
    features = torch.Tensor(features)
    in_feats = features.shape[1]
    n_edges = edges.shape[0]
    # train / val / test
    test_split = np.floor(n_edges * 0.8).astype('int64')
    val_split = np.floor(test_split * 0.8).astype('int64')
    val_set = edges.iloc[: test_split, :]
    train_set = edges.iloc[:val_split, :]
    train_eid = torch.LongTensor(np.arange(0, val_split))
    val_eid = torch.LongTensor(np.arange(val_split, test_split))
    test_eid = torch.LongTensor(np.arange(test_split, n_edges))
    # graph
    graph = DGLGraph((train_set['src'].values, train_set['dst'].values))
    graph_v = DGLGraph((val_set['src'].values, val_set['dst'].values))
    graph_t = DGLGraph((edges['src'].values, edges['dst'].values))
    # 转为HetroGraph
    g = dgl.graph(graph.all_edges())
    g.ndata['features'] = features
    g_v = dgl.graph(graph_v.all_edges())
    g_v.ndata['features'] = features
    g_t = dgl.graph(graph_t.all_edges())
    g_t.ndata['features'] = features

    gpu = args.gpu
    use_cuda = gpu >= 0 and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(gpu)
        g.to(torch.device('cuda:{}'.format(gpu)))
        g_v.to(torch.device('cuda:{}'.format(gpu)))
        g_t.to(torch.device('cuda:{}'.format(gpu)))


    fanouts = list(map(int, args.fan_out.split(',')))
    sampler = Sample(g, fanouts, args.num_neg)
    # 将数据集打乱顺序，分多个batch，每个batch采样两个B
    batch_size = args.batch_size
    num_workers = args.num_workers
    dataloader = DataLoader(dataset=train_eid, batch_size=batch_size, collate_fn=sampler.obtain_Bs,
            shuffle=True, drop_last=False, num_workers=num_workers, pin_memory=True)

    # 设定模型
    num_hid = args.num_hidden
    ks = args.num_layers
    dropout_r = args.dropout
    agg = args.agg
    bias = args.bias
    norm = args.norm
    model = GraphSAGE(in_feats, num_hid, 16, ks, bias=bias,
                      aggregator=agg, activation=F.relu, norm=norm, dropout=dropout_r, use_cuda=use_cuda)
    if use_cuda:
        model.cuda()
    loss_fcn = Unsuper_Cross_Entropy()
    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # link-prediction
    def link_prediction(model, g_v, g_t, batch_size, val_id, test_id):
        model.eval()
        with torch.no_grad():
            logits_v = model.infer(g_v, batch_size)
            logits_t = model.infer(g_t, batch_size)
        model.train()
        # val
        heads_v, tails_v = g_v.find_edges(val_id)
        src_v = logits_v[heads_v, :]
        dst_v = logits_v[tails_v, :]
        y_pred_v = nn.Sigmoid()((src_v * dst_v).sum(dim=1))
        precision_v = (y_pred_v >= 0.5).float().sum() / val_id.shape[0]
        # test
        heads_t, tails_t = g_t.find_edges(test_id)
        src_t = logits_t[heads_t, :]
        dst_t = logits_t[tails_t, :]
        y_pred_t = nn.Sigmoid()((src_t * dst_t).sum(dim=1))
        precision_t = (y_pred_t >= 0.5).float().sum() / test_id.shape[0]
        return precision_v, precision_t

    # 训练、验证与测试
    n_epochs = args.num_epochs
    log_every = args.log_every
    eval_every = args.eval_every
    iter_pos = []
    iter_neg = []
    iter_d = []
    iter_t = []
    best_eval_precision = 0
    best_test_precision = 0
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
        print('\n')
        print('Eval-ing...')
        time_ev_0 = time.time()
        val_precision, test_precision = link_prediction(model, g_v, g_t, batch_size, val_eid, test_eid)
        if val_precision > best_eval_precision:
            best_eval_precision = val_precision
            best_test_precision = test_precision
        time_ev_1 = time.time()
        print('Eval Precision {:.4f} | Test Precision {:.4f} | Eval Time(s): {:.4f}'.format(val_precision,
                                                                                            test_precision,
                                                                                            time_ev_1 - time_ev_0))
        print('Best Eval Precision {:.4f} | Best Test Precision {:.4f}'.format(best_eval_precision,
                                                                                best_test_precision))

        time_epoch_1 = time.time()
        print('Epoch Time(s): {:.4f}'.format(time_epoch_1 - time_epoch_0))

    print('\n')
    print('Finish!')

if __name__ == '__main__':
    argparser = argparse.ArgumentParser("multi-gpu training")
    argparser.add_argument('--gpu', type=int, default=0,
        help="GPU device ID. Use -1 for CPU training")
    argparser.add_argument('--num-epochs', type=int, default=50)
    argparser.add_argument('--num-hidden', type=int, default=16)
    argparser.add_argument('--num-layers', type=int, default=2)
    argparser.add_argument('--fan-out', type=str, default='10,25')
    argparser.add_argument('--agg', type=str, default='mean')
    argparser.add_argument('--batch-size', type=int, default=1000)
    argparser.add_argument('--log-every', type=int, default=5)
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
    # profiler = Profiler()
    # profiler.start()
    main(args)
    # profiler.stop()
    # print(profiler.output_text(unicode=True, color=True))
