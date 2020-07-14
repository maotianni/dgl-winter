import pandas as pd
import numpy as np
import scipy.sparse as sp
import dgl
from dgl import DGLGraph
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score, roc_auc_score
import time
import argparse
from pyinstrument import Profiler
from sample_graphSAGE_unsupervised import Unsuper_Cross_Entropy, Sample, GraphSAGE

def main(args):
    df = {'src': [], 'dst': [], 't': []}
    with open('../data/CollegeMsg.txt', 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip('\n').split(' ')
            line = list(map(lambda x: int(x) - 1, line))
            df['src'].append(line[0])
            df['dst'].append(line[1])
            df['t'].append(line[2])
    edges = pd.DataFrame(df)
    del df
    nodes = pd.DataFrame({'nodeID': pd.concat([edges['src'], edges['dst']], axis=0).unique().astype('object')})
    nodes = pd.get_dummies(nodes)
    features = nodes.values
    # tarin/val/test indices
    features = torch.Tensor(features)
    in_feats = features.shape[1]
    n_edges = edges.shape[0]
    # train / val / test
    test_split = np.floor(n_edges * 0.8).astype('int64')
    val_split = np.floor(test_split * 0.8).astype('int64')
    train_set = edges.iloc[:val_split, :]
    val_set = edges.iloc[: test_split, :]
    train_eid = torch.LongTensor(np.arange(0, val_split))
    train_nds = pd.concat([train_set['src'], train_set['dst']], axis=0).unique()
    val_nds = pd.concat([val_set['src'], val_set['dst']], axis=0).unique()
    head, tail = edges['src'].values[: val_split], edges['dst'].values[: val_split]
    head_v, tail_v = edges['src'].values[val_split: test_split], edges['dst'].values[val_split: test_split]
    head_t, tail_t = edges['src'].values[test_split:], edges['dst'].values[test_split]
    # 验证集和测试集原有的边抹去
    coo_train = sp.coo_matrix((np.ones(train_set.shape[0]), (head, tail)),
                              shape=(train_nds.shape[0], train_nds.shape[0]))
    coo_val = sp.coo_matrix((np.ones(train_set.shape[0]), (head, tail)),
                            shape=(val_nds.shape[0], val_nds.shape[0]))
    coo_test = sp.coo_matrix((np.ones(val_set.shape[0]),
                              (np.concatenate([head, head_v]), np.concatenate([tail, tail_v]))),
                             shape=(nodes.shape[0], nodes.shape[0]))
    # graph
    g = dgl.graph(coo_train)
    g.ndata['features'] = features[train_nds, :]
    g_v = dgl.graph(coo_val)
    g_v.ndata['features'] = features[val_nds, :]
    g_t = dgl.graph(coo_test)
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
    def link_prediction(model, g_v, g_t, batch_size, head, tail, head_v, tail_v, head_t, tail_t):
        model.eval()
        with torch.no_grad():
            logits_v = model.infer(g_v, batch_size)
            logits_t = model.infer(g_t, batch_size)
        model.train()
        link_pre_res = {}
        # val
        y_pred_v = nn.Sigmoid()(torch.matmul(logits_v, logits_v.T))
        y_pred_v = y_pred_v.cpu().numpy()
        truth_v = np.zeros((g_v.number_of_nodes(), g_v.number_of_nodes()))
        truth_v[head, tail] = -1            # 训练集已有的边不管
        truth_v[head_v, tail_v] = 1
        y_pred_v = y_pred_v[np.where(truth_v > -1)].reshape(-1)
        truth_v = truth_v[np.where(truth_v > -1)].reshape(-1)
        link_pre_res['Val AP'] = average_precision_score(truth_v, y_pred_v)
        link_pre_res['Val AUC'] = roc_auc_score(truth_v, y_pred_v)
        # test
        y_pred_t = nn.Sigmoid()(torch.matmul(logits_t, logits_t.T))
        y_pred_t = y_pred_t.cpu().numpy()
        truth_t = np.zeros((g_t.number_of_nodes(), g_t.number_of_nodes()))
        truth_t[np.concatenate([head, head_v]), np.concatenate([tail, tail_v])] = -1
        truth_t[head_t, tail_t] = 1
        y_pred_t = y_pred_t[np.where(truth_t > -1)].reshape(-1)
        truth_t = truth_t[np.where(truth_t > -1)].reshape(-1)
        link_pre_res['Test AP'] = average_precision_score(truth_t, y_pred_t)
        link_pre_res['Test AUC'] = roc_auc_score(truth_t, y_pred_t)
        return link_pre_res

    # 训练、验证与测试
    n_epochs = args.num_epochs
    log_every = args.log_every
    eval_every = args.eval_every
    iter_pos = []
    iter_neg = []
    iter_d = []
    iter_t = []
    best_eval_ap = 0
    best_test_ap = 0
    best_eval_auc = 0
    best_test_auc = 0
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
        link_pre_res = link_prediction(model, g_v, g_t, batch_size, head, tail, head_v, tail_v, head_t, tail_t)
        if link_pre_res['Val AUC'] > best_eval_auc:
            best_eval_auc = link_pre_res['Val AUC']
            best_eval_ap = link_pre_res['Val AP']
            best_test_auc = link_pre_res['Test AUC']
            best_test_ap = link_pre_res['Test AP']
        time_ev_1 = time.time()
        print('Eval AUC {:.4f} | Eval AP {:.4f} | Test AUC {:.4f} |'
              ' Test AP {:.4f} | Eval Time(s): {:.4f}'.format(link_pre_res['Val AUC'],
                                                              link_pre_res['Val AP'],
                                                              link_pre_res['Test AUC'],
                                                              link_pre_res['Test AP'],
                                                              time_ev_1 - time_ev_0))
        print('Best Eval AUC {:.4f}, Eval AP {:.4f} |'
              ' Best Test AUC {:.4f}, Test AP {:.4f}'.format(best_eval_auc,
                                                             best_eval_ap,
                                                             best_test_auc,
                                                             best_test_ap))

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
