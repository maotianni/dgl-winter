import numpy as np
import dgl
import torch
import torch.nn as nn
# 其他
import time
import argparse
# 自定义
import Model
import utils
import Data


def future_task(n_epochs, batch_size, batch_size_test, use_cuda, gpu,
                model, nc, loss_nc, opt_nc, sampler, num_neg,
                n_users, n_items, in_feats_s, out_feats,
                n_train, n_val,
                g_v, g_t,
                head, tail, head_v, tail_v, head_t, tail_t,
                features_u, features_v, features_e, t, label,
                log_every, eval_every, validation, advanced=False):
    iter_pos = []
    iter_neg = []
    iter_d = []
    iter_t = []
    eval_res = {'eval_auc': [], 'best_eval_auc': 0, 'test_auc': [], 'best_test_auc': 0}
    for epoch in range(n_epochs):
        time_epoch_0 = time.time()
        # each epoch
        si, sj = torch.zeros(n_users, in_feats_s), torch.zeros(n_items, in_feats_s)
        #zi, zj = torch.zeros(n_users, out_feats), torch.zeros(n_items, out_feats)
        ## cuda
        if use_cuda:
            si, sj = si.cuda(), sj.cuda()
            #zi, zj = zi.cuda(), zj.cuda()
        time_step = time.time()
        for start in range(0, head.shape[0], batch_size):
            step = start // batch_size + 1
            end = start + batch_size
            if end > head.shape[0]:
                end = head.shape[0]
            head_b = head[start:end]
            tail_b = tail[start:end]
            # sample
            pos_graph, pos_graph_r, neg_graph, \
            pos_graph_v, neg_graph_v, \
            extra_v_u_id, extra_u_v_id, extra_neg_id = sampler.obtain_Bs(head_b, tail_b, start)
            ## cuda
            if use_cuda:
                pos_graph.to(torch.device('cuda:{}'.format(gpu)))
                pos_graph_r.to(torch.device('cuda:{}'.format(gpu)))
                neg_graph.to(torch.device('cuda:{}'.format(gpu)))
                pos_graph_v.to(torch.device('cuda:{}'.format(gpu)))
                neg_graph_v.to(torch.device('cuda:{}'.format(gpu)))
            # id
            head_id = pos_graph.srcdata[dgl.NID]
            tail_id = pos_graph.dstdata[dgl.NID]
            head_id_r = pos_graph_r.srcdata[dgl.NID]
            tail_id_r = pos_graph_r.dstdata[dgl.NID]
            head_id_neg = neg_graph.srcdata[dgl.NID]
            tail_id_neg = neg_graph.dstdata[dgl.NID]
            head_id_out = pos_graph_v.srcdata[dgl.NID]
            tail_id_out = pos_graph_v.dstdata[dgl.NID]
            # input
            si_b, sj_b = si[head_id], sj[tail_id]
            si_b_r, sj_b_r = sj[head_id_r], si[tail_id_r]
            si_b_n, sj_b_n = si[head_id_neg], sj[tail_id_neg]
            vi_b, vj_b = features_u[head_id], features_v[tail_id]
            vi_b_r, vj_b_r = features_v[head_id_r], features_u[tail_id_r]
            vi_b_n, vj_b_n = features_u[head_id_neg], features_v[tail_id_neg]
            e_b = torch.cat([features_e[extra_u_v_id], features_e[start: end]], dim=0)
            e_b_r = torch.cat([features_e[extra_v_u_id], features_e[start: end]], dim=0)
            e_b_n = torch.cat([features_e[extra_neg_id], features_e[start: end]], dim=0)
            t_b = torch.cat([t[extra_u_v_id], t[start: end]])
            t_b_r = torch.cat([t[extra_v_u_id], t[start: end]])
            t_b_n = torch.cat([t[extra_neg_id], t[start: end]])
            time_load = time.time()
            # forward
            if advanced:
                zi_b, zj_b, zn_b, si_b2, sj_b2 = model.infer(pos_graph, pos_graph_r, neg_graph,
                                                             si_b, sj_b, si_b_r, sj_b_r, si_b_n, sj_b_n,
                                                             e_b, e_b_r, e_b_n, t_b, t_b_r, t_b_n,
                                                             vi_b, vj_b, vi_b_r, vj_b_r, vi_b_n, vj_b_n)
            else:
                zi_b, zj_b, zn_b = model.forward(pos_graph, pos_graph_r, neg_graph,
                                                 si_b, sj_b, si_b_r, sj_b_r, si_b_n, sj_b_n,
                                                 e_b, e_b_r, e_b_n, t_b, t_b_r, t_b_n,
                                                 vi_b, vj_b, vi_b_r, vj_b_r, vi_b_n, vj_b_n)
                si_b2, sj_b2 = model.evolve(pos_graph, pos_graph_r, si_b, sj_b, si_b_r, sj_b_r, t_b, t_b_r, e_b, e_b_r)
            # save
            si[head_id_out], sj[tail_id_out] = si_b2, sj_b2
            #zi[head_id_out], zj[tail_id_out] = zi_b, zj_b
            # node
            zi_l, zj_l = zi_b.detach(), zj_b.detach()
            score = nc(zi_l, zj_l, pos_graph_v)
            target = label[start:end].float()
            loss_2 = loss_nc(score, target)
            opt_nc.zero_grad()
            loss_2.backward()
            opt_nc.step()
            # log time
            time_train = time.time()
            # log
            edge_pos = pos_graph.number_of_edges()
            edge_neg = neg_graph.number_of_edges()
            iter_pos.append(edge_pos / (time_train - time_step))
            iter_neg.append(edge_neg / (time_train - time_step))
            iter_d.append(time_load - time_step)
            iter_t.append(time_train - time_load)
            if step % log_every == 0:
                if step // 3 == 0:
                    print(
                        'Epoch {:05d} | Step {:05d} | NCLoss {:.4f} | '
                        'Speed (samples/sec) {:.4f} & {:.4f} | Load Time(sec) {:.4f} | Train Time(sec) {:.4f}'.format(
                            epoch + 1, step, loss_2.item(), np.mean(iter_pos),
                            np.mean(iter_neg), np.mean(iter_d), np.mean(iter_t)))
                else:
                    print(
                        'Epoch {:05d} | Step {:05d} | NCLoss {:.4f} | '
                        'Speed (samples/sec) {:.4f} & {:.4f} | Load Time(sec) {:.4f} | Train Time(sec) {:.4f}'.format(
                            epoch + 1, step, loss_2.item(), np.mean(iter_pos[3:]),
                            np.mean(iter_neg[3:]), np.mean(iter_d[3:]), np.mean(iter_t[3:])))
            time_step = time.time()
        print('\n')
        print('Embedding Has Been Trained!')
        if epoch % eval_every == 0:
            print('Start Evaling...')
            time_ev_0 = time.time()
            if validation:
                node_cls_res = utils.node_class(model, nc, utils.Sample(g_v, num_negs=num_neg, num_nei=args.num_nei),
                                                n_train,
                                                head_v, tail_v, batch_size_test,
                                                features_u[:n_train+n_val], features_v[:n_train+n_val],
                                                features_e[:n_train+n_val], t[:n_train+n_val], label[:n_train+n_val],
                                                n_users, n_items, in_feats_s, out_feats,use_cuda, gpu)
                if epoch > 0:
                    eval_res['eval_auc'].append(node_cls_res['AUC'])
                if epoch >= 20 and node_cls_res['AUC'] < min(eval_res['eval_auc'][-10:]):
                    break
                if epoch > 0 and node_cls_res['AUC'] > eval_res['best_eval_auc']:
                    eval_res['best_eval_auc'] = node_cls_res['AUC']
                    print('Testing...')
                    test_res = utils.node_class(model, nc, utils.Sample(g_t, num_negs=num_neg, num_nei=args.num_nei),
                                                n_train+n_val,
                                                head_t, tail_t, batch_size_test, features_u, features_v,
                                                features_e, t, label, n_users, n_items, in_feats_s, out_feats,
                                                use_cuda, gpu, advanced)
                    eval_res['best_test_auc'] = test_res['AUC']
                time_ev_1 = time.time()
                print('Eval AUC {:.4f} | Eval Time(s): {:.4f}'.format(node_cls_res['AUC'],
                                                                      time_ev_1 - time_ev_0))
                print('Best Eval AUC {:.4f} | Best Test AUC {:.4f}'.format(eval_res['best_eval_auc'],
                                                                           eval_res['best_test_auc']))
            else:
                node_cls_res = utils.node_class(model, nc, utils.Sample(g_t, num_negs=num_neg, num_nei=args.num_nei),
                                                n_train,
                                                head_t, tail_t, batch_size_test, features_u, features_v, features_e,
                                                t, label, n_users, n_items, in_feats_s, out_feats,
                                                use_cuda, gpu, advanced)
                if epoch > 0:
                    eval_res['test_auc'].append(node_cls_res['AUC'])
                if epoch >= 20 and node_cls_res['AUC'] < min(eval_res['test_auc'][-10:]):
                    break
                if epoch > 0 and node_cls_res['AUC'] > eval_res['best_test_auc']:
                    eval_res['best_test_auc'] = node_cls_res['AUC']
                time_ev_1 = time.time()
                print('Test AUC {:.4f} | Eval Time(s): {:.4f}'.format(node_cls_res['AUC'],
                                                                      time_ev_1 - time_ev_0))
                print('Best Test AUC {:.4f}'.format(eval_res['best_test_auc']))
            print('\n')

        time_epoch_1 = time.time()
        print('Epoch Time(s): {:.4f}'.format(time_epoch_1 - time_epoch_0))


# 直接运行脚本时使用：
def main(args):
    # validation
    validation = args.validation
    # data
    dl = Data.LoadData(args.dataset, validation=validation)
    n_users = dl.n_users
    n_items = dl.n_items
    u_feats = dl.u_feats
    v_feats = dl.v_feats
    features_u = dl.features_u
    features_v = dl.features_v
    features_e = dl.e_feats
    t = dl.t
    label = dl.label
    # head & tail
    if validation:
        head, tail = dl.train
        head_v, tail_v = dl.val
        head_t, tail_t = dl.test
        n_train = dl.n_train
        n_val = dl.n_val
        n_test = dl.n_test
    else:
        head, tail = dl.train
        head_t, tail_t = dl.test
        n_train = dl.n_train
        n_test = dl.n_test
    # graph
    g = dgl.bipartite(list(zip(head, tail)), 'user', 'edit', 'item', num_nodes=(n_users, n_items))
    if validation:
        g_v = dgl.bipartite(list(zip(np.concatenate([head, head_v]), np.concatenate([tail, tail_v]))),
                            'user', 'edit', 'item', num_nodes=(n_users, n_items))
        g_t = dgl.bipartite(list(zip(np.concatenate([head, head_v, head_t]), np.concatenate([tail, tail_v, tail_t]))),
                            'user', 'edit', 'item', num_nodes=(n_users, n_items))
    else:
        g_t = dgl.bipartite(list(zip(np.concatenate([head, head_t]), np.concatenate([tail, tail_t]))),
                            'user', 'edit', 'item', num_nodes=(n_users, n_items))
    # cuda
    gpu = args.gpu
    use_cuda = gpu >= 0 and torch.cuda.is_available()
    if use_cuda:
        features_u = features_u.cuda()
        features_v = features_v.cuda()
        features_e = features_e.cuda()
        t = t.cuda()
        label = label.cuda()
        g.to(torch.device('cuda:{}'.format(gpu)))
        if validation:
            g_v.to(torch.device('cuda:{}'.format(gpu)))
        g_t.to(torch.device('cuda:{}'.format(gpu)))

    # sampler
    sampler = utils.Sample(g, num_negs=args.num_neg, num_nei=args.num_nei)

    # batch, train
    n_edges = g.number_of_edges()
    learn = args.learn
    batch_size = args.batch_size
    batch_size_test = args.batch_size_test
    num_heads = args.num_heads
    in_feats_u = features_u.shape[1]
    in_feats_v = features_v.shape[1]
    in_feats_t = args.in_feats_t
    in_feats_e = features_e.shape[1]
    in_feats_s = args.in_feats_s
    in_feats_m = in_feats_s * 2 + in_feats_t + in_feats_e
    out_feats = args.out_feats
    # model, loss function, optimizer
    if args.advanced:
        model = Model.AdvancedTGN(in_feats_u, in_feats_v, in_feats_m, in_feats_t, in_feats_e, in_feats_s, out_feats,
                                  num_heads, activation=torch.tanh, method=args.message,
                                  dropout=args.dropout, use_cuda=use_cuda)
    else:
        model = Model.TGNBasic(in_feats_m, in_feats_u, in_feats_v, in_feats_t,
                               in_feats_e, in_feats_s, out_feats,
                               num_heads, activation=torch.tanh, method=args.message,
                               dropout=args.dropout, use_cuda=use_cuda)
    if use_cuda:
        model.cuda()
    nc = utils.GraphNC(out_feats * 2, 80, 10, args.dropout)
    if use_cuda:
        nc.cuda()
    nc.train()
    opt_nc = torch.optim.Adam(nc.parameters(), lr=args.lr_p, weight_decay=args.decay)
    loss_nc = nn.BCELoss()
    # training loop
    n_epochs = args.n_epochs
    log_every = args.log_every
    eval_every = args.eval_every
    # load params
    if args.advanced:
        model.load_state_dict(torch.load('best_params_advanced_{}.pth'.format(args.dataset)))
        model.eval()
    else:
        model.load_state_dict(torch.load('best_params_{}.pth'.format(args.dataset)))
        model.eval()
    if validation:
        future_task(n_epochs, batch_size, batch_size_test, use_cuda, gpu,
                    model, nc, loss_nc, opt_nc, sampler, args.num_neg,
                    n_users, n_items, in_feats_s, out_feats,
                    n_train, n_val, g_v, g_t,
                    head, tail, head_v, tail_v, head_t, tail_t,
                    features_u, features_v, features_e, t, label,
                    log_every, eval_every, validation, args.advanced)
    else:
        future_task(n_epochs, batch_size, batch_size_test, use_cuda, gpu,
                    model, nc, loss_nc, opt_nc, sampler, args.num_neg,
                    n_users, n_items, in_feats_s, out_feats,
                    n_train, n_train, g_t, g_t,
                    head, tail, head_t, tail_t, head_t, tail_t,
                    features_u, features_v, features_e, t, label,
                    log_every, eval_every, validation, args.advanced)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser("TGN Reference")
    argparser.add_argument('--gpu', type=int, default=-1,
                           help="GPU device ID. Use -1 for CPU training")
    argparser.add_argument('--dataset', type=str, default='wikipedia',
                           help='wikipedia or reddit')
    argparser.add_argument("--validation", default=False, action='store_true',
                           help="validation")
    argparser.add_argument("--advanced", default=False, action='store_true',
                           help="advanced")
    argparser.add_argument('--learn', type=str, default='None',
                           help='future tasks, Link Prediction or Node Classification')
    argparser.add_argument('--message', type=str, default='last',
                           help='reduce function')
    argparser.add_argument('--batch-size', type=int, default=100, help='batch size for training')
    argparser.add_argument('--batch-size-test', type=int, default=100, help='batch size for evaling')
    argparser.add_argument('--num-heads', type=int, default=2, help='Multi Head Attention heads')
    argparser.add_argument('--in-feats-t', type=int, default=100, help='time embedding feats')
    argparser.add_argument('--in-feats-s', type=int, default=172, help='memory feats')
    argparser.add_argument('--out-feats', type=int, default=100, help='node embedding feats')
    argparser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    argparser.add_argument('--lr', type=float, default=0.0003, help='lr for embedding')
    argparser.add_argument('--lr-p', type=float, default=0.0003, help='lr for future task(s)')
    argparser.add_argument('--decay', type=float, default=1e-4, help='l2 norm for future task(s)')
    argparser.add_argument('--n-epochs', type=int, default=50, help='number of epoch(s)')
    argparser.add_argument('--log-every', type=int, default=50, help='print training results every xx step(s)')
    argparser.add_argument('--eval-every', type=int, default=1, help='eval the model every xx epoch(s)')
    argparser.add_argument('--num-neg', type=int, default=1, help='for each edge, sample xx negative node pairs')
    argparser.add_argument('--num-nei', type=int, default=10, help='for each edge, aggregate xx neighbours')

    args = argparser.parse_args()
    print(args)
    main(args)
