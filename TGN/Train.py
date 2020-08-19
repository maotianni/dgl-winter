import numpy as np
import dgl
import torch
import torch.nn as nn
# 其他
import time
import argparse
# 自定义
import NodeCL
import Model
import utils
import Data


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
        new_val = dl.new_val
        new_test = dl.new_test
    else:
        head, tail = dl.train
        head_t, tail_t = dl.test
        n_train = dl.n_train
        n_test = dl.n_test
        new_test = dl.new_test
    # graph
    g = dgl.bipartite(list(zip(head, tail)), 'user', 'edit', 'item', num_nodes=(n_users, n_items))
    if validation:
        g_v = dgl.bipartite(list(zip(head_v, tail_v)), 'user', 'edit', 'item', num_nodes=(n_users, n_items))
    g_t = dgl.bipartite(list(zip(head_t, tail_t)), 'user', 'edit', 'item', num_nodes=(n_users, n_items))
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
    sampler = utils.Sample(g, num_negs=args.num_neg)

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
    model = Model.TGNBasic(in_feats_m, in_feats_u, in_feats_v, in_feats_t,
                           in_feats_e, in_feats_s, out_feats,
                           num_heads, activation=torch.tanh,
                           method=args.message, dropout=args.dropout, use_cuda=use_cuda)
    if use_cuda:
        model.cuda()
    loss_func = utils.Unsuper_Cross_Entropy()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    model.train()
    # edge or learn
    if learn == 'node':
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
    iter_pos = []
    iter_neg = []
    iter_d = []
    iter_t = []
    eval_res = {'eval_ap': [], 'eval_auc': [], 'best_eval_ap': 0, 'best_eval_auc': 0,
                'test_ap': [], 'test_auc': [], 'best_test_ap': 0, 'best_test_auc': 0}
    print('Training Embedding...')
    for epoch in range(n_epochs):
        time_epoch_0 = time.time()
        # each epoch
        si, sj = torch.zeros(n_users, in_feats_s), torch.zeros(n_items, in_feats_s)
        zi, zj = torch.zeros(n_users, out_feats), torch.zeros(n_items, out_feats)
        ## cuda
        if use_cuda:
            si, sj = si.cuda(), sj.cuda()
            zi, zj = zi.cuda(), zj.cuda()
        time_step = time.time()
        for start in range(0, head.shape[0], batch_size):
            step = start // batch_size + 1
            end = start + batch_size
            if end > head.shape[0]:
                end = head.shape[0]
            head_b = head[start:end]
            tail_b = tail[start:end]
            # sample
            pos_graph, pos_graph_r, neg_graph = sampler.obtain_Bs(head_b, tail_b)
            ## cuda
            if use_cuda:
                pos_graph.to(torch.device('cuda:{}'.format(gpu)))
                pos_graph_r.to(torch.device('cuda:{}'.format(gpu)))
                neg_graph.to(torch.device('cuda:{}'.format(gpu)))
            # id
            head_id = pos_graph.srcdata[dgl.NID]
            tail_id = pos_graph.dstdata[dgl.NID]
            neg_id = neg_graph.dstdata[dgl.NID]
            # input
            si_b, sj_b, sn_b = si[head_id], sj[tail_id], sj[neg_id]
            vi_b, vj_b, vn_b = features_u[head_id], features_v[tail_id], features_v[neg_id]
            e_b = features_e[start:end]
            t_b = t[start:end]
            time_load = time.time()
            # forward
            zi_b, zj_b, zn_b = model.forward(pos_graph, pos_graph_r, neg_graph,
                                             si_b, sj_b, sn_b,
                                             e_b, t_b,
                                             vi_b, vj_b, vn_b)
            si_b2, sj_b2 = model.evolve(pos_graph, pos_graph_r, si_b, sj_b, t_b, e_b)
            # loss / backward
            loss = loss_func(zi_b, zj_b, zn_b, pos_graph, neg_graph, use_cuda)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # log time
            time_train = time.time()
            # overwrite
            model.eval()
            with torch.no_grad():
                si[head_id], sj[tail_id] = si_b2, sj_b2
                zi[head_id], zj[tail_id] = zi_b, zj_b
            model.train()
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
                        'Epoch {:05d} | Step {:05d} | Loss {:.4f} | '
                        'Speed (samples/sec) {:.4f} & {:.4f} | Load Time(sec) {:.4f} | Train Time(sec) {:.4f}'.format(
                            epoch+1, step, loss.item(), np.mean(iter_pos),
                            np.mean(iter_neg), np.mean(iter_d), np.mean(iter_t)))
                else:
                    print(
                        'Epoch {:05d} | Step {:05d} | Loss {:.4f} | '
                        'Speed (samples/sec) {:.4f} & {:.4f} | Load Time(sec) {:.4f} | Train Time(sec) {:.4f}'.format(
                            epoch+1, step, loss.item(), np.mean(iter_pos[3:]),
                            np.mean(iter_neg[3:]), np.mean(iter_d[3:]), np.mean(iter_t[3:])))
            time_step = time.time()
        print('\n')
        print('Embedding Has Been Trained!')
        if epoch % eval_every == 0:
            print('Start Evaling...')
            time_ev_0 = time.time()
            if validation:
                link_pre_res = utils.link_pre(model, utils.Sample(g_v, num_negs=args.num_neg), n_train,
                                              head_v, tail_v, batch_size_test,
                                              features_u[:n_train+n_val], features_v[:n_train+n_val],
                                              features_e[:n_train+n_val], t[:n_train+n_val],
                                              n_users, n_items, in_feats_s, out_feats,
                                              args.inductive, new_val,
                                              out_feats, use_cuda, gpu)
                if epoch > 0:
                    eval_res['eval_ap'].append(link_pre_res['AP'])
                if epoch >= 20 and link_pre_res['AP'] < min(eval_res['eval_ap'][-10:]):
                    break
                if epoch > 0 and link_pre_res['AP'] > eval_res['best_eval_ap']:
                    if args.inductive:
                        torch.save(model.state_dict(), 'best_params_inductive_{}.pth'.format(args.dataset))
                    elif args.message != 'last':
                        torch.save(model.state_dict(), 'best_params_{}_{}.pth'.format(args.message, args.dataset))
                    else:
                        torch.save(model.state_dict(), 'best_params_{}.pth'.format(args.dataset))
                    eval_res['best_eval_ap'] = link_pre_res['AP']
                    eval_res['best_eval_auc'] = link_pre_res['AUC']
                    print('Testing...')
                    test_res = utils.link_pre(model, utils.Sample(g_t, num_negs=args.num_neg), n_train+n_val,
                                              head_t, tail_t, batch_size_test, features_u, features_v, features_e, t,
                                              n_users, n_items, in_feats_s, out_feats,
                                              args.inductive, new_test,
                                              use_cuda, gpu)
                    eval_res['best_test_ap'] = test_res['AP']
                    eval_res['best_test_auc'] = test_res['AUC']
                time_ev_1 = time.time()
                print('Eval AP {:.4f} | Eval AUC {:.4f} | Eval Time(s): {:.4f}'.format(link_pre_res['AP'],
                                                                                       link_pre_res['AUC'],
                                                                                       time_ev_1 - time_ev_0))
                print('Best Eval AP {:.4f} | Best Eval AUC {:.4f} | '
                      'Best Test AP {:.4f} | Best Test AUC {:.4f}'.format(eval_res['best_eval_ap'],
                                                                          eval_res['best_eval_auc'],
                                                                          eval_res['best_test_ap'],
                                                                          eval_res['best_test_auc']))
            else:
                link_pre_res = utils.link_pre(model, utils.Sample(g_t, num_negs=args.num_neg), n_train,
                                              head_t, tail_t, batch_size_test, features_u, features_v,
                                              features_e, t, n_users, n_items, in_feats_s, out_feats,
                                              args.inductive, new_test,
                                              use_cuda, gpu)
                if epoch > 0:
                    eval_res['test_ap'].append(link_pre_res['AP'])
                if epoch >= 20 and link_pre_res['AP'] < min(eval_res['test_ap'][-10:]):
                    break
                if epoch > 0 and link_pre_res['AP'] > eval_res['best_test_ap']:
                    if args.inductive:
                        torch.save(model.state_dict(), 'best_params_inductive_{}.pth'.format(args.dataset))
                    elif args.message != 'last':
                        torch.save(model.state_dict(), 'best_params_{}_{}.pth'.format(args.message, args.dataset))
                    else:
                        torch.save(model.state_dict(), 'best_params_{}.pth'.format(args.dataset))
                    eval_res['best_test_ap'] = link_pre_res['AP']
                    eval_res['best_test_auc'] = link_pre_res['AUC']
                time_ev_1 = time.time()
                print('Test AP {:.4f} | Test AUC {:.4f} | Eval Time(s): {:.4f}'.format(link_pre_res['AP'],
                                                                                       link_pre_res['AUC'],
                                                                                       time_ev_1 - time_ev_0))
                print('Best Test AP {:.4f} | Best Test AUC {:.4f}'.format(eval_res['best_test_ap'],
                                                                          eval_res['best_test_auc']))
            print('\n')

        time_epoch_1 = time.time()
        print('Epoch Time(s): {:.4f}'.format(time_epoch_1 - time_epoch_0))

    # Future Dynamic Node Classification
    model.load_state_dict(torch.load('best_params_{}.pth'.format(args.dataset)))
    model.eval()
    if learn == 'node':
        if validation:
            NodeCL.future_task(n_epochs, batch_size, batch_size_test, use_cuda, gpu,
                               model, nc, loss_nc, opt_nc, sampler, args.num_neg,
                               n_users, n_items, in_feats_s, out_feats,
                               n_train, n_val, g_v, g_t,
                               head, tail, head_v, tail_v, head_t, tail_t,
                               features_u, features_v, features_e, t, label,
                               log_every, eval_every, validation)
        else:
            NodeCL.future_task(n_epochs, batch_size, batch_size_test, use_cuda, gpu,
                               model, nc, loss_nc, opt_nc, sampler, args.num_neg,
                               n_users, n_items, in_feats_s, out_feats,
                               n_train, n_train, g_t, g_t,
                               head, tail, head_t, tail_t, head_t, tail_t,
                               features_u, features_v, features_e, t, label,
                               log_every, eval_every, validation)

    print('\n')
    print('Finish!!')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser("TGN Reference")
    argparser.add_argument('--gpu', type=int, default=-1,
                           help="GPU device ID. Use -1 for CPU training")
    argparser.add_argument('--dataset', type=str, default='wikipedia',
                           help='wikipedia or reddit')
    argparser.add_argument("--validation", default=False, action='store_true',
                           help="validation")
    argparser.add_argument("--inductive", default=False, action='store_true',
                           help="inductive")
    argparser.add_argument('--learn', type=str, default='None',
                           help='future tasks, Link Prediction or Node Classification')
    argparser.add_argument('--message', type=str, default='last',
                           help='reduce function')
    argparser.add_argument('--batch-size', type=int, default=200, help='batch size for training')
    argparser.add_argument('--batch-size-test', type=int, default=200, help='batch size for evaling')
    argparser.add_argument('--num-heads', type=int, default=2, help='Multi Head Attention heads')
    argparser.add_argument('--in-feats-t', type=int, default=100, help='time embedding feats')
    argparser.add_argument('--in-feats-s', type=int, default=172, help='memory feats')
    argparser.add_argument('--out-feats', type=int, default=100, help='node embedding feats')
    argparser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    argparser.add_argument('--lr', type=float, default=0.0001, help='lr for embedding')
    argparser.add_argument('--lr-p', type=float, default=0.0001, help='lr for future task(s)')
    argparser.add_argument('--decay', type=float, default=1e-4, help='l2 norm for future task(s)')
    argparser.add_argument('--n-epochs', type=int, default=50, help='number of epoch(s)')
    argparser.add_argument('--log-every', type=int, default=50, help='print training results every xx step(s)')
    argparser.add_argument('--eval-every', type=int, default=1, help='eval the model every xx epoch(s)')
    argparser.add_argument('--num-neg', type=int, default=1, help='for each edge, sample xx negative node pairs')

    args = argparser.parse_args()
    print(args)
    main(args)

