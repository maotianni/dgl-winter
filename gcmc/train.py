import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl
import argparse
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from model import GraphConvMatrixCompletion


def node_features():
    users = pd.read_table('data/ml-100k/u.user', sep="|",
                          names=['user_id', 'age', 'sex', 'occupation', 'zip_code'],
                          encoding='latin-1', engine='python')
    movies = pd.read_table('data/ml-100k/u.item', engine='python', sep='|',
                           header=None, encoding='latin-1',
                           names=['movie_id', 'title', 'release_date', 'video_release_date',
                                  'IMDb_URL', 'unknown', 'Action', 'Adventure',
                                  'Animation', 'Children', 'Comedy', 'Crime', 'Documentary',
                                  'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
                                  'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'])
    users['sex'] = pd.get_dummies(data=users['sex'], prefix='sex').iloc[:, 1]
    temp_1 = users.iloc[:, 1:3].values
    temp_2 = pd.get_dummies(data=users['occupation'], prefix='occupation').iloc[:, 1:].values
    temp_3 = np.zeros((943, 18))
    x_u = th.Tensor(np.concatenate((temp_1, temp_2, temp_3), axis=1))
    temp_4 = np.zeros((1682, 22))
    temp_5 = movies.iloc[:, 6:].values
    x_v = th.Tensor(np.concatenate((temp_4, temp_5), axis=1))
    return x_u, x_v


def main(args):
    features = node_features()

    # full data
    users = pd.read_table('data/ml-100k/u.user', sep="|",
                          names=['user_id', 'age', 'sex', 'occupation', 'zip_code'],
                          encoding='latin-1', engine='python')
    movies = pd.read_table('data/ml-100k/u.item', engine='python', sep='|',
                           header=None, encoding='latin-1',
                           names=['movie_id', 'title', 'release_date', 'video_release_date',
                                  'IMDb_URL', 'unknown', 'Action', 'Adventure',
                                  'Animation', 'Children', 'Comedy', 'Crime', 'Documentary',
                                  'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
                                  'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'])
    # train / test
    all_train_rating_info = pd.read_csv(
                                        'data/ml-100k/u1.base', sep='\t', header=None,
                                        names=['user_id', 'movie_id', 'rating', 'timestamp'],
                                        dtype={'user_id': np.int32, 'movie_id' : np.int32,
                                               'ratings': np.float32, 'timestamp': np.int64}, engine='python'
                                       )
    test_rating_info = pd.read_csv(
                                   'data/ml-100k/u1.test', sep='\t', header=None,
                                   names=['user_id', 'movie_id', 'rating', 'timestamp'],
                                   dtype={'user_id': np.int32, 'movie_id' : np.int32,
                                          'ratings': np.float32, 'timestamp': np.int64}, engine='python'
                                   )
    # validation
    validation = args.validation
    if validation:
        train_rating_info = all_train_rating_info.sample(frac=0.8)
        rowlist = []
        for indexs in train_rating_info.index:
            rowlist.append(indexs)
        validation_rating_info = all_train_rating_info.drop(rowlist, axis=0)
    else:
        train_rating_info = all_train_rating_info
        validation_rating_info = test_rating_info


    # create graphs
    semi_param = args.semi_param
    # train
    edge_u_train = train_rating_info.user_id.values - 1
    edge_v_train = train_rating_info.movie_id.values - 1
    edge_r_train = th.Tensor(train_rating_info.rating.values)

    rates = []
    for i in range(5):
        index = np.argwhere(train_rating_info.rating == (i + 1)).T[0, :]
        e_user = train_rating_info.user_id.values[index] - 1
        e_item = train_rating_info.movie_id.values[index] - 1
        edge = list(zip(e_user, e_item))
        edge_rev = list(zip(e_item, e_user))
        rates.append(dgl.bipartite(edge, 'user', '{}'.format(i+1), 'item',
                                   card=(users.values.shape[0], movies.values.shape[0])))
        rates.append(dgl.bipartite(edge_rev, 'item', 'rev{}'.format(i+1), 'user',
                                   card=(movies.values.shape[0], users.values.shape[0])))
    g = dgl.hetero_from_relations(rates)
    user_ci = []
    user_cj = []
    item_ci = []
    item_cj = []
    for i in range(5):
        user_ci.append(g['rev{}'.format(i+1)].in_degrees())
        item_ci.append(g['{}'.format(i+1)].in_degrees())
        user_cj.append(g['{}'.format(i+1)].out_degrees())
        item_cj.append(g['rev{}'.format(i+1)].out_degrees())
    uci = np.sqrt(1 / (sum(user_ci).numpy()).astype('float64'))
    uci[uci == np.inf] = 1
    vci = np.sqrt(1 / (sum(item_ci).numpy()).astype('float64'))
    vci[vci == np.inf] = 1
    ucj = np.sqrt(1 / (sum(user_cj).numpy()).astype('float64'))
    ucj[ucj == np.inf] = 1
    vcj = np.sqrt(1 / (sum(item_cj).numpy()).astype('float64'))
    vcj[vcj == np.inf] = 1
    user_ci = th.Tensor(uci)
    item_ci = th.Tensor(vci)
    user_cj = th.Tensor(ucj)
    item_cj = th.Tensor(vcj)
    g.nodes['user'].data.update({'ci': user_ci, 'cj': user_cj})
    g.nodes['item'].data.update({'ci': item_ci, 'cj': item_cj})

    # validation
    edge_u_val = validation_rating_info.user_id.values - 1
    edge_v_val = validation_rating_info.movie_id.values - 1
    edge_r_val = th.Tensor(validation_rating_info.rating.values)
    # test
    edge_u_test = test_rating_info.user_id.values - 1
    edge_v_test = test_rating_info.movie_id.values - 1
    edge_r_test = th.Tensor(test_rating_info.rating.values)
    rates_test = []
    for i in range(5):
        index = np.argwhere(test_rating_info.rating == (i + 1)).T[0, :]
        e_user = test_rating_info.user_id.values[index] - 1
        e_item = test_rating_info.movie_id.values[index] - 1
        edge = list(zip(e_user, e_item))
        edge_rev = list(zip(e_item, e_user))
        rates_test.append(dgl.bipartite(edge, 'user', '{}'.format(i+1), 'item',
                                   card=(users.values.shape[0], movies.values.shape[0])))
        rates_test.append(dgl.bipartite(edge_rev, 'item', 'rev{}'.format(i+1), 'user',
                                   card=(movies.values.shape[0], users.values.shape[0])))
    g_test = dgl.hetero_from_relations(rates_test)
    user_ci_test = []
    user_cj_test = []
    item_ci_test = []
    item_cj_test = []
    for i in range(5):
        user_ci_test.append(g['rev{}'.format(i+1)].in_degrees())
        item_ci_test.append(g['{}'.format(i+1)].in_degrees())
        user_cj_test.append(g['{}'.format(i+1)].out_degrees())
        item_cj_test.append(g['rev{}'.format(i+1)].out_degrees())
    uci_test = np.sqrt(1 / (sum(user_ci_test).numpy()).astype('float64'))
    uci_test[uci_test == np.inf] = 1
    vci_test = np.sqrt(1 / (sum(item_ci_test).numpy()).astype('float64'))
    vci_test[vci_test == np.inf] = 1
    ucj_test = np.sqrt(1 / (sum(user_cj_test).numpy()).astype('float64'))
    ucj_test[ucj_test == np.inf] = 1
    vcj_test = np.sqrt(1 / (sum(item_cj_test).numpy()).astype('float64'))
    vcj_test[vcj_test == np.inf] = 1
    user_ci_test = th.Tensor(uci_test)
    item_ci_test = th.Tensor(vci_test)
    user_cj_test = th.Tensor(ucj_test)
    item_cj_test = th.Tensor(vcj_test)
    g_test.nodes['user'].data.update({'ci': user_ci_test, 'cj': user_cj_test})
    g_test.nodes['item'].data.update({'ci': item_ci_test, 'cj': item_cj_test})


    # features
    g.nodes['user'].data['x'] = features[0]
    g.nodes['item'].data['x'] = features[1]
    g_test.nodes['user'].data['x'] = features[0]
    g_test.nodes['item'].data['x'] = features[1]

    # check cuda
    gpu = args.gpu
    use_cuda = gpu >= 0 and th.cuda.is_available()
    if use_cuda:
        th.cuda.set_device(gpu)
        g.to(th.device('cuda:{}'.format(gpu)))
        g_test.to(th.device('cuda:{}'.format(gpu)))
        edge_r_train = edge_r_train.cuda()
        edge_r_val = edge_r_val.cuda()
        edge_r_test = edge_r_test.cuda()

    # create model
    u_num = users.values.shape[0]
    v_num = movies.values.shape[0]
    n_ins = args.n_ins
    n_hidden = args.n_hidden
    n_z = args.n_z
    n_bases = args.n_bases
    accum = args.accum
    w_sharing = args.w_sharing
    side_information = args.side_information
    bias = args.bias
    dropout = args.dropout
    norm = args.norm

    model = GraphConvMatrixCompletion(u_num, v_num, 40, n_ins, n_hidden, n_z, 5, n_bases,
                                      norm=norm, accum=accum, w_sharing=w_sharing,
                                      activation=F.relu, side_information=side_information, bias=bias,
                                      dropout=dropout, use_cuda=use_cuda)
    if use_cuda:
        model.cuda()

    # optimizer
    lr = args.lr
    l2norm = args.l2norm
    optimizer = th.optim.Adam(model.parameters(), lr=lr, weight_decay=l2norm)

    # training loop
    n_epochs = args.n_epochs
    print("start training...")
    forward_time = []
    backward_time = []
    model.train()

    # plot the experiment
    t_loss, t_rmse, v_loss, v_rmse = [], [], [], []

    r = th.Tensor([1, 2, 3, 4, 5])
    loss_function = nn.MSELoss()
    if use_cuda:
        r = r.cuda()
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        t0 = time.time()
        logits = model(g, features[0], features[1])[0]
        logits_partial = logits[:, edge_u_train, edge_v_train].T
        if use_cuda:
            logits_partial = logits_partial.cuda()
        loss = F.cross_entropy(logits_partial, (edge_r_train - 1).long())
        t1 = time.time()
        loss.backward(retain_graph=True)
        optimizer.step()
        t2 = time.time()

        forward_time.append(t1 - t0)
        backward_time.append(t2 - t1)
        print("Epoch {:05d} | Train Forward Time(s) {:.4f} | Backward Time(s) {:.4f}".
              format(epoch, forward_time[-1], backward_time[-1]))

        M_partial = th.matmul(r, logits_partial.T)
        train_mse = loss_function(M_partial, edge_r_train)
        train_rmse = train_mse.sqrt()

        logits_partial_v = logits[:, edge_u_val, edge_v_val].T
        if use_cuda:
            logits_partial_v = logits_partial.cuda()
        val_loss = F.cross_entropy(logits_partial_v, (edge_r_val-1).long())
        M_partial_v = th.matmul(r, logits_partial_v.T)
        val_mse = loss_function(M_partial_v, edge_r_val)
        val_rmse = val_mse.sqrt()

        print("Train Loss: {:.4f} | Train RMSE: {:.4f} | Validation Loss: {:.4f} | Validation RMSE: {:.4f}".
              format(loss.item(), train_rmse.item(), val_loss.item(), val_rmse.item()))

        # data for plots
        t_loss.append(loss.item()), t_rmse.append(train_rmse.item())
        v_loss.append(val_loss.item()), v_rmse.append(val_rmse.item())

    print()

    model.eval()
    logits = model.forward(g_test, features[0], features[1])[0]
    logits_partial_t = logits[:, edge_u_test, edge_v_test].T
    if use_cuda:
        logits_partial_t = logits_partial_t.cuda()
    test_loss = F.cross_entropy(logits_partial_t, (edge_r_test-1).long())
    M_partial_t = th.matmul(r, logits_partial_t.T)
    test_mse = loss_function(M_partial_t, edge_r_test)
    test_rmse = test_mse.sqrt()
    print("Test Loss: {:.4f} | Test RMSE: {:.4f}".format(test_loss.item(), test_rmse.item()))
    print()

    print("Mean forward time: {:4f}".format(np.mean(forward_time[len(forward_time) // 4:])))
    print("Mean backward time: {:4f}".format(np.mean(backward_time[len(backward_time) // 4:])))


    # plots!!!
    e_s = list(range(n_epochs))
    plt.figure()
    plt.plot(e_s, t_loss, label='train loss')
    plt.plot(e_s, v_loss, label='val loss')
    plt.legend()
    plt.xlabel('epoch', fontsize=16)
    plt.xticks(fontsize=16)
    plt.ylabel('loss', fontsize=16)
    plt.yticks(fontsize=16)
    plt.title('Cross Entropy Loss')
    plt.savefig('Cross Entropy Loss.png')
    plt.close()

    plt.figure()
    plt.plot(e_s, t_rmse, label='train rmse')
    plt.plot(e_s, v_rmse, label='val rmse')
    plt.legend()
    plt.xlabel('epoch', fontsize=16)
    plt.xticks(fontsize=16)
    plt.ylabel('loss', fontsize=16)
    plt.yticks(fontsize=16)
    plt.title('RMSE Loss')
    plt.savefig('RMSE Loss.png')
    plt.close()

    # test result!!!
    f = open('test.txt', 'w')
    f.write('{}'.format(args))
    f.write('\n\n')
    f.write("Test Loss: {:.4f} | Test RMSE: {:.4f}".format(test_loss.item(), test_rmse.item()))
    f.write('\n\n')
    f.write("Mean forward time: {:4f}\n".format(np.mean(forward_time[len(forward_time) // 4:])))
    f.write("Mean backward time: {:4f}".format(np.mean(backward_time[len(backward_time) // 4:])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GC-MC')
    parser.add_argument("--validation", type=bool, default=True,
                help="use validation sets")
    parser.add_argument("--semi-param", type=float, default=0.05,
                help="a semi-param in normalization")
    parser.add_argument("--gpu", type=int, default=-1,
                help="gpu")
    parser.add_argument("--n-ins", type=int, default=10,
                help="d of dense layer")
    parser.add_argument("--n-hidden", type=int, default=500,
                help="d of h layer")
    parser.add_argument("--n-z", type=int, default=75,
                help="d of z layer")
    parser.add_argument("--n-bases", type=int, default=2,
                            help="n of w-sharing bases")
    parser.add_argument("--accum", type=str, default='stack',
                            help="method of accum")
    parser.add_argument("--w-sharing", type=bool, default=True,
                help="w-sharing")
    parser.add_argument("--side-information", type=bool, default=True,
                help="side-information")
    parser.add_argument("--bias", type=bool, default=True,
                help="bias")
    parser.add_argument("--dropout", type=float, default=0.7,
                help="dropout rate")
    parser.add_argument("--lr", type=float, default=1e-2,
                help="learning rate")
    parser.add_argument("--l2norm", type=float, default=0.0,
                help="L2 norm")
    parser.add_argument("--n-epochs", type=int, default=1000,
                help="number of epochs")
    parser.add_argument("--norm", type=str, default='left',
                        help="left or symmetric")

    args = parser.parse_args()
    print(args)
    main(args)
