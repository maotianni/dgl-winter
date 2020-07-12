import torch as th
import torch.nn as nn
import torch.nn.functional as F
import argparse
import time
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt

from model import GraphConvMatrixCompletion
from data import node_features, load_feats


def main(args):
    data_set = args.data_set
    features = node_features(data_set)
    users, movies, all_train_rating_info, test_rating_info = load_feats(data_set)

    # validation
    validation = args.validation
    if validation:
        train_rating_info = all_train_rating_info.sample(frac=0.8, random_state=35)
        rowlist = []
        for indexs in train_rating_info.index:
            rowlist.append(indexs)
        validation_rating_info = all_train_rating_info.drop(rowlist, axis=0)
    else:
        train_rating_info = all_train_rating_info
        validation_rating_info = test_rating_info


    # create graphs
    if data_set == 'ml-10m':
        n_ratings = 10
    else:
        n_ratings = 5
    # train
    edge_u_train = train_rating_info.user_id.values
    edge_v_train = train_rating_info.movie_id.values
    edge_r_train = th.Tensor(train_rating_info.rating.values)
    rates = []
    rates_rev = []
    for i in range(n_ratings):
        index = np.argwhere(train_rating_info.rating == (i + 1)).T[0, :]
        e_user = train_rating_info.user_id.values[index]
        e_item = train_rating_info.movie_id.values[index]
        e = np.ones(len(index))
        A_ur = coo_matrix((e, (e_user, e_item)), shape=(users.values.shape[0], movies.values.shape[0])).toarray()
        A_vr = coo_matrix((e, (e_item, e_user)), shape=(movies.values.shape[0], users.values.shape[0])).toarray()
        rates.append(A_ur)
        rates_rev.append(A_vr)
    A_u = th.Tensor(rates)
    A_v = th.Tensor(rates_rev)
    A = sum(rates)
    D_u = np.eye(users.values.shape[0]) * sum(A.T)
    D_v = np.eye(movies.values.shape[0]) * sum(A)
    D_u = D_u ** (-1 / 2)
    D_v = D_v ** (-1 / 2)
    D_u[D_u == np.inf] = 0
    D_v[D_v == np.inf] = 0
    D_u = th.Tensor(D_u)
    D_v = th.Tensor(D_v)

    # validation
    edge_u_val = validation_rating_info.user_id.values
    edge_v_val = validation_rating_info.movie_id.values
    edge_r_val = th.Tensor(validation_rating_info.rating.values)
    rates_val = []
    rates_rev_val = []
    for i in range(n_ratings):
        index = np.argwhere(validation_rating_info.rating == (i + 1)).T[0, :]
        e_user = validation_rating_info.user_id.values[index]
        e_item = validation_rating_info.movie_id.values[index]
        e = np.ones(len(index))
        A_ur = coo_matrix((e, (e_user, e_item)), shape=(users.values.shape[0], movies.values.shape[0])).toarray()
        A_vr = coo_matrix((e, (e_item, e_user)), shape=(movies.values.shape[0], users.values.shape[0])).toarray()
        rates_val.append(A_ur)
        rates_rev_val.append(A_vr)
    A_u_val = th.Tensor(rates_val)
    A_v_val = th.Tensor(rates_rev_val)
    A_val = sum(rates_val)
    D_u_val = np.eye(users.values.shape[0]) * sum(A_val.T)
    D_v_val = np.eye(movies.values.shape[0]) * sum(A_val)
    D_u_val = D_u_val ** (-1 / 2)
    D_v_val = D_v_val ** (-1 / 2)
    D_u_val[D_u_val == np.inf] = 0
    D_v_val[D_v_val == np.inf] = 0
    D_u_val = th.Tensor(D_u_val)
    D_v_val = th.Tensor(D_v_val)

    # test
    edge_u_test = test_rating_info.user_id.values
    edge_v_test = test_rating_info.movie_id.values
    edge_r_test = th.Tensor(test_rating_info.rating.values)
    rates_test = []
    rates_rev_test = []
    for i in range(n_ratings):
        index = np.argwhere(test_rating_info.rating == (i + 1)).T[0, :]
        e_user = test_rating_info.user_id.values[index]
        e_item = test_rating_info.movie_id.values[index]
        e = np.ones(len(index))
        A_ur = coo_matrix((e, (e_user, e_item)), shape=(users.values.shape[0], movies.values.shape[0])).toarray()
        A_vr = coo_matrix((e, (e_item, e_user)), shape=(movies.values.shape[0], users.values.shape[0])).toarray()
        rates_test.append(A_ur)
        rates_rev_test.append(A_vr)
    A_u_test = th.Tensor(rates_test)
    A_v_test = th.Tensor(rates_rev_test)
    A_test = sum(rates_test)
    D_u_test = np.eye(users.values.shape[0]) * sum(A_test.T)
    D_v_test = np.eye(movies.values.shape[0]) * sum(A_test)
    D_u_test = D_u_test ** (-1 / 2)
    D_v_test = D_v_test ** (-1 / 2)
    D_u_test[D_u_test == np.inf] = 0
    D_v_test[D_v_test == np.inf] = 0
    D_u_test = th.Tensor(D_u_test)
    D_v_test = th.Tensor(D_v_test)


    # features
    X_u = features[0]
    X_v = features[1]
    X_u_val = features[0]
    X_v_val = features[1]
    X_u_test = features[0]
    X_v_test = features[1]

    # check cuda
    gpu = args.gpu
    use_cuda = gpu >= 0 and th.cuda.is_available()
    if use_cuda:
        th.cuda.set_device(gpu)
        X_u = X_u.cuda()
        X_v = X_v.cuda()
        X_u_val = X_u_val.cuda()
        X_v_val = X_v_val.cuda()
        X_u_test = X_u_test.cuda()
        X_v_test = X_v_test.cuda()
        A_u = A_u.cuda()
        A_v = A_v.cuda()
        A_u_val = A_u_val.cuda()
        A_v_val = A_v_val.cuda()
        A_u_test = A_u_test.cuda()
        A_v_test = A_v_test.cuda()
        D_u = D_u.cuda()
        D_v = D_v.cuda()
        D_u_val = D_u_val.cuda()
        D_v_val = D_v_val.cuda()
        D_u_test = D_u_test.cuda()
        D_v_test = D_v_test.cuda()
        edge_r_train = edge_r_train.cuda()
        edge_r_val = edge_r_val.cuda()
        edge_r_test = edge_r_test.cuda()

    # create model
    u_num = users.values.shape[0]
    v_num = movies.values.shape[0]
    n_in = X_u.shape[1]
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

    model = GraphConvMatrixCompletion(u_num, v_num, n_in, n_ins, n_hidden, n_z, n_ratings, n_bases,
                                      norm=norm, accum=accum, w_sharing=w_sharing,
                                      activation=F.leaky_relu, side_information=side_information, bias=bias,
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

    if data_set == 'ml-10m':
        r = th.Tensor([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5])
    else:
        r = th.Tensor([1, 2, 3, 4, 5])
    loss_function = nn.MSELoss()
    if use_cuda:
        r = r.cuda()
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        t0 = time.time()
        logits = model(X_u, X_v, A_u, A_v, D_u, D_v, X_u, X_v)[0]
        logits_partial = logits[:, edge_u_train, edge_v_train].T
        loss = F.cross_entropy(logits_partial, (edge_r_train - 1).long()).mean()
        t1 = time.time()
        loss.backward(retain_graph=True)
        t2 = time.time()
        M_partial = th.matmul(r, logits_partial.T)
        train_mse = loss_function(M_partial, edge_r_train)
        train_rmse = train_mse.sqrt()
        optimizer.step()
        

        forward_time.append(t1 - t0)
        backward_time.append(t2 - t1)
        print("Epoch {:05d} | Train Forward Time(s) {:.4f} | Backward Time(s) {:.4f}".
              format(epoch, forward_time[-1], backward_time[-1]))

        model.eval()
        logits_v = model.forward(X_u_val, X_v_val, A_u_val, A_v_val, D_u_val, D_v_val, X_u_val, X_v_val)[0]
        logits_partial_v = logits_v[:, edge_u_val, edge_v_val].T
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
    logits = model.forward(X_u_test, X_v_test, A_u_test, A_v_test, D_u_test, D_v_test, X_u_test, X_v_test)[0]
    logits_partial_t = logits[:, edge_u_test, edge_v_test].T
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
    parser.add_argument("--data-set", type=str, default='ml-100k',
                        help="choose the dataset")
    parser.add_argument("--validation", default=True, action='store_false',
                help="use validation sets")
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
    parser.add_argument("--w-sharing", default=True, action='store_false',
                help="w-sharing")
    parser.add_argument("--side-information", default=True, action='store_false',
                help="side-information")
    parser.add_argument("--bias", default=True, action='store_false',
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
