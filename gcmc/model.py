import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl
from scipy.sparse import coo_matrix
import argparse
import time
import pandas as pd
import numpy as np

from layers import GraphCovLayer, EmbeddingLayer, BilinearDecoder


class Template(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, num_of_ratings, num_of_bases,
                 w_sharing=False, activation=None,
                 side_information=False, bias=False, dropout=0.0, use_cuda=False):
        super(Template, self).__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.num_of_ratings = num_of_ratings
        self.num_of_bases = num_of_bases
        self.w_sharing = w_sharing
        self.activation = activation
        self.side_information = side_information
        self.bias = bias
        self.num_of_ratings = num_of_ratings
        self.dropout = dropout
        self.use_cuda = use_cuda
        # 构建模型
        self.build_model()

    def build_model(self):
        self.layers = nn.ModuleList()
        # i2h
        i2h = self.build_GConv_layer()
        if i2h is not None:
            self.layers.append(i2h)
        # h2z
        h2z = self.build_Emb_layer()
        if h2z is not None:
            self.layers.append(h2z)
        # z2o
        z2o  = self.build_Decode_layer()
        if z2o is not None:
            self.layers.append(z2o)

    def build_GConv_layer(self):
        return None

    def build_Emb_layer(self):
        return None

    def build_Decode_layer(self):
        return None

    def forward(self, g):
        for layer in self.layers:
            g = layer(g)
        return g


class EncoderNDecoder(Template):
    def build_GConv_layer(self):
        return GraphCovLayer(self.in_dim, self.hid_dim, self.num_of_ratings,
                             w_sharing=self.w_sharing, activation=self.activation, dropout=self.dropout)

    def build_Emb_layer(self):
        return EmbeddingLayer(self.in_dim, self.hid_dim, self.out_dim,
                              activation=self.activation, side_information=self.side_information,
                              bias=self.bias, dropout=self.dropout)

    def build_Decode_layer(self):
        return BilinearDecoder(self.out_dim, self.num_of_ratings, self.num_of_bases, w_sharing=self.w_sharing)


class GraphConvMatrixCompletion(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, num_of_ratings, num_bases,
                 w_sharing=False, activation=None,
                 side_information=False, bias=False, dropout=0.0, use_cuda=False):
        super(GraphConvMatrixCompletion, self).__init__()
        self.encoderNdecoder = EncoderNDecoder(in_dim, hid_dim, out_dim, num_of_ratings, num_bases,
                                    w_sharing=w_sharing, activation=activation,
                                    side_information=side_information, bias=bias,
                                    dropout=dropout, use_cuda=use_cuda)

    def forward(self, g):
        return self.encoderNdecoder.forward(g)


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


features = node_features()

# full data
users = pd.read_table('ml-100k/u.user', sep="|",
                      names=['user_id', 'age', 'sex', 'occupation', 'zip_code'],
                      encoding='latin-1', engine='python')
movies = pd.read_table('ml-100k/u.item', engine='python', sep='|',
                       header=None, encoding='latin-1',
                       names=['movie_id', 'title', 'release_date', 'video_release_date',
                              'IMDb_URL', 'unknown', 'Action', 'Adventure',
                              'Animation', 'Children', 'Comedy', 'Crime', 'Documentary',
                              'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
                              'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'])
# train / test
all_train_rating_info = pd.read_csv(
                                    'ml-100k/u1.base', sep='\t', header=None,
                                    names=['user_id', 'movie_id', 'rating', 'timestamp'],
                                    dtype={'user_id': np.int32, 'movie_id' : np.int32,
                                           'ratings': np.float32, 'timestamp': np.int64}, engine='python'
                                   )
test_rating_info = pd.read_csv(
                               'ml-100k/u1.test', sep='\t', header=None,
                               names=['user_id', 'movie_id', 'rating', 'timestamp'],
                               dtype={'user_id': np.int32, 'movie_id' : np.int32,
                                      'ratings': np.float32, 'timestamp': np.int64}, engine='python'
                               )
# validation
validation = True
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
# train
edge_u_train = train_rating_info.user_id.values - 1
edge_v_train = train_rating_info.movie_id.values - 1
edge_r_train = train_rating_info.rating.values
spmat = coo_matrix((edge_r_train, (edge_u_train, edge_v_train)), shape=(users.values.shape[0], movies.values.shape[0]))
g = dgl.bipartite(spmat, 'user', 'rate', 'item')
g.edges['rate'].data['rate'] = edge_r_train - 1
u_s_train = g.edges()[0]
v_s_train = g.edges()[1]
label_raw_train = g.edges['rate'].data['rate']
label_raw_train = th.Tensor(label_raw_train).long()
# validation
edge_u_val = validation_rating_info.user_id.values - 1
edge_v_val = validation_rating_info.movie_id.values - 1
edge_r_val = validation_rating_info.rating.values
spmat_v = coo_matrix((edge_r_val, (edge_u_val, edge_v_val)), shape=(users.values.shape[0], movies.values.shape[0]))
g_val = dgl.bipartite(spmat_v, 'user', 'rate', 'item')
g_val.edges['rate'].data['rate'] = edge_r_val - 1
# test
edge_u_test = test_rating_info.user_id.values - 1
edge_v_test = test_rating_info.movie_id.values - 1
edge_r_test = test_rating_info.rating.values
spmat_t = coo_matrix((edge_r_test, (edge_u_test, edge_v_test)), shape=(users.values.shape[0], movies.values.shape[0]))
g_test = dgl.bipartite(spmat_t, 'user', 'rate', 'item')
g_test.edges['rate'].data['rate'] = edge_r_test - 1

# check cuda
gpu = 0
use_cuda = gpu >= 0 and th.cuda.is_available()
if use_cuda:
    th.cuda.set_device(gpu)
    u_s_train = u_s_train.cuda()
    v_s_train = v_s_train.cuda()
    label_raw_train = label_raw_train.cuda()

# features
g.nodes['user'].data['x'] = features[0]
g.nodes['item'].data['x'] = features[1]

# create model
n_hidden = 10
n_bases = 3
w_sharing = True
side_information = True
bias = True
dropout = 0.7

model = GraphConvMatrixCompletion(40, n_hidden, n_hidden, 5, n_bases, w_sharing=w_sharing,
                                  activation=F.relu, side_information=side_information, bias=bias,
                                  dropout=dropout, use_cuda=use_cuda)
if use_cuda:
    model.cuda()

# optimizer
lr = 1e-2
l2norm = 0.0
optimizer = th.optim.Adam(model.parameters(), lr=lr, weight_decay=l2norm)

# training loop
n_epochs = 50
print("start training...")
forward_time = []
backward_time = []
model.train()

for epoch in range(n_epochs):
    optimizer.zero_grad()
    t0 = time.time()
    logits = model(g)
    loss = F.cross_entropy(logits[:, u_s_train, v_s_train].T, label_raw_train)
    t1 = time.time()
    loss.backward()
    optimizer.step()
    t2 = time.time()

    forward_time.append(t1 - t0)
    backward_time.append(t2 - t1)
    print("Epoch {:05d} | Train Forward Time(s) {:.4f} | Backward Time(s) {:.4f}".
          format(epoch, forward_time[-1], backward_time[-1]))


