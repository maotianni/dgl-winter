import pandas as pd
import numpy as np
import torch


def load_wiki(val):
    # 读取
    data = pd.read_csv('..\data\wikipedia.csv', skiprows=[0], header=None)
    # 列名
    columns = pd.read_csv('..\data\wikipedia.csv', nrows=1, header=None)
    # 替换列名
    columns_n = [''] * data.shape[1]
    columns_n[:4] = columns.values[0].tolist()[:4]
    columns_n[4:] = ['feat_{}'.format(i) for i in range(data.shape[1] - 4)]
    # 替换
    data.columns = columns_n
    # Users / Items
    users = pd.DataFrame({'user_id': data.user_id.unique().astype('object')})
    items = pd.DataFrame({'item_id': data.item_id.unique().astype('object')})
    n_users = users.shape[0]
    n_items = items.shape[0]
    # features
    # one-hot
    users = pd.get_dummies(users)
    items = pd.get_dummies(items)
    features_u = torch.Tensor(users.values)
    features_v = torch.Tensor(items.values)
    features_e = torch.Tensor(data.iloc[:, 4:].values)
    t = torch.Tensor(data['timestamp'])
    label = torch.Tensor(data['state_label'])
    # train-val-test
    test_split = np.floor(data.shape[0] * 0.85).astype('int64')
    val_split = np.floor(data.shape[0] * 0.70).astype('int64')
    all_train_set = data.iloc[:test_split, :]
    train_set = data.iloc[:val_split, :]
    val_set = data.iloc[val_split: test_split, :]
    test_set = data.iloc[test_split:, :]
    del data
    head, tail = train_set['user_id'].values, train_set['item_id'].values
    head_a, tail_a = all_train_set['user_id'].values, all_train_set['item_id'].values
    head_v, tail_v = val_set['user_id'].values, val_set['item_id'].values
    head_t, tail_t = test_set['user_id'].values, test_set['item_id'].values
    # dict
    out = {'features_u': features_u, 'features_v': features_v,
           'n_users': n_users, 'n_items': n_items, 'u_feats': users.shape[0], 'v_feats': items.shape[0],
           'edge_feats': features_e, 't': t, 'label': label}
    if val:
        out['train'] = (head, tail)
        out['val'] = (head_v, tail_v)
        out['test'] = (head_t, tail_t)
    else:
        out['train'] = (head_a, tail_a)
        out['test'] = (head_t, tail_t)
    return out


class LoadData(object):
    def __init__(self, dataset, val=False):
        self.dataset = dataset
        self.val = val
        if self.dataset == 'wikipedia':
            self.loader = load_wiki(self.val)
        self.n_users = self.loader['n_users']
        self.n_items = self.loader['n_items']
        self.u_feats = self.loader['u_feats']
        self.v_feats = self.loader['v_feats']
        self.features_u = self.loader['features_u']
        self.features_v = self.loader['features_v']
        self.e_feats = self.loader['edge_feats']
        self.t = self.loader['t']
        self.label = self.loader['label']
        self.train = self.loader['train']
        if self.val:
            self.val = self.loader['val']
        self.test = self.loader['test']
