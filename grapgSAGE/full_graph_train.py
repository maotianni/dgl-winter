import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import dgl
import dgl.function as fn
from full_graph import *
import time


def load_data():
    # 导入数据：分隔符为空格
    raw_data = pd.read_csv('cora/cora.content',sep = '\t',header = None)
    num_nodes = raw_data.shape[0]
    # 将论文的编号转[0,2707]
    a = list(raw_data.index)
    b = list(raw_data[0])
    c = zip(b,a)
    map = dict(c)
    # 将词向量提取为特征,第二行到倒数第二行
    features =raw_data.iloc[:,1:-1]
    labels = pd.get_dummies(raw_data[1434])
    # 引用数据
    raw_data_cites = pd.read_csv('cora/cora.cites',sep = '\t',header = None)
    # 创建一个规模和邻接矩阵一样大小的矩阵
    x = []
    y = []
    # 创建邻接矩阵
    for i ,j in zip(raw_data_cites[0],raw_data_cites[1]):
        x.append(map[i])
        y.append(map[j])  #替换论文编号为[0,2707]
    #idx_train = range(140)
    #idx_val = range(200, 500)
    #idx_test = range(500, 1500)
    rand_indices = np.random.permutation(num_nodes)
    idx_test = rand_indices[:1000]
    idx_val = rand_indices[1000:1500]
    idx_train = list(rand_indices[1500:])

    src = np.array(x)
    dst = np.array(y)
    features = torch.FloatTensor(features.values)
    labels = torch.LongTensor(labels.values)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return src, dst, features, labels, idx_train, idx_val, idx_test

# data
data = load_data()
src = data[0]
dst = data[1]
features = data[2]
labels = data[3]
target_class = torch.topk(labels, 1)[1].squeeze(1)       # one-hot转会普通的数值
train_id = data[4]
val_id = data[5]
test_id = data[6]

g = dgl.DGLGraph()
g.add_nodes(features.shape[0])
g.add_edges(src, dst)
g.add_edges(dst, src)
g.ndata['x'] = features

gpu = 0
use_cuda = gpu >= 0 and torch.cuda.is_available()
if use_cuda:
    torch.cuda.set_device(gpu)
    g.to(torch.device('cuda:{}'.format(gpu)))
    target_class = target_class.cuda()
    train_id = train_id.cuda()
    val_id = val_id.cuda()
    test_id = test_id.cuda()

in_feats = features.shape[1]
n_classes = labels.shape[1]
n_edges = g.number_of_edges()

#model
model = FullGraphSAGE(in_feats, 16, n_classes, 3, bias=True, aggregator='lstm', activation=F.relu, dropout=0.5)
if use_cuda:
    model.cuda()
loss_fcn = torch.nn.CrossEntropyLoss()

# use optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)

# eval
def evaluate(model, g, target_class, id):
    model.eval()
    with torch.no_grad():
        logits = model(g)
        logits = logits[id]
        target_class = target_class[id]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == target_class)
        return correct.item() * 1.0 / len(target_class)

# train
print("start training...")
forward_time = []
backward_time = []
# model.train()

for epoch in range(200):
    model.train()
    # forward
    optimizer.zero_grad()
    t0 = time.time()
    logits = model(g)
    loss = loss_fcn(logits[train_id], target_class[train_id])
    t1 = time.time()
    loss.backward()
    optimizer.step()
    t2 = time.time()
    forward_time.append(t1 - t0)
    backward_time.append(t2 - t1)
    acc = evaluate(model, g, target_class, val_id)
    print("Epoch {:05d} | Forward Time(s) {:.4f} | Backward Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
          .format(epoch, forward_time[-1], backward_time[-1], loss.item(), acc))

print()
acc = evaluate(model, g, target_class, test_id)
print("Test Accuracy {:.4f}".format(acc))
