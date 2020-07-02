import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LR_classification(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(LR_classification, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.logsoftmax = nn.LogSoftmax()

    def forward(self, X):
        y_pred = self.logsoftmax(self.linear(X))
        return y_pred

class Link_Prediction(nn.Module):
    def __init__(self, in_feats, hid_feats, n_layers=2):
        super(Link_Prediction, self).__init__()
        self.n_layers = n_layers
        self.linear_1 = nn.Linear(in_feats, hid_feats)
        self.linear_2 = nn.Linear(hid_feats, hid_feats)
        self.linear_3 = nn.Linear(hid_feats, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X):
        hid = F.relu(self.linear_1(X))
        for layer in range(self.n_layers - 2):
            hid = F.relu(self.linear_2(hid))
        y_pred = F.sigmoid(self.linear_3(hid))
        return y_pred
