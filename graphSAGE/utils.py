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
