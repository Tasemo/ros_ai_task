#!/usr/bin/env python

import torch.nn as nn
import torch.nn.functional as F

class MNISTModel(nn.Module):
    def __init__(self, width, height):
        super().__init__()
        self.size = width * height
        self.layer1 = nn.Linear(in_features=self.size, out_features=200)
        self.layer2 = nn.Linear(in_features=200, out_features=200)
        self.layer3 = nn.Linear(in_features=200, out_features=10)

    def forward(self, x):
        x = x.view(-1, self.size)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return F.log_softmax(self.layer3(x), dim=1)
