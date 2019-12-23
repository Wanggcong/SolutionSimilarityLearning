from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

# simple demo
class MLPNet(nn.Module):
    def __init__(self, num_class=2):
        super(MLPNet, self).__init__()
        n_hidden = 500

        self.fc1 = nn.Linear(32*32*3, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, n_hidden)
        self.fc4 = nn.Linear(n_hidden, num_class)

    def forward(self, x):
        n,c,h,w = x.size()
        x = x.view(n,-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return x