import torch
import torch.nn as nn
from torch.autograd import Variable


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax()
        self.rnn = nn.GRU(input_size, hidden_size, self.num_layers)

    def forward(self, x, h0):
        x, hn = self.rnn(x, h0)
        x = self.fc(x[x.size(0)-1,:,:])
        x = self.softmax(x)
        return x
    def initHidden(self):
        return Variable(torch.zeros(self.num_layers, 1, self.hidden_size))

    def initCell(self):
        return Variable(torch.zeros(self.num_layers, 1, self.hidden_size))


