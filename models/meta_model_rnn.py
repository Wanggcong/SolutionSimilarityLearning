import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class sup_namedata_rnn(nn.Module):
    def __init__(self, last_dims):
        super(sup_namedata_rnn, self).__init__()
        self.fc_size = [128,128]
        self.meta_fcs = []
        self.meta_fcs.append(nn.Linear(2*2, last_dims))  
        self.meta_fcs.append(nn.Linear(58*58,last_dims))  
        self.meta_fcs.append(nn.Linear(128*128, last_dims))  
        self.meta_fcs.append(nn.Linear(128*128, last_dims))  

        self.meta_fcs=nn.ModuleList(self.meta_fcs)

    def forward(self, x_list):
        x_out=[]
        batchsize = x_list[0].size(0)
        x1=torch.matmul(x_list[1].permute(0,2,1) ,x_list[1])# add
        x = self.meta_fcs[1](x1.view(batchsize,-1))
        x = x.view(batchsize,-1) #uncomment
        return x


class sup_namedata_rnn_notalign(nn.Module):
    def __init__(self, last_dims):
        super(sup_namedata_rnn_notalign, self).__init__()
        self.fc_size = [128,128]
        self.meta_fcs = []        
        self.meta_fcs.append(nn.Linear(2*2, last_dims))  
        self.meta_fcs.append(nn.Linear(384*58,last_dims))  
        self.meta_fcs.append(nn.Linear(128*128, last_dims))  
        self.meta_fcs.append(nn.Linear(128*128, last_dims))  

        self.meta_fcs=nn.ModuleList(self.meta_fcs)

    def forward(self, x_list):
        x_out=[]
        batchsize = x_list[0].size(0)
        x1=x_list[1]# add
        x = self.meta_fcs[1](x1.view(batchsize,-1))
        x = x.view(batchsize,-1) #uncomment
        return x