import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class sup_cifar_mlp(nn.Module):
    def __init__(self, last_dims, selected_layers=[1]):
        super(sup_cifar_mlp, self).__init__()
        self.fc_size = [3072,3072,3072,3072,3072]
        self.meta_fcs = []
        for i in range(1):
            self.meta_fcs.append(nn.Linear(self.fc_size[i]*self.fc_size[i], last_dims))      
        self.meta_fcs=nn.ModuleList(self.meta_fcs)
        self.selected_layers = selected_layers


    def forward(self, x_list):
        x_out=[]
        batchsize = x_list[0].size(0)
        dims = 2

        if self.selected_layers[0]==1:
            x = x_list[0].permute(0,2,1)
            x = torch.matmul(x,x.permute(0,2,1)).view(batchsize,-1)
            x = self.meta_fcs[0](x) 
        else: 
            if self.selected_layers[0]>=2:
                x=torch.matmul(F.normalize(x_list[0].permute(0,2,1),2,dims),F.normalize(x_list[1].permute(0,2,1),2,dims))# add
            if self.selected_layers[0]>=3:
                x=torch.matmul(x,F.normalize(x_list[2].permute(0,2,1),2,dims))
            if self.selected_layers[0]>=4:
                x=torch.matmul(x,F.normalize(x_list[3].permute(0,2,1),2,dims)) 
            x = torch.matmul(x,x.permute(0,2,1)).view(batchsize,-1) 

            x=self.meta_fcs[0](x) 
        return x



class sup_cifar_mlp_notalign(nn.Module):
    def __init__(self, last_dims, selected_layers=[1]):
        super(sup_cifar_mlp_notalign, self).__init__()
        self.fc_size = [3072,500,500,500,2]
        self.meta_fcs = []
        layer = selected_layers[0]-1
        self.meta_fcs.append(nn.Linear(self.fc_size[layer]*self.fc_size[layer+1], last_dims))      
        self.meta_fcs=nn.ModuleList(self.meta_fcs)
        self.selected_layers = selected_layers

    def forward(self, x_list):
        x_out=[]
        batchsize = x_list[0].size(0)
        dims = 1
        layer = self.selected_layers[0]-1
        x = x_list[layer]
        x = F.normalize(x,2,dims)

        x = x.contiguous().view(batchsize,-1) 
        x=self.meta_fcs[0](x) 
        return x
        return x




  

