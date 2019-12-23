import torch
import torch.nn as nn
import math

class sup_cifar(nn.Module):
    def __init__(self, last_dims, selected_layers=[1]):
        super(sup_cifar, self).__init__()
        # in_channels, out_channels, kernel_size, stride=1
        self.cnn_other_size = [27,288,567,1152]
        # self.cnn_other_size = [27,32,64,128]
        self.cnn_chn = [32,64,128,256]
        self.meta_chn = [100,100,100,100]

        self.meta_cnns = []
        for i in range(len(self.cnn_other_size)):
            self.meta_cnns.append(nn.Conv2d(1, self.meta_chn[i], (1,self.cnn_other_size[i]), (1,self.cnn_other_size[i])))
        self.meta_fcs = []
        for i in range(len(self.cnn_other_size)):
            self.meta_fcs.append(nn.Linear(self.cnn_chn[i]*self.meta_chn[i], last_dims))

        self.meta_cnns=nn.ModuleList(self.meta_cnns)
        self.meta_fcs=nn.ModuleList(self.meta_fcs)

        self.selected_layers = selected_layers

    def align_weights_first_layer(self, x, conv, fc):
        batchsize, c_out, c_in,ker_h, ker_w = x.size()
        x = x.view(batchsize,1,c_out, -1)
        x1 = conv(x)
        x1, _ = torch.sort(x1,2) ### remove this line for a baseline  
        x1 = x1.view(batchsize,-1) 
        x1 = fc(x1)
        return x1

    def align_weights_other_layer(self, x, conv, fc):
        batchsize, c_out, c_in,ker_h, ker_w = x.size()
        x = x.view(batchsize,1,c_out, -1)


        xx = x.squeeze().view(batchsize,1,c_out,-1,9)
        conv_weight = conv.weight.squeeze().view(conv.weight.data.size(0),1,-1,9)  ### 
        temp = torch.matmul(xx,conv_weight.permute(0,1,3,2))
        # print('xxxx.size:',temp.size())
        max_res,_= torch.max(temp,3)                          
        sum_res = torch.sum(max_res,3)
        sort_res,_ = torch.sort(sum_res,2)      ### replace this line for a baseline 
        sort_res = sum_res

        sort_res = sort_res.view(batchsize,-1)
        x2 = fc(sort_res)   ####
        return x2


    def forward(self, x_list):
        x1 = self.align_weights_first_layer(x_list[0], self.meta_cnns[0],self.meta_fcs[0])   #####
        x2 = self.align_weights_other_layer(x_list[1], self.meta_cnns[1],self.meta_fcs[1])   #####
        x3 = self.align_weights_other_layer(x_list[2], self.meta_cnns[2],self.meta_fcs[2])   #####
        x4 = self.align_weights_other_layer(x_list[3], self.meta_cnns[3],self.meta_fcs[3])   #####
    
        x_out=[]
        x_out.append(x1)
        x_out.append(x2)
        x_out.append(x3)
        x_out.append(x4)

        x = x_out[self.selected_layers[0]-1]
        if len(self.selected_layers)==1:
            pass
        else:
            for i in range(1,len(self.selected_layers)):
                x = x +x_out[self.selected_layers[i]-1]
        return x

class sup_cifar_chain(nn.Module):
    def __init__(self, last_dims, selected_layers=[1]):
        super(sup_cifar_chain, self).__init__()        
        self.cnn_chn = [27,27*9,27*9*9,27*9*9*9,2]
        self.meta_fcs = []
        for i in range(3):
            self.meta_fcs.append(nn.Linear(self.cnn_chn[i]*self.cnn_chn[i], last_dims))
        self.meta_fcs=nn.ModuleList(self.meta_fcs)
        self.selected_layers = selected_layers

    # chain
    def forward(self, x_list):
        x_out=[]
        n,c_out,c_in,ker_h,ker_in = x_list[0].size()
        x = x_list[0]
        if not x.is_contiguous():
            x = x.contiguous()
        x = x.view(n,c_out,-1)
        x_chain = x_list[0].view(n,c_out,-1)# 1,32,27
        for i in range(3):
            if i>0:
                n,c_out,c_in,ker_h,ker_in = x_list[i].size()  #1,32,3,3,3;  1,64,32,3,3
                x_chain = torch.matmul(x_chain.permute(0,2,1),x_list[i].permute(0,2,1,3,4).contiguous().view(n,c_in,-1))##
                n,c1,c2 = x_chain.size()
                x_chain = x_chain.view(n,c1,-1,9).permute(0,2,1,3).contiguous().view(n,c2/9,-1)  #1,64,27*9
            x = torch.matmul(x_chain.permute(0,2,1),x_chain)
            x=x.view(x.size(0),-1)
            x=self.meta_fcs[i](x)
            x_out.append(x) 
        x = x_out[self.selected_layers[0]-1]
        if len(self.selected_layers)==1:
            pass
        else:
            for i in range(1,len(self.selected_layers)):
                x = x +x_out[self.selected_layers[i]-1]
        return x



class sup_cifar_chain_notalign(nn.Module):
    def __init__(self, last_dims, selected_layers=[1]):
        super(sup_cifar_chain_notalign, self).__init__()        
        self.cnn_chn = [3*32*9,32*64*9,64*128*9,128*256*9,2]
        #'G' : [32,     'M', 64,      'M', 128,           'M', 256],
        self.meta_fcs = []
        for i in range(3):
            self.meta_fcs.append(nn.Linear(self.cnn_chn[i], last_dims))
        self.meta_fcs=nn.ModuleList(self.meta_fcs)
        self.selected_layers = selected_layers

    # chain
    def forward(self, x_list):
        x_out=[]
        for i in range(3):
            x = x_list[i]
            x=x.view(x.size(0),-1)
            x=self.meta_fcs[i](x)
            x_out.append(x) 

        x = x_out[self.selected_layers[0]-1]
        if len(self.selected_layers)==1:
            pass
        else:
            for i in range(1,len(self.selected_layers)):
                x = x +x_out[self.selected_layers[i]-1]
        return x