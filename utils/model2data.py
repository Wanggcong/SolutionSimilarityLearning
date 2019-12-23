import torch
import torch.nn as nn
class MetaData():
    def __init__(self, model, model_path, kernel_size=3):
        self.model = model
        self.model_path = model_path
        self.kernel_size = kernel_size
        self.model_data = []
    def re_load(self):
        checkpoint = torch.load(self.model_path)
        self.model.load_state_dict(checkpoint)
    def param2data_vanilla(self):
        self.re_load()
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                if m.weight.data.size(2) ==self.kernel_size:
                    c_out, c_in,ker_h, ker_w = m.weight.data.size()
                    self.model_data.append(m.weight.data)
            if isinstance(m, nn.Linear):
                self.model_data.append(m.weight.data)   

            if isinstance(m, nn.GRU):
                self.model_data.append(m.weight_ih_l0.data)   
                self.model_data.append(m.weight_ih_l1.data)    
                self.model_data.append(m.weight_hh_l0.data)   
                self.model_data.append(m.weight_hh_l1.data)   

    def param2data(self):
        self.re_load()
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                if m.weight.data.size(2) ==self.kernel_size:
                    c_out, c_in,ker_h, ker_w = m.weight.data.size()
                    self.model_data.append(m.weight.data.view(1,c_out, -1))
    
    def param2data_svd(self):
        self.re_load()
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                if m.weight.data.size(2) ==self.kernel_size:
                    c_out, c_in,ker_h, ker_w = m.weight.data.size()
                    # add svd here
                    layer_para = m.weight.data.view(c_out, -1)
                    u, s, v = torch.svd(layer_para)
                    layer_para_compressed = torch.mm(s*torch.eye(c_out),v.t()) 
                    self.model_data.append(layer_para_compressed.view(1, c_out, -1))

    def debug_each_layer_para(self):
        for i in range(len(self.model_data)):
            para = self.model_data[i]
            print('layer and para:', i, para.size())


