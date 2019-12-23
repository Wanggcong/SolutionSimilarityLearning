import argparse
import os
import time
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
from random import sample
import numpy as np
from sklearn.metrics import average_precision_score
from models import *
from utils import *
from datasets import *
from options import *

from utils.model2data import MetaData


def re_load(model,model_path):
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)
    return model


def model2vector(meta_model,model_path):
    cls_or_retr=True
    if cls_or_retr:
        class_num=50
    else:
        class_num=30

    if meta_model=='cifar_mlp':
        # model =  MLPNet() 
        sup_model =  sup_cifar_mlp(class_num)  
    elif meta_model=='cifar_chain':
        # model =  plain_net5()  
        sup_model =  sup_cifar_chain(class_num)
    else:
        # model =  plain_net5()  
        sup_model =  sup_cifar(class_num) 

    # print('sup_model:',sup_model)
    sup_model = nn.DataParallel(sup_model).cuda()
    # load data
    model_data = re_load(sup_model,model_path)
    # model_data1 = meta_data_transformer1.param2data_vanilla()
    # print('model_data1:',model_data.module.meta_fcs[0].weight.data)
    vec = model_data.module.meta_fcs[0].weight.data

    # to vector
    print('vec size:', vec.size())
    vec = vec.view(-1,1)
    #norm
    vec = vec/torch.norm(vec)
    return vec

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='PyTorch Cifar100 Example')
    dataset_name = 'cifar100'                                                       
    myMetaOptions=MetaOptions(parser,dataset_name)
    myMetaOptions.initialize_datasets()
    opts = myMetaOptions.parser.parse_args()
    print('configures:',opts._get_kwargs)
    # torch.manual_seed(opts.seed)                                                    
    
   
    model1_path = 'weights/meta_cifar100/cifar_mlp_l1v2.pt'  # cifar_mlp_l1v2.pt, cifar_chain_l2v2.pt
    model2_path = 'weights/meta_cifar100/cifar_mlp_l1v3.pt'   
    meta_model = 'cifar_mlp' # cifar_chain

    vec1 = model2vector(meta_model, model1_path)
    vec2 = model2vector(meta_model, model2_path)

    sim = torch.mm(vec1.t(),vec2)
    print('sim:', sim)




