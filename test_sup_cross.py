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


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='PyTorch Cifar100 Example')
    dataset_name = 'cifar100'    #########
    # dataset_name = 'TinyImageNet'   ######
    # dataset_name = 'NameData'       ########
    myMetaOptions=MetaOptions(parser,dataset_name)
    myMetaOptions.initialize_datasets()
    opts = myMetaOptions.parser.parse_args()
    print('configures:',opts._get_kwargs)
    torch.manual_seed(opts.seed)          ########


    cls_or_retr= not opts.cls_or_retr
   

    if dataset_name=='cifar100' or dataset_name=='TinyImageNet':
        class_num=50
        # if cls_or_retr:
        #     class_num=50
        # else:
        #     class_num=30
    else:   # NameData
        class_num=9
        # if cls_or_retr:
        #     class_num=9
        # else:
        #     class_num=5        

    meta_set = split_datasets(opts.root_path, cls_or_retr=cls_or_retr, train_classes=class_num, query_num=20)  ####### query_num=20
    meta_set.split_all_test()
    if cls_or_retr:
        print('debug if overlap exits between training and test set:',set(meta_set.train_set).intersection(set(meta_set.test_set)))
    else:
        # print('meta_set.query_set:',meta_set.query_set)
        print('debug if overlap exits between training and query set:',set(meta_set.train_set).intersection(set(meta_set.query_set)))
        print('debug if overlap exits between training and gallay set:',set(meta_set.train_set).intersection(set(meta_set.gal_set)))
        print('debug if overlap exits between gallay and query set:',set(meta_set.gal_set).intersection(set(meta_set.query_set)))

    selected_layers=[int(i) for i in opts.selected_layers]
    print('selected_layers:',selected_layers)
    if opts.meta_model=='cifar_mlp':
        model =  MLPNet() 
        sup_model =  sup_cifar_mlp(class_num,selected_layers)  
    elif opts.meta_model=='cifar_chain':
        model =  plain_net5()  
        sup_model =  sup_cifar_chain(class_num,selected_layers)
    elif opts.meta_model=='cifar_chain_leaky_relu':
        # model =  plain_net5()  
        model = plain_net5_leaky_relu(2).cuda()  
        sup_model =  sup_cifar_chain(class_num,selected_layers)   
    elif opts.meta_model=='cifar_chain_diff':
        model = plain_net6_diff(2).cuda()   
        sup_model =  sup_cifar_chain(class_num,selected_layers)   
    elif opts.meta_model=='cifar_chain_resnet5':
        model = resnet5(2).cuda()      
        sup_model =  sup_cifar_chain(class_num,selected_layers)                
    elif opts.meta_model=='TinyImageNet_mlp':
        model =  MLPNet(4) 
        sup_model =  sup_cifar_mlp(class_num,selected_layers)  
    elif opts.meta_model=='TinyImageNet_chain':
        model =  plain_net5(4)  
        sup_model =  sup_cifar_chain(class_num,selected_layers)

    elif opts.meta_model=='rnn':
        input_size = len(string.ascii_letters + " .,;'-")                 
        model =  RNN(input_size, 128, 2)   #input_size, hidden_size, output_size
        sup_model =  sup_namedata_rnn(class_num)
    else:
        model =  plain_net5()             
        sup_model =  sup_cifar(class_num) # no selected_layers

    print('model:',model)


    # loading the super model:
    my_path = os.path.join("weights/",'meta_'+dataset_name)
    file_path = os.path.join(my_path, opts.target_model+'_'+opts.model_path+".pt")
    checkpoint = torch.load(file_path) 
    sup_model = nn.DataParallel(sup_model).cuda()
    sup_model.load_state_dict(checkpoint)

    meta_set.test_set = sorted(meta_set.test_set)
    # print('meta_set.test_set:',meta_set.test_set)
    if cls_or_retr:
        dataset = Cifar100Meta(model, cls_or_retr, meta_set.train_set, meta_set.train_targets, meta_set.test_set, meta_set.test_targets)                                        #########
        test_loader = dataset.initialize_one_set(opts, meta_set.test_set, meta_set.test_targets)            
    else:
        dataset = Cifar100Meta(model, cls_or_retr, meta_set.train_set, meta_set.train_targets, query_dataset=meta_set.query_set,
            query_targets=meta_set.query_targets,gal_dataset=meta_set.gal_set,gal_targets=meta_set.gal_targets)
        # train_loader, query_loader, gal_loader = dataset.initialize(opts)
        query_loader = dataset.initialize_one_set(opts, meta_set.query_set, meta_set.query_targets) 
        gal_loader = dataset.initialize_one_set(opts, meta_set.gal_set, meta_set.gal_targets) 

    # optimizer = optim.SGD(sup_model.parameters(), lr=opts.lr, momentum=opts.momentum)   ###
    optimizer = optim.Adam(model.parameters(), lr=opts.lr*0.01, betas=(0.9,0.999))
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[opts.step1], gamma=0.1)
    criterion = nn.CrossEntropyLoss().cuda()

    # trainer and tester
    log_file = './logs/'+opts.meta_model+'_'+opts.model_path+'_'+opts.log_file+'.log'
    myTrainTest = TrainTest(sup_model, criterion, optimizer)

    # train and test
    if cls_or_retr:
        myTrainTest.test_cls(test_loader, opts, log_file) 
    else:
        myTrainTest.test_retr(query_loader, gal_loader, opts, log_file) 
    print('configures:',opts._get_kwargs)
    