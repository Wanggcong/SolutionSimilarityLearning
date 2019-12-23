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
    # dataset_name = 'cifar100'    #####
    # dataset_name = 'TinyImageNet'  ######
    dataset_name = 'NameData'    #######
    myMetaOptions=MetaOptions(parser,dataset_name)
    myMetaOptions.initialize_datasets()
    opts = myMetaOptions.parser.parse_args()
    print('configures:',opts._get_kwargs)
    torch.manual_seed(opts.seed)   ######
    # cls_or_retr=True  #####
    # cls_or_retr=False   ####

    cls_or_retr= not opts.cls_or_retr
   

    if dataset_name=='cifar100' or dataset_name=='TinyImageNet':
        if cls_or_retr:
            class_num=50
        else:
            class_num=30
    else:   # NameData
        if cls_or_retr:
            class_num=9
        else:
            class_num=5        

    # data prepare
    meta_set = split_datasets(opts.root_path, cls_or_retr=cls_or_retr, train_classes=class_num)
    meta_set.split()
    # print('train_set files',meta_set.train_set)
    # print('test_set files',meta_set.test_set)
    if cls_or_retr:
        print('debug if overlap exits between training and test set:',set(meta_set.train_set).intersection(set(meta_set.test_set)))
    else:
        # print('meta_set.query_set:',meta_set.query_set)
        print('debug if overlap exits between training and query set:',set(meta_set.train_set).intersection(set(meta_set.query_set)))
        print('debug if overlap exits between training and gallay set:',set(meta_set.train_set).intersection(set(meta_set.gal_set)))
        print('debug if overlap exits between gallay and query set:',set(meta_set.gal_set).intersection(set(meta_set.query_set)))

    selected_layers=[int(i) for i in opts.selected_layers]
    print('selected_layers:',selected_layers)
    if opts.meta_model=='cifar_mlp_notalign':
        model =  MLPNet() 
        sup_model =  sup_cifar_mlp_notalign(class_num,selected_layers)  
    elif opts.meta_model=='cifar_chain_notalign':
        model =  plain_net5()  
        sup_model =  sup_cifar_chain_notalign(class_num,selected_layers)
    elif opts.meta_model=='TinyImageNet_mlp_notalign':
        model =  MLPNet(4) 
        sup_model =  sup_cifar_mlp_notalign(class_num,selected_layers)  
    elif opts.meta_model=='TinyImageNet_chain_notalign':
        model =  plain_net5(4)  
        sup_model =  sup_cifar_chain_notalign(class_num,selected_layers)
    elif opts.meta_model=='rnn_notalign':
        input_size = len(string.ascii_letters + " .,;'-")                 
        model =  RNN(input_size, 128, 2)   #input_size, hidden_size, output_size
        sup_model =  sup_namedata_rnn_notalign(class_num)
    else:
        if dataset_name=='cifar100':
            model =  plain_net5()             
        else:                               
            model =  plain_net5(4)             ## for tiny ImageNet
        sup_model =  sup_cifar(class_num, selected_layers) # no selected_layers            

    if cls_or_retr:
        dataset = Cifar100Meta(model, cls_or_retr, meta_set.train_set, meta_set.train_targets, meta_set.test_set, meta_set.test_targets)   ####
        train_loader, test_loader = dataset.initialize(opts)            
    else:
        dataset = Cifar100Meta(model, cls_or_retr, meta_set.train_set, meta_set.train_targets, query_dataset=meta_set.query_set,
            query_targets=meta_set.query_targets,gal_dataset=meta_set.gal_set,gal_targets=meta_set.gal_targets)
        train_loader, query_loader, gal_loader = dataset.initialize(opts)


    sup_model = nn.DataParallel(sup_model).cuda()

    # optimization prepare
    optimizer = optim.SGD(sup_model.parameters(), lr=opts.lr, momentum=opts.momentum)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[opts.step1], gamma=0.1)
    criterion = nn.CrossEntropyLoss().cuda()

    # trainer and tester
    log_file = './logs/'+opts.meta_model+'_'+opts.model_path+'_'+opts.log_file+'.log'
    myTrainTest = TrainTest(sup_model, criterion, optimizer)

    # train and test
    for epoch in range(1, opts.epochs + 1):  
        myTrainTest.train(train_loader, epoch, opts, log_file)  
        if cls_or_retr:
            myTrainTest.test_cls(test_loader, opts, log_file) 
            # myTrainTest.test_cls_topk(test_loader, opts, log_file) 
        else:
            myTrainTest.test_retr(query_loader, gal_loader, opts, log_file) 
        scheduler.step()    
        # if epoch%opts.log_interval==0:
        print('configures:',opts._get_kwargs)
    
    # save model or not
    my_path = os.path.join("weights/",'meta_'+dataset_name)
    if not os.path.exists(my_path):
        os.mkdir(my_path)
    if (opts.not_save_model):                                       
        print('save super model ...')
        file_path = os.path.join(my_path, opts.meta_model+'_'+opts.model_path+".pt")  ####
        torch.save(sup_model.state_dict(),file_path)  
