import argparse
import os
import time
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from random import sample
import numpy as np
from torch.optim import lr_scheduler
from models import *
from options import *
from datasets import *
from utils import *

def train(opts, model, criterion, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % opts.log_interval == 0:
            print('Train Epoch: {} [running({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, 100. * batch_idx / len(train_loader), loss.item()))

def test(opts, model, criterion, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    point_numbers = 0
    batch_num =0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
        # for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            point_numbers = point_numbers + data.size(0)
            batch_num = batch_num+1
    test_loss /= batch_num
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, point_numbers, 100. * correct / point_numbers))

if __name__=='__main__':
    datasets = 'cifar100'    # change this line when using other datasets
    # datasets = 'tinyImageNet'    # change this line when using other datasets
    # model_path = 'cifar100_mlp' ############ in opts
    parser = argparse.ArgumentParser(description='PyTorch Cifar100 Example')
    parser_trained=TrainOptions(parser)
    if datasets == 'mnist':
        parser_trained.initialize_mnist()
    elif datasets == 'cifar100':
        parser_trained.initialize_cifar100()
    else:
        parser_trained.initialize_tinyImageNet()                #####
    opts = parser_trained.parser.parse_args()
    for ii in range(opts.tasks,opts.tasks+5):    #cifar100, 50 tasks
        # continue if models have been generated.
        my_path = os.path.join("weights/",opts.model_path,str(ii))            
        if not os.path.exists(os.path.join("weights/",opts.model_path)):
            os.mkdir(os.path.join("weights/",opts.model_path))
        if not os.path.exists(my_path):
            os.mkdir(my_path)

        use_cuda = not opts.no_cuda and torch.cuda.is_available()
        criterion = nn.CrossEntropyLoss()
        device = torch.device("cuda" if use_cuda else "cpu")  
        if datasets == 'mnist':
            train_loader, test_loader = mnist().initialize(opts, use_cuda)
        elif datasets == 'cifar100':
            train_loader, test_loader = cifar100().initialize(opts, use_cuda,ii)    ##### identicial data split
        else:
            if opts.model_name=='mlp':
                train_loader, test_loader = TinyImageNet2().initialize(opts, use_cuda,ii)    ##### identicial data split 
            else:
                train_loader, test_loader = TinyImageNet().initialize(opts, use_cuda,ii)    ##### identicial data split 

        for kk in range(100):     
            file_path = os.path.join(my_path,str(kk)+".pt")  
            if os.path.exists(file_path): 
                print('This model has been generated:',ii,kk)
                continue
            # torch.manual_seed(opts.seed) # remove this line to obtain different initialized trained models.
            if datasets == 'mnist':
                model = Net().cuda()
            elif datasets == 'cifar100':
                # model = vgg11_bn().cuda()
                # model = vgg7_like_bn().cuda()
                if opts.model_name=='plain_net5':
                    model = plain_net5(2).cuda()
                elif opts.model_name=='plain_net6_diff':
                    model = plain_net6_diff(2).cuda()
                elif opts.model_name=='resnet8':
                    model = resnet8(2).cuda()                        
                elif opts.model_name=='plain_net5_leaky_relu':
                    model = plain_net5_leaky_relu(2).cuda()                                                
                elif opts.model_name=='resnet5':
                    model = resnet5(2).cuda() 
                else:
                    model = MLPNet(2).cuda()
            else:
                if opts.model_name=='plain_net5':
                    model = plain_net5(4).cuda()
                elif opts.model_name=='plain_net6_diff':
                    model = plain_net6_diff(4).cuda()
                elif opts.model_name=='resnet8':
                    model = resnet8(4).cuda()                    
                else:
                    model = MLPNet(4).cuda()   
            print(model)             
            # train_loader, test_loader = cifar100().initialize(opts, use_cuda,ii)  
            if opts.optimizer=='sgd':
                optimizer = optim.SGD(model.parameters(), lr=opts.lr, momentum=opts.momentum)
            else:
                optimizer = optim.Adam(model.parameters(), lr=opts.lr*0.01, betas=(0.9,0.999))
            scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[opts.step1], gamma=0.1)
            for epoch in range(1, opts.epochs + 1):
                print('which subset and which rounds:',ii, kk)
                train(opts, model, criterion, device, train_loader, optimizer, epoch)
                test(opts, model, criterion, device, test_loader)
                scheduler.step()    
            if not (opts.not_save_model):
                print('save model ...')
                file_path = os.path.join(my_path,str(kk)+".pt")
                torch.save(model.state_dict(),file_path)