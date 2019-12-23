from __future__ import print_function
import torch
from torchvision import datasets, transforms
import torch.utils.data.sampler as S 
# from utils import *
from utils.getSubset import getSubset,getSubset4

def tuple_modified(one_tuple,new_label):
    one_list = list(one_tuple)
    one_list[1] =new_label
    one_tuple_new = tuple(one_list)
    return one_tuple_new


class TinyImageNet():
    def __init__(self, root_path_train='./data/tiny-imagenet-200/train/', root_path_test='./data/tiny-imagenet-200/val_new/'):
        """Reset the class; indicates the class hasn't been initailized"""
        self.root_path_train = root_path_train
        self.root_path_test = root_path_test

    def initialize(self, opts, use_cuda, rounds):
        print('=> loading TinyImageNet data...')
        normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])    #####
        kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}
        train_dataset = datasets.ImageFolder(self.root_path_train,
            transform=transforms.Compose([
                transforms.Resize(size=(32,32),interpolation=2), # added, 2019.09.11
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

        # change labels
        train_labels = [train_dataset.imgs[i][1] for i in range(train_dataset.__len__())]
        train_indices = getSubset4(train_labels, 4*rounds, 4*rounds+1, 4*rounds+2, 4*rounds+3)

        # print('train_indices:',len(train_indices))
        for ind in train_indices:
            if train_labels[ind]==4*rounds:            ###
                train_dataset.imgs[ind]=tuple_modified(train_dataset.imgs[ind],0)
            elif train_labels[ind]==4*rounds+1: 
                train_dataset.imgs[ind]=tuple_modified(train_dataset.imgs[ind],1)
            elif train_labels[ind]==4*rounds+2: 
                train_dataset.imgs[ind]=tuple_modified(train_dataset.imgs[ind],2)
            else:
                train_dataset.imgs[ind]=tuple_modified(train_dataset.imgs[ind],3)

        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=opts.batch_size, sampler =S.SubsetRandomSampler(train_indices), **kwargs)     

        test_dataset = datasets.ImageFolder(self.root_path_test,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]))
        test_labels = [test_dataset.imgs[i][1] for i in range(test_dataset.__len__())]
        test_indices = getSubset4(test_labels, 4*rounds, 4*rounds+1, 4*rounds+2, 4*rounds+3)
        
        for ind in test_indices:
            if test_labels[ind]==4*rounds:            ###
                test_dataset.imgs[ind]=tuple_modified(test_dataset.imgs[ind],0)
            elif test_labels[ind]==4*rounds+1: 
                test_dataset.imgs[ind]=tuple_modified(test_dataset.imgs[ind],1)
            elif test_labels[ind]==4*rounds+2: 
                test_dataset.imgs[ind]=tuple_modified(test_dataset.imgs[ind],2)
            else:
                test_dataset.imgs[ind]=tuple_modified(test_dataset.imgs[ind],3)

        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=100, sampler =S.SubsetRandomSampler(test_indices), **kwargs)        
        return trainloader, testloader


class TinyImageNet2():
    def __init__(self, root_path_train='./data/tiny-imagenet-200/train/', root_path_test='./data/tiny-imagenet-200/val_new/'):
        """Reset the class; indicates the class hasn't been initailized"""
        self.root_path_train = root_path_train
        self.root_path_test = root_path_test

    def initialize(self, opts, use_cuda, rounds):
        print('=> loading TinyImageNet data...')
        normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])    #####
        kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}
        train_dataset = datasets.ImageFolder(self.root_path_train,
            transform=transforms.Compose([
                transforms.Resize(size=(32,32),interpolation=2),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

        # change labels
        train_labels = [train_dataset.imgs[i][1] for i in range(train_dataset.__len__())]
        train_indices = getSubset4(train_labels, 4*rounds, 4*rounds+1, 4*rounds+2, 4*rounds+3)

        # print('train_indices:',len(train_indices))
        for ind in train_indices:
            if train_labels[ind]==4*rounds:     
                train_dataset.imgs[ind]=tuple_modified(train_dataset.imgs[ind],0)
            elif train_labels[ind]==4*rounds+1: 
                train_dataset.imgs[ind]=tuple_modified(train_dataset.imgs[ind],1)
            elif train_labels[ind]==4*rounds+2: 
                train_dataset.imgs[ind]=tuple_modified(train_dataset.imgs[ind],2)
            else:
                train_dataset.imgs[ind]=tuple_modified(train_dataset.imgs[ind],3)

        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=opts.batch_size, sampler =S.SubsetRandomSampler(train_indices), **kwargs)    

        test_dataset = datasets.ImageFolder(self.root_path_test,
            transform=transforms.Compose([
                transforms.Resize(size=(32,32),interpolation=2),
                transforms.ToTensor(),
                normalize,
            ]))
        test_labels = [test_dataset.imgs[i][1] for i in range(test_dataset.__len__())]
        test_indices = getSubset4(test_labels, 4*rounds, 4*rounds+1, 4*rounds+2, 4*rounds+3)
        
        for ind in test_indices:
            if test_labels[ind]==4*rounds:            ###
                test_dataset.imgs[ind]=tuple_modified(test_dataset.imgs[ind],0)
            elif test_labels[ind]==4*rounds+1: 
                test_dataset.imgs[ind]=tuple_modified(test_dataset.imgs[ind],1)
            elif test_labels[ind]==4*rounds+2: 
                test_dataset.imgs[ind]=tuple_modified(test_dataset.imgs[ind],2)
            else:
                test_dataset.imgs[ind]=tuple_modified(test_dataset.imgs[ind],3)

        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=100, sampler =S.SubsetRandomSampler(test_indices), **kwargs)        
        return trainloader, testloader
