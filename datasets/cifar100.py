from __future__ import print_function
import torch
from torchvision import datasets, transforms
import torch.utils.data.sampler as S 
# from utils import *
from utils.getSubset import getSubset


class cifar100():
    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, opts, use_cuda, rounds):
        print('=> loading cifar100 data...')
        normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])    
        kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}
        train_dataset = datasets.CIFAR100(
            root='./data/cifar100',
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
        # change labels
        train_indices = getSubset(train_dataset.train_labels, 2*rounds, 2*rounds+1)
        print('train_indices:',len(train_indices))
        for ind in train_indices:
            if train_dataset.train_labels[ind]==2*rounds:            ###
                train_dataset.train_labels[ind] = 0
            else:
                train_dataset.train_labels[ind] = 1    
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=opts.batch_size, sampler =S.SubsetRandomSampler(train_indices), **kwargs)    

        test_dataset = datasets.CIFAR100(
            root='./data/cifar100',
            train=False,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]))

        test_indices = getSubset(test_dataset.test_labels, 2*rounds, 2*rounds+1)
        print('test_indices:',len(test_indices))
        
        for ind in test_indices:
            if test_dataset.test_labels[ind]==2*rounds:
                test_dataset.test_labels[ind] = 0
            else:
                test_dataset.test_labels[ind] = 1    
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=100, sampler =S.SubsetRandomSampler(test_indices), **kwargs)
        
        return trainloader, testloader