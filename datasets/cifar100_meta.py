from __future__ import print_function
import torch
from torchvision import datasets, transforms
import torch.utils.data.sampler as S 
from utils.model2data import MetaData


class Preprocessor(object):
    def __init__(self, model, dataset_path_list, target_list):
        self.dataset_path_list = dataset_path_list  
        self.target_list = target_list 
        self.length = 0
        self.model = model

    def __len__(self):
        return len(self.dataset_path_list)

    def __getitem__(self,index):
        model_path = self.dataset_path_list
        meta_data_transformer = MetaData(self.model,self.dataset_path_list[index])
        meta_data_transformer.param2data_vanilla()
        target = self.target_list[index]
        return meta_data_transformer.model_data, target 


class Cifar100Meta():
    def __init__(self, model, cls_or_retr, train_dataset, train_targets, test_dataset=None,test_targets=None,
        query_dataset=None,query_targets=None,gal_dataset=None,gal_targets=None):
        self.cls_or_retr = cls_or_retr   
        self.model = model   
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.query_dataset = query_dataset
        self.gal_dataset = gal_dataset

        self.train_targets = train_targets
        self.test_targets = test_targets
        self.query_targets = query_targets
        self.gal_targets = gal_targets

    def initialize(self, opts):
        trainloader = torch.utils.data.DataLoader(Preprocessor(self.model, self.train_dataset, self.train_targets), 
            batch_size=opts.batch_size, shuffle=True)   
        if self.cls_or_retr:
            testloader = torch.utils.data.DataLoader(Preprocessor(self.model, self.test_dataset,self.test_targets), 
                batch_size=1, shuffle=False)
            return trainloader, testloader
        else:
            queryloader = torch.utils.data.DataLoader(Preprocessor(self.model, self.query_dataset,self.query_targets), 
                batch_size=1, shuffle=False)
            galloader = torch.utils.data.DataLoader(Preprocessor(self.model, self.gal_dataset,self.gal_targets), 
                batch_size=1, shuffle=False)
            return trainloader, queryloader, galloader
    # regard the training set as a common set
    def initialize_one_set(self, opts, one_set, one_target):
        oneloader = torch.utils.data.DataLoader(Preprocessor(self.model, one_set, one_target), 
            batch_size=opts.batch_size, shuffle=False)  
        return oneloader
