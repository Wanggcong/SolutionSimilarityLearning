import argparse
import os


class MetaOptions():
    def __init__(self,parser,dataset_name):
        """Reset the class; indicates the class hasn't been initailized"""
        self.parser = parser
        self.dataset_name = dataset_name

    def initialize(self):
        self.parser.add_argument('--seed', type=int, default=1, metavar='S',
                            help='random seed (default: 1)')  
    
    def initialize_datasets(self):
        """Define the common options that are used in both training and test."""
        # basic parameters
        self.initialize()
        if self.dataset_name == 'mnist':
            self.parser.add_argument('--root-path', type=str, 
                                default='/media/data2/anonymous/projects/LearnableParameterSimilarity/weights/mnist', metavar='RP',
                                help='root path for weights')    
            self.parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                                help='input batch size for training (default: 64)')
            self.parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                                help='input batch size for testing (default: 1000)')
            self.parser.add_argument('--epochs', type=int, default=100, metavar='N',
                                help='number of epochs to train (default: 10)')
            self.parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                                help='learning rate (default: 0.01)')
            self.parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                                help='SGD momentum (default: 0.5)')
            self.parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                                help='how many batches to wait before logging training status')
            self.parser.add_argument('--not-save-model', action='store_true', default=True,
                                help='For Saving the current Model')
            self.parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', 
                help='weight decay (default: 1e-4)')
            self.parser.add_argument('--meta-model', type=str, default='cifar_mlp', metavar='M',
                                help='meta model type')          
            self.parser.add_argument('--step1', default=30, type=int, metavar='N', 
                help='step1 lr') 
            self.parser.add_argument('--log-file', type=str, default='', metavar='M',
                                help='log file') 
            self.parser.add_argument('--selected-layers', type=str, default='0', metavar='M',
                               help='selected layers')        
            self.parser.add_argument('--cls-or-retr', action='store_true', 
                                help='True for classification, False for retrieval.')                                   
        elif self.dataset_name == 'cifar100' or self.dataset_name == 'TinyImageNet':
            self.parser.add_argument('--root-path', type=str, 
                                default='/media/data2/anonymous/projects/LearnableParameterSimilarity/weights/cifar100_100', metavar='RP',
                                help='root path for weights')                                        
            self.parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                                help='input batch size for training (default: 64)')
            self.parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                                help='input batch size for testing (default: 1000)')
            self.parser.add_argument('--epochs', type=int, default=100, metavar='N',
                                help='number of epochs to train (default: 10)')
            self.parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                                help='learning rate (default: 0.01)')
            self.parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                                help='SGD momentum (default: 0.5)')
            self.parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                                help='how many batches to wait before logging training status')
            self.parser.add_argument('--not-save-model', action='store_true', default=True,
                                help='For Saving the current Model')
            self.parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', 
                help='weight decay (default: 1e-4)')
            self.parser.add_argument('--meta-model', type=str, default='cifar_mlp', metavar='M',
                                help='meta model type')     
            self.parser.add_argument('--target-model', type=str, default='cifar_mlp', metavar='M',
                                help='target model type')                                               
            self.parser.add_argument('--log-file', type=str, default='', metavar='M',
                                help='log file')   
            self.parser.add_argument('--step1', default=30, type=int, metavar='N', 
                help='step1 lr') 
            self.parser.add_argument('--model-path', type=str, default='v1', metavar='M',
                                help='model path')   
            self.parser.add_argument('--selected-layers', type=str, default='0', metavar='M',
                               help='selected layers') 
            self.parser.add_argument('--cls-or-retr', action='store_true',
                                help='True for classification, False for retrieval.')                                                                             
        else:
            self.parser.add_argument('--root-path', type=str, 
                                default='/media/data2/anonymous/projects/LearnableParameterSimilarity/weights/cifar100_rnn_v1', metavar='RP',
                                help='root path for weights')                                        
            self.parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                                help='input batch size for training (default: 64)')
            self.parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                                help='input batch size for testing (default: 1000)')
            self.parser.add_argument('--epochs', type=int, default=100, metavar='N',
                                help='number of epochs to train (default: 10)')
            self.parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                                help='learning rate (default: 0.01)')
            self.parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                                help='SGD momentum (default: 0.5)')
            self.parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                                help='how many batches to wait before logging training status')
            self.parser.add_argument('--not-save-model', action='store_true', default=True,
                                help='For Saving the current Model')
            self.parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', 
                help='weight decay (default: 1e-4)')
            self.parser.add_argument('--meta-model', type=str, default='cifar_mlp', metavar='M',
                                help='meta model type')  
            self.parser.add_argument('--target-model', type=str, default='cifar_mlp', metavar='M',
                                help='target model type')                                                 
            self.parser.add_argument('--log-file', type=str, default='', metavar='M',
                                help='log file')   
            self.parser.add_argument('--step1', default=30, type=int, metavar='N', 
                help='step1 lr') 
            self.parser.add_argument('--model-path', type=str, default='v1', metavar='M',
                                help='model path')   
            self.parser.add_argument('--selected-layers', type=str, default='0', metavar='M',
                               help='selected layers')    
            self.parser.add_argument('--cls-or-retr', action='store_true', 
                                help='True for classification, False for retrieval.')             

# note: --meta-model is source model, and --target-model is target model.