import argparse
import os



class TrainOptions():
    def __init__(self,parser):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False
        self.parser = parser

    def initialize(self):
        self.parser.add_argument('--no-cuda', action='store_true', default=False,
                            help='disables CUDA training')
        self.parser.add_argument('--seed', type=int, default=1, metavar='S',
                            help='random seed (default: 1)')  
    
    def initialize_mnist(self):
        """Define the common options that are used in both training and test."""
        # basic parameters
        self.initialize()
        self.parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                            help='input batch size for training (default: 64)')
        self.parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                            help='input batch size for testing (default: 1000)')
        self.parser.add_argument('--epochs', type=int, default=10, metavar='N',
                            help='number of epochs to train (default: 10)')
        self.parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                            help='learning rate (default: 0.01)')
        self.parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                            help='SGD momentum (default: 0.5)')
        self.parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                            help='how many batches to wait before logging training status')
        self.parser.add_argument('--save-model', action='store_true',
                            help='For Saving the current Model')
        self.parser.add_argument('--step1', default=30, type=int, metavar='N', 
                            help='step1 lr')     
        self.parser.add_argument('--model-name', type=str, default='MLPNet', metavar='M',
                            help='meta model type')
        self.parser.add_argument('--model-path', type=str, default='mnist_v3', metavar='M',              
                            help='meta model type')     
        self.parser.add_argument('--optimizer', type=str, default='sgd', metavar='M',             
                            help='meta model type')                                   
    def initialize_cifar100(self):  
        self.initialize()  
        self.parser.add_argument('--epochs', default=50, type=int, metavar='N',                   
                            help='number of total epochs to run')
        self.parser.add_argument('--step1', default=30, type=int, metavar='N',                     
                            help='step1 lr')   
        self.parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N', 
                            help='mini-batch size (default: 128),only used for train')
        self.parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', 
                            help='initial learning rate')
        self.parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
        self.parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', 
                            help='weight decay (default: 1e-4)')
        self.parser.add_argument('--log-interval', type=int, default=10, metavar='N',                      
                            help='how many batches to wait before logging training status')
        self.parser.add_argument('--not-save-model', action='store_true', 
                            help='For Saving the current Model')
        self.parser.add_argument('--model-name', type=str, default='plain_net5', metavar='M',
                            help='meta model type')  
        self.parser.add_argument('--model-path', type=str, default='cifar100_v3', metavar='M',          
                            help='meta model type')  
        self.parser.add_argument('--tasks', type=int, default=0, metavar='N',                      
                            help='tasks xx begins...')        
        self.parser.add_argument('--optimizer', type=str, default='sgd', metavar='M',             
                            help='meta model type')
    def initialize_namedata(self):  
        self.initialize()  
        self.parser.add_argument('--epochs', default=10000, type=int, metavar='N',                  
                            help='number of total epochs to run')
        self.parser.add_argument('--step1', default=7000, type=int, metavar='N',                    
                            help='step1 lr')   
        self.parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N', 
                            help='mini-batch size (default: 128),only used for train')
        self.parser.add_argument('--lr', '--learning-rate', default=0.005, type=float, metavar='LR', 
                            help='initial learning rate')
        self.parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
        self.parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', 
                            help='weight decay (default: 1e-4)')
        self.parser.add_argument('--log-interval', type=int, default=5000, metavar='N',                    
                            help='how many batches to wait before logging training status')
        self.parser.add_argument('--not-save-model', action='store_true', 
                            help='For Saving the current Model')
        self.parser.add_argument('--model-name', type=str, default='rnn', metavar='M',
                            help='meta model type')  
        self.parser.add_argument('--model-path', type=str, default='rnn_v2', metavar='M',         
                            help='meta model type')  
        self.parser.add_argument('--tasks', type=int, default=0, metavar='N',                    
                            help='tasks xx begins...') 
        self.parser.add_argument('--n-hidden', type=int, default=128, metavar='N',                   
                            help='n-hidden') 
        self.parser.add_argument('--optimizer', type=str, default='sgd', metavar='M',            
                            help='meta model type')
    def initialize_tinyImageNet(self):  
        self.initialize()  
        self.parser.add_argument('--epochs', default=150, type=int, metavar='N',                 
                            help='number of total epochs to run')
        self.parser.add_argument('--step1', default=100, type=int, metavar='N',                    
                            help='step1 lr')   
        self.parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N', 
                            help='mini-batch size (default: 128),only used for train')
        self.parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', 
                            help='initial learning rate')
        self.parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
        self.parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', 
                            help='weight decay (default: 1e-4)')
        self.parser.add_argument('--log-interval', type=int, default=10, metavar='N',                   
                            help='how many batches to wait before logging training status')
        self.parser.add_argument('--not-save-model', action='store_true', 
                            help='For Saving the current Model')
        self.parser.add_argument('--model-name', type=str, default='plain_net5', metavar='M',
                            help='meta model type')  
        self.parser.add_argument('--model-path', type=str, default='cifar100_v3', metavar='M',           
                            help='meta model type')  
        self.parser.add_argument('--tasks', type=int, default=0, metavar='N',                     
                            help='tasks xx begins...')         
        self.parser.add_argument('--optimizer', type=str, default='sgd', metavar='M',              
                            help='meta model type')
