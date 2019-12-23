import torchvision
import torchvision.transforms as transforms
import random
# from utils import *
from utils.read_nested_folders import read_folders

class split_datasets():
    def __init__(self, root_path,cls_or_retr=True,train_classes=30, train_num=60,query_num=20):
    # def __init__(self, root_path,cls_or_retr=True,train_classes=30, train_num=6,query_num=2):
        self.root_path = root_path
        self.train_num = train_num

        self.train_classes = train_classes
        self.query_num = query_num
        self.cls_or_retr = cls_or_retr # True for classification, False for retrieval
        self.train_set = []
        self.train_targets = []
        self.test_set = []
        self.test_targets = []

        self.query_set = []
        self.query_targets = []
        self.gal_set = []
        self.gal_targets = []        
    def split(self):
        # read file lists
        files = read_folders(self.root_path)
        files.get_nested_folders()
        nested_files = files.nested_files
        print('nested_files:',nested_files)

        # random splits
        if self.cls_or_retr:
            # for classification
            for label, one_folder in enumerate(nested_files):
                train_instances = random.sample(one_folder,self.train_num)
                self.train_set+=train_instances
                test_instances=list(set(one_folder)-set(train_instances))
                self.test_set += test_instances

                self.train_targets+=[label for i in range(len(train_instances))]
                self.test_targets+=[label for i in range(len(test_instances))]
        else:
            # for retrieval
            random.shuffle(nested_files)
            for i in range(len(nested_files)):
                if i<self.train_classes:
                    self.train_set+=nested_files[i]
                    self.train_targets+=[i for ii in range(len(nested_files[i]))]
                else:
                    one_folder = nested_files[i]
                    query_instances = random.sample(one_folder,self.query_num)

                    self.query_set+=query_instances
                    gal_instances=list(set(one_folder)-set(query_instances))
                    self.gal_set += gal_instances    

                    self.query_targets+=[i for ii in range(len(query_instances))]
                    self.gal_targets+=[i for ii in range(len(gal_instances))]
            print('#### lalala ####')

    def split_all_test(self):
        # read file lists
        files = read_folders(self.root_path)
        files.get_nested_folders()
        nested_files = files.nested_files

        # random splits
        if self.cls_or_retr:
            # fro classification
            for label, one_folder in enumerate(nested_files):
                self.test_set+=one_folder
                self.test_targets+=[label for i in range(len(one_folder))]
        else:
            # for retrieval
            random.shuffle(nested_files)
            for i in range(len(nested_files)):
                one_folder = nested_files[i]
                query_instances = random.sample(one_folder,self.query_num)

                self.query_set+=query_instances
                gal_instances=list(set(one_folder)-set(query_instances))
                self.gal_set += gal_instances    

                self.query_targets+=[i for ii in range(len(query_instances))]
                self.gal_targets+=[i for ii in range(len(gal_instances))]
            print('#### lalala ####')