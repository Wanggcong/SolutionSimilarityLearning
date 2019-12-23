# -*- coding: utf-8 -*
import torch
import glob
import unicodedata
import string
import random


def findFiles(path): return glob.glob(path)
# Find letter index from all_letters, e.g. "a" = 0

class NameData():
    def __init__(self, root_path='./data/names/*.txt'):
        self.root_path = root_path 
        self.all_letters = string.ascii_letters + " .,;'-"
        self.n_letters = len(self.all_letters)
        # Build the category_lines dictionary, a list of lines per category
        self.category_lines = {}
        self.all_categories = []
        self.root_path = root_path
        self.n_categories = len(self.all_categories)

    # Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
    def unicodeToAscii(self,s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
            and c in self.all_letters
        )

    # Read a file and split into lines
    def readLines(self,filename):
        lines = open(filename).read().strip().split('\n')
        # return [self.unicodeToAscii(line) for line in lines]
        return [self.unicodeToAscii(line.decode("UTF-8")) for line in lines]

    def read_dataset(self):
        for filename in findFiles(self.root_path):
            category = filename.split('/')[-1].split('.')[0]
            self.all_categories.append(category)
            lines = self.readLines(filename)
            self.category_lines[category] = lines
        self.n_categories = len(self.all_categories)

    def initialize(self, indexes):
        # read dataset
        self.read_dataset()
        train_category_lines, test_category_lines = {}, {}
        for i in range(2):
            one_class = self.all_categories[indexes[i]]
            # one_class = name_category[i]
            one_cls_list = self.category_lines[one_class]
            # one list is split into two lists
            train_instances = random.sample(one_cls_list,int(len(one_cls_list)/2))
            test_instances = list(set(one_cls_list)-set(train_instances))
            train_category_lines[one_class] = train_instances
            test_category_lines[one_class] = test_instances
        return train_category_lines, test_category_lines

if __name__ == '__main__':
    root_path = '../data/names/*.txt'
    name_dataset = NameData(root_path)
    train_category_lines, test_category_lines = name_dataset.initialize()










