from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
"""vgg in pytorch


[1] Karen Simonyan, Andrew Zisserman

    Very Deep Convolutional Networks for Large-Scale Image Recognition.
    https://arxiv.org/abs/1409.1556v6
"""
'''VGG11/13/16/19 in Pytorch.'''




# vgg
cfg = {
    'A' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'F' : [32,     'M', 64,      'M', 128,           'M', 256,           'M', 512,           'M'],
    'G' : [32,     'M', 64,      'M', 128,           'M', 256],
    'H' : [32,     'M', 32,      'M', 64,            'M', 128,       'M', 256],
    'I' : [32,     'M', 64,      'M', 128,            'M', 256,       'M', 256],
}

class VGG(nn.Module):

    def __init__(self, features, num_class=2):
        super(VGG, self).__init__()
        self.features = features

        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_class)
        )

    def forward(self, x):
        output = self.features(x)
        output = output.view(output.size()[0], -1)
        output = self.classifier(output)
    
        return output
class VGG_like(nn.Module):

    def __init__(self, features, num_class=2):
        super(VGG_like, self).__init__()
        self.features = features

        self.classifier = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, num_class)
        )

    def forward(self, x):
        output = self.features(x)
        output = output.view(output.size()[0], -1)
        output = self.classifier(output)
    
        return output

class plain_net(nn.Module):
    def __init__(self, features, num_class=2):
        super(plain_net, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(256, num_class),
        )

    def forward(self, x):
        output = self.features(x)
        output = F.avg_pool2d(output,(output.size(2),output.size(3)))
        output = output.view(output.size()[0], -1)
        output = self.classifier(output)
        return output

class plain_net_leaky_relu(nn.Module):
    def __init__(self, features, num_class=2):
        super(plain_net_leaky_relu, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(256, num_class),
        )

    def forward(self, x):
        output = self.features(x)
        output = F.avg_pool2d(output,(output.size(2),output.size(3)))
        output = output.view(output.size()[0], -1)
        output = self.classifier(output)
        return output

def make_layers(cfg, batch_norm=False):
    layers = []

    input_channel = 3
    for l in cfg:
        if l == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            continue
        
        layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]

        if batch_norm:
            layers += [nn.BatchNorm2d(l)]
        
        layers += [nn.ReLU(inplace=True)]
        input_channel = l
    
    return nn.Sequential(*layers)

# aim to design a very different nets, no pooling layers 
def make_layers2(cfg, batch_norm=False):
    layers = []

    input_channel = 3
    for i, l in enumerate(cfg):
        if l == 'M':
            # layers += [nn.MaxPool2d(kernel_size=2, stride=2)]            # no pooling, different depth
            continue
        if i==0:
            layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]
            print('********************** this is the first layer **************************')
        else:
            layers += [nn.Conv2d(input_channel, l, kernel_size=2, stride=2, padding=1)]  # different kernel size, different strides
        if batch_norm:
            layers += [nn.BatchNorm2d(l)]
        
        layers += [nn.ReLU(inplace=True)]
        input_channel = l
    
    return nn.Sequential(*layers)

def make_layers_leaky_relu(cfg, batch_norm=False):
    layers = []

    input_channel = 3
    for i, l in enumerate(cfg):
        if l == 'M':
            # layers += [nn.MaxPool2d(kernel_size=2, stride=2)]            # no pooling, different depth
            continue
        if i==0:
            layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]
            print('********************** this is the first layer **************************')
        else:
            layers += [nn.Conv2d(input_channel, l, kernel_size=2, stride=2, padding=1)]  # different kernel size, different strides
        if batch_norm:
            layers += [nn.BatchNorm2d(l)]
        
        layers += [nn.LeakyReLU()]
        input_channel = l
    
    return nn.Sequential(*layers)


def make_layers_leaky_relu3x3(cfg, batch_norm=False):
    layers = []

    input_channel = 3
    for l in cfg:
        if l == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            continue
        
        layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]

        if batch_norm:
            layers += [nn.BatchNorm2d(l)]
        
        layers += [nn.LeakyReLU()]
        input_channel = l
    
    return nn.Sequential(*layers)
    


def vgg11_bn():
    return VGG(make_layers(cfg['A'], batch_norm=True))

def vgg13_bn():
    return VGG(make_layers(cfg['B'], batch_norm=True))

def vgg16_bn():
    return VGG(make_layers(cfg['D'], batch_norm=True))

def vgg19_bn():
    return VGG(make_layers(cfg['E'], batch_norm=True))

def vgg7_like_bn():
    return VGG_like(make_layers(cfg['F'], batch_norm=True))

def plain_net5(num_class=2):
    return plain_net(make_layers(cfg['G'], batch_norm=True), num_class=num_class)

def plain_net6_diff(num_class):
    # return plain_net(make_layers2(cfg['H'], batch_norm=True), num_class=num_class)
    # return plain_net(make_layers(cfg['H'], batch_norm=True), num_class=num_class)
    return plain_net(make_layers(cfg['I'], batch_norm=True), num_class=num_class)


def plain_net5_leaky_relu(num_class=2):
    # return plain_net_leaky_relu(make_layers_leaky_relu(cfg['G'], batch_norm=True), num_class=num_class)
    return plain_net_leaky_relu(make_layers_leaky_relu3x3(cfg['G'], batch_norm=True), num_class=num_class)

