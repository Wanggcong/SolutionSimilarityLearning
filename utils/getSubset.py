import numpy as np


def getSubset(labels, c1, c2):
    selected_ind =[]
    for i in range(len(labels)):
        if labels[i]==c1 or labels[i]==c2:
            selected_ind.append(i)
    return selected_ind

def getSubset4(labels, c1, c2, c3, c4):
    selected_ind =[]
    for i in range(len(labels)):
        if labels[i]==c1 or labels[i]==c2 or labels[i]==c3 or labels[i]==c4:
            selected_ind.append(i)
    return selected_ind