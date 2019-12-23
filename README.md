[Paper Link](https://openreview.net/forum?id=rJgCOySYwH&noteId=rJgCOySYwH&invitationId=ICLR.cc/2020/Conference/Paper1824)
## New!
2019.12.23: This is one of my favor papers. A clearer theoretical explanation is comming soon.









## Example
Some related absolute paths could be invalid. The code is simple. It could take too much time to create a solution set (e.g., 5000 trained models), we strongly suggest the reviewers focus on these files:



### A.Solution set generation:
#### --train.py
#### --model/mlp.py
#### --model/vgg_like.py
#### --model/mlp.py 

### B.Solution classification/retrieval, includes chain alignment rule and linear projection:
#### -- train_sup.py
#### --./model/meta_mdoel_cnn.py
#### --./model/meta_mdoel_mlp.py
#### --./model/meta_mdoel_rnn.py

### C.How to read weights as “training data”:
#### --datasets/cifar100_meta.py
#### --datasets/name_data.py
#### --tiny_ImageNet.py


### Related scripts are at ./scripts

# Citation:
```
@misc{
wang2020function,
title={Function Feature Learning of Neural Networks},
author={Guangcong Wang and Jianhuang Lai and Guangrun Wang and Wenqi Liang},
year={2020},
url={https://openreview.net/forum?id=rJgCOySYwH}
}
```
