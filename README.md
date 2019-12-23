## New!
Please click this link for updated version: 

https://github.com/Wanggcong/SolutionSimilarityLearning.git

This repo will be not updated.




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
