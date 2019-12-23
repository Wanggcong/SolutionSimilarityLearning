# CUDA_VISIBLE_DEVICES=2 python train_sup.py

CUDA_VISIBLE_DEVICES=7 python train_sup.py --meta-model cifar_mlp --selected-layers 1 --model-path l1v1 --epochs 10 --batch-size 1 --step1 6 --log-file layer1 --root-path /media/data2/anonymous/projects/LearnableParameterSimilarity/weights/cifar100_mlp100
CUDA_VISIBLE_DEVICES=7 python train_sup.py --meta-model cifar_chain --selected-layers 1 --model-path 0827l1v1 --epochs 10 --batch-size 1 --step1 6 --log-file layer1 
CUDA_VISIBLE_DEVICES=3 python train_sup.py --meta-model rnn --model-path v1 --epochs 10 --batch-size 1 --step1 6 --log-file v1 --root-path /media/data2/anonymous/projects/LearnableParameterSimilarity/weights/cifar100_rnn_v1 --log-interval 50


#tinyImageNet:
#mlp:
CUDA_VISIBLE_DEVICES=7 python train_sup.py --meta-model TinyImageNet_mlp --model-path tiny_norm_layer3 --epochs 10 --batch-size 1 --step1 6 --root-path /media/data2/anonymous/projects/LearnableParameterSimilarity/weights/tinyImageNet_mlp/ --selected-layers 3
#cnn:
CUDA_VISIBLE_DEVICES=7 python train_sup.py --meta-model TinyImageNet_chain --epochs 10 --batch-size 1 --step1 6 --root-path /media/data2/anonymous/projects/LearnableParameterSimilarity/weights/tinyImageNetv2plain5/  --model-path tiny_norm_cnn_layer3 --selected-layers 3