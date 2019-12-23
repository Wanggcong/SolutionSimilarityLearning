
# tinyImageNet
CUDA_VISIBLE_DEVICES=2 python train.py --model-name  plain_net5 --model-path tinyImageNet_100 --step1 30 --epochs 50 --tasks 0


CUDA_VISIBLE_DEVICES=4 python train.py --model-name  plain_net5 --model-path tinyImageNetv2plain5 --step1 70 --epochs 100 --tasks 10



# about optimizer:
CUDA_VISIBLE_DEVICES=2 python train.py --model-name  plain_net5 --model-path cifar100_100_adam --step1 30 --epochs 50 --optimizer adam --tasks 0





CUDA_VISIBLE_DEVICES=0 python train.py --model-name  plain_net5_leaky_relu --model-path cifar100_100_leaky_relu_v2 --step1 30 --epochs 50 --tasks 0


CUDA_VISIBLE_DEVICES=0 python train.py --model-name  plain_net6_diff --model-path cifar100_100_net6_diff_3x3 --step1 30 --epochs 50 --tasks 0
