
#tinyImageNet, not align, mlp, cls
CUDA_VISIBLE_DEVICES=7 python train_sup_baselines.py --meta-model TinyImageNet_mlp_notalign --epochs 10 --batch-size 1 --step1 6 --root-path /media/data2/anonymous/projects/LearnableParameterSimilarity/weights/tinyImageNet_mlp/ --model-path tiny_norm_layer1_notalign --selected-layers 1
#tinyImageNet, not align, cnn, cls
CUDA_VISIBLE_DEVICES=7 python train_sup_baselines.py --meta-model TinyImageNet_chain_notalign --epochs 10 --batch-size 1 --step1 6 --root-path /media/data2/anonymous/projects/LearnableParameterSimilarity/weights/tinyImageNetv2plain5/ --model-path tiny_norm_layer1_notalign_cnn --selected-layers 1


#cifar100, not align, mlp, cls
CUDA_VISIBLE_DEVICES=7 python train_sup_baselines.py --meta-model cifar_mlp_notalign --epochs 10 --batch-size 1 --step1 6 --root-path /media/data2/anonymous/projects/LearnableParameterSimilarity/weights/cifar100_mlp100_h500/ --model-path cifar_norm_layer1_notalign --selected-layers 1


#cifar100, not align, cnn, cls
CUDA_VISIBLE_DEVICES=7 python train_sup_baselines.py --meta-model cifar_chain_notalign --epochs 10 --batch-size 1 --step1 6 --root-path /media/data2/anonymous/projects/LearnableParameterSimilarity/weights/cifar100_100/  --model-path cifar_norm_cnn_layer1_notalign --selected-layers 1

